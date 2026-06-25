from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from typing import Any, Iterable, Mapping

from .models import BenchmarkRow
from .scorecard import HEADLINE_SCORE_FIELD, scorecard_for_rows
from .suite import OFFICIAL_RUNTIME_BASELINES, SUITE_AGGREGATE_SOURCE_GROUPS, normalize_source_group


GENERATION_TIME_TRACK = "generation_time"
REFERENCE_CODE_TRACK = "reference_code"
_BENCHMARK_ROW_FIELDS = {field.name for field in fields(BenchmarkRow)}


def _row_from_payload(payload: Mapping[str, Any]) -> BenchmarkRow:
    data = {key: value for key, value in payload.items() if key in _BENCHMARK_ROW_FIELDS}
    data.setdefault("metadata", {})
    return BenchmarkRow(**data)


def _report_rows(report: Mapping[str, Any]) -> list[BenchmarkRow]:
    rows = report.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [_row_from_payload(row) for row in rows if isinstance(row, Mapping)]


def _report_identifier(report: Mapping[str, Any], *, index: int) -> str:
    output_path = str(report.get("output_path", "")).strip()
    if output_path:
        return output_path
    config = dict(report.get("config", {}))
    metadata = dict(config.get("metadata", {}))
    project = dict(metadata.get("project", {}))
    project_name = str(project.get("name", "")).strip()
    if project_name:
        return project_name
    return f"report_{index}"


def _row_evaluation_track(row: BenchmarkRow) -> str:
    explicit = str(row.evaluation_track).strip()
    if explicit:
        return explicit
    metadata = dict(row.metadata) if isinstance(row.metadata, Mapping) else {}
    provider_mode = str(metadata.get("provider_mode", "")).strip().lower()
    if provider_mode in {"local_hf", "local_command", "watermark_runtime"}:
        return GENERATION_TIME_TRACK
    model_label = str(row.model_label).strip().lower()
    if model_label and model_label not in {"reference_oracle", "unspecified"}:
        return GENERATION_TIME_TRACK
    if str(row.method_origin).strip().lower() == "upstream":
        return GENERATION_TIME_TRACK
    if str(row.watermark_scheme).strip().lower().endswith("_runtime"):
        return GENERATION_TIME_TRACK
    return REFERENCE_CODE_TRACK


def _source_group(row: BenchmarkRow) -> str:
    source_group = str(row.source_group).strip()
    if source_group:
        return source_group
    return str(row.dataset).strip() or "unspecified"


def _language_value(row: BenchmarkRow) -> str:
    return str(row.language).strip() or "unspecified"


def _model_value(row: BenchmarkRow) -> str:
    return str(row.model_label).strip() or "unspecified"


def _attack_value(row: BenchmarkRow) -> str:
    return str(row.attack_name).strip() or "unspecified"


def _reference_kind_value(row: BenchmarkRow) -> str:
    return str(row.reference_kind).strip() or "unspecified"


def _task_identity(row: BenchmarkRow) -> str:
    if str(row.task_id).strip():
        return str(row.task_id).strip()
    if str(row.prompt_digest).strip():
        return str(row.prompt_digest).strip()
    return str(row.example_id).strip() or "unspecified"


_SUITE_PUBLIC_HUMANEVAL_OVERLAP_SOURCES = {
    normalize_source_group("public_humaneval_plus"),
    normalize_source_group("public_humaneval_x"),
}
_SUITE_PUBLIC_MBPP_OVERLAP_SOURCES = {
    normalize_source_group("public_mbpp_plus"),
    normalize_source_group("public_mbxp_5lang"),
}


def _overlap_normalized_source(row: BenchmarkRow, *, collapse_cross_source_overlaps: bool) -> str:
    source_group = normalize_source_group(_source_group(row))
    if not collapse_cross_source_overlaps:
        return source_group
    language = _language_value(row).strip().lower()
    if language != "python":
        return source_group
    if source_group in _SUITE_PUBLIC_HUMANEVAL_OVERLAP_SOURCES:
        return "suite_public_humaneval_python"
    if source_group in _SUITE_PUBLIC_MBPP_OVERLAP_SOURCES:
        return "suite_public_mbpp_python"
    return source_group


def _overlap_task_identity(row: BenchmarkRow, *, collapse_cross_source_overlaps: bool) -> str:
    if not collapse_cross_source_overlaps:
        return _task_identity(row)
    source_group = normalize_source_group(_source_group(row))
    language = _language_value(row).strip().lower()
    if language == "python" and source_group in _SUITE_PUBLIC_HUMANEVAL_OVERLAP_SOURCES:
        prompt_digest = str(row.prompt_digest).strip().lower()
        if prompt_digest:
            return prompt_digest
    if language == "python" and source_group in _SUITE_PUBLIC_MBPP_OVERLAP_SOURCES:
        task_id = str(row.task_id).strip().lower()
        if task_id:
            return task_id
    return _task_identity(row)


def _support_signature(
    row: BenchmarkRow,
    *,
    include_model: bool,
    collapse_cross_source_overlaps: bool = False,
) -> tuple[Any, ...]:
    signature = (
        _overlap_normalized_source(row, collapse_cross_source_overlaps=collapse_cross_source_overlaps),
        _language_value(row),
        _attack_value(row),
        _reference_kind_value(row),
        _overlap_task_identity(row, collapse_cross_source_overlaps=collapse_cross_source_overlaps),
    )
    if include_model:
        return (_model_value(row), *signature)
    return signature


def _canonical_row_key(row: BenchmarkRow, *, collapse_cross_source_overlaps: bool = False) -> tuple[Any, ...]:
    return (
        str(row.watermark_scheme).strip() or "unspecified",
        str(row.method_origin).strip() or "unspecified",
        _row_evaluation_track(row),
        str(row.model_label).strip() or "unspecified",
        str(row.baseline_family).strip(),
        str(row.baseline_origin).strip(),
        str(row.baseline_upstream_commit).strip(),
        str(row.attack_name).strip(),
        str(row.language).strip(),
        _overlap_normalized_source(row, collapse_cross_source_overlaps=collapse_cross_source_overlaps),
        str(row.reference_kind).strip() or "unspecified",
        _overlap_task_identity(row, collapse_cross_source_overlaps=collapse_cross_source_overlaps),
        round(float(row.watermark_strength), 6),
    )


def _dedupe_rows(
    rows: Iterable[BenchmarkRow],
    *,
    collapse_cross_source_overlaps: bool = False,
) -> tuple[list[BenchmarkRow], int]:
    unique: dict[tuple[Any, ...], BenchmarkRow] = {}
    duplicates_removed = 0
    for row in rows:
        key = _canonical_row_key(row, collapse_cross_source_overlaps=collapse_cross_source_overlaps)
        if key in unique:
            duplicates_removed += 1
            continue
        unique[key] = row
    return list(unique.values()), duplicates_removed


def _group_metadata(rows: list[BenchmarkRow]) -> dict[str, Any]:
    return {
        "datasets": sorted({row.dataset for row in rows if row.dataset}),
        "source_groups": sorted({_source_group(row) for row in rows if _source_group(row)}),
        "languages": sorted({row.language for row in rows if row.language}),
        "methods": sorted({row.watermark_scheme for row in rows if row.watermark_scheme}),
        "models": sorted({row.model_label for row in rows if row.model_label}),
        "evaluation_tracks": sorted({_row_evaluation_track(row) for row in rows if _row_evaluation_track(row)}),
    }


def _allowed_source_groups(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {normalize_source_group(value) for value in values if normalize_source_group(value)}
    return normalized or None


def _row_in_allowed_sources(row: BenchmarkRow, allowed: set[str] | None) -> bool:
    if not allowed:
        return True
    return normalize_source_group(_source_group(row)) in allowed


def _allowed_methods(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalized or None


def _row_in_allowed_methods(row: BenchmarkRow, allowed: set[str] | None) -> bool:
    if not allowed:
        return True
    return str(row.watermark_scheme).strip().lower() in allowed


def _support_intersection(items: list[set[str]]) -> list[str]:
    nonempty = [set(item) for item in items if item]
    if not nonempty:
        return []
    if len(nonempty) == 1:
        return sorted(nonempty[0])
    intersection = set.intersection(*nonempty)
    return sorted(intersection)


def _leaderboard_entry(
    rows: list[BenchmarkRow],
    *,
    report_count: int,
    evaluation_track: str,
    raw_row_count: int,
    duplicate_rows_removed: int,
    noncomparable_rows_removed: int,
    comparable_models: list[str],
    comparable_source_groups: list[str],
    comparable_languages: list[str],
    comparable_attacks: list[str],
    comparable_reference_kinds: list[str],
    comparable_task_count: int,
    restrict_source_groups: set[str] | None = None,
    balance_by_source_group: bool = False,
) -> dict[str, Any]:
    if not rows:
        return {}
    scorecard = scorecard_for_rows(
        rows,
        restrict_source_groups=restrict_source_groups,
        balance_by_source_group=balance_by_source_group,
    )
    first = rows[0]
    return {
        "method": first.watermark_scheme,
        "origin": first.method_origin or "unspecified",
        "evaluation_track": evaluation_track,
        "model": first.model_label or "unspecified",
        "row_count": len(rows),
        "raw_row_count": raw_row_count,
        "report_count": report_count,
        "duplicate_rows_removed": duplicate_rows_removed,
        "noncomparable_rows_removed": noncomparable_rows_removed,
        "comparable_models": comparable_models,
        "comparable_source_groups": comparable_source_groups,
        "comparable_languages": comparable_languages,
        "comparable_attacks": comparable_attacks,
        "comparable_reference_kinds": comparable_reference_kinds,
        "comparable_task_count": comparable_task_count,
        **_group_metadata(rows),
        **scorecard,
    }


def _iter_report_rows(reports: Iterable[Mapping[str, Any]]) -> list[tuple[str, BenchmarkRow]]:
    materialized: list[tuple[str, BenchmarkRow]] = []
    for index, report in enumerate(reports):
        report_id = _report_identifier(report, index=index)
        for row in _report_rows(report):
            materialized.append((report_id, row))
    return materialized


def collect_report_rows(
    reports: Iterable[Mapping[str, Any]],
    *,
    track: str | None = None,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    dedupe: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[BenchmarkRow]:
    allowed = _allowed_source_groups(allowed_source_groups)
    methods = _allowed_methods(allowed_methods)
    rows = [
        row
        for _, row in _iter_report_rows(reports)
        if (track is None or _row_evaluation_track(row) == track)
        and _row_in_allowed_sources(row, allowed)
        and _row_in_allowed_methods(row, methods)
    ]
    if not dedupe:
        return rows
    unique_rows, _ = _dedupe_rows(rows, collapse_cross_source_overlaps=collapse_cross_source_overlaps)
    return unique_rows


def build_method_model_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    allowed = _allowed_source_groups(allowed_source_groups)
    methods = _allowed_methods(allowed_methods)
    materialized = [
        (report_id, row)
        for report_id, row in _iter_report_rows(reports)
        if _row_in_allowed_sources(row, allowed) and _row_in_allowed_methods(row, methods)
    ]
    entry_rows: dict[tuple[str, str, str, str], list[BenchmarkRow]] = defaultdict(list)
    entry_report_ids: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    cluster_supports: dict[tuple[str, str], dict[tuple[str, str], set[tuple[Any, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for report_id, row in materialized:
        track = _row_evaluation_track(row)
        model = str(row.model_label).strip() or "unspecified"
        method_key = (
            str(row.watermark_scheme).strip() or "unspecified",
            str(row.method_origin).strip() or "unspecified",
        )
        entry_key = (track, method_key[0], method_key[1], model)
        cluster_key = (track, model)
        entry_rows[entry_key].append(row)
        entry_report_ids[entry_key].add(report_id)
        cluster_supports[cluster_key][method_key].add(
            _support_signature(
                row,
                include_model=False,
                collapse_cross_source_overlaps=collapse_cross_source_overlaps,
            )
        )

    entries: list[dict[str, Any]] = []
    for entry_key, rows in entry_rows.items():
        track, method, origin, model = entry_key
        cluster_key = (track, model)
        comparable_support = _support_intersection(list(cluster_supports[cluster_key].values()))
        filtered_rows = [
            row
            for row in rows
            if (
                not comparable_support
                or _support_signature(
                    row,
                    include_model=False,
                    collapse_cross_source_overlaps=collapse_cross_source_overlaps,
                )
                in comparable_support
            )
        ]
        unique_rows, duplicates_removed = _dedupe_rows(
            filtered_rows,
            collapse_cross_source_overlaps=collapse_cross_source_overlaps,
        )
        noncomparable_rows_removed = max(0, len(rows) - len(filtered_rows))
        if not unique_rows:
            continue
        comparable_source_groups = sorted({_source_group(row) for row in unique_rows})
        comparable_languages = sorted({_language_value(row) for row in unique_rows})
        comparable_attacks = sorted({_attack_value(row) for row in unique_rows})
        comparable_reference_kinds = sorted({_reference_kind_value(row) for row in unique_rows})
        entry = _leaderboard_entry(
            unique_rows,
            report_count=len(entry_report_ids[entry_key]),
            evaluation_track=track,
            raw_row_count=len(rows),
            duplicate_rows_removed=duplicates_removed,
            noncomparable_rows_removed=noncomparable_rows_removed,
            comparable_models=[model],
            comparable_source_groups=comparable_source_groups,
            comparable_languages=comparable_languages,
            comparable_attacks=comparable_attacks,
            comparable_reference_kinds=comparable_reference_kinds,
            comparable_task_count=len(comparable_support)
            if comparable_support
            else len(
                {
                    _support_signature(
                        row,
                        include_model=False,
                        collapse_cross_source_overlaps=collapse_cross_source_overlaps,
                    )
                    for row in unique_rows
                }
            ),
            restrict_source_groups=allowed,
            balance_by_source_group=balance_by_source_group,
        )
        entry["method"] = method
        entry["origin"] = origin
        entries.append(entry)

    return sorted(
        entries,
        key=lambda item: (
            str(item.get("evaluation_track", "")),
            -float(item.get(HEADLINE_SCORE_FIELD, 0.0)),
            str(item.get("method", "")),
            str(item.get("model", "")),
        ),
    )


def build_track_method_model_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    track: str = GENERATION_TIME_TRACK,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in build_method_model_leaderboard(
            reports,
            allowed_source_groups=allowed_source_groups,
            allowed_methods=allowed_methods,
            balance_by_source_group=balance_by_source_group,
            collapse_cross_source_overlaps=collapse_cross_source_overlaps,
        )
        if str(entry.get("evaluation_track", "")).strip() == track
    ]


def _build_track_master_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    track: str,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    allowed = _allowed_source_groups(allowed_source_groups)
    methods = _allowed_methods(allowed_methods)
    materialized = [
        (report_id, row)
        for report_id, row in _iter_report_rows(reports)
        if _row_evaluation_track(row) == track
        and _row_in_allowed_sources(row, allowed)
        and _row_in_allowed_methods(row, methods)
    ]
    method_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    method_report_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    method_supports: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)

    for report_id, row in materialized:
        key = (
            str(row.watermark_scheme).strip() or "unspecified",
            str(row.method_origin).strip() or "unspecified",
        )
        method_rows[key].append(row)
        method_report_ids[key].add(report_id)
        method_supports[key].add(
            _support_signature(
                row,
                include_model=True,
                collapse_cross_source_overlaps=collapse_cross_source_overlaps,
            )
        )

    comparable_support = _support_intersection(list(method_supports.values()))

    entries: list[dict[str, Any]] = []
    for key, rows in method_rows.items():
        filtered_rows = [
            row
            for row in rows
            if (
                not comparable_support
                or _support_signature(
                    row,
                    include_model=True,
                    collapse_cross_source_overlaps=collapse_cross_source_overlaps,
                )
                in comparable_support
            )
        ]
        unique_rows, duplicates_removed = _dedupe_rows(
            filtered_rows,
            collapse_cross_source_overlaps=collapse_cross_source_overlaps,
        )
        noncomparable_rows_removed = max(0, len(rows) - len(filtered_rows))
        if not unique_rows:
            continue
        comparable_models = sorted({_model_value(row) for row in unique_rows})
        comparable_source_groups = sorted({_source_group(row) for row in unique_rows})
        comparable_languages = sorted({_language_value(row) for row in unique_rows})
        comparable_attacks = sorted({_attack_value(row) for row in unique_rows})
        comparable_reference_kinds = sorted({_reference_kind_value(row) for row in unique_rows})
        entry = _leaderboard_entry(
            unique_rows,
            report_count=len(method_report_ids[key]),
            evaluation_track=track,
            raw_row_count=len(rows),
            duplicate_rows_removed=duplicates_removed,
            noncomparable_rows_removed=noncomparable_rows_removed,
            comparable_models=comparable_models,
            comparable_source_groups=comparable_source_groups,
            comparable_languages=comparable_languages,
            comparable_attacks=comparable_attacks,
            comparable_reference_kinds=comparable_reference_kinds,
            comparable_task_count=len(comparable_support)
            if comparable_support
            else len(
                {
                    _support_signature(
                        row,
                        include_model=True,
                        collapse_cross_source_overlaps=collapse_cross_source_overlaps,
                    )
                    for row in unique_rows
                }
            ),
            restrict_source_groups=allowed,
            balance_by_source_group=balance_by_source_group,
        )
        entry.pop("model", None)
        entry["method"] = key[0]
        entry["origin"] = key[1]
        entries.append(entry)

    return sorted(entries, key=lambda item: (-float(item.get(HEADLINE_SCORE_FIELD, 0.0)), str(item.get("method", ""))))


def build_method_master_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    track: str = GENERATION_TIME_TRACK,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    return _build_track_master_leaderboard(
        reports,
        track=track,
        allowed_source_groups=allowed_source_groups,
        allowed_methods=allowed_methods,
        balance_by_source_group=balance_by_source_group,
        collapse_cross_source_overlaps=collapse_cross_source_overlaps,
    )


def build_reference_track_master_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    return _build_track_master_leaderboard(
        reports,
        track=REFERENCE_CODE_TRACK,
        allowed_source_groups=allowed_source_groups,
        allowed_methods=allowed_methods,
        balance_by_source_group=balance_by_source_group,
        collapse_cross_source_overlaps=collapse_cross_source_overlaps,
    )


def build_reference_track_model_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    return build_track_method_model_leaderboard(
        reports,
        track=REFERENCE_CODE_TRACK,
        allowed_source_groups=allowed_source_groups,
        allowed_methods=allowed_methods,
        balance_by_source_group=balance_by_source_group,
        collapse_cross_source_overlaps=collapse_cross_source_overlaps,
    )


def build_upstream_only_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    allowed_source_groups: Iterable[str] | None = None,
    allowed_methods: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
    collapse_cross_source_overlaps: bool = False,
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in build_method_master_leaderboard(
            reports,
            track=GENERATION_TIME_TRACK,
            allowed_source_groups=allowed_source_groups,
            allowed_methods=allowed_methods,
            balance_by_source_group=balance_by_source_group,
            collapse_cross_source_overlaps=collapse_cross_source_overlaps,
        )
        if entry.get("origin") == "upstream"
    ]


def _suite_source_groups_for_entry(entry: Mapping[str, Any]) -> set[str]:
    raw_groups = entry.get("source_groups")
    if isinstance(raw_groups, str):
        values = [item.strip(" '\"") for item in raw_groups.strip("[]").split(",") if item.strip()]
    elif isinstance(raw_groups, Iterable):
        values = [str(item).strip() for item in raw_groups if str(item).strip()]
    else:
        values = []
    return {normalize_source_group(value) for value in values if normalize_source_group(value)}


def _annotate_suite_atomic_source_roster(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {normalize_source_group(value) for value in SUITE_AGGREGATE_SOURCE_GROUPS}
    for entry in entries:
        observed = _suite_source_groups_for_entry(entry)
        missing = sorted(required - observed)
        entry["suite_atomic_source_complete"] = not missing
        entry["missing_source_groups"] = missing
    return entries


def build_suite_method_master_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    track: str = GENERATION_TIME_TRACK,
    allowed_methods: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    methods = OFFICIAL_RUNTIME_BASELINES if allowed_methods is None and track == GENERATION_TIME_TRACK else allowed_methods
    return _annotate_suite_atomic_source_roster(
        build_method_master_leaderboard(
            reports,
            track=track,
            allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
            allowed_methods=methods,
            balance_by_source_group=True,
            collapse_cross_source_overlaps=True,
        )
    )


def build_suite_method_model_leaderboard(
    reports: Iterable[Mapping[str, Any]],
    *,
    track: str = GENERATION_TIME_TRACK,
    allowed_methods: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    methods = OFFICIAL_RUNTIME_BASELINES if allowed_methods is None and track == GENERATION_TIME_TRACK else allowed_methods
    return _annotate_suite_atomic_source_roster(
        build_track_method_model_leaderboard(
            reports,
            track=track,
            allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
            allowed_methods=methods,
            balance_by_source_group=True,
            collapse_cross_source_overlaps=True,
        )
    )
