from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path} must contain JSON objects on each line")
            rows.append(payload)
    return rows


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _quota_select(
    rows: Sequence[dict[str, Any]],
    *,
    quota_per_language: Mapping[str, int] | None = None,
    quota_per_source_group: Mapping[str, int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    language_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        language = str(row.get("language", "")).lower()
        source_group = str(row.get("source_group", "")).lower()
        if quota_per_language and language in quota_per_language and language_counter[language] >= int(quota_per_language[language]):
            continue
        if quota_per_source_group and source_group in quota_per_source_group and source_counter[source_group] >= int(quota_per_source_group[source_group]):
            continue
        selected.append(dict(row))
        language_counter[language] += 1
        source_counter[source_group] += 1
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _reference_kind_priority_map(reference_kinds: Sequence[str] | None) -> dict[str, int]:
    if not reference_kinds:
        return {}
    return {
        str(reference_kind).strip().lower(): index
        for index, reference_kind in enumerate(reference_kinds)
        if str(reference_kind).strip()
    }


def _sort_rows_for_selection(
    rows: Sequence[dict[str, Any]],
    *,
    reference_kind_priority: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    priority_map = _reference_kind_priority_map(reference_kind_priority)

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        reference_kind = str(row.get("reference_kind", "canonical")).lower()
        return (
            str(row.get("language", "")).lower(),
            priority_map.get(reference_kind, len(priority_map)),
            str(row.get("source_group", "")).lower(),
            str(row.get("family_id", "")).lower(),
            str(row.get("task_id", "")).lower(),
        )

    return sorted((dict(row) for row in rows), key=sort_key)


def _resolve_language_balance(
    rows: Sequence[dict[str, Any]],
    *,
    include_languages: Sequence[str] | None,
    quota_per_language: Mapping[str, int] | None,
    balance_languages: Mapping[str, Any] | None,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    if not balance_languages or not bool(balance_languages.get("enabled", False)):
        return quota_per_language if quota_per_language is not None else None, {}

    available_counts = Counter(str(row.get("language", "")).lower() for row in rows if str(row.get("language", "")).strip())
    configured_languages = [str(language).lower() for language in include_languages or [] if str(language).strip()]
    candidate_languages = configured_languages or sorted(available_counts)
    non_empty_languages = [language for language in candidate_languages if available_counts.get(language, 0) > 0]
    if not non_empty_languages:
        return quota_per_language if quota_per_language is not None else None, {
            "enabled": True,
            "applied": False,
            "reason": "no_languages_available",
        }

    raw_target = balance_languages.get("target", "max_feasible")
    if isinstance(raw_target, int):
        target_count = int(raw_target)
        target_label = str(raw_target)
    else:
        target_label = str(raw_target).strip().lower() or "max_feasible"
        if target_label == "max_feasible":
            target_count = min(available_counts[language] for language in non_empty_languages)
        else:
            target_count = int(target_label)

    derived_quotas = {language: min(int(target_count), int(available_counts.get(language, 0))) for language in non_empty_languages}
    if quota_per_language:
        merged = {str(key).lower(): int(value) for key, value in dict(quota_per_language).items()}
        for language, value in derived_quotas.items():
            merged[language] = min(int(merged.get(language, value)), int(value))
        effective_quotas = merged
    else:
        effective_quotas = derived_quotas

    metadata = {
        "enabled": True,
        "applied": True,
        "target": target_label,
        "target_count": int(target_count),
        "available_language_counts": dict(sorted(available_counts.items())),
        "effective_quota_per_language": dict(sorted(effective_quotas.items())),
        "reference_kind_priority": [
            str(reference_kind).strip().lower()
            for reference_kind in balance_languages.get("reference_kind_priority", [])
            if str(reference_kind).strip()
        ],
    }
    return effective_quotas, metadata


def _row_runtime_validation_summary(
    rows: Sequence[dict[str, Any]],
    *,
    claimed_language_count: int,
) -> dict[str, Any]:
    annotation_available = any("clean_reference_validation_available" in row for row in rows)
    compile_annotation_available = any("clean_reference_compile_success" in row for row in rows)
    pass_annotation_available = any("clean_reference_passed" in row for row in rows)
    if not annotation_available:
        return {
            "runtime_validation_supported_languages": [],
            "runtime_semantic_validation_rate": None,
            "runtime_semantic_validation_language_rate": None,
            "runtime_unvalidated_languages": [],
            "clean_reference_compile_rate": None,
            "clean_reference_pass_rate": None,
            "runtime_validation_basis": "unavailable",
            "runtime_validation_annotations_available": False,
        }

    runtime_validation_supported_languages = sorted(
        {
            str(row.get("language", "")).lower()
            for row in rows
            if bool(row.get("clean_reference_validation_available"))
        }
    )
    runtime_semantic_validation_rate = round(
        sum(1 for row in rows if bool(row.get("clean_reference_validation_available"))) / max(1, len(rows)),
        4,
    )
    runtime_semantic_validation_language_rate = round(
        len(set(runtime_validation_supported_languages)) / max(1, claimed_language_count),
        4,
    )
    clean_reference_compile_rate = (
        round(
            sum(1 for row in rows if row.get("clean_reference_compile_success") is True) / max(1, len(rows)),
            4,
        )
        if compile_annotation_available
        else None
    )
    clean_reference_pass_rate = (
        round(
            sum(1 for row in rows if row.get("clean_reference_passed") is True) / max(1, len(rows)),
            4,
        )
        if pass_annotation_available
        else None
    )
    runtime_unvalidated_languages = sorted(
        {
            str(row.get("language", "")).lower()
            for row in rows
            if not bool(row.get("clean_reference_validation_available"))
        }
    )
    return {
        "runtime_validation_supported_languages": runtime_validation_supported_languages,
        "runtime_semantic_validation_rate": runtime_semantic_validation_rate,
        "runtime_semantic_validation_language_rate": runtime_semantic_validation_language_rate,
        "runtime_unvalidated_languages": runtime_unvalidated_languages,
        "clean_reference_compile_rate": clean_reference_compile_rate,
        "clean_reference_pass_rate": clean_reference_pass_rate,
        "runtime_validation_basis": "row_annotations",
        "runtime_validation_annotations_available": True,
    }


def compose_benchmark_collection(
    inputs: Sequence[str | Path],
    *,
    include_languages: Sequence[str] | None = None,
    include_source_groups: Sequence[str] | None = None,
    include_origin_types: Sequence[str] | None = None,
    include_difficulties: Sequence[str] | None = None,
    include_reference_kinds: Sequence[str] | None = None,
    quota_per_language: Mapping[str, int] | None = None,
    quota_per_source_group: Mapping[str, int] | None = None,
    balance_languages: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_inputs = [Path(item) for item in inputs]
    manifests = {str(path): _load_manifest(path) for path in normalized_inputs}
    include_languages_set = {str(item).lower() for item in include_languages or [] if str(item).strip()}
    include_source_groups_set = {str(item).lower() for item in include_source_groups or [] if str(item).strip()}
    include_origin_types_set = {str(item).lower() for item in include_origin_types or [] if str(item).strip()}
    include_difficulties_set = {str(item).lower() for item in include_difficulties or [] if str(item).strip()}
    include_reference_kinds_set = {str(item).lower() for item in include_reference_kinds or [] if str(item).strip()}

    rows: list[dict[str, Any]] = []
    per_input_counts: dict[str, int] = {}
    for path in normalized_inputs:
        input_rows = _read_jsonl(path)
        filtered: list[dict[str, Any]] = []
        for row in input_rows:
            language = str(row.get("language", "")).lower()
            source_group = str(row.get("source_group", "")).lower()
            origin_type = str(row.get("origin_type", "")).lower()
            difficulty = str(row.get("difficulty", "")).lower()
            reference_kind = str(row.get("reference_kind", "canonical")).lower()
            if include_languages_set and language not in include_languages_set:
                continue
            if include_source_groups_set and source_group not in include_source_groups_set:
                continue
            if include_origin_types_set and origin_type not in include_origin_types_set:
                continue
            if include_difficulties_set and difficulty not in include_difficulties_set:
                continue
            if include_reference_kinds_set and reference_kind not in include_reference_kinds_set:
                continue
            filtered.append(dict(row))
        per_input_counts[str(path)] = len(filtered)
        rows.extend(filtered)

    effective_quota_per_language, language_balance_metadata = _resolve_language_balance(
        rows,
        include_languages=sorted(include_languages_set),
        quota_per_language=quota_per_language,
        balance_languages=balance_languages,
    )
    reference_kind_priority = []
    if language_balance_metadata.get("enabled"):
        reference_kind_priority = list(language_balance_metadata.get("reference_kind_priority", []))
    rows = _quota_select(
        _sort_rows_for_selection(rows, reference_kind_priority=reference_kind_priority),
        quota_per_language=effective_quota_per_language,
        quota_per_source_group=quota_per_source_group,
        limit=limit,
    )

    language_counts = Counter(str(row.get("language", "")).lower() for row in rows if str(row.get("language", "")).strip())
    source_group_counts = Counter(str(row.get("source_group", "")).lower() for row in rows if str(row.get("source_group", "")).strip())
    origin_type_counts = Counter(str(row.get("origin_type", "")).lower() for row in rows if str(row.get("origin_type", "")).strip())
    difficulty_counts = Counter(str(row.get("difficulty", "")).lower() for row in rows if str(row.get("difficulty", "")).strip())
    reference_kind_counts = Counter(str(row.get("reference_kind", "canonical")).lower() for row in rows)
    family_counts = Counter(str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip())
    claimed_language_count = len(include_languages_set or language_counts)
    runtime_summary = _row_runtime_validation_summary(rows, claimed_language_count=claimed_language_count)

    source_groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for path, manifest in manifests.items():
        for source_manifest in manifest.get("source_manifests", []) if isinstance(manifest.get("source_manifests"), list) else []:
            source_groups[str(source_manifest.get("source_group", source_manifest.get("benchmark", ""))).lower()] = dict(source_manifest)
        if manifest:
            source_groups[str(manifest.get("source_group", Path(path).stem)).lower()] = dict(manifest)

    collection_manifest = {
        "schema_version": 1,
        "record_count": len(rows),
        "observed_languages": sorted(language_counts),
        "claimed_languages": sorted(include_languages_set or language_counts),
        "validation_supported_languages": sorted(
            {
                str(row.get("language", "")).lower()
                for row in rows
                if bool(row.get("validation_supported"))
            }
        ),
        "datasets": sorted({str(row.get("dataset", "")).strip() for row in rows if str(row.get("dataset", "")).strip()}),
        "language_counts": dict(language_counts),
        "source_group_counts": dict(source_group_counts),
        "origin_type_counts": dict(origin_type_counts),
        "difficulty_counts": dict(difficulty_counts),
        "family_count": len(family_counts),
        "family_counts": {key: value for key, value in family_counts.items() if key},
        "inputs": [str(path) for path in normalized_inputs],
        "input_filtered_counts": per_input_counts,
        "include_languages": sorted(include_languages_set),
        "include_source_groups": sorted(include_source_groups_set),
        "include_origin_types": sorted(include_origin_types_set),
        "include_difficulties": sorted(include_difficulties_set),
        "include_reference_kinds": sorted(include_reference_kinds_set),
        "quota_per_language": {str(key): int(value) for key, value in dict(effective_quota_per_language or {}).items()},
        "quota_per_source_group": {str(key): int(value) for key, value in dict(quota_per_source_group or {}).items()},
        "language_balance": language_balance_metadata,
        "reference_kind_counts": dict(reference_kind_counts),
        "runtime_validation_supported_languages": list(runtime_summary["runtime_validation_supported_languages"]),
        "coverage": {
            "observed_language_count": len(language_counts),
            "claimed_language_count": claimed_language_count,
            "observed_coverage_rate": round(
                len(set(language_counts) & set(include_languages_set or language_counts)) / max(1, claimed_language_count),
                4,
            ),
            "declared_semantic_validation_rate": round(
                sum(1 for row in rows if bool(row.get("validation_supported"))) / max(1, len(rows)),
                4,
            ),
            "declared_semantic_validation_language_rate": round(
                len({str(row.get("language", "")).lower() for row in rows if bool(row.get("validation_supported"))})
                / max(1, claimed_language_count),
                4,
            ),
            "runtime_semantic_validation_rate": runtime_summary["runtime_semantic_validation_rate"],
            "runtime_semantic_validation_language_rate": runtime_summary["runtime_semantic_validation_language_rate"],
            "semantic_validation_rate": runtime_summary["runtime_semantic_validation_rate"],
            "semantic_validation_language_rate": runtime_summary["runtime_semantic_validation_language_rate"],
            "clean_reference_compile_rate": runtime_summary["clean_reference_compile_rate"],
            "clean_reference_pass_rate": runtime_summary["clean_reference_pass_rate"],
            "runtime_validation_basis": runtime_summary["runtime_validation_basis"],
            "runtime_validation_annotations_available": runtime_summary["runtime_validation_annotations_available"],
            "missing_claimed_languages": sorted(set(include_languages_set or language_counts) - set(language_counts)),
            "declared_unvalidated_languages": sorted(
                {str(row.get("language", "")).lower() for row in rows if not bool(row.get("validation_supported"))}
            ),
            "runtime_unvalidated_languages": list(runtime_summary["runtime_unvalidated_languages"]),
            "unvalidated_languages": list(runtime_summary["runtime_unvalidated_languages"]),
        },
        "source_manifests": sorted(source_groups.values(), key=lambda item: str(item.get("benchmark") or item.get("dataset_label") or item.get("source_group") or "")),
    }
    return rows, collection_manifest


def write_composed_collection(output_path: str | Path, rows: Iterable[dict[str, Any]], manifest: Mapping[str, Any]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
    output.with_suffix(".manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output
