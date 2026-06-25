from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SUITE_SPEC = importlib.util.spec_from_file_location("_posteditbench_suite", ROOT / "posteditbench" / "suite.py")
if _SUITE_SPEC is None or _SUITE_SPEC.loader is None:  # pragma: no cover - import-time guard
    raise RuntimeError("unable to load posteditbench suite metadata")
_SUITE_MODULE = importlib.util.module_from_spec(_SUITE_SPEC)
sys.modules[_SUITE_SPEC.name] = _SUITE_MODULE
_SUITE_SPEC.loader.exec_module(_SUITE_MODULE)

_UTILS_SPEC = importlib.util.spec_from_file_location("_posteditbench_utils", ROOT / "posteditbench" / "utils.py")
if _UTILS_SPEC is None or _UTILS_SPEC.loader is None:  # pragma: no cover - import-time guard
    raise RuntimeError("unable to load posteditbench utility helpers")
_UTILS_MODULE = importlib.util.module_from_spec(_UTILS_SPEC)
sys.modules[_UTILS_SPEC.name] = _UTILS_MODULE
_UTILS_SPEC.loader.exec_module(_UTILS_MODULE)

OFFICIAL_RUNTIME_BASELINES = _SUITE_MODULE.OFFICIAL_RUNTIME_BASELINES
OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES = _SUITE_MODULE.OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES
SUITE_ATOMIC_SOURCE_LABELS = _SUITE_MODULE.SUITE_ATOMIC_SOURCE_LABELS
SUITE_ATOMIC_SOURCE_ORDER = _SUITE_MODULE.SUITE_ATOMIC_SOURCE_ORDER
SUITE_MODEL_FAMILIES = _SUITE_MODULE.SUITE_MODEL_FAMILIES
SUITE_MODEL_ROSTER = _SUITE_MODULE.SUITE_MODEL_ROSTER
SUITE_MODEL_REVISIONS = _SUITE_MODULE.SUITE_MODEL_REVISIONS
SUITE_MODEL_SLUGS = _SUITE_MODULE.SUITE_MODEL_SLUGS
suite_benchmark_roster = _SUITE_MODULE.suite_benchmark_roster
suite_experiment_languages = _SUITE_MODULE.suite_experiment_languages
suite_model_revision = _SUITE_MODULE.suite_model_revision
suite_source_by_slug = _SUITE_MODULE.suite_source_by_slug
stable_hash = _UTILS_MODULE.stable_hash

try:
    from _shared import dump_json, read_jsonl, write_jsonl
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import dump_json, read_jsonl, write_jsonl


HEAVY_STAGE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
MODEL_PRIORITY_ORDER = (
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "bigcode/starcoder2-7b",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
)
METHOD_PRIORITY_ORDER = (
    "ewd_runtime",
    "kgw_runtime",
    "sweet_runtime",
    "stone_runtime",
)
SOURCE_PRIORITY_ORDER = (
    "crafted_original",
    "crafted_translation",
    "crafted_stress",
    "mbxp_5lang",
    "humaneval_x",
    "mbpp_plus",
    "humaneval_plus",
)
_MODEL_PRIORITY_RANK = {name: len(MODEL_PRIORITY_ORDER) - index for index, name in enumerate(MODEL_PRIORITY_ORDER)}
_METHOD_PRIORITY_RANK = {name: len(METHOD_PRIORITY_ORDER) - index for index, name in enumerate(METHOD_PRIORITY_ORDER)}
_SOURCE_PRIORITY_RANK = {name: len(SOURCE_PRIORITY_ORDER) - index for index, name in enumerate(SOURCE_PRIORITY_ORDER)}

_SOURCE_CONFIGS = {
    "humaneval_plus": "configs/public_humaneval_plus_stone_runtime.yaml",
    "mbpp_plus": "configs/public_mbpp_plus_stone_runtime.yaml",
    "humaneval_x": "configs/humanevalx_only.yaml",
    "mbxp_5lang": "configs/mbxp_only.yaml",
    "crafted_original": "configs/crafted_original_only.yaml",
    "crafted_translation": "configs/crafted_translation_only.yaml",
    "crafted_stress": "configs/crafted_stress_only.yaml",
}

_BASE_CONFIGS = {
    "stone_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_stone_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_stone_runtime.yaml",
        "default": "configs/public_humaneval_plus_stone_runtime.yaml",
    },
    "sweet_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_sweet_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_sweet_runtime.yaml",
        "default": "configs/public_humaneval_plus_sweet_runtime.yaml",
    },
    "ewd_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_ewd_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_ewd_runtime.yaml",
        "default": "configs/public_humaneval_plus_ewd_runtime.yaml",
    },
    "kgw_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_kgw_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_kgw_runtime.yaml",
        "default": "configs/public_humaneval_plus_kgw_runtime.yaml",
    },
}

_SOURCE_TAGS = {
    "humaneval_plus": "heplus",
    "mbpp_plus": "mbppplus",
    "humaneval_x": "humanevalx",
    "mbxp_5lang": "mbxp",
    "crafted_original": "crafted_original",
    "crafted_translation": "crafted_translation",
    "crafted_stress": "crafted_stress",
}

_RELEASE_INPUT_PATHS = {
    "humaneval_plus": ROOT / "data" / "release" / "sources" / "suite_humaneval_plus_release.normalized.jsonl",
    "mbpp_plus": ROOT / "data" / "release" / "sources" / "suite_mbpp_plus_release.normalized.jsonl",
    "humaneval_x": ROOT / "data" / "release" / "sources" / "suite_humanevalx_release.normalized.jsonl",
    "mbxp_5lang": ROOT / "data" / "release" / "sources" / "suite_mbxp_release.normalized.jsonl",
    "crafted_original": ROOT / "data" / "release" / "sources" / "crafted_original_release.normalized.jsonl",
    "crafted_translation": ROOT / "data" / "release" / "sources" / "crafted_translation_release.normalized.jsonl",
    "crafted_stress": ROOT / "data" / "release" / "sources" / "crafted_stress_release.normalized.jsonl",
}


def _load_json(path: Path) -> dict[str, Any]:
    # Accept UTF-8 files with or without BOM so cross-platform rewrites do not
    # break the release-source manifest rebuild path.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relpath(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _config_payload(path: str) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"configuration file {path} must contain a mapping")
    return dict(payload)


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _parse_csv_values(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical release-suite inputs and emit full or filtered suite manifests."
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Optional custom manifest path. If omitted, the default suite manifests are refreshed in place.",
    )
    parser.add_argument(
        "--output",
        dest="output_manifest",
        type=Path,
        default=None,
        help="Alias for --output-manifest.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="suite_all_models_methods",
        help="Profile name for --output-manifest subset generation.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model roster filter for a subset manifest.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method filter for a subset manifest.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated source slug filter for a subset manifest.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional description override for --output-manifest subset generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-run benchmark limit override for reviewer subset manifests.",
    )
    parser.add_argument(
        "--skip-refresh-prepared-inputs",
        action="store_true",
        help="Build the requested manifest without regenerating canonical prepared inputs first.",
    )
    return parser.parse_args()


def _stable_row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("family_id", "")).strip().lower(),
        str(row.get("task_id", "")).strip().lower(),
        _normalize(row.get("language")),
        _normalize(row.get("difficulty")),
        str(row.get("prompt_digest", "")).strip().lower(),
        str(row.get("source_digest", "")).strip().lower(),
    )


def _drain_round_robin(
    buckets: dict[object, list[dict[str, Any]]],
    *,
    ordered_keys: list[object],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets.get(key) for key in ordered_keys):
        for key in ordered_keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(dict(bucket.pop(0)))
            if len(selected) >= limit:
                break
    return selected


def _difficulty_round_robin(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered_difficulties = ["hard", "medium", "easy"]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_normalize(row.get("difficulty"))].append(dict(row))
    for bucket in buckets.values():
        bucket.sort(key=_stable_row_sort_key)
    return _drain_round_robin(buckets, ordered_keys=ordered_difficulties, limit=limit)


def _candidate_multilingual_families(
    rows: list[dict[str, Any]],
    *,
    languages: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        language = _normalize(row.get("language"))
        if not family_id or language not in languages:
            continue
        current = by_family[family_id].get(language)
        candidate = dict(row)
        if current is None or _stable_row_sort_key(candidate) < _stable_row_sort_key(current):
            by_family[family_id][language] = candidate
    return {
        family_id: bundle
        for family_id, bundle in by_family.items()
        if all(language in bundle for language in languages)
    }


def _filter_release_reference_rows(
    rows: list[dict[str, Any]],
    *,
    include_reference_kinds: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed = {_normalize(value) for value in include_reference_kinds if _normalize(value)}
    if not allowed:
        return [dict(row) for row in rows]
    return [
        dict(row)
        for row in rows
        if _normalize(row.get("reference_kind") or "canonical") in allowed
    ]


def _family_round_robin_selection(
    family_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    languages: tuple[str, ...],
    family_limit: int,
    stratum_fields: tuple[str, ...],
) -> list[str]:
    buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for family_id, bundle in family_rows.items():
        representative = bundle[languages[0]]
        key = tuple(_normalize(representative.get(field)) or "unknown" for field in stratum_fields)
        buckets[key].append(family_id)
    ordered_keys = sorted(buckets, key=lambda key: tuple(str(part) for part in key))
    for bucket in buckets.values():
        bucket.sort()
    selected: list[str] = []
    while len(selected) < family_limit and any(buckets.get(key) for key in ordered_keys):
        for key in ordered_keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= family_limit:
                break
    return selected


def _category_balanced_family_selection(
    family_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    languages: tuple[str, ...],
    family_limit: int,
    difficulty_field: str = "difficulty",
) -> list[str]:
    category_buckets: dict[str, list[str]] = defaultdict(list)
    representative_rows: dict[str, dict[str, Any]] = {}
    for family_id, bundle in family_rows.items():
        representative = bundle[languages[0]]
        representative_rows[family_id] = representative
        category_buckets[_normalize(representative.get("category")) or "unknown"].append(family_id)

    ordered_categories = sorted(category_buckets)
    if not ordered_categories:
        return []
    base = family_limit // len(ordered_categories)
    remainder = family_limit % len(ordered_categories)
    quotas = {category: base for category in ordered_categories}
    availability_order = sorted(
        ordered_categories,
        key=lambda category: (-len(category_buckets[category]), category),
    )
    for category in availability_order[:remainder]:
        quotas[category] += 1

    selected: list[str] = []
    for category in ordered_categories:
        family_ids = category_buckets[category]
        difficulty_buckets: dict[str, list[str]] = defaultdict(list)
        for family_id in family_ids:
            representative = representative_rows[family_id]
            difficulty_buckets[_normalize(representative.get(difficulty_field)) or "unknown"].append(family_id)
        for bucket in difficulty_buckets.values():
            bucket.sort()
        category_selected: list[str] = []
        ordered_difficulties = ["hard", "medium", "easy", "unknown"]
        while len(category_selected) < min(int(quotas[category]), len(family_ids)) and any(
            difficulty_buckets.get(key) for key in ordered_difficulties
        ):
            for difficulty in ordered_difficulties:
                bucket = difficulty_buckets.get(difficulty) or []
                if not bucket:
                    continue
                category_selected.append(bucket.pop(0))
                if len(category_selected) >= min(int(quotas[category]), len(family_ids)):
                    break
        selected.extend(category_selected)

    if len(selected) < family_limit:
        remaining = sorted(set(family_rows) - set(selected))
        selected.extend(remaining[: family_limit - len(selected)])
    return selected[:family_limit]


def _selected_family_rows(
    family_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    selected_family_ids: list[str],
    languages: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family_id in selected_family_ids:
        bundle = family_rows[family_id]
        for language in languages:
            selected.append(dict(bundle[language]))
    return selected


def _coverage_from_rows(rows: list[dict[str, Any]], *, claimed_languages: tuple[str, ...]) -> dict[str, Any]:
    languages = sorted({_normalize(row.get("language")) for row in rows if _normalize(row.get("language"))})
    validation_supported_languages = sorted(
        {
            _normalize(row.get("language"))
            for row in rows
            if _normalize(row.get("language")) and bool(row.get("validation_supported"))
        }
    )
    runtime_annotation_available = any("clean_reference_validation_available" in row for row in rows)
    compile_annotation_available = any("clean_reference_compile_success" in row for row in rows)
    pass_annotation_available = any("clean_reference_passed" in row for row in rows)
    runtime_validation_supported_languages = (
        sorted(
            {
                _normalize(row.get("language"))
                for row in rows
                if _normalize(row.get("language")) and bool(row.get("clean_reference_validation_available"))
            }
        )
        if runtime_annotation_available
        else []
    )
    clean_reference_compile_rate = (
        round(sum(1 for row in rows if row.get("clean_reference_compile_success") is True) / max(1, len(rows)), 4)
        if compile_annotation_available
        else None
    )
    clean_reference_pass_rate = (
        round(sum(1 for row in rows if row.get("clean_reference_passed") is True) / max(1, len(rows)), 4)
        if pass_annotation_available
        else None
    )
    claimed = list(claimed_languages) or languages
    return {
        "observed_language_count": len(languages),
        "claimed_language_count": len(claimed),
        "observed_coverage_rate": round(len(set(languages) & set(claimed)) / max(1, len(claimed)), 4),
        "declared_semantic_validation_rate": round(sum(1 for row in rows if bool(row.get("validation_supported"))) / max(1, len(rows)), 4),
        "declared_semantic_validation_language_rate": round(len(set(validation_supported_languages)) / max(1, len(claimed)), 4),
        "runtime_semantic_validation_rate": (
            round(sum(1 for row in rows if row.get("clean_reference_validation_available")) / max(1, len(rows)), 4)
            if runtime_annotation_available
            else None
        ),
        "runtime_semantic_validation_language_rate": (
            round(len(set(runtime_validation_supported_languages)) / max(1, len(claimed)), 4)
            if runtime_annotation_available
            else None
        ),
        "semantic_validation_rate": (
            round(sum(1 for row in rows if row.get("clean_reference_validation_available")) / max(1, len(rows)), 4)
            if runtime_annotation_available
            else None
        ),
        "semantic_validation_language_rate": (
            round(len(set(runtime_validation_supported_languages)) / max(1, len(claimed)), 4)
            if runtime_annotation_available
            else None
        ),
        "clean_reference_compile_rate": clean_reference_compile_rate,
        "clean_reference_pass_rate": clean_reference_pass_rate,
        "runtime_validation_basis": "row_annotations" if runtime_annotation_available else "unavailable",
        "runtime_validation_annotations_available": runtime_annotation_available,
        "missing_claimed_languages": sorted(set(claimed) - set(languages)),
        "declared_unvalidated_languages": sorted(
            {
                _normalize(row.get("language"))
                for row in rows
                if _normalize(row.get("language")) and not bool(row.get("validation_supported"))
            }
        ),
        "runtime_unvalidated_languages": (
            sorted(
                {
                    _normalize(row.get("language"))
                    for row in rows
                    if _normalize(row.get("language")) and not bool(row.get("clean_reference_validation_available"))
                }
            )
            if runtime_annotation_available
            else []
        ),
        "unvalidated_languages": (
            sorted(
                {
                    _normalize(row.get("language"))
                    for row in rows
                    if _normalize(row.get("language")) and not bool(row.get("clean_reference_validation_available"))
                }
            )
            if runtime_annotation_available
            else []
        ),
    }


def _source_manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("source_manifests")
    if isinstance(entries, list) and entries:
        return [dict(item) for item in entries]
    return [dict(manifest)]


def _write_release_collection(
    *,
    source_slug: str,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    input_candidate_count: int,
    claimed_languages: tuple[str, ...],
    selection_policy: dict[str, Any],
) -> None:
    base_manifest = _load_json(input_path.with_suffix(".manifest.json"))
    spec = suite_source_by_slug(source_slug)
    if spec is None:
        raise KeyError(f"Unknown suite source slug: {source_slug}")
    rows = sorted((dict(row) for row in rows), key=_stable_row_sort_key)
    write_jsonl(output_path, rows)
    language_counts = Counter(_normalize(row.get("language")) for row in rows if _normalize(row.get("language")))
    source_group_counts = Counter(_normalize(row.get("source_group")) for row in rows if _normalize(row.get("source_group")))
    origin_type_counts = Counter(_normalize(row.get("origin_type")) for row in rows if _normalize(row.get("origin_type")))
    difficulty_counts = Counter(_normalize(row.get("difficulty")) for row in rows if _normalize(row.get("difficulty")))
    reference_kind_counts = Counter(_normalize(row.get("reference_kind") or "canonical") for row in rows)
    family_counts = Counter(str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip())
    category_counts = Counter(_normalize(row.get("category")) for row in rows if _normalize(row.get("category")))
    template_family_counts = Counter(_normalize(row.get("template_family")) for row in rows if _normalize(row.get("template_family")))
    raw_category_counts = Counter(str(row.get("category", "")).strip() for row in rows if str(row.get("category", "")).strip())
    raw_template_family_counts = Counter(
        str(row.get("template_family", "")).strip() for row in rows if str(row.get("template_family", "")).strip()
    )
    validation_backend_counts = Counter(
        _normalize(row.get("validation_backend")) for row in rows if _normalize(row.get("validation_backend"))
    )
    family_languages: dict[str, set[str]] = defaultdict(set)
    family_difficulty: dict[str, str] = {}
    family_category: dict[str, str] = {}
    family_template_family: dict[str, str] = {}
    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        if not family_id:
            continue
        language = _normalize(row.get("language"))
        if language:
            family_languages[family_id].add(language)
        difficulty = _normalize(row.get("difficulty"))
        if difficulty and family_id not in family_difficulty:
            family_difficulty[family_id] = difficulty
        category = str(row.get("category", "")).strip()
        if category and family_id not in family_category:
            family_category[family_id] = category
        template_family = str(row.get("template_family", "")).strip()
        if template_family and family_id not in family_template_family:
            family_template_family[family_id] = template_family
    sample_ids_path = output_path.with_suffix(".sample_ids.json")
    dump_json(sample_ids_path, {"record_count": len(rows), "sample_ids": [str(row.get("task_id", "")).strip() for row in rows if str(row.get("task_id", "")).strip()]})
    is_crafted_source = source_slug.startswith("crafted_")
    manifest = {
        "schema_version": 2 if is_crafted_source else 1,
        "collection_name": spec.collection_name or output_path.stem.replace(".normalized", ""),
        "benchmark": str(base_manifest.get("benchmark") or spec.dataset_label),
        "dataset_label": str(spec.dataset_label),
        "record_count": len(rows),
        "task_count": len(rows),
        "observed_languages": sorted(language_counts),
        "claimed_languages": list(claimed_languages),
        "validation_supported_languages": sorted({_normalize(row.get("language")) for row in rows if _normalize(row.get("language")) and bool(row.get("validation_supported"))}),
        "datasets": sorted({str(row.get("dataset", "")).strip() for row in rows if str(row.get("dataset", "")).strip()}),
        "language_counts": dict(sorted(language_counts.items())),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "origin_type_counts": dict(sorted(origin_type_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "inputs": [_relpath(input_path)],
        "input_filtered_counts": {_relpath(input_path): int(input_candidate_count)},
        "include_languages": list(claimed_languages),
        "include_source_groups": sorted(source_group_counts),
        "include_origin_types": sorted(origin_type_counts),
        "include_difficulties": sorted(difficulty_counts),
        "include_reference_kinds": sorted(reference_kind_counts),
        "quota_per_language": dict(sorted(language_counts.items())),
        "quota_per_source_group": {},
        "reference_kind_counts": dict(sorted(reference_kind_counts.items())),
        "reference_kind_total": int(sum(reference_kind_counts.values())),
        "canonical_reference_count": int(reference_kind_counts.get("canonical", 0)),
        "smoke_overlay_reference_count": int(reference_kind_counts.get("smoke_overlay", 0)),
        "coverage": _coverage_from_rows(rows, claimed_languages=claimed_languages),
        "source_manifests": _source_manifest_entries(base_manifest),
        "sample_ids_path": _relpath(sample_ids_path),
        "suite_selection_policy": selection_policy,
    }
    construction_note = str(base_manifest.get("construction_note", "")).strip()
    if construction_note:
        manifest["construction_note"] = construction_note
    if is_crafted_source:
        claimed_language_count = max(1, len(claimed_languages))
        family_task_count = int(len(rows) / max(1, len(family_counts))) if family_counts else 0
        family_difficulty_counts = Counter(family_difficulty.values())
        family_category_counts = Counter(family_category.values())
        family_template_counts = Counter(family_template_family.values())
        family_language_coverage_rate = (
            round(
                mean(len(languages) / claimed_language_count for languages in family_languages.values()),
                4,
            )
            if family_languages
            else 0.0
        )
        manifest.update(
            {
                "contract_drift_families": [],
                "family_difficulty_counts": dict(sorted(family_difficulty_counts.items())),
                "family_language_coverage_rate": family_language_coverage_rate,
                "languages": list(claimed_languages),
                "normalized_path": _relpath(output_path),
                "origin_type": sorted(origin_type_counts)[0] if len(origin_type_counts) == 1 else "",
                "source_group": sorted(source_group_counts)[0] if len(source_group_counts) == 1 else "",
                "task_count_per_family": family_task_count,
                "taxonomy_categories": sorted(raw_category_counts),
                "validation_backend_counts": dict(sorted(validation_backend_counts.items())),
            }
        )
        if family_category_counts:
            manifest["category_counts"] = dict(sorted(family_category_counts.items()))
        if family_template_counts:
            manifest["template_family_counts"] = dict(sorted(family_template_counts.items()))
    else:
        if category_counts:
            manifest["category_counts"] = dict(sorted(category_counts.items()))
        if template_family_counts:
            manifest["template_family_counts"] = dict(sorted(template_family_counts.items()))
    dump_json(output_path.with_suffix(".manifest.json"), manifest)


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _build_public_release_copy(source_slug: str) -> None:
    spec = suite_source_by_slug(source_slug)
    if spec is None:
        raise KeyError(f"Unknown suite source slug: {source_slug}")
    input_path = _RELEASE_INPUT_PATHS[source_slug]
    output_path = ROOT / spec.prepared_output
    rows = _filter_release_reference_rows(
        read_jsonl(input_path),
        include_reference_kinds=tuple(spec.include_reference_kinds),
    )
    if len(rows) < int(spec.full_limit):
        raise ValueError(
            f"{source_slug} exposes only {len(rows)} rows, but the canonical release suite expects {spec.full_limit}"
        )
    selected = [dict(row) for row in rows[: int(spec.full_limit)]]
    _write_release_collection(
        source_slug=source_slug,
        input_path=input_path,
        output_path=output_path,
        rows=selected,
        input_candidate_count=len(rows),
        claimed_languages=tuple(spec.languages),
        selection_policy={
            "type": "full_retained_release_copy",
            "target_record_count": int(spec.full_limit),
            "input_record_count": len(rows),
            "allowed_reference_kinds": list(spec.include_reference_kinds),
        },
    )


def _build_multilingual_family_release_slice(
    source_slug: str,
    *,
    stratum_fields: tuple[str, ...],
    category_balanced: bool = False,
) -> None:
    spec = suite_source_by_slug(source_slug)
    if spec is None:
        raise KeyError(f"Unknown suite source slug: {source_slug}")
    input_path = _RELEASE_INPUT_PATHS[source_slug]
    output_path = ROOT / spec.prepared_output
    rows = _filter_release_reference_rows(
        read_jsonl(input_path),
        include_reference_kinds=tuple(spec.include_reference_kinds),
    )
    languages = tuple(suite_experiment_languages(spec))
    family_rows = _candidate_multilingual_families(rows, languages=languages)
    family_limit = int(spec.full_limit) // max(1, len(languages))
    if category_balanced:
        selected_family_ids = _category_balanced_family_selection(
            family_rows,
            languages=languages,
            family_limit=family_limit,
        )
    else:
        selected_family_ids = _family_round_robin_selection(
            family_rows,
            languages=languages,
            family_limit=family_limit,
            stratum_fields=stratum_fields,
        )
    selected_rows = _selected_family_rows(family_rows, selected_family_ids=selected_family_ids, languages=languages)
    _write_release_collection(
        source_slug=source_slug,
        input_path=input_path,
        output_path=output_path,
        rows=selected_rows,
        input_candidate_count=len(family_rows) * len(languages),
        claimed_languages=languages,
        selection_policy={
            "type": "category_balanced_multilingual_release_slice"
            if category_balanced
            else "family_balanced_multilingual_release_slice",
            "target_record_count": int(spec.full_limit),
            "target_family_count": family_limit,
            "languages": list(languages),
            "strata": list(stratum_fields),
            "allowed_reference_kinds": list(spec.include_reference_kinds),
        },
    )


def _build_canonical_crafted_source(
    source_slug: str,
    *,
    stratum_fields: tuple[str, ...],
    category_balanced: bool = False,
) -> None:
    spec = suite_source_by_slug(source_slug)
    if spec is None:
        raise KeyError(f"Unknown suite source slug: {source_slug}")
    source_path = ROOT / spec.prepared_output
    if not source_path.exists():
        raise FileNotFoundError(f"missing canonical crafted source input: {source_path}")
    rows = read_jsonl(source_path)
    languages = tuple(suite_experiment_languages(spec))
    family_rows = _candidate_multilingual_families(rows, languages=languages)
    family_limit = int(spec.full_limit) // max(1, len(languages))
    if len(family_rows) < family_limit:
        raise ValueError(
            f"{source_slug} exposes only {len(family_rows)} complete multilingual families; "
            f"{family_limit} are required for the canonical {spec.full_limit}-row release source"
        )
    if category_balanced:
        selected_family_ids = _category_balanced_family_selection(
            family_rows,
            languages=languages,
            family_limit=family_limit,
        )
    else:
        selected_family_ids = _family_round_robin_selection(
            family_rows,
            languages=languages,
            family_limit=family_limit,
            stratum_fields=stratum_fields,
        )
    selected_rows = _selected_family_rows(family_rows, selected_family_ids=selected_family_ids, languages=languages)
    _write_release_collection(
        source_slug=source_slug,
        input_path=source_path,
        output_path=source_path,
        rows=selected_rows,
        input_candidate_count=len(family_rows) * len(languages),
        claimed_languages=languages,
        selection_policy={
            "type": "category_balanced_canonical_release_source" if category_balanced else "family_balanced_canonical_release_source",
            "target_record_count": int(spec.full_limit),
            "target_family_count": family_limit,
            "languages": list(languages),
            "strata": list(stratum_fields),
        },
    )
    current_count = _count_rows(source_path)
    if current_count != int(spec.full_limit):
        raise ValueError(f"{source_slug} canonical release source expected {spec.full_limit} rows, found {current_count}")


def _build_release_prepared_inputs(selected_sources: list[str] | tuple[str, ...] | None = None) -> None:
    selected = {str(source).strip() for source in (selected_sources or SUITE_ATOMIC_SOURCE_ORDER) if str(source).strip()}
    if "humaneval_plus" in selected:
        _build_public_release_copy("humaneval_plus")
    if "mbpp_plus" in selected:
        _build_public_release_copy("mbpp_plus")
    if "humaneval_x" in selected:
        _build_multilingual_family_release_slice("humaneval_x", stratum_fields=("difficulty",))
    if "mbxp_5lang" in selected:
        _build_multilingual_family_release_slice("mbxp_5lang", stratum_fields=("difficulty",))
    if "crafted_original" in selected:
        _build_canonical_crafted_source("crafted_original", stratum_fields=("category", "difficulty"), category_balanced=True)
    if "crafted_translation" in selected:
        _build_canonical_crafted_source("crafted_translation", stratum_fields=("category", "difficulty"), category_balanced=True)
    if "crafted_stress" in selected:
        _build_canonical_crafted_source("crafted_stress", stratum_fields=("category", "difficulty"), category_balanced=True)


def _source_metadata(source_key: str) -> dict[str, Any]:
    config_path = _SOURCE_CONFIGS[source_key]
    payload = _config_payload(config_path)
    benchmark = dict(payload.get("benchmark", {}))
    paths = dict(payload.get("paths", {}))
    source_spec = suite_source_by_slug(source_key)
    if source_spec is None:
        raise KeyError(f"Unknown suite source slug: {source_key}")
    prepared = str(
        source_spec.prepared_output
        or source_spec.prepared_benchmark
        or benchmark.get("prepared_output")
        or paths.get("prepared_benchmark")
        or benchmark.get("source")
        or ""
    ).strip()
    manifest = _load_json((ROOT / prepared).with_suffix(".manifest.json"))
    dataset_label = str(source_spec.dataset_label).strip() or SUITE_ATOMIC_SOURCE_LABELS[source_key]
    source_group_counts = dict(manifest.get("source_group_counts", {}))
    source_group = str(source_spec.source_group or benchmark.get("source_group") or next(iter(source_group_counts.keys()), source_key)).strip()
    observed_languages = [str(language).strip() for language in manifest.get("observed_languages", []) if str(language).strip()]
    experiment_languages = [str(language).strip() for language in suite_experiment_languages(source_spec) if str(language).strip()]
    if observed_languages:
        observed_languages = [language for language in observed_languages if language in experiment_languages]
    if not observed_languages:
        observed_languages = experiment_languages
    benchmark_override = dict(benchmark)
    benchmark_override["dataset_label"] = dataset_label
    benchmark_override["source"] = prepared
    benchmark_override["prepared_output"] = prepared
    benchmark_override["languages"] = observed_languages
    benchmark_override["source_group"] = source_group
    benchmark_override.pop("collection_sources", None)
    if source_spec.collection_name:
        benchmark_override["collection_name"] = source_spec.collection_name
    return {
        "key": source_key,
        "config": config_path,
        "dataset_label": dataset_label,
        "prepared_benchmark": prepared,
        "benchmark": benchmark_override,
        "full_limit": int(source_spec.full_limit),
        "source_group": source_group,
        "languages": observed_languages,
        "tag": _SOURCE_TAGS[source_key],
    }


def _base_config(method: str, source_key: str) -> str:
    spec = _BASE_CONFIGS[method]
    return spec.get(source_key, spec["default"])


def _project_name(method: str, model: str, source_key: str, stage: str) -> str:
    model_slug = SUITE_MODEL_SLUGS[model]
    return f"posteditbench-{stage}-{method}-{model_slug}-{_SOURCE_TAGS[source_key]}"


def _shared_seed(*, model: str, source_key: str, stage: str) -> int:
    digest = stable_hash(f"{stage}:{model}:{source_key}", digest_size=8)
    return 1000 + (int(digest, 16) % 900_000)


def _estimated_priority(
    *,
    model: str,
    method: str,
    source_key: str,
    benchmark_limit: int,
    method_priority_rank: dict[str, int] | None = None,
) -> int:
    model_rank = int(_MODEL_PRIORITY_RANK.get(model, 0))
    rank_map = method_priority_rank or _METHOD_PRIORITY_RANK
    method_rank = int(rank_map.get(method, 0))
    source_rank = int(_SOURCE_PRIORITY_RANK.get(source_key, 0))
    capped_limit = max(0, min(int(benchmark_limit), 99_999))
    return model_rank * 1_000_000_000 + method_rank * 10_000_000 + source_rank * 100_000 + capped_limit


def _runtime_watermark_overrides(model: str) -> dict[str, Any]:
    from posteditbench.baselines.stone_family.official_runtime import runtime_compatibility_profile_name

    payload: dict[str, Any] = {"model_name": model}
    revision = suite_model_revision(model)
    if revision:
        payload["revision"] = revision
    profile = runtime_compatibility_profile_name(model)
    if profile != "generic_completion":
        payload["compatibility_profile"] = profile
    return payload


def _run_item(
    *,
    run_id: str,
    profile: str,
    config: str,
    model: str,
    source: dict[str, Any],
    stage: str,
    benchmark_limit: int | None,
    baseline_eval_sample_limit: int,
    tags: list[str],
    method_priority_rank: dict[str, int] | None = None,
) -> dict[str, Any]:
    if benchmark_limit is None:
        raise ValueError(f"{run_id} requires an explicit benchmark limit for heavy-first priority planning")
    benchmark_override = json.loads(json.dumps(source["benchmark"]))
    benchmark_override["limit"] = int(benchmark_limit)
    return {
        "run_id": run_id,
        "profile": profile,
        "config": config,
        "model": model,
        "model_revision": suite_model_revision(model),
        "method": tags[-1],
        "source_slug": source["key"],
        "config_overrides": {
            "project": {"name": _project_name(tags[-1], model, source["key"], stage), "seed": _shared_seed(model=model, source_key=source["key"], stage=stage)},
            "paths": {"prepared_benchmark": source["prepared_benchmark"]},
            "benchmark": benchmark_override,
            "watermark": _runtime_watermark_overrides(model),
        },
        "resource": "gpu",
        "gpu_pool": "runtime",
        "baseline_eval": True,
        "baseline_eval_sample_limit": int(baseline_eval_sample_limit),
        "priority": _estimated_priority(
            model=model,
            method=tags[-1],
            source_key=source["key"],
            benchmark_limit=int(benchmark_limit),
            method_priority_rank=method_priority_rank,
        ),
        "tags": tags,
    }


def _suite_run_items() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    source_specs = [_source_metadata(source_key) for source_key in SUITE_ATOMIC_SOURCE_ORDER]
    for model in SUITE_MODEL_ROSTER:
        model_slug = SUITE_MODEL_SLUGS[model]
        for source in source_specs:
            for method in OFFICIAL_RUNTIME_BASELINES:
                runs.append(
                    _run_item(
                        run_id=f"suite_{model_slug}_{source['tag']}_{method}",
                        profile="suite_all_models_methods",
                        config=_base_config(method, source["key"]),
                        model=model,
                        source=source,
                        stage="suite",
                        benchmark_limit=int(source["full_limit"]),
                        baseline_eval_sample_limit=64,
                        tags=["suite", model_slug, source["tag"], method, method],
                    )
                )
    return runs


def _stage_a_run_items() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    model = HEAVY_STAGE_MODEL
    model_slug = SUITE_MODEL_SLUGS[model]
    for source_key in SUITE_ATOMIC_SOURCE_ORDER:
        source = _source_metadata(source_key)
        source_spec = suite_source_by_slug(source_key)
        if source_spec is None:
            raise KeyError(f"Unknown suite source slug: {source_key}")
        for method in OFFICIAL_RUNTIME_BASELINES:
            runs.append(
                _run_item(
                    run_id=f"stage_a_{model_slug}_{source['tag']}_{method}",
                    profile="suite_canary_heavy",
                    config=_base_config(method, source["key"]),
                    model=model,
                    source=source,
                    stage="stage-a",
                    benchmark_limit=int(source_spec.stage_a_limit),
                    baseline_eval_sample_limit=16,
                    tags=["precheck", "stage_a", model_slug, source["tag"], method, method],
                )
            )
    return runs


def _stage_b_run_items() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    smoke_sources: list[tuple[dict[str, Any], int]] = []
    for source_key in SUITE_ATOMIC_SOURCE_ORDER:
        source = _source_metadata(source_key)
        source_spec = suite_source_by_slug(source_key)
        if source_spec is None:
            raise KeyError(f"Unknown suite source slug: {source_key}")
        if int(source_spec.stage_b_limit) <= 0:
            continue
        smoke_sources.append((source, int(source_spec.stage_b_limit)))
    for model in SUITE_MODEL_ROSTER:
        if model == HEAVY_STAGE_MODEL:
            continue
        model_slug = SUITE_MODEL_SLUGS[model]
        for source, stage_b_limit in smoke_sources:
            for method in OFFICIAL_RUNTIME_BASELINES:
                runs.append(
                    _run_item(
                        run_id=f"stage_b_{model_slug}_{source['tag']}_{method}",
                        profile="model_invocation_smoke",
                        config=_base_config(method, source["key"]),
                        model=model,
                        source=source,
                        stage="stage-b",
                        benchmark_limit=stage_b_limit,
                        baseline_eval_sample_limit=4,
                        tags=["precheck", "stage_b", model_slug, source["tag"], method, method],
                    )
                )
    return runs


def _stage_b_source_keys() -> list[str]:
    selected: list[str] = []
    for source_key in SUITE_ATOMIC_SOURCE_ORDER:
        source_spec = suite_source_by_slug(source_key)
        if source_spec is None:
            raise KeyError(f"Unknown suite source slug: {source_key}")
        if int(source_spec.stage_b_limit) > 0:
            selected.append(source_key)
    return selected


def _filter_run_items(
    runs: list[dict[str, Any]],
    *,
    models: list[str],
    methods: list[str],
    sources: list[str],
) -> list[dict[str, Any]]:
    model_filter = {item.strip() for item in models if item.strip()}
    method_filter = {item.strip() for item in methods if item.strip()}
    source_filter = {item.strip() for item in sources if item.strip()}
    available_models = {str(item.get("model", "")).strip() for item in runs if str(item.get("model", "")).strip()}
    available_methods = {str(item.get("method", "")).strip() for item in runs if str(item.get("method", "")).strip()}
    available_sources = {str(item.get("source_slug", "")).strip() for item in runs if str(item.get("source_slug", "")).strip()}
    unknown_models = sorted(model_filter - available_models)
    unknown_methods = sorted(method_filter - available_methods)
    unknown_sources = sorted(source_filter - available_sources)
    if unknown_models or unknown_methods or unknown_sources:
        details = {
            "unknown_models": unknown_models,
            "unknown_methods": unknown_methods,
            "unknown_sources": unknown_sources,
            "available_models": sorted(available_models),
            "available_methods": sorted(available_methods),
            "available_sources": sorted(available_sources),
        }
        raise ValueError(
            "subset filters reference values outside the canonical release roster: "
            + json.dumps(details, indent=2, sort_keys=True)
        )
    filtered: list[dict[str, Any]] = []
    for item in runs:
        if model_filter and str(item.get("model", "")).strip() not in model_filter:
            continue
        if method_filter and str(item.get("method", "")).strip() not in method_filter:
            continue
        if source_filter and str(item.get("source_slug", "")).strip() not in source_filter:
            continue
        filtered.append(dict(item))
    return filtered


def _manifest_payload(
    *,
    profile: str,
    description: str,
    runs: list[dict[str, Any]],
    model_roster: list[str] | tuple[str, ...],
    benchmark_roster: list[str] | tuple[str, ...],
    atomic_benchmark_sources: list[str] | tuple[str, ...],
    fairness_rule: str | None = None,
    method_roster: list[str] | tuple[str, ...] | None = None,
    required_watermark_methods: list[str] | tuple[str, ...] | None = None,
    required_provider_modes: list[str] | tuple[str, ...] | None = None,
    required_gpu_pools: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_models = [str(model).strip() for model in model_roster if str(model).strip()]
    selected_source_slugs = [str(source).strip() for source in atomic_benchmark_sources if str(source).strip()]
    suite_inventory_dataset_counts = {
        SUITE_ATOMIC_SOURCE_LABELS[source_slug]: int(suite_source_by_slug(source_slug).full_limit)
        for source_slug in selected_source_slugs
        if source_slug in SUITE_ATOMIC_SOURCE_LABELS and suite_source_by_slug(source_slug) is not None
    }
    suite_inventory_source_aliases = {
        source_slug: {
            "label": SUITE_ATOMIC_SOURCE_LABELS[source_slug],
            "release_file": _relpath(_RELEASE_INPUT_PATHS[source_slug]) if source_slug in _RELEASE_INPUT_PATHS else "",
            "release_file_basename": _RELEASE_INPUT_PATHS[source_slug].name if source_slug in _RELEASE_INPUT_PATHS else "",
        }
        for source_slug in selected_source_slugs
        if source_slug in SUITE_ATOMIC_SOURCE_LABELS
    }
    return {
        "schema_version": 1,
        "profile": profile,
        "description": description,
        "fairness_rule": (
            fairness_rule
            or "For each local model backbone, compare all four official imported baselines on the same executable benchmark rows; cross-model execution may run in parallel but aggregation remains model-conditioned."
        ),
        "model_roster": selected_models,
        "model_revisions": {
            model: suite_model_revision(model)
            for model in selected_models
            if suite_model_revision(model)
        },
        "model_roster_metadata": [
            {
                "model": model,
                "revision": suite_model_revision(model),
                "slug": SUITE_MODEL_SLUGS.get(model, ""),
                "family": SUITE_MODEL_FAMILIES.get(model, "unspecified"),
            }
            for model in selected_models
        ],
        "benchmark_roster": list(benchmark_roster),
        "atomic_benchmark_sources": selected_source_slugs,
        "suite_inventory_datasets": list(suite_benchmark_roster()),
        "suite_inventory_dataset_counts": suite_inventory_dataset_counts,
        "suite_inventory_source_aliases": suite_inventory_source_aliases,
        "method_roster": list(method_roster or OFFICIAL_RUNTIME_BASELINES),
        "required_watermark_methods": list(required_watermark_methods or OFFICIAL_RUNTIME_BASELINES),
        "required_provider_modes": list(required_provider_modes or ["offline_mock"]),
        "required_gpu_pools": list(required_gpu_pools or ["runtime"]),
        "runs": runs,
    }


def _subset_manifest_payload(
    *,
    profile: str,
    description: str,
    runs: list[dict[str, Any]],
    limit: int | None = None,
) -> dict[str, Any]:
    profile_runs: list[dict[str, Any]] = []
    for run in runs:
        payload = dict(run)
        payload["profile"] = profile
        if limit is not None:
            capped_limit = max(1, int(limit))
            config_overrides = dict(payload.get("config_overrides", {}))
            benchmark_override = dict(config_overrides.get("benchmark", {}))
            benchmark_override["limit"] = capped_limit
            config_overrides["benchmark"] = benchmark_override
            payload["config_overrides"] = config_overrides
            payload["baseline_eval_sample_limit"] = max(
                1,
                min(int(payload.get("baseline_eval_sample_limit", capped_limit)), capped_limit),
            )
            payload["priority"] = _estimated_priority(
                model=str(payload.get("model", "")).strip(),
                method=str(payload.get("method", "")).strip(),
                source_key=str(payload.get("source_slug", "")).strip(),
                benchmark_limit=capped_limit,
            )
        profile_runs.append(payload)
    selected_models = [
        model
        for model in SUITE_MODEL_ROSTER
        if any(str(run.get("model", "")).strip() == model for run in profile_runs)
    ]
    selected_sources = [
        source_slug
        for source_slug in SUITE_ATOMIC_SOURCE_ORDER
        if any(str(run.get("source_slug", "")).strip() == source_slug for run in profile_runs)
    ]
    selected_methods = [
        method
        for method in OFFICIAL_RUNTIME_BASELINES
        if any(str(run.get("method", "")).strip() == method for run in profile_runs)
    ]
    payload = _manifest_payload(
        profile=profile,
        description=description,
        runs=profile_runs,
        model_roster=selected_models,
        benchmark_roster=[
            SUITE_ATOMIC_SOURCE_LABELS[source_slug]
            for source_slug in selected_sources
            if source_slug in SUITE_ATOMIC_SOURCE_LABELS
        ],
        atomic_benchmark_sources=selected_sources,
        fairness_rule="Reviewer subset fairness rule: compare only the explicitly selected model/method/source slice under the same benchmark rows and execution settings captured by this subset manifest.",
        method_roster=selected_methods,
        required_watermark_methods=selected_methods,
    )
    payload["suite_inventory_datasets"] = [
        SUITE_ATOMIC_SOURCE_LABELS[source_slug]
        for source_slug in selected_sources
        if source_slug in SUITE_ATOMIC_SOURCE_LABELS
    ]
    payload["suite_inventory_dataset_counts"] = {
        SUITE_ATOMIC_SOURCE_LABELS[source_slug]: int(suite_source_by_slug(source_slug).full_limit)
        for source_slug in selected_sources
        if source_slug in SUITE_ATOMIC_SOURCE_LABELS and suite_source_by_slug(source_slug) is not None
    }
    payload["suite_inventory_source_aliases"] = {
        source_slug: {
            "label": SUITE_ATOMIC_SOURCE_LABELS[source_slug],
            "release_file": _relpath(_RELEASE_INPUT_PATHS[source_slug]) if source_slug in _RELEASE_INPUT_PATHS else "",
            "release_file_basename": _RELEASE_INPUT_PATHS[source_slug].name if source_slug in _RELEASE_INPUT_PATHS else "",
        }
        for source_slug in selected_sources
        if source_slug in SUITE_ATOMIC_SOURCE_LABELS
    }
    return payload


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_manifest = None
    if args.output_manifest is not None:
        output_manifest = args.output_manifest if args.output_manifest.is_absolute() else (ROOT / args.output_manifest)
    requested_source_filters = _parse_csv_values(args.sources)
    if not args.skip_refresh_prepared_inputs:
        if output_manifest is not None and requested_source_filters:
            _build_release_prepared_inputs(requested_source_filters)
        else:
            _build_release_prepared_inputs()
    profile_name = str(args.profile).strip() or "suite_subset"
    profile_runs = _suite_run_items()
    if output_manifest is not None:
        selected_runs = _filter_run_items(
            profile_runs,
            models=_parse_csv_values(args.models),
            methods=_parse_csv_values(args.methods),
            sources=_parse_csv_values(args.sources),
        )
        if not selected_runs:
            raise ValueError(
                "the requested subset filters produced an empty manifest inside the canonical release roster; "
                "check the requested model/method/source combination or use a custom manifest workflow for non-canonical experiments"
            )
        payload = _subset_manifest_payload(
            profile=profile_name,
            description=(
                str(args.description).strip()
                or "PostEditBench release-suite reviewer subset generated from the canonical full-suite manifest builder."
            ),
            runs=selected_runs,
            limit=args.limit,
        )
        _write_manifest(output_manifest, payload)
        print(
            json.dumps(
                {
                    "manifest": _relpath(output_manifest),
                    "profile": payload["profile"],
                    "run_count": len(payload["runs"]),
                    "models": payload["model_roster"],
                    "sources": payload["atomic_benchmark_sources"],
                    "methods": payload["method_roster"],
                    "limit": args.limit,
                },
                indent=2,
            )
        )
        return 0
    stage_b_source_keys = _stage_b_source_keys()
    manifests = {
        ROOT / "configs" / "matrices" / "suite_all_models_methods.json": _manifest_payload(
            profile="suite_all_models_methods",
            description="PostEditBench release-suite full matrix: 5 local backbones x 4 pinned baseline implementations across 7 executed benchmark sources.",
            runs=_suite_run_items(),
            model_roster=SUITE_MODEL_ROSTER,
            benchmark_roster=suite_benchmark_roster(),
            atomic_benchmark_sources=SUITE_ATOMIC_SOURCE_ORDER,
        ),
        ROOT / "configs" / "matrices" / "suite_canary_heavy.json": _manifest_payload(
            profile="suite_canary_heavy",
            description="Stage A heavy precheck: Qwen2.5-Coder-14B across the release-suite atomic-source roster with bounded sample counts.",
            runs=_stage_a_run_items(),
            model_roster=(HEAVY_STAGE_MODEL,),
            benchmark_roster=suite_benchmark_roster(),
            atomic_benchmark_sources=SUITE_ATOMIC_SOURCE_ORDER,
        ),
        ROOT / "configs" / "matrices" / "model_invocation_smoke.json": _manifest_payload(
            profile="model_invocation_smoke",
            description="Stage B smoke precheck: remaining backbones across the full active atomic-source roster with bounded sample counts.",
            runs=_stage_b_run_items(),
            model_roster=tuple(model for model in SUITE_MODEL_ROSTER if model != HEAVY_STAGE_MODEL),
            benchmark_roster=tuple(
                SUITE_ATOMIC_SOURCE_LABELS[source_slug]
                for source_slug in stage_b_source_keys
                if source_slug in SUITE_ATOMIC_SOURCE_LABELS
            ),
            atomic_benchmark_sources=tuple(stage_b_source_keys),
        ),
    }
    for path, payload in manifests.items():
        _write_manifest(path, payload)
    print(json.dumps({str(path.relative_to(ROOT)): len(payload["runs"]) for path, payload in manifests.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
