from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from math import comb
from statistics import mean
from typing import Any

from .models import BenchmarkConfig, BenchmarkReport, BenchmarkRow, supported_attack_rows
from .metrics import (
    attack_breakdown,
    attack_robustness,
    classification_metrics,
    budget_curve_summary,
    clean_retention,
    language_breakdown,
    mean_robustness_score,
    mean_watermark_retention,
    semantic_attack_robustness,
    semantic_clean_retention,
    semantic_preservation_rate,
    semantic_watermarked_retention,
    semantic_validation_rate,
    watermarked_retention,
)
from .metrics.detection import threshold_prediction
from .scorecard import scorecard_for_rows


@dataclass(frozen=True, slots=True)
class SummaryStat:
    name: str
    value: float


def _language_names(rows: list[BenchmarkRow]) -> list[str]:
    names: list[str] = []
    for row in rows:
        if row.language and row.language not in names:
            names.append(row.language)
    return names


def _declared_validation_available(row: BenchmarkRow) -> bool:
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    example_metadata = metadata.get("example_metadata")
    if isinstance(example_metadata, dict) and "validation_supported" in example_metadata:
        return bool(example_metadata.get("validation_supported"))
    if "validation_supported" in metadata:
        return bool(metadata.get("validation_supported"))
    return bool(row.semantic_validation_available)


def _evaluation_track_name(row: BenchmarkRow) -> str:
    explicit = str(row.evaluation_track).strip()
    if explicit:
        return explicit
    return "unspecified"


def _effective_provider_modes(rows: list[BenchmarkRow]) -> list[str]:
    values: list[str] = []
    for row in rows:
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        provider_mode = str(metadata.get("provider_mode", "")).strip()
        if provider_mode and provider_mode not in values:
            values.append(provider_mode)
    return values


def _example_row_key(row: BenchmarkRow) -> tuple[str, str, str]:
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    example_metadata = metadata.get("example_metadata") if isinstance(metadata.get("example_metadata"), dict) else {}
    source_group = str(row.source_group or example_metadata.get("source_group", "")).strip() or "unspecified"
    model_label = str(row.model_label or "unspecified").strip() or "unspecified"
    example_id = str(row.example_id or "").strip() or "unspecified"
    return (model_label, source_group, example_id)


def _unique_clean_scores(rows: list[BenchmarkRow]) -> list[float]:
    scores: dict[tuple[str, str, str], float] = {}
    for row in rows:
        scores.setdefault(_example_row_key(row), row.clean_score)
    return list(scores.values())


def _calibration_points(rows: list[BenchmarkRow]) -> dict[str, Any]:
    supported_rows = supported_attack_rows(rows)
    clean_scores = _unique_clean_scores(supported_rows)
    attacked_scores = [row.attacked_score for row in supported_rows]
    thresholds = [round(value / 10.0, 1) for value in range(1, 10)]
    points: list[dict[str, Any]] = []
    best = {"threshold": 0.0, "youden_j": float("-inf")}
    for threshold in thresholds:
        labels = [False] * len(clean_scores) + [True] * len(attacked_scores)
        predictions = [threshold_prediction(score, threshold) for score in clean_scores + attacked_scores]
        metrics = classification_metrics(labels, predictions)
        fpr = sum(1 for score in clean_scores if threshold_prediction(score, threshold)) / max(1, len(clean_scores))
        tpr = sum(1 for score in attacked_scores if threshold_prediction(score, threshold)) / max(1, len(attacked_scores))
        youden_j = tpr - fpr
        points.append(
            {
                "threshold": threshold,
                "false_positive_rate": round(fpr, 4),
                "true_positive_rate": round(tpr, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "accuracy": round(metrics.accuracy, 4),
                "youden_j": round(youden_j, 4),
            }
        )
        if youden_j > best["youden_j"]:
            best = {"threshold": threshold, "youden_j": youden_j}
    return {
        "clean_score_count": len(clean_scores),
        "attacked_score_count": len(attacked_scores),
        "thresholds": points,
        "recommended_threshold": round(float(best["threshold"]), 4),
        "recommended_youden_j": round(float(best["youden_j"]), 4),
    }


def _example_clean_trials(rows: list[BenchmarkRow]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = _example_row_key(row)
        if key in grouped:
            continue
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        trials = metadata.get("clean_functional_trials", [])
        if isinstance(trials, list):
            grouped[key] = [dict(item) for item in trials if isinstance(item, dict)]
    return grouped


def _row_stage_timing(row: BenchmarkRow) -> dict[str, Any]:
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    stage_timing = metadata.get("stage_timing", {})
    return dict(stage_timing) if isinstance(stage_timing, dict) else {}


def _mapping_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _mapping_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


def _stage_timing_metrics(rows: list[BenchmarkRow]) -> dict[str, Any]:
    if not rows:
        return {
            "task_count": 0,
            "attack_row_count": 0,
            "clean_generation_seconds_total": 0.0,
            "watermarked_generation_seconds_total": 0.0,
            "attack_seconds_total": 0.0,
            "validation_seconds_total": 0.0,
            "detection_seconds_total": 0.0,
            "total_example_seconds_total": 0.0,
            "clean_generation_seconds_mean_per_task": 0.0,
            "watermarked_generation_seconds_mean_per_task": 0.0,
            "clean_generation_seconds_per_1k_token": 0.0,
            "watermarked_generation_seconds_per_1k_token": 0.0,
            "attack_seconds_mean_per_row": 0.0,
            "validation_seconds_mean_per_task": 0.0,
            "detection_seconds_mean_per_task": 0.0,
            "total_example_seconds_mean_per_task": 0.0,
            "clean_generation_standardized_token_count_total": 0,
            "watermarked_generation_standardized_token_count_total": 0,
            "attacked_standardized_token_count_total": 0,
            "clean_generation_line_count_total": 0,
            "watermarked_line_count_total": 0,
            "attacked_line_count_total": 0,
        }

    shared_clean_generation = 0.0
    shared_watermarked_generation = 0.0
    shared_clean_validation = 0.0
    shared_watermarked_validation = 0.0
    shared_negative_control_detection = 0.0
    shared_watermarked_detection = 0.0

    clean_generation_token_count_total = 0
    watermarked_generation_token_count_total = 0
    attacked_token_count_total = 0
    clean_generation_line_count_total = 0
    watermarked_line_count_total = 0
    attacked_line_count_total = 0

    attack_seconds_total = 0.0
    attacked_validation_seconds_total = 0.0
    attacked_detection_seconds_total = 0.0

    seen_examples: set[tuple[str, str, str]] = set()
    for row in rows:
        timing = _row_stage_timing(row)
        row_components = timing.get("row_stage_components", {})
        if not isinstance(row_components, dict):
            row_components = {}
        attack_seconds_total += _mapping_float(row_components, "attack_seconds")
        attacked_validation_seconds_total += _mapping_float(row_components, "attacked_validation_seconds")
        attacked_detection_seconds_total += _mapping_float(row_components, "attacked_detection_seconds")
        attacked_token_count_total += _mapping_int(timing, "attacked_standardized_token_count")
        attacked_line_count_total += _mapping_int(timing, "attacked_line_count")

        example_key = _example_row_key(row)
        if example_key in seen_examples:
            continue
        seen_examples.add(example_key)
        shared_components = timing.get("shared_stage_components", {})
        if not isinstance(shared_components, dict):
            shared_components = {}
        shared_clean_generation += _mapping_float(shared_components, "clean_generation_seconds")
        shared_watermarked_generation += _mapping_float(shared_components, "watermarked_generation_seconds")
        shared_clean_validation += _mapping_float(shared_components, "clean_validation_seconds")
        shared_watermarked_validation += _mapping_float(shared_components, "watermarked_validation_seconds")
        shared_negative_control_detection += _mapping_float(shared_components, "negative_control_detection_seconds")
        shared_watermarked_detection += _mapping_float(shared_components, "watermarked_detection_seconds")
        clean_generation_token_count_total += _mapping_int(timing, "clean_generation_standardized_token_count")
        watermarked_generation_token_count_total += _mapping_int(timing, "watermarked_generation_standardized_token_count")
        clean_generation_line_count_total += _mapping_int(timing, "clean_generation_line_count")
        watermarked_line_count_total += _mapping_int(timing, "watermarked_line_count")

    task_count = len(seen_examples)
    validation_seconds_total = shared_clean_validation + shared_watermarked_validation + attacked_validation_seconds_total
    detection_seconds_total = shared_negative_control_detection + shared_watermarked_detection + attacked_detection_seconds_total
    total_example_seconds_total = (
        shared_clean_generation
        + shared_watermarked_generation
        + attack_seconds_total
        + validation_seconds_total
        + detection_seconds_total
    )
    clean_generation_seconds_per_1k_token = (
        (shared_clean_generation / max(float(clean_generation_token_count_total), 1.0)) * 1000.0
        if clean_generation_token_count_total > 0
        else 0.0
    )
    watermarked_generation_seconds_per_1k_token = (
        (shared_watermarked_generation / max(float(watermarked_generation_token_count_total), 1.0)) * 1000.0
        if watermarked_generation_token_count_total > 0
        else 0.0
    )
    return {
        "task_count": task_count,
        "attack_row_count": len(rows),
        "clean_generation_seconds_total": round(shared_clean_generation, 4),
        "watermarked_generation_seconds_total": round(shared_watermarked_generation, 4),
        "attack_seconds_total": round(attack_seconds_total, 4),
        "validation_seconds_total": round(validation_seconds_total, 4),
        "detection_seconds_total": round(detection_seconds_total, 4),
        "total_example_seconds_total": round(total_example_seconds_total, 4),
        "clean_generation_seconds_mean_per_task": round(shared_clean_generation / max(1, task_count), 4),
        "watermarked_generation_seconds_mean_per_task": round(shared_watermarked_generation / max(1, task_count), 4),
        "clean_generation_seconds_per_1k_token": round(clean_generation_seconds_per_1k_token, 4),
        "watermarked_generation_seconds_per_1k_token": round(watermarked_generation_seconds_per_1k_token, 4),
        "attack_seconds_mean_per_row": round(attack_seconds_total / max(1, len(rows)), 4),
        "validation_seconds_mean_per_task": round(validation_seconds_total / max(1, task_count), 4),
        "detection_seconds_mean_per_task": round(detection_seconds_total / max(1, task_count), 4),
        "total_example_seconds_mean_per_task": round(total_example_seconds_total / max(1, task_count), 4),
        "clean_generation_standardized_token_count_total": clean_generation_token_count_total,
        "watermarked_generation_standardized_token_count_total": watermarked_generation_token_count_total,
        "attacked_standardized_token_count_total": attacked_token_count_total,
        "clean_generation_line_count_total": clean_generation_line_count_total,
        "watermarked_line_count_total": watermarked_line_count_total,
        "attacked_line_count_total": attacked_line_count_total,
    }


def _pass_at_k(successes: int, samples: int, k: int) -> float:
    if samples <= 0:
        return 0.0
    k = max(1, min(k, samples))
    if successes <= 0:
        return 0.0
    if samples - successes < k:
        return 1.0
    return 1.0 - (comb(samples - successes, k) / comb(samples, k))


def _clean_functional_metrics(rows: list[BenchmarkRow]) -> dict[str, Any]:
    grouped = _example_clean_trials(rows)
    all_trials = [trial for trials in grouped.values() for trial in trials]
    compile_values = [1.0 if trial.get("compile_success") else 0.0 for trial in all_trials if trial.get("compile_success") is not None]
    pass_values = [1.0 if trial.get("passed") else 0.0 for trial in all_trials if trial.get("passed") is not None]
    error_taxonomy: dict[str, int] = {}
    pass_at_values: dict[int, list[float]] = {1: [], 5: [], 10: []}
    for trials in grouped.values():
        successes = sum(1 for trial in trials if trial.get("passed") is True)
        samples = len(trials)
        for k in pass_at_values:
            pass_at_values[k].append(_pass_at_k(successes, samples, k))
        for trial in trials:
            label = str(trial.get("error_kind", "")).strip() or "passed"
            error_taxonomy[label] = error_taxonomy.get(label, 0) + 1
    return {
        "task_count": len(grouped),
        "sample_count": len(all_trials),
        "compile_success_rate": round(mean(compile_values), 4) if compile_values else 0.0,
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "pass@1": round(mean(pass_at_values[1]), 4) if pass_at_values[1] else 0.0,
        "pass@5": round(mean(pass_at_values[5]), 4) if pass_at_values[5] else 0.0,
        "pass@10": round(mean(pass_at_values[10]), 4) if pass_at_values[10] else 0.0,
        "error_taxonomy": error_taxonomy,
    }


def _attacked_functional_metrics(rows: list[BenchmarkRow]) -> dict[str, Any]:
    rows = supported_attack_rows(rows)
    validations: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        attacked = metadata.get("attacked_validation", {})
        clean_summary = metadata.get("clean_functional_summary", {})
        if isinstance(attacked, dict):
            validations.append(
                {
                    "available": attacked.get("available"),
                    "passed": attacked.get("passed"),
                    "compile_success": (attacked.get("metadata") or {}).get("compile_success"),
                    "error_kind": (attacked.get("metadata") or {}).get("error_kind"),
                    "clean_compile_success_rate": clean_summary.get("compile_success_rate", 0.0) if isinstance(clean_summary, dict) else 0.0,
                }
            )
    available = [item for item in validations if item.get("available")]
    compile_values = [1.0 if item.get("compile_success") else 0.0 for item in available if item.get("compile_success") is not None]
    pass_values = [1.0 if item.get("passed") else 0.0 for item in available if item.get("passed") is not None]
    gated = [item for item in available if float(item.get("clean_compile_success_rate", 0.0)) > 0.0]
    gated_pass = [1.0 if item.get("passed") else 0.0 for item in gated if item.get("passed") is not None]
    taxonomy: dict[str, int] = {}
    for item in available:
        label = str(item.get("error_kind", "")).strip() or "passed"
        taxonomy[label] = taxonomy.get(label, 0) + 1
    return {
        "validated_rows": len(available),
        "compile_success_rate": round(mean(compile_values), 4) if compile_values else 0.0,
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "clean_compile_subset_pass_rate": round(mean(gated_pass), 4) if gated_pass else 0.0,
        "error_taxonomy": taxonomy,
    }


def _watermarked_functional_metrics(rows: list[BenchmarkRow]) -> dict[str, Any]:
    validations: list[dict[str, Any]] = []
    seen_examples: set[tuple[str, str, str]] = set()
    for row in rows:
        key = _example_row_key(row)
        if key in seen_examples:
            continue
        seen_examples.add(key)
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        validation = metadata.get("watermarked_validation", metadata.get("clean_validation", {}))
        if isinstance(validation, dict):
            validations.append(
                {
                    "available": validation.get("available"),
                    "passed": validation.get("passed"),
                    "compile_success": (validation.get("metadata") or {}).get("compile_success"),
                    "error_kind": (validation.get("metadata") or {}).get("error_kind"),
                }
            )
    available = [item for item in validations if item.get("available")]
    compile_values = [1.0 if item.get("compile_success") else 0.0 for item in available if item.get("compile_success") is not None]
    pass_values = [1.0 if item.get("passed") else 0.0 for item in available if item.get("passed") is not None]
    taxonomy: dict[str, int] = {}
    for item in available:
        label = str(item.get("error_kind", "")).strip() or "passed"
        taxonomy[label] = taxonomy.get(label, 0) + 1
    pass_at_1 = round(mean(pass_values), 4) if pass_values else 0.0
    return {
        "validated_tasks": len(available),
        "compile_success_rate": round(mean(compile_values), 4) if compile_values else 0.0,
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "pass@1": pass_at_1,
        "error_taxonomy": taxonomy,
    }


def _slice_breakdown(rows: list[BenchmarkRow], attribute: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        if attribute == "evaluation_track":
            value = _evaluation_track_name(row)
        else:
            value = str(getattr(row, attribute, "")).strip()
        if not value and isinstance(row.metadata, dict):
            value = str((row.metadata.get("example_metadata") or {}).get(attribute, "")).strip()
        value = value or "unspecified"
        grouped.setdefault(value, []).append(row)
    breakdown: dict[str, dict[str, Any]] = {}
    for value, group_rows in grouped.items():
        supported_rows = supported_attack_rows(group_rows)
        breakdown[value] = {
            "count": float(len(group_rows)),
            "supported_count": float(len(supported_rows)),
            "support_rate": round(float(len(supported_rows)) / max(1.0, float(len(group_rows))), 4),
            "attacked_detect_rate": round(mean(1.0 if row.attacked_detected else 0.0 for row in supported_rows), 4)
            if supported_rows
            else 0.0,
            "mean_quality_score": round(mean(row.quality_score for row in supported_rows), 4) if supported_rows else 0.0,
            "semantic_validation_rate": round(semantic_validation_rate(group_rows), 4),
            "semantic_preservation_rate": round(semantic_preservation_rate(group_rows), 4),
            "clean_functional_metrics": _clean_functional_metrics(group_rows),
            "attacked_functional_metrics": _attacked_functional_metrics(group_rows),
            "scorecard": scorecard_for_rows(group_rows, include_generalization=False),
        }
    return breakdown


def _declared_validation_by_language(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.language or "unspecified", []).append(row)
    breakdown: dict[str, dict[str, float]] = {}
    for language, language_rows in grouped.items():
        breakdown[language] = {
            "count": float(len(language_rows)),
            "declared_semantic_validation_rate": round(
                mean(1.0 if _declared_validation_available(row) else 0.0 for row in language_rows),
                4,
            ),
        }
    return breakdown


def _reference_kind_breakdown(rows: list[BenchmarkRow]) -> dict[str, dict[str, Any]]:
    return _slice_breakdown(rows, "reference_kind")


def _baseline_family_breakdown(rows: list[BenchmarkRow]) -> dict[str, dict[str, Any]]:
    grouped = [row for row in rows if str(row.baseline_family).strip()]
    if not grouped:
        return {}
    return _slice_breakdown(grouped, "baseline_family")


def _runtime_validation_coverage(rows: list[BenchmarkRow]) -> dict[str, float]:
    if not rows:
        return {
            "runtime_semantic_validation_rate": 0.0,
            "runtime_semantic_validation_language_rate": 0.0,
        }
    runtime_rate = semantic_validation_rate(rows)
    validation_by_language = language_breakdown(rows)
    validated_languages = [language for language, stats in validation_by_language.items() if stats["semantic_validation_rate"] > 0.0]
    runtime_language_rate = len(validated_languages) / max(1, len(validation_by_language))
    return {
        "runtime_semantic_validation_rate": runtime_rate,
        "runtime_semantic_validation_language_rate": runtime_language_rate,
    }


def _coerce_optional_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _coverage_runtime_unavailable(coverage: dict[str, Any]) -> bool:
    return str(coverage.get("runtime_validation_basis", "")).strip().lower() == "unavailable"


def _ordered_unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _row_source_groups(rows: list[BenchmarkRow]) -> list[str]:
    values: list[str] = []
    for row in rows:
        source_group = str(row.source_group).strip() or str(row.dataset).strip()
        if source_group:
            values.append(source_group)
    return _ordered_unique_strings(values)


def _manifest_source_groups(benchmark_manifest: dict[str, Any] | None) -> list[str]:
    manifest = dict(benchmark_manifest or {})
    source_group_counts = manifest.get("source_group_counts", {})
    if isinstance(source_group_counts, dict) and source_group_counts:
        return _ordered_unique_strings([str(source_group) for source_group in source_group_counts.keys()])
    source_groups = manifest.get("source_groups", [])
    if isinstance(source_groups, list):
        return _ordered_unique_strings([str(source_group) for source_group in source_groups])
    return []


def resolve_benchmark_source_metadata(
    rows: list[BenchmarkRow],
    *,
    benchmark_manifest: dict[str, Any] | None = None,
    configured_source: str = "",
) -> dict[str, Any]:
    benchmark_sources = _manifest_source_groups(benchmark_manifest)
    if not benchmark_sources:
        benchmark_sources = _row_source_groups(rows)
    configured = str(configured_source).strip()
    if not benchmark_sources and configured:
        benchmark_sources = [configured]
    if len(benchmark_sources) == 1:
        benchmark_source = benchmark_sources[0]
    elif benchmark_sources:
        benchmark_source = "suite_aggregate"
    else:
        benchmark_source = configured
    return {
        "benchmark_source": benchmark_source,
        "benchmark_sources": list(benchmark_sources),
    }


def summarize_rows(rows: list[BenchmarkRow], *, benchmark_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if not rows:
        return {
            "schema_version": 3,
            "run_dir": "",
            "record_count": 0,
            "supported_attack_row_count": 0,
            "unsupported_attack_row_count": 0,
            "attack_support_rate": 0.0,
            "rows": 0.0,
            "mean_detection_score": 0.0,
            "mean_quality_score": 0.0,
            "mean_stealth_score": 0.0,
            "mean_mutation_distance": 0.0,
            "mean_watermark_retention": 0.0,
            "mean_robustness_score": 0.0,
            "mean_attack_outcome_score": 0.0,
            "clean_detect_rate": 0.0,
            "watermarked_detect_rate": 0.0,
            "attacked_detect_rate": 0.0,
            "avg_quality": 0.0,
            "avg_stealth": 0.0,
            "avg_mutation_distance": 0.0,
            "semantic_validation_rate": 0.0,
            "semantic_preservation_rate": 0.0,
            "semantic_clean_detect_rate": 0.0,
            "semantic_watermarked_detect_rate": 0.0,
            "declared_semantic_validation_rate": 0.0,
            "declared_semantic_validation_language_rate": 0.0,
            "runtime_semantic_validation_rate": 0.0,
            "runtime_semantic_validation_language_rate": 0.0,
            "claimed_languages": [],
            "observed_languages": [],
            "claimed_language_coverage": 0.0,
            "semantic_validation_coverage": 0.0,
            "semantic_validation_by_language": {},
            "declared_semantic_validation_by_language": {},
            "runtime_semantic_validation_by_language": {},
            "language_coverage": {},
            "coverage_gaps": {},
            "declared_validation_supported_languages": [],
            "runtime_validation_supported_languages": [],
            "runtime_validation_basis": "row_annotations",
            "runtime_validation_annotations_available": False,
            "evaluation_tracks": [],
            "paper_primary_track": "",
            "pass_threshold": 0.0,
            "benchmark_source": "",
            "benchmark_sources": [],
            "stage_timing": _stage_timing_metrics([]),
            "clean_retention": 0.0,
            "watermarked_retention": 0.0,
            "clean_functional_metrics": {},
            "watermarked_functional_metrics": {},
            "attacked_functional_metrics": {},
            "by_evaluation_track": {},
            "by_reference_kind": {},
            "by_baseline_family": {},
            "scorecard": scorecard_for_rows([]),
        }
    observed_languages = _language_names(rows)
    manifest = dict(benchmark_manifest or {})
    supported_rows = supported_attack_rows(rows)
    claimed_languages = list(manifest.get("claimed_languages", observed_languages) or observed_languages)
    claimed_set = set(str(language).lower() for language in claimed_languages if str(language).strip())
    observed_set = set(language.lower() for language in observed_languages)
    runtime_validation_by_language = language_breakdown(rows)
    declared_validation_by_language = _declared_validation_by_language(rows)
    language_coverage = dict(manifest.get("language_summary", {}))
    coverage = dict(manifest.get("coverage", {}))
    runtime_validation_unavailable = _coverage_runtime_unavailable(coverage)
    claimed_language_coverage = float(coverage.get("observed_coverage_rate", 0.0))
    declared_semantic_validation_rate = float(
        coverage.get(
            "declared_semantic_validation_rate",
            round(mean(1.0 if _declared_validation_available(row) else 0.0 for row in rows), 4) if rows else 0.0,
        )
    )
    declared_validated_languages = [
        language
        for language, stats in declared_validation_by_language.items()
        if float(stats.get("declared_semantic_validation_rate", 0.0)) > 0.0
    ]
    declared_semantic_validation_language_rate = float(
        coverage.get(
            "declared_semantic_validation_language_rate",
            round(len(declared_validated_languages) / max(1, len(claimed_languages)), 4),
        )
    )
    runtime_validation = _runtime_validation_coverage(rows)
    clean_functional = _clean_functional_metrics(rows)
    watermarked_functional = _watermarked_functional_metrics(rows)
    attacked_functional = _attacked_functional_metrics(rows)
    stage_timing = _stage_timing_metrics(rows)
    explicit_evaluation_tracks = sorted({str(row.evaluation_track).strip() for row in rows if str(row.evaluation_track).strip()})
    evaluation_tracks = explicit_evaluation_tracks
    paper_track_ready = bool(explicit_evaluation_tracks) and len(explicit_evaluation_tracks) == 1 and all(
        str(row.evaluation_track).strip() for row in rows
    )
    paper_primary_track = explicit_evaluation_tracks[0] if paper_track_ready else ""
    runtime_unvalidated_languages = [language for language, stats in runtime_validation_by_language.items() if stats["semantic_validation_rate"] == 0.0]
    clean_reference_compile_rate = coverage.get("clean_reference_compile_rate", 0.0)
    clean_reference_pass_rate = coverage.get("clean_reference_pass_rate", 0.0)
    runtime_semantic_validation_rate = runtime_validation["runtime_semantic_validation_rate"]
    return {
        "schema_version": 3,
        "run_dir": "",
        "record_count": len(rows),
        "supported_attack_row_count": len(supported_rows),
        "unsupported_attack_row_count": len(rows) - len(supported_rows),
        "attack_support_rate": round(float(len(supported_rows)) / max(1.0, float(len(rows))), 4),
        "rows": float(len(rows)),
        "mean_detection_score": mean(row.attacked_score for row in supported_rows) if supported_rows else 0.0,
        "mean_quality_score": mean(row.quality_score for row in supported_rows) if supported_rows else 0.0,
        "mean_stealth_score": mean(row.stealth_score for row in rows),
        "mean_mutation_distance": mean(row.mutation_distance for row in supported_rows) if supported_rows else 0.0,
        "mean_watermark_retention": mean_watermark_retention(rows),
        "mean_robustness_score": mean_robustness_score(rows),
        "mean_attack_outcome_score": mean_robustness_score(rows),
        "clean_detect_rate": mean(1.0 if row.clean_detected else 0.0 for row in supported_rows) if supported_rows else 0.0,
        "watermarked_detect_rate": mean(1.0 if row.positive_detected else 0.0 for row in supported_rows) if supported_rows else 0.0,
        "attacked_detect_rate": mean(1.0 if row.attacked_detected else 0.0 for row in supported_rows) if supported_rows else 0.0,
        "avg_quality": mean(row.quality_score for row in supported_rows) if supported_rows else 0.0,
        "avg_stealth": mean(row.stealth_score for row in rows),
        "avg_mutation_distance": mean(row.mutation_distance for row in supported_rows) if supported_rows else 0.0,
        "semantic_validation_rate": semantic_validation_rate(rows),
        "semantic_preservation_rate": semantic_preservation_rate(rows),
        "semantic_clean_detect_rate": semantic_clean_retention(rows),
        "semantic_watermarked_detect_rate": semantic_watermarked_retention(rows),
        "declared_semantic_validation_rate": declared_semantic_validation_rate,
        "declared_semantic_validation_language_rate": declared_semantic_validation_language_rate,
        "runtime_semantic_validation_rate": runtime_semantic_validation_rate,
        "runtime_semantic_validation_language_rate": runtime_validation["runtime_semantic_validation_language_rate"],
        "runtime_validation_basis": str(coverage.get("runtime_validation_basis", "row_annotations")),
        "runtime_validation_annotations_available": bool(
            coverage.get("runtime_validation_annotations_available", not runtime_validation_unavailable)
        ),
        "evaluation_tracks": evaluation_tracks,
        "paper_primary_track": paper_primary_track,
        "paper_track_ready": paper_track_ready,
        "claimed_languages": claimed_languages,
        "observed_languages": observed_languages,
        "claimed_language_coverage": claimed_language_coverage,
        "semantic_validation_coverage": semantic_validation_rate(rows),
        "semantic_validation_by_language": runtime_validation_by_language,
        "declared_semantic_validation_by_language": declared_validation_by_language,
        "runtime_semantic_validation_by_language": runtime_validation_by_language,
        "language_coverage": language_coverage,
        "stage_timing": stage_timing,
        "clean_functional_metrics": clean_functional,
        "watermarked_functional_metrics": watermarked_functional,
        "attacked_functional_metrics": attacked_functional,
        "clean_reference_compile_rate": (
            None if runtime_validation_unavailable and clean_reference_compile_rate is None
            else _coerce_optional_float(clean_reference_compile_rate, 0.0)
        ),
        "clean_reference_pass_rate": (
            None if runtime_validation_unavailable and clean_reference_pass_rate is None
            else _coerce_optional_float(clean_reference_pass_rate, 0.0)
        ),
        "declared_validation_supported_languages": list(manifest.get("validation_supported_languages", [])),
        "runtime_validation_supported_languages": list(
            manifest.get("runtime_validation_supported_languages", [])
            or [language for language, stats in runtime_validation_by_language.items() if stats["semantic_validation_rate"] > 0.0]
        ),
        "compile_success_rate": clean_functional.get("compile_success_rate", 0.0),
        "test_pass_rate": clean_functional.get("test_pass_rate", 0.0),
        "pass@1": clean_functional.get("pass@1", 0.0),
        "pass@5": clean_functional.get("pass@5", 0.0),
        "pass@10": clean_functional.get("pass@10", 0.0),
        "error_taxonomy": clean_functional.get("error_taxonomy", {}),
        "coverage_gaps": {
            "missing_claimed_languages": sorted(claimed_set - observed_set),
            "declared_unvalidated_languages": list(coverage.get("declared_unvalidated_languages", coverage.get("unvalidated_languages", []))),
            "runtime_unvalidated_languages": list(
                runtime_unvalidated_languages
                if runtime_validation_unavailable
                else coverage.get("runtime_unvalidated_languages", runtime_unvalidated_languages)
            ),
            "unvalidated_languages": runtime_unvalidated_languages,
        },
        "pass_threshold": 0.0,
        "by_evaluation_track": _slice_breakdown(rows, "evaluation_track"),
        "by_reference_kind": _reference_kind_breakdown(rows),
        "by_baseline_family": _baseline_family_breakdown(rows),
        "scorecard": scorecard_for_rows(rows),
    }


def build_report(
    config: BenchmarkConfig,
    rows: list[BenchmarkRow],
    output_path: str | None = None,
    *,
    benchmark_manifest: dict[str, Any] | None = None,
) -> BenchmarkReport:
    summary = summarize_rows(rows, benchmark_manifest=benchmark_manifest)
    summary.update(
        {
            "by_attack": attack_breakdown(rows),
            "by_language": _slice_breakdown(rows, "language"),
            "language_breakdown_attack_view": language_breakdown(rows),
            "attack_breakdown": attack_breakdown(rows),
            "attack_robustness": attack_robustness(rows),
            "semantic_attack_robustness": semantic_attack_robustness(rows),
            "budget_curves": budget_curve_summary(rows),
            "clean_retention": clean_retention(rows),
            "watermarked_retention": watermarked_retention(rows),
            "by_source_group": _slice_breakdown(rows, "source_group"),
            "by_origin_type": _slice_breakdown(rows, "origin_type"),
            "by_difficulty": _slice_breakdown(rows, "difficulty"),
            "by_task_category": _slice_breakdown(rows, "task_category"),
            "by_method_origin": _slice_breakdown(rows, "method_origin"),
            "by_evaluation_track": _slice_breakdown(rows, "evaluation_track"),
            "by_model_label": _slice_breakdown(rows, "model_label"),
            "by_reference_kind": _reference_kind_breakdown(rows),
            "by_baseline_family": _baseline_family_breakdown(rows),
            "benchmark_manifest": dict(benchmark_manifest or {}),
            "detection_calibration": _calibration_points(rows),
        }
    )
    reporting = {}
    if hasattr(config, "metadata"):
        reporting = dict(getattr(config, "metadata", {}).get("reporting", {}) or {})
    summary["pass_threshold"] = float(reporting.get("pass_threshold", summary.get("pass_threshold", 0.0) or 0.0))
    configured_provider_mode = str(getattr(config, "provider_mode", "offline_mock"))
    effective_provider_modes = _effective_provider_modes(rows)
    summary["configured_provider_mode"] = configured_provider_mode
    if len(effective_provider_modes) == 1:
        summary["provider_mode"] = effective_provider_modes[0]
    elif effective_provider_modes:
        summary["provider_mode"] = "mixed"
    else:
        summary["provider_mode"] = configured_provider_mode
    summary["provider_modes"] = effective_provider_modes or [configured_provider_mode]
    summary["validation_scope"] = str(getattr(config, "validation_scope", "python_first"))
    summary["baseline_families"] = sorted({row.baseline_family for row in rows if row.baseline_family})
    summary["baseline_origins"] = sorted({row.baseline_origin for row in rows if row.baseline_origin})
    summary["baseline_upstream_commits"] = sorted({row.baseline_upstream_commit for row in rows if row.baseline_upstream_commit})
    configured_benchmark_source = str(
        getattr(config, "corpus_parameters", {}).get("public_source")
        or getattr(config, "corpus_parameters", {}).get("source")
        or ""
    )
    summary.update(
        resolve_benchmark_source_metadata(
            rows,
            benchmark_manifest=benchmark_manifest,
            configured_source=configured_benchmark_source,
        )
    )
    if output_path:
        summary["run_dir"] = str(Path(output_path).parent)
    return BenchmarkReport(config=config, rows=tuple(rows), summary=summary, output_path=output_path)
