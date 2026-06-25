from __future__ import annotations

from collections import defaultdict
from math import exp, isfinite, log
from statistics import mean
from typing import Any, Iterable, Mapping

from .attacks.registry import CORE_ATTACKS, STRESS_ATTACKS, attack_tier
from .baselines.stone_family.evaluation import binary_auroc
from .models import BenchmarkRow, attack_row_supported, supported_attack_rows
from .suite import (
    SUITE_MULTILINGUAL_AGGREGATE_SOURCE_GROUPS,
    model_family_for_label,
    normalize_source_group,
)


EPSILON = 1e-9
SPARSE_TASK_MIN_ROWS = 20
HEADLINE_SOFT_FLOOR_EPSILON = 0.05
UNSUPPORTED_GENERALIZATION_NEUTRAL = 0.5
SCORE_VERSION = "posteditbench-suite-v8-core-stress-public-summary"
NEGATIVE_CONTROL_FAMILIES = ("human_reference", "clean_generation")
HEADLINE_SCORE_FIELD = "PostEditScore"
GENERALIZATION_AXIS_ORDER = (
    "source_stability",
    "task_stability",
    "language_stability",
    "cross_family_transfer",
)
SOURCE_BALANCED_COMPONENT_FIELDS = (
    "detection_separability",
    "robustness",
    "raw_robustness_strict",
    "robustness_support_rate",
    "stress_robustness",
    "utility",
    "raw_utility_strict",
    "utility_support_rate",
    "stealth",
    "efficiency",
    "stealth_conditioned",
    "efficiency_conditioned",
    "core_score",
    "raw_core_score_strict",
    "negative_control_fpr",
    "negative_vs_watermarked_auroc",
    "negative_control_support_rate",
    "semantic_validation_rate",
    "declared_semantic_validation_rate",
    "semantic_preservation_rate",
    "quality_score_mean",
    "watermark_retention_mean",
    "attacked_detected_semantic_rate",
    "watermarked_pass_preservation",
    "attacked_pass_preservation",
)
NULLABLE_SOURCE_BALANCED_COMPONENT_FIELDS = {
    "robustness",
    "raw_robustness_strict",
    "stress_robustness",
    "utility",
    "raw_utility_strict",
}
PUBLIC_SOURCE_BALANCED_COMPONENT_FIELDS = {
    "detection_separability",
    "robustness",
    "utility",
    "stealth",
    "efficiency",
    "stealth_conditioned",
    "efficiency_conditioned",
    "core_score",
    "semantic_validation_rate",
    "declared_semantic_validation_rate",
    "semantic_preservation_rate",
    "quality_score_mean",
    "watermark_retention_mean",
    "attacked_detected_semantic_rate",
    "attacked_pass_preservation",
}


def _clamp01(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator <= 0 else 1.0
    return _clamp01(float(numerator) / max(float(denominator), EPSILON))


def _gate_value(
    *,
    watermarked_pass_preservation: float,
    negative_control_fpr: float,
    negative_control_support_rate: float,
) -> float:
    return _clamp01(
        min(
            float(watermarked_pass_preservation),
            1.0 - float(negative_control_fpr),
            float(negative_control_support_rate),
        )
    )


def _geometric_mean(values: Iterable[float]) -> float:
    materialized = [_clamp01(float(value)) for value in values]
    if not materialized:
        return 0.0
    if any(value <= 0.0 for value in materialized):
        return 0.0
    return _clamp01(exp(mean(log(value) for value in materialized)))


def _headline_soft_floor(value: float) -> float:
    clamped = _clamp01(float(value))
    return _clamp01(HEADLINE_SOFT_FLOOR_EPSILON + ((1.0 - HEADLINE_SOFT_FLOOR_EPSILON) * clamped))


def _arithmetic_mean(values: Iterable[float]) -> float | None:
    materialized = [_clamp01(float(value)) for value in values]
    if not materialized:
        return None
    return _clamp01(mean(materialized))


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(_clamp01(float(value)), 4)


def _support_rate(supported_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return round(_clamp01(float(supported_count) / float(total_count)), 4)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    materialized = [_clamp01(float(value)) for value in values if value is not None]
    if not materialized:
        return None
    return _clamp01(mean(materialized))


def _public_status(*, supported: bool, value: float | None) -> str:
    if not supported or value is None:
        return "unsupported"
    if _clamp01(value) <= 0.0:
        return "supported_zero"
    return "supported_nonzero"


def _generalization_status(*, supported: bool, generalization: float | None) -> str:
    if not supported:
        return "unsupported"
    if _clamp01(generalization or 0.0) <= 0.0:
        return "supported_zero"
    return "supported_nonzero"


def _public_core_supported(component: Mapping[str, Any]) -> bool:
    return (
        str(component.get("robustness_status", "unsupported")).strip() != "unsupported"
        and str(component.get("utility_status", "unsupported")).strip() != "unsupported"
    )


def _row_metadata(row: BenchmarkRow) -> dict[str, Any]:
    return dict(row.metadata) if isinstance(row.metadata, Mapping) else {}


def _example_metadata(row: BenchmarkRow) -> dict[str, Any]:
    return dict(_row_metadata(row).get("example_metadata", {}))


def _attack_row_supported(row: BenchmarkRow) -> bool:
    return attack_row_supported(row)


def _example_row_key(row: BenchmarkRow) -> tuple[str, str, str]:
    source_group = normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", ""))) or "unspecified"
    model_label = str(row.model_label or "unspecified").strip() or "unspecified"
    example_id = str(row.example_id or "").strip() or "unspecified"
    return (model_label, source_group, example_id)


def _declared_validation_available(row: BenchmarkRow) -> bool:
    example_metadata = _example_metadata(row)
    if "validation_supported" in example_metadata:
        return bool(example_metadata.get("validation_supported"))
    metadata = _row_metadata(row)
    if "validation_supported" in metadata:
        return bool(metadata.get("validation_supported"))
    return bool(row.semantic_validation_available)


def _executed_validation_available(row: BenchmarkRow) -> bool:
    return bool(row.semantic_validation_available)


def _unique_example_rows(rows: Iterable[BenchmarkRow]) -> list[BenchmarkRow]:
    unique: dict[tuple[str, str, str], BenchmarkRow] = {}
    for row in rows:
        unique.setdefault(_example_row_key(row), row)
    return list(unique.values())


def _clean_functional_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in _unique_example_rows(rows):
        trials = _row_metadata(row).get("clean_functional_trials", [])
        if isinstance(trials, list):
            grouped[_example_row_key(row)] = [dict(item) for item in trials if isinstance(item, Mapping)]

    if not grouped:
        return {"sample_count": 0.0, "compile_success_rate": 0.0, "test_pass_rate": 0.0, "pass@1": 0.0}

    all_trials = [trial for trials in grouped.values() for trial in trials]
    compile_values = [1.0 if trial.get("compile_success") else 0.0 for trial in all_trials if trial.get("compile_success") is not None]
    pass_values = [1.0 if trial.get("passed") else 0.0 for trial in all_trials if trial.get("passed") is not None]
    pass_at_1: list[float] = []
    for trials in grouped.values():
        successes = sum(1 for trial in trials if trial.get("passed") is True)
        pass_at_1.append(1.0 if successes > 0 else 0.0)
    return {
        "sample_count": float(len(all_trials)),
        "compile_success_rate": round(mean(compile_values), 4) if compile_values else 0.0,
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "pass@1": round(mean(pass_at_1), 4) if pass_at_1 else 0.0,
    }


def _watermarked_functional_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    validations: list[dict[str, Any]] = []
    for row in _unique_example_rows(rows):
        metadata = _row_metadata(row)
        validation = metadata.get("watermarked_validation", metadata.get("clean_validation", {}))
        if isinstance(validation, Mapping):
            validations.append(dict(validation))
    available = [item for item in validations if item.get("available")]
    pass_values = [1.0 if item.get("passed") else 0.0 for item in available if item.get("passed") is not None]
    return {
        "validated_tasks": float(len(available)),
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "pass@1": round(mean(pass_values), 4) if pass_values else 0.0,
    }


def _attacked_functional_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    rows = supported_attack_rows(rows)
    validations: list[dict[str, Any]] = []
    for row in rows:
        attacked = _row_metadata(row).get("attacked_validation", {})
        if isinstance(attacked, Mapping):
            validations.append(dict(attacked))
    available = [item for item in validations if item.get("available")]
    pass_values = [1.0 if item.get("passed") else 0.0 for item in available if item.get("passed") is not None]
    return {
        "validated_rows": float(len(available)),
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
    }


def _negative_control_applicable(row: BenchmarkRow, control: object, *, family: str) -> bool:
    if not isinstance(control, Mapping):
        return family == "human_reference"
    if "applicable" in control:
        return bool(control.get("applicable"))
    if family == "human_reference":
        return True
    if str(row.evaluation_track).strip() == "reference_code":
        return False
    metadata = _row_metadata(row)
    provider_mode = str(metadata.get("provider_mode", "")).strip().lower()
    if provider_mode in {"local_hf", "local_command", "watermark_runtime"}:
        return True
    example_metadata = _example_metadata(row)
    if "provider_generation_succeeded" in example_metadata:
        return True
    return bool(control.get("available"))


def _negative_control_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, Any]:
    positives: list[float] = []
    negative_scores: list[float] = []
    negative_flags: list[bool] = []
    coverage = {
        "positive_examples": 0,
        "human_reference_negatives": 0,
        "clean_generation_negatives": 0,
        "human_reference_applicable": 0,
        "clean_generation_applicable": 0,
    }
    family_coverage_rates: dict[str, float] = {}

    for row in _unique_example_rows(rows):
        positives.append(float(row.positive_score))
        coverage["positive_examples"] += 1
        negative_controls = _row_metadata(row).get("negative_controls", {})
        if not isinstance(negative_controls, Mapping):
            negative_controls = {}
        for family in NEGATIVE_CONTROL_FAMILIES:
            control = negative_controls.get(family, {})
            applicable = _negative_control_applicable(row, control, family=family)
            if applicable:
                coverage[f"{family}_applicable"] += 1
            if not (applicable and isinstance(control, Mapping) and control.get("available")):
                continue
            negative_scores.append(float(control.get("score", 0.0)))
            negative_flags.append(bool(control.get("detected", False)))
            coverage[f"{family}_negatives"] += 1

    applicable_types: list[str] = []
    missing_types: list[str] = []
    observed_types: list[str] = []
    support_rates: list[float] = []
    for family in NEGATIVE_CONTROL_FAMILIES:
        applicable = int(coverage[f"{family}_applicable"])
        observed = int(coverage[f"{family}_negatives"])
        if applicable <= 0:
            continue
        applicable_types.append(family)
        rate = _clamp01(float(observed) / max(float(applicable), 1.0))
        family_coverage_rates[family] = round(rate, 4)
        support_rates.append(rate)
        if observed > 0:
            observed_types.append(family)
        else:
            missing_types.append(family)

    support_rate = round(mean(support_rates), 4) if support_rates else 0.0
    if negative_flags:
        fpr = mean(1.0 if detected else 0.0 for detected in negative_flags)
    elif applicable_types:
        fpr = 1.0
    else:
        fpr = 0.0
    auroc = binary_auroc(negative_scores, positives) if negative_scores and positives else 0.0

    coverage.update(
        {
            "negative_control_support_rate": support_rate,
            "negative_control_applicable_types": applicable_types,
            "negative_control_observed_types": observed_types,
            "negative_control_missing_types": missing_types,
            "negative_control_family_coverage": family_coverage_rates,
        }
    )
    return {
        "negative_control_fpr": round(_clamp01(fpr), 4),
        "negative_vs_watermarked_auroc": round(_clamp01(auroc), 4),
        "negative_control_support_rate": support_rate,
        "coverage": coverage,
    }


def _task_slice_key(row: BenchmarkRow) -> str:
    category = str(row.task_category or _example_metadata(row).get("category", "")).strip()
    if category:
        return category
    source_group = str(row.source_group or _example_metadata(row).get("source_group", "unspecified")).strip() or "unspecified"
    difficulty = str(row.difficulty or _example_metadata(row).get("difficulty", "unspecified")).strip() or "unspecified"
    reference_kind = str(row.reference_kind or _example_metadata(row).get("reference_kind", "unspecified")).strip() or "unspecified"
    return f"{source_group}:{difficulty}:{reference_kind}"


def _language_slice_key(row: BenchmarkRow) -> str:
    return str(row.language or _example_metadata(row).get("language", "")).strip().lower() or "unspecified"


def _group_rows(rows: Iterable[BenchmarkRow], key_fn) -> dict[str, list[BenchmarkRow]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        key = str(key_fn(row)).strip() or "unspecified"
        grouped[key].append(row)
    return dict(grouped)


def _restrict_rows(rows: Iterable[BenchmarkRow], *, restrict_source_groups: Iterable[str] | None) -> list[BenchmarkRow]:
    materialized = list(rows)
    if restrict_source_groups is None:
        return materialized
    allowed = {normalize_source_group(value) for value in restrict_source_groups if normalize_source_group(value)}
    if not allowed:
        return []
    return [
        row
        for row in materialized
        if normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", ""))) in allowed
    ]


def _collapse_sparse_groups(groups: dict[str, list[BenchmarkRow]], *, min_rows: int) -> tuple[dict[str, list[BenchmarkRow]], int]:
    if not groups:
        return {}, 0
    dense: dict[str, list[BenchmarkRow]] = {}
    other: list[BenchmarkRow] = []
    folded = 0
    for name, group_rows in groups.items():
        if len(group_rows) < min_rows:
            other.extend(group_rows)
            folded += len(group_rows)
            continue
        dense[name] = group_rows
    if other:
        dense["other"] = other
    return dense or groups, folded


def _example_shared_stage_timing(row: BenchmarkRow) -> dict[str, Any]:
    stage_timing = _row_metadata(row).get("stage_timing", {})
    if not isinstance(stage_timing, Mapping):
        return {}
    shared = stage_timing.get("shared_stage_components", {})
    if isinstance(shared, Mapping) and shared:
        return dict(shared)
    return dict(stage_timing)


def _example_stage_timing(row: BenchmarkRow) -> dict[str, Any]:
    stage_timing = _row_metadata(row).get("stage_timing", {})
    return dict(stage_timing) if isinstance(stage_timing, Mapping) else {}


def _efficiency_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    unique_rows = _unique_example_rows(rows)
    if not unique_rows:
        return {
            "efficiency_raw": 0.0,
        }
    clean_generation_seconds_total = 0.0
    watermarked_generation_seconds_total = 0.0
    clean_generation_token_count_total = 0
    watermarked_generation_token_count_total = 0
    for row in unique_rows:
        timing = _example_shared_stage_timing(row)
        full_timing = _example_stage_timing(row)
        try:
            clean_generation_seconds_total += float(timing.get("clean_generation_seconds", 0.0) or 0.0)
        except Exception:
            clean_generation_seconds_total += 0.0
        try:
            watermarked_generation_seconds_total += float(timing.get("watermarked_generation_seconds", 0.0) or 0.0)
        except Exception:
            watermarked_generation_seconds_total += 0.0
        try:
            clean_generation_token_count_total += int(full_timing.get("clean_generation_standardized_token_count", 0) or 0)
        except Exception:
            clean_generation_token_count_total += 0
        try:
            watermarked_generation_token_count_total += int(
                full_timing.get("watermarked_generation_standardized_token_count", 0) or 0
            )
        except Exception:
            watermarked_generation_token_count_total += 0
    clean_seconds_per_token = clean_generation_seconds_total / max(float(clean_generation_token_count_total), 1.0)
    watermarked_seconds_per_token = watermarked_generation_seconds_total / max(float(watermarked_generation_token_count_total), 1.0)
    if (
        clean_generation_seconds_total <= 0.0
        or watermarked_generation_seconds_total <= 0.0
        or clean_generation_token_count_total <= 0
        or watermarked_generation_token_count_total <= 0
    ):
        efficiency_raw = 0.0
    else:
        efficiency_raw = _clamp01(clean_seconds_per_token / max(watermarked_seconds_per_token, EPSILON))
    return {
        "efficiency_raw": round(efficiency_raw, 4),
    }


def _empty_components() -> dict[str, Any]:
    coverage = {
        "positive_examples": 0,
        "human_reference_negatives": 0,
        "clean_generation_negatives": 0,
        "human_reference_applicable": 0,
        "clean_generation_applicable": 0,
        "negative_control_support_rate": 0.0,
        "negative_control_applicable_types": [],
        "negative_control_observed_types": [],
        "negative_control_missing_types": [],
        "negative_control_family_coverage": {},
    }
    return {
        "detection_separability": 0.0,
        "robustness": None,
        "raw_robustness_strict": None,
        "robustness_status": "unsupported",
        "robustness_support_rate": 0.0,
        "stress_robustness": None,
        "headline_core_score": 0.0,
        "headline_generalization": UNSUPPORTED_GENERALIZATION_NEUTRAL,
        "generalization_status": "unsupported",
        "generalization": None,
        "utility": None,
        "raw_utility_strict": None,
        "utility_status": "unsupported",
        "utility_support_rate": 0.0,
        "stealth": 0.0,
        "efficiency": 0.0,
        "stealth_conditioned": 0.0,
        "efficiency_conditioned": 0.0,
        "core_score": 0.0,
        "raw_core_score_strict": 0.0,
        "negative_control_fpr": 1.0,
        "negative_vs_watermarked_auroc": 0.0,
        "negative_control_support_rate": 0.0,
        "semantic_validation_rate": 0.0,
        "declared_semantic_validation_rate": 0.0,
        "semantic_preservation_rate": 0.0,
        "quality_score_mean": 0.0,
        "watermark_retention_mean": 0.0,
        "attacked_detected_semantic_rate": 0.0,
        "watermarked_pass_preservation": 0.0,
        "attacked_pass_preservation": 0.0,
        "gate": 0.0,
        "raw_composite_strict": 0.0,
        "score_coverage": coverage,
    }


def _score_components_raw(rows: list[BenchmarkRow]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return _empty_components()

    supported_rows = supported_attack_rows(rows)
    executed_rate = (
        _clamp01(mean(1.0 if _executed_validation_available(row) else 0.0 for row in supported_rows))
        if supported_rows
        else 0.0
    )
    declared_rate = (
        _clamp01(mean(1.0 if _declared_validation_available(row) else 0.0 for row in supported_rows))
        if supported_rows
        else 0.0
    )
    negative_metrics = _negative_control_metrics(rows)
    clean_functional = _clean_functional_metrics(rows)
    watermarked_functional = _watermarked_functional_metrics(rows)
    attacked_functional = _attacked_functional_metrics(supported_rows)
    semantic_attack_rows = [
        row for row in supported_rows if row.semantic_validation_available and row.semantic_preserving is True
    ]
    semantic_attack = mean(1.0 if row.attacked_detected else 0.0 for row in semantic_attack_rows) if semantic_attack_rows else 0.0
    semantic_rows = [row for row in supported_rows if row.semantic_validation_available]
    semantic_preservation = mean(1.0 if row.semantic_preserving else 0.0 for row in semantic_rows) if semantic_rows else 0.0
    quality_mean = _clamp01(mean(row.quality_score for row in supported_rows)) if supported_rows else 0.0
    stealth = _clamp01(mean(row.stealth_score for row in rows))
    retention_mean = _clamp01(mean(row.watermark_retention for row in supported_rows)) if supported_rows else 0.0
    watermarked_pass_preservation = _safe_ratio(
        float(watermarked_functional["pass@1"]),
        float(clean_functional["pass@1"]),
    )
    attacked_validation_supported = bool(attacked_functional["validated_rows"])
    attacked_pass_preservation = (
        _safe_ratio(
            float(attacked_functional["test_pass_rate"]),
            float(clean_functional["test_pass_rate"]),
        )
        if attacked_validation_supported
        else 0.0
    )

    detection_separability = _clamp01(float(negative_metrics["negative_vs_watermarked_auroc"]))

    attack_rows_by_name: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        attack_rows_by_name[str(row.attack_name).strip() or "unspecified"].append(row)

    attack_breakdown: dict[str, dict[str, Any]] = {}
    core_attack_public_values: list[float] = []
    stress_attack_public_values: list[float] = []
    raw_core_attack_values: list[float] = []
    supported_core_attack_count = 0
    supported_stress_attack_count = 0
    for attack_name, attack_rows in attack_rows_by_name.items():
        supported_attack_rows_for_attack = [row for row in attack_rows if _attack_row_supported(row)]
        attack_clean_functional = _clean_functional_metrics(supported_attack_rows_for_attack)
        attack_attacked_functional = _attacked_functional_metrics(supported_attack_rows_for_attack)
        attack_semantic_rows = [
            row
            for row in supported_attack_rows_for_attack
            if row.semantic_validation_available and row.semantic_preserving is True
        ]
        factor_values: list[float] = []
        factor_payload = {
            "attack_retention": (
                _rounded_or_none(_clamp01(mean(row.watermark_retention for row in supported_attack_rows_for_attack)))
                if supported_attack_rows_for_attack
                else None
            ),
            "attack_attacked_detected_semantic_rate": (
                _rounded_or_none(mean(1.0 if row.attacked_detected else 0.0 for row in attack_semantic_rows))
                if attack_semantic_rows
                else None
            ),
            "attack_attacked_pass_preservation": (
                _rounded_or_none(
                    _safe_ratio(
                        float(attack_attacked_functional["test_pass_rate"]),
                        float(attack_clean_functional["test_pass_rate"]),
                    )
                )
                if int(attack_attacked_functional["validated_rows"]) > 0
                else None
            ),
        }
        for key in (
            "attack_retention",
            "attack_attacked_detected_semantic_rate",
            "attack_attacked_pass_preservation",
        ):
            value = factor_payload[key]
            if value is not None:
                factor_values.append(float(value))
        attack_support_rate = _support_rate(len(factor_values), 3)
        raw_attack_robustness_strict = _rounded_or_none(_geometric_mean(factor_values)) if factor_values else None
        attack_robustness = round(_clamp01((_arithmetic_mean(factor_values) or 0.0)), 4) if factor_values else None
        tier = attack_tier(attack_name)
        if attack_robustness is not None:
            if tier == "core":
                core_attack_public_values.append(float(attack_robustness))
                supported_core_attack_count += 1
            else:
                stress_attack_public_values.append(float(attack_robustness))
                supported_stress_attack_count += 1
        if raw_attack_robustness_strict is not None and tier == "core":
            raw_core_attack_values.append(float(raw_attack_robustness_strict))
        attack_breakdown[attack_name] = {
            "attack_name": attack_name,
            "attack_tier": tier,
            "attack_supported_rows": len(supported_attack_rows_for_attack),
            "attack_total_rows": len(attack_rows),
            **factor_payload,
            "attack_support_rate": attack_support_rate,
            "raw_attack_robustness_strict": raw_attack_robustness_strict,
            "attack_robustness": attack_robustness,
            "attack_status": _public_status(
                supported=bool(supported_attack_rows_for_attack) and bool(factor_values),
                value=attack_robustness,
            ),
        }

    robustness_support_rate = _support_rate(supported_core_attack_count, len(CORE_ATTACKS))
    robustness = round(_clamp01((_arithmetic_mean(core_attack_public_values) or 0.0)), 4) if core_attack_public_values else None
    raw_robustness_strict = _rounded_or_none(_geometric_mean(raw_core_attack_values)) if raw_core_attack_values else None
    stress_support_rate = _support_rate(supported_stress_attack_count, len(STRESS_ATTACKS))
    stress_robustness = round(_clamp01((_arithmetic_mean(stress_attack_public_values) or 0.0)), 4) if stress_attack_public_values else None

    utility_factors: dict[str, float | None] = {
        "quality_score_mean": round(quality_mean, 4) if supported_rows else None,
        "semantic_preservation_rate": _rounded_or_none(_clamp01(semantic_preservation)) if semantic_rows else None,
        "semantic_validation_rate": round(executed_rate, 4) if declared_rate > 0.0 else None,
    }
    supported_utility_values = [float(value) for value in utility_factors.values() if value is not None]
    utility_support_rate = _support_rate(len(supported_utility_values), 3)
    utility = round(_clamp01((_arithmetic_mean(supported_utility_values) or 0.0)), 4) if supported_utility_values else None
    raw_utility_strict = _rounded_or_none(_geometric_mean(supported_utility_values)) if supported_utility_values else None

    efficiency_metrics = _efficiency_metrics(rows)
    utility_scalar = float(utility or 0.0)
    robustness_scalar = float(robustness or 0.0)
    raw_utility_scalar = float(raw_utility_strict or 0.0)
    raw_robustness_scalar = float(raw_robustness_strict or 0.0)
    stealth_conditioned = (stealth * utility_scalar) ** 0.5 if stealth > 0.0 and utility_scalar > 0.0 else 0.0
    efficiency_conditioned = (
        float(efficiency_metrics["efficiency_raw"]) * utility_scalar
    ) ** 0.5 if float(efficiency_metrics["efficiency_raw"]) > 0.0 and utility_scalar > 0.0 else 0.0
    raw_stealth_conditioned_strict = (
        (stealth * raw_utility_scalar) ** 0.5 if stealth > 0.0 and raw_utility_scalar > 0.0 else 0.0
    )
    raw_efficiency_conditioned_strict = (
        float(efficiency_metrics["efficiency_raw"]) * raw_utility_scalar
    ) ** 0.5 if float(efficiency_metrics["efficiency_raw"]) > 0.0 and raw_utility_scalar > 0.0 else 0.0
    core_score = _geometric_mean(
        [
            detection_separability,
            robustness_scalar,
            utility_scalar,
            stealth_conditioned,
            efficiency_conditioned,
        ]
    )
    raw_core_score_strict = _geometric_mean(
        [
            detection_separability,
            raw_robustness_scalar,
            raw_utility_scalar,
            raw_stealth_conditioned_strict,
            raw_efficiency_conditioned_strict,
        ]
    )
    gate = _gate_value(
        watermarked_pass_preservation=watermarked_pass_preservation,
        negative_control_fpr=float(negative_metrics["negative_control_fpr"]),
        negative_control_support_rate=float(negative_metrics["negative_control_support_rate"]),
    )
    return {
        "detection_separability": round(detection_separability, 4),
        "robustness": robustness,
        "raw_robustness_strict": raw_robustness_strict,
        "robustness_status": _public_status(supported=bool(core_attack_public_values), value=robustness),
        "robustness_support_rate": robustness_support_rate,
        "stress_robustness": stress_robustness,
        "utility": utility,
        "raw_utility_strict": raw_utility_strict,
        "utility_status": _public_status(supported=bool(supported_utility_values), value=utility),
        "utility_support_rate": utility_support_rate,
        "stealth": round(stealth, 4),
        "efficiency": round(_clamp01(efficiency_metrics["efficiency_raw"]), 4),
        "stealth_conditioned": round(_clamp01(stealth_conditioned), 4),
        "efficiency_conditioned": round(_clamp01(efficiency_conditioned), 4),
        "core_score": round(_clamp01(core_score), 4),
        "raw_core_score_strict": round(_clamp01(raw_core_score_strict), 4),
        "negative_control_fpr": float(negative_metrics["negative_control_fpr"]),
        "negative_vs_watermarked_auroc": float(negative_metrics["negative_vs_watermarked_auroc"]),
        "negative_control_support_rate": float(negative_metrics["negative_control_support_rate"]),
        "semantic_validation_rate": round(executed_rate, 4),
        "declared_semantic_validation_rate": round(declared_rate, 4),
        "semantic_preservation_rate": round(_clamp01(semantic_preservation), 4),
        "quality_score_mean": round(quality_mean, 4),
        "watermark_retention_mean": round(retention_mean, 4),
        "attacked_detected_semantic_rate": round(_clamp01(semantic_attack), 4),
        "watermarked_pass_preservation": round(watermarked_pass_preservation, 4),
        "attacked_pass_preservation": round(attacked_pass_preservation, 4),
        "gate": round(gate, 4),
        "attack_supported_row_count": len(supported_rows),
        "attack_unsupported_row_count": len(rows) - len(supported_rows),
        "attack_support_rate": _support_rate(len(supported_rows), len(rows)),
        "score_coverage": {
            **dict(negative_metrics["coverage"]),
            "attack_breakdown": attack_breakdown,
            "attack_supported_row_count": len(supported_rows),
            "attack_unsupported_row_count": len(rows) - len(supported_rows),
            "attack_support_rate": _support_rate(len(supported_rows), len(rows)),
            "utility_factor_breakdown": utility_factors,
        },
    }


def _score_components(rows: list[BenchmarkRow], *, balance_by_source_group: bool = False) -> dict[str, Any]:
    components = _score_components_raw(rows)
    if not balance_by_source_group:
        return components

    source_groups = _group_rows(
        rows,
        lambda row: normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", ""))) or "unspecified",
    )
    if len(source_groups) <= 1:
        coverage = dict(components.get("score_coverage", {}))
        coverage.update(
            {
                "aggregation_mode": "source_balanced",
                "source_group_count": len(source_groups),
                "aggregated_source_groups": sorted(source_groups),
            }
        )
        components["score_coverage"] = coverage
        return components

    per_source = {name: _score_components_raw(group_rows) for name, group_rows in source_groups.items()}
    supported_public_sources = [
        component for component in per_source.values() if _public_core_supported(component)
    ]
    for key in SOURCE_BALANCED_COMPONENT_FIELDS:
        if key in PUBLIC_SOURCE_BALANCED_COMPONENT_FIELDS:
            if key in NULLABLE_SOURCE_BALANCED_COMPONENT_FIELDS:
                value = _mean_optional(component.get(key) for component in supported_public_sources)
                components[key] = round(value, 4) if value is not None else None
                continue
            values = [float(component.get(key, 0.0) or 0.0) for component in supported_public_sources]
            components[key] = round(_clamp01(mean(values)), 4) if values else 0.0
            continue
        if key in NULLABLE_SOURCE_BALANCED_COMPONENT_FIELDS:
            value = _mean_optional(component.get(key) for component in per_source.values())
            components[key] = round(value, 4) if value is not None else None
            continue
        values = [float(component.get(key, 0.0) or 0.0) for component in per_source.values()]
        components[key] = round(_clamp01(mean(values)), 4) if values else 0.0
    components["gate"] = round(
        _gate_value(
            watermarked_pass_preservation=float(components.get("watermarked_pass_preservation", 0.0)),
            negative_control_fpr=float(components.get("negative_control_fpr", 1.0)),
            negative_control_support_rate=float(components.get("negative_control_support_rate", 0.0)),
        ),
        4,
    )
    components["robustness_status"] = _public_status(
        supported=bool(components.get("robustness_support_rate", 0.0)) and components.get("robustness") is not None,
        value=components.get("robustness"),
    )
    components["utility_status"] = _public_status(
        supported=bool(components.get("utility_support_rate", 0.0)) and components.get("utility") is not None,
        value=components.get("utility"),
    )

    coverage = dict(components.get("score_coverage", {}))
    coverage.update(
        {
            "aggregation_mode": "source_balanced",
            "source_group_count": len(source_groups),
            "aggregated_source_groups": sorted(source_groups),
            "source_balanced_sources": {
                source_group: {
                    "row_count": len(source_groups[source_group]),
                    "detection_separability": float(component.get("detection_separability", 0.0)),
                    "robustness": float(component.get("robustness", 0.0) or 0.0),
                    "utility": float(component.get("utility", 0.0) or 0.0),
                    "stealth": float(component.get("stealth", 0.0)),
                    "efficiency": float(component.get("efficiency", 0.0)),
                    "core_score": component.get("core_score") if _public_core_supported(component) else None,
                    "raw_core_score_strict": float(component.get("raw_core_score_strict", 0.0)),
                }
                for source_group, component in per_source.items()
            },
        }
    )
    components["score_coverage"] = coverage
    return components


def _stability_from_components(
    components: Mapping[str, dict[str, Any]],
    *,
    field: str = "core_score",
) -> float | None:
    if not components or len(components) < 2:
        return None
    values: list[float] = []
    for component in components.values():
        if component.get(field) is None:
            continue
        if field == "core_score" and not _public_core_supported(component):
            continue
        values.append(_clamp01(float(component.get(field, 0.0))))
    if len(values) < 2:
        return None
    arithmetic_mean = mean(values)
    if arithmetic_mean <= 0.0:
        # An axis with enough slices but uniformly zero core performance is still
        # structurally supported; it should surface as measured zero transfer.
        return 0.0
    return round(_clamp01(_geometric_mean(values) / max(arithmetic_mean, EPSILON)), 4)


def _slice_component_map(
    groups: dict[str, list[BenchmarkRow]],
    *,
    balance_by_source_group: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        name: _score_components(group_rows, balance_by_source_group=balance_by_source_group)
        for name, group_rows in groups.items()
    }


def _family_component_map(
    model_components: Mapping[str, dict[str, Any]],
    *,
    field: str = "core_score",
) -> dict[str, dict[str, Any]]:
    grouped_values: dict[str, list[float]] = defaultdict(list)
    grouped_models: dict[str, list[str]] = defaultdict(list)
    for model_label, component in model_components.items():
        family = model_family_for_label(model_label)
        value = component.get(field)
        if value is None:
            continue
        if field == "core_score" and not _public_core_supported(component):
            continue
        grouped_values[family].append(_clamp01(float(value)))
        grouped_models[family].append(str(model_label))
    family_components: dict[str, dict[str, Any]] = {}
    for family, values in grouped_values.items():
        # Family transfer is a headline-score axis, so each family gets one vote.
        # We use an arithmetic family summary here to avoid leaking within-family
        # scale variance back into the headline score; that variance is exposed
        # separately through the released scale-consistency diagnostic.
        payload = {
            field: round(_clamp01(mean(values)) if values else 0.0, 4),
            "family": family,
            "family_model_count": len(values),
            "models": sorted(grouped_models[family]),
        }
        if field == "core_score":
            payload["robustness_status"] = "supported_nonzero"
            payload["utility_status"] = "supported_nonzero"
        family_components[family] = payload
    return family_components


def _scale_consistency_payload(
    model_components: Mapping[str, dict[str, Any]],
    *,
    field: str = "raw_core_score_strict",
) -> dict[str, Any]:
    grouped_values: dict[str, list[float]] = defaultdict(list)
    for model_label, component in model_components.items():
        value = component.get(field)
        if value is None:
            continue
        grouped_values[model_family_for_label(model_label)].append(_clamp01(float(value)))
    family_scores: dict[str, float] = {}
    for family, values in grouped_values.items():
        if len(values) < 2:
            continue
        arithmetic_mean = mean(values)
        if arithmetic_mean <= 0.0:
            family_scores[family] = 0.0
            continue
        family_scores[family] = round(
            _clamp01(_geometric_mean(values) / max(arithmetic_mean, EPSILON)),
            4,
        )
    supported_families = sorted(family_scores)
    scale_consistency = _geometric_mean(family_scores.values()) if family_scores else None
    return {
        "scale_consistency": round(scale_consistency, 4) if scale_consistency is not None else None,
        "scale_supported_families": supported_families,
        "scale_supported_family_count": len(supported_families),
        "scale_consistency_by_family": dict(family_scores),
    }


def _generalization_payload(
    *,
    source_components: dict[str, dict[str, Any]],
    task_components: dict[str, dict[str, Any]],
    language_components: dict[str, dict[str, Any]],
    family_components: dict[str, dict[str, Any]],
    raw_family_components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_stability = _stability_from_components(source_components, field="core_score")
    task_stability = _stability_from_components(task_components, field="core_score")
    language_stability = _stability_from_components(language_components, field="core_score")
    cross_family = _stability_from_components(family_components, field="core_score")
    raw_source_stability = _stability_from_components(source_components, field="raw_core_score_strict")
    raw_task_stability = _stability_from_components(task_components, field="raw_core_score_strict")
    raw_language_stability = _stability_from_components(language_components, field="raw_core_score_strict")
    raw_cross_family = _stability_from_components(raw_family_components, field="raw_core_score_strict")
    available = [
        ("source_stability", source_stability),
        ("task_stability", task_stability),
        ("language_stability", language_stability),
        ("cross_family_transfer", cross_family),
    ]
    raw_available = [
        ("source_stability", raw_source_stability),
        ("task_stability", raw_task_stability),
        ("language_stability", raw_language_stability),
        ("cross_family_transfer", raw_cross_family),
    ]
    available_values = [value for _, value in available if value is not None]
    raw_available_values = [value for _, value in raw_available if value is not None]
    if not available_values:
        generalization = None
        supported = False
        available_axes: list[str] = []
    else:
        generalization = round(_clamp01((_arithmetic_mean(available_values) or 0.0)), 4)
        supported = True
        available_axes = [name for name, value in available if value is not None]
    return {
        "generalization": generalization,
        "raw_generalization_strict": _rounded_or_none(_geometric_mean(raw_available_values)) if raw_available_values else None,
        "generalization_supported": supported,
        "generalization_available_axes": available_axes,
        "source_stability": source_stability,
        "task_stability": task_stability,
        "language_stability": language_stability,
        "cross_family_transfer": cross_family,
    }


def scorecard_for_rows(
    rows: Iterable[BenchmarkRow],
    *,
    include_generalization: bool = True,
    restrict_source_groups: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
) -> dict[str, Any]:
    materialized = _restrict_rows(rows, restrict_source_groups=restrict_source_groups)
    components = _score_components(materialized, balance_by_source_group=balance_by_source_group)
    scorecard = dict(components)

    model_groups = _group_rows(materialized, lambda row: row.model_label or "unspecified")
    source_groups = _group_rows(
        materialized,
        lambda row: normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", ""))) or "unspecified",
    )
    task_groups, folded_rows = _collapse_sparse_groups(
        _group_rows(materialized, _task_slice_key),
        min_rows=SPARSE_TASK_MIN_ROWS,
    )
    multilingual_rows = _restrict_rows(
        materialized,
        restrict_source_groups=SUITE_MULTILINGUAL_AGGREGATE_SOURCE_GROUPS,
    )
    language_groups = _group_rows(multilingual_rows, _language_slice_key)
    model_components = _slice_component_map(model_groups, balance_by_source_group=balance_by_source_group)
    source_components = _slice_component_map(source_groups, balance_by_source_group=balance_by_source_group)
    task_components = _slice_component_map(task_groups, balance_by_source_group=balance_by_source_group)
    language_components = _slice_component_map(language_groups, balance_by_source_group=balance_by_source_group)
    family_components = _family_component_map(model_components, field="core_score")
    raw_family_components = _family_component_map(model_components, field="raw_core_score_strict")
    scale_payload = _scale_consistency_payload(model_components, field="raw_core_score_strict")
    if include_generalization:
        generalization_payload = _generalization_payload(
            source_components=source_components,
            task_components=task_components,
            language_components=language_components,
            family_components=family_components,
            raw_family_components=raw_family_components,
        )
    else:
        generalization_payload = {
            "generalization": None,
            "raw_generalization_strict": None,
            "generalization_supported": False,
            "generalization_available_axes": [],
            "source_stability": None,
            "task_stability": None,
            "language_stability": None,
            "cross_family_transfer": None,
        }

    raw_robustness = float(scorecard.get("robustness", 0.0) or 0.0)
    raw_utility = float(scorecard.get("utility", 0.0) or 0.0)
    raw_stealth_conditioned = float(scorecard.get("stealth_conditioned", 0.0))
    raw_efficiency_conditioned = float(scorecard.get("efficiency_conditioned", 0.0))
    headline_core_score = round(
        _clamp01(
            _geometric_mean(
                [
                    _headline_soft_floor(float(scorecard.get("detection_separability", 0.0))),
                    _headline_soft_floor(raw_robustness),
                    _headline_soft_floor(raw_utility),
                    _headline_soft_floor(raw_stealth_conditioned),
                    _headline_soft_floor(raw_efficiency_conditioned),
                ]
            )
        ),
        4,
    )
    generalization_supported = bool(generalization_payload["generalization_supported"])
    raw_generalization = generalization_payload["generalization"]
    generalization_status = _generalization_status(
        supported=generalization_supported,
        generalization=raw_generalization,
    )
    headline_generalization = round(
        UNSUPPORTED_GENERALIZATION_NEUTRAL
        if not generalization_supported
        else _headline_soft_floor(float(raw_generalization or 0.0)),
        4,
    )
    raw_generalization_strict = float(generalization_payload.get("raw_generalization_strict") or 0.0)
    raw_composite_strict = round(
        _clamp01(
            float(scorecard.get("gate", 0.0))
            * float(scorecard.get("raw_core_score_strict", 0.0))
            * raw_generalization_strict
        ),
        4,
    )
    headline_score = round(
        _clamp01(
            float(scorecard.get("gate", 0.0))
            * _geometric_mean([headline_core_score, headline_generalization])
        ),
        4,
    )
    scorecard.update(
        {
            **generalization_payload,
            **scale_payload,
            "headline_core_score": headline_core_score,
            "headline_generalization": headline_generalization,
            "generalization_status": generalization_status,
            "raw_composite_strict": raw_composite_strict,
            HEADLINE_SCORE_FIELD: headline_score,
            "score_version": SCORE_VERSION,
        }
    )
    scorecard.pop("scale_consistency_by_family", None)
    coverage = dict(scorecard.get("score_coverage", {}))
    available_axes = list(generalization_payload.get("generalization_available_axes", []))
    all_axes = list(GENERALIZATION_AXIS_ORDER)
    coverage.update(
        {
            "model_slice_count": len(model_components),
            "source_slice_count": len(source_components),
            "task_slice_count": len(task_components),
            "language_slice_count": len(language_components),
            "family_slice_count": len(family_components),
            "folded_sparse_task_rows": folded_rows,
            "generalization_axes_used": available_axes,
            "generalization_axes_missing": [axis for axis in all_axes if axis not in available_axes],
            "scale_supported_families": list(scale_payload.get("scale_supported_families", [])),
            "scale_supported_family_count": int(scale_payload.get("scale_supported_family_count", 0)),
            "scale_consistency_by_family": dict(scale_payload.get("scale_consistency_by_family", {})),
        }
    )
    scale_payload = dict(scale_payload)
    scale_payload.pop("scale_consistency_by_family", None)
    scorecard["score_coverage"] = coverage
    return scorecard
