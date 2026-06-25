from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from statistics import mean
from typing import Iterable

from ..models import BenchmarkRow, attack_row_supported


def attack_breakdown(rows: Iterable[BenchmarkRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.attack_name].append(row)
    breakdown: dict[str, dict[str, float]] = {}
    for attack_name, attack_rows in grouped.items():
        supported_rows = [row for row in attack_rows if attack_row_supported(row)]
        breakdown[attack_name] = {
            "count": float(len(attack_rows)),
            "supported_count": float(len(supported_rows)),
            "support_rate": round(float(len(supported_rows)) / max(1.0, float(len(attack_rows))), 4),
            "mean_detection_score": mean(row.attacked_score for row in supported_rows) if supported_rows else 0.0,
            "attacked_detect_rate": mean(1.0 if row.attacked_detected else 0.0 for row in supported_rows) if supported_rows else 0.0,
            "avg_quality": mean(row.quality_score for row in supported_rows) if supported_rows else 0.0,
            "avg_stealth": mean(row.stealth_score for row in supported_rows) if supported_rows else 0.0,
            "mean_watermark_retention": mean(row.watermark_retention for row in supported_rows) if supported_rows else 0.0,
            "mean_robustness_score": mean(row.robustness_score for row in supported_rows) if supported_rows else 0.0,
            "mean_attack_outcome_score": mean(row.robustness_score for row in supported_rows) if supported_rows else 0.0,
        }
    return breakdown


def language_breakdown(rows: Iterable[BenchmarkRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.language].append(row)
    breakdown: dict[str, dict[str, float]] = {}
    for language, language_rows in grouped.items():
        supported_rows = [row for row in language_rows if attack_row_supported(row)]
        semantic_rows = [row for row in supported_rows if row.semantic_validation_available]
        breakdown[language] = {
            "count": float(len(language_rows)),
            "supported_count": float(len(supported_rows)),
            "support_rate": round(float(len(supported_rows)) / max(1.0, float(len(language_rows))), 4),
            "mean_detection_score": mean(row.attacked_score for row in supported_rows) if supported_rows else 0.0,
            "attacked_detect_rate": mean(1.0 if row.attacked_detected else 0.0 for row in supported_rows) if supported_rows else 0.0,
            "avg_quality": mean(row.quality_score for row in supported_rows) if supported_rows else 0.0,
            "avg_stealth": mean(row.stealth_score for row in supported_rows) if supported_rows else 0.0,
            "mean_watermark_retention": mean(row.watermark_retention for row in supported_rows) if supported_rows else 0.0,
            "mean_robustness_score": mean(row.robustness_score for row in supported_rows) if supported_rows else 0.0,
            "mean_attack_outcome_score": mean(row.robustness_score for row in supported_rows) if supported_rows else 0.0,
            "semantic_validation_rate": mean(1.0 if row.semantic_validation_available else 0.0 for row in supported_rows) if supported_rows else 0.0,
            "semantic_preservation_rate": mean(1.0 if row.semantic_preserving else 0.0 for row in semantic_rows) if semantic_rows else 0.0,
        }
    return breakdown


def confidence_band(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    if not samples:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(samples), "min": min(samples), "max": max(samples)}


def budget_curve_summary(rows: Iterable[BenchmarkRow]) -> dict[str, list[dict[str, float | int]]]:
    grouped: dict[str, dict[int, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not attack_row_supported(row):
            continue
        metadata = row.metadata if isinstance(row.metadata, Mapping) else {}
        curve = metadata.get("budget_curve")
        if not curve and isinstance(metadata.get("attack_metadata"), Mapping):
            curve = metadata["attack_metadata"].get("budget_curve")
        if not curve:
            continue
        for point in curve:
            try:
                budget = int(point.get("budget", 0))
            except Exception:
                continue
            grouped[row.attack_name][budget].append(point)

    summary: dict[str, list[dict[str, float | int]]] = {}
    for attack_name, budget_points in grouped.items():
        summary[attack_name] = []
        for budget in sorted(budget_points):
            points = budget_points[budget]
            summary[attack_name].append(
                {
                    "budget": budget,
                    "count": len(points),
                    "mean_detector_score": round(mean(float(point.get("detector_score", 0.0)) for point in points), 4),
                    "mean_quality_score": round(mean(float(point.get("quality_score", 0.0)) for point in points), 4),
                    "semantic_preserving_rate": round(
                        mean(
                            1.0 if value else 0.0
                            for value in [point.get("semantic_preserving") for point in points]
                            if value is not None
                        ),
                        4,
                    )
                    if any(point.get("semantic_preserving") is not None for point in points)
                    else 0.0,
                }
            )
    return summary
