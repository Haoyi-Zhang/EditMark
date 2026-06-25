from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from ..models import BenchmarkRow, supported_attack_rows


def attack_robustness(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    rows = supported_attack_rows(rows)
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.attack_name].append(row)
    summary: dict[str, float] = {}
    for attack_name, attack_rows in grouped.items():
        summary[attack_name] = mean(1.0 if row.attacked_detected else 0.0 for row in attack_rows)
    summary["overall"] = mean(summary.values()) if summary else 0.0
    return summary


def clean_retention(rows: Iterable[BenchmarkRow]) -> float:
    rows = supported_attack_rows(rows)
    if not rows:
        return 0.0
    return mean(1.0 if row.clean_detected else 0.0 for row in rows)


def watermarked_retention(rows: Iterable[BenchmarkRow]) -> float:
    rows = supported_attack_rows(rows)
    if not rows:
        return 0.0
    return mean(1.0 if row.positive_detected else 0.0 for row in rows)


def mean_watermark_retention(rows: Iterable[BenchmarkRow]) -> float:
    rows = supported_attack_rows(rows)
    if not rows:
        return 0.0
    return mean(row.watermark_retention for row in rows)


def mean_robustness_score(rows: Iterable[BenchmarkRow]) -> float:
    rows = supported_attack_rows(rows)
    if not rows:
        return 0.0
    return mean(row.robustness_score for row in rows)


def semantic_rows(rows: Iterable[BenchmarkRow]) -> list[BenchmarkRow]:
    return [
        row
        for row in supported_attack_rows(rows)
        if row.semantic_validation_available and row.semantic_preserving is True
    ]


def semantic_validation_rate(rows: Iterable[BenchmarkRow]) -> float:
    rows = supported_attack_rows(rows)
    if not rows:
        return 0.0
    return mean(1.0 if row.semantic_validation_available else 0.0 for row in rows)


def semantic_preservation_rate(rows: Iterable[BenchmarkRow]) -> float:
    rows = [row for row in supported_attack_rows(rows) if row.semantic_validation_available]
    if not rows:
        return 0.0
    return mean(1.0 if row.semantic_preserving else 0.0 for row in rows)


def semantic_clean_retention(rows: Iterable[BenchmarkRow]) -> float:
    rows = semantic_rows(rows)
    if not rows:
        return 0.0
    return mean(1.0 if row.clean_detected else 0.0 for row in rows)


def semantic_watermarked_retention(rows: Iterable[BenchmarkRow]) -> float:
    rows = semantic_rows(rows)
    if not rows:
        return 0.0
    return mean(1.0 if row.positive_detected else 0.0 for row in rows)


def semantic_attack_robustness(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    rows = semantic_rows(rows)
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.attack_name].append(row)
    summary: dict[str, float] = {}
    for attack_name, attack_rows in grouped.items():
        summary[attack_name] = mean(1.0 if row.attacked_detected else 0.0 for row in attack_rows)
    summary["overall"] = mean(summary.values()) if summary else 0.0
    return summary
