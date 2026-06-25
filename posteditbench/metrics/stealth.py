from __future__ import annotations

from ..utils import edit_distance_ratio, line_count


def watermark_footprint(original: str, watermarked: str) -> float:
    return 1.0 - edit_distance_ratio(original, watermarked)


def line_impact(original: str, watermarked: str) -> float:
    original_lines = max(line_count(original), 1)
    return abs(line_count(watermarked) - line_count(original)) / original_lines


def stealth_score(original: str, watermarked: str) -> float:
    footprint = watermark_footprint(original, watermarked)
    impact = line_impact(original, watermarked)
    return max(0.0, min(1.0, 1.0 - 0.7 * footprint - 0.3 * impact))
