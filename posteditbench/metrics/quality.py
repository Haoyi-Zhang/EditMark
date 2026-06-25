from __future__ import annotations

from ..utils import edit_distance_ratio, jaccard, tokenize


def lexical_preservation(original: str, mutated: str) -> float:
    return jaccard(tokenize(original), tokenize(mutated))


def structural_similarity(original: str, mutated: str) -> float:
    return edit_distance_ratio(original, mutated)


def overall_quality_score(original: str, mutated: str) -> float:
    lexical = lexical_preservation(original, mutated)
    structural = structural_similarity(original, mutated)
    return max(0.0, min(1.0, 0.5 * lexical + 0.5 * structural))
