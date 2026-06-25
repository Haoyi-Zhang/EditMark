from __future__ import annotations

from math import comb
from typing import Iterable


def pass_at_k(successes: int, samples: int, k: int) -> float:
    if samples <= 0:
        return 0.0
    k = max(1, min(k, samples))
    if successes <= 0:
        return 0.0
    if samples - successes < k:
        return 1.0
    return 1.0 - (comb(samples - successes, k) / comb(samples, k))


def binary_auroc(negative_scores: Iterable[float], positive_scores: Iterable[float]) -> float:
    negatives = [float(score) for score in negative_scores]
    positives = [float(score) for score in positive_scores]
    if not negatives or not positives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def calculate_stem(pass_k_score: float, auroc: float, perplexity_reference: float, perplexity_watermarked: float) -> float:
    if perplexity_reference <= 0:
        return 0.0
    correctness = float(pass_k_score) / 2.0
    detectability = float(auroc)
    naturalness = 1.0 - (abs(float(perplexity_watermarked) - float(perplexity_reference)) / float(perplexity_reference))
    return round((correctness + detectability + naturalness) / 3.0, 4)
