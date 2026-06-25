from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float


def classification_metrics(labels: list[bool], predictions: list[bool]) -> ClassificationMetrics:
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    tn = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return ClassificationMetrics(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def threshold_prediction(score: float, threshold: float) -> bool:
    return score >= threshold
