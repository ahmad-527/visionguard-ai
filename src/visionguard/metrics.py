"""VisionGuard-owned metric contracts with explicit undefined cases."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class MetricInputError(ValueError):
    """Raised for malformed labels, scores, predictions, or shapes."""


@dataclass(frozen=True)
class MetricResult:
    """A metric value or a documented undefined result."""

    name: str
    level: str
    value: float | None
    status: str
    reason: str | None = None


def _binary_labels(values: Sequence[int], location: str) -> list[int]:
    labels = list(values)
    if not labels:
        raise MetricInputError(f"{location} must not be empty")
    if any(isinstance(item, bool) or item not in {0, 1} for item in labels):
        raise MetricInputError(f"{location} must contain only integer 0/1 values")
    return labels


def _scores(values: Sequence[float], expected: int) -> list[float]:
    scores = [float(value) for value in values]
    if len(scores) != expected:
        raise MetricInputError("labels and scores must have identical lengths")
    if any(not math.isfinite(value) for value in scores):
        raise MetricInputError("scores must be finite")
    return scores


def binary_auroc(
    labels: Sequence[int], scores: Sequence[float], *, level: str = "image"
) -> MetricResult:
    """Compute tie-aware binary AUROC using the Mann-Whitney rank identity."""

    checked_labels = _binary_labels(labels, "labels")
    checked_scores = _scores(scores, len(checked_labels))
    positives = sum(checked_labels)
    negatives = len(checked_labels) - positives
    if positives == 0 or negatives == 0:
        return MetricResult(
            "auroc",
            level,
            None,
            "undefined",
            "AUROC requires at least one positive and one negative label",
        )
    ordered = sorted(enumerate(checked_scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, checked_labels, strict=True) if label == 1
    )
    value = (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return MetricResult("auroc", level, value, "defined")


def binary_f1(
    labels: Sequence[int], predictions: Sequence[int], *, level: str = "image"
) -> MetricResult:
    """Compute binary F1 with an explicit zero-positive-prediction policy."""

    checked_labels = _binary_labels(labels, "labels")
    checked_predictions = _binary_labels(predictions, "predictions")
    if len(checked_labels) != len(checked_predictions):
        raise MetricInputError("labels and predictions must have identical lengths")
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(checked_labels, checked_predictions, strict=True)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(checked_labels, checked_predictions, strict=True)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(checked_labels, checked_predictions, strict=True)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return MetricResult(
            "f1",
            level,
            None,
            "undefined",
            "F1 is undefined when labels and predictions contain no positives",
        )
    return MetricResult("f1", level, 2 * true_positive / denominator, "defined")


def flatten_pixel_data(
    labels: Sequence[Sequence[int]], scores: Sequence[Sequence[float]]
) -> tuple[list[int], list[float]]:
    """Validate matching 2-D pixel arrays and flatten them for metric functions."""

    if len(labels) != len(scores) or not labels:
        raise MetricInputError(
            "pixel label and score row counts must match and be nonzero"
        )
    flat_labels: list[int] = []
    flat_scores: list[float] = []
    for label_row, score_row in zip(labels, scores, strict=True):
        if len(label_row) != len(score_row) or not label_row:
            raise MetricInputError(
                "pixel label and score shapes must match with nonempty rows"
            )
        flat_labels.extend(_binary_labels(label_row, "pixel labels"))
        flat_scores.extend(_scores(score_row, len(label_row)))
    return flat_labels, flat_scores
