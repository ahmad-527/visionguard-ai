"""VisionGuard-owned metric contracts with explicit undefined cases."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeAlias

BinaryImage: TypeAlias = Sequence[Sequence[int]]
ScoreImage: TypeAlias = Sequence[Sequence[float]]


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


def _checked_images(
    labels: Sequence[BinaryImage], scores: Sequence[ScoreImage]
) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
    if not labels or len(labels) != len(scores):
        raise MetricInputError("label and score image counts must match and be nonzero")
    checked_labels: list[list[list[int]]] = []
    checked_scores: list[list[list[float]]] = []
    for label_image, score_image in zip(labels, scores, strict=True):
        if not label_image or len(label_image) != len(score_image):
            raise MetricInputError("label and score image shapes must match")
        label_rows: list[list[int]] = []
        score_rows: list[list[float]] = []
        for label_row, score_row in zip(label_image, score_image, strict=True):
            if not label_row or len(label_row) != len(score_row):
                raise MetricInputError("label and score image shapes must match")
            label_rows.append(_binary_labels(label_row, "pixel labels"))
            score_rows.append(_scores(score_row, len(label_row)))
        checked_labels.append(label_rows)
        checked_scores.append(score_rows)
    return checked_labels, checked_scores


def _components(image: list[list[int]]) -> list[set[tuple[int, int]]]:
    height, width = len(image), len(image[0])
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if image[row][col] == 1
    }
    components: list[set[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def au_pro(
    labels: Sequence[BinaryImage],
    scores: Sequence[ScoreImage],
    *,
    fpr_limit: float = 0.05,
) -> MetricResult:
    """Compute tie-grouped, four-connected AU-PRO normalized over an FPR range."""

    if not 0.0 < fpr_limit <= 1.0:
        raise MetricInputError("fpr_limit must be in (0, 1]")
    checked_labels, checked_scores = _checked_images(labels, scores)
    regions: list[tuple[int, set[tuple[int, int]]]] = []
    normal_pixels = 0
    unique_scores: set[float] = set()
    for image_index, (label_image, score_image) in enumerate(
        zip(checked_labels, checked_scores, strict=True)
    ):
        regions.extend(
            (image_index, component) for component in _components(label_image)
        )
        for label_row, score_row in zip(label_image, score_image, strict=True):
            normal_pixels += sum(label == 0 for label in label_row)
            unique_scores.update(score_row)
    if not regions or normal_pixels == 0:
        return MetricResult(
            "au_pro_0.05",
            "pixel_region",
            None,
            "undefined",
            "AU-PRO requires anomalous regions and normal pixels",
        )

    points = [(0.0, 0.0)]
    for threshold in sorted(unique_scores, reverse=True):
        false_positives = 0
        predicted: list[set[tuple[int, int]]] = []
        for label_image, score_image in zip(
            checked_labels, checked_scores, strict=True
        ):
            image_prediction: set[tuple[int, int]] = set()
            for row, (label_row, score_row) in enumerate(
                zip(label_image, score_image, strict=True)
            ):
                for col, (label, score) in enumerate(
                    zip(label_row, score_row, strict=True)
                ):
                    if score >= threshold:
                        image_prediction.add((row, col))
                        false_positives += label == 0
            predicted.append(image_prediction)
        pro = sum(
            len(component & predicted[index]) / len(component)
            for index, component in regions
        ) / len(regions)
        fpr = false_positives / normal_pixels
        points.append((fpr, pro))
        if fpr >= fpr_limit:
            break

    clipped: list[tuple[float, float]] = [points[0]]
    for current in points[1:]:
        previous = clipped[-1]
        if current[0] <= fpr_limit:
            clipped.append(current)
            if current[0] == fpr_limit:
                break
            continue
        if current[0] == previous[0]:
            clipped[-1] = (previous[0], max(previous[1], current[1]))
            continue
        weight = (fpr_limit - previous[0]) / (current[0] - previous[0])
        clipped.append((fpr_limit, previous[1] + weight * (current[1] - previous[1])))
        break
    if clipped[-1][0] < fpr_limit:
        clipped.append((fpr_limit, clipped[-1][1]))
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in pairwise(clipped)
    )
    return MetricResult("au_pro_0.05", "pixel_region", area / fpr_limit, "defined")
