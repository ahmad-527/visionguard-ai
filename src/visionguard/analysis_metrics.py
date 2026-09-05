"""Deterministic descriptive metrics for frozen benchmark evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from visionguard.benchmark_metrics import Float16AuProAccumulator
from visionguard.metrics import MetricResult, binary_auroc


class AnalysisMetricError(ValueError):
    """Raised when comparative-analysis metric inputs are malformed."""


@dataclass
class ConfusionCounts:
    """Binary confusion counts with explicit true-negative preservation."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def add(self, other: ConfusionCounts) -> None:
        """Add another set of counts in place."""

        self.true_positive += other.true_positive
        self.false_positive += other.false_positive
        self.true_negative += other.true_negative
        self.false_negative += other.false_negative

    def as_dict(self) -> dict[str, int]:
        """Return stable JSON field names."""

        return asdict(self)


def _checked_binary(values: Sequence[int], name: str) -> list[int]:
    checked = list(values)
    if not checked:
        raise AnalysisMetricError(f"{name} must not be empty")
    if any(isinstance(value, bool) or value not in {0, 1} for value in checked):
        raise AnalysisMetricError(f"{name} must contain only integer 0/1 values")
    return checked


def _rate(numerator: int, denominator: int, *, name: str) -> dict[str, Any]:
    if denominator == 0:
        return {
            "name": name,
            "status": "undefined",
            "value": None,
            "reason": f"{name} denominator is zero",
        }
    return {
        "name": name,
        "status": "defined",
        "value": numerator / denominator,
        "reason": None,
    }


def metric_payload(result: MetricResult) -> dict[str, Any]:
    """Convert the repository metric value object to stable JSON."""

    return asdict(result)


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], scores: Sequence[float]
) -> dict[str, Any]:
    """Calculate frozen-threshold image metrics and the existing AUROC."""

    checked_labels = _checked_binary(labels, "labels")
    checked_predictions = _checked_binary(predictions, "predictions")
    checked_scores = [float(value) for value in scores]
    if not (len(checked_labels) == len(checked_predictions) == len(checked_scores)):
        raise AnalysisMetricError(
            "labels, predictions, and scores must have identical lengths"
        )
    if any(not math.isfinite(value) for value in checked_scores):
        raise AnalysisMetricError("scores must be finite")

    counts = ConfusionCounts()
    for label, prediction in zip(checked_labels, checked_predictions, strict=True):
        if label == 1 and prediction == 1:
            counts.true_positive += 1
        elif label == 0 and prediction == 1:
            counts.false_positive += 1
        elif label == 0 and prediction == 0:
            counts.true_negative += 1
        else:
            counts.false_negative += 1

    f1_denominator = (
        2 * counts.true_positive + counts.false_positive + counts.false_negative
    )
    return {
        "confusion": counts.as_dict(),
        "sensitivity": _rate(
            counts.true_positive,
            counts.true_positive + counts.false_negative,
            name="sensitivity",
        ),
        "specificity": _rate(
            counts.true_negative,
            counts.true_negative + counts.false_positive,
            name="specificity",
        ),
        "precision": _rate(
            counts.true_positive,
            counts.true_positive + counts.false_positive,
            name="precision",
        ),
        "image_f1": _rate(
            2 * counts.true_positive,
            f1_denominator,
            name="image_f1",
        ),
        "image_auroc": metric_payload(
            binary_auroc(checked_labels, checked_scores, level="image")
        ),
    }


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise AnalysisMetricError("quantile probability must be in [0, 1]")
    if not sorted_values:
        raise AnalysisMetricError("quantiles require at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - weight)
        + float(sorted_values[upper]) * weight
    )


def describe_scores(values: Sequence[float]) -> dict[str, Any]:
    """Describe finite values using a documented deterministic quantile rule."""

    checked = [float(value) for value in values]
    if not checked:
        raise AnalysisMetricError("score distribution must not be empty")
    if any(not math.isfinite(value) for value in checked):
        raise AnalysisMetricError("score distribution must be finite")
    ordered = sorted(checked)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q1": _linear_quantile(ordered, 0.25),
        "median": _linear_quantile(ordered, 0.5),
        "q3": _linear_quantile(ordered, 0.75),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
        "population_standard_deviation": statistics.pstdev(ordered),
        "quantile_method": "linear_index_n_minus_1",
    }


def score_distributions(
    labels: Sequence[int], scores: Sequence[float]
) -> dict[str, dict[str, Any]]:
    """Separate public image anomaly scores by ground-truth image label."""

    checked_labels = _checked_binary(labels, "labels")
    checked_scores = [float(value) for value in scores]
    if len(checked_labels) != len(checked_scores):
        raise AnalysisMetricError("labels and scores must have identical lengths")
    return {
        "normal_public_images": describe_scores(
            [
                score
                for label, score in zip(checked_labels, checked_scores, strict=True)
                if label == 0
            ]
        ),
        "anomalous_public_images": describe_scores(
            [
                score
                for label, score in zip(checked_labels, checked_scores, strict=True)
                if label == 1
            ]
        ),
    }


def _histogram_quantile(
    values: Any, counts: Any, probability: float, total: int
) -> float:
    """Return the expanded-sample linear quantile without materializing pixels."""

    np = _numpy()
    position = (total - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    cumulative = np.cumsum(counts, dtype=np.uint64)
    lower_bin = int(np.searchsorted(cumulative, lower_index + 1, side="left"))
    upper_bin = int(np.searchsorted(cumulative, upper_index + 1, side="left"))
    if lower_bin == upper_bin:
        return float(values[lower_bin])
    weight = position - lower_index
    return float(values[lower_bin]) * (1.0 - weight) + float(values[upper_bin]) * weight


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - analysis environment contract
        raise AnalysisMetricError("Map analysis requires NumPy") from exc
    return np


class Float16PixelAnalysisAccumulator:
    """Accumulate exact frozen-map pixel metrics and diagnostic distributions."""

    def __init__(self) -> None:
        np = _numpy()
        self._np = np
        self._positive_histogram = np.zeros(65536, dtype=np.uint64)
        self._negative_histogram = np.zeros(65536, dtype=np.uint64)
        self._confusion = ConfusionCounts()
        self._au_pro = Float16AuProAccumulator()
        self._image_count = 0

    @property
    def confusion(self) -> ConfusionCounts:
        """Return a copy of accumulated pixel counts."""

        return ConfusionCounts(**self._confusion.as_dict())

    def update(self, labels: Any, scores: Any, predictions: Any) -> ConfusionCounts:
        """Add one public mask, continuous float16 map, and frozen binary map."""

        np = self._np
        label_array = np.asarray(labels)
        score_array = np.asarray(scores)
        prediction_array = np.asarray(predictions)
        if (
            label_array.ndim != 2
            or score_array.ndim != 2
            or prediction_array.ndim != 2
            or label_array.shape != score_array.shape
            or label_array.shape != prediction_array.shape
            or label_array.size == 0
        ):
            raise AnalysisMetricError(
                "pixel labels, scores, and predictions must be equal "
                "nonempty 2-D arrays"
            )
        if score_array.dtype != np.float16:
            raise AnalysisMetricError("continuous evidence maps must be float16")
        if not bool(np.isfinite(score_array).all()):
            raise AnalysisMetricError("continuous evidence maps must be finite")
        if not bool(np.isin(label_array, (0, 1, False, True)).all()):
            raise AnalysisMetricError("pixel labels must be binary")
        if not bool(np.isin(prediction_array, (0, 1, False, True)).all()):
            raise AnalysisMetricError("thresholded evidence maps must be binary")

        binary_labels = label_array.astype(bool, copy=False)
        binary_predictions = prediction_array.astype(bool, copy=False)
        counts = ConfusionCounts(
            true_positive=int(np.count_nonzero(binary_labels & binary_predictions)),
            false_positive=int(np.count_nonzero(~binary_labels & binary_predictions)),
            true_negative=int(np.count_nonzero(~binary_labels & ~binary_predictions)),
            false_negative=int(np.count_nonzero(binary_labels & ~binary_predictions)),
        )
        self._confusion.add(counts)
        codes = score_array.view(np.uint16)
        self._positive_histogram += np.bincount(
            codes[binary_labels], minlength=65536
        ).astype(np.uint64, copy=False)
        self._negative_histogram += np.bincount(
            codes[~binary_labels], minlength=65536
        ).astype(np.uint64, copy=False)
        self._au_pro.update(binary_labels, score_array)
        self._image_count += 1
        return counts

    def _ordered_histogram(self, histogram: Any) -> tuple[Any, Any]:
        np = self._np
        used_codes = np.flatnonzero(histogram).astype(np.uint16)
        values = used_codes.view(np.float16).astype(np.float64)
        order = np.argsort(values, kind="stable")
        return values[order], histogram[used_codes][order]

    def _histogram_description(self, histogram: Any) -> dict[str, Any]:
        np = self._np
        total = int(histogram.sum(dtype=np.uint64))
        if total == 0:
            return {
                "count": 0,
                "status": "undefined",
                "reason": "pixel class is absent",
            }
        values, counts = self._ordered_histogram(histogram)
        weights = counts.astype(np.float64)
        mean = float(np.sum(values * weights) / total)
        variance = float(np.sum(((values - mean) ** 2) * weights) / total)
        return {
            "count": total,
            "status": "defined",
            "minimum": float(values[0]),
            "q1": _histogram_quantile(values, counts, 0.25, total),
            "median": _histogram_quantile(values, counts, 0.5, total),
            "q3": _histogram_quantile(values, counts, 0.75, total),
            "maximum": float(values[-1]),
            "mean": mean,
            "population_standard_deviation": math.sqrt(max(variance, 0.0)),
            "quantile_method": "linear_index_n_minus_1",
        }

    def _pixel_auroc(self) -> dict[str, Any]:
        np = self._np
        positives = int(self._positive_histogram.sum(dtype=np.uint64))
        negatives = int(self._negative_histogram.sum(dtype=np.uint64))
        if positives == 0 or negatives == 0:
            return {
                "name": "pixel_auroc_diagnostic",
                "level": "pixel",
                "status": "undefined",
                "value": None,
                "reason": "pixel AUROC requires both mask classes",
            }
        combined = self._positive_histogram + self._negative_histogram
        used_codes = np.flatnonzero(combined).astype(np.uint16)
        code_values = used_codes.view(np.float16).astype(np.float64)
        order = np.argsort(code_values, kind="stable")
        ordered_codes = used_codes[order]
        ordered_values = code_values[order]
        negative_before = 0
        concordant = 0.0
        index = 0
        while index < len(ordered_codes):
            value = ordered_values[index]
            stop = index + 1
            while stop < len(ordered_codes) and ordered_values[stop] == value:
                stop += 1
            tied_codes = ordered_codes[index:stop]
            positive_count = int(
                self._positive_histogram[tied_codes].sum(dtype=np.uint64)
            )
            negative_count = int(
                self._negative_histogram[tied_codes].sum(dtype=np.uint64)
            )
            concordant += positive_count * (negative_before + 0.5 * negative_count)
            negative_before += negative_count
            index = stop
        return {
            "name": "pixel_auroc_diagnostic",
            "level": "pixel",
            "status": "defined",
            "value": concordant / (positives * negatives),
            "reason": None,
        }

    def result(self) -> dict[str, Any]:
        """Return exact aggregate pixel results for one category/seed cell."""

        if self._image_count == 0:
            raise AnalysisMetricError("pixel analysis requires at least one image")
        counts = self._confusion
        f1_denominator = (
            2 * counts.true_positive + counts.false_positive + counts.false_negative
        )
        return {
            "confusion": counts.as_dict(),
            "pixel_precision": _rate(
                counts.true_positive,
                counts.true_positive + counts.false_positive,
                name="pixel_precision",
            ),
            "pixel_sensitivity": _rate(
                counts.true_positive,
                counts.true_positive + counts.false_negative,
                name="pixel_sensitivity",
            ),
            "pixel_specificity": _rate(
                counts.true_negative,
                counts.true_negative + counts.false_positive,
                name="pixel_specificity",
            ),
            "pixel_f1": _rate(
                2 * counts.true_positive,
                f1_denominator,
                name="pixel_f1",
            ),
            "au_pro_0.05": metric_payload(self._au_pro.result(fpr_limit=0.05)),
            "pixel_auroc_diagnostic": self._pixel_auroc(),
            "continuous_score_distributions": {
                "ground_truth_anomaly_pixels": self._histogram_description(
                    self._positive_histogram
                ),
                "ground_truth_background_pixels": self._histogram_description(
                    self._negative_histogram
                ),
            },
        }


def localization_diagnostics(
    labels: Any, scores: Any, predictions: Any
) -> dict[str, Any]:
    """Assign documented, non-exclusive localization-failure indicators."""

    np = _numpy()
    label_array = np.asarray(labels)
    score_array = np.asarray(scores)
    prediction_array = np.asarray(predictions)
    if (
        label_array.ndim != 2
        or score_array.ndim != 2
        or prediction_array.ndim != 2
        or label_array.shape != score_array.shape
        or label_array.shape != prediction_array.shape
        or label_array.size == 0
    ):
        raise AnalysisMetricError(
            "localization inputs must be equal nonempty 2-D arrays"
        )
    try:
        valid_scores = bool(np.isfinite(score_array).all())
    except TypeError as exc:
        raise AnalysisMetricError("localization scores must be finite numbers") from exc
    if not valid_scores:
        raise AnalysisMetricError("localization scores must be finite numbers")
    if not bool(np.isin(label_array, (0, 1, False, True)).all()):
        raise AnalysisMetricError("localization labels must be binary")
    if not bool(np.isin(prediction_array, (0, 1, False, True)).all()):
        raise AnalysisMetricError("localization predictions must be binary")
    label_array = label_array.astype(bool, copy=False)
    prediction_array = prediction_array.astype(bool, copy=False)
    true_positive = int(np.count_nonzero(label_array & prediction_array))
    false_positive_mask = ~label_array & prediction_array
    false_positive = int(np.count_nonzero(false_positive_mask))
    true_negative = int(np.count_nonzero(~label_array & ~prediction_array))
    false_negative = int(np.count_nonzero(label_array & ~prediction_array))
    ground_truth_positive = true_positive + false_negative
    predicted_positive = true_positive + false_positive
    continuous_minimum = float(np.min(score_array))
    continuous_maximum = float(np.max(score_array))

    flags: list[str] = []
    if ground_truth_positive > 0 and true_positive == 0:
        flags.append("missed_anomaly")
    if (
        ground_truth_positive > 0
        and true_positive > 0
        and false_negative > true_positive
    ):
        flags.append("under_localization")
    if (
        ground_truth_positive > 0
        and true_positive > 0
        and false_positive > true_positive
    ):
        flags.append("over_localization")

    height, width = label_array.shape
    row_split = height // 2
    column_split = width // 2
    quadrants = (
        false_positive_mask[:row_split, :column_split],
        false_positive_mask[:row_split, column_split:],
        false_positive_mask[row_split:, :column_split],
        false_positive_mask[row_split:, column_split:],
    )
    if all(quadrant.size and bool(np.any(quadrant)) for quadrant in quadrants):
        flags.append("diffuse_false_positive_map")

    continuous_nonconstant = continuous_maximum > continuous_minimum
    total_pixels = int(label_array.size)
    if continuous_nonconstant and (
        (ground_truth_positive > 0 and predicted_positive == 0)
        or predicted_positive == total_pixels
    ):
        flags.append("threshold_collapse")
    if not continuous_nonconstant and ground_truth_positive > 0 and true_positive == 0:
        flags.append("constant_continuous_map")

    return {
        "confusion": ConfusionCounts(
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
        ).as_dict(),
        "ground_truth_positive_pixels": ground_truth_positive,
        "predicted_positive_pixels": predicted_positive,
        "total_pixels": total_pixels,
        "continuous_minimum": continuous_minimum,
        "continuous_maximum": continuous_maximum,
        "flags": flags,
    }
