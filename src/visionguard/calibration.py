"""Normal-only threshold calibration utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from visionguard.experiment import ThresholdConfig


class CalibrationError(ValueError):
    """Raised when normal-only calibration inputs violate the contract."""


@dataclass(frozen=True)
class OrderStatisticCalibration:
    """Highest finite order statistic and its exchangeable marginal coverage."""

    threshold: float
    sample_count: int
    order_rank: int
    marginal_coverage: float
    guarantee: str = "exchangeability-dependent marginal coverage only"


def highest_order_statistic(
    normal_scores: Sequence[float], *, minimum_samples: int
) -> OrderStatisticCalibration:
    """Calibrate from normal-only independent units without distribution fitting."""

    if len(normal_scores) < minimum_samples:
        raise CalibrationError(
            f"Calibration requires at least {minimum_samples} scores; "
            f"received {len(normal_scores)}"
        )
    values = [float(score) for score in normal_scores]
    if any(not math.isfinite(score) for score in values):
        raise CalibrationError("Calibration scores must be finite")
    count = len(values)
    return OrderStatisticCalibration(
        threshold=max(values),
        sample_count=count,
        order_rank=count,
        marginal_coverage=count / (count + 1),
    )


def per_image_pixel_maxima(pixel_maps: Sequence[Sequence[float]]) -> list[float]:
    """Reduce correlated pixel maps to one calibration unit per image."""

    if not pixel_maps:
        raise CalibrationError("Pixel calibration requires at least one image")
    maxima: list[float] = []
    for pixel_map in pixel_maps:
        values = [float(score) for score in pixel_map]
        if not values:
            raise CalibrationError("Each pixel calibration image must be non-empty")
        if any(not math.isfinite(score) for score in values):
            raise CalibrationError("Calibration scores must be finite")
        maxima.append(max(values))
    return maxima


def empirical_quantile(
    normal_scores: Sequence[float], config: ThresholdConfig
) -> float:
    """Compute a deterministic linearly interpolated empirical quantile.

    Inputs are validation-normal scores only. The caller is responsible for keeping
    image and pixel score populations separate.
    """

    if config.method != "empirical_quantile":
        raise CalibrationError(f"Unsupported calibration method: {config.method}")
    if len(normal_scores) < config.minimum_samples:
        raise CalibrationError(
            f"Calibration requires at least {config.minimum_samples} scores; "
            f"received {len(normal_scores)}"
        )
    values = [float(score) for score in normal_scores]
    if any(not math.isfinite(score) for score in values):
        raise CalibrationError("Calibration scores must be finite")
    values.sort()
    position = (len(values) - 1) * config.quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
