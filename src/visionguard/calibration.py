"""Normal-only threshold calibration utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

from visionguard.experiment import ThresholdConfig


class CalibrationError(ValueError):
    """Raised when normal-only calibration inputs violate the contract."""


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
