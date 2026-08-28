"""Streaming metrics for official-format MVTec anomaly maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from visionguard.metrics import MetricInputError, MetricResult


@dataclass
class BinaryCountAccumulator:
    """Accumulate binary classification counts without retaining every pixel."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, labels: Any, predictions: Any) -> None:
        """Add equally shaped NumPy boolean arrays to the running counts."""

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - ML environment contract
            raise MetricInputError("Streaming pixel metrics require NumPy") from exc
        label_array = np.asarray(labels)
        prediction_array = np.asarray(predictions)
        if label_array.shape != prediction_array.shape or label_array.size == 0:
            raise MetricInputError("Pixel labels and predictions must have equal shape")
        if not np.all(np.isin(label_array, (0, 1, False, True))):
            raise MetricInputError("Pixel labels must be binary")
        if not np.all(np.isin(prediction_array, (0, 1, False, True))):
            raise MetricInputError("Pixel predictions must be binary")
        checked_labels = label_array.astype(bool, copy=False)
        checked_predictions = prediction_array.astype(bool, copy=False)
        self.true_positive += int(
            np.count_nonzero(checked_labels & checked_predictions)
        )
        self.false_positive += int(
            np.count_nonzero(~checked_labels & checked_predictions)
        )
        self.false_negative += int(
            np.count_nonzero(checked_labels & ~checked_predictions)
        )

    def result(self, *, level: str = "pixel") -> MetricResult:
        """Return F1 under the frozen undefined-case policy."""

        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        if denominator == 0:
            return MetricResult(
                "f1",
                level,
                None,
                "undefined",
                "F1 is undefined when labels and predictions contain no positives",
            )
        return MetricResult(
            "f1",
            level,
            2 * self.true_positive / denominator,
            "defined",
        )


class Float16AuProAccumulator:
    """Accumulate exact tie groups for official float16 anomaly-map scores.

    MVTec's submission checker requires continuous anomaly TIFFs to be float16.
    Float16 has 65,536 bit patterns, so counts can be accumulated per exact score
    without retaining or sorting every original-resolution pixel.
    """

    def __init__(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - ML environment contract
            raise MetricInputError("Streaming AU-PRO requires NumPy") from exc
        self._np = np
        self._false_positive_changes = np.zeros(65536, dtype=np.uint64)
        self._pro_changes = np.zeros(65536, dtype=np.float64)
        self._normal_pixels = 0
        self._regions = 0
        self._images = 0

    def update(self, labels: Any, scores: Any) -> None:
        """Add one original-resolution binary mask and float16 anomaly map."""

        np = self._np
        try:
            from scipy.ndimage import label as connected_components
        except ImportError as exc:  # pragma: no cover - ML environment contract
            raise MetricInputError("Streaming AU-PRO requires SciPy") from exc
        label_array = np.asarray(labels)
        score_array = np.asarray(scores)
        if label_array.ndim != 2 or score_array.ndim != 2:
            raise MetricInputError("AU-PRO inputs must be 2-D images")
        if label_array.shape != score_array.shape or label_array.size == 0:
            raise MetricInputError("AU-PRO label and score shapes must match")
        if not np.all(np.isin(label_array, (0, 1, False, True))):
            raise MetricInputError("AU-PRO labels must be binary")
        if score_array.dtype != np.float16:
            raise MetricInputError("Official-format AU-PRO scores must be float16")
        if not bool(np.isfinite(score_array).all()):
            raise MetricInputError("AU-PRO scores must be finite")

        binary = label_array.astype(bool, copy=False)
        codes = score_array.view(np.uint16)
        normal_codes = codes[~binary]
        self._false_positive_changes += np.bincount(
            normal_codes, minlength=65536
        ).astype(np.uint64, copy=False)
        self._normal_pixels += int(normal_codes.size)

        labeled, count = connected_components(
            binary, structure=np.ones((3, 3), dtype=int)
        )
        for region_id in range(1, int(count) + 1):
            region_codes = codes[labeled == region_id]
            self._pro_changes += np.bincount(region_codes, minlength=65536) / float(
                region_codes.size
            )
        self._regions += int(count)
        self._images += 1

    def result(self, *, fpr_limit: float = 0.05) -> MetricResult:
        """Integrate the exact tie-grouped PRO curve through ``fpr_limit``."""

        if not 0.0 < fpr_limit <= 1.0:
            raise MetricInputError("fpr_limit must be in (0, 1]")
        if self._images == 0:
            raise MetricInputError("AU-PRO requires at least one image")
        if self._regions == 0 or self._normal_pixels == 0:
            return MetricResult(
                "au_pro_0.05",
                "pixel_region",
                None,
                "undefined",
                "AU-PRO requires anomalous regions and normal pixels",
            )

        np = self._np
        used = (self._false_positive_changes != 0) | (self._pro_changes != 0)
        codes = np.flatnonzero(used).astype(np.uint16)
        values = codes.view(np.float16).astype(np.float64)
        if not bool(np.isfinite(values).all()):
            raise MetricInputError("AU-PRO score bins must be finite")
        order = np.argsort(values)[::-1]
        ordered_codes = codes[order]
        false_positive_changes = self._false_positive_changes[ordered_codes]
        pro_changes = self._pro_changes[ordered_codes]
        fprs = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(false_positive_changes, dtype=np.uint64).astype(np.float64)
                / self._normal_pixels,
            )
        )
        pros = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(pro_changes, dtype=np.float64) / self._regions,
            )
        )
        np.clip(fprs, None, 1.0, out=fprs)
        np.clip(pros, None, 1.0, out=pros)

        above = np.flatnonzero(fprs >= fpr_limit)
        if above.size == 0:
            clipped_fprs = np.append(fprs, fpr_limit)
            clipped_pros = np.append(pros, pros[-1])
        else:
            right = int(above[0])
            if fprs[right] == fpr_limit:
                clipped_fprs = fprs[: right + 1]
                clipped_pros = pros[: right + 1]
            else:
                left = right - 1
                if math.isclose(float(fprs[right]), float(fprs[left])):
                    interpolated = max(float(pros[left]), float(pros[right]))
                else:
                    weight = (fpr_limit - float(fprs[left])) / (
                        float(fprs[right]) - float(fprs[left])
                    )
                    interpolated = float(pros[left]) + weight * (
                        float(pros[right]) - float(pros[left])
                    )
                clipped_fprs = np.append(fprs[:right], fpr_limit)
                clipped_pros = np.append(pros[:right], interpolated)
        area = float(np.trapezoid(clipped_pros, clipped_fprs))
        return MetricResult("au_pro_0.05", "pixel_region", area / fpr_limit, "defined")
