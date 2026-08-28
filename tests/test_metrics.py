from __future__ import annotations

import pytest

from visionguard.calibration import CalibrationError, empirical_quantile
from visionguard.experiment import ThresholdConfig
from visionguard.metrics import (
    MetricInputError,
    binary_auroc,
    binary_f1,
    flatten_pixel_data,
)


def test_auroc_perfect_predictions() -> None:
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).value == 1.0


def test_auroc_inverted_predictions() -> None:
    assert binary_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]).value == 0.0


def test_auroc_constant_predictions_are_tie_aware() -> None:
    assert binary_auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]).value == 0.5


def test_auroc_without_positive_is_explicitly_undefined() -> None:
    result = binary_auroc([0, 0], [0.1, 0.2])

    assert result.value is None
    assert result.status == "undefined"
    assert "positive" in str(result.reason)


def test_f1_perfect_and_inverted_predictions() -> None:
    assert binary_f1([0, 1, 1], [0, 1, 1]).value == 1.0
    assert binary_f1([0, 1], [1, 0]).value == 0.0


def test_f1_empty_positive_case_is_undefined() -> None:
    assert binary_f1([0, 0], [0, 0]).status == "undefined"


def test_metric_shape_and_invalid_input_checks() -> None:
    with pytest.raises(MetricInputError, match="identical lengths"):
        binary_auroc([0, 1], [0.2])
    with pytest.raises(MetricInputError, match="0/1"):
        binary_auroc([0, 2], [0.2, 0.8])
    with pytest.raises(MetricInputError, match="finite"):
        binary_auroc([0, 1], [0.2, float("nan")])


def test_pixel_shape_mismatch_is_rejected() -> None:
    with pytest.raises(MetricInputError, match="shapes"):
        flatten_pixel_data([[0, 1]], [[0.1]])


def test_pixel_data_flattens_after_validation() -> None:
    labels, scores = flatten_pixel_data([[0, 1], [1, 0]], [[0.1, 0.9], [0.8, 0.2]])

    assert binary_auroc(labels, scores, level="pixel").value == 1.0


def test_empirical_quantile_is_deterministic() -> None:
    config = ThresholdConfig("empirical_quantile", 0.5, 2)

    assert empirical_quantile([3.0, 1.0, 2.0, 4.0], config) == 2.5


def test_calibration_rejects_too_few_or_nonfinite_scores() -> None:
    config = ThresholdConfig("empirical_quantile", 0.9, 2)
    with pytest.raises(CalibrationError, match="at least"):
        empirical_quantile([1.0], config)
    with pytest.raises(CalibrationError, match="finite"):
        empirical_quantile([1.0, float("inf")], config)
