from __future__ import annotations

import pytest

from visionguard.calibration import (
    CalibrationError,
    empirical_quantile,
    highest_order_statistic,
    per_image_pixel_maxima,
)
from visionguard.experiment import ThresholdConfig
from visionguard.metrics import (
    MetricInputError,
    au_pro,
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


def test_highest_order_statistic_records_finite_sample_coverage() -> None:
    result = highest_order_statistic([0.3, 0.1, 0.2], minimum_samples=3)

    assert result.threshold == 0.3
    assert result.order_rank == 3
    assert result.marginal_coverage == 0.75
    assert "exchangeability" in result.guarantee


def test_pixel_calibration_uses_one_maximum_per_image() -> None:
    maxima = per_image_pixel_maxima([[0.1, 0.7, 0.2], [0.3, 0.4]])

    assert maxima == [0.7, 0.4]
    assert highest_order_statistic(maxima, minimum_samples=2).threshold == 0.7


def test_order_statistic_rejects_invalid_inputs() -> None:
    with pytest.raises(CalibrationError, match="at least"):
        highest_order_statistic([0.1], minimum_samples=2)
    with pytest.raises(CalibrationError, match="finite"):
        per_image_pixel_maxima([[float("inf")]])


def test_au_pro_perfect_and_reversed_predictions() -> None:
    labels = [[[1, 0], [0, 0]]]

    assert au_pro(labels, [[[0.9, 0.3], [0.2, 0.1]]]).value == 1.0
    assert au_pro(labels, [[[0.1, 0.9], [0.8, 0.7]]]).value == 0.0


def test_au_pro_ties_are_processed_as_one_threshold_group() -> None:
    result = au_pro([[[1, 0], [0, 0]]], [[[0.5, 0.5], [0.5, 0.5]]])

    assert result.value == pytest.approx(0.025)


def test_au_pro_handles_single_pixel_regions_equally() -> None:
    labels = [[[1, 0, 1], [0, 0, 0]]]
    scores = [[[0.9, 0.1, 0.8], [0.1, 0.1, 0.1]]]

    assert au_pro(labels, scores).value == 1.0


def test_au_pro_undefined_and_invalid_cases_are_explicit() -> None:
    assert au_pro([[[0, 0]]], [[[0.1, 0.2]]]).status == "undefined"
    assert au_pro([[[1, 1]]], [[[0.1, 0.2]]]).status == "undefined"
    with pytest.raises(MetricInputError, match="shapes"):
        au_pro([[[0, 1]]], [[[0.1]]])
    with pytest.raises(MetricInputError, match="finite"):
        au_pro([[[0, 1]]], [[[0.1, float("nan")]]])
