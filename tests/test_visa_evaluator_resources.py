"""Calculated scenarios, never test-image shape measurements."""

import pytest

from visionguard.visa import CATEGORIES
from visionguard.visa_evaluator import EvaluationError
from visionguard.visa_evaluator_resources import (
    RUN_BUDGET,
    SAFETY_FLOOR,
    check_capacity,
    estimate,
)


def test_independent_arithmetic_capacity_and_floor():
    counts = dict.fromkeys(CATEGORIES, 10)
    result = estimate(counts, 2, 3)
    assert result["model_image_calls"] == 720
    assert result["map_payload_bytes"] == 720 * 2 * 3 * 3
    assert result["required_free_bytes"] == RUN_BUDGET + SAFETY_FLOOR
    check_capacity(result["required_free_bytes"], result)
    with pytest.raises(EvaluationError, match="Insufficient"):
        check_capacity(result["required_free_bytes"] - 1, result)
    for invalid in (0, -1, True):
        with pytest.raises(EvaluationError):
            estimate(counts, invalid, 3)
    with pytest.raises(EvaluationError, match="budget"):
        estimate(counts, 100000, 100000)
    with pytest.raises(EvaluationError):
        estimate({"candle": 1})
