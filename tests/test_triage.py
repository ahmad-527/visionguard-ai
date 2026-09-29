"""Entirely synthetic policy and metric tests; no benchmark outcomes are used."""

import inspect
import itertools
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from visionguard.triage import (
    ALLOWED_SEEDS,
    Decision,
    TriageError,
    TriageResult,
    triage,
    triage_same_seed,
    validate_seed_pair,
)
from visionguard.triage_metrics import calculate_triage_metrics


@pytest.mark.parametrize(
    ("pc", "ead", "outcome"),
    [
        (False, False, Decision.PASS),
        (True, True, Decision.REJECT),
        (False, True, Decision.REVIEW),
        (True, False, Decision.REVIEW),
    ],
)
def test_truth_table(pc, ead, outcome):
    result = triage(2 if pc else 0, 1, 2000 if ead else 0, 1000)
    assert result.patchcore_decision is pc
    assert result.efficientad_decision is ead
    assert result.triage_decision is outcome


def test_strict_threshold_boundaries_and_independent_scales():
    assert triage(1, 1, 1000, 1000).triage_decision is Decision.PASS
    assert (
        triage(math.nextafter(1, math.inf), 1, 1000, 1000).triage_decision
        is Decision.REVIEW
    )
    assert (
        triage(1, 1, math.nextafter(1000, math.inf), 1000).triage_decision
        is Decision.REVIEW
    )
    assert triage(-2, -1, -2000, -1000).triage_decision is Decision.PASS
    assert triage(-0.0, 0.0, 0.0, -0.0).triage_decision is Decision.PASS
    # Coercing these integers to float would erase the strict inequality.
    assert triage(2**60 + 1, 2**60, 0, 0).triage_decision is Decision.REVIEW


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize(
    "invalid",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        None,
        True,
        False,
        "1.0",
        [],
        {},
        complex(1),
    ],
)
def test_invalid_score_or_threshold_fails_closed(position, invalid):
    values = [0.0, 1.0, 2.0, 3.0]
    values[position] = invalid
    with pytest.raises(TriageError):
        triage(*values)


def test_missing_evidence_no_defaults():
    with pytest.raises(TypeError):
        triage(1, 2, 3)
    with pytest.raises(TypeError):
        triage_same_seed(1, 2, 3, 4)


def test_deterministic_immutable_output():
    result = triage(1, 0, 0, 1)
    assert all(triage(1, 0, 0, 1) == result for _ in range(100))
    with pytest.raises(FrozenInstanceError):
        result.triage_decision = Decision.PASS


@pytest.mark.parametrize("decision", list(Decision))
def test_no_implicit_binary_conversion(decision):
    with pytest.raises(TypeError, match="not binary"):
        bool(decision)
    with pytest.raises(TypeError):
        int(decision)
    with pytest.raises(TypeError):
        bool(triage(1, 0, 0, 1))


@pytest.mark.parametrize("pc,ead", list(itertools.product(ALLOWED_SEEDS, repeat=2)))
def test_seed_pairing(pc, ead):
    if pc == ead:
        assert validate_seed_pair(pc, ead) == pc
        assert triage_same_seed(
            0, 1, 2, 1, patchcore_seed=pc, efficientad_seed=ead
        ) == triage(0, 1, 2, 1)
    else:
        with pytest.raises(TriageError, match="same seed"):
            triage_same_seed(0, 1, 2, 1, patchcore_seed=pc, efficientad_seed=ead)


@pytest.mark.parametrize("invalid", [None, True, "42", 42.0, 0, -1, 2027])
def test_invalid_seeds(invalid):
    for seeds in [(invalid, 42), (42, invalid)]:
        with pytest.raises(TriageError):
            validate_seed_pair(*seeds)


def test_no_category_or_label_input_path():
    assert list(inspect.signature(triage).parameters) == [
        "patchcore_score",
        "patchcore_threshold",
        "efficientad_score",
        "efficientad_threshold",
    ]
    for keyword in ["category", "label", "override", "weights"]:
        with pytest.raises(TypeError):
            triage(0, 1, 2, 1, **{keyword: "synthetic"})


@pytest.mark.parametrize(
    "pc,ead,decision",
    [
        (False, True, Decision.PASS),
        (True, True, Decision.REVIEW),
        (0, 1, Decision.REVIEW),
        (False, True, "REVIEW"),
    ],
)
def test_forged_results_rejected(pc, ead, decision):
    with pytest.raises(TriageError):
        TriageResult(pc, ead, decision)


def test_module_imports_no_models_and_performs_no_file_access():
    code = """
import sys
from visionguard.triage import triage, Decision
assert not any(name in sys.modules for name in ("torch", "anomalib", "numpy"))
def audit(event, args):
    if event in ("open", "socket.connect", "subprocess.Popen"):
        raise AssertionError(event)
sys.addaudithook(audit)
assert triage(0, 1, 2, 1).triage_decision is Decision.REVIEW
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_hand_calculated_three_way_metrics():
    # Synthetic counts: normals P=2,V=1,R=1; anomalies P=1,V=2,R=3.
    p = triage(0, 1, 0, 1)
    r = triage(2, 1, 2, 1)
    pc_review = triage(2, 1, 0, 1)
    ead_review = triage(0, 1, 2, 1)
    results = [p, p, pc_review, r, p, pc_review, ead_review, r, r, r]
    metrics = calculate_triage_metrics(results, [False] * 4 + [True] * 6)
    assert metrics.sample_count == 10
    assert metrics.decision_counts == {"PASS": 3, "REVIEW": 3, "REJECT": 4}
    expected = {
        "review_rate": (3, 10),
        "automatic_decision_coverage": (7, 10),
        "selective_accuracy": (5, 7),
        "anomaly_pass_through_rate": (1, 6),
        "normal_reject_rate": (1, 4),
        "anomaly_review_capture": (2, 6),
        "normal_review_rate": (1, 4),
        "reject_precision": (3, 4),
        "pass_negative_predictive_value": (2, 3),
    }
    assert set(metrics.rates) == set(expected)
    for name, (numerator, denominator) in expected.items():
        rate = metrics.rates[name]
        assert (rate.numerator, rate.denominator) == (numerator, denominator)
        assert rate.value == pytest.approx(numerator / denominator)
        assert rate.undefined_reason is None
    assert metrics.class_decision_counts["normal"] == {
        "PASS": 2,
        "REVIEW": 1,
        "REJECT": 1,
    }
    assert metrics.class_decision_counts["anomalous"] == {
        "PASS": 1,
        "REVIEW": 2,
        "REJECT": 3,
    }
    assert metrics.disagreement_counts == {
        "patchcore_anomalous_efficientad_normal": 2,
        "patchcore_normal_efficientad_anomalous": 1,
    }
    assert (
        metrics.class_disagreement_counts["normal"][
            "patchcore_normal_efficientad_anomalous"
        ]
        == 0
    )
    assert metrics == calculate_triage_metrics(results, [False] * 4 + [True] * 6)


@pytest.mark.parametrize(
    "results,labels,undefined",
    [
        (
            [],
            [],
            {
                "review_rate",
                "automatic_decision_coverage",
                "selective_accuracy",
                "anomaly_pass_through_rate",
                "normal_reject_rate",
                "anomaly_review_capture",
                "normal_review_rate",
                "reject_precision",
                "pass_negative_predictive_value",
            },
        ),
        (
            [triage(1, 0, 0, 1)],
            [True],
            {
                "selective_accuracy",
                "normal_reject_rate",
                "normal_review_rate",
                "reject_precision",
                "pass_negative_predictive_value",
            },
        ),
        (
            [triage(0, 1, 0, 1)],
            [False],
            {"anomaly_pass_through_rate", "anomaly_review_capture", "reject_precision"},
        ),
        (
            [triage(1, 0, 1, 0)],
            [True],
            {
                "normal_reject_rate",
                "normal_review_rate",
                "pass_negative_predictive_value",
            },
        ),
    ],
)
def test_undefined_rates(results, labels, undefined):
    metrics = calculate_triage_metrics(results, labels)
    assert {
        name for name, rate in metrics.rates.items() if rate.value is None
    } == undefined
    for name in undefined:
        assert metrics.rates[name].denominator == 0
        assert metrics.rates[name].undefined_reason == "zero_denominator"
    if results and all(r.triage_decision is Decision.REVIEW for r in results):
        assert metrics.rates["automatic_decision_coverage"].value == 0
        assert metrics.rates["anomaly_review_capture"].value == 1


@pytest.mark.parametrize(
    "results,labels",
    [
        ([triage(0, 1, 0, 1)], []),
        ([None], [True]),
        ([triage(0, 1, 0, 1)], [1]),
        ([triage(0, 1, 0, 1)], ["anomalous"]),
    ],
)
def test_malformed_metric_evidence_rejected(results, labels):
    with pytest.raises(TriageError):
        calculate_triage_metrics(results, labels)


def test_structural_subset_properties_not_empirical_performance():
    for pc, ead in itertools.product([False, True], repeat=2):
        result = triage(int(pc), 0, int(ead), 0)
        if result.triage_decision is Decision.PASS:
            assert not pc and not ead
        if result.triage_decision is Decision.REJECT:
            assert pc and ead
