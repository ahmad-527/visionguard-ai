"""Entirely artificial arrays/scores: never benchmark results or dataset assets."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from visionguard.metrics import au_pro, binary_auroc
from visionguard.triage import Decision, triage_same_seed
from visionguard.visa_evaluator import (
    EvaluationError,
    ModelSpec,
    aggregate_matrix,
    evaluate_synthetic_cell,
    finite_scalar,
    model_specs,
    predict_synthetic,
    summarize,
)
from visionguard.visa_evaluator_synthetic import fixture, synthetic_report


def test_hand_calculated_answers_and_reference_pro():
    samples, pc, ea, specs = fixture()
    result = evaluate_synthetic_cell(samples, pc, ea, specs)
    assert [r["triage"] for r in result["images"]] == [
        "PASS",
        "REVIEW",
        "REVIEW",
        "REJECT",
    ]
    assert result["models"]["patchcore"]["image"]["confusion"] == {
        "true_positive": 2,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 0,
    }
    assert result["models"]["efficientad"]["image"]["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    for model, image_f1, image_auc, pixel_f1, pro in (
        ("patchcore", 1.0, 1.0, 1.0, 1.0),
        ("efficientad", 0.5, 0.75, 2 / 3, 0.5125),
    ):
        metrics = result["models"][model]
        assert metrics["image"]["image_f1"]["value"] == image_f1
        assert metrics["image"]["image_auroc"]["value"] == image_auc
        assert metrics["pixel"]["pixel_f1"]["value"] == pixel_f1
        # Two equally weighted singleton regions. EA PRO=.5+.5*FPR after
        # the first correct peak, integral / .05 = .5 + .25*.05 = .5125.
        assert metrics["pixel"]["au_pro_0.05"]["value"] == pytest.approx(pro)
        predictions = pc if model == "patchcore" else ea
        independent = au_pro(
            [s.mask.tolist() for s in samples],
            [p.restored_map.tolist() for p in predictions],
            fpr_limit=0.05,
        )
        assert independent.value == pytest.approx(pro)
    rates = result["triage"]["rates"]
    expected = {
        "review_rate": 0.5,
        "automatic_decision_coverage": 0.5,
        "selective_accuracy": 1.0,
        "anomaly_pass_through_rate": 0.0,
        "normal_reject_rate": 0.0,
        "anomaly_review_capture": 0.5,
        "normal_review_rate": 0.5,
        "reject_precision": 1.0,
        "pass_negative_predictive_value": 1.0,
    }
    assert {k: v["value"] for k, v in rates.items()} == expected
    assert result["triage"]["disagreement_counts"] == {
        "patchcore_anomalous_efficientad_normal": 1,
        "patchcore_normal_efficientad_anomalous": 1,
    }


@pytest.mark.parametrize(
    "p,e,outcome",
    [
        (2.0, 2.0, "PASS"),
        (3.0, 2.0, "REVIEW"),
        (2.0, 3.0, "REVIEW"),
        (3.0, 3.0, "REJECT"),
    ],
)
def test_truth_table_equality_and_review_not_binary(p, e, outcome):
    result = triage_same_seed(p, 2.0, e, 2.0, patchcore_seed=42, efficientad_seed=42)
    assert result.triage_decision.value == outcome
    with pytest.raises(TypeError):
        bool(result)
    with pytest.raises(TypeError):
        bool(Decision.REVIEW)


def test_float32_threshold_before_float16_rounding():
    samples, pc, ea, specs = fixture()
    values = np.full((2, 2), np.float32(2.0001))
    assert values.astype(np.float16)[0, 0] == 2.0
    pc[2] = replace(pc[2], restored_map=values)
    result = evaluate_synthetic_cell(samples, pc, ea, specs)
    assert result["models"]["patchcore"]["pixel"]["confusion"]["false_positive"] == 3


def test_ties_pairwise_expected_half_credit():
    labels, scores = [0, 0, 1, 1], [1.0, 2.0, 2.0, 2.0]
    # All four positive/negative pairs: two wins and two ties => 3/4.
    expected = sum((a > n) + 0.5 * (a == n) for a in scores[2:] for n in scores[:2]) / 4
    assert expected == 0.75 == binary_auroc(labels, scores).value


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, None, "2", 10**400]
)
def test_bad_thresholds_and_scores(value):
    with pytest.raises(EvaluationError):
        finite_scalar(value, "test")
    with pytest.raises(EvaluationError):
        ModelSpec("patchcore", "candle", 42, "a" * 64, value, 2.0)
    samples, pc, ea, specs = fixture()
    pc[0] = replace(pc[0], score=value)
    with pytest.raises(EvaluationError):
        evaluate_synthetic_cell(samples, pc, ea, specs)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "reorder",
        "duplicate",
        "wrong_seed",
        "wrong_category",
        "wrong_hash",
        "wrong_threshold",
        "bad_shape",
        "nan_map",
        "inf_map",
        "overflow",
        "float64",
        "bad_mask",
        "empty_mask",
        "float_mask",
        "label_conflict",
        "real_id",
        "override_binary",
        "quantized_without_binary",
    ],
)
def test_adversarial_cells_fail_closed(change):
    samples, pc, ea, specs = fixture()
    if change == "missing":
        ea.pop()
    elif change == "reorder":
        ea.reverse()
    elif change == "duplicate":
        samples[1], pc[1], ea[1] = samples[0], pc[0], ea[0]
    elif change.startswith("wrong_"):
        fields = {
            "wrong_seed": {"seed": 123},
            "wrong_category": {"category": "cashew"},
            "wrong_hash": {"model_sha256": "c" * 64},
            "wrong_threshold": {"image_threshold": 2.1},
        }
        pc[0] = replace(pc[0], spec=replace(specs[0], **fields[change]))
    elif change in (
        "bad_shape",
        "nan_map",
        "inf_map",
        "overflow",
        "float64",
        "quantized_without_binary",
    ):
        arrays = {
            "bad_shape": np.zeros((3, 3), dtype=np.float32),
            "nan_map": np.full((2, 2), np.nan, dtype=np.float32),
            "inf_map": np.full((2, 2), np.inf, dtype=np.float32),
            "overflow": np.full((2, 2), 65536, dtype=np.float32),
            "float64": np.zeros((2, 2)),
            "quantized_without_binary": np.zeros((2, 2), dtype=np.float16),
        }
        pc[0] = replace(pc[0], restored_map=arrays[change])
    elif change in ("bad_mask", "empty_mask", "float_mask"):
        masks = {
            "bad_mask": np.full((2, 2), 255, dtype=np.uint8),
            "empty_mask": np.zeros((0, 2), dtype=np.uint8),
            "float_mask": np.zeros((2, 2)),
        }
        samples[0] = replace(samples[0], mask=masks[change])
    elif change == "label_conflict":
        samples[0] = replace(samples[0], label=1)
    elif change == "real_id":
        samples[0] = replace(samples[0], sample_id="visa/candle/test/0")
    else:
        pc[0] = replace(pc[0], thresholded_map=np.ones((2, 2), dtype=bool))
    with pytest.raises(EvaluationError):
        evaluate_synthetic_cell(samples, pc, ea, specs)


def test_pair_seed_and_empty_cell_rejected():
    s, p, e, specs = fixture()
    with pytest.raises(EvaluationError, match="seed"):
        evaluate_synthetic_cell(s, p, e, (specs[0], replace(specs[1], seed=123)))
    with pytest.raises(EvaluationError, match="Empty"):
        evaluate_synthetic_cell([], [], [], specs)


def test_all_review_and_absent_classes_undefined_not_zero():
    s, p, e, specs = fixture()
    p = [replace(x, score=3.0) for x in p]
    e = [replace(x, score=2.0) for x in e]
    result = evaluate_synthetic_cell(s[:2], p[:2], e[:2], specs)
    assert result["triage"]["rates"]["selective_accuracy"]["value"] is None
    assert (
        result["triage"]["rates"]["anomaly_pass_through_rate"]["undefined_reason"]
        == "zero_denominator"
    )
    assert result["models"]["patchcore"]["image"]["image_auroc"]["value"] is None
    assert result["models"]["patchcore"]["pixel"]["au_pro_0.05"]["value"] is None
    assert result["triage"]["decision_counts"]["REVIEW"] == 2


def test_synthetic_backend_has_no_truth_in_decision():
    s, p, _, specs = fixture()
    assert predict_synthetic(s[0], specs[0], lambda _: p[0]) == p[0]
    with pytest.raises(EvaluationError):
        predict_synthetic(
            s[0], specs[0], lambda _: replace(p[0], spec=replace(specs[0], seed=123))
        )


def test_matrix_determinism_macro_micro_pairing():
    report = synthetic_report()
    assert report == synthetic_report()
    aggregate = report["aggregate"]
    seed = aggregate["seeds"]["42"]
    assert seed["category_macro"]["efficientad"]["image_auroc"]["mean"] == 0.75
    assert seed["paired_patchcore_minus_efficientad"]["image_auroc"]["mean"] == 0.25
    assert (
        seed["model_pooled_confusion"]["efficientad"]["image"]["confusion"][
            "false_positive"
        ]
        == 12
    )
    assert seed["triage_pooled_counts"]["decision_counts"]["REVIEW"] == 24
    assert (
        aggregate["descriptive_across_three_seeds"]["patchcore"]["image_f1"][
            "sample_std"
        ]
        == 0.0
    )
    with pytest.raises(EvaluationError):
        aggregate_matrix(report["cells"][:-1])
    with pytest.raises(EvaluationError):
        aggregate_matrix(report["cells"] + report["cells"][:1])
    cells = deepcopy(report["cells"])
    cells[0]["models"]["patchcore"]["image"]["image_auroc"]["value"] = None
    changed = aggregate_matrix(cells)["seeds"]["42"]
    assert changed["category_macro"]["patchcore"]["image_auroc"]["mean"] is None
    assert (
        changed["category_macro"]["patchcore"]["image_auroc"]["defined_group_count"]
        == 11
    )
    assert changed["paired_patchcore_minus_efficientad"]["image_auroc"]["mean"] is None
    assert summarize([1.0, 2.0, 3.0])["sample_std"] == 1.0


def test_inventory_all_72_required():
    from visionguard.visa import CATEGORIES

    cells = {
        f"{m}:{c}:{s}": {
            "final_model_sha256": "a" * 64,
            "image_threshold": {"threshold": 2.0},
            "pixel_threshold": {"threshold": 2.0},
        }
        for m in ("patchcore", "efficientad")
        for c in CATEGORIES
        for s in (42, 123, 2026)
    }
    assert len(model_specs({"cells": cells})) == 72
    cells.pop("patchcore:candle:42")
    with pytest.raises(EvaluationError):
        model_specs({"cells": cells})


def test_cross_seed_membership_must_be_identical():
    report = synthetic_report()
    report["cells"][0]["membership_sha256"] = "0" * 64
    with pytest.raises(EvaluationError, match="membership"):
        aggregate_matrix(report["cells"])
