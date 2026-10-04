"""Artificial genuine-shaped IDs only; independent B1 equivalence oracle."""

import ast
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from visionguard import heldout_metrics as real
from visionguard import visa_evaluator as b1
from visionguard.visa_evaluator_synthetic import fixture


def converted(category="candle", seed=42):
    samples, pc, ea, specs = fixture(category, seed)
    ids = [
        f"{category}/Data/Images/{'Anomaly' if s.label else 'Normal'}/"
        f"artificial-{i}.png"
        for i, s in enumerate(samples)
    ]
    return (
        [
            real.Sample(i, s.category, s.label, s.mask)
            for i, s in zip(ids, samples, strict=True)
        ],
        *[
            [
                real.Prediction(i, p.spec, p.score, p.restored_map, p.thresholded_map)
                for i, p in zip(ids, pp, strict=True)
            ]
            for pp in (pc, ea)
        ],
        specs,
    )


def numbers(value):
    if isinstance(value, dict):
        return {
            k: numbers(v)
            for k, v in value.items()
            if k not in ("sample_id", "membership_sha256", "evidence_class")
        }
    if isinstance(value, list):
        return [numbers(v) for v in value]
    return value


def test_exact_numeric_equivalence_all_pairs():
    from visionguard.visa import CATEGORIES

    actual = []
    expected = []
    for c in CATEGORIES:
        for s in (42, 123, 2026):
            a = real.evaluate_cell(*converted(c, s))
            b = b1.evaluate_synthetic_cell(*fixture(c, s))
            assert numbers(a) == numbers(b)
            actual.append(a)
            expected.append(b)
    assert numbers(real.aggregate_matrix(actual)) == numbers(
        b1.aggregate_matrix(expected)
    )


def test_independently_counted_known_answers_and_undefined():
    result = real.evaluate_cell(*converted())
    assert result["triage"]["decision_counts"] == {"PASS": 1, "REVIEW": 2, "REJECT": 1}
    assert result["triage"]["rates"]["selective_accuracy"]["value"] == 1
    assert result["triage"]["rates"]["automatic_decision_coverage"]["value"] == 0.5
    assert result["triage"]["rates"]["anomaly_pass_through_rate"]["value"] == 0
    pc, ea = (result["models"][m] for m in ("patchcore", "efficientad"))
    assert pc["image"]["confusion"] == {
        "true_positive": 2,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 0,
    }
    assert ea["image"]["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    # Anomalous scores2,4 beat normal scores1,3 in3 of4 ordered pairs.
    assert ea["image"]["image_auroc"]["value"] == 3 / 4
    assert ea["pixel"]["pixel_f1"]["value"] == 2 / 3
    assert pc["pixel"]["au_pro_0.05"]["value"] == 1
    samples, pp, ee, specs = converted()
    normal = real.evaluate_cell(samples[:1], pp[:1], ee[:1], specs)
    assert normal["models"]["patchcore"]["image"]["image_auroc"]["value"] is None
    assert normal["triage"]["rates"]["reject_precision"]["value"] is None
    assert normal["triage"]["rates"]["anomaly_pass_through_rate"]["value"] is None


def test_numerical_ast_equivalence():
    def functions(module):
        tree = ast.parse(Path(module.__file__).read_text())
        return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    old = functions(b1)
    new = functions(real)

    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = node.name.replace("evaluate_synthetic_cell", "evaluate_cell")
            node.body = [
                n
                for n in node.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)
                )
            ]
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == "SyntheticSample":
                node.id = "Sample"
            return node

        def visit_Constant(self, node):
            if node.value == "synthetic_engineering_only":
                node.value = "real_id_interface"
            if isinstance(node.value, str) and "synthetic" in node.value.lower():
                node.value = "interface message"
            if node.value == "Controlled real-ID evidence interface required":
                node.value = "interface message"
            return node

    for name in (
        "validate_prediction",
        "evaluate_synthetic_cell",
        "summarize",
        "aggregate_matrix",
        "sum_nested_counts",
        "pooled_confusion",
    ):
        target = name.replace("evaluate_synthetic_cell", "evaluate_cell")
        assert ast.dump(Normalize().visit(copy.deepcopy(old[name]))) == ast.dump(
            Normalize().visit(copy.deepcopy(new[target]))
        )


def test_b1_still_rejects_genuine_id():
    s = converted()[0][0]
    with pytest.raises(ValueError):
        b1.validate_sample(b1.SyntheticSample(s.sample_id, s.category, s.label, s.mask))


@pytest.mark.parametrize(
    "identifier",
    [
        "synthetic:candle:0",
        "../Data/Images/Normal/a.png",
        "candle/Data/Images/Normal/../a.png",
        "candle\\Data\\Images\\Normal\\a.png",
        "capsules/Data/Images/Normal/a.png",
    ],
)
def test_bad_ids(identifier):
    with pytest.raises(ValueError):
        real.validate_sample(replace(converted()[0][0], sample_id=identifier))


@pytest.mark.parametrize(
    "change",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"sample_id": "candle/Data/Images/Normal/other.png"},
        {"restored_map": np.zeros((3, 3), np.float32)},
        {"spec": replace(fixture()[3][0], seed=123)},
    ],
)
def test_adversarial_predictions(change):
    ss, pc, _, specs = converted()
    with pytest.raises(ValueError):
        real.validate_prediction(ss[0], replace(pc[0], **change), specs[0])


def test_native_operations_ast_equivalent():
    from visionguard import heldout_backend, visa_evaluator_backend

    def method(module, name):
        tree = ast.parse(Path(module.__file__).read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        node.name = "predict"
        node.args.args[1].annotation.id = "FrameInterface"
        return ast.dump(node)

    assert method(heldout_backend, "predict") == method(
        visa_evaluator_backend, "predict_synthetic"
    )
