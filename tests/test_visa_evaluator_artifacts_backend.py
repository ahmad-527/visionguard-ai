"""Published metadata and synthetic artifact faults; never dataset images."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from visionguard.visa_acquire import VisaIntegrityError
from visionguard.visa_evaluator import EvaluationError, model_specs
from visionguard.visa_evaluator_artifacts import (
    DEVELOPMENT_FREEZE,
    verify_local,
    verify_published,
)
from visionguard.visa_evaluator_backend import SyntheticFrame
from visionguard.visa_evaluator_synthetic import fixture

ROOT = Path(__file__).resolve().parents[1]


def test_fake_native_synthetic_inference_no_truth_or_training(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import visionguard.preprocessing as pre
    from visionguard.visa_evaluator_backend import FrozenBackend

    _, _, _, specs = fixture()
    calls = []

    class FakeTensor:
        def unsqueeze(self, _):
            return self

        def to(self, _):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((2, 3), dtype=np.float32)

    class FakeNative:
        def eval(self):
            calls.append("eval")

        def requires_grad_(self, value):
            assert value is False
            calls.append("grad_disabled")

        def __call__(self, _):
            calls.append("predict_synthetic")
            return SimpleNamespace(pred_score=[2.0], anomaly_map=np.zeros((1, 1, 2, 3)))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(pre, "restore_anomaly_map", lambda _, shape: FakeTensor())
    backend = FrozenBackend(specs[0], FakeNative(), lambda _: FakeTensor(), "cpu")
    frame = SyntheticFrame(
        "synthetic:candle:0", "candle", np.zeros((2, 3, 3), dtype=np.uint8)
    )
    prediction = backend.predict_synthetic(frame)
    assert prediction.score == 2.0 and prediction.restored_map.shape == (2, 3)
    assert calls == ["eval", "grad_disabled", "predict_synthetic"]
    with pytest.raises(EvaluationError):
        backend.predict_synthetic(
            SyntheticFrame("forbidden_real_id", "candle", frame.rgb)
        )
    assert len(calls) == 3


def test_published_inventory_and_no_model_fallback():
    frozen = verify_published(ROOT)
    assert len(model_specs(frozen)) == 72
    assert frozen["validated_cells"] == 72
    assert frozen["final_test_lock"] == "closed"
    assert len(frozen["cells"]["efficientad:cashew:42"]["attempt_history"]) == 3
    assert (
        frozen["cells"]["efficientad:pcb1:42"]["attempt_history"][0]["status"]
        == "interrupted"
    )


def test_corrupt_published_freeze_rejected_before_read(monkeypatch):
    import visionguard.visa_evaluator_artifacts as artifacts

    monkeypatch.setattr(artifacts, "sha256_file", lambda _: "0" * 64)
    with pytest.raises(VisaIntegrityError, match="Published freeze differs"):
        verify_published(ROOT)


@pytest.mark.parametrize(
    "fault",
    [
        "matrix",
        "local_freeze",
        "membership",
        "source",
        "origin",
        "final_model",
        "calibration",
        "normalization",
        "fit_membership",
    ],
)
def test_local_provenance_faults_fail_closed_before_any_inference(
    tmp_path, monkeypatch, fault
):
    import visionguard.visa_evaluator_artifacts as a

    freeze = deepcopy(verify_published(ROOT))
    identity = {
        "implementation_commit": a.IMPLEMENTATION,
        "membership_sha256": a.MEMBERSHIP,
        "protocol_fingerprint": freeze["execution_contract"]["protocols"]["patchcore"],
    }
    matrix = {"identities": {"patchcore": identity}}
    if fault == "origin":
        identity["implementation_commit"] = "0" * 40

    def digest(path):
        name = path.name
        if name == "matrix-manifest.json":
            return "0" * 64 if fault == "matrix" else freeze["matrix_manifest_sha256"]
        if name == "development-freeze.json":
            return (
                "0" * 64
                if fault == "local_freeze"
                else freeze["full_local_freeze_sha256"]
            )
        if name == "development-membership.json":
            return "0" * 64 if fault == "membership" else a.MEMBERSHIP
        return (
            "0" * 64
            if fault == "source"
            else next(
                v
                for k, v in freeze["execution_contract"]["source_hashes"].items()
                if path.as_posix().endswith(k)
            )
        )

    def validator(_root, model, category, seed, _identity, _membership):
        cell = freeze["cells"][f"{model}:{category}:{seed}"]
        result = {
            "canonical_model_sha256": cell["canonical_model_sha256"],
            "files": {
                "final_model": {"sha256": cell["final_model_sha256"]},
                "calibration": {"sha256": cell["calibration_sha256"]},
            },
            "calibration": {
                "image": cell["image_threshold"],
                "pixel": cell["pixel_threshold"],
            },
            "normalization": cell["normalization"],
            "fit_inventory_sha256": cell["fit_inventory_sha256"],
            "calibration_inventory_sha256": cell["calibration_inventory_sha256"],
        }
        if fault == "final_model":
            result["files"]["final_model"]["sha256"] = "0" * 64
        if fault == "calibration":
            result["calibration"]["image"] = {"threshold": -1}
        if fault == "normalization":
            result["normalization"] = {"synthetic": "wrong"}
        if fault == "fit_membership":
            result["fit_inventory_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(a, "verify_published", lambda _: freeze)
    monkeypatch.setattr(a, "sha256_file", digest)
    monkeypatch.setattr(
        a, "load", lambda p: matrix if p.name == "matrix-manifest.json" else {}
    )
    monkeypatch.setattr(a, "validate_cell", validator)
    with pytest.raises(VisaIntegrityError):
        verify_local(tmp_path, tmp_path / "development")


def test_frame_contract_label_free_and_no_path():
    _, _, _, specs = fixture()
    frame = SyntheticFrame(
        "synthetic:candle:0", "candle", np.zeros((2, 3, 3), dtype=np.uint8)
    )
    frame.validate(specs[0])
    for bad in (
        SyntheticFrame("real_test_image", "candle", frame.rgb),
        SyntheticFrame(frame.sample_id, "cashew", frame.rgb),
        SyntheticFrame(frame.sample_id, frame.category, np.zeros((2, 2))),
        SyntheticFrame("synthetic:candle:/path", frame.category, frame.rgb),
    ):
        with pytest.raises(EvaluationError):
            bad.validate(specs[0])
    assert not hasattr(frame, "label") and not hasattr(frame, "mask")
    assert (
        DEVELOPMENT_FREEZE
        == "d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061"
    )
