"""Manufactured CPU tensor adapter test, NOT trained-artifact/native ML acceptance."""

import hashlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from visionguard_inspection.native import NativeInspectionBackend

from test_inspection_registry import fixture_entry
from visionguard.inspection_contract import (
    ModelManifest,
    prepare_image,
    validate_output,
)

torch = pytest.importorskip("torch")


def test_cpu_adapter_inference_mode_and_original_coordinate_restoration():
    class FixtureModel:
        def __call__(self, tensor):
            assert torch.is_inference_mode_enabled()
            assert tensor.device.type == "cpu" and tensor.shape == (1, 3, 2, 2)
            return SimpleNamespace(
                pred_score=torch.tensor([0.75]),
                anomaly_map=torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]),
            )

    native = SimpleNamespace(
        model=FixtureModel(), transform=lambda _: torch.zeros(3, 2, 2)
    )
    manifest = ModelManifest("fixture", "a" * 64, "b" * 64, 0.5)
    backend = NativeInspectionBackend(manifest, native)
    encoded = io.BytesIO()
    Image.new("RGB", (7, 3)).save(encoded, format="PNG")
    image = prepare_image(encoded.getvalue())
    result = backend.predict(image)
    assert validate_output(image, manifest, result) == "ANOMALOUS"
    assert len(result.anomaly_map) == 3 and len(result.anomaly_map[0]) == 7
    assert result.anomaly_map[0][0] == 0 and result.anomaly_map[0][-1] == 1
    backend.close()
    assert backend.native is None


def test_safe_loader_binds_manufactured_state_and_refuses_wrong_identity(
    tmp_path, monkeypatch
):
    from visionguard_inspection import native as inspection_native

    from visionguard import visa_evaluator_backend
    from visionguard.efficientad import canonical_checkpoint_sha256

    item = fixture_entry(tmp_path)
    state = {"manufactured": torch.ones(2)}
    digest = canonical_checkpoint_sha256(state)
    payload = {
        "identity": {"model": "patchcore", "category": "candle", "seed": 42},
        "model_state": state,
        "canonical_model_sha256": digest,
    }

    def save():
        encoded = io.BytesIO()
        torch.save(payload, encoded)
        raw = encoded.getvalue()
        (tmp_path / "fixture.pt").write_bytes(raw)
        item["artifact"] = {
            "path": "fixture.pt",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        item["canonical_state_sha256"] = digest
        import json

        permission_path = tmp_path / "permission.json"
        approval = json.loads(permission_path.read_text())
        approval.update(
            artifact_sha256=item["artifact"]["sha256"], canonical_state_sha256=digest
        )
        approval_raw = json.dumps(approval).encode()
        permission_path.write_bytes(approval_raw)
        item["application_permission"]["sha256"] = hashlib.sha256(
            approval_raw
        ).hexdigest()

    save()
    monkeypatch.setattr(
        inspection_native,
        "version",
        lambda name: {"torch": "2.9.1", "torchvision": "0.24.1", "anomalib": "2.6.0"}[
            name
        ],
    )
    calls = []

    def manufactured_constructor(spec, loaded, science, device):
        calls.append(spec)
        assert device == "cpu" and torch.equal(
            loaded["manufactured"], state["manufactured"]
        )
        return object()

    monkeypatch.setattr(
        visa_evaluator_backend, "native_from_state", manufactured_constructor
    )
    backend = inspection_native.approved_registration(tmp_path, item).factory()
    assert calls[0].image_threshold == 0.123 and backend.manifest.threshold == 0.123
    payload["identity"]["seed"] = 2025
    save()
    with pytest.raises(ValueError, match="binding mismatch"):
        inspection_native.approved_registration(tmp_path, item).factory()
    assert len(calls) == 1
