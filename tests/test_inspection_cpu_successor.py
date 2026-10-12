"""Actual native CPU execution with random/generated states, NOT trained acceptance.

No optimizer, fit, feature-bank training, calibration, pretrained download or
existing artifact is used. All serialized states and permissions are manufactured
pytest temporary fixtures. The fixture thresholds are constants, not calibration.
"""

import gc
import hashlib
import io
import json
import os
import weakref
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

pytest.importorskip("visionguard_inspection")
from visionguard_inspection.native import approved_registration
from visionguard_inspection.registry import Registry
from visionguard_inspection.runtime import MODEL_VERSIONS, PROFILE_ID, verify_runtime
from visionguard_inspection.service import Engine, create_app

from visionguard.efficientad import canonical_checkpoint_sha256
from visionguard.inspection_contract import (
    InspectionError,
    prepare_image,
    validate_output,
)

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    os.environ.get("VISIONGUARD_CPU_SUCCESSOR_TESTS") != "1",
    reason="Explicit manufactured-native CPU successor suite not requested",
)


def bound(root, name, raw):
    (root / name).write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def document(root, name, value):
    return bound(root, name, json.dumps(value).encode())


@pytest.fixture(scope="module", params=["patchcore", "efficientad"])
def generated_state(request):
    verify_runtime()
    assert torch.version.cuda is None
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    if request.param == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        model = PatchcoreModel(
            backbone="wide_resnet50_2.racm_in1k",
            layers=["layer2", "layer3"],
            pre_trained=False,
            num_neighbors=9,
        )
        # Constant synthetic neighbours, not fitted from any images/features.
        model.memory_bank = torch.linspace(0, 1, 16 * 1536).reshape(16, 1536)
    else:
        from anomalib.models.image.efficient_ad.torch_model import (
            EfficientAdModel,
            EfficientAdModelSize,
        )

        model = EfficientAdModel(
            teacher_out_channels=384,
            model_size=EfficientAdModelSize.S,
            padding=False,
            pad_maps=True,
        )
        # Manufactured normalization/quantile buffers, NOT learned/calibrated.
        with torch.no_grad():
            model.mean_std["std"].fill_(1)
            model.quantiles["qb_st"].fill_(1)
            model.quantiles["qb_ae"].fill_(1)
    state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    del model
    return request.param, state


def manufacture(root, model, state, *, identity=None, declared=None, extra=None):
    science = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / f"configs/protocols/{model}-visa-v1.yaml"
        ).read_text()
    )["protocol"]["scientific"]
    digest = canonical_checkpoint_sha256(state)
    encoded = io.BytesIO()
    torch.save(
        {
            "identity": identity or {"model": model, "category": "candle", "seed": 42},
            "model_state": state,
            "canonical_model_sha256": declared or digest,
            **(extra or {}),
        },
        encoded,
    )
    item = {
        "model_id": "manufactured-native",
        "model": model,
        "category": "candle",
        "seed": 42,
        "canonical_state_sha256": digest,
        "artifact": bound(root, "generated.pt", encoded.getvalue()),
        "science": document(root, "science.json", science),
        "calibration": document(
            root,
            "calibration.json",
            {
                "image": {"threshold": 0.123},
                "pixel": {"threshold": 0.456},
            },
        ),
    }
    item["application_permission"] = document(
        root,
        "permission.json",
        {
            "schema_version": 1,
            "scope": "application-development-inference",
            "source_role": "development",
            "model_id": item["model_id"],
            "model": model,
            "category": "candle",
            "seed": 42,
            "canonical_state_sha256": digest,
            "artifact_sha256": item["artifact"]["sha256"],
            "science_sha256": item["science"]["sha256"],
            "calibration_sha256": item["calibration"]["sha256"],
            "device": "cpu",
            "input_scope": "generated-or-development-non-held-out",
            "runtime_profile_id": PROFILE_ID,
            "runtime_versions": dict(MODEL_VERSIONS),
            "human_attestation": "MANUFACTURED NATIVE TEST FIXTURE ONLY - NOT APPROVAL",
            "expires_at_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
    )
    return item


def test_native_safe_restoration_inference_coordinates_and_lifecycle(
    tmp_path,
    generated_state,
    monkeypatch,
):
    model, state = generated_state
    item = manufacture(tmp_path, model, state)
    original_load = torch.load
    calls = []

    def safe_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return original_load(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "PT2/JIT/pretrained loading is outside the admitted tensor-state path"
        )

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(torch.export, "load", forbidden)
    monkeypatch.setattr(torch.jit, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    registration = approved_registration(tmp_path, item)
    backend = registration.factory()
    assert len(calls) == 1
    assert (
        registration.manifest.threshold == 0.123
        and registration.pixel_threshold == 0.456
    )
    native = backend.native
    assert not native.model.training
    assert all(
        not p.requires_grad and p.device.type == "cpu"
        for p in native.model.parameters()
    )
    assert (
        canonical_checkpoint_sha256(native.model.state_dict())
        == item["canonical_state_sha256"]
    )
    encoded = io.BytesIO()
    # Deliberately non-square original dimensions; no external assets.
    rgb = Image.new("RGB", (7, 3), (64, 128, 192))
    rgb.save(encoded, format="PNG")
    prepared = prepare_image(encoded.getvalue())
    tensor = native.transform(rgb)
    assert tensor.shape == (3, 256, 256) and tensor.dtype == torch.float32
    expected = torch.tensor([64, 128, 192], dtype=torch.float32) / 255
    if model == "patchcore":
        expected = (expected - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
            [0.229, 0.224, 0.225]
        )
    torch.testing.assert_close(tensor[:, 128, 128], expected)
    result = backend.predict(prepared)
    assert len(result.anomaly_map) == 3 and len(result.anomaly_map[0]) == 7
    validate_output(
        prepared, registration.manifest, result
    )  # finite score/map + all identities
    # Equality is NORMAL; do not change existing calibration during inference.
    assert (
        validate_output(
            prepared, replace(registration.manifest, threshold=result.score), result
        )
        == "NORMAL"
    )
    assert registration.manifest.threshold == 0.123
    with torch.inference_mode():
        raw = native.model(tensor.unsqueeze(0))
        expected_map = torch.nn.functional.interpolate(
            raw.anomaly_map, size=(3, 7), mode="bilinear", align_corners=False
        )[0, 0]
    torch.testing.assert_close(
        torch.tensor(result.anomaly_map), expected_map, rtol=1e-5, atol=1e-5
    )
    reference = weakref.ref(native.model)
    del native, raw
    backend.close()
    gc.collect()
    assert backend.native is None and reference() is None
    # Reachability/CPU lifecycle only, not proof about the historical GPU allocator.


@pytest.mark.parametrize("damage", ["identity", "declared", "canonical", "strict-keys"])
def test_actual_safe_loading_refuses_mismatched_or_incomplete_states(
    tmp_path, generated_state, damage
):
    model, original = generated_state
    state = dict(original)
    kwargs = {}
    if damage == "identity":
        kwargs["identity"] = {"model": model, "category": "candle", "seed": 2025}
    elif damage == "declared":
        kwargs["declared"] = "0" * 64
    elif damage == "strict-keys":
        # Hash-bound but incomplete state must still fail strict restoration.
        state.pop(next(iter(state)))
    item = manufacture(tmp_path, model, state, **kwargs)
    if damage == "canonical":
        item["canonical_state_sha256"] = "0" * 64
        approval = json.loads((tmp_path / "permission.json").read_text())
        approval["canonical_state_sha256"] = "0" * 64
        item["application_permission"] = document(tmp_path, "permission.json", approval)
    registration = approved_registration(tmp_path, item)
    with pytest.raises((InspectionError, RuntimeError)):
        registration.factory()


class InnocentOpaqueObject:
    """Non-tensor pickle global: no code execution payload or harmful behaviour."""


def test_weights_only_rejects_opaque_global_without_unsafe_fallback(
    tmp_path, monkeypatch
):
    verify_runtime()
    item = manufacture(
        tmp_path,
        "patchcore",
        {"manufactured": torch.ones(2)},
        extra={"opaque": InnocentOpaqueObject()},
    )
    import pickle

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        approved_registration(tmp_path, item).factory()
    assert InnocentOpaqueObject not in torch.serialization.get_safe_globals()


def test_actual_constructor_failure_preserves_exception_and_quarantine(tmp_path):
    verify_runtime()
    # Hash-valid tensor envelope, wrong native keys. No trained state.
    registration = approved_registration(
        tmp_path,
        manufacture(tmp_path, "efficientad", {"manufactured": torch.ones(2)}),
    )
    engine = Engine(None)
    image = io.BytesIO()
    Image.new("RGB", (7, 3)).save(image, format="PNG")
    with pytest.raises(RuntimeError, match="state_dict") as original:
        engine.run(registration, image.getvalue())
    assert original.value.__cause__ is None  # constructor exception not replaced
    assert engine.quarantined and engine.resident is None
    assert engine.lifecycle_failure == "backend_initialization_failed"
    with pytest.raises(InspectionError, match="quarantined"):
        engine.run(registration, image.getvalue())
    with pytest.raises(InspectionError, match="cleanup unverified"):
        engine.shutdown()


def test_native_generated_state_through_http_and_revoked_permission(
    tmp_path, generated_state
):
    model, state = generated_state
    registration = approved_registration(tmp_path, manufacture(tmp_path, model, state))
    encoded = io.BytesIO()
    Image.new("RGB", (7, 3), (64, 128, 192)).save(encoded, format="PNG")
    headers = {
        "content-type": "image/png",
        "x-visionguard-client": "inspection-v1",
        "x-visionguard-input-role": "generated-or-development-non-held-out",
    }
    with TestClient(
        create_app(Registry((registration,))), base_url="http://127.0.0.1"
    ) as api:
        assert api.get("/api/v1/ready").json()["native_ready"] is False
        result = api.post(
            "/api/v1/inspect/manufactured-native",
            content=encoded.getvalue(),
            headers=headers,
        )
        assert result.status_code == 200
        body = result.json()
        assert (body["width"], body["height"]) == (7, 3)
        assert body["threshold"] == 0.123
        assert body["model"]["artifact_sha256"] == registration.manifest.artifact_sha256
        assert body["decision"] == ("ANOMALOUS" if body["score"] > 0.123 else "NORMAL")
        assert api.get("/api/v1/ready").json()["native_ready"] is True
        (tmp_path / "permission.json").write_bytes(b"{}")
        assert api.get("/api/v1/ready").status_code == 503
        refused = api.post(
            "/api/v1/inspect/manufactured-native",
            content=encoded.getvalue(),
            headers=headers,
        )
        assert refused.status_code == 422 and refused.json()["decision"] is None


@pytest.mark.parametrize("change", ["permission", "science", "calibration", "expired"])
def test_actual_runtime_permission_configuration_rechecked_before_loading(
    tmp_path, change, monkeypatch
):
    # Tiny envelope: admission must fail before construction/deserialization.
    item = manufacture(tmp_path, "patchcore", {"manufactured": torch.ones(2)})
    entry = approved_registration(tmp_path, item)
    if change == "expired":

        class Later(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(UTC) + timedelta(hours=2)

        monkeypatch.setattr("visionguard_inspection.native.datetime", Later)
    else:
        (tmp_path / f"{change}.json").write_bytes(b"{}")

    def forbidden(*args, **kwargs):
        pytest.fail("Rejected permission/configuration must not deserialize tensors")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(InspectionError):
        entry.factory()
