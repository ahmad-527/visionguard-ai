from __future__ import annotations

import copy
from pathlib import Path

import pytest

from visionguard.efficientad_protocol import (
    EXPECTED_EFFICIENTAD_FINGERPRINT,
    IMAGENETTE_ARCHIVE_SHA256,
    TEACHER_SMALL_SHA256,
    EfficientAdGateInputs,
    EfficientAdProtocolError,
    authorize_engineering_split,
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
    validate_future_benchmark_prerequisites,
)
from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS

PROTOCOL = Path("configs/protocols/efficientad-mvtecad2-v1.yaml")


def valid_gate(document: dict[str, object]) -> EfficientAdGateInputs:
    return EfficientAdGateInputs(
        explicit_benchmark_mode=True,
        evaluation_split="test_public",
        git_dirty=False,
        dataset_audit_status="passed",
        teacher_weight_sha256=TEACHER_SMALL_SHA256,
        auxiliary_archive_sha256=IMAGENETTE_ARCHIVE_SHA256,
        resolved_versions={
            "anomalib": "2.6.0",
            "lightning": "2.6.5",
            "torch": "2.9.1+cu126",
            "torchvision": "0.24.1+cu126",
        },
        categories=OFFICIAL_CATEGORIES,
        seeds=PROTOCOL_SEEDS,
        recorded_fingerprint=efficientad_protocol_fingerprint(document),
    )


def test_protocol_loads_with_reviewed_fingerprint() -> None:
    document = load_efficientad_protocol(PROTOCOL)

    assert (
        efficientad_protocol_fingerprint(document) == EXPECTED_EFFICIENTAD_FINGERPRINT
    )


def test_local_runtime_paths_do_not_change_fingerprint() -> None:
    document = load_efficientad_protocol(PROTOCOL)
    changed = copy.deepcopy(document)
    changed["runtime"] = {
        "dataset_root": "C:/private/mvtec",
        "weight_cache": "/var/cache/weights",
    }

    assert efficientad_protocol_fingerprint(
        changed
    ) == efficientad_protocol_fingerprint(document)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "variant"), "pdn_medium"),
        (("training", "max_steps"), 1),
        (("preprocessing", "resize"), [128, 128]),
        (("calibration", "normal_only"), False),
        (("auxiliary_data", "archive_sha256"), "0" * 64),
    ],
)
def test_material_changes_change_fingerprint(
    path: tuple[str, str], value: object
) -> None:
    document = load_efficientad_protocol(PROTOCOL)
    changed = copy.deepcopy(document)
    changed["protocol"][path[0]][path[1]] = value

    assert efficientad_protocol_fingerprint(
        changed
    ) != efficientad_protocol_fingerprint(document)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_phase3a_engineering_splits_are_allowed(split: str) -> None:
    authorize_engineering_split(split)


@pytest.mark.parametrize(
    "split", ["test_public", "test_private", "test_private_mixed", "private"]
)
def test_phase3a_rejects_public_and_private_splits(split: str) -> None:
    with pytest.raises(EfficientAdProtocolError, match="only train and validation"):
        authorize_engineering_split(split)


def test_complete_future_gate_remains_deliberately_locked() -> None:
    document = load_efficientad_protocol(PROTOCOL)

    with pytest.raises(EfficientAdProtocolError, match="Phase 3A lock"):
        validate_future_benchmark_prerequisites(document, valid_gate(document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("explicit_benchmark_mode", False, "explicitly"),
        ("evaluation_split", "validation", "test_public"),
        ("git_dirty", True, "dirty"),
        ("dataset_audit_status", "failed", "audit"),
        ("teacher_weight_sha256", "0" * 64, "teacher"),
        ("auxiliary_archive_sha256", "0" * 64, "auxiliary"),
        ("categories", ("can",), "category"),
        ("seeds", (42,), "seed"),
        ("recorded_fingerprint", "0" * 64, "fingerprint"),
    ],
)
def test_future_gate_validates_every_prerequisite(
    field: str, value: object, message: str
) -> None:
    document = load_efficientad_protocol(PROTOCOL)
    inputs = valid_gate(document)
    changed = EfficientAdGateInputs(**{**inputs.__dict__, field: value})  # type: ignore[arg-type]

    with pytest.raises(EfficientAdProtocolError, match=message):
        validate_future_benchmark_prerequisites(document, changed)
