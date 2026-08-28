from __future__ import annotations

import copy
from pathlib import Path

import pytest

from visionguard.protocol import (
    OFFICIAL_CATEGORIES,
    PROTOCOL_SEEDS,
    BenchmarkGateInputs,
    ProtocolError,
    authorize_public_benchmark,
    load_protocol,
    protocol_fingerprint,
    reject_private_evaluation,
)

PROTOCOL_PATH = Path("configs/protocols/patchcore-mvtecad2-v1.yaml")


def valid_gate(document: dict[str, object]) -> BenchmarkGateInputs:
    return BenchmarkGateInputs(
        explicit_benchmark_mode=True,
        evaluation_split="test_public",
        git_dirty=False,
        dataset_audit_status="passed",
        weight_sha256="a" * 64,
        resolved_versions={
            "anomalib": "2.6.0",
            "timm": "1.0.28",
            "torch": "2.9.1+cu126",
            "torchvision": "0.24.1+cu126",
        },
        categories=OFFICIAL_CATEGORIES,
        seeds=PROTOCOL_SEEDS,
        recorded_fingerprint=protocol_fingerprint(document),
    )


def test_frozen_protocol_loads_and_has_stable_fingerprint() -> None:
    first = load_protocol(PROTOCOL_PATH)
    second = load_protocol(PROTOCOL_PATH)

    assert protocol_fingerprint(first) == protocol_fingerprint(second)
    assert len(protocol_fingerprint(first)) == 64


def test_benchmark_relevant_change_changes_fingerprint() -> None:
    original = load_protocol(PROTOCOL_PATH)
    changed = copy.deepcopy(original)
    changed["protocol"]["model"]["num_neighbors"] = 1

    assert protocol_fingerprint(changed) != protocol_fingerprint(original)


def test_runtime_local_paths_do_not_change_fingerprint() -> None:
    original = load_protocol(PROTOCOL_PATH)
    with_runtime = copy.deepcopy(original)
    with_runtime["runtime"] = {"dataset_root": "C:/private/mvtec"}

    assert protocol_fingerprint(with_runtime) == protocol_fingerprint(original)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("explicit_benchmark_mode", False, "explicitly"),
        ("evaluation_split", "validation", "test_public"),
        ("git_dirty", True, "dirty"),
        ("dataset_audit_status", "failed", "audit"),
        ("weight_sha256", None, "weight"),
        ("categories", ("can",), "category"),
        ("seeds", (42,), "seed"),
        ("recorded_fingerprint", "0" * 64, "fingerprint"),
    ],
)
def test_benchmark_gate_rejects_each_missing_prerequisite(
    field: str, value: object, message: str
) -> None:
    document = load_protocol(PROTOCOL_PATH)
    inputs = valid_gate(document)
    changed = BenchmarkGateInputs(
        **{**inputs.__dict__, field: value}  # type: ignore[arg-type]
    )

    with pytest.raises(ProtocolError, match=message):
        authorize_public_benchmark(document, changed)


def test_benchmark_gate_accepts_complete_public_request() -> None:
    document = load_protocol(PROTOCOL_PATH)

    assert authorize_public_benchmark(document, valid_gate(document)) == (
        protocol_fingerprint(document)
    )


def test_private_evaluation_remains_separately_gated() -> None:
    with pytest.raises(ProtocolError, match="human authorization"):
        reject_private_evaluation()
