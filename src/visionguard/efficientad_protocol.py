"""EfficientAD v1 identity and deliberately locked Phase 3A benchmark gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS

EFFICIENTAD_PROTOCOL_ID = "efficientad-mvtecad2-v1"
EXPECTED_EFFICIENTAD_FINGERPRINT = (
    "e9d6a66e7a52f2993e984ec20278c4ca4c710198cc466df15f947adff763f69f"
)
TEACHER_SMALL_SHA256 = (
    "a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d"
)
IMAGENETTE_ARCHIVE_SHA256 = (
    "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd"
)


class EfficientAdProtocolError(ValueError):
    """Raised when EfficientAD protocol identity or authorization is invalid."""


@dataclass(frozen=True)
class EfficientAdGateInputs:
    """Phase 3B prerequisites required by the frozen public benchmark gate."""

    explicit_benchmark_mode: bool
    evaluation_split: str
    git_dirty: bool
    dataset_audit_status: str
    teacher_weight_sha256: str | None
    auxiliary_archive_sha256: str | None
    resolved_versions: Mapping[str, str]
    categories: tuple[str, ...]
    seeds: tuple[int, ...]
    recorded_fingerprint: str


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EfficientAdProtocolError(f"{location} must be a mapping")
    return value


def _fingerprint_mapping(protocol: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        protocol,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def efficientad_protocol_fingerprint(document: Mapping[str, Any]) -> str:
    """Hash scientific configuration only, excluding runtime/local paths."""

    return _fingerprint_mapping(_mapping(document.get("protocol"), "protocol"))


def load_efficientad_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the proposed frozen EfficientAD protocol."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EfficientAdProtocolError(
            f"Unable to load protocol {path}: {exc}"
        ) from exc
    root = _mapping(document, "protocol document")
    if set(root) != {"schema_version", "protocol"} or root["schema_version"] != 1:
        raise EfficientAdProtocolError(
            "Protocol document must contain schema_version 1"
        )
    validate_efficientad_snapshot(_mapping(root["protocol"], "protocol"))
    return root


def validate_efficientad_snapshot(protocol: Mapping[str, Any]) -> None:
    """Reject material drift from the reviewed Phase 3A protocol."""

    if protocol.get("id") != EFFICIENTAD_PROTOCOL_ID:
        raise EfficientAdProtocolError("EfficientAD protocol identity is invalid")
    if protocol.get("status") != "proposed_frozen_phase3a":
        raise EfficientAdProtocolError("EfficientAD protocol status is invalid")
    if protocol.get("phase3a_public_evaluation_lock") is not True:
        raise EfficientAdProtocolError("Phase 3A public-evaluation lock is required")
    dataset = _mapping(protocol.get("dataset"), "protocol.dataset")
    if tuple(dataset.get("categories", ())) != OFFICIAL_CATEGORIES:
        raise EfficientAdProtocolError("Every official category is required in order")
    reproducibility = _mapping(protocol.get("reproducibility"), "reproducibility")
    if tuple(reproducibility.get("seeds", ())) != PROTOCOL_SEEDS:
        raise EfficientAdProtocolError("Seeds differ from the comparison contract")
    model = _mapping(protocol.get("model"), "protocol.model")
    if (
        model.get("variant") != "pdn_small"
        or model.get("teacher_weight_sha256") != TEACHER_SMALL_SHA256
    ):
        raise EfficientAdProtocolError("PDN-S model or teacher identity has drifted")
    auxiliary = _mapping(protocol.get("auxiliary_data"), "auxiliary_data")
    if (
        auxiliary.get("required") is not True
        or auxiliary.get("archive_sha256") != IMAGENETTE_ARCHIVE_SHA256
    ):
        raise EfficientAdProtocolError("Required penalty-data identity has drifted")
    training = _mapping(protocol.get("training"), "training")
    if training.get("max_steps") != 70000 or training.get("batch_size") != 1:
        raise EfficientAdProtocolError("Reference training-step contract has drifted")
    calibration = _mapping(protocol.get("calibration"), "calibration")
    if (
        calibration.get("normal_only") is not True
        or calibration.get("split") != "validation"
    ):
        raise EfficientAdProtocolError("Calibration must remain validation-normal only")
    if _fingerprint_mapping(protocol) != EXPECTED_EFFICIENTAD_FINGERPRINT:
        raise EfficientAdProtocolError(
            "EfficientAD protocol differs from its reviewed fingerprint"
        )


def validate_future_benchmark_prerequisites(
    document: Mapping[str, Any], inputs: EfficientAdGateInputs
) -> str:
    """Authorize Phase 3B only when every frozen prerequisite matches."""

    protocol = _mapping(document.get("protocol"), "protocol")
    validate_efficientad_snapshot(protocol)
    fingerprint = efficientad_protocol_fingerprint(document)
    failures: list[str] = []
    if not inputs.explicit_benchmark_mode:
        failures.append("benchmark mode was not explicitly requested")
    if inputs.evaluation_split != "test_public":
        failures.append("future public evaluation split must be test_public")
    if inputs.git_dirty:
        failures.append("Git worktree is dirty")
    if inputs.dataset_audit_status != "passed":
        failures.append("dataset audit has not passed")
    if inputs.teacher_weight_sha256 != TEACHER_SMALL_SHA256:
        failures.append("teacher weight identity does not match")
    if inputs.auxiliary_archive_sha256 != IMAGENETTE_ARCHIVE_SHA256:
        failures.append("auxiliary-data identity does not match")
    if inputs.categories != OFFICIAL_CATEGORIES:
        failures.append("category set differs from the protocol")
    if inputs.seeds != PROTOCOL_SEEDS:
        failures.append("seed set differs from the protocol")
    if inputs.recorded_fingerprint != fingerprint:
        failures.append("protocol fingerprint does not match")
    expected = _mapping(protocol.get("dependencies"), "dependencies")
    for package in ("anomalib", "lightning", "torch", "torchvision"):
        if inputs.resolved_versions.get(package, "").split("+")[0] != str(
            expected[package]
        ):
            failures.append(f"{package} version does not match")
    if failures:
        raise EfficientAdProtocolError(
            "Benchmark prerequisites denied: " + "; ".join(failures)
        )
    return fingerprint


def authorize_engineering_split(split: str) -> None:
    """Allow only train/good and validation/good work during Phase 3A."""

    if split not in {"train", "validation"}:
        raise EfficientAdProtocolError(
            "Phase 3A permits only train and validation normal data"
        )
