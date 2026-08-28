"""Frozen benchmark protocol identity and deliberate authorization gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_ID = "patchcore-mvtecad2-v1"
EXPECTED_PROTOCOL_FINGERPRINT = (
    "106a668e9fac47afd6746b7337b276ecb6f02999822da92ac95f650d4b5f99af"
)
OFFICIAL_CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
PROTOCOL_SEEDS = (42, 123, 2026)


class ProtocolError(ValueError):
    """Raised when a protocol or benchmark authorization is invalid."""


@dataclass(frozen=True)
class BenchmarkGateInputs:
    """Measured prerequisites required before a public benchmark may start."""

    explicit_benchmark_mode: bool
    evaluation_split: str
    git_dirty: bool
    dataset_audit_status: str
    weight_sha256: str | None
    resolved_versions: Mapping[str, str]
    categories: tuple[str, ...]
    seeds: tuple[int, ...]
    recorded_fingerprint: str


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{location} must be a mapping")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the frozen protocol document without runtime paths."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Unable to load protocol {path}: {exc}") from exc
    root = _mapping(document, "protocol document")
    if set(root) != {"schema_version", "protocol"} or root["schema_version"] != 1:
        raise ProtocolError("Protocol document must contain schema_version 1")
    protocol = _mapping(root["protocol"], "protocol")
    _validate_frozen_protocol(protocol)
    return root


def _validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("id") != PROTOCOL_ID or protocol.get("status") != "frozen":
        raise ProtocolError("Benchmark protocol must have the frozen v1 identity")
    dataset = _mapping(protocol.get("dataset"), "protocol.dataset")
    if tuple(dataset.get("categories", ())) != OFFICIAL_CATEGORIES:
        raise ProtocolError("Protocol must contain every official category in order")
    reproducibility = _mapping(
        protocol.get("reproducibility"), "protocol.reproducibility"
    )
    if tuple(reproducibility.get("seeds", ())) != PROTOCOL_SEEDS:
        raise ProtocolError("Protocol seed set differs from the frozen seed policy")
    preprocessing = _mapping(protocol.get("preprocessing"), "protocol.preprocessing")
    if preprocessing.get("center_crop") is not None:
        raise ProtocolError("The v1 protocol forbids center cropping")
    calibration = _mapping(protocol.get("calibration"), "protocol.calibration")
    if calibration.get("normal_only") is not True:
        raise ProtocolError("Benchmark calibration must be normal-only")
    metrics = _mapping(protocol.get("metrics"), "protocol.metrics")
    ranking = _mapping(metrics.get("official_ranking"), "official ranking metric")
    if ranking.get("name") != "au_pro" or ranking.get("fpr_limit") != 0.05:
        raise ProtocolError("Official ranking metric must be AU-PRO at FPR 0.05")
    if _fingerprint_mapping(protocol) != EXPECTED_PROTOCOL_FINGERPRINT:
        raise ProtocolError(
            "Frozen protocol content differs from its reviewed identity"
        )


def validate_protocol_snapshot(protocol: Mapping[str, Any]) -> None:
    """Reject a snapshot that differs from the reviewed immutable protocol."""

    _validate_frozen_protocol(protocol)


def _fingerprint_mapping(protocol: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        protocol,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def protocol_fingerprint(document: Mapping[str, Any]) -> str:
    """Hash only the canonical scientific protocol, excluding runtime context."""

    protocol = _mapping(document.get("protocol"), "protocol")
    return _fingerprint_mapping(protocol)


def authorize_public_benchmark(
    document: Mapping[str, Any], inputs: BenchmarkGateInputs
) -> str:
    """Authorize an explicit public benchmark only when every frozen gate passes."""

    protocol = _mapping(document.get("protocol"), "protocol")
    _validate_frozen_protocol(protocol)
    expected_fingerprint = protocol_fingerprint(document)
    failures: list[str] = []
    if not inputs.explicit_benchmark_mode:
        failures.append("benchmark mode was not explicitly requested")
    if inputs.evaluation_split != "test_public":
        failures.append("only test_public is authorized by the public benchmark gate")
    if inputs.git_dirty:
        failures.append("Git worktree is dirty")
    if inputs.dataset_audit_status != "passed":
        failures.append("dataset audit has not passed")
    if not inputs.weight_sha256 or len(inputs.weight_sha256) != 64:
        failures.append("pretrained weight SHA-256 is unverified")
    if inputs.categories != OFFICIAL_CATEGORIES:
        failures.append("category set differs from the frozen protocol")
    if inputs.seeds != PROTOCOL_SEEDS:
        failures.append("seed set differs from the frozen protocol")
    if inputs.recorded_fingerprint != expected_fingerprint:
        failures.append("protocol fingerprint does not match")
    expected_versions = _mapping(protocol.get("dependencies"), "dependencies")
    for package in ("anomalib", "timm", "torch", "torchvision"):
        actual = inputs.resolved_versions.get(package, "").split("+")[0]
        if actual != str(expected_versions[package]):
            failures.append(f"{package} version does not match the protocol")
    if failures:
        raise ProtocolError("Benchmark authorization denied: " + "; ".join(failures))
    return expected_fingerprint


def reject_private_evaluation() -> None:
    """Keep private-server activity behind a separate future human gate."""

    raise ProtocolError("Private evaluation requires separate human authorization")
