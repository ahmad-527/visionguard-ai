"""Schema-v3 engineering artifacts for frozen EfficientAD preparation."""

from __future__ import annotations

import math
from typing import Any

from visionguard.efficientad_protocol import (
    EFFICIENTAD_PROTOCOL_ID,
    efficientad_protocol_fingerprint,
    validate_efficientad_snapshot,
)
from visionguard.paths import portable_relative_path
from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS


class EfficientAdArtifactError(ValueError):
    """Raised when an EfficientAD artifact violates schema v3."""


def validate_efficientad_artifact(artifact: dict[str, Any]) -> None:
    """Validate provenance-rich engineering evidence without benchmark claims."""

    required = {
        "artifact_schema_version",
        "protocol_id",
        "protocol_fingerprint",
        "protocol_snapshot",
        "experiment_id",
        "run_kind",
        "benchmark_claim",
        "git",
        "dataset",
        "category",
        "seed",
        "implementation",
        "model",
        "training",
        "preprocessing",
        "auxiliary_data",
        "environment",
        "reproducibility",
        "weights",
        "calibration",
        "thresholds",
        "model_state",
        "predictions",
        "metrics",
        "resources",
        "warnings",
        "failures",
        "status",
    }
    missing = required - set(artifact)
    if missing:
        raise EfficientAdArtifactError(
            "Artifact is missing: " + ", ".join(sorted(missing))
        )
    if artifact["artifact_schema_version"] != 3:
        raise EfficientAdArtifactError("EfficientAD artifacts require schema version 3")
    if artifact["protocol_id"] != EFFICIENTAD_PROTOCOL_ID:
        raise EfficientAdArtifactError("EfficientAD protocol identity is invalid")
    if artifact["category"] not in OFFICIAL_CATEGORIES:
        raise EfficientAdArtifactError("Artifact category is outside the protocol")
    if artifact["seed"] not in PROTOCOL_SEEDS:
        raise EfficientAdArtifactError("Artifact seed is outside the protocol")
    if (
        artifact["benchmark_claim"] is not False
        or artifact["run_kind"] != "phase3a_engineering_non_benchmark"
    ):
        raise EfficientAdArtifactError(
            "Phase 3A artifacts cannot claim benchmark status"
        )
    snapshot = artifact["protocol_snapshot"]
    try:
        validate_efficientad_snapshot(snapshot)
    except ValueError as exc:
        raise EfficientAdArtifactError("Protocol snapshot is invalid") from exc
    if artifact["protocol_fingerprint"] != efficientad_protocol_fingerprint(
        {"protocol": snapshot}
    ):
        raise EfficientAdArtifactError("Protocol fingerprint does not match")
    if artifact.get("evaluation_split") is not None:
        raise EfficientAdArtifactError(
            "Phase 3A artifact must not contain an evaluation split"
        )
    for prediction in artifact["predictions"]:
        try:
            portable_relative_path(prediction["sample_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EfficientAdArtifactError(
                "Prediction sample_id is not portable"
            ) from exc
        score = prediction.get("anomaly_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise EfficientAdArtifactError("Prediction score must be finite")
        anomaly_map = prediction.get("anomaly_map")
        if (
            not isinstance(anomaly_map, dict)
            or len(str(anomaly_map.get("sha256", ""))) != 64
        ):
            raise EfficientAdArtifactError("Prediction anomaly map requires SHA-256")
    checkpoint = artifact["model_state"].get("checkpoint_sha256")
    if checkpoint is not None and len(str(checkpoint)) != 64:
        raise EfficientAdArtifactError("Checkpoint identity must be SHA-256")
    weights = artifact["weights"]
    if not isinstance(weights, list) or not weights:
        raise EfficientAdArtifactError("At least one verified weight is required")
    if any(len(str(weight.get("sha256", ""))) != 64 for weight in weights):
        raise EfficientAdArtifactError("Every weight requires SHA-256 identity")
