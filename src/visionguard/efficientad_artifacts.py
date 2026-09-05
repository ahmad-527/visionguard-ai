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
    run_kind = artifact["run_kind"]
    if run_kind == "phase3a_engineering_non_benchmark":
        if artifact["benchmark_claim"] is not False:
            raise EfficientAdArtifactError(
                "Phase 3A artifacts cannot claim benchmark status"
            )
        if artifact.get("evaluation_split") is not None:
            raise EfficientAdArtifactError(
                "Phase 3A artifact must not contain an evaluation split"
            )
    elif run_kind == "phase3b_protocol_authorized_public_benchmark":
        if artifact["benchmark_claim"] is not True:
            raise EfficientAdArtifactError("Phase 3B benchmark claim is required")
        if artifact.get("evaluation_split") != "test_public":
            raise EfficientAdArtifactError(
                "Phase 3B evaluation split must be test_public"
            )
        if (
            not isinstance(artifact["git"], dict)
            or artifact["git"].get("dirty") is not False
        ):
            raise EfficientAdArtifactError("Phase 3B requires a clean Git state")
    else:
        raise EfficientAdArtifactError("EfficientAD artifact run_kind is invalid")
    snapshot = artifact["protocol_snapshot"]
    try:
        validate_efficientad_snapshot(snapshot)
    except ValueError as exc:
        raise EfficientAdArtifactError("Protocol snapshot is invalid") from exc
    if artifact["protocol_fingerprint"] != efficientad_protocol_fingerprint(
        {"protocol": snapshot}
    ):
        raise EfficientAdArtifactError("Protocol fingerprint does not match")
    sample_ids: list[str] = []
    for prediction in artifact["predictions"]:
        try:
            sample_id = portable_relative_path(prediction["sample_id"]).as_posix()
        except (KeyError, TypeError, ValueError) as exc:
            raise EfficientAdArtifactError(
                "Prediction sample_id is not portable"
            ) from exc
        sample_ids.append(sample_id)
        score = prediction.get("anomaly_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise EfficientAdArtifactError("Prediction score must be finite")
        anomaly_map = prediction.get("anomaly_map")
        if (
            not isinstance(anomaly_map, dict)
            or len(str(anomaly_map.get("sha256", ""))) != 64
        ):
            raise EfficientAdArtifactError("Prediction anomaly map requires SHA-256")
    if len(sample_ids) != len(set(sample_ids)):
        raise EfficientAdArtifactError("Prediction sample_id values must be unique")
    checkpoint = artifact["model_state"].get("checkpoint_sha256")
    if checkpoint is not None and len(str(checkpoint)) != 64:
        raise EfficientAdArtifactError("Checkpoint identity must be SHA-256")
    weights = artifact["weights"]
    if not isinstance(weights, list) or not weights:
        raise EfficientAdArtifactError("At least one verified weight is required")
    if any(len(str(weight.get("sha256", ""))) != 64 for weight in weights):
        raise EfficientAdArtifactError("Every weight requires SHA-256 identity")
    if (
        run_kind == "phase3b_protocol_authorized_public_benchmark"
        and artifact["status"] == "completed"
    ):
        metrics = artifact.get("category_metrics")
        if not isinstance(metrics, dict) or set(metrics) != {
            "au_pro_0.05",
            "pixel_f1",
            "image_f1",
            "image_auroc",
        }:
            raise EfficientAdArtifactError(
                "Completed Phase 3B artifacts require every frozen metric"
            )
        if not artifact["predictions"]:
            raise EfficientAdArtifactError(
                "Completed Phase 3B artifacts require public predictions"
            )
        if artifact["training"].get("final_optimization_step") != 70000:
            raise EfficientAdArtifactError(
                "Completed Phase 3B artifacts require optimization step 70000"
            )
        calibration = artifact["calibration"]
        if (
            not isinstance(calibration, dict)
            or calibration.get("normal_only") is not True
            or calibration.get("split") != "validation"
        ):
            raise EfficientAdArtifactError(
                "Completed Phase 3B calibration must be validation-normal only"
            )
        thresholds = artifact["thresholds"]
        if not isinstance(thresholds, dict) or any(
            not isinstance(thresholds.get(name), (int, float))
            or not math.isfinite(float(thresholds[name]))
            for name in ("image", "pixel")
        ):
            raise EfficientAdArtifactError(
                "Completed Phase 3B thresholds must be finite"
            )
        for name, metric in metrics.items():
            if (
                not isinstance(metric, dict)
                or metric.get("status") != "defined"
                or not isinstance(metric.get("value"), (int, float))
                or not math.isfinite(float(metric["value"]))
            ):
                raise EfficientAdArtifactError(
                    f"Completed Phase 3B metric {name} must be finite and defined"
                )
        for sample_id, prediction in zip(
            sample_ids, artifact["predictions"], strict=True
        ):
            if not sample_id.startswith(f"{artifact['category']}/test_public/"):
                raise EfficientAdArtifactError(
                    "Completed Phase 3B predictions must use test_public"
                )
            anomaly_map = prediction["anomaly_map"]
            try:
                portable_relative_path(anomaly_map["path"])
                portable_relative_path(anomaly_map["thresholded_path"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EfficientAdArtifactError(
                    "Completed Phase 3B map paths must be portable"
                ) from exc
            if len(str(anomaly_map.get("thresholded_sha256", ""))) != 64:
                raise EfficientAdArtifactError(
                    "Completed Phase 3B binary map requires SHA-256"
                )
