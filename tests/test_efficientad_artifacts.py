from __future__ import annotations

import copy
from pathlib import Path

import pytest

from visionguard.artifacts import validate_artifact
from visionguard.efficientad_artifacts import (
    EfficientAdArtifactError,
    validate_efficientad_artifact,
)
from visionguard.efficientad_protocol import (
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
)


def artifact() -> dict[str, object]:
    document = load_efficientad_protocol(
        Path("configs/protocols/efficientad-mvtecad2-v1.yaml")
    )
    protocol = copy.deepcopy(document["protocol"])
    return {
        "artifact_schema_version": 3,
        "protocol_id": protocol["id"],
        "protocol_fingerprint": efficientad_protocol_fingerprint(document),
        "protocol_snapshot": protocol,
        "experiment_id": "phase3a-can-42",
        "run_kind": "phase3a_engineering_non_benchmark",
        "benchmark_claim": False,
        "evaluation_split": None,
        "git": {"commit": "a" * 40, "dirty": False},
        "dataset": {"status": "passed", "sha256": "b" * 64},
        "category": "can",
        "seed": 42,
        "implementation": {"name": "anomalib", "version": "2.6.0"},
        "model": {"variant": "pdn_small"},
        "training": {"steps": 2},
        "preprocessing": protocol["preprocessing"],
        "auxiliary_data": {"sha256": "c" * 64},
        "environment": {},
        "reproducibility": {},
        "weights": [{"sha256": "d" * 64}],
        "calibration": {"normal_only": True},
        "thresholds": {},
        "model_state": {"checkpoint_sha256": "e" * 64},
        "predictions": [
            {
                "sample_id": "can/validation/good/000.png",
                "anomaly_score": 0.1,
                "anomaly_map": {"sha256": "f" * 64},
            }
        ],
        "metrics": [],
        "resources": {},
        "warnings": [],
        "failures": [],
        "status": "completed",
    }


def test_schema_v3_accepts_nonbenchmark_engineering_evidence() -> None:
    validate_efficientad_artifact(artifact())
    validate_artifact(artifact())


def test_schema_v3_rejects_public_evaluation_and_benchmark_claim() -> None:
    value = artifact()
    value["evaluation_split"] = "test_public"
    value["benchmark_claim"] = True

    with pytest.raises(EfficientAdArtifactError, match="cannot claim"):
        validate_efficientad_artifact(value)


def test_schema_v3_rejects_nonfinite_prediction() -> None:
    value = artifact()
    value["predictions"][0]["anomaly_score"] = float("nan")  # type: ignore[index]

    with pytest.raises(EfficientAdArtifactError, match="finite"):
        validate_efficientad_artifact(value)


def test_schema_v3_rejects_protocol_drift_and_absolute_sample_path() -> None:
    drift = artifact()
    drift["protocol_snapshot"]["training"]["max_steps"] = 1  # type: ignore[index]
    with pytest.raises(EfficientAdArtifactError, match="snapshot"):
        validate_efficientad_artifact(drift)

    absolute = artifact()
    absolute["predictions"][0]["sample_id"] = "C:/private/image.png"  # type: ignore[index]
    with pytest.raises(EfficientAdArtifactError, match="portable"):
        validate_efficientad_artifact(absolute)


@pytest.mark.parametrize(("field", "value"), [("category", "bottle"), ("seed", 7)])
def test_schema_v3_rejects_category_and_seed_drift(field: str, value: object) -> None:
    changed = artifact()
    changed[field] = value

    with pytest.raises(EfficientAdArtifactError, match=field):
        validate_efficientad_artifact(changed)
