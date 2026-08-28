from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.artifacts import (
    ArtifactError,
    capture_git_state,
    dataset_audit_identity,
    new_experiment_artifact,
    validate_artifact,
    write_artifact,
)


def artifact() -> dict[str, object]:
    return new_experiment_artifact(
        experiment_id="synthetic",
        git={"commit": "a" * 40, "branch": "test", "dirty": False},
        dataset={"sha256": "b" * 64, "status": "passed"},
        configuration={"schema_version": 1},
        environment={"schema_version": 1},
        reproducibility={"python": {"status": "configured", "seed": 1}},
    )


def test_artifact_schema_accepts_unmeasured_skeleton() -> None:
    validate_artifact(artifact())


def test_git_state_capture_handles_repository_path() -> None:
    state = capture_git_state(Path.cwd().resolve())

    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)


def test_artifact_cannot_claim_benchmark() -> None:
    value = artifact()
    value["benchmark_claim"] = True

    with pytest.raises(ArtifactError, match="cannot claim"):
        validate_artifact(value)


def test_artifact_rejects_nonfinite_prediction() -> None:
    value = artifact()
    value["predictions"] = [{"sample_id": "one", "anomaly_score": float("nan")}]

    with pytest.raises(ArtifactError, match="finite"):
        validate_artifact(value)


def test_artifact_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_artifact(path, artifact())

    with pytest.raises(ArtifactError, match="overwrite"):
        write_artifact(path, artifact())


def test_dataset_audit_identity_omits_local_root(tmp_path: Path) -> None:
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {"name": "sample", "version": "one"},
                "root": "C:/private/dataset",
                "summary": {
                    "status": "passed",
                    "image_count": 1,
                    "mask_count": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "expected_overlap_group_count": 0,
                    "unexpected_overlap_group_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    identity = dataset_audit_identity(report_path)

    assert identity["status"] == "passed"
    assert "root" not in identity
    assert len(identity["sha256"]) == 64


def test_failed_dataset_audit_is_rejected(tmp_path: Path) -> None:
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {"name": "sample", "version": "one"},
                "summary": {"status": "failed"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="passed"):
        dataset_audit_identity(report_path)


def test_dataset_audit_must_match_runtime_root(tmp_path: Path) -> None:
    report_path = tmp_path / "audit.json"
    measured_root = tmp_path / "measured"
    measured_root.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {"name": "sample", "version": "one"},
                "root": str(measured_root),
                "summary": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="does not match"):
        dataset_audit_identity(report_path, expected_root=tmp_path / "different")
