from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.artifacts import (
    ArtifactError,
    capture_git_state,
    dataset_audit_identity,
    new_benchmark_artifact,
    new_experiment_artifact,
    validate_artifact,
    write_artifact,
)
from visionguard.protocol import load_protocol


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


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/private/sample.png",
        "C:/private/sample.png",
        r"C:\private\sample.png",
        r"\\server\share\sample.png",
        "../sample.png",
        r"images\..\..\sample.png",
    ],
)
def test_artifact_rejects_nonportable_sample_paths(invalid_path: str) -> None:
    value = artifact()
    value["predictions"] = [{"sample_id": invalid_path, "anomaly_score": 0.1}]

    with pytest.raises(ArtifactError, match="portable relative path"):
        validate_artifact(value)


@pytest.mark.parametrize(
    "invalid_path",
    ["//server/share/map.npy", "C:/private/map.npy", "maps/../../map.npy"],
)
def test_artifact_rejects_nonportable_anomaly_map_paths(invalid_path: str) -> None:
    value = artifact()
    value["predictions"] = [
        {
            "sample_id": "test/widget.png",
            "anomaly_score": 0.1,
            "anomaly_map": {"path": invalid_path},
        }
    ]

    with pytest.raises(ArtifactError, match="portable relative path"):
        validate_artifact(value)


def test_artifact_accepts_relative_paths_with_either_slash_convention() -> None:
    value = artifact()
    value["predictions"] = [
        {
            "sample_id": r"test\widget.png",
            "anomaly_score": 0.1,
            "anomaly_map": {"path": "anomaly-maps/widget.npy"},
        }
    ]

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


def benchmark_artifact() -> dict[str, object]:
    return new_benchmark_artifact(
        protocol_document=load_protocol(
            Path("configs/protocols/patchcore-mvtecad2-v1.yaml")
        ),
        experiment_id="benchmark-can-42",
        git={"commit": "a" * 40, "branch": "benchmark", "dirty": False},
        dataset={"sha256": "b" * 64, "status": "passed"},
        category="can",
        seed=42,
        environment={"resolved_packages": {}},
        weight={"sha256": "c" * 64},
        calibration={"normal_only": True},
    )


def test_benchmark_artifact_is_bound_to_protocol_fingerprint() -> None:
    value = benchmark_artifact()

    validate_artifact(value)
    value["protocol_snapshot"]["model"]["num_neighbors"] = 1  # type: ignore[index]
    with pytest.raises(ArtifactError, match="protocol"):
        validate_artifact(value)


@pytest.mark.parametrize(("field", "value"), [("category", "easy"), ("seed", 7)])
def test_benchmark_artifact_rejects_protocol_drift(field: str, value: object) -> None:
    artifact_value = benchmark_artifact()
    artifact_value[field] = value

    with pytest.raises(ArtifactError, match=field):
        validate_artifact(artifact_value)
