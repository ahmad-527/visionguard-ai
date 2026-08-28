"""Versioned machine-readable experiment artifact contracts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.paths import portable_relative_path
from visionguard.protocol import (
    OFFICIAL_CATEGORIES,
    PROTOCOL_ID,
    PROTOCOL_SEEDS,
    ProtocolError,
    protocol_fingerprint,
    validate_protocol_snapshot,
)


class ArtifactError(ValueError):
    """Raised when artifact data is incomplete, malformed, or mutable."""


def _artifact_relative_path(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{location} must be a non-empty relative path")
    try:
        portable_relative_path(value.strip())
    except ValueError as exc:
        raise ArtifactError(
            f"{location} must be a portable relative path without '..'"
        ) from exc


def sha256_file(path: Path) -> str:
    """Measure the SHA-256 identity of an existing file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_git_state(repository: Path) -> dict[str, Any]:
    """Capture Git revision and dirty state without changing the repository."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={repository.as_posix()}", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        branch = git("branch", "--show-current")
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or type(exc).__name__
        raise ArtifactError(f"Unable to capture Git state: {detail}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactError(
            f"Unable to capture Git state: {type(exc).__name__}"
        ) from exc
    return {"commit": revision, "branch": branch, "dirty": bool(status)}


def dataset_audit_identity(
    report_path: Path, *, expected_root: Path | None = None
) -> dict[str, Any]:
    """Validate a passed audit and retain only portable identity/status fields."""

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Unable to read dataset audit report: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 2:
        raise ArtifactError("Dataset audit report must use schema version 2")
    summary = report.get("summary")
    dataset = report.get("dataset")
    if not isinstance(summary, dict) or summary.get("status") != "passed":
        raise ArtifactError("Dataset audit report must have passed status")
    if not isinstance(dataset, dict):
        raise ArtifactError("Dataset audit report is missing dataset identity")
    if expected_root is not None:
        reported_root = report.get("root")
        if not isinstance(reported_root, str):
            raise ArtifactError("Dataset audit report is missing its measured root")
        if Path(reported_root).resolve() != expected_root.resolve():
            raise ArtifactError(
                "Dataset root does not match the root measured by the audit report"
            )
    return {
        "sha256": sha256_file(report_path),
        "schema_version": 2,
        "dataset": {
            "name": dataset.get("name"),
            "version": dataset.get("version"),
        },
        "status": "passed",
        "summary": {
            key: summary.get(key)
            for key in (
                "image_count",
                "mask_count",
                "error_count",
                "warning_count",
                "expected_overlap_group_count",
                "unexpected_overlap_group_count",
            )
        },
    }


def new_experiment_artifact(
    *,
    experiment_id: str,
    git: dict[str, Any],
    dataset: dict[str, Any],
    configuration: dict[str, Any],
    environment: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    """Create an unmeasured artifact skeleton populated only from runtime inputs."""

    return {
        "artifact_schema_version": 1,
        "experiment_id": experiment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_kind": "engineering_smoke_non_benchmark",
        "benchmark_claim": False,
        "git": git,
        "dataset": dataset,
        "configuration": configuration,
        "environment": environment,
        "reproducibility": reproducibility,
        "weights": [],
        "model_state": {},
        "thresholds": {},
        "predictions": [],
        "metrics": [],
        "resources": {},
        "failures": [],
        "warnings": [],
        "status": "initialized",
    }


def new_benchmark_artifact(
    *,
    protocol_document: dict[str, Any],
    experiment_id: str,
    git: dict[str, Any],
    dataset: dict[str, Any],
    category: str,
    seed: int,
    environment: dict[str, Any],
    weight: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Create a protocol-bound benchmark artifact after an external gate passes."""

    protocol = protocol_document["protocol"]
    return {
        "artifact_schema_version": 2,
        "protocol_id": protocol["id"],
        "protocol_fingerprint": protocol_fingerprint(protocol_document),
        "protocol_snapshot": protocol,
        "experiment_id": experiment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_kind": "protocol_authorized_public_benchmark",
        "benchmark_claim": True,
        "git": git,
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "environment": environment,
        "weight": weight,
        "calibration": calibration,
        "thresholds": {},
        "predictions": [],
        "metric_implementation": {},
        "category_metrics": {},
        "aggregation": protocol["metrics"]["official_ranking"]["category_aggregation"],
        "resources": {},
        "warnings": [],
        "failures": [],
        "status": "initialized",
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required fields and measured-value types for schema version 1."""

    if artifact.get("artifact_schema_version") == 2:
        _validate_benchmark_artifact(artifact)
        return

    required = {
        "artifact_schema_version",
        "experiment_id",
        "generated_at",
        "run_kind",
        "benchmark_claim",
        "git",
        "dataset",
        "configuration",
        "environment",
        "reproducibility",
        "weights",
        "model_state",
        "thresholds",
        "predictions",
        "metrics",
        "resources",
        "failures",
        "warnings",
        "status",
    }
    missing = required - set(artifact)
    if missing:
        raise ArtifactError(f"Artifact is missing: {', '.join(sorted(missing))}")
    if artifact["artifact_schema_version"] != 1:
        raise ArtifactError("Unsupported artifact schema version")
    if artifact["benchmark_claim"] is not False:
        raise ArtifactError("Phase 2A artifacts cannot claim benchmark status")
    if artifact["run_kind"] != "engineering_smoke_non_benchmark":
        raise ArtifactError("Phase 2A artifact run_kind is invalid")
    if not isinstance(artifact["predictions"], list):
        raise ArtifactError("Artifact predictions must be a list")
    for prediction in artifact["predictions"]:
        if not isinstance(prediction, dict):
            raise ArtifactError("Each prediction must be a mapping")
        _artifact_relative_path(prediction.get("sample_id"), "Prediction sample_id")
        anomaly_map = prediction.get("anomaly_map")
        if anomaly_map is not None:
            if not isinstance(anomaly_map, dict):
                raise ArtifactError("Prediction anomaly_map must be a mapping")
            _artifact_relative_path(
                anomaly_map.get("path"), "Prediction anomaly_map.path"
            )
        score = prediction.get("anomaly_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ArtifactError("Prediction anomaly_score must be finite")


def _validate_benchmark_artifact(artifact: dict[str, Any]) -> None:
    required = {
        "artifact_schema_version",
        "protocol_id",
        "protocol_fingerprint",
        "protocol_snapshot",
        "experiment_id",
        "generated_at",
        "run_kind",
        "benchmark_claim",
        "git",
        "dataset",
        "category",
        "seed",
        "environment",
        "weight",
        "calibration",
        "thresholds",
        "predictions",
        "metric_implementation",
        "category_metrics",
        "aggregation",
        "resources",
        "warnings",
        "failures",
        "status",
    }
    missing = required - set(artifact)
    if missing:
        raise ArtifactError(
            f"Benchmark artifact is missing: {', '.join(sorted(missing))}"
        )
    if (
        artifact["protocol_id"] != PROTOCOL_ID
        or artifact["run_kind"] != "protocol_authorized_public_benchmark"
        or artifact["benchmark_claim"] is not True
    ):
        raise ArtifactError("Benchmark artifact has an invalid protocol identity")
    document = {"protocol": artifact["protocol_snapshot"]}
    try:
        validate_protocol_snapshot(artifact["protocol_snapshot"])
    except ProtocolError as exc:
        raise ArtifactError(
            "Benchmark artifact protocol snapshot is not frozen"
        ) from exc
    if artifact["protocol_fingerprint"] != protocol_fingerprint(document):
        raise ArtifactError("Benchmark artifact protocol fingerprint does not match")
    if artifact["category"] not in OFFICIAL_CATEGORIES:
        raise ArtifactError("Benchmark artifact category is outside the protocol")
    if artifact["seed"] not in PROTOCOL_SEEDS:
        raise ArtifactError("Benchmark artifact seed is outside the protocol")
    if (
        not isinstance(artifact["git"], dict)
        or artifact["git"].get("dirty") is not False
    ):
        raise ArtifactError("Benchmark artifacts require a clean Git state")
    if (
        not isinstance(artifact["dataset"], dict)
        or artifact["dataset"].get("status") != "passed"
    ):
        raise ArtifactError("Benchmark artifacts require a passing dataset audit")
    weight = artifact["weight"]
    if not isinstance(weight, dict) or len(str(weight.get("sha256", ""))) != 64:
        raise ArtifactError("Benchmark artifacts require verified weight identity")
    if not isinstance(artifact["predictions"], list):
        raise ArtifactError("Benchmark artifact predictions must be a list")
    for prediction in artifact["predictions"]:
        if not isinstance(prediction, dict):
            raise ArtifactError("Each benchmark prediction must be a mapping")
        _artifact_relative_path(prediction.get("sample_id"), "Prediction sample_id")
        score = prediction.get("anomaly_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ArtifactError("Prediction anomaly_score must be finite")
        anomaly_map = prediction.get("anomaly_map")
        if not isinstance(anomaly_map, dict):
            raise ArtifactError("Benchmark prediction requires an anomaly map")
        _artifact_relative_path(anomaly_map.get("path"), "Prediction anomaly_map.path")
        if len(str(anomaly_map.get("sha256", ""))) != 64:
            raise ArtifactError("Benchmark anomaly map requires SHA-256 identity")


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Validate and create an artifact without overwriting prior run evidence."""

    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise ArtifactError(f"Refusing to overwrite artifact: {path}") from exc
