from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from visionguard.artifacts import sha256_file
from visionguard.efficientad_benchmark import (
    EfficientAdBenchmarkError,
    _atomic_torch_save,
    _canonical_json_sha256,
    _checkpoint_relative,
    _load_checkpoint,
    _new_stream_state,
    _next_stream_index,
    _restore_rng_state,
    _rng_state,
    _validate_checkpoint_identity,
    _validate_manifest,
    _verify_completed_cell,
    aggregate_benchmark,
)
from visionguard.efficientad_protocol import (
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
)
from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS


def checkpoint_identity() -> dict[str, object]:
    return {
        "protocol_id": "efficientad-mvtecad2-v1",
        "protocol_fingerprint": "a" * 64,
        "benchmark_git_commit": "b" * 40,
        "dataset_audit_sha256": "c" * 64,
        "teacher_weight_sha256": "d" * 64,
        "teacher_archive_sha256": "e" * 64,
        "imagenette_archive_sha256": "f" * 64,
        "environment_sha256": "1" * 64,
        "category": "can",
        "seed": 42,
    }


def manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "protocol_id": "efficientad-mvtecad2-v1",
        "protocol_fingerprint": "a" * 64,
        "benchmark_git_commit": "b" * 40,
        "dataset_audit_sha256": "c" * 64,
        "teacher_weight_sha256": "d" * 64,
        "teacher_archive_sha256": "e" * 64,
        "imagenette_archive_sha256": "f" * 64,
        "environment_sha256": "1" * 64,
        "cells": {
            f"{category}:{seed}": {
                "category": category,
                "seed": seed,
                "status": "pending",
                "attempts": [],
                "interruption_history": [],
                "failure_history": [],
            }
            for category in OFFICIAL_CATEGORIES
            for seed in PROTOCOL_SEEDS
        },
    }


def expected_manifest_identity() -> dict[str, object]:
    value = manifest()
    return {
        key: value[key]
        for key in (
            "protocol_id",
            "protocol_fingerprint",
            "benchmark_git_commit",
            "dataset_audit_sha256",
            "teacher_weight_sha256",
            "teacher_archive_sha256",
            "imagenette_archive_sha256",
            "environment_sha256",
        )
    }


def test_stream_position_resumes_exactly(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    uninterrupted = _new_stream_state(7, 42)
    expected = [_next_stream_index(uninterrupted, 7) for _ in range(25)]

    interrupted = _new_stream_state(7, 42)
    first = [_next_stream_index(interrupted, 7) for _ in range(11)]
    state_path = tmp_path / "stream-state.pt"
    serialized = torch.save(interrupted, state_path)
    del serialized
    restored = torch.load(state_path, weights_only=False)
    second = [_next_stream_index(restored, 7) for _ in range(14)]

    assert first + second == expected


def test_rng_state_resumes_python_numpy_torch_and_cuda() -> None:
    import numpy as np

    torch = pytest.importorskip("torch")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    state = _rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.random()
    np.random.rand()
    torch.rand(1)
    _restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert actual == expected


def test_checkpoint_atomic_roundtrip_and_hash_validation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "latest.pt"
    payload = {
        "identity": checkpoint_identity(),
        "step": 1000,
        "tensor": torch.arange(3),
    }
    digest = _atomic_torch_save(path, payload)

    loaded = _load_checkpoint(path, digest)

    assert loaded["step"] == 1000
    assert torch.equal(loaded["tensor"], payload["tensor"])
    assert not path.with_suffix(".pt.tmp").exists()


def test_missing_corrupt_and_hash_mismatched_checkpoints_fail_closed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    missing = tmp_path / "missing.pt"
    with pytest.raises(EfficientAdBenchmarkError, match="missing"):
        _load_checkpoint(missing, "0" * 64)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    import hashlib

    digest = hashlib.sha256(corrupt.read_bytes()).hexdigest()
    with pytest.raises(EfficientAdBenchmarkError, match="corrupt"):
        _load_checkpoint(corrupt, digest)
    with pytest.raises(EfficientAdBenchmarkError, match="SHA-256"):
        _load_checkpoint(corrupt, "0" * 64)


def test_checkpoint_identity_and_step_drift_fail_closed() -> None:
    identity = checkpoint_identity()
    checkpoint = {"identity": identity, "step": 1000}
    _validate_checkpoint_identity(checkpoint, identity)

    changed = dict(identity)
    changed["benchmark_git_commit"] = "0" * 40
    with pytest.raises(EfficientAdBenchmarkError, match="identity"):
        _validate_checkpoint_identity(checkpoint, changed)
    with pytest.raises(EfficientAdBenchmarkError, match="step"):
        _validate_checkpoint_identity({"identity": identity, "step": 70001}, identity)


@pytest.mark.parametrize(
    "field",
    [
        "protocol_fingerprint",
        "benchmark_git_commit",
        "dataset_audit_sha256",
        "teacher_weight_sha256",
        "imagenette_archive_sha256",
        "environment_sha256",
    ],
)
def test_resume_manifest_identity_drift_fails_closed(field: str) -> None:
    value = manifest()
    expected = expected_manifest_identity()
    expected[field] = "0" * len(str(expected[field]))

    with pytest.raises(EfficientAdBenchmarkError, match="identity drift"):
        _validate_manifest(value, expected)


def test_manifest_requires_exact_complete_matrix() -> None:
    value = manifest()
    _validate_manifest(value, expected_manifest_identity())
    del value["cells"]["can:42"]  # type: ignore[index]

    with pytest.raises(EfficientAdBenchmarkError, match="matrix"):
        _validate_manifest(value, expected_manifest_identity())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "fabric"),
        ("seed", 7),
        ("status", "mystery"),
        ("attempts", {}),
        ("interruption_history", {}),
        ("failure_history", {}),
    ],
)
def test_manifest_cell_integrity_fails_closed(field: str, value: object) -> None:
    document = manifest()
    document["cells"]["can:42"][field] = value  # type: ignore[index]

    with pytest.raises(EfficientAdBenchmarkError, match="cell integrity"):
        _validate_manifest(document, expected_manifest_identity())


def test_partially_written_manifest_is_not_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": 2,', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        json.loads(path.read_text(encoding="utf-8"))


def test_aggregation_refuses_incomplete_matrix(tmp_path: Path) -> None:
    with pytest.raises(EfficientAdBenchmarkError, match="incomplete"):
        aggregate_benchmark(manifest(), tmp_path)


def test_checkpoint_path_is_attempt_scoped() -> None:
    assert _checkpoint_relative("can", 42, 3).as_posix() == (
        "runs/can/seed-42/attempt-3/latest-checkpoint.pt"
    )


def test_completed_cell_skip_requires_verified_artifact_maps_and_checkpoint(
    tmp_path: Path,
) -> None:
    document = load_efficientad_protocol(
        Path("configs/protocols/efficientad-mvtecad2-v1.yaml")
    )
    protocol = document["protocol"]
    run_dir = tmp_path / "runs/can/seed-42/attempt-1"
    map_dir = run_dir / "maps"
    map_dir.mkdir(parents=True)
    checkpoint = run_dir / "latest-checkpoint.pt"
    continuous = map_dir / "continuous.tiff"
    binary = map_dir / "binary.png"
    checkpoint.write_bytes(b"checkpoint")
    continuous.write_bytes(b"continuous")
    binary.write_bytes(b"binary")
    current_manifest = manifest()
    current_manifest.update(
        {
            "protocol_id": protocol["id"],
            "protocol_fingerprint": efficientad_protocol_fingerprint(document),
            "environment_sha256": _canonical_json_sha256({}),
        }
    )
    artifact = {
        "artifact_schema_version": 3,
        "protocol_id": protocol["id"],
        "protocol_fingerprint": current_manifest["protocol_fingerprint"],
        "protocol_snapshot": protocol,
        "experiment_id": "phase3b-can-seed-42",
        "run_kind": "phase3b_protocol_authorized_public_benchmark",
        "benchmark_claim": True,
        "evaluation_split": "test_public",
        "git": {
            "commit": current_manifest["benchmark_git_commit"],
            "dirty": False,
        },
        "dataset": {
            "status": "passed",
            "sha256": current_manifest["dataset_audit_sha256"],
        },
        "category": "can",
        "seed": 42,
        "implementation": {"name": "anomalib", "version": "2.6.0"},
        "model": protocol["model"],
        "training": {"final_optimization_step": 70000},
        "preprocessing": protocol["preprocessing"],
        "auxiliary_data": {
            "archive_sha256": current_manifest["imagenette_archive_sha256"]
        },
        "environment": {},
        "environment_sha256": current_manifest["environment_sha256"],
        "reproducibility": protocol["reproducibility"],
        "weights": [
            {
                "sha256": current_manifest["teacher_weight_sha256"],
                "archive_sha256": current_manifest["teacher_archive_sha256"],
            }
        ],
        "calibration": {"normal_only": True, "split": "validation"},
        "thresholds": {"image": 0.1, "pixel": 0.2},
        "model_state": {
            "checkpoint_path": checkpoint.relative_to(tmp_path).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
        "predictions": [
            {
                "sample_id": "can/test_public/good/000.png",
                "anomaly_score": 0.1,
                "anomaly_map": {
                    "path": "maps/continuous.tiff",
                    "sha256": sha256_file(continuous),
                    "thresholded_path": "maps/binary.png",
                    "thresholded_sha256": sha256_file(binary),
                },
            }
        ],
        "metrics": [],
        "category_metrics": {
            name: {"status": "defined", "value": 0.5}
            for name in ("au_pro_0.05", "pixel_f1", "image_f1", "image_auroc")
        },
        "resources": {},
        "warnings": [],
        "failures": [],
        "status": "completed",
    }
    artifact_path = run_dir / "benchmark-artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    entry = current_manifest["cells"]["can:42"]  # type: ignore[index]
    entry.update(
        {
            "status": "completed",
            "artifact_path": artifact_path.relative_to(tmp_path).as_posix(),
            "artifact_sha256": sha256_file(artifact_path),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    )

    _verify_completed_cell(entry, tmp_path, current_manifest)
    continuous.write_bytes(b"tampered")
    with pytest.raises(EfficientAdBenchmarkError, match="anomaly-map identity"):
        _verify_completed_cell(entry, tmp_path, current_manifest)
