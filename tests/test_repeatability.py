from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.repeatability import RepeatabilityError, compare_artifacts


def write_run(root: Path, *, score_delta: float = 0.0) -> Path:
    np = pytest.importorskip("numpy")
    root.mkdir()
    map_path = root / "anomaly-maps" / "sample.npy"
    map_path.parent.mkdir()
    with map_path.open("wb") as stream:
        np.save(stream, np.asarray([[0.1 + score_delta, 0.2]]), allow_pickle=False)
    import hashlib

    map_hash = hashlib.sha256(map_path.read_bytes()).hexdigest()
    artifact = {
        "status": "completed",
        "benchmark_claim": False,
        "configuration": {"dataset": {"evaluation_split": None}},
        "dataset": {"status": "passed"},
        "weights": [{"sha256": "a" * 64}],
        "reproducibility": {"seed": 42},
        "model_state": {"memory_bank": {"sha256": "b" * 64, "shape": [2, 2]}},
        "predictions": [
            {
                "sample_id": "validation/good/sample.png",
                "anomaly_score": 0.3 + score_delta,
                "anomaly_map": {"path": "anomaly-maps/sample.npy", "sha256": map_hash},
            }
        ],
    }
    artifact_path = root / "experiment-artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def test_identical_repeatability_artifacts_are_exact(tmp_path: Path) -> None:
    first = write_run(tmp_path / "first")
    second = write_run(tmp_path / "second")

    result = compare_artifacts(first, second)

    assert result["memory_bank"]["exact"] is True
    assert result["image_scores"]["bitwise_equal"] is True
    assert result["anomaly_maps"]["all_sha256_equal"] is True
    assert result["materially_nondeterministic"] is False


def test_repeatability_reports_numeric_difference(tmp_path: Path) -> None:
    first = write_run(tmp_path / "first")
    second = write_run(tmp_path / "second", score_delta=1e-4)

    result = compare_artifacts(first, second)

    assert result["image_scores"]["bitwise_equal"] is False
    assert result["materially_nondeterministic"] is True


def test_repeatability_rejects_evaluation_data(tmp_path: Path) -> None:
    first = write_run(tmp_path / "first")
    second = write_run(tmp_path / "second")
    value = json.loads(first.read_text(encoding="utf-8"))
    value["configuration"]["dataset"]["evaluation_split"] = "test_public"
    first.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RepeatabilityError, match="evaluation split"):
        compare_artifacts(first, second)
