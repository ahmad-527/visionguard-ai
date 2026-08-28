"""Compare two train/validation-only engineering artifacts without test data."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from visionguard.calibration import highest_order_statistic, per_image_pixel_maxima


class RepeatabilityError(ValueError):
    """Raised when repeatability artifacts are incomplete or incomparable."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepeatabilityError(f"Unable to read artifact {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("status") != "completed":
        raise RepeatabilityError("Repeatability requires two completed artifacts")
    if value.get("benchmark_claim") is not False:
        raise RepeatabilityError("Repeatability inputs must be non-benchmark artifacts")
    configuration = value.get("configuration", {})
    if configuration.get("dataset", {}).get("evaluation_split") is not None:
        raise RepeatabilityError(
            "Repeatability inputs must not use an evaluation split"
        )
    return value


def _map_maxima(path: Path, artifact: dict[str, Any]) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RepeatabilityError("Map comparison requires NumPy") from exc
    maxima: list[float] = []
    for prediction in artifact["predictions"]:
        map_path = path.parent / prediction["anomaly_map"]["path"]
        anomaly_map = np.load(map_path, allow_pickle=False)
        maxima.append(float(anomaly_map.max()))
    return per_image_pixel_maxima([[value] for value in maxima])


def compare_artifacts(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Return measured exact/tolerance comparisons for two identical runs."""

    first, second = _load(first_path), _load(second_path)
    for key in ("configuration", "dataset", "weights", "reproducibility"):
        if first.get(key) != second.get(key):
            raise RepeatabilityError(f"Artifacts differ in required input field {key}")
    first_predictions = first["predictions"]
    second_predictions = second["predictions"]
    first_ids = [prediction["sample_id"] for prediction in first_predictions]
    second_ids = [prediction["sample_id"] for prediction in second_predictions]
    if first_ids != second_ids:
        raise RepeatabilityError("Prediction sample order differs")
    first_scores = [
        float(prediction["anomaly_score"]) for prediction in first_predictions
    ]
    second_scores = [
        float(prediction["anomaly_score"]) for prediction in second_predictions
    ]
    score_differences = [
        abs(left - right)
        for left, right in zip(first_scores, second_scores, strict=True)
    ]
    first_pixel_maxima = _map_maxima(first_path, first)
    second_pixel_maxima = _map_maxima(second_path, second)
    image_first = highest_order_statistic(first_scores, minimum_samples=1)
    image_second = highest_order_statistic(second_scores, minimum_samples=1)
    pixel_first = highest_order_statistic(first_pixel_maxima, minimum_samples=1)
    pixel_second = highest_order_statistic(second_pixel_maxima, minimum_samples=1)
    first_hashes = [
        prediction["anomaly_map"]["sha256"] for prediction in first_predictions
    ]
    second_hashes = [
        prediction["anomaly_map"]["sha256"] for prediction in second_predictions
    ]
    memory_first = first.get("model_state", {}).get("memory_bank", {})
    memory_second = second.get("model_state", {}).get("memory_bank", {})
    return {
        "schema_version": 1,
        "run_kind": "train_validation_repeatability_non_benchmark",
        "benchmark_claim": False,
        "sample_count": len(first_scores),
        "memory_bank": {
            "exact": memory_first.get("sha256") == memory_second.get("sha256"),
            "first_sha256": memory_first.get("sha256"),
            "second_sha256": memory_second.get("sha256"),
            "shape_equal": memory_first.get("shape") == memory_second.get("shape"),
        },
        "image_scores": {
            "bitwise_equal": first_scores == second_scores,
            "max_absolute_difference": max(score_differences, default=0.0),
            "within_absolute_tolerance_1e-7": all(
                difference <= 1e-7 for difference in score_differences
            ),
        },
        "thresholds": {
            "image_exact": image_first.threshold == image_second.threshold,
            "image_absolute_difference": abs(
                image_first.threshold - image_second.threshold
            ),
            "pixel_exact": pixel_first.threshold == pixel_second.threshold,
            "pixel_absolute_difference": abs(
                pixel_first.threshold - pixel_second.threshold
            ),
        },
        "anomaly_maps": {
            "all_sha256_equal": first_hashes == second_hashes,
            "equal_count": sum(
                left == right
                for left, right in zip(first_hashes, second_hashes, strict=True)
            ),
            "total_count": len(first_hashes),
        },
        "materially_nondeterministic": not (
            memory_first.get("sha256") == memory_second.get("sha256")
            and all(difference <= 1e-7 for difference in score_differences)
            and math.isclose(
                pixel_first.threshold, pixel_second.threshold, abs_tol=1e-7
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two non-benchmark train/validation-only artifacts."
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_artifacts(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
