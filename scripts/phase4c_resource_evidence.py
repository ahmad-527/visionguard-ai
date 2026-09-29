"""Capacity evidence only: existing benchmark resources and VisA integrity shapes.

No prediction values, test metrics, qualitative images, or model calls are used.
"""

import argparse
import json
import statistics
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository
    reports = root / "reports/phase4c-visa-readiness"
    audit = json.loads((reports / "audit-summary.json").read_text())
    if sha256_file(args.inventory) != audit["inventory_sha256"]:
        raise VisaIntegrityError("Capacity inventory identity mismatch")
    inventory = json.loads(args.inventory.read_text())["records"]
    pixels = sum(
        r["image_identity"]["width"] * r["image_identity"]["height"]
        for r in inventory
        if r["split"] == "test"
    )
    historical = {}
    for model, directory in (
        ("patchcore", "phase2c-public-benchmark"),
        ("efficientad", "phase3b-efficientad-public-benchmark"),
    ):
        base = root / "outputs" / directory
        manifest = json.loads((base / "benchmark-manifest.json").read_text())
        observations = []
        for key, cell in sorted(manifest["cells"].items()):
            path = base / cell["artifact_path"]
            if sha256_file(path) != cell["artifact_sha256"]:
                raise VisaIntegrityError("Historical resource artifact mismatch")
            artifact = json.loads(path.read_text())
            resources = artifact["resources"]
            row = {"cell": key, "artifact_sha256": cell["artifact_sha256"]}
            if model == "patchcore":
                row["seconds"] = resources["wall_time_seconds"]["value"]
                row["cuda_peak_allocated_bytes"] = resources["cuda_memory"][
                    "peak_allocated_bytes"
                ]
            else:
                row["seconds"] = resources["active_training_seconds"]
                checkpoint = base / artifact["model_state"]["checkpoint_path"]
                # Stored file size is a capacity observation, not checkpoint validation.
                row["checkpoint_size_bytes"] = checkpoint.stat().st_size
                row["cuda_peak_allocated_bytes"] = resources[
                    "cuda_peak_allocated_bytes"
                ]
            observations.append(row)
        seconds = [r["seconds"] for r in observations]
        historical[model] = {
            "measurement": "whole_cell_wall_time"
            if model == "patchcore"
            else "active_training_time_not_wall_time",
            "observed_cells": len(observations),
            "seconds_min_median_max": [
                min(seconds),
                statistics.median(seconds),
                max(seconds),
            ],
            "naive_36_cell_hours_min_median_max": [
                n * 36 / 3600
                for n in (min(seconds), statistics.median(seconds), max(seconds))
            ],
            "observations": observations,
        }
    fit_counts = [r["fit"] for r in audit["allocation_counts"].values()]
    # Frozen 256x256 input, layer2 32x32 spatial grid, 512+1024 float32 channels.
    embedding_bytes = max(fit_counts) * 32 * 32 * 1536 * 4
    atomic_json(
        reports / "capacity-evidence.json",
        {
            "schema_version": 1,
            "purpose": "ENGINEERING_CAPACITY_ESTIMATE_NOT_PERFORMANCE",
            "historical": historical,
            "visa_integrity": {
                "inventory_sha256": audit["inventory_sha256"],
                "test_image_count": sum(r["split"] == "test" for r in inventory),
                "test_total_pixels": pixels,
                "fit_count_min_max": [min(fit_counts), max(fit_counts)],
            },
            "extrapolated": {
                "model_cells": 72,
                "map_uncompressed_payload_bytes_two_models_three_seeds": pixels * 18,
                "map_formula": (
                    "total_pixels * 2_models * 3_seeds * (2_float16+1_binary)"
                ),
                "patchcore_max_cell_embedding_bytes": embedding_bytes,
                "patchcore_stack_peak_lower_bound_bytes": embedding_bytes * 2,
                "patchcore_coreset_payload_bytes_all_cells": sum(
                    int(n * 1024 * 0.01) * 1536 * 4 * 3 for n in fit_counts
                ),
                "efficientad_checkpoint_bytes_36_cells": max(
                    r["checkpoint_size_bytes"]
                    for r in historical["efficientad"]["observations"]
                )
                * 36,
            },
            "unknown": [
                "VisA full-cell runtime",
                "thermal sustained throughput",
                "PNG compression and metric scratch space",
                "PatchCore full-category GPU fit feasibility",
            ],
            "test_performance_evaluated": False,
            "final_test_lock": "closed",
        },
    )


if __name__ == "__main__":
    main()
