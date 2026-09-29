"""Publish compact evidence only after complete independent 72-cell validation."""

import argparse
from datetime import datetime
from pathlib import Path

from visionguard.visa_acquire import atomic_json, sha256_file
from visionguard.visa_matrix import file_bytes
from visionguard.visa_matrix_contract import load_contract, order
from visionguard.visa_matrix_validation import load, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repository, root = args.repository.resolve(), args.root.resolve()
    contract, fingerprint = load_contract(repository)
    matrix = load(root / "matrix-manifest.json")
    freeze = load(root / "development-freeze.json")
    require(
        matrix["status"] == "completed" and matrix["counts"]["completed_total"] == 72,
        "Partial development matrix cannot be published as complete",
    )
    require(
        freeze["execution_fingerprint"] == fingerprint
        and freeze["matrix_manifest_sha256"]
        == sha256_file(root / "matrix-manifest.json"),
        "Final freeze identity mismatch",
    )
    cells = {}
    for key in order():
        row = matrix["cells"][key]
        path = root / row["validation"]["path"]
        require(
            sha256_file(path) == row["validation"]["sha256"],
            "Validation receipt differs",
        )
        evidence = load(path)
        require(
            evidence == freeze["cells"][key] and evidence["status"] == "validated",
            "Cell freeze differs",
        )
        model, category, seed = key.split(":")
        independent = load(root / f"{model}-development-summary.json")
        require(
            sha256_file(root / f"{model}-development-summary.json")
            == matrix["completed_model_validations"][model],
            "Model validation differs",
        )
        require(
            sha256_file(
                root / "independent-validation" / f"{model}-{category}-{seed}.json"
            )
            == independent["validation_sha256"][key],
            "Independent receipt differs",
        )
        cells[key] = {
            "final_model_sha256": evidence["files"]["final_model"]["sha256"],
            "canonical_model_sha256": evidence["canonical_model_sha256"],
            "calibration_sha256": evidence["files"]["calibration"]["sha256"],
            "image_threshold": evidence["calibration"]["image"],
            "pixel_threshold": evidence["calibration"]["pixel"],
            "normalization": evidence.get("normalization"),
            "fit_inventory_sha256": evidence["fit_inventory_sha256"],
            "calibration_inventory_sha256": evidence["calibration_inventory_sha256"],
            "fit_count": evidence["fit_count"],
            "calibration_count": evidence["calibration_count"],
            "final_optimization_step": evidence.get("final_optimization_step"),
            "active_seconds": evidence.get(
                "active_fit_seconds", evidence.get("active_training_seconds")
            ),
            "active_time_limit": evidence["active_time_limit"],
            "attempt_history": [
                {
                    "attempt": a["attempt"],
                    "status": a["status"],
                    "history": a["history"],
                }
                for a in evidence["attempts"]
            ],
            "validation_sha256": row["validation"]["sha256"],
            "wall_history": row["history"],
            "retained_cell_bytes": row["retained_cell_bytes"],
            "compaction": row.get("compaction"),
        }
    output = repository / "reports/phase4d-a-visa-development-matrix"
    compact = {
        "execution_contract": contract,
        "execution_fingerprint": fingerprint,
        "implementation_commit": matrix["implementation_commit"],
        "full_local_freeze_sha256": sha256_file(root / "development-freeze.json"),
        "matrix_manifest_sha256": sha256_file(root / "matrix-manifest.json"),
        "cells": cells,
        "validated_cells": 72,
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
        "phase4d_b_started": False,
        "human_access_history": "UNKNOWN",
        "independent_reservation": "NOT ESTABLISHED",
    }
    atomic_json(output / "development-freeze.json", compact)
    summary = {
        "status": "72_development_cells_validated",
        "counts": matrix["counts"],
        "freeze_sha256": sha256_file(output / "development-freeze.json"),
        "retained_output_bytes": file_bytes(root),
        "elapsed_wall_seconds": (
            datetime.fromisoformat(matrix["finished_at"])
            - datetime.fromisoformat(matrix["started_at"])
        ).total_seconds(),
        "active_recorded_seconds": sum(c["active_seconds"] for c in cells.values()),
        "active_time_caveat": (
            "model-specific recorded fit/training timers, "
            "not end-to-end active time or speed comparison"
        ),
        "failed_attempts": sum(
            a["status"] == "failed"
            for c in cells.values()
            for a in c["attempt_history"]
        ),
        "interrupted_attempts": sum(
            a["status"] == "interrupted"
            for c in cells.values()
            for a in c["attempt_history"]
        ),
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
    }
    atomic_json(output / "development-summary.json", summary)


if __name__ == "__main__":
    main()
