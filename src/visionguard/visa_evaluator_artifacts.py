"""Read-only verification of development artifacts; never reads dataset assets."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import sha256_file
from visionguard.visa_guard import MEMBERSHIP, verified_context
from visionguard.visa_matrix_contract import order
from visionguard.visa_matrix_validation import load, require, validate_cell

DEVELOPMENT_FREEZE = "d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061"
IMPLEMENTATION = "cca5e47ec8c3f68adfa6c90e17a17770b48049d0"
EXECUTION = "31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3"
MERGE = "ab6f47402e11ae55c835ba7323c681e995ef4082"


def verify_published(repository: Path) -> dict:
    """Verify canonical committed evidence and exact 72-cell membership."""
    path = (
        repository / "reports/phase4d-a-visa-development-matrix/development-freeze.json"
    )
    require(sha256_file(path) == DEVELOPMENT_FREEZE, "Published freeze differs")
    freeze = load(path)
    require(
        freeze["implementation_commit"] == IMPLEMENTATION
        and freeze["execution_fingerprint"] == EXECUTION
        and freeze["validated_cells"] == 72
        and set(freeze["cells"]) == set(order())
        and freeze["final_test_lock"] == "closed"
        and freeze["test_performance_evaluated"] is False
        and freeze["phase4d_b_started"] is False
        and freeze["human_access_history"] == "UNKNOWN"
        and freeze["independent_reservation"] == "NOT ESTABLISHED",
        "Published provenance/boundary mismatch",
    )
    for model in ("patchcore", "efficientad"):
        verified_context(repository, model)
    for key, cell in freeze["cells"].items():
        model, category, seed = key.split(":")
        require(category in CATEGORIES and int(seed) in (42, 123, 2026), "Wrong cell")
        counts = freeze["execution_contract"]["counts"][category]
        require(
            cell["fit_count"] == counts["fit"]
            and cell["calibration_count"] == counts["calibration"]
            and cell["attempt_history"][-1]["status"] == "development_complete",
            "Missing development completion/membership",
        )
        if model == "efficientad":
            require(cell["final_optimization_step"] == 70000, "Wrong final step")
    return freeze


def verify_local(repository: Path, root: Path) -> dict:
    """Hash/CPU semantic checks only on declared completed development files.

    Does not enumerate any dataset root, instantiate a trained model, infer,
    recalibrate, repair state, or overwrite a prior validation receipt. The
    native frozen validator checks final/fit/normalized checkpoints and the
    final training state, including normalization, calibration and attempts.
    """
    repository, root = repository.resolve(), root.resolve()
    freeze = verify_published(repository)
    matrix_path = root / "matrix-manifest.json"
    require(
        sha256_file(matrix_path) == freeze["matrix_manifest_sha256"],
        "Local matrix differs",
    )
    require(
        sha256_file(root / "development-freeze.json")
        == freeze["full_local_freeze_sha256"],
        "Local freeze differs",
    )
    matrix = load(matrix_path)
    membership_path = (
        repository / "reports/phase4c-visa-readiness/development-membership.json"
    )
    require(sha256_file(membership_path) == MEMBERSHIP, "Membership differs")
    membership = load(membership_path)
    # Check only bound, preexisting scientific files. New B1 modules are not
    # retroactively included in the frozen Phase4D-A implementation identity.
    for relative, digest in freeze["execution_contract"]["source_hashes"].items():
        require(sha256_file(repository / relative) == digest, "Frozen source differs")
    cells = {}
    for key in order():
        model, category, seed = key.split(":")
        identity = matrix["identities"][model]
        require(
            identity["implementation_commit"] == IMPLEMENTATION
            and identity["membership_sha256"] == MEMBERSHIP
            and identity["protocol_fingerprint"]
            == freeze["execution_contract"]["protocols"][model],
            "Local model provenance differs",
        )
        result = validate_cell(
            root / model, model, category, int(seed), identity, membership
        )
        published = freeze["cells"][key]
        require(
            result["canonical_model_sha256"] == published["canonical_model_sha256"]
            and result["files"]["final_model"]["sha256"]
            == published["final_model_sha256"]
            and result["files"]["calibration"]["sha256"]
            == published["calibration_sha256"]
            and result["calibration"]["image"] == published["image_threshold"]
            and result["calibration"]["pixel"] == published["pixel_threshold"]
            and result.get("normalization") == published["normalization"]
            and result["fit_inventory_sha256"] == published["fit_inventory_sha256"]
            and result["calibration_inventory_sha256"]
            == published["calibration_inventory_sha256"],
            "Local scientific artifacts disagree with published freeze",
        )
        cells[key] = {
            "status": "verified_read_only",
            "final_model_sha256": published["final_model_sha256"],
            "canonical_model_sha256": result["canonical_model_sha256"],
            "checkpoint_files_verified": len(result["files"]),
            "attempt_statuses": [a["status"] for a in result["attempts"]],
        }
    # A second metadata read detects any writer changing the completed matrix.
    require(
        sha256_file(matrix_path) == freeze["matrix_manifest_sha256"], "Concurrent drift"
    )
    return {
        "status": "72_development_cells_verified_read_only",
        "published_freeze_sha256": DEVELOPMENT_FREEZE,
        "implementation_commit": IMPLEMENTATION,
        "execution_fingerprint": EXECUTION,
        "cells": cells,
        "final_test_lock": "closed",
        "test_asset_access": False,
        "model_inference_performed": False,
        "test_performance_evaluated": False,
    }


def main() -> None:
    """Explicit development-output root only; no test or inference interface."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--development-artifacts", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_local(args.repository, args.development_artifacts), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
