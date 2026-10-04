"""Publish compact synthetic-only B1 evidence and calculated capacity scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from visionguard.visa_acquire import atomic_json
from visionguard.visa_evaluator_protocol import verify_readiness_freeze
from visionguard.visa_evaluator_resources import estimate
from visionguard.visa_evaluator_storage import canonical_bytes
from visionguard.visa_evaluator_synthetic import synthetic_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repository.resolve()
    snapshot = verify_readiness_freeze(repo)
    full = synthetic_report()
    audit = json.loads(
        (repo / "reports/phase4c-visa-readiness/audit-summary.json").read_text()
    )
    counts = {
        c: n["test_normal"] + n["test_anomaly"]
        for c, n in audit["category_counts"].items()
    }
    resources = estimate(counts)
    resources["observed_local_free_bytes"] = shutil.disk_usage(repo).free
    resources["future_capacity_gate_satisfied_now"] = (
        resources["observed_local_free_bytes"] >= resources["required_free_bytes"]
    )
    evidence = {
        "schema_version": 1,
        "evidence_class": "synthetic_engineering_only",
        "evaluator_implementation_fingerprint": snapshot["fingerprint"],
        "synthetic_paired_cells": len(full["cells"]),
        "full_synthetic_matrix_canonical_sha256": hashlib.sha256(
            canonical_bytes(full)
        ).hexdigest(),
        "known_answer_example": full["cells"][0],
        "aggregate": full["aggregate"],
        "real_test_asset_access": False,
        "real_test_performance_evaluations": 0,
        "final_test_lock": "closed",
        "phase4d_b2_started": False,
    }
    target = repo / "reports/phase4d-b1-visa-evaluator-readiness"
    for name, document in (
        ("synthetic-metric-evidence.json", evidence),
        ("resource-budget.json", resources),
    ):
        path = target / name
        if path.exists():
            raise ValueError("Immutable evidence already exists")
        atomic_json(path, document)
    print(
        json.dumps(
            {"fingerprint": snapshot["fingerprint"], "resources": resources},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
