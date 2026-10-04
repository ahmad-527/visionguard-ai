"""Durable artificial36-pair acceptance, fault preservation and exact B1 replay."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from visionguard.heldout_admission import make_artificial
from visionguard.heldout_runner import run_artificial
from visionguard.heldout_storage import json_once
from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import (
    aggregate_matrix,
    evaluate_synthetic_cell,
    require,
)
from visionguard.visa_evaluator_synthetic import fixture


def numeric(document):
    if isinstance(document, dict):
        return {
            k: numeric(v)
            for k, v in document.items()
            if k not in ("sample_id", "membership_sha256", "evidence_class")
        }
    if isinstance(document, list):
        return [numeric(v) for v in document]
    return document


def reference(category, seed):
    """Supply B1 exactly the new admission's canonical path order, not row index."""
    samples, pc, ea, specs = fixture(category, seed)
    order = sorted(
        range(len(samples)),
        key=lambda i: (
            f"{category}/Data/Images/{'Anomaly' if samples[i].label else 'Normal'}/"
            f"artificial-{i}.png"
        ),
    )
    return evaluate_synthetic_cell(
        [samples[i] for i in order],
        [pc[i] for i in order],
        [ea[i] for i in order],
        specs,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    permit = make_artificial(root / "inputs")
    matrix = root / "matrix"
    start = time.perf_counter()
    try:
        run_artificial(matrix, permit, interrupt=("candle", 42, "efficientad"))
    except InterruptedError:
        pass
    else:
        raise ValueError("Declared interruption not observed")
    failed = matrix / "candle/seed-42/efficientad/attempt-1/failure.json"
    failure_sha = sha256_file(failed)
    result = run_artificial(matrix, permit)
    replay = run_artificial(matrix, permit)
    expected = [reference(c, s) for c in CATEGORIES for s in (42, 123, 2026)]
    require(
        numeric(result["cells"]) == numeric(expected)
        and numeric(result["aggregate"]) == numeric(aggregate_matrix(expected)),
        "Exact B1 known-answer equivalence failure",
    )
    require(
        result["cells"] == replay["cells"]
        and replay["new_inference_calls"] == 0
        and replay["validated_skip_stages"] == 72,
        "Restart equivalence failure",
    )
    require(sha256_file(failed) == failure_sha, "Interrupted attempt changed")
    calls = json.loads(failed.read_text())["completed_inference_calls"]
    report = {
        "evidence_class": "artificial_real_id_acceptance_only",
        "paired_cells": 36,
        "models": 72,
        "artificial_unique_images": 48,
        "effective_scope_calls": 288,
        "interrupted_partial_calls": calls,
        "new_resume_calls": result["new_inference_calls"],
        "completed_stage_skips_on_resume": result["validated_skip_stages"],
        "completed_stage_skips_on_full_replay": replay["validated_skip_stages"],
        "new_full_replay_calls": 0,
        "bitwise_numeric_B1_equivalence": True,
        "all_seed_category_aggregates_exact": True,
        "interrupted_failure_sha256": failure_sha,
        "failure_preserved": True,
        "triage_counts_per_pair": result["cells"][0]["triage"]["decision_counts"],
        "wall_seconds": time.perf_counter() - start,
        "retained_matrix_bytes": sum(
            p.stat().st_size for p in matrix.rglob("*") if p.is_file()
        ),
        "observed_D_free_bytes": shutil.disk_usage(root).free,
        "real_test_asset_access": False,
        "real_test_performance_evaluations": 0,
        "final_test_lock": "CLOSED",
    }
    json_once(root / "acceptance.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
