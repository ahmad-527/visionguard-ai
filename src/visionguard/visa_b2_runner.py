"""Sequential B2 readiness orchestration. All real entry points remain CLOSED."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from visionguard.triage import ALLOWED_SEEDS
from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import sha256_file
from visionguard.visa_b2_admission import admit_fixture, fixture_origin, make_fixture
from visionguard.visa_b2_storage import json_once, publish_cell, replay_cell
from visionguard.visa_evaluator import aggregate_matrix, require
from visionguard.visa_evaluator_synthetic import fixture


def plan() -> dict:
    return {
        "models": 72,
        "paired_cells": [[c, s] for c in CATEGORIES for s in ALLOWED_SEEDS],
        "expected_real_images": 2162,
        "expected_real_model_image_calls": 12972,
        "real_execution_authorized": False,
        "gpu_workloads_at_once": 1,
    }


def run_synthetic(output: Path, permit, *, stop_after: int | None = None) -> dict:
    """Use minted artificial inputs only; never load a trained model implicitly.

    Valid complete cells are SHA-replayed rather than recomputed. Corrupt/partial
    cells do not count. Interrupted attempts are immutable and get a new attempt.
    An unclean process death leaves the exclusive lock for human process review;
    no automatic stale-lock deletion or hardware/publication retry exists.
    """
    frames = admit_fixture(permit)
    source_origin = fixture_origin(permit)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "writer.lock"
    handle = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    owner = f"{os.getpid()}:{uuid4().hex}".encode()
    os.write(handle, owner)
    os.fsync(handle)
    os.close(handle)
    results, skipped, attempts = [], 0, []
    try:
        for category in CATEGORIES:
            for seed in ALLOWED_SEEDS:
                samples, pc, ea, specs = fixture(category, seed)
                expected = {
                    s.sample_id: (s, p, e)
                    for s, p, e in zip(samples, pc, ea, strict=True)
                }
                require(
                    set(expected) == {f.sample.sample_id for f in frames[category]},
                    "Input/prediction alignment differs",
                )
                for frame in frames[category]:
                    reference = expected[frame.sample.sample_id][0]
                    require(
                        frame.sample.label == reference.label
                        and np.array_equal(frame.sample.mask, reference.mask),
                        "Known-answer truth differs",
                    )
                cell = output / category / f"seed-{seed}"
                prior = (
                    sorted(
                        cell.glob("attempt-*"), key=lambda p: int(p.name.split("-")[-1])
                    )
                    if cell.exists()
                    else []
                )
                completed = bool(prior and (prior[-1] / "complete.json").exists())
                parent = (
                    prior[-2]
                    if completed and len(prior) > 1
                    else (prior[-1] if prior and not completed else None)
                )
                origin = source_origin | {
                    "parent_failure_sha256": sha256_file(parent / "failure.json")
                    if parent
                    else None
                }
                if completed:
                    complete = prior[-1] / "complete.json"
                    # Expected SHA from independently published receipt, not itself.
                    receipt = json.loads((prior[-1] / "validated.json").read_text())
                    require(
                        sha256_file(complete) == receipt["sha256"],
                        "Completion binding changed",
                    )
                    result = replay_cell(prior[-1], receipt["sha256"], specs, origin)
                    skipped += 1
                else:
                    if prior:
                        require(
                            (prior[-1] / "failure.json").exists(),
                            "Unclassified interruption requires human review",
                        )
                        failure = json.loads((prior[-1] / "failure.json").read_text())
                        require(
                            failure["status"] == "interrupted",
                            "Failed filesystem/hardware attempt requires human review",
                        )
                    attempt = cell / f"attempt-{len(prior) + 1}"
                    records = (
                        (f.sample, *expected[f.sample.sample_id][1:])
                        for f in frames[category]
                    )
                    receipt = publish_cell(attempt, records, specs, origin)
                    json_once(attempt / "validated.json", receipt)
                    result = replay_cell(attempt, receipt["sha256"], specs, origin)
                    attempts.append(
                        {
                            "category": category,
                            "seed": seed,
                            "attempt": len(prior) + 1,
                            "parent_failure_preserved": bool(prior),
                        }
                    )
                results.append(result)
                if stop_after is not None and len(results) == stop_after:
                    raise InterruptedError(
                        "Declared between-cell synthetic interruption"
                    )
        aggregate = aggregate_matrix(results)
        report = {
            "evidence_class": "synthetic_engineering_only",
            "paired_cells": len(results),
            "cells": results,
            "aggregate": aggregate,
            "validated_skip_completed": skipped,
            "new_attempts": attempts,
            "real_test_access": False,
            "real_execution_authorized": False,
        }
        # A fresh immutable summary for each invocation; no replace-live metadata.
        json_once(output / f"matrix-{uuid4().hex}.json", report)
        return report
    finally:
        # Only this normally unwinding owner's lock. Process death does not run it.
        require(
            lock.read_bytes() == owner,
            "Writer lock ownership changed; human review required",
        )
        lock.unlink()


def evaluate_real(*, repository: Path, test_root: object = None, **kwargs):
    """Fail before any operation on the supplied real test-root object."""
    from visionguard.visa_b2_gate import deny_real_execution

    deny_real_execution(repository=repository, test_root=test_root, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(plan(), sort_keys=True))
    elif args.synthetic:
        require(args.output is not None, "Explicit synthetic output required")
        from visionguard.visa_b2_contract import verify_freeze

        verify_freeze(args.repository)
        permit = make_fixture(args.output / "inputs")
        print(
            json.dumps(
                run_synthetic(args.output / "matrix", permit),
                allow_nan=False,
                sort_keys=True,
            )
        )
    else:
        evaluate_real(repository=args.repository)


if __name__ == "__main__":
    main()
