"""Actual controlled route plus full artificial acceptance; never authorizes itself."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from visionguard.heldout_admission import admit, admit_artificial, make_artificial
from visionguard.heldout_authorization import authorize, check_permission
from visionguard.heldout_backend import load_backend
from visionguard.heldout_contract import context, verify_freeze
from visionguard.heldout_metrics import Prediction, aggregate_matrix
from visionguard.heldout_paths import checked
from visionguard.heldout_resources import ACTIVE_BUDGET, Budget, monitor
from visionguard.heldout_stages import (
    fail,
    paired_result,
    publish_pair,
    publish_stage,
    verify_stage,
)
from visionguard.heldout_storage import json_once
from visionguard.triage import ALLOWED_SEEDS
from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_evaluator_storage import canonical_bytes
from visionguard.visa_evaluator_synthetic import fixture


def plan():
    return {
        "pairs": [[c, s] for c in CATEGORIES for s in ALLOWED_SEEDS],
        "models": 72,
        "expected_real_calls": 12972,
        "test_lock": "CLOSED absent separate reviewed human approval",
    }


class ArtificialBackend:
    def __init__(self, spec):
        self.spec = spec

    def predict(self, frame):
        frame.validate(self.spec)
        i = int(Path(frame.sample_id).stem.removeprefix("artificial-"))
        p = fixture(self.spec.category, self.spec.seed)[
            1 if self.spec.model == "patchcore" else 2
        ][i]
        return Prediction(frame.sample_id, self.spec, p.score, p.restored_map.copy())


def _prior(parent):
    if not parent.exists():
        return []
    checked(parent)
    paths = sorted(parent.glob("attempt-*"), key=lambda p: int(p.name.split("-")[-1]))
    require(
        [p.name for p in paths] == [f"attempt-{i}" for i in range(1, len(paths) + 1)],
        "Attempt chronology gap",
    )
    for path in paths:
        checked(path)
    return paths


def _resume_parent(paths):
    if not paths:
        return None
    require(
        (paths[-1] / "failure.json").is_file(),
        "Unclassified partial attempt: human review required",
    )
    failure = json.loads((paths[-1] / "failure.json").read_text())
    require(failure["status"] == "interrupted", "Failed attempt: no automatic retry")
    return sha256_file(paths[-1] / "failure.json")


def run_matrix(
    inputs,
    output,
    specs,
    origin,
    backend_factory,
    *,
    real=False,
    interrupt=None,
    watcher=None,
    provenance_check=lambda: None,
    permission=None,
):
    """One model at a time. Complete stages independently verified before skip.

    Model-stage maps are retained once; pair publications only reference stages.
    A failed/unclassified writer leaves its lock for human process review.
    """
    if real:
        check_permission(permission)
    require(inputs.artificial is (not real), "Execution/input evidence class mismatch")
    checked(output, missing=True)
    output.mkdir(parents=True, exist_ok=True)
    budget = Budget(output)
    token = ACTIVE_BUDGET.set(budget)
    lock = output / "writer.lock"
    checked(lock, missing=True)
    owner = {
        "pid": os.getpid(),
        "nonce": uuid4().hex,
        "started": datetime.now(UTC).isoformat(),
        "origin": origin,
    }
    lock_created = False
    safe_exit = False
    invocation = output / f"invocation-{owner['nonce']}"
    invocation.mkdir()
    results = []
    calls = 0
    skipped_stages = 0
    try:
        json_once(lock, owner)
        lock_created = True
        if real:
            require(watcher is not None, "Independent owner watcher required")
            watcher.start(output, owner)

        def check():
            if watcher:
                watcher.check()
            row = monitor(output, budget, real)
            if real:
                now = time.monotonic()
                if now - check.last >= 60:
                    provenance_check()
                    budget.refresh()
                    json_once(invocation / f"telemetry-{uuid4().hex}.json", row)
                    check.last = now

        check.last = 0
        for category in CATEGORIES:
            for seed in ALLOWED_SEEDS:
                pair_specs = tuple(
                    specs[f"{m}:{category}:{seed}"]
                    for m in ("patchcore", "efficientad")
                )
                pair_root = output / category / f"seed-{seed}"
                stages = []
                for spec in pair_specs:
                    parent = pair_root / spec.model
                    paths = _prior(parent)
                    prior_complete = bool(
                        paths and (paths[-1] / "complete.json").exists()
                    )
                    if prior_complete:
                        require(
                            not (paths[-1] / "failure.json").exists(),
                            "Completion with failure requires human review",
                        )
                        receipt = json.loads(
                            checked(paths[-1] / "validated.json").read_text()
                        )
                        old_origin = json.loads(
                            (paths[-1] / "origin.json").read_text()
                        )["source"]
                        expected = origin | {
                            "parent_failure_sha256": sha256_file(
                                paths[-2] / "failure.json"
                            )
                            if len(paths) > 1
                            else None
                        }
                        require(old_origin == expected, "Stage resume lineage mismatch")
                        verify_stage(paths[-1], receipt["sha256"], spec, expected)
                        skipped_stages += 1
                        stage_path = paths[-1]
                    else:
                        stage_origin = origin | {
                            "parent_failure_sha256": _resume_parent(paths)
                        }
                        stage_path = parent / f"attempt-{len(paths) + 1}"
                        backend = None
                        started = time.perf_counter()
                        check()
                        try:
                            if real:
                                import torch

                                torch.cuda.reset_peak_memory_stats()
                            backend = backend_factory(spec)
                            digest = backend.state_digest() if real else None
                            receipt = publish_stage(
                                stage_path,
                                inputs.frames(category),
                                spec,
                                stage_origin,
                                backend,
                                interrupt_after=1
                                if interrupt == (category, seed, spec.model)
                                else None,
                                check=check,
                            )
                            calls += len(inputs.rows[category])
                            if real and backend.state_digest() != digest:
                                error = ValueError(
                                    "Native state mutated during inference"
                                )
                                fail(stage_path, error, len(inputs.rows[category]))
                                raise error
                        finally:
                            del backend
                            gc.collect()
                            if real:
                                import torch

                                torch.cuda.empty_cache()
                        json_once(
                            invocation / f"model-{category}-{seed}-{spec.model}.json",
                            {
                                "seconds": time.perf_counter() - started,
                                "spec": vars(spec),
                                "observations": monitor(output, budget, real),
                            },
                        )
                    stages.append((stage_path, receipt["sha256"]))
                # Stage lineage differs by model, so paired reducer checks each
                # independently then supplies common immutable source origin.
                stage_origins = [
                    json.loads((p / "origin.json").read_text())["source"]
                    for p, _ in stages
                ]
                pair_parent = pair_root / "pair"
                previous = _prior(pair_parent)
                paired_origin = origin | {"model_stage_origins": stage_origins}
                # paired_result accepts per-stage origins (see staged reducer).
                pair_stages = [
                    (p, sha, o)
                    for (p, sha), o in zip(stages, stage_origins, strict=True)
                ]
                started = time.perf_counter()
                if previous and (previous[-1] / "complete.json").exists():
                    require(
                        not (previous[-1] / "failure.json").exists(),
                        "Failed pair requires human review",
                    )
                    receipt = json.loads(
                        checked(previous[-1] / "validated.json").read_text()
                    )
                    require(
                        sha256_file(previous[-1] / "complete.json")
                        == receipt["sha256"],
                        "Pair completion corrupt",
                    )
                    complete = json.loads((previous[-1] / "complete.json").read_text())
                    require(
                        complete["origin"] == paired_origin
                        and complete["specs"] == [vars(s) for s in pair_specs]
                        and complete["stages"]
                        == [
                            {"path": str(p.relative_to(pair_root)), "sha256": sha}
                            for p, sha in stages
                        ],
                        "Pair stage lineage changed",
                    )
                    result = paired_result(pair_stages, pair_specs, paired_origin)
                    require(
                        canonical_bytes(result)
                        == (previous[-1] / "result.json").read_bytes()
                        and sha256_file(previous[-1] / "result.json")
                        == complete["result"]["sha256"],
                        "Pair replay changed",
                    )
                else:
                    _resume_parent(previous)
                    result, receipt = publish_pair(
                        pair_parent / f"attempt-{len(previous) + 1}",
                        pair_stages,
                        pair_specs,
                        paired_origin,
                    )
                results.append(result)
                json_once(
                    invocation / f"pair-{category}-{seed}.json",
                    {
                        "publication_replay_seconds": time.perf_counter() - started,
                        "receipt": receipt,
                    },
                )
        aggregate = aggregate_matrix(results)
        budget.refresh()
        require(len(results) == 36, "Incomplete matrix")
        expected_calls = sum(len(r) for r in inputs.rows.values()) * 6
        if real:
            require(expected_calls == 12972, "Scope changed")
        report = {
            "schema_version": 1,
            "evidence_class": inputs.origin["evidence_class"],
            "origin": origin,
            "paired_cells": 36,
            "model_stages": 72,
            "new_inference_calls": calls,
            "total_scope_calls": expected_calls,
            "validated_skip_stages": skipped_stages,
            "aggregate": aggregate,
            "cells": results,
            "retained_bytes_before_summary": budget.used,
            "real_test_access": real,
        }
        json_once(invocation / "complete.json", report)
        if not (output / "complete.json").exists():
            json_once(
                output / "complete.json",
                {
                    "origin": origin,
                    "invocation": invocation.name,
                    "report_sha256": sha256_file(invocation / "complete.json"),
                },
            )
        safe_exit = True
        return report
    except BaseException as exc:
        fail(invocation, exc)
        safe_exit = isinstance(exc, InterruptedError)
        raise
    finally:
        if watcher:
            watcher.finish(safe_exit)
        if lock_created and safe_exit:
            require(json.loads(lock.read_text()) == owner, "Writer ownership changed")
            lock.unlink()
        ACTIVE_BUDGET.reset(token)


def run_artificial(parent, permit, *, interrupt=None):
    inputs = admit_artificial(permit)
    specs = {
        f"{s.model}:{c}:{seed}": s
        for c in CATEGORIES
        for seed in ALLOWED_SEEDS
        for s in fixture(c, seed)[3]
    }
    return run_matrix(
        inputs, parent, specs, inputs.origin, ArtificialBackend, interrupt=interrupt
    )


def evaluate_real(
    repository: Path, approval=None, *, resume_claim=None, test_root=None
):
    """No caller dataset path. Denials do not touch even a hostile root sentinel."""
    permission = authorize(repository, approval, resume_claim=resume_claim)
    require(test_root is None, "Caller test-root override forbidden")
    ctx = context(repository)
    frozen = verify_freeze(repository)
    inputs = admit(repository, permission)
    pixels = sum(
        r["image_identity"]["width"] * r["image_identity"]["height"]
        for rows in inputs.rows.values()
        for row in rows
        for r in [inputs.identities[row.image]]
    )
    # Conservative uncompressed maps + binary/truth PNG upper allowance. All
    # old failed attempts are charged separately, never removed to make room.
    projection = pixels * 6 * 4 + 12972 * 16384 + 36 * 2**21
    budget = Budget(permission.run_root)
    budget.before(projection)
    from visionguard.heldout_watcher import Watcher

    origin = inputs.origin | {
        "authorization_id": permission.authorization_id,
        "claim_sha256": permission.claim_sha256,
        "activation_fingerprint": frozen["fingerprint"],
    }
    return run_matrix(
        inputs,
        permission.run_root,
        ctx["specs"],
        origin,
        lambda spec: load_backend(
            repository, repository / "outputs/phase4d-a-visa-development-matrix", spec
        ),
        real=True,
        watcher=Watcher(),
        permission=permission,
        provenance_check=lambda: (
            check_permission(permission),
            verify_freeze(repository),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--artificial", type=Path)
    # Deliberately no execution shortcut; separate explicit signed document.
    parser.add_argument("--human-approval", type=Path)
    parser.add_argument("--resume-claim-sha256")
    args = parser.parse_args()
    if args.plan:
        result = plan()
    elif args.artificial:
        verify_freeze(args.repository)
        args.artificial = args.artificial.absolute()
        permit = make_artificial(args.artificial / "inputs")
        result = run_artificial(args.artificial / "matrix", permit)
    else:
        approval = (
            json.loads(args.human_approval.read_text()) if args.human_approval else None
        )
        result = evaluate_real(
            args.repository, approval, resume_claim=args.resume_claim_sha256
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
