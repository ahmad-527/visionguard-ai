"""Durable VisA development state and a deliberately closed final-test gate.

This CLI initializes/plans state; it does not launch training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from visionguard.triage import ALLOWED_SEEDS
from visionguard.visa import CATEGORIES, safe_asset
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_protocol import load_visa_protocol, verify_audit


def reject_final_test(
    *,
    confirmed: bool = False,
    supplied_fingerprint: str | None = None,
    expected_fingerprint: str | None = None,
    supplied_audit_sha256: str | None = None,
    expected_audit_sha256: str | None = None,
) -> None:
    """No flag combination authorizes final test in Phase 4C.

    Validate proposed identity arguments for useful diagnostics, then remain
    closed. A later reviewed Phase 4D authorization implementation is required.
    """
    if confirmed is not True:
        raise VisaIntegrityError("Final-test lock CLOSED: explicit confirmation absent")
    if not expected_fingerprint or supplied_fingerprint != expected_fingerprint:
        raise VisaIntegrityError(
            "Final-test lock CLOSED: protocol fingerprint mismatch"
        )
    if not expected_audit_sha256 or supplied_audit_sha256 != expected_audit_sha256:
        raise VisaIntegrityError("Final-test lock CLOSED: dataset audit mismatch")
    raise VisaIntegrityError(
        "Final-test lock CLOSED: Phase 4D has no execution authorization"
    )


def cells() -> tuple[str, ...]:
    return tuple(
        f"{category}:{seed}" for category in CATEGORIES for seed in ALLOWED_SEEDS
    )


@contextmanager
def exclusive_writer(root: Path):
    """One writer; stale locks require inspection, never automatic deletion."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(datetime.now(UTC).isoformat())
    except FileExistsError as exc:
        raise VisaIntegrityError(
            "Execution writer already active or stale lock needs review"
        ) from exc
    try:
        yield
    finally:
        lock.unlink()


class ExecutionState:
    """Attempt metadata persists independently of model-specific training loops."""

    def __init__(self, root: Path, identity: dict) -> None:
        for field, length in (
            ("protocol_fingerprint", 64),
            ("audit_sha256", 64),
            ("membership_sha256", 64),
            ("implementation_commit", 40),
            ("source_fingerprint", 64),
        ):
            if not re.fullmatch(rf"[0-9a-f]{{{length}}}", str(identity.get(field, ""))):
                raise VisaIntegrityError(f"Missing or malformed provenance: {field}")
        if (
            identity.get("model") not in ("patchcore", "efficientad")
            or not isinstance(identity.get("environment"), dict)
            or not identity["environment"]
        ):
            raise VisaIntegrityError("Model and environment provenance required")
        self.root = root
        self.path = root / "manifest.json"
        self.identity = deepcopy(identity)

    def initialize(self) -> dict:
        with exclusive_writer(self.root):
            if self.path.exists():
                return self.read()
            document = {
                "schema_version": 1,
                "identity": self.identity,
                "matrix": list(cells()),
                "cells": {key: {"attempts": []} for key in cells()},
                "final_test_lock": "closed",
                "test_performance_evaluated": False,
            }
            atomic_json(self.path, document)
            return document

    def read(self) -> dict:
        document = json.loads(self.path.read_text())
        if document.get("identity") != self.identity or document.get("matrix") != list(
            cells()
        ):
            raise VisaIntegrityError("Resume identity or matrix mismatch")
        if (
            set(document["cells"]) != set(cells())
            or document.get("final_test_lock") != "closed"
            or document.get("test_performance_evaluated") is not False
        ):
            raise VisaIntegrityError("Execution manifest contract mismatch")
        return document

    def begin(self, category: str, seed: int) -> tuple[int, Path]:
        if (
            category not in CATEGORIES
            or type(seed) is not int
            or seed not in ALLOWED_SEEDS
        ):
            raise VisaIntegrityError("Invalid category/seed cell")
        key = f"{category}:{seed}"
        with exclusive_writer(self.root):
            document = self.read()
            attempts = document["cells"][key]["attempts"]
            if attempts and attempts[-1]["status"] not in ("failed", "interrupted"):
                raise VisaIntegrityError(
                    "Cell is active or already completed; validate before skipping"
                )
            index = len(attempts) + 1
            relative = f"runs/{category}/seed-{seed}/attempt-{index}"
            directory = self.root / relative
            directory.mkdir(parents=True, exist_ok=False)
            attempt = {
                "attempt": index,
                "directory": relative,
                "status": "created",
                "stage": "created",
                "history": [{"stage": "created", "at": datetime.now(UTC).isoformat()}],
                "checkpoint": None,
                "artifacts": {},
            }
            atomic_json(
                directory / "origin.json",
                {
                    "identity": self.identity,
                    "category": category,
                    "seed": seed,
                    "attempt": index,
                },
            )
            attempts.append(attempt)
            atomic_json(self.path, document)
            return index, directory

    def update(
        self,
        category: str,
        seed: int,
        *,
        stage: str,
        status: str = "active",
        checkpoint: Path | None = None,
        artifacts: dict[str, Path] | None = None,
        reason: str | None = None,
    ) -> None:
        if stage not in (
            "fit",
            "normalization",
            "calibration",
            "artifact_write",
            "development_complete",
        ):
            raise VisaIntegrityError(
                "Unsupported development stage; no test stage exists"
            )
        if status not in ("active", "failed", "interrupted", "development_complete"):
            raise VisaIntegrityError("Invalid attempt status")
        with exclusive_writer(self.root):
            document = self.read()
            attempt = document["cells"][f"{category}:{seed}"]["attempts"][-1]
            if attempt["status"] in ("failed", "interrupted", "development_complete"):
                raise VisaIntegrityError("Terminal attempts are immutable")
            if stage == "development_complete" and status != "development_complete":
                raise VisaIntegrityError("Completion stage and status must agree")
            if status in ("failed", "interrupted") and not reason:
                raise VisaIntegrityError(
                    "Failure/interruption reason must be preserved"
                )

            def bind(path: Path) -> dict:
                relative = path.resolve().relative_to(self.root.resolve()).as_posix()
                safe_asset(self.root, relative)
                if not relative.startswith(attempt["directory"] + "/"):
                    raise VisaIntegrityError(
                        "Artifact must belong to the active attempt"
                    )
                return {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }

            if checkpoint is not None:
                attempt["checkpoint"] = bind(checkpoint)
            for name, path in (artifacts or {}).items():
                attempt["artifacts"][name] = bind(path)
            if status == "development_complete" and (
                stage != "development_complete"
                or not attempt["checkpoint"]
                or "calibration" not in attempt["artifacts"]
            ):
                raise VisaIntegrityError(
                    "Complete development requires checkpoint and calibration evidence"
                )
            attempt.update(stage=stage, status=status)
            attempt["history"].append(
                {
                    "stage": stage,
                    "status": status,
                    "reason": reason,
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_json(self.path, document)

    def validated_checkpoint(self, category: str, seed: int) -> Path:
        attempt = self.read()["cells"][f"{category}:{seed}"]["attempts"][-1]
        record = attempt["checkpoint"]
        if not record:
            raise VisaIntegrityError("No durable checkpoint available")
        path = safe_asset(self.root, record["path"])
        if sha256_file(path) != record["sha256"]:
            raise VisaIntegrityError("Checkpoint corruption detected")
        return path

    def validate_completed(self, category: str, seed: int) -> bool:
        attempts = self.read()["cells"][f"{category}:{seed}"]["attempts"]
        if not attempts or attempts[-1]["status"] != "development_complete":
            return False
        self.validated_checkpoint(category, seed)
        for record in attempts[-1]["artifacts"].values():
            if sha256_file(safe_asset(self.root, record["path"])) != record["sha256"]:
                raise VisaIntegrityError("Completed artifact corruption detected")
        return True

    def matrix_status(self) -> dict:
        completed = sum(
            self.validate_completed(category, seed)
            for category in CATEGORIES
            for seed in ALLOWED_SEEDS
        )
        return {
            "expected_cells": 36,
            "development_complete_cells": completed,
            "development_matrix_complete": completed == 36,
            "confirmatory_evaluation_complete": False,
            "final_test_lock": "closed",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-fingerprint", required=True)
    parser.add_argument("--dataset-audit", required=True, type=Path)
    parser.add_argument("--dataset-audit-sha", required=True)
    parser.add_argument("--confirm-independent-test-evaluation", action="store_true")
    args = parser.parse_args(argv)
    if args.confirm_independent_test_evaluation:
        reject_final_test(
            confirmed=True,
            supplied_fingerprint=args.protocol_fingerprint,
            expected_fingerprint=args.protocol_fingerprint,
            supplied_audit_sha256=args.dataset_audit_sha,
            expected_audit_sha256=args.dataset_audit_sha,
        )
    protocol = load_visa_protocol(
        args.protocol, expected_fingerprint=args.protocol_fingerprint
    )
    verify_audit(args.dataset_audit, args.dataset_audit_sha)
    if protocol["protocol"]["dataset"]["audit_sha256"] != args.dataset_audit_sha:
        raise VisaIntegrityError("Protocol/audit binding mismatch")
    print(
        json.dumps(
            {
                "status": "development_plan_only",
                "cells": 36,
                "final_test_lock": "closed",
                "protocol_id": protocol["protocol"]["id"],
                "training_started": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
