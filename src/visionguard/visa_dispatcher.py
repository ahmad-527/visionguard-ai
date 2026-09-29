"""Single-cell, normal-only VisA development dispatcher; no final-test command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from visionguard.embedding_journal import EmbeddingJournal, publish_journal
from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_development_worker import file_record, validate_receipt
from visionguard.visa_execution import ExecutionState, cells, exclusive_writer
from visionguard.visa_guard import verified_context


def execution_scope(model: str, engineering: bool) -> dict:
    return {
        "purpose": "ENGINEERING_ONLY"
        if engineering
        else "NORMAL_ONLY_FULL_DEVELOPMENT",
        "fit_limit": (8 if model == "patchcore" else 4) if engineering else None,
        "calibration_limit": 19 if engineering and model == "efficientad" else None,
        "steps": 2 if engineering else 70000,
    }


class DevelopmentDispatcher:
    """Parent survives worker termination and binds only validated durable state."""

    def __init__(self, root: Path, identity: dict):
        self.root = root.resolve()
        self.identity = identity
        self.state = ExecutionState(self.root, identity)
        self.state.initialize()

    def prepare(
        self, category: str, seed: int, *, resume: bool
    ) -> tuple[Path, dict | None]:
        if (
            category not in CATEGORIES
            or type(seed) is not int
            or seed not in (42, 123, 2026)
        ):
            raise VisaIntegrityError("Invalid frozen cell")
        if self.identity["scope"]["purpose"] == "ENGINEERING_ONLY" and (
            category,
            seed,
        ) != ("candle", 42):
            raise VisaIntegrityError("Declared engineering fixture is candle seed 42")
        prior = None
        cell = self.state.read()["cells"][f"{category}:{seed}"]
        if self.state.validate_completed(category, seed):
            raise VisaIntegrityError("Completed cell already validated; no overwrite")
        if resume:
            if not cell["attempts"] or cell["attempts"][-1]["status"] not in (
                "failed",
                "interrupted",
            ):
                raise VisaIntegrityError(
                    "Resume requires a terminal interrupted/failed attempt"
                )
            if any(
                event.get("reason") == "hardware_failure_stop"
                for event in cell["attempts"][-1]["history"]
            ):
                raise VisaIntegrityError(
                    "Hardware failure requires human review, not resume"
                )
            path = self.state.validated_checkpoint(category, seed)
            prior = file_record(self.root, path)
            validate_receipt(
                self.root, prior, {**self.identity, "category": category, "seed": seed}
            )
        _, directory = self.state.begin(category, seed)
        self.state.update(category, seed, stage="fit")
        return directory, prior

    def finalize(
        self, category: str, seed: int, directory: Path, returncode: int
    ) -> dict:
        try:
            return self._finalize(category, seed, directory, returncode)
        except (ValueError, KeyError, OSError, TypeError):
            current = self.state.read()["cells"][f"{category}:{seed}"]["attempts"][-1]
            if current["status"] in ("created", "active"):
                self.state.update(
                    category,
                    seed,
                    stage=current["stage"],
                    status="failed",
                    reason="receipt_integrity_failure",
                )
            raise

    def _finalize(
        self, category: str, seed: int, directory: Path, returncode: int
    ) -> dict:
        identity = {**self.identity, "category": category, "seed": seed}
        path = directory / "worker-state.json"
        if not path.exists():
            self.state.update(
                category,
                seed,
                stage="fit",
                status="failed",
                reason="worker_preflight_failed",
            )
            raise VisaIntegrityError("Worker failed before durable state")
        document = json.loads(path.read_text())
        if document["identity"] != identity:
            raise VisaIntegrityError("Worker origin mismatch")
        # Recover only atomically completed metadata after a hard worker exit.
        journal_path = directory / "embeddings.json"
        if journal_path.exists() and "embedding_journal" not in document["files"]:
            journal_data = json.loads(journal_path.read_text())
            journal = EmbeddingJournal(
                self.root, directory, identity, journal_data["sample_ids"]
            )
            journal.validate(journal_data)
            document["files"]["embedding_journal"] = file_record(
                self.root, journal_path
            )
        training_path = directory / "training-state.json"
        if training_path.exists() and "training_checkpoint" not in document["files"]:
            training = json.loads(training_path.read_text())
            if training["identity"] != identity:
                raise VisaIntegrityError("Training origin mismatch")
            record = training["entry"].get("latest_valid_checkpoint")
            if record:
                checkpoint = self.root / record["path"]
                if sha256_file(checkpoint) != record["sha256"]:
                    raise VisaIntegrityError("Training checkpoint corruption")
                document["files"]["training_checkpoint"] = file_record(
                    self.root, checkpoint
                )
        publish_journal(path, document)
        receipt = file_record(self.root, path)
        validate_receipt(self.root, receipt, identity)
        stage = document["stage"]
        if stage == "created":
            stage = "fit"
        if returncode == 0 and document["status"] == "development_complete":
            if (
                not {"final_model", "calibration", "normalized_model", "fit_model"}
                <= document["files"].keys()
            ):
                raise VisaIntegrityError("Incomplete development output contract")
            artifacts = {
                name: self.root / record["path"]
                for name, record in document["files"].items()
            }
            artifacts["receipt"] = path
            self.state.update(
                category,
                seed,
                stage="development_complete",
                status="development_complete",
                checkpoint=self.root / document["files"]["final_model"]["path"],
                artifacts=artifacts,
            )
        else:
            log = (
                (directory / "worker.log").read_text(errors="replace")
                if (directory / "worker.log").exists()
                else ""
            )
            hardware = any(
                term in log
                for term in ("OutOfMemoryError", "CUDA error", "device-side assert")
            )
            reason = (
                "hardware_failure_stop"
                if hardware
                else (
                    "intentional_worker_exit" if returncode == 75 else "worker_failed"
                )
            )
            self.state.update(
                category,
                seed,
                stage=stage,
                status="interrupted" if returncode == 75 else "failed",
                checkpoint=path,
                reason=reason,
            )
        return self.state.read()["cells"][f"{category}:{seed}"]["attempts"][-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--model", choices=("patchcore", "efficientad"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--identity-only", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--run-cell", nargs=2, metavar=("CATEGORY", "SEED"))
    action.add_argument("--resume-cell", nargs=2, metavar=("CATEGORY", "SEED"))
    parser.add_argument("--development", type=Path)
    for name in ("weight", "teacher", "imagenette-root", "imagenette-archive"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--confirm-full-development-fit", action="store_true")
    parser.add_argument(
        "--stop-at",
        choices=(
            "embedding:4",
            "before_coreset",
            "after_fit",
            "step:1",
            "after_normalization",
            "after_calibration",
        ),
    )
    args = parser.parse_args(argv)
    repository, root = args.repository.resolve(), (args.output / args.model).resolve()
    verified_context(repository, args.model)
    if args.identity_only:
        from visionguard.visa_engineering import execution_identity

        _, identity = execution_identity(
            repository,
            args.model,
            extra_sources=(
                "src/visionguard/visa_dispatcher.py",
                "src/visionguard/visa_development_worker.py",
                "configs/engineering/visa-dispatcher-acceptance-v1.yaml",
            ),
        )
        print(json.dumps(identity))
        return 0
    if args.plan:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "cells": cells(),
                    "final_test_lock": "closed",
                    "training_started": False,
                }
            )
        )
        return 0
    if args.status:
        if not (root / "manifest.json").exists():
            print(
                json.dumps(
                    {
                        "status": "not_initialized",
                        "expected_cells": 36,
                        "final_test_lock": "closed",
                    }
                )
            )
        else:
            saved = json.loads((root / "manifest.json").read_text())
            state = ExecutionState(root, saved["identity"])
            print(
                json.dumps(
                    {**state.matrix_status(), "scope": saved["identity"]["scope"]}
                )
            )
        return 0
    if not args.engineering_smoke and not args.confirm_full_development_fit:
        raise VisaIntegrityError(
            "Full normal-only development requires explicit confirmation"
        )
    if args.development is None:
        raise VisaIntegrityError("Development-only root required")
    if args.stop_at and not args.engineering_smoke:
        raise VisaIntegrityError("Declared fault injection is engineering-only")
    # Isolate CUDA environment inspection so the orchestration parent never holds
    # a second CUDA context/VRAM allocation while the fitting worker is running.
    with exclusive_writer(repository / "outputs/visa-development-gpu-lease"):
        checked = subprocess.run(
            [
                sys.executable,
                "-m",
                "visionguard.visa_dispatcher",
                "--repository",
                str(repository),
                "--model",
                args.model,
                "--output",
                str(args.output),
                "--identity-only",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    identity = json.loads(checked.stdout)
    identity["scope"] = execution_scope(args.model, args.engineering_smoke)
    if args.model == "patchcore":
        if args.weight is None:
            raise VisaIntegrityError("Frozen PatchCore weight required")
        identity["weight_sha256"] = sha256_file(args.weight)
    else:
        if any(
            getattr(args, name) is None
            for name in ("teacher", "imagenette_root", "imagenette_archive")
        ):
            raise VisaIntegrityError("Frozen teacher and ImageNette assets required")
        identity["teacher_sha256"] = sha256_file(args.teacher)
        identity["imagenette_sha256"] = sha256_file(args.imagenette_archive)
        science = verified_context(repository, args.model)["scientific"]
        if (
            identity["teacher_sha256"] != science["model"]["teacher_weight_sha256"]
            or identity["imagenette_sha256"]
            != science["auxiliary_data"]["archive_sha256"]
        ):
            raise VisaIntegrityError("Frozen EfficientAD asset identity mismatch")
    category, seed_text = args.run_cell or args.resume_cell
    seed = int(seed_text)
    if category not in CATEGORIES or seed not in (42, 123, 2026):
        raise VisaIntegrityError("Invalid frozen category/seed")
    with exclusive_writer(repository / "outputs/visa-development-gpu-lease"):
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise VisaIntegrityError(
                "At least 20 GiB development disk headroom required"
            )
        dispatcher = DevelopmentDispatcher(root, identity)
        if dispatcher.state.validate_completed(category, seed):
            print(
                json.dumps(
                    {"status": "validated_skip_completed", "final_test_lock": "closed"}
                )
            )
            return 0
        directory, resume = dispatcher.prepare(
            category, seed, resume=args.resume_cell is not None
        )
        request = {
            "repository": str(repository),
            "root": str(root),
            "directory": str(directory),
            "development": str(args.development.resolve()),
            "model": args.model,
            "category": category,
            "seed": seed,
            "identity": identity,
            "resume": resume,
            "stop_at": args.stop_at,
        }
        for name in ("weight", "teacher", "imagenette_root", "imagenette_archive"):
            value = getattr(args, name)
            request[name] = None if value is None else str(value.resolve())
        atomic_json(directory / "request.json", request)
        with (directory / "worker.log").open("x", encoding="utf-8") as log:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "visionguard.visa_development_worker",
                    "--request",
                    str(directory / "request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            atomic_json(
                directory / "worker-process.json",
                {
                    "parent_pid": os.getpid(),
                    "worker_pid": worker.pid,
                    "started_at": datetime.now(UTC).isoformat(),
                    "request_sha256": sha256_file(directory / "request.json"),
                    "purpose": identity["scope"]["purpose"],
                },
            )
            try:
                code = worker.wait()
            except KeyboardInterrupt:
                worker.terminate()
                worker.wait()
                code = 75
        result = dispatcher.finalize(category, seed, directory, code)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "attempt": result["attempt"],
                    "final_test_lock": "closed",
                }
            )
        )
        return 0 if code in (0, 75) else 2


if __name__ == "__main__":
    raise SystemExit(main())
