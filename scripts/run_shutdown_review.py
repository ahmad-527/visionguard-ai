"""Record allowlisted isolated engineering checks, preserving every attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REVIEW_SOURCES = (
    "src/visionguard/heldout_cleanup.py",
    "src/visionguard/heldout_runner.py",
    "src/visionguard/accounting_successor.py",
    "src/visionguard/cleanup_successor.py",
    "src/visionguard/heldout_contract.py",
    "tests/test_heldout_cleanup.py",
    "tests/test_heldout_execution.py",
    "tests/test_heldout_pause.py",
    "tests/test_cleanup_successor.py",
    "tests/test_accounting_successor.py",
    "tests/test_operational_successor.py",
    "tests/test_heldout_accounting.py",
    "scripts/cleanup_successor_freeze.py",
    "scripts/probe_heldout_cleanup.py",
    "scripts/run_shutdown_review.py",
)


def source_hashes(repository: Path) -> dict[str, str]:
    """Read only the explicit engineering review source set."""
    return {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name in REVIEW_SOURCES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "native",
            "regression",
            "acceptance",
            "focused",
            "receipt-fault",
            "lint",
        ),
    )
    parser.add_argument("--native-python", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    report_root = repository / "reports/shutdown-failure-review"
    run = report_root / (
        args.case + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.case == "native":
        if args.native_python is None:
            parser.error("native case requires the existing native runtime")
        command = [
            str(args.native_python),
            "-B",
            "scripts/probe_heldout_cleanup.py",
            "--output",
            str(run / "native-observations.json"),
        ]
    elif args.case in {"regression", "acceptance", "focused", "receipt-fault"}:
        targets = [
            "tests/test_heldout_cleanup.py",
            "tests/test_heldout_execution.py",
            "tests/test_heldout_pause.py",
        ]
        if args.case == "acceptance":
            targets = [
                "tests/test_heldout_accounting.py",
                "tests/test_accounting_successor.py",
                "tests/test_operational_successor.py",
                "tests/test_cleanup_successor.py",
            ]
        elif args.case == "focused":
            targets = [
                targets[0],
                "tests/test_heldout_pause.py::test_independent_cpu_watcher_safe_exit_handshake",
            ]
        elif args.case == "receipt-fault":
            targets = [
                "tests/test_heldout_cleanup.py::test_cleanup_receipt_refusal_does_not_mask_primary_failure",
            ]
        command = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            *targets,
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp=" + str(run / "manufactured-temp"),
            "--junitxml=" + str(run / "pytest-results.xml"),
        ]
    else:
        command = [sys.executable, "-B", "-m", "ruff", "check", *REVIEW_SOURCES]
    before = source_hashes(repository)
    started = datetime.now(UTC).isoformat()
    with (
        (run / "stdout.log").open("xb") as stdout,
        (run / "stderr.log").open("xb") as stderr,
    ):
        result = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=3600,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    record = {
        "evidence_class": "ISOLATED_NON_HELDOUT_ENGINEERING_CHECK",
        "case": args.case,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "working_directory": str(repository),
        "exit_code": result.returncode,
        "original_run_access": False,
        "previous_attempts_removed": False,
        "review_source_sha256_before": before,
        "review_source_sha256_after": source_hashes(repository),
    }
    record["review_sources_unchanged_during_check"] = (
        record["review_source_sha256_before"] == record["review_source_sha256_after"]
    )
    with (run / "command-result.json").open("x", encoding="utf-8") as target:
        json.dump(record, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({"run_directory": str(run), **record}, indent=2))
    print((run / "stdout.log").read_text(encoding="utf-8", errors="replace"))
    print((run / "stderr.log").read_text(encoding="utf-8", errors="replace"))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
