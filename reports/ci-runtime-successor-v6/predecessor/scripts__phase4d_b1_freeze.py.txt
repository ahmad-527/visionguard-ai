"""Freeze/reproduce B1 engineering source and methodology, never access test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionguard.visa_acquire import atomic_json, sha256_file
from visionguard.visa_evaluator_artifacts import verify_published
from visionguard.visa_evaluator_protocol import (
    FREEZE_PATH,
    PROTOCOL_PATH,
    verify_readiness_freeze,
)
from visionguard.visa_protocol import canonical_fingerprint


def build(repository: Path) -> dict:
    published = verify_published(repository)
    paths = set(published["execution_contract"]["source_hashes"])
    paths.update(
        p.relative_to(repository).as_posix()
        for p in (repository / "src/visionguard").glob("visa_evaluator*.py")
    )
    paths.update(
        p.relative_to(repository).as_posix()
        for p in (repository / "tests").glob("test_visa_evaluator*.py")
    )
    paths.update(
        {
            "scripts/phase4d_b1_freeze.py",
            ".github/workflows/ci.yml",
            "src/visionguard/analysis_metrics.py",
            "src/visionguard/triage_protocol.py",
            "src/visionguard/visa_protocol.py",
            "src/visionguard/visa_guard.py",
            "configs/schemas/visa-evaluator-report-v1.json",
            "scripts/phase4d_b1_evidence.py",
            "tests/test_visa_matrix.py",
        }
    )
    native_receipt = (
        "reports/phase4d-b1-visa-evaluator-readiness/native-backend-restoration.json"
    )
    paths.add(native_receipt)
    document = {
        "schema_version": 1,
        "protocol": json.loads(
            (repository / PROTOCOL_PATH).read_text(encoding="utf-8")
        ),
        "source_sha256": {p: sha256_file(repository / p) for p in sorted(paths)},
        "native_environment": published["execution_contract"]["environment"],
        "native_source_sha256": json.loads((repository / native_receipt).read_text())[
            "native_source_sha256"
        ],
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--write", action="store_true", help="Create only; cannot overwrite a freeze"
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    snapshot = build(repository)
    if args.write:
        path = repository / FREEZE_PATH
        if path.exists():
            raise ValueError("Immutable B1 freeze already exists")
        atomic_json(path, snapshot)
    else:
        existing = verify_readiness_freeze(repository)
        if existing != snapshot:
            raise ValueError("B1 freeze reproduction differs")
    print(snapshot["fingerprint"])


if __name__ == "__main__":
    main()
