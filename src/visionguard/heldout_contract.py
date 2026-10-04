"""Controlled activation freeze layered on immutable B1/B2 scientific inputs."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_b2_contract import context as readiness_context
from visionguard.visa_b2_contract import verify_freeze as verify_readiness
from visionguard.visa_evaluator import require
from visionguard.visa_protocol import canonical_fingerprint

PROTOCOL = "configs/protocols/visa-controlled-activation-v1.json"
FREEZE = "reports/phase4d-b2-controlled-activation/implementation-freeze.json"


def context(repository: Path) -> dict:
    old = readiness_context(repository)
    ready = verify_readiness(repository)
    protocol = json.loads((repository / PROTOCOL).read_text())
    require(
        protocol["b2_readiness_fingerprint"] == ready["fingerprint"],
        "Readiness identity drift",
    )
    for key in (
        "b1_fingerprint",
        "development_freeze_sha256",
        "scope",
        "output_relative",
    ):
        require(protocol[key] == old["protocol"][key], "Scientific contract drift")
    audit = json.loads(
        (repository / "reports/phase4c-visa-readiness/audit-summary.json").read_text()
    )
    return old | {"activation": protocol, "audit": audit}


def build_freeze(repository: Path) -> dict:
    ctx = context(repository)
    paths = [
        p.relative_to(repository).as_posix()
        for pattern in (
            "src/visionguard/heldout_*.py",
            "tests/test_heldout_*.py",
            "scripts/heldout_activation_*.py",
        )
        for p in repository.glob(pattern)
    ]
    paths += [PROTOCOL, ".github/workflows/controlled-activation.yml"]
    document = {
        "schema_version": 1,
        "contract": ctx["activation"],
        "source_sha256": {p: sha256_file(repository / p) for p in sorted(paths)},
        "environment": ctx["published"]["execution_contract"]["environment"],
        "models": {k: vars(v) for k, v in ctx["specs"].items()},
        "evidence_sha256": {
            p.relative_to(repository).as_posix(): sha256_file(p)
            for p in sorted(
                (repository / "reports/phase4d-b2-controlled-activation").glob("*.json")
            )
            if p.name not in ("implementation-freeze.json", "ci-verification.json")
        },
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository: Path) -> dict:
    saved = json.loads((repository / FREEZE).read_text())
    require(saved == build_freeze(repository), "Activation/source fingerprint drift")
    return saved
