"""Controlled activation freeze layered on immutable B1/B2 scientific inputs."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_b2_contract import context as readiness_context
from visionguard.visa_b2_contract import verify_freeze as verify_readiness
from visionguard.visa_evaluator import require

PROTOCOL = "configs/protocols/visa-controlled-activation-v1.json"
FREEZE = "reports/phase4d-b2-controlled-activation/implementation-freeze.json"


def context(repository: Path) -> dict:
    from visionguard.ci_runtime_successor import historical_repository

    repository = historical_repository(repository)
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
    """Explicit CI/runtime v6 candidate; historical freezes remain immutable."""
    from visionguard.ci_runtime_successor import build_freeze as successor

    return successor(repository)


def verify_freeze(repository: Path) -> dict:
    """Require the current successor AND immutable historical/scientific bindings."""
    from visionguard.ci_runtime_successor import verify_freeze as successor

    return successor(repository)
