"""Separate B2 authorization design. Readiness never grants real-test access."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from visionguard.visa_b2_contract import context, verify_freeze
from visionguard.visa_evaluator import EvaluationError, require
from visionguard.visa_protocol import canonical_fingerprint


def implementation_head(repository: Path) -> str:
    """Independent checkout identity, never supplied-as-expected by a caller."""
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, env=environment, text=True
    ).strip()


def expected_intent(repository: Path) -> dict:
    """Draft only. Actual checkout identity is not reviewed permission.

    The future execution gate must get the reviewed commit from a trusted human
    approval ledger AND compare actual Git HEAD. This readiness draft can never
    authorize a run, even if a human text/flag or matching SHA is supplied.
    """
    ctx = context(repository)
    freeze = verify_freeze(repository)
    return {
        "schema_version": 1,
        "reviewed_implementation_commit": implementation_head(repository),
        "b2_fingerprint": freeze["fingerprint"],
        "b1_fingerprint": ctx["protocol"]["b1_fingerprint"],
        "development_freeze_sha256": ctx["protocol"]["development_freeze_sha256"],
        "dataset_audit_sha256": ctx["protocol"]["dataset_audit_sha256"],
        "scope": ctx["protocol"]["scope"],
        "output_relative": ctx["protocol"]["output_relative"],
        "output_destination": str(
            repository.resolve() / ctx["protocol"]["output_relative"]
        ),
        "contract_sha256": canonical_fingerprint(ctx["protocol"]),
        "models": {k: asdict(v) for k, v in ctx["specs"].items()},
        "status": "DRAFT_NOT_AUTHORIZED",
    }


def _id_used(repository: Path, authorization_id: str) -> bool:
    """Independent design ledger location; no caller supplies the expected ledger.

    No readiness command creates or consumes an authorization record. A future
    reviewed gate must authenticate the human approval and claim its ID atomically.
    """
    return (
        repository
        / "outputs/phase4d-b2-authorization-used"
        / f"{authorization_id}.json"
    ).exists()


def validate_intent(
    repository: Path,
    candidate: dict,
    *,
    free_bytes: int,
) -> dict:
    """Pure design validation, NEVER permission or admission.

    Expected scientific identities and destination come independently from the
    canonical repository freeze and independently read checkout HEAD;
    real entrypoints never call this as an authority or accept it as approval.
    """
    expected = expected_intent(repository)
    require(
        type(candidate) is dict
        and set(candidate) == set(expected) | {"authorization_id"},
        "Missing/malformed intent",
    )
    authorization_id = candidate["authorization_id"]
    require(
        type(authorization_id) is str
        and len(authorization_id) == 32
        and set(authorization_id) <= set("0123456789abcdef")
        and not _id_used(repository, authorization_id),
        "Reused/invalid authorization ID",
    )
    require(
        canonical_fingerprint(
            {k: v for k, v in candidate.items() if k != "authorization_id"}
        )
        == canonical_fingerprint(expected),
        "Intent provenance/output/model mismatch",
    )
    require(
        type(free_bytes) is int
        and free_bytes
        >= context(repository)["protocol"]["capacity"]["minimum_free_bytes"],
        "Insufficient capacity",
    )
    return {
        "status": "intent_matches_design_only_not_permission",
        "real_execution_authorized": False,
    }


def deny_real_execution(
    *,
    repository: Path,
    test_root: object = None,
    authorization: object = None,
    **kwargs,
) -> NoReturn:
    """Always deny; never resolve/stat/hash/iterate any supplied test-root object."""
    try:
        verify_freeze(repository)
    except (OSError, ValueError, KeyError) as exc:
        raise EvaluationError(
            "Final-test lock CLOSED: B2 identity unavailable"
        ) from exc
    raise EvaluationError(
        "Final-test lock CLOSED: B2 readiness is not human execution authorization"
    )
