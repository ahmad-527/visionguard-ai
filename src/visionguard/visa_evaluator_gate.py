"""Permanently closed B1 final-test gate; identity is never authorization."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from visionguard.visa_evaluator import EvaluationError


def deny_final_test(
    *,
    repository: Path,
    supplied_fingerprint: str | None = None,
    supplied_audit: str | None = None,
    human_authorization: object = None,
    test_root: object = None,
) -> NoReturn:
    """Always deny before resolving/enumerating/stat-ing any supplied test root.

    Canonical expected identities come independently from frozen repository
    documents, not supplied assertions. Even matching identities, arbitrary
    authorization text or an environment variable cannot open the B1 gate.
    A later reviewed stage must separately implement authorization/lifecycle.
    """
    from visionguard.visa_evaluator_protocol import verify_readiness_freeze

    try:
        expected = verify_readiness_freeze(repository)
    except (OSError, ValueError, KeyError) as exc:
        raise EvaluationError(
            "Final-test lock CLOSED: repository identity unavailable"
        ) from exc
    if supplied_fingerprint != expected["fingerprint"]:
        raise EvaluationError("Final-test lock CLOSED: evaluator identity mismatch")
    if supplied_audit != expected["document"]["protocol"]["dataset_audit_sha256"]:
        raise EvaluationError("Final-test lock CLOSED: audit identity mismatch")
    raise EvaluationError(
        "Final-test lock CLOSED: B1 never authorizes real test execution"
    )


def evaluate_final_test(**kwargs: object) -> NoReturn:
    """No opening path, no dataset or trained-model loader behind this API."""
    deny_final_test(**kwargs)
