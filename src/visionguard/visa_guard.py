"""Repository-derived identities; no caller-asserted expected authorization values."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, sha256_file
from visionguard.visa_protocol import (
    load_visa_protocol,
    verify_audit,
    verify_implementation,
)

FROZEN = {
    "patchcore": "3ffcdc37cf3117d319da3e970383c6d0bdb52e2cd42dca87ad3161c5a9bc3191",
    "efficientad": "78b27feeee044f560287a1ce45000800452344190e7b68003eefb3e2e853f1f9",
}
AUDIT = "06e227bb5d2cd26f38010b2c304c62f14f383a81c64f5b2e4c48f1019128f58f"
MEMBERSHIP = "10e7a6c898fb18fbd1b93a115a5a94f3e2ebcecb7a1796d99d8c1e6f3567ce0b"


def verified_context(repository: Path, model: str) -> dict:
    if model not in FROZEN:
        raise VisaIntegrityError("Unknown frozen model")
    reports = repository / "reports/phase4c-visa-readiness"
    freeze = json.loads((reports / "protocol-freeze.json").read_text())
    if (
        freeze["protocol_fingerprints"] != FROZEN
        or freeze["audit_sha256"] != AUDIT
        or freeze["membership_sha256"] != MEMBERSHIP
        or freeze["final_test_lock"] != "closed"
    ):
        raise VisaIntegrityError("Canonical freeze identity drift")
    protocol = load_visa_protocol(
        repository / f"configs/protocols/{model}-visa-v1.yaml",
        expected_fingerprint=FROZEN[model],
    )["protocol"]
    verify_implementation(repository, protocol)
    verify_audit(reports / "audit-summary.json", AUDIT)
    if sha256_file(reports / "development-membership.json") != MEMBERSHIP:
        raise VisaIntegrityError("Canonical membership identity drift")
    return protocol


def reject_test_request(
    repository: Path,
    model: str,
    *,
    supplied_fingerprint: str,
    supplied_audit: str,
    confirmed: bool,
) -> None:
    verified_context(repository, model)
    if not confirmed:
        raise VisaIntegrityError("Final-test lock CLOSED: confirmation absent")
    if supplied_fingerprint != FROZEN[model]:
        raise VisaIntegrityError(
            "Final-test lock CLOSED: protocol fingerprint mismatch"
        )
    if supplied_audit != AUDIT:
        raise VisaIntegrityError("Final-test lock CLOSED: dataset audit mismatch")
    raise VisaIntegrityError("Final-test lock CLOSED: Phase 4D is not authorized")
