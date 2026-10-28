"""Explicit successor source freeze, preserving immutable predecessor evidence."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_protocol import canonical_fingerprint

PREDECESSOR = "reports/phase4d-b2-controlled-activation/implementation-freeze.json"
PREDECESSOR_FINGERPRINT = (
    "960811034e904c6f86a8e127ef0b149a67fb2ec5715e96333e813d461d4d431f"
)
PREDECESSOR_MERGE = "91b5b45a82ccef042ad13535d2456ea7d6013885"
PREDECESSOR_HEAD = "58062446d7a6d7bded8515e2d884de04981c843f"
PROTOCOL = "configs/protocols/visa-authorization-amendment-v2.json"
FREEZE = "reports/stage-b-acl-compatibility/implementation-freeze-v2.json"
ARCHIVES = {
    f"src/visionguard/{name}.py": (
        f"reports/stage-b-acl-compatibility/predecessor/{name}.py.txt"
    )
    for name in ("heldout_authorization", "heldout_contract")
}
ADDITIONAL = (
    "src/visionguard/windows_trust_acl.py",
    "src/visionguard/authorization_amendment.py",
    "scripts/security_amendment_freeze.py",
    "scripts/security_acl_diagnostic.py",
    "tests/test_windows_trust_acl.py",
    "tests/test_authorization_amendment.py",
    ".github/workflows/authorization-amendment.yml",
    PROTOCOL,
)


def verify_predecessor(repository: Path) -> dict:
    """Historical v1 is evidence, never a current-execution fallback."""
    saved = json.loads((repository / PREDECESSOR).read_text())
    require(
        saved["fingerprint"]
        == PREDECESSOR_FINGERPRINT
        == canonical_fingerprint(saved["document"]),
        "Predecessor fingerprint drift",
    )
    for name, expected in saved["document"]["source_sha256"].items():
        actual = ARCHIVES.get(name, name)
        require(sha256_file(repository / actual) == expected, "Historical source drift")
    for name, expected in saved["document"]["evidence_sha256"].items():
        require(sha256_file(repository / name) == expected, "Historical evidence drift")
    return saved


def build_freeze(repository: Path) -> dict:
    from visionguard.heldout_contract import context

    old = verify_predecessor(repository)
    ctx = context(repository)
    protocol = json.loads((repository / PROTOCOL).read_text())
    require(
        protocol["predecessor_activation_fingerprint"] == old["fingerprint"]
        and protocol["predecessor_merge"] == PREDECESSOR_MERGE
        and protocol["predecessor_head"] == PREDECESSOR_HEAD,
        "Amendment predecessor drift",
    )
    require(
        ctx["activation"] == old["document"]["contract"], "Scientific contract drift"
    )
    models = {k: vars(v) for k, v in ctx["specs"].items()}
    environment = ctx["published"]["execution_contract"]["environment"]
    require(models == old["document"]["models"], "Frozen models/thresholds drift")
    require(environment == old["document"]["environment"], "Frozen environment drift")
    paths = set(old["document"]["source_sha256"]) | set(ADDITIONAL)
    paths.update(ARCHIVES.values())
    paths.add(PREDECESSOR)
    document = {
        "schema_version": 2,
        "amendment": protocol,
        "predecessor_fingerprint": old["fingerprint"],
        "predecessor_source_sha256": old["document"]["source_sha256"],
        "predecessor_evidence_sha256": old["document"]["evidence_sha256"],
        "contract": ctx["activation"],
        "models": models,
        "environment": environment,
        "source_sha256": {p: sha256_file(repository / p) for p in sorted(paths)},
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository: Path) -> dict:
    saved = json.loads((repository / FREEZE).read_text())
    require(saved == build_freeze(repository), "Successor/source fingerprint drift")
    return saved
