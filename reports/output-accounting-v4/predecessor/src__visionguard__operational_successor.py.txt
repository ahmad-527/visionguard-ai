"""Explicit v3 operational freeze; old source is verified as historical evidence."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard import authorization_amendment as v2
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_protocol import canonical_fingerprint

PREDECESSOR_MERGE = "808319e29164d69a72d7de6286fc6b014bc546c3"
PREDECESSOR_HEAD = "63e2787989cfe84e3363bfcfd21f37941620560d"
PREDECESSOR_FINGERPRINT = (
    "8b30e6b923c8bb49f457b50860a0adcc62d6bf1031405e28e60ab1c28ddd00b0"
)
PROTOCOL = "configs/protocols/visa-controlled-pause-v3.json"
FREEZE = "reports/controlled-pause-resume-v3/implementation-freeze-v3.json"
CHANGED = (
    "src/visionguard/heldout_runner.py",
    "src/visionguard/heldout_authorization.py",
    "src/visionguard/heldout_contract.py",
    "src/visionguard/heldout_watcher.py",
    "tests/test_heldout_authorization.py",
    "tests/test_authorization_amendment.py",
    "scripts/security_amendment_freeze.py",
)
ARCHIVES = {
    p: "reports/controlled-pause-resume-v3/predecessor/" + p.replace("/", "__") + ".txt"
    for p in CHANGED
}
ADDITIONAL = (
    "src/visionguard/operational_successor.py",
    "src/visionguard/heldout_pause.py",
    "tests/test_heldout_pause.py",
    "tests/test_operational_successor.py",
    "scripts/operational_successor_freeze.py",
    ".github/workflows/controlled-pause.yml",
    PROTOCOL,
)


def historical(repository: Path) -> tuple[dict, dict]:
    """No execution fallback and no regeneration of the old freezes."""
    first = json.loads((repository / v2.PREDECESSOR).read_text())
    second = json.loads((repository / v2.FREEZE).read_text())
    for saved, fp, archives in (
        (first, v2.PREDECESSOR_FINGERPRINT, ARCHIVES | v2.ARCHIVES),
        (second, PREDECESSOR_FINGERPRINT, ARCHIVES),
    ):
        require(
            saved["fingerprint"] == fp == canonical_fingerprint(saved["document"]),
            "Historical activation fingerprint drift",
        )
        for name, expected in saved["document"]["source_sha256"].items():
            require(
                sha256_file(repository / archives.get(name, name)) == expected,
                f"Historical source drift: {name}",
            )
        for name, expected in saved["document"].get("evidence_sha256", {}).items():
            require(sha256_file(repository / name) == expected, "Evidence drift")
    require(
        second["document"]["predecessor_fingerprint"] == first["fingerprint"],
        "Historical chronology drift",
    )
    for field in ("contract", "models", "environment"):
        require(first["document"][field] == second["document"][field], "Science drift")
    return first, second


def build_freeze(repository: Path) -> dict:
    from visionguard.heldout_contract import context

    first, second = historical(repository)
    ctx = context(repository)
    protocol = json.loads((repository / PROTOCOL).read_text())
    require(
        protocol["predecessor_fingerprint"] == second["fingerprint"]
        and protocol["predecessor_merge"] == PREDECESSOR_MERGE
        and protocol["predecessor_head"] == PREDECESSOR_HEAD,
        "Operational predecessor drift",
    )
    models = {k: vars(v) for k, v in ctx["specs"].items()}
    environment = ctx["published"]["execution_contract"]["environment"]
    require(
        ctx["activation"] == second["document"]["contract"]
        and models == second["document"]["models"]
        and environment == second["document"]["environment"],
        "Frozen scientific identity drift",
    )
    paths = set(second["document"]["source_sha256"]) | set(ADDITIONAL)
    paths.update(ARCHIVES.values())
    paths.update((v2.FREEZE, v2.PREDECESSOR))
    document = {
        "schema_version": 3,
        "operational_protocol": protocol,
        "historical_fingerprints": [first["fingerprint"], second["fingerprint"]],
        "contract": ctx["activation"],
        "models": models,
        "environment": environment,
        "source_sha256": {p: sha256_file(repository / p) for p in sorted(paths)},
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository: Path) -> dict:
    saved = json.loads((repository / FREEZE).read_text())
    require(saved == build_freeze(repository), "Operational/source fingerprint drift")
    return saved
