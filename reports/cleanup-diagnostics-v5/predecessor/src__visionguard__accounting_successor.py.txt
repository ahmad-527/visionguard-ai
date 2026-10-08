"""Distinct operational v4; the live v3 and all scientific freezes remain frozen."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_protocol import canonical_fingerprint

PREDECESSOR_MERGE = "6e0bfb0c5cb3496efe760b42871e47a84995c778"
PREDECESSOR_FINGERPRINT = (
    "0334a17b5b5c8f954e692c4caa47f93eaead0228acb6c45a082a75ca3fbc1c47"
)
PROTOCOL = "configs/protocols/visa-output-accounting-v4.json"
FREEZE = "reports/output-accounting-v4/implementation-freeze-v4.json"
CHANGED = (
    "src/visionguard/heldout_runner.py",
    "src/visionguard/heldout_watcher.py",
    "src/visionguard/heldout_resources.py",
    "src/visionguard/heldout_storage.py",
    "src/visionguard/heldout_stages.py",
    "src/visionguard/heldout_contract.py",
    "src/visionguard/operational_successor.py",
    "tests/test_operational_successor.py",
    "tests/test_authorization_amendment.py",
)
ARCHIVES = {
    p: "reports/output-accounting-v4/predecessor/" + p.replace("/", "__") + ".txt"
    for p in CHANGED
}
ADDITIONAL = (
    "src/visionguard/accounting_successor.py",
    "src/visionguard/heldout_accounting.py",
    "src/visionguard/heldout_changes.py",
    "tests/test_heldout_accounting.py",
    "tests/test_accounting_successor.py",
    "scripts/output_accounting_freeze.py",
    "scripts/benchmark_output_accounting.py",
    ".github/workflows/output-accounting.yml",
    PROTOCOL,
)


def build_freeze(repository: Path):
    from visionguard.heldout_contract import context
    from visionguard.operational_successor import FREEZE as old_path
    from visionguard.operational_successor import verify_freeze as old_verify

    old = old_verify(repository)
    require(old["fingerprint"] == PREDECESSOR_FINGERPRINT, "V3 predecessor drift")
    ctx = context(repository)
    models = {k: vars(v) for k, v in ctx["specs"].items()}
    environment = ctx["published"]["execution_contract"]["environment"]
    require(
        ctx["activation"] == old["document"]["contract"]
        and models == old["document"]["models"]
        and environment == old["document"]["environment"],
        "Scientific identity drift",
    )
    protocol = json.loads((repository / PROTOCOL).read_text())
    require(
        protocol["predecessor_merge"] == PREDECESSOR_MERGE
        and protocol["predecessor_fingerprint"] == old["fingerprint"],
        "Accounting predecessor drift",
    )
    paths = set(old["document"]["source_sha256"]) | set(ADDITIONAL)
    paths.update(ARCHIVES.values())
    paths.add(old_path)
    document = {
        "schema_version": 4,
        "operational_protocol": protocol,
        "historical_fingerprints": old["document"]["historical_fingerprints"]
        + [old["fingerprint"]],
        "contract": ctx["activation"],
        "models": models,
        "environment": environment,
        "source_sha256": {p: sha256_file(repository / p) for p in sorted(paths)},
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository: Path):
    saved = json.loads((repository / FREEZE).read_text())
    require(saved == build_freeze(repository), "Accounting/source fingerprint drift")
    return saved
