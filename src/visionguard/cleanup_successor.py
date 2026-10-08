"""Create-once diagnostics v5 candidate over immutable v3/v4 operational evidence."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_protocol import canonical_fingerprint

PREDECESSOR_HEAD = "d5aae83010e32524bb4787afdcc9ff32aeceb6ee"
PREDECESSOR_FINGERPRINT = (
    "68e5e1bc3db4170e9dcebc6d7e844e8343e32ee16cce0032c6749e78c894c469"
)
PROTOCOL = "configs/protocols/visa-cleanup-diagnostics-v5.json"
FREEZE = "reports/cleanup-diagnostics-v5/implementation-freeze-v5.json"
CHANGED = (
    "src/visionguard/accounting_successor.py",
    "src/visionguard/heldout_runner.py",
    "src/visionguard/heldout_contract.py",
    ".github/workflows/output-accounting.yml",
)
ARCHIVES = {
    name: "reports/cleanup-diagnostics-v5/predecessor/"
    + name.replace("/", "__")
    + ".txt"
    for name in CHANGED
}
ADDITIONAL = (
    "src/visionguard/cleanup_successor.py",
    "src/visionguard/heldout_cleanup.py",
    "tests/test_cleanup_successor.py",
    "tests/test_heldout_cleanup.py",
    "scripts/cleanup_successor_freeze.py",
    "scripts/probe_heldout_cleanup.py",
    "scripts/run_shutdown_review.py",
    "docs/heldout-cleanup-diagnostics-v5.md",
    PROTOCOL,
)


def build_freeze(repository: Path) -> dict:
    """Bind current source, verified predecessors, and unchanged scientific identity."""
    from visionguard.accounting_successor import FREEZE as old_path
    from visionguard.accounting_successor import verify_freeze as old_verify
    from visionguard.heldout_contract import context

    old = old_verify(repository)
    require(old["fingerprint"] == PREDECESSOR_FINGERPRINT, "V4 predecessor drift")
    ctx = context(repository)
    models = {key: vars(spec) for key, spec in ctx["specs"].items()}
    environment = ctx["published"]["execution_contract"]["environment"]
    require(
        ctx["activation"] == old["document"]["contract"]
        and models == old["document"]["models"]
        and environment == old["document"]["environment"],
        "Scientific identity drift",
    )
    protocol = json.loads((repository / PROTOCOL).read_bytes())
    require(
        protocol["predecessor_head"] == PREDECESSOR_HEAD
        and protocol["predecessor_fingerprint"] == old["fingerprint"],
        "Cleanup predecessor drift",
    )
    paths = set(old["document"]["source_sha256"]) | set(ADDITIONAL)
    paths.update(ARCHIVES.values())
    paths.add(old_path)
    document = {
        "schema_version": 5,
        "operational_protocol": protocol,
        "historical_fingerprints": old["document"]["historical_fingerprints"]
        + [old["fingerprint"]],
        "contract": ctx["activation"],
        "models": models,
        "environment": environment,
        "source_sha256": {
            name: sha256_file(repository / name) for name in sorted(paths)
        },
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository: Path) -> dict:
    """No historical regeneration, fallback, or authorization fingerprint reuse."""
    saved = json.loads((repository / FREEZE).read_bytes())
    require(saved == build_freeze(repository), "Cleanup/source fingerprint drift")
    return saved
