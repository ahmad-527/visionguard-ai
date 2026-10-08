"""Metadata-only freeze acceptance; manufactured mutations, no held-out access."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from visionguard import cleanup_successor as successor
from visionguard.accounting_successor import verify_freeze as verify_v4
from visionguard.heldout_contract import verify_freeze as verify_current
from visionguard.operational_successor import historical
from visionguard.operational_successor import verify_freeze as verify_v3
from visionguard.visa_acquire import sha256_file
from visionguard.visa_b2_storage import json_once
from visionguard.visa_protocol import canonical_fingerprint

REPO = Path(__file__).resolve().parents[1]
PAUSE_TEST_SHA256 = "76d4aa677b860b8a0eb76d221d4a7f753cd004eee6164a3037c7411b26752195"


def test_v5_chain_preserves_exact_predecessors_and_scientific_inputs():
    v3 = verify_v3(REPO)
    v4 = verify_v4(REPO)
    v5 = successor.verify_freeze(REPO)
    assert v3["fingerprint"] == (
        "0334a17b5b5c8f954e692c4caa47f93eaead0228acb6c45a082a75ca3fbc1c47"
    )
    assert v4["fingerprint"] == successor.PREDECESSOR_FINGERPRINT
    assert v5 == successor.build_freeze(REPO) == verify_current(REPO)
    assert v5["document"]["schema_version"] == 5
    assert v5["fingerprint"] not in v5["document"]["historical_fingerprints"]
    assert v5["document"]["historical_fingerprints"] == (
        v4["document"]["historical_fingerprints"] + [v4["fingerprint"]]
    )
    assert v5["document"]["operational_protocol"]["evaluation_lock"] == "CLOSED"
    for field in ("contract", "models", "environment"):
        assert v5["document"][field] == v4["document"][field] == v3["document"][field]
    assert len(v5["document"]["models"]) == 72
    assert sha256_file(REPO / "tests/test_heldout_pause.py") == PAUSE_TEST_SHA256
    for name, archive in successor.ARCHIVES.items():
        assert sha256_file(REPO / archive) == v4["document"]["source_sha256"][name]


@pytest.mark.parametrize(
    "field",
    [
        "operational_protocol",
        "contract",
        "models",
        "environment",
        "source_sha256",
        "historical_fingerprints",
    ],
)
def test_v5_changes_cannot_reuse_authorization_fingerprint(field):
    original = successor.build_freeze(REPO)
    changed = copy.deepcopy(original["document"])
    changed[field] = "manufactured change"
    assert canonical_fingerprint(changed) != original["fingerprint"]


@pytest.fixture
def manufactured_repository(tmp_path, monkeypatch):
    saved = successor.build_freeze(REPO)
    first, _ = historical(REPO)
    names = set(saved["document"]["source_sha256"]) | set(
        first["document"].get("evidence_sha256", {})
    )
    names.add(successor.FREEZE)
    for name in sorted(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
    from visionguard import heldout_contract

    ctx = heldout_contract.context(REPO)
    monkeypatch.setattr(heldout_contract, "context", lambda _: ctx)
    return tmp_path, ctx


@pytest.mark.parametrize(
    "name",
    [
        "tests/test_heldout_pause.py",
        "src/visionguard/heldout_cleanup.py",
        "tests/test_heldout_cleanup.py",
        ".github/workflows/output-accounting.yml",
        "reports/output-accounting-v4/implementation-freeze-v4.json",
        successor.ARCHIVES["src/visionguard/heldout_runner.py"],
        successor.FREEZE,
        successor.PROTOCOL,
    ],
)
def test_v5_fails_closed_on_current_or_historical_drift(manufactured_repository, name):
    root, _ = manufactured_repository
    path = root / name
    if path.suffix == ".json":
        content = json.loads(path.read_bytes())
        content["manufactured_corruption"] = True
        path.write_text(json.dumps(content))
    else:
        path.write_bytes(path.read_bytes() + b"\nmanufactured corruption\n")
    with pytest.raises(ValueError, match="drift"):
        successor.verify_freeze(root)


def test_v5_scientific_identity_drift_refused(manufactured_repository):
    root, ctx = manufactured_repository
    ctx["activation"] = {"manufactured_drift": True}
    with pytest.raises(ValueError, match="drift"):
        successor.build_freeze(root)


def test_candidate_publication_refuses_overwrite_and_preserves_both_predecessors(
    manufactured_repository,
):
    from visionguard.accounting_successor import FREEZE as v4_path
    from visionguard.operational_successor import FREEZE as v3_path

    root, _ = manufactured_repository
    originals = {name: (root / name).read_bytes() for name in (v3_path, v4_path)}
    candidate = root / successor.FREEZE
    before = candidate.read_bytes()
    with pytest.raises(FileExistsError):
        json_once(candidate, successor.build_freeze(root))
    assert candidate.read_bytes() == before
    assert {name: (root / name).read_bytes() for name in originals} == originals


def test_dedicated_ci_runs_cleanup_and_freeze_chain_acceptance():
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/output-accounting.yml").read_text()
    )
    steps = workflow["jobs"]["accounting"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    for name in (
        "scripts/operational_successor_freeze.py",
        "scripts/output_accounting_freeze.py",
        "scripts/cleanup_successor_freeze.py",
        "tests/test_cleanup_successor.py",
        "tests/test_heldout_cleanup.py",
        "tests/test_accounting_successor.py",
    ):
        assert name in commands
    assert "--create" not in commands
