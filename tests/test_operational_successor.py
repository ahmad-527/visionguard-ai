"""Frozen science and historical source checks; no model or input loading."""

import copy
import json
from pathlib import Path

import pytest

from visionguard import operational_successor as successor
from visionguard.visa_protocol import canonical_fingerprint

REPOSITORY = Path(__file__).resolve().parents[1]


def test_historical_source_archives_and_72_models_exact():
    first, second = successor.historical(REPOSITORY)
    from visionguard.heldout_contract import context

    ctx = context(REPOSITORY)
    assert {k: vars(v) for k, v in ctx["specs"].items()} == second["document"]["models"]
    assert len(ctx["specs"]) == 72
    assert (
        first["document"]["contract"]
        == second["document"]["contract"]
        == ctx["activation"]
    )


def test_successor_fingerprint_stable_scientific_and_operational_binding():
    before = successor.build_freeze(REPOSITORY)
    assert successor.build_freeze(REPOSITORY) == before
    assert before["document"]["schema_version"] == 3
    for field in ("models", "contract", "operational_protocol", "source_sha256"):
        changed = copy.deepcopy(before["document"])
        changed[field] = {"manufactured_change": True}
        assert canonical_fingerprint(changed) != before["fingerprint"]


@pytest.mark.parametrize("kind", ["archive", "science", "protocol"])
def test_fail_closed_drift(tmp_path, monkeypatch, kind):
    saved = successor.build_freeze(REPOSITORY)
    first, _ = successor.historical(REPOSITORY)
    paths = set(saved["document"]["source_sha256"]) | set(
        first["document"]["evidence_sha256"]
    )
    # Manufacturing a repository of text/config evidence, not model artifacts.
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY / relative).read_bytes())
    from visionguard import heldout_contract

    ctx = heldout_contract.context(REPOSITORY)
    monkeypatch.setattr(heldout_contract, "context", lambda _: ctx)
    if kind == "archive":
        path = tmp_path / next(iter(successor.ARCHIVES.values()))
        path.write_bytes(path.read_bytes() + b"manufactured corruption")
    elif kind == "science":
        ctx["activation"] = {"wrong": True}
    else:
        path = tmp_path / successor.PROTOCOL
        protocol = json.loads(path.read_bytes())
        protocol["predecessor_fingerprint"] = "0" * 64
        path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="drift"):
        successor.build_freeze(tmp_path)
