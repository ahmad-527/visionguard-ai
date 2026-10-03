"""Repository freeze and closed-gate negatives, with synthetic sentinels only."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from visionguard.visa_evaluator import EvaluationError
from visionguard.visa_evaluator_gate import evaluate_final_test
from visionguard.visa_evaluator_protocol import (
    FREEZE_PATH,
    PROTOCOL_PATH,
    verify_readiness_freeze,
)
from visionguard.visa_protocol import canonical_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_fingerprint_reproduces_stably():
    first = verify_readiness_freeze(ROOT)
    assert first == verify_readiness_freeze(ROOT)
    assert canonical_fingerprint(first["document"]) == first["fingerprint"]


@pytest.mark.parametrize(
    "rule",
    [
        "triage",
        "metrics",
        "seeds",
        "retention",
        "gate",
        "historical_access",
        "development_freeze_sha256",
    ],
)
def test_scientific_mutations_alter_fingerprint(rule):
    snapshot = verify_readiness_freeze(ROOT)
    changed = deepcopy(snapshot["document"])
    changed["protocol"][rule] = "synthetic deliberate mutation"
    assert canonical_fingerprint(changed) != snapshot["fingerprint"]


def test_bound_source_or_config_changes_rejected(tmp_path):
    snapshot = verify_readiness_freeze(ROOT)
    source = "synthetic-source.txt"
    (tmp_path / source).write_bytes(b"artificial bound bytes")
    from visionguard.visa_acquire import sha256_file

    snapshot["document"]["source_sha256"] = {source: sha256_file(tmp_path / source)}
    snapshot["fingerprint"] = canonical_fingerprint(snapshot["document"])
    for relative, value in (
        (FREEZE_PATH, snapshot),
        (PROTOCOL_PATH, snapshot["document"]["protocol"]),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    verify_readiness_freeze(tmp_path)
    (tmp_path / source).write_bytes(b"corrupt synthetic implementation")
    with pytest.raises(EvaluationError, match="implementation drift"):
        verify_readiness_freeze(tmp_path)
    changed = deepcopy(snapshot["document"]["protocol"])
    changed["seeds"] = [42]
    (tmp_path / PROTOCOL_PATH).write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(EvaluationError, match="methodology"):
        verify_readiness_freeze(tmp_path)


class NoAccess:
    def __fspath__(self):
        raise AssertionError("Gate must not resolve a test root")

    def __str__(self):
        raise AssertionError("Gate must not inspect a supplied object")


@pytest.mark.parametrize(
    "mode",
    [
        "arbitrary_fp",
        "arbitrary_audit",
        "correct",
        "human_text",
        "unavailable_repository",
    ],
)
def test_gate_never_uses_self_asserted_expected_values_or_opens(
    mode, tmp_path, monkeypatch
):
    snapshot = verify_readiness_freeze(ROOT)
    fp, audit = (
        snapshot["fingerprint"],
        snapshot["document"]["protocol"]["dataset_audit_sha256"],
    )
    if mode == "arbitrary_fp":
        fp = "a" * 64
    if mode == "arbitrary_audit":
        audit = "a" * 64
    monkeypatch.setenv("VISIONGUARD_TEST_AUTHORIZED", "true")
    with pytest.raises(EvaluationError, match="CLOSED"):
        evaluate_final_test(
            repository=tmp_path if mode == "unavailable_repository" else ROOT,
            supplied_fingerprint=fp,
            supplied_audit=audit,
            human_authorization="I authorize B2"
            if mode == "human_text"
            else NoAccess(),
            test_root=NoAccess(),
        )
