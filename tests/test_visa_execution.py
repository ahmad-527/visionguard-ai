"""Synthetic durable-state and closed-test-gate checks; no scientific runs."""

import json

import pytest

from visionguard.visa_acquire import VisaIntegrityError
from visionguard.visa_execution import (
    ExecutionState,
    cells,
    exclusive_writer,
    reject_final_test,
)


def synthetic_identity(version="v1"):
    return {
        "model": "patchcore",
        "protocol_fingerprint": "a" * 64,
        "audit_sha256": "b" * 64,
        "membership_sha256": "c" * 64,
        "implementation_commit": "d" * 40,
        "source_fingerprint": "e" * 64,
        "environment": {"synthetic": version},
    }


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "confirmation"),
        ({"confirmed": True}, "fingerprint"),
        (
            {
                "confirmed": True,
                "supplied_fingerprint": "a",
                "expected_fingerprint": "b",
            },
            "fingerprint",
        ),
        (
            {
                "confirmed": True,
                "supplied_fingerprint": "a",
                "expected_fingerprint": "a",
                "supplied_audit_sha256": "b",
                "expected_audit_sha256": "c",
            },
            "audit",
        ),
        (
            {
                "confirmed": True,
                "supplied_fingerprint": "a",
                "expected_fingerprint": "a",
                "supplied_audit_sha256": "b",
                "expected_audit_sha256": "b",
            },
            "Phase 4D",
        ),
    ],
)
def test_final_test_lock_cannot_open_in_phase4c(kwargs, message):
    with pytest.raises(VisaIntegrityError, match=message):
        reject_final_test(**kwargs)


def test_partial_matrix_never_complete_and_resume_identity(tmp_path):
    state = ExecutionState(tmp_path, synthetic_identity())
    state.initialize()
    assert len(cells()) == 36
    assert state.matrix_status() == {
        "expected_cells": 36,
        "development_complete_cells": 0,
        "development_matrix_complete": False,
        "confirmatory_evaluation_complete": False,
        "final_test_lock": "closed",
    }
    with pytest.raises(VisaIntegrityError, match="identity"):
        ExecutionState(tmp_path, synthetic_identity("v2")).initialize()


def test_attempts_checkpoint_corruption_and_immutable_history(tmp_path):
    state = ExecutionState(tmp_path, synthetic_identity())
    state.initialize()
    index, attempt = state.begin("candle", 42)
    assert index == 1
    checkpoint = attempt / "synthetic-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic checkpoint, not a trained model")
    state.update("candle", 42, stage="fit", checkpoint=checkpoint)
    assert state.validated_checkpoint("candle", 42) == checkpoint
    state.update(
        "candle",
        42,
        stage="fit",
        status="interrupted",
        reason="synthetic intentional stop",
    )
    with pytest.raises(VisaIntegrityError, match="immutable"):
        state.update("candle", 42, stage="fit")
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(VisaIntegrityError, match="corruption"):
        state.validated_checkpoint("candle", 42)
    _, next_attempt = state.begin("candle", 42)
    assert next_attempt != attempt
    assert (attempt / "origin.json").exists()
    state.update("candle", 42, stage="fit", status="failed", reason="synthetic failure")
    histories = state.read()["cells"]["candle:42"]["attempts"]
    assert [a["status"] for a in histories] == ["interrupted", "failed"]


def test_completed_requires_verified_checkpoint_and_artifact(tmp_path):
    state = ExecutionState(tmp_path, synthetic_identity())
    state.initialize()
    _, directory = state.begin("candle", 42)
    with pytest.raises(VisaIntegrityError, match="requires checkpoint"):
        state.update(
            "candle", 42, stage="development_complete", status="development_complete"
        )
    checkpoint, calibration = directory / "checkpoint", directory / "calibration.json"
    checkpoint.write_bytes(b"synthetic")
    calibration.write_text(json.dumps({"synthetic": True}))
    state.update(
        "candle",
        42,
        stage="development_complete",
        status="development_complete",
        checkpoint=checkpoint,
        artifacts={"calibration": calibration},
    )
    assert state.validate_completed("candle", 42)
    assert not state.matrix_status()["development_matrix_complete"]
    calibration.write_text("changed")
    with pytest.raises(VisaIntegrityError, match="corruption"):
        state.validate_completed("candle", 42)


def test_concurrent_writer_rejected(tmp_path):
    with (
        exclusive_writer(tmp_path),
        pytest.raises(VisaIntegrityError, match="writer"),
        exclusive_writer(tmp_path),
    ):
        raise AssertionError("must not enter")


@pytest.mark.parametrize(
    "category,seed",
    [("unknown", 42), ("candle", 42.0), ("candle", True), ("candle", 0)],
)
def test_invalid_cell(tmp_path, category, seed):
    state = ExecutionState(tmp_path, synthetic_identity())
    state.initialize()
    with pytest.raises(VisaIntegrityError):
        state.begin(category, seed)


def test_incomplete_provenance_fails_closed(tmp_path):
    with pytest.raises(VisaIntegrityError, match="provenance"):
        ExecutionState(tmp_path, {})
