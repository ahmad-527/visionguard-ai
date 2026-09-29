"""Synthetic dispatcher/stage faults; no trained models or benchmark outcomes."""

import json
from pathlib import Path

import pytest

from visionguard.embedding_journal import EmbeddingJournal
from visionguard.visa_acquire import VisaIntegrityError, atomic_json
from visionguard.visa_development_worker import (
    advance,
    file_record,
    validate_receipt,
    validate_request_scope,
)
from visionguard.visa_dispatcher import DevelopmentDispatcher, execution_scope, main


def identity(model="patchcore"):
    return {
        "model": model,
        "protocol_fingerprint": "a" * 64,
        "audit_sha256": "b" * 64,
        "membership_sha256": "c" * 64,
        "implementation_commit": "d" * 40,
        "source_fingerprint": "e" * 64,
        "environment": {"synthetic": True},
        "scope": execution_scope(model, True),
    }


def worker_state(dispatcher, directory, stage="fit"):
    document = {
        "identity": {**dispatcher.identity, "category": "candle", "seed": 42},
        "stage": stage,
        "stage_history": ["created", stage],
        "files": {},
        "status": "running",
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
    }
    atomic_json(directory / "worker-state.json", document)
    return document


@pytest.mark.parametrize("model", ["patchcore", "efficientad"])
@pytest.mark.parametrize("stage", ["fit", "normalization", "calibration"])
def test_postfit_stage_interruption_new_attempt_and_immutable_origin(
    tmp_path, model, stage
):
    dispatcher = DevelopmentDispatcher(tmp_path, identity(model))
    directory, prior = dispatcher.prepare("candle", 42, resume=False)
    assert prior is None
    before = (directory / "origin.json").read_bytes()
    document = worker_state(dispatcher, directory, stage)
    if stage != "fit":
        model_file = directory / "synthetic-fit.pt"
        model_file.write_bytes(b"synthetic, not a model")
        document["files"]["fit_model"] = file_record(tmp_path, model_file)
        atomic_json(directory / "worker-state.json", document)
    result = dispatcher.finalize("candle", 42, directory, 75)
    assert result["status"] == "interrupted"
    following, receipt = dispatcher.prepare("candle", 42, resume=True)
    assert following != directory
    assert receipt is not None
    assert (directory / "origin.json").read_bytes() == before
    assert dispatcher.state.matrix_status()["confirmatory_evaluation_complete"] is False


def test_embedding_hard_exit_is_recovered_only_after_chunk_validation(tmp_path):
    dispatcher = DevelopmentDispatcher(tmp_path, identity())
    directory, _ = dispatcher.prepare("candle", 42, resume=False)
    doc = worker_state(dispatcher, directory)
    journal = EmbeddingJournal(tmp_path, directory, doc["identity"], ["synthetic-id"])
    journal_doc = journal.create()
    path = directory / "chunks" / "0.pt"
    path.parent.mkdir()
    path.write_bytes(b"synthetic tensor placeholder")
    journal.append(journal_doc, path, tensor_sha256="f" * 64)
    interrupted = dispatcher.finalize("candle", 42, directory, 75)
    assert interrupted["checkpoint"] is not None
    recovered = json.loads((directory / "worker-state.json").read_text())
    assert "embedding_journal" in recovered["files"]


def test_efficientad_latest_checkpoint_is_recovered(tmp_path):
    dispatcher = DevelopmentDispatcher(tmp_path, identity("efficientad"))
    directory, _ = dispatcher.prepare("candle", 42, resume=False)
    document = worker_state(dispatcher, directory)
    checkpoint = directory / "synthetic-training.pt"
    checkpoint.write_bytes(b"synthetic optimizer scheduler RNG stream placeholder")
    atomic_json(
        directory / "training-state.json",
        {
            "identity": document["identity"],
            "entry": {
                "latest_valid_checkpoint": {
                    **file_record(tmp_path, checkpoint),
                    "step": 1,
                }
            },
        },
    )
    dispatcher.finalize("candle", 42, directory, 75)
    recovered = json.loads((directory / "worker-state.json").read_text())
    assert "training_checkpoint" in recovered["files"]


def test_completed_skip_requires_all_file_hashes_and_is_immutable(tmp_path):
    dispatcher = DevelopmentDispatcher(tmp_path, identity())
    directory, _ = dispatcher.prepare("candle", 42, resume=False)
    document = worker_state(dispatcher, directory, "development_complete")
    document["status"] = "development_complete"
    for name in ("fit_model", "normalized_model", "final_model", "calibration"):
        path = directory / f"{name}.synthetic"
        path.write_bytes(b"synthetic fixture")
        document["files"][name] = file_record(tmp_path, path)
    atomic_json(directory / "worker-state.json", document)
    dispatcher.finalize("candle", 42, directory, 0)
    assert dispatcher.state.validate_completed("candle", 42)
    with pytest.raises(VisaIntegrityError, match="Completed"):
        dispatcher.prepare("candle", 42, resume=False)
    (directory / "calibration.synthetic").write_bytes(b"corrupt")
    with pytest.raises(VisaIntegrityError, match="corruption"):
        dispatcher.state.validate_completed("candle", 42)


def test_corrupt_receipt_marks_failed_not_active(tmp_path):
    dispatcher = DevelopmentDispatcher(tmp_path, identity())
    directory, _ = dispatcher.prepare("candle", 42, resume=False)
    doc = worker_state(dispatcher, directory)
    path = directory / "bad.synthetic"
    path.write_bytes(b"original")
    doc["files"]["fit_model"] = file_record(tmp_path, path)
    atomic_json(directory / "worker-state.json", doc)
    path.write_bytes(b"corrupt")
    with pytest.raises(VisaIntegrityError, match="corruption"):
        dispatcher.finalize("candle", 42, directory, 75)
    assert (
        dispatcher.state.read()["cells"]["candle:42"]["attempts"][-1]["status"]
        == "failed"
    )
    following, _ = dispatcher.prepare("candle", 42, resume=False)
    assert following != directory


def test_hardware_failure_cannot_be_automatically_resumed(tmp_path):
    dispatcher = DevelopmentDispatcher(tmp_path, identity())
    directory, _ = dispatcher.prepare("candle", 42, resume=False)
    worker_state(dispatcher, directory)
    (directory / "worker.log").write_text("synthetic OutOfMemoryError")
    dispatcher.finalize("candle", 42, directory, 1)
    with pytest.raises(VisaIntegrityError, match="Hardware"):
        dispatcher.prepare("candle", 42, resume=True)


def test_stage_regression_and_identity_drift_fail(tmp_path):
    state = {"stage": "calibration", "stage_history": []}
    with pytest.raises(VisaIntegrityError, match="regress"):
        advance(tmp_path, state, "fit")
    dispatcher = DevelopmentDispatcher(tmp_path, identity())
    directory, _ = dispatcher.prepare("candle", 42, resume=False)
    document = worker_state(dispatcher, directory)
    with pytest.raises(VisaIntegrityError, match="identity"):
        validate_receipt(
            tmp_path,
            file_record(tmp_path, directory / "worker-state.json"),
            {**document["identity"], "seed": 123},
        )


def test_plan_and_full_fit_gate_do_not_load_models(tmp_path, capsys):
    repository = Path(__file__).resolve().parents[1]
    common = [
        "--repository",
        str(repository),
        "--model",
        "patchcore",
        "--output",
        str(tmp_path),
    ]
    assert main([*common, "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["cells"]) == 36
    assert plan["cells"][:3] == ["candle:42", "candle:123", "candle:2026"]
    with pytest.raises(VisaIntegrityError, match="explicit confirmation"):
        main([*common, "--run-cell", "candle", "42"])


def test_smoke_scope_cannot_expand(tmp_path):
    dispatcher = DevelopmentDispatcher(tmp_path, identity())
    with pytest.raises(VisaIntegrityError, match="fixture"):
        dispatcher.prepare("pcb3", 42, resume=False)
    with pytest.raises(VisaIntegrityError, match="Invalid"):
        dispatcher.prepare("candle", 0, resume=False)


@pytest.mark.parametrize("mutation", ["seed", "category", "steps", "fit", "test_root"])
def test_worker_cannot_trust_self_asserted_scope(mutation):
    request = {
        "model": "efficientad",
        "category": "candle",
        "seed": 42,
        "stop_at": None,
        "identity": identity("efficientad"),
    }
    if mutation == "seed":
        request["seed"] = 7
    elif mutation == "category":
        request["category"] = "pcb3"
    elif mutation == "steps":
        request["identity"]["scope"]["steps"] = 100
    elif mutation == "fit":
        request["identity"]["scope"]["fit_limit"] = 100
    else:
        request["test_root"] = "synthetic_forbidden"
    with pytest.raises(VisaIntegrityError):
        validate_request_scope(request)
