"""Synthetic operational faults only; no data/model training or test outcomes."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from visionguard.calibration import highest_order_statistic
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_matrix import command, counts, disk_check
from visionguard.visa_matrix_contract import MODELS, order, source_hashes
from visionguard.visa_matrix_validation import compact_embeddings, validate_calibration
from visionguard.visa_protocol import canonical_fingerprint


def test_canonical_72_order_and_no_test_cli(tmp_path):
    schedule = order()
    assert len(schedule) == len(set(schedule)) == 72
    assert schedule[:3] == [
        "patchcore:candle:42",
        "patchcore:candle:123",
        "patchcore:candle:2026",
    ]
    assert schedule[35] == "patchcore:pipe_fryum:2026"
    assert schedule[36] == "efficientad:candle:42"
    local = {
        k: "synthetic"
        for k in (
            "weight",
            "teacher",
            "imagenette_root",
            "imagenette_archive",
            "development",
        )
    }
    for model in MODELS:
        cmd = command(tmp_path, tmp_path, local, model, "candle", 42, True)
        assert "--resume-cell" in cmd
        assert "--confirm-full-development-fit" in cmd
        assert "--confirm-independent-test-evaluation" not in cmd
        assert "--engineering-smoke" not in cmd


def test_partial_status_is_not_complete():
    document = {"cells": {k: {"status": "pending"} for k in order()}}
    document["cells"][order()[0]]["status"] = "completed"
    document["cells"][order()[1]]["status"] = "interrupted"
    document["cells"][order()[2]]["status"] = "failed"
    document["cells"][order()[3]]["status"] = "active"
    result = counts(document)
    assert result["completed_total"] == 1
    assert result["patchcore"] == {
        "completed": 1,
        "active": 1,
        "interrupted": 1,
        "failed": 1,
        "pending": 32,
    }
    assert result["efficientad"]["pending"] == 36


def test_disk_gate_fails_before_exhaustion(tmp_path, monkeypatch):
    class Disk:
        free = 10

    monkeypatch.setattr("visionguard.visa_matrix.shutil.disk_usage", lambda _: Disk())
    with pytest.raises(VisaIntegrityError, match="Disk pressure"):
        disk_check(tmp_path, {"storage": {"minimum_free_bytes": 20}})


def test_code_changes_change_operational_fingerprint(tmp_path):
    (tmp_path / "src/visionguard").mkdir(parents=True)
    source = tmp_path / "src/visionguard/synthetic.py"
    source.write_text("# first")
    (tmp_path / "pyproject.toml").write_text("# synthetic")
    before = canonical_fingerprint(source_hashes(tmp_path))
    source.write_text("# changed")
    assert canonical_fingerprint(source_hashes(tmp_path)) != before


def test_committed_contract_reproduces_and_binds_every_cell():
    import yaml

    from visionguard.visa_matrix_contract import CONTRACT, load_contract

    repository = Path(__file__).resolve().parents[1]
    # Historical D-A source bytes remain immutable. Later-phase additions must
    # NOT weaken its live execution guard or permit restarting that old matrix.
    snapshot = yaml.safe_load((repository / CONTRACT).read_text())
    document, fingerprint = snapshot["execution"], snapshot["fingerprint"]
    for relative, digest in document["source_hashes"].items():
        assert sha256_file(repository / relative) == digest
    assert (
        fingerprint
        == "31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3"
    )
    if source_hashes(repository) != document["source_hashes"]:
        with pytest.raises(VisaIntegrityError, match="STOP MATRIX"):
            load_contract(repository)
    else:
        assert load_contract(repository) == (document, fingerprint)
    assert document["order"] == order()
    assert document["checkpoint_policy"]["efficientad_interval_steps"] == 1000
    assert document["checkpoint_policy"]["efficientad_final_step"] == 70000
    assert document["human_access_history"] == "UNKNOWN"
    assert document["independent_reservation"] == "NOT ESTABLISHED"
    assert document["phase4d_b_authorized"] is False
    assert document["final_test_lock"] == "closed"
    assert fingerprint == canonical_fingerprint(document)
    changed = copy.deepcopy(document)
    changed["order"].reverse()
    assert canonical_fingerprint(changed) != fingerprint


def test_matrix_modules_are_lightweight_and_have_no_test_command():
    import subprocess
    import sys

    for module in ("visa_matrix", "visa_matrix_contract", "visa_matrix_validation"):
        result = subprocess.run(
            [sys.executable, "-m", f"visionguard.{module}", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "evaluate-test" not in result.stdout
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import visionguard.visa_matrix; "
            "assert 'torch' not in sys.modules; assert 'anomalib' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def calibration():
    records = [
        {
            "path": f"candle/calibration/{i}.JPG",
            "image_identity": {"height": 2, "width": 3},
        }
        for i in range(19)
    ]
    rows = [
        {
            "normal_id": f"{i}.JPG",
            "score": float(i),
            "pixel_maximum": float(i + 1),
            "shape": [2, 3],
            "map_sha256": "a" * 64,
        }
        for i in range(19)
    ]
    return records, {
        "inputs": rows,
        "image": asdict(
            highest_order_statistic([r["score"] for r in rows], minimum_samples=19)
        ),
        "pixel": asdict(
            highest_order_statistic(
                [r["pixel_maximum"] for r in rows], minimum_samples=19
            )
        ),
    }


@pytest.mark.parametrize("fault", [None, "order", "count", "threshold", "nan", "shape"])
def test_exact_calibration_contract(calibration, fault):
    records, evidence = calibration
    if fault == "order":
        evidence["inputs"].reverse()
    elif fault == "count":
        evidence["inputs"].pop()
    elif fault == "threshold":
        evidence["image"]["threshold"] += 1
    elif fault == "nan":
        evidence["inputs"][0]["score"] = float("nan")
    elif fault == "shape":
        evidence["inputs"][0]["shape"] = [3, 2]
    if fault:
        with pytest.raises(VisaIntegrityError):
            validate_calibration(evidence, records)
    else:
        validate_calibration(evidence, records)


@pytest.fixture
def compactable(tmp_path):
    root = tmp_path / "patchcore"
    directory = root / "runs/candle/seed-42/attempt-1"
    chunks = directory / "chunks"
    chunks.mkdir(parents=True)
    path = chunks / "000000.pt"
    path.write_bytes(b"synthetic not a model")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    ]
    atomic_json(directory / "embeddings.json", {"chunks": rows})
    attempts = [{"status": "development_complete"}]
    atomic_json(
        root / "manifest.json", {"cells": {"candle:42": {"attempts": attempts}}}
    )
    result = {
        "model": "patchcore",
        "category": "candle",
        "seed": 42,
        "status": "validated",
        "attempts": attempts,
        "chunk_inventory_sha256": canonical_fingerprint(rows),
        "fit_evidence": {
            "ordered_embeddings_sha256": "a" * 64,
            "memory_bank_sha256": "b" * 64,
        },
        "identity": {"membership_sha256": "c" * 64},
        "coreset_indices_sha256": "d" * 64,
        "files": {
            "embedding_journal": {
                "path": (directory / "embeddings.json").relative_to(root).as_posix(),
                "sha256": sha256_file(directory / "embeddings.json"),
            },
            "final_model": {"sha256": "e" * 64},
            "calibration": {"sha256": "f" * 64},
        },
    }
    return root, result, tmp_path / "compaction.json", path


def test_compaction_retains_inventory_and_is_idempotent(compactable):
    root, result, receipt, chunk = compactable
    before = (chunk.parent.parent / "embeddings.json").read_bytes()
    summary = compact_embeddings(root, result, receipt)
    assert summary["status"] == "completed"
    assert not chunk.exists()
    assert (chunk.parent.parent / "embeddings.json").read_bytes() == before
    assert compact_embeddings(root, result, receipt) == summary


@pytest.mark.parametrize(
    "fault", ["active", "unvalidated", "corrupt", "missing", "lock", "escape"]
)
def test_compaction_refuses_unsafe_or_incomplete(compactable, fault):
    root, result, receipt, chunk = compactable
    result = copy.deepcopy(result)
    if fault == "active":
        result["attempts"][-1]["status"] = "active"
    elif fault == "unvalidated":
        result["status"] = "pending"
    elif fault == "corrupt":
        chunk.write_bytes(b"corrupt")
    elif fault == "missing":
        chunk.unlink()
    elif fault == "lock":
        (root / ".writer.lock").write_text("synthetic lock")
    else:
        journal = root / result["files"]["embedding_journal"]["path"]
        document = json.loads(journal.read_text())
        document["chunks"][0]["path"] = "../outside.pt"
        atomic_json(journal, document)
        result["files"]["embedding_journal"]["sha256"] = sha256_file(journal)
        result["chunk_inventory_sha256"] = canonical_fingerprint(document["chunks"])
    with pytest.raises(VisaIntegrityError):
        compact_embeddings(root, result, receipt)
    assert not receipt.exists()


def test_compaction_crash_after_intent_can_resume(compactable, monkeypatch):
    root, result, receipt, chunk = compactable
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path == chunk:
            raise OSError("synthetic interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(OSError):
        compact_embeddings(root, result, receipt)
    assert json.loads(receipt.read_text())["status"] == "prepared"
    monkeypatch.setattr(Path, "unlink", original)
    assert compact_embeddings(root, result, receipt)["status"] == "completed"


@pytest.mark.parametrize("fail_first", [False, True])
def test_supervisor_schedule_validation_barrier_and_fail_closed(
    tmp_path, monkeypatch, fail_first
):
    import visionguard.visa_matrix as module

    root = tmp_path / "outputs/matrix"
    contract = {
        "order": order(),
        "monitor_seconds": 60,
        "storage": {"minimum_free_bytes": 1},
    }
    monkeypatch.setattr(
        module, "verify_execution", lambda *_: (contract, "frozen", "commit")
    )
    monkeypatch.setattr(module, "hardware_observation", lambda _: {"boot": "synthetic"})
    monkeypatch.setattr(module, "check_assets", lambda *_: None)
    monkeypatch.setattr(module, "expected_identity", lambda *_: {"synthetic": True})
    launched, validated = [], []

    class Worker:
        pid = 123

        def __init__(self, cmd, **_):
            model = cmd[cmd.index("--model") + 1]
            index = cmd.index("--run-cell")
            category, seed = cmd[index + 1 : index + 3]
            if model == "efficientad":
                assert len(validated) >= 72  # 36 cells + fresh independent pass
            launched.append(f"{model}:{category}:{seed}")
            path = root / model / "manifest.json"
            manifest = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "identity": {"synthetic": True},
                    "cells": {
                        ":".join(k.split(":")[1:]): {"attempts": []}
                        for k in order()
                        if k.startswith(model + ":")
                    },
                }
            )
            manifest["cells"][f"{category}:{seed}"] = {
                "attempts": [{"status": "development_complete"}]
            }
            atomic_json(path, manifest)
            (root / model / "runs" / category / f"seed-{seed}").mkdir(parents=True)

        def wait(self, timeout=None):
            return 1 if fail_first else 0

    def validate(repository, output_root, model, category, seed, identity, output):
        validated.append(f"{model}:{category}:{seed}")
        value = {"synthetic": True}
        atomic_json(output, value)
        return value

    def compact(model_root, result, output):
        atomic_json(output, {"synthetic": True})

    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    monkeypatch.setattr(module, "validate_subprocess", validate)
    monkeypatch.setattr(module, "compact_embeddings", compact)
    local = {
        k: "synthetic"
        for k in (
            "weight",
            "teacher",
            "imagenette_root",
            "imagenette_archive",
            "development",
        )
    }
    if fail_first:
        with pytest.raises(VisaIntegrityError, match="Dispatcher failed"):
            module.run(tmp_path, root, local, "frozen")
        assert launched == [order()[0]]
        assert not (root / "development-freeze.json").exists()
        assert (
            json.loads((root / "matrix-manifest.json").read_text())["status"]
            == "stopped"
        )
    else:
        module.run(tmp_path, root, local, "frozen")
        assert launched == order()
        assert len(validated) == 144
        freeze = json.loads((root / "development-freeze.json").read_text())
        assert freeze["validated_cells"] == 72
        assert freeze["final_test_lock"] == "closed"
