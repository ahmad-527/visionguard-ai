"""Entirely artificial fixtures; no real test paths, trained models or results."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from visionguard.visa import CATEGORIES
from visionguard.visa_b2_admission import (
    _ISSUED,
    FixturePermit,
    admit_fixture,
    make_fixture,
)
from visionguard.visa_b2_contract import build_freeze, context
from visionguard.visa_b2_gate import (
    deny_real_execution,
    expected_intent,
    validate_intent,
)
from visionguard.visa_b2_runner import evaluate_real, plan, run_synthetic
from visionguard.visa_b2_storage import publish_cell, replay_cell, write_once
from visionguard.visa_evaluator import EvaluationError, evaluate_synthetic_cell
from visionguard.visa_evaluator_storage import canonical_bytes
from visionguard.visa_evaluator_synthetic import fixture
from visionguard.visa_protocol import canonical_fingerprint

REPO = Path(__file__).resolve().parents[1]


def test_exact_audit_scope_and_closed_config():
    ctx = context(REPO)
    assert sum(ctx["counts"].values()) == 2162
    assert len(ctx["specs"]) == 72
    assert plan()["expected_real_model_image_calls"] == 12972
    assert plan()["paired_cells"] == [
        [c, s] for c in CATEGORIES for s in (42, 123, 2026)
    ]
    assert ctx["protocol"]["real_execution_enabled"] is False


def test_unissued_root_capability_rejected_before_path_io(monkeypatch):
    monkeypatch.setattr(Path, "lstat", lambda _: pytest.fail("Unissued root touched"))
    with pytest.raises(EvaluationError, match="Unissued"):
        admit_fixture(FixturePermit("arbitrary"))


def test_synthetic_admission_official_shape_order_and_masks(tmp_path):
    token = make_fixture(tmp_path)
    admitted = admit_fixture(token)
    assert set(admitted) == set(CATEGORIES)
    for frames in admitted.values():
        assert [f.sample.sample_id.rsplit(":", 1)[1] for f in frames] == [
            "2",
            "3",
            "0",
            "1",
        ]
        assert [f.sample.label for f in frames] == [1, 1, 0, 0]
        assert all(f.rgb.shape[:2] == f.sample.mask.shape == (2, 2) for f in frames)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "corrupt",
        "wrong-mask",
        "wrong-dimension",
        "reordered-split",
        "wrong-membership",
        "duplicate",
        "extra-file",
    ],
)
def test_source_faults_fail_closed(tmp_path, problem):
    token = make_fixture(tmp_path)
    _, root, origin = _ISSUED[token.nonce]
    path = root / "candle/Data/Images/Normal/synthetic-0.png"
    if problem == "missing":
        path.unlink()
    elif problem == "corrupt":
        path.write_bytes(b"corrupt")
    elif problem in ("wrong-mask", "wrong-dimension"):
        from PIL import Image

        relative = "candle/Data/Masks/Anomaly/synthetic-2.png"
        target = root / relative
        array = np.full(
            (2, 2) if problem == "wrong-mask" else (3, 2),
            128 if problem == "wrong-mask" else 255,
            dtype=np.uint8,
        )
        Image.fromarray(array).save(target)
        # Independently designated synthetic oracle changed to reach semantic check.
        origin["files"][relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    elif problem == "extra-file":
        (root / "unexpected.txt").write_bytes(b"synthetic unrelated data")
    else:
        split = root / "synthetic-split.csv"
        rows = split.read_text().splitlines()
        if problem == "reordered-split":
            index = next(i for i, row in enumerate(rows) if ",test," in row)
            rows[index] = rows[index].replace(",test,", ",train,")
        elif problem == "duplicate":
            rows.append(rows[1])
        else:
            rows.pop(1)
        data = ("\n".join(rows) + "\n").encode()
        split.write_bytes(data)
        origin["split_sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises((EvaluationError, OSError, ValueError)):
        admit_fixture(token)


def test_immutable_publication_and_exact_metric_replay(tmp_path):
    samples, pc, ea, specs = fixture()
    expected = evaluate_synthetic_cell(samples, pc, ea, specs)
    directory = tmp_path / "attempt-1"
    origin = {"source_sha256": "c" * 64}
    receipt = publish_cell(directory, zip(samples, pc, ea, strict=True), specs, origin)
    assert replay_cell(directory, receipt["sha256"], specs, origin) == expected
    assert replay_cell(directory, receipt["sha256"], specs, origin) == expected
    assert expected["triage"]["decision_counts"] == {
        "PASS": 1,
        "REVIEW": 2,
        "REJECT": 1,
    }
    # Independent four-image / sixteen-pixel arithmetic, not another reducer.
    assert expected["models"]["patchcore"]["image"]["confusion"] == {
        "true_positive": 2,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 0,
    }
    assert expected["models"]["efficientad"]["image"]["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert expected["models"]["efficientad"]["image"]["image_auroc"]["value"] == 0.75
    assert expected["models"]["efficientad"]["pixel"]["pixel_f1"][
        "value"
    ] == pytest.approx(2 / 3)
    assert (
        expected["models"]["efficientad"]["pixel"]["pixel_auroc_diagnostic"]["value"]
        == 0.75
    )
    assert expected["triage"]["rates"]["review_rate"]["value"] == 0.5
    assert expected["triage"]["rates"]["selective_accuracy"]["value"] == 1.0
    assert expected["triage"]["rates"]["anomaly_pass_through_rate"]["value"] == 0.0
    with pytest.raises(FileExistsError):
        publish_cell(directory, zip(samples, pc, ea, strict=True), specs, origin)
    with pytest.raises(FileExistsError):
        write_once(directory / "origin.json", b"different")


@pytest.mark.parametrize(
    "fault", ["map", "mask", "completion", "seed", "model", "origin", "origin-file"]
)
def test_replay_corruption_and_provenance(tmp_path, fault):
    samples, pc, ea, specs = fixture()
    origin = {"source_sha256": "c" * 64}
    directory = tmp_path / "attempt-1"
    receipt = publish_cell(directory, zip(samples, pc, ea, strict=True), specs, origin)
    if fault in ("map", "mask", "completion", "origin-file"):
        path = (
            directory
            / {
                "map": "000000-patchcore.tiff",
                "mask": "000000-truth.png",
                "completion": "complete.json",
                "origin-file": "origin.json",
            }[fault]
        )
        path.write_bytes(b"corrupt")
    elif fault == "origin":
        origin = {"source_sha256": "d" * 64}
    else:
        from dataclasses import replace

        specs = (
            replace(
                specs[0],
                **({"seed": 123} if fault == "seed" else {"model_sha256": "d" * 64}),
            ),
            specs[1],
        )
    with pytest.raises((EvaluationError, ValueError)):
        replay_cell(directory, receipt["sha256"], specs, origin)


def test_partial_attempt_never_complete_and_new_attempt_preserves_failure(tmp_path):
    samples, pc, ea, specs = fixture()
    root = tmp_path / "partial"
    with pytest.raises(InterruptedError):
        publish_cell(
            root, zip(samples, pc, ea, strict=True), specs, {}, interrupt_after=1
        )
    assert not (root / "complete.json").exists()
    assert json.loads((root / "failure.json").read_text())["status"] == "interrupted"
    before = (root / "failure.json").read_bytes()
    publish_cell(tmp_path / "new-attempt", zip(samples, pc, ea, strict=True), specs, {})
    assert (root / "failure.json").read_bytes() == before


def test_full_36_pair_matrix_known_answers_and_restart(tmp_path):
    permit = make_fixture(tmp_path / "inputs")
    output = tmp_path / "run"
    with pytest.raises(InterruptedError):
        run_synthetic(output, permit, stop_after=1)
    first = (output / "candle/seed-42/attempt-1/complete.json").read_bytes()
    result = run_synthetic(output, permit)
    assert result["paired_cells"] == 36
    assert result["validated_skip_completed"] == 1
    assert all(
        c["sample_count"] == 4
        and c["triage"]["decision_counts"] == {"PASS": 1, "REVIEW": 2, "REJECT": 1}
        for c in result["cells"]
    )
    assert (
        result["aggregate"]["seeds"]["42"]["category_macro"]["patchcore"][
            "image_auroc"
        ]["mean"]
        == 1.0
    )
    assert (output / "candle/seed-42/attempt-1/complete.json").read_bytes() == first
    replay = run_synthetic(output, permit)
    assert replay["validated_skip_completed"] == 36
    assert canonical_bytes(replay["aggregate"]) == canonical_bytes(result["aggregate"])


def test_multiple_writer_and_stale_lock_refusal(tmp_path):
    token = make_fixture(tmp_path / "inputs")
    output = tmp_path / "run"
    output.mkdir()
    lock = output / "writer.lock"
    lock.write_bytes(b"unknown-owner")
    with pytest.raises(FileExistsError):
        run_synthetic(output, token)
    assert lock.read_bytes() == b"unknown-owner"


def test_completed_corrupt_cell_not_silently_recomputed(tmp_path):
    token = make_fixture(tmp_path / "inputs")
    output = tmp_path / "run"
    with pytest.raises(InterruptedError):
        run_synthetic(output, token, stop_after=1)
    cell = output / "candle/seed-42/attempt-1"
    (cell / "000000-patchcore.tiff").write_bytes(b"corrupt")
    with pytest.raises(EvaluationError, match="corruption"):
        run_synthetic(output, token)
    assert not (cell.parent / "attempt-2").exists()


@pytest.fixture
def intent(monkeypatch):
    import visionguard.visa_b2_gate as gate

    # Before a freeze exists locally, exercise the exact builder as identity oracle.
    monkeypatch.setattr(gate, "verify_freeze", build_freeze)
    candidate = expected_intent(REPO) | {"authorization_id": "a" * 32}
    return candidate


@pytest.mark.parametrize(
    "field",
    [
        "reviewed_implementation_commit",
        "b1_fingerprint",
        "b2_fingerprint",
        "development_freeze_sha256",
        "dataset_audit_sha256",
        "output_relative",
        "output_destination",
        "contract_sha256",
        "models",
        "scope",
        "status",
        "schema_version",
    ],
)
def test_authorization_intent_rejects_self_assertion(intent, field):
    broken = copy.deepcopy(intent)
    broken[field] = (
        True
        if field == "schema_version"
        else ({} if field in ("models", "scope") else "self-asserted")
    )
    with pytest.raises(EvaluationError):
        validate_intent(REPO, broken, free_bytes=148 * 1024**3)


def test_authorization_missing_reused_invalid_capacity_and_never_permission(
    intent, monkeypatch
):
    with pytest.raises(EvaluationError):
        validate_intent(REPO, {}, free_bytes=148 * 1024**3)
    with pytest.raises(EvaluationError, match="capacity"):
        validate_intent(REPO, intent, free_bytes=148 * 1024**3 - 1)
    import visionguard.visa_b2_gate as gate

    with monkeypatch.context() as patch:
        patch.setattr(gate, "_id_used", lambda repository, authorization_id: True)
        with pytest.raises(EvaluationError, match="Reused"):
            validate_intent(REPO, intent, free_bytes=148 * 1024**3)
    malformed = intent | {"authorization_id": "invalid"}
    with pytest.raises(EvaluationError):
        validate_intent(REPO, malformed, free_bytes=148 * 1024**3)
    assert (
        validate_intent(REPO, intent, free_bytes=148 * 1024**3)[
            "real_execution_authorized"
        ]
        is False
    )


def test_real_entrypoints_never_touch_test_root_even_matching_draft(intent):
    class ForbiddenRoot:
        def __getattribute__(self, name):
            raise AssertionError(f"Real root accessed: {name}")

    for entry in (deny_real_execution, evaluate_real):
        with pytest.raises(EvaluationError, match="CLOSED"):
            entry(repository=REPO, test_root=ForbiddenRoot(), authorization=intent)


def test_missing_corrupt_receipts_close_before_root_access(tmp_path):
    with pytest.raises(EvaluationError, match="CLOSED"):
        deny_real_execution(repository=tmp_path, test_root=object())


def test_fingerprint_changes_for_scientific_or_source_changes():
    snapshot = build_freeze(REPO)
    assert canonical_fingerprint(snapshot["document"]) == snapshot["fingerprint"]
    for key in ("protocol", "source_sha256"):
        modified = copy.deepcopy(snapshot["document"])
        if key == "protocol":
            modified[key]["runner"]["binary"] = "incorrect_ge_threshold"
        else:
            modified[key][next(iter(modified[key]))] = "0" * 64
        assert canonical_fingerprint(modified) != snapshot["fingerprint"]


def test_failed_publication_preserved_no_automatic_retry(tmp_path, monkeypatch):
    import visionguard.visa_b2_storage as storage

    samples, pc, ea, specs = fixture()
    original = storage._png
    calls = 0

    def denied(path, array):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("Synthetic WinError 5 fault")
        return original(path, array)

    monkeypatch.setattr(storage, "_png", denied)
    with pytest.raises(PermissionError):
        publish_cell(tmp_path / "failed", zip(samples, pc, ea, strict=True), specs, {})
    assert calls == 2
    assert (
        json.loads((tmp_path / "failed/failure.json").read_text())["status"] == "failed"
    )
    assert not (tmp_path / "failed/complete.json").exists()


def test_new_process_resume_skips_validated_cells(tmp_path):
    import os
    import subprocess
    import sys

    output = tmp_path / "run"
    code = (
        "from pathlib import Path; import sys; "
        "from visionguard.visa_b2_admission import make_fixture; "
        "from visionguard.visa_b2_runner import run_synthetic; "
        "root=Path(sys.argv[1]); p=make_fixture(root/'inputs'); "
        "r=run_synthetic(root/'run',p,stop_after=int(sys.argv[2]) or None); "
        "assert r['validated_skip_completed']==1; "
        "assert r['paired_cells']==36"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    first = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), "1"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode != 0 and "InterruptedError" in first.stderr
    before = (output / "candle/seed-42/attempt-1/complete.json").read_bytes()
    second = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), "0"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert (output / "candle/seed-42/attempt-1/complete.json").read_bytes() == before


def test_incomplete_post_publication_validation_not_counted(tmp_path):
    token = make_fixture(tmp_path / "inputs")
    output = tmp_path / "run"
    with pytest.raises(InterruptedError):
        run_synthetic(output, token, stop_after=1)
    (output / "candle/seed-42/attempt-1/validated.json").unlink()
    with pytest.raises(FileNotFoundError):
        run_synthetic(output, token)
    assert not (output / "candle/seed-42/attempt-2").exists()


@pytest.mark.parametrize("fault", ["nonfinite", "dimension", "seed"])
def test_invalid_predictions_never_complete(tmp_path, fault):
    from dataclasses import replace

    samples, pc, ea, specs = fixture()
    broken = replace(
        pc[0],
        **{
            "nonfinite": {"score": float("nan")},
            "dimension": {"restored_map": np.zeros((3, 2), dtype=np.float32)},
            "seed": {"spec": replace(specs[0], seed=123)},
        }[fault],
    )
    with pytest.raises(EvaluationError):
        publish_cell(
            tmp_path / "failed",
            zip(samples, [broken, *pc[1:]], ea, strict=True),
            specs,
            {},
        )
    assert not (tmp_path / "failed/complete.json").exists()


def test_failure_receipt_cannot_hide_original_error(tmp_path, monkeypatch):
    import visionguard.visa_b2_storage as storage

    original = storage.json_once

    def deny_receipt(path, payload):
        if path.name == "failure.json":
            raise OSError("Synthetic secondary receipt failure")
        return original(path, payload)

    monkeypatch.setattr(storage, "json_once", deny_receipt)
    samples, pc, ea, specs = fixture()
    broken = [__import__("dataclasses").replace(pc[0], score=float("inf")), *pc[1:]]
    with pytest.raises(EvaluationError) as error:
        publish_cell(
            tmp_path / "failed", zip(samples, broken, ea, strict=True), specs, {}
        )
    assert any(
        "receipt publication also failed" in note for note in error.value.__notes__
    )
