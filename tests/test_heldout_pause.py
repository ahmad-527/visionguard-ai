"""Manufactured per-image pause/resume equivalence and fail-closed controls."""

import copy
import json
import os
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest

from visionguard import heldout_pause as pause
from visionguard import heldout_runner as runner
from visionguard.heldout_admission import make_artificial
from visionguard.visa_acquire import sha256_file


def hook_at(target, occurrence=1):
    seen = 0

    def hook(root, nonce, label):
        nonlocal seen
        if label == target:
            seen += 1
            if seen == occurrence:
                a = pause.request(root, nonce)
                assert a == pause.request(root, nonce)  # immutable repeated request

    return hook


@pytest.mark.parametrize(
    "checkpoint,occurrence,skipped",
    [
        ("before-first-stage", 1, 0),
        ("before-image", 3, 0),
        ("after-stage", 1, 1),
        ("between-pairs", 1, 2),
    ],
)
def test_artificial_pause_resume_exact(
    tmp_path, monkeypatch, checkpoint, occurrence, skipped
):
    permit = make_artificial(tmp_path / "inputs")
    # Full-scope equivalence once, not a repeated full matrix for every checkpoint.
    baseline = (
        runner.run_artificial(tmp_path / "baseline", permit)
        if checkpoint == "before-image"
        else None
    )
    out = tmp_path / "resumed"
    stopped = runner.run_artificial(
        out, permit, checkpoint_hook=hook_at(checkpoint, occurrence)
    )
    assert stopped["scientific_completion"] is False
    assert not (out / "complete.json").exists() and not (out / "writer.lock").exists()
    nonce = stopped["invocation"]
    with pytest.raises(ValueError, match="Runner exit"):
        pause.status(out, nonce)  # still in this process: never falsely PAUSED
    # Artificial independent-process probe only; cross-process test below is native.
    monkeypatch.setattr(pause, "absent", lambda *a: True)
    assert pause.status(out, nonce)["status"] == "SAFE_PAUSED"
    before = {p: sha256_file(p) for p in out.rglob("*") if p.is_file()}
    report = runner.run_artificial(
        out,
        permit,
        resume_invocation=nonce,
        checkpoint_hook=None if baseline else hook_at("before-stage"),
    )
    if baseline:
        assert (
            report["cells"] == baseline["cells"]
            and report["aggregate"] == baseline["aggregate"]
        )
        assert report["validated_skip_stages"] == skipped
        partial = out / "candle/seed-42/patchcore/attempt-1/failure.json"
        assert json.loads(partial.read_bytes())["completed_inference_calls"] == 1
        from visionguard.visa_protocol import canonical_fingerprint

        (tmp_path / "equivalence.json").write_text(
            json.dumps(
                {
                    "evidence_class": "manufactured_only",
                    "pairs": 36,
                    "stages": 72,
                    "logical_calls": 288,
                    "additional_partial_calls": 1,
                    "scientific_cells_sha256": canonical_fingerprint(report["cells"]),
                    "scientific_aggregate_sha256": canonical_fingerprint(
                        report["aggregate"]
                    ),
                    "exact_uninterrupted_resume_metric_equality": True,
                    "preserved_prior_files": len(before),
                },
                sort_keys=True,
            )
        )
    else:
        assert report["scientific_completion"] is False
        complete_stages = [
            p
            for p in out.glob("*/seed-*/*/attempt-*/validated.json")
            if p.parent.parent.name != "pair"
        ]
        assert len(complete_stages) == skipped
    assert all(sha256_file(p) == digest for p, digest in before.items())
    # Compare validated scientific predictions/maps, not changing attempt origins.
    for original in (tmp_path / "baseline").glob("*/seed-*/*/attempt-1/complete.json"):
        if original.parent.parent.name == "pair":
            continue
        relative = original.parent.parent.relative_to(tmp_path / "baseline")
        latest = sorted((out / relative).glob("attempt-*"))[-1]
        rows_a = json.loads(original.read_bytes())["images"]
        rows_b = json.loads((latest / "complete.json").read_bytes())["images"]
        assert rows_a == rows_b


def test_restarted_invocation_and_stale_request(tmp_path, monkeypatch):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    first = runner.run_artificial(
        out, permit, checkpoint_hook=hook_at("before-image", 3)
    )
    monkeypatch.setattr(pause, "absent", lambda *a: True)
    second = runner.run_artificial(
        out,
        permit,
        resume_invocation=first["invocation"],
        checkpoint_hook=hook_at("after-stage"),
    )
    assert second["invocation"] != first["invocation"]
    with pytest.raises(ValueError, match="Later invocation"):
        pause.verify_safe(out, first["invocation"])
    final = runner.run_artificial(out, permit, resume_invocation=second["invocation"])
    assert final["model_stages"] == 72
    assert (out / "candle/seed-42/patchcore/attempt-1/failure.json").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        pause.request(out, first["invocation"])


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "fields",
        "time",
        "claim",
        "ack",
        "missing-safe",
        "lock",
        "corrupt-stage",
    ],
)
def test_bad_pause_resume_evidence_fails_closed(tmp_path, monkeypatch, fault):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    if fault in ("stale", "fields", "time", "claim"):

        def corrupt(root, nonce, label):
            if label != "before-first-stage":
                return
            pause.request(root, nonce)
            path = root / ("invocation-" + nonce) / "pause-request.json"
            value = json.loads(path.read_bytes())
            if fault == "stale":
                value["invocation"] = "0" * 32
            elif fault == "fields":
                value["unexpected"] = True
            elif fault == "time":
                value["requested_at"] = "2100-01-01T00:00:00+00:00"
            else:
                value["claim_sha256"] = "0" * 64
            path.write_text(json.dumps(value))

        with pytest.raises(ValueError):
            runner.run_artificial(out, permit, checkpoint_hook=corrupt)
        assert (out / "writer.lock").exists()
        assert not list(out.glob("invocation-*/safe-exit.json"))
        return
    stopped = runner.run_artificial(out, permit, checkpoint_hook=hook_at("after-stage"))
    directory = out / ("invocation-" + stopped["invocation"])
    monkeypatch.setattr(pause, "absent", lambda *a: True)
    if fault == "ack":
        (directory / "pause-acknowledged.json").write_text("{}")
    elif fault == "missing-safe":
        (directory / "safe-exit.json").unlink()  # manufactured corruption only
    elif fault == "lock":
        (out / "writer.lock").write_text("blocked synthetic owner")
    else:
        next((out / "candle/seed-42/patchcore/attempt-1").glob("*.tiff")).write_bytes(
            b"corrupt"
        )
    with pytest.raises((ValueError, KeyError, OSError)):
        runner.run_artificial(out, permit, resume_invocation=stopped["invocation"])


def test_gpu_cleanup_verification_only_mocked(monkeypatch):
    events = []
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        empty_cache=lambda: events.append("empty"),
        memory_allocated=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert runner.release_gpu(True) == 0 and events == ["sync", "empty"]
    cuda.memory_allocated = lambda: 1
    with pytest.raises(ValueError, match="allocations remain"):
        runner.release_gpu(True)


def test_watcher_failure_never_safe_pause(tmp_path):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"

    class BadWatcher:
        def check(self):
            pass

        def finish(self, safe):
            raise ValueError("manufactured watcher shutdown failure")

    from visionguard.heldout_admission import admit_artificial
    from visionguard.visa import CATEGORIES
    from visionguard.visa_evaluator_synthetic import fixture

    inputs = admit_artificial(permit)
    specs = {
        f"{s.model}:{c}:{seed}": s
        for c in CATEGORIES
        for seed in (42, 123, 2026)
        for s in fixture(c, seed)[3]
    }
    with pytest.raises(ValueError, match="watcher shutdown"):
        runner.run_matrix(
            inputs,
            out,
            specs,
            inputs.origin,
            runner.ArtificialBackend,
            watcher=BadWatcher(),
            checkpoint_hook=hook_at("before-first-stage"),
        )
    assert (out / "writer.lock").exists() and not list(
        out.glob("invocation-*/safe-exit.json")
    )


@pytest.mark.parametrize("fault", ["gpu", "ownership"])
def test_cleanup_fault_retains_lock_and_receipt(tmp_path, monkeypatch, fault):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    hook = hook_at("before-first-stage")
    if fault == "gpu":

        def failure(_):
            raise ValueError("manufactured GPU cleanup failure")

        monkeypatch.setattr(runner, "release_gpu", failure)
    else:
        request_hook = hook

        def changed_owner(root, nonce, label):
            request_hook(root, nonce, label)
            if label == "before-first-stage":
                (root / "writer.lock").write_text("{}")

        hook = changed_owner
    with pytest.raises(ValueError):
        runner.run_artificial(out, permit, checkpoint_hook=hook)
    assert (out / "writer.lock").exists()
    assert list(out.glob("invocation-*/cleanup-failure.json"))
    assert not list(out.glob("invocation-*/safe-exit.json"))


def test_resume_gate_before_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "admit", lambda *a: pytest.fail("Real input admission reached")
    )
    with pytest.raises(ValueError, match="CLOSED"):
        runner.evaluate_real(
            tmp_path, resume_claim="0" * 64, resume_invocation="0" * 32
        )
    permission = SimpleNamespace(
        run_root=tmp_path, authorization_id="a" * 32, claim_sha256="b" * 64
    )
    monkeypatch.setattr(runner, "authorize", lambda *a, **kw: permission)
    with pytest.raises(ValueError, match="claim AND"):
        runner.evaluate_real(tmp_path, resume_claim="0" * 64)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.evaluate_real(
            tmp_path, resume_claim="0" * 64, resume_invocation="0" * 32
        )
    (tmp_path / "watcher-stop.json").write_text("manufactured STOP")
    with pytest.raises(ValueError, match="watcher STOP"):
        runner.evaluate_real(
            tmp_path, resume_claim="0" * 64, resume_invocation="0" * 32
        )


def test_blocked_process_inspection_not_absence(monkeypatch):
    def blocked(*a, **kw):
        raise PermissionError("manufactured blocked process inspection")

    if os.name == "nt":
        monkeypatch.setattr(pause.subprocess, "check_output", blocked)
    else:
        monkeypatch.setattr(pause, "process_identity", blocked)
    with pytest.raises(PermissionError):
        pause.absent(os.getpid(), "manufactured")


def test_exact_120_hour_window_mock_trust(artificial_trust):  # noqa: F811
    # Fixture imported below from the existing authorization test module.
    from visionguard import heldout_authorization as gate

    root, _, approval = artificial_trust
    changed = copy.deepcopy(approval)
    from datetime import datetime

    changed["payload"]["expires"] = (
        datetime.fromisoformat(changed["payload"]["expires"]) + timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(ValueError, match="exactly 120"):
        gate.authorize(root, changed)
    assert not (root / "output").exists()


def test_expired_original_claim_cannot_resume(artificial_trust, monkeypatch):  # noqa: F811
    from datetime import datetime

    from visionguard import heldout_authorization as gate

    root, _, approval = artificial_trust
    permission = gate.authorize(root, approval)

    class ExpiredClock:
        fromisoformat = staticmethod(datetime.fromisoformat)

        @staticmethod
        def now(_):
            return permission.expires + timedelta(seconds=1)

    monkeypatch.setattr(gate, "datetime", ExpiredClock)
    with pytest.raises(ValueError, match="expired"):
        gate.check_permission(permission)
    with pytest.raises(ValueError, match="Expired"):
        gate.authorize(root, approval, resume_claim=permission.claim_sha256)


from test_heldout_authorization import artificial_trust  # noqa: E402,F401


def test_missing_mismatched_real_claim_output_metadata_only(tmp_path):
    root = tmp_path / "outputs/runs" / ("a" * 32)
    root.mkdir(parents=True)
    owner = {
        "artificial": False,
        "origin": {"authorization_id": "a" * 32, "claim_sha256": "b" * 64},
        "nonce": "c" * 32,
    }
    bound = pause.binding(owner, "d" * 64)
    with pytest.raises((ValueError, FileNotFoundError)):
        pause.validate_claim(root, owner, bound)
    claims = root.parent.parent / "authorization-claims"
    claims.mkdir()
    (claims / ("a" * 32 + ".json")).write_text(json.dumps({"run_root": str(root)}))
    with pytest.raises(ValueError, match="claim"):
        pause.validate_claim(root, owner, bound)


def test_cpu_subprocess_pause_and_fresh_process_resume(tmp_path):
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1]
    # Each process mints freshly manufactured, byte-identical fixture inputs.
    # No existing tree is reclassified as an artificial capability.
    # Child owns the pause request; parent independently observes actual exit.
    code = """
import json,sys
from pathlib import Path
from visionguard.heldout_admission import make_artificial
from visionguard import heldout_runner as r, heldout_pause as p
root=Path(sys.argv[1])
permit=make_artificial(root/'inputs')
def hook(out,nonce,label):
 if label=='after-stage': p.request(out,nonce)
result=r.run_artificial(root/'matrix',permit,checkpoint_hook=hook)
print(json.dumps(result)); sys.exit(75)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 75, child.stderr
    stopped = json.loads(child.stdout)
    assert (
        pause.status(tmp_path / "matrix", stopped["invocation"])["status"]
        == "SAFE_PAUSED"
    )
    code_resume = """
import json,sys
from pathlib import Path
from visionguard.heldout_admission import make_artificial
from visionguard import heldout_runner as r
root=Path(sys.argv[1]); permit=make_artificial(root/'inputs')
print(json.dumps(r.run_artificial(root/'matrix',permit,resume_invocation=sys.argv[2])))
"""
    # See admission token contract below; no real permission is constructed.
    resumed = subprocess.run(
        [sys.executable, "-c", code_resume, str(tmp_path), stopped["invocation"]],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    report = json.loads(resumed.stdout)
    assert report["model_stages"] == 72 and report["validated_skip_stages"] == 1


def test_independent_cpu_watcher_safe_exit_handshake(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    from uuid import uuid4

    from visionguard.heldout_watcher import Watcher

    root = tmp_path / "watcher"
    root.mkdir()
    owner = {
        "pid": os.getpid(),
        "nonce": uuid4().hex,
        "started": datetime.now(UTC).isoformat(),
    }
    actual_popen = subprocess.Popen

    def synthetic_guard(command, **kwargs):
        if command[1:3] != ["-m", "visionguard.heldout_watcher"]:
            return actual_popen(command, **kwargs)
        return actual_popen(
            [
                command[0],
                "-c",
                "from visionguard import heldout_watcher as w; "
                "w.windows_warnings=lambda _:[]; w.main()",
                *command[3:],
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", synthetic_guard)
    watcher = Watcher()
    child = None
    try:
        watcher.start(root, owner)
        child, directory = watcher.child, watcher.directory
        result = watcher.finish(True)
        assert result["exit_code"] == 0 and child.poll() == 0
        assert result["sha256"] == sha256_file(directory / "stopped.json")
        assert not (root / "watcher-stop.json").exists()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()  # only freshly minted synthetic CPU test child
            child.wait(timeout=5)
