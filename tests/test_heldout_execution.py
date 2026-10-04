import json

import pytest

from visionguard.heldout_admission import make_artificial
from visionguard.heldout_resources import Budget
from visionguard.heldout_runner import run_artificial
from visionguard.visa_acquire import sha256_file


def test_complete_matrix_restart_exact(tmp_path):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    a = run_artificial(out, permit)
    assert a["paired_cells"] == 36 and a["model_stages"] == 72
    assert a["new_inference_calls"] == a["total_scope_calls"] == 288
    b = run_artificial(out, permit)
    assert b["new_inference_calls"] == 0 and b["validated_skip_stages"] == 72
    assert a["cells"] == b["cells"] and a["aggregate"] == b["aggregate"]
    assert not a["real_test_access"]


def test_mid_stage_interruption_preserved_and_resume(tmp_path):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    with pytest.raises(InterruptedError):
        run_artificial(out, permit, interrupt=("candle", 42, "efficientad"))
    failure = out / "candle/seed-42/efficientad/attempt-1/failure.json"
    old = sha256_file(failure)
    pc = sha256_file(out / "candle/seed-42/patchcore/attempt-1/complete.json")
    report = run_artificial(out, permit)
    assert report["validated_skip_stages"] == 1 and report["new_inference_calls"] == 284
    assert (
        sha256_file(failure) == old
        and sha256_file(out / "candle/seed-42/patchcore/attempt-1/complete.json") == pc
    )
    origin = json.loads(
        (out / "candle/seed-42/efficientad/attempt-2/origin.json").read_text()
    )
    assert origin["source"]["parent_failure_sha256"] == old


def test_multiple_writer_refusal(tmp_path):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    out.mkdir()
    (out / "writer.lock").write_text("existing owner")
    with pytest.raises(FileExistsError):
        run_artificial(out, permit)
    assert (out / "writer.lock").read_text() == "existing owner"


@pytest.mark.parametrize(
    "fault", ["map", "binary", "completion", "unclassified", "failed"]
)
def test_no_silent_recovery_bad_evidence(tmp_path, fault):
    permit = make_artificial(tmp_path / "inputs")
    out = tmp_path / "matrix"
    with pytest.raises(InterruptedError):
        run_artificial(out, permit, interrupt=("candle", 42, "efficientad"))
    pc = out / "candle/seed-42/patchcore/attempt-1"
    ea = out / "candle/seed-42/efficientad/attempt-1"
    if fault == "map":
        next(pc.glob("*.tiff")).write_bytes(b"corrupt")
    elif fault == "binary":
        next(pc.glob("*-binary.png")).write_bytes(b"corrupt")
    elif fault == "completion":
        (pc / "complete.json").write_text("{}")
    elif fault == "unclassified":
        (ea / "failure.json").unlink()
    else:
        (ea / "failure.json").write_text('{"status":"failed"}')
    with pytest.raises((ValueError, OSError, KeyError)):
        run_artificial(out, permit)
    assert (out / "writer.lock").exists()


def test_budget_counts_existing_failed_attempts(tmp_path):
    (tmp_path / "retained-failure.bin").write_bytes(b"x" * 1024)
    budget = Budget(tmp_path, limit=1024, reserve=0)
    budget.before()
    with pytest.raises(ValueError):
        budget.before(1)


def test_volume_replacement_stops_without_retry(tmp_path, monkeypatch):
    import visionguard.heldout_resources as resources

    budget = Budget(tmp_path, reserve=0)
    monkeypatch.setattr(resources, "volume_identity", lambda *a: "different-volume")
    with pytest.raises(ValueError, match="changed/disconnected"):
        budget.before()


def test_mutating_artificial_native_backend_never_becomes_skippable(
    tmp_path, monkeypatch
):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    import visionguard.heldout_runner as runner
    from visionguard.heldout_admission import admit_artificial
    from visionguard.visa import CATEGORIES
    from visionguard.visa_evaluator_synthetic import fixture

    permit = make_artificial(tmp_path / "inputs")
    inputs = admit_artificial(permit)
    inputs = replace(
        inputs, artificial=False
    )  # Manufactured fixture, mock native route ONLY.
    monkeypatch.setattr(runner, "check_permission", lambda p: None)
    specs = {
        f"{s.model}:{c}:{seed}": s
        for c in CATEGORIES
        for seed in (42, 123, 2026)
        for s in fixture(c, seed)[3]
    }

    class Mutating(runner.ArtificialBackend):
        def __init__(self, spec):
            super().__init__(spec)
            self.calls = 0

        def state_digest(self):
            return str(self.calls)

        def predict(self, frame):
            self.calls += 1
            return super().predict(frame)

    class ArtificialWatcher:
        def start(self, *args):
            pass

        def check(self):
            pass

        def finish(self, *args):
            pass

    monkeypatch.setattr(runner, "monitor", lambda *args: {"artificial_only": True})
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda: None, empty_cache=lambda: None
            )
        ),
    )
    output = tmp_path / "matrix"
    with pytest.raises(ValueError, match="state mutated"):
        runner.run_matrix(
            inputs,
            output,
            specs,
            inputs.origin,
            Mutating,
            real=True,
            watcher=ArtificialWatcher(),
            permission="ARTIFICIAL MOCK ONLY",
        )
    stage = output / "candle/seed-42/patchcore/attempt-1"
    assert (stage / "failure.json").exists() and (output / "writer.lock").exists()


def test_disconnected_drive_no_infinite_parent_walk(tmp_path, monkeypatch):
    monkeypatch.setattr(type(tmp_path), "exists", lambda self: False)
    with pytest.raises(ValueError, match="Disconnected output drive"):
        Budget(tmp_path)


def test_reparse_stat_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from visionguard.heldout_paths import checked

    actual = type(tmp_path).lstat
    monkeypatch.setattr(
        type(tmp_path),
        "lstat",
        lambda p: (
            SimpleNamespace(st_file_attributes=0x400, st_mode=actual(p).st_mode)
            if p == tmp_path
            else actual(p)
        ),
    )
    with pytest.raises(ValueError, match="Reparse/cloud"):
        checked(tmp_path)


def test_watcher_identity_and_disappearance(tmp_path):
    import os

    from visionguard.heldout_watcher import process_identity

    assert process_identity(os.getpid()) == process_identity(os.getpid())
    import subprocess

    with pytest.raises((OSError, ValueError, subprocess.CalledProcessError)):
        process_identity(2**30)


def test_native_binary_before_float16_rounding(tmp_path):
    import numpy as np

    from visionguard.heldout_admission import Frame
    from visionguard.heldout_metrics import Prediction, Sample
    from visionguard.heldout_stages import publish_stage, read_stage
    from visionguard.visa_evaluator_synthetic import fixture

    spec = fixture()[3][0]
    identifier = "candle/Data/Images/Anomaly/artificial-rounding.png"
    sample = Sample(identifier, "candle", 1, np.array([[1, 0], [0, 0]], dtype=np.uint8))

    class Backend:
        def predict(self, frame):
            return Prediction(
                frame.sample_id,
                spec,
                2.0,
                np.array([[2.0001, 0], [0, 0]], dtype=np.float32),
            )

    origin = {"membership": {"candle": [identifier]}}
    receipt = publish_stage(
        tmp_path / "stage",
        [Frame(sample, np.zeros((2, 2, 3), np.uint8))],
        spec,
        origin,
        Backend(),
    )
    _, prediction = next(
        read_stage(tmp_path / "stage", receipt["sha256"], spec, origin)
    )
    assert prediction.restored_map[0, 0] == 2
    assert prediction.thresholded_map[0, 0]


def test_independent_watcher_records_artificial_owner_disappearance(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from datetime import UTC, datetime

    from visionguard.heldout_watcher import process_identity

    root = tmp_path / "watcher-test"
    root.mkdir()
    directory = root / "guard"
    directory.mkdir()
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    guard = None
    try:
        identity = process_identity(owner.pid)
        # Synthetic fault injection: no real event-log access in this unit test.
        guard = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from visionguard import heldout_watcher as w; "
                "w.windows_warnings=lambda _:[]; w.main()",
                "--root",
                str(root),
                "--directory",
                str(directory),
                "--owner-pid",
                str(owner.pid),
                "--owner-identity",
                identity,
                "--started",
                datetime.now(UTC).isoformat(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        deadline = time.monotonic() + 20
        while (
            not (directory / "ready.json").exists()
            and guard.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        assert (directory / "ready.json").exists()
        owner.terminate()
        owner.wait(timeout=5)  # ONLY freshly minted idle test process.
        assert guard.wait(timeout=15) != 0
        evidence = json.loads((root / "watcher-stop.json").read_text())
        assert (
            evidence["owner_pid"] == owner.pid
            and evidence["owner_identity"] == identity
        )
    finally:
        for child in (owner, guard):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
