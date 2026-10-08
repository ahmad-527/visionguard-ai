"""Manufactured allocation/reference probes; no original data or checkpoints."""

import gc
import json
import os
import subprocess
import sys
import weakref
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from visionguard import heldout_cleanup as cleanup
from visionguard import heldout_runner as runner
from visionguard.heldout_admission import admit_artificial, make_artificial
from visionguard.heldout_backend import NativeBackend
from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator_synthetic import fixture


class ManufacturedAllocation:
    """Weakly observed Python ownership, not a measurement of native GPU use."""

    live = weakref.WeakSet()

    def __init__(self, size):
        self.size = size
        self.live.add(self)


class ManufacturedCuda:
    def __init__(self):
        self.events = []

    def memory_allocated(self):
        return sum(allocation.size for allocation in ManufacturedAllocation.live)

    def memory_reserved(self):
        return 8192

    def current_device(self):
        return 0

    def synchronize(self):
        self.events.append("synchronize")

    def empty_cache(self):
        self.events.append("empty_cache")


@pytest.fixture
def manufactured_cuda(monkeypatch):
    gc.collect()
    assert not list(ManufacturedAllocation.live)
    cuda = ManufacturedCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return cuda


class ManufacturedModel:
    def __init__(self):
        self.tensor = ManufacturedAllocation(4096)
        self.cycle = self

    def eval(self):
        return self

    def requires_grad_(self, _):
        return self


def backend():
    return NativeBackend(None, ManufacturedModel(), None, "manufactured_cuda")


def test_backend_wrapper_release_collects_model_cycle(manufactured_cuda):
    native = backend()
    model_ref, tensor_ref = weakref.ref(native.model), weakref.ref(native.model.tensor)
    assert manufactured_cuda.memory_allocated() == 4096
    del native
    assert cleanup.release_gpu(True) == 0
    assert model_ref() is None and tensor_ref() is None
    assert manufactured_cuda.events == ["synchronize", "empty_cache"]


def test_borrowed_tensor_reference_keeps_guard_closed(manufactured_cuda):
    native = backend()
    borrowed = native.model.tensor
    del native
    with pytest.raises(ValueError, match="allocations remain") as failure:
        cleanup.release_gpu(True)
    diagnostic = failure.value.gpu_cleanup_diagnostics
    assert diagnostic["allocated_bytes_after_empty_cache"] == 4096
    assert diagnostic["reserved_bytes_after_empty_cache"] == 8192
    assert diagnostic["phase"] == "allocation_zero_check"
    del borrowed
    assert cleanup.release_gpu(True) == 0


def test_exception_traceback_keeps_backend_until_detached(manufactured_cuda):
    references = []

    def native_failure():
        native = backend()
        references.append(weakref.ref(native.model))
        raise ValueError("manufactured backend exception")

    try:
        native_failure()
    except ValueError as error:
        retained = error
    with pytest.raises(ValueError, match="allocations remain"):
        cleanup.release_gpu(True)
    assert references[0]() is not None
    retained.__traceback__ = None
    retained.__context__ = None
    retained.__cause__ = None
    assert cleanup.release_gpu(True) == 0
    assert references[0]() is None


def test_reserved_cache_is_not_live_allocation_failure(manufactured_cuda):
    assert manufactured_cuda.memory_reserved() > 0
    assert cleanup.release_gpu(True) == 0


@pytest.mark.parametrize("phase", ["cuda_synchronize", "cuda_empty_cache"])
def test_original_cleanup_exception_and_phase_preserved(
    manufactured_cuda, monkeypatch, phase
):
    original = RuntimeError("manufactured native cleanup API error")

    def fail():
        raise original

    method = "synchronize" if phase == "cuda_synchronize" else "empty_cache"
    monkeypatch.setattr(manufactured_cuda, method, fail)
    with pytest.raises(RuntimeError) as failure:
        cleanup.release_gpu(True)
    assert failure.value is original
    assert failure.value.gpu_cleanup_diagnostics["phase"] == phase
    assert failure.value.gpu_cleanup_diagnostics["allocated_bytes_after_gc"] == 0


@pytest.mark.parametrize("watcher_fails", [False, True])
def test_post_publication_gpu_failure_truthful_handshake_and_bound_receipt(
    tmp_path, monkeypatch, manufactured_cuda, watcher_fails
):
    retained = ManufacturedAllocation(4096)
    assert retained.size == 4096
    permit = make_artificial(tmp_path / "manufactured-inputs")
    inputs = admit_artificial(permit)
    specs = {
        f"{spec.model}:{category}:{seed}": spec
        for category in CATEGORIES
        for seed in (42, 123, 2026)
        for spec in fixture(category, seed)[3]
    }
    monkeypatch.setattr(runner, "release_gpu", lambda _: cleanup.release_gpu(True))
    calls = []

    class ManufacturedWatcher:
        def check(self):
            pass

        def finish(self, safe):
            calls.append(safe)
            assert safe is False
            if watcher_fails:
                raise RuntimeError("manufactured secondary watcher failure")
            return {"exit_code": 0, "sha256": "a" * 64}

    output = tmp_path / "manufactured-output"
    with pytest.raises(ValueError, match="allocations remain"):
        runner.run_matrix(
            inputs,
            output,
            specs,
            inputs.origin,
            runner.ArtificialBackend,
            watcher=ManufacturedWatcher(),
        )
    pointer = json.loads((output / "complete.json").read_bytes())
    invocation = output / pointer["invocation"]
    failure = json.loads((invocation / "cleanup-failure.json").read_bytes())
    assert calls == [False]
    assert failure["cleanup_phase"] == "gpu_release"
    assert failure["completion_pointer_for_this_invocation_published"] is True
    assert failure["safe_exit_unproven"] is True
    assert datetime.fromisoformat(failure["recorded_at_utc"]).utcoffset() is not None
    assert failure["writer_lock_released_by_cleanup"] is False
    assert failure["gpu_cleanup"]["allocated_bytes_after_empty_cache"] == 4096
    assert failure["binding"]["invocation"] == invocation.name[11:]
    assert failure["binding"]["owner_sha256"] == sha256_file(invocation / "owner.json")
    assert failure["watcher_shutdown"]["safe_owner_exit"] is False
    assert (failure["watcher_shutdown"]["secondary_error"] is not None) is watcher_fails
    assert (failure["watcher_shutdown"]["receipt"] is None) is watcher_fails
    assert (output / "writer.lock").read_bytes() == (
        invocation / "owner.json"
    ).read_bytes()
    assert not (invocation / "safe-exit.json").exists()
    assert not (invocation / "failure.json").exists()
    assert len(list(invocation.glob("model-*.json"))) == 72
    assert len(list(invocation.glob("pair-*.json"))) == 36


def test_new_diagnostics_and_tests_are_in_future_source_freeze():
    from visionguard.cleanup_successor import ADDITIONAL

    assert "src/visionguard/heldout_cleanup.py" in ADDITIONAL
    assert "tests/test_heldout_cleanup.py" in ADDITIONAL
    assert "scripts/probe_heldout_cleanup.py" in ADDITIONAL
    assert "scripts/run_shutdown_review.py" in ADDITIONAL


def test_independent_cpu_watcher_reports_unsafe_owner_exit(tmp_path, monkeypatch):
    """Fresh synthetic CPU child truthfully acknowledges an unsafe owner exit."""
    from visionguard.heldout_watcher import Watcher

    root = tmp_path / "manufactured-watcher"
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
        result = watcher.finish(False)
        assert result["exit_code"] == 0 and child.poll() == 0
        assert result["sha256"] == sha256_file(directory / "stopped.json")
        stopped = json.loads((directory / "stopped.json").read_bytes())
        assert stopped["safe_owner_exit"] is False
        assert not (root / "watcher-stop.json").exists()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()  # only this freshly manufactured synthetic test child
            child.wait(timeout=5)


@pytest.mark.parametrize("publication_error", [OSError, ValueError])
def test_cleanup_receipt_refusal_does_not_mask_primary_failure(
    tmp_path, monkeypatch, publication_error
):
    """Manufactured interruption and publishing refusal; no native/data access."""
    primary = RuntimeError("Manufactured primary GPU cleanup failure")
    original_publish = runner.json_once
    calls = []

    def refuse_receipt(path, value):
        if path.name == "cleanup-failure.json":
            raise publication_error("Manufactured receipt publication refusal")
        return original_publish(path, value)

    def fail_cleanup(_):
        raise primary

    class ManufacturedWatcher:
        def check(self):
            pass

        def finish(self, safe):
            calls.append(safe)
            return {"exit_code": 0, "sha256": "a" * 64}

    permit = make_artificial(tmp_path / "manufactured-inputs")
    inputs = admit_artificial(permit)
    specs = {
        f"{spec.model}:{category}:{seed}": spec
        for category in CATEGORIES
        for seed in (42, 123, 2026)
        for spec in fixture(category, seed)[3]
    }
    monkeypatch.setattr(runner, "json_once", refuse_receipt)
    monkeypatch.setattr(runner, "release_gpu", fail_cleanup)
    output = tmp_path / "manufactured-output"
    with pytest.raises(RuntimeError) as failure:
        runner.run_matrix(
            inputs,
            output,
            specs,
            inputs.origin,
            runner.ArtificialBackend,
            interrupt=("candle", 42, "patchcore"),
            watcher=ManufacturedWatcher(),
        )
    assert failure.value is primary
    assert calls == [False]
    assert any("receipt also failed" in note for note in primary.__notes__)
    assert any("publication refusal" in note for note in primary.__notes__)
    invocation = next(output.glob("invocation-*"))
    assert (output / "writer.lock").exists()
    assert (invocation / "failure.json").exists()
    assert not (invocation / "cleanup-failure.json").exists()
    assert not (invocation / "safe-exit.json").exists()
    assert not (output / "complete.json").exists()
