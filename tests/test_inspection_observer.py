"""Untrained reference-lifetime and durable-observer regressions."""

import gc
import json
import threading
import weakref

import pytest
from scripts.inspection_test_observer import StackObserver, python_stacks


def test_truncated_stack_snapshot_does_not_hold_finished_worker_allocations():
    ready, finish = threading.Event(), threading.Event()
    refs = []

    class Allocation:
        pass

    def worker():
        owned = Allocation()
        refs.append(weakref.ref(owned))
        ready.set()
        finish.wait(5)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(5)
        stacks = python_stacks(max_depth=1)
        assert stacks and all(len(x["stack"]) <= 1 for x in stacks)
        assert "locals" not in json.dumps(stacks)
    finally:
        finish.set()
        thread.join(5)
    gc.collect()
    assert not thread.is_alive() and refs[0]() is None


def test_observer_writes_stops_and_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "stacks.jsonl"
    observer = StackObserver(path, interval=0.01)
    # Wait for a snapshot through the journal, without selecting process timings.
    for _ in range(100):
        if observer.snapshots:
            break
        threading.Event().wait(0.01)
    status = observer.close()
    assert status["thread_stopped"] and not status["failures"]
    assert status["snapshots"] >= 1 and status["frame_locals_read"] is False
    assert all(
        json.loads(line)["frame_locals_read"] is False
        for line in path.read_text().splitlines()
    )
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        StackObserver(path)
    assert path.read_bytes() == before


def test_cap_failure_is_explicit_and_thread_is_stopped(tmp_path):
    observer = StackObserver(tmp_path / "small.jsonl", interval=0.01, limit_bytes=1)
    observer.thread.join(3)
    status = observer.close()
    assert status["thread_stopped"] and status["snapshots"] == 0
    assert status["failures"] == ["Stack journal cap reached"]
