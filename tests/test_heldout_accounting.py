import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from visionguard.heldout_accounting import Ledger, snapshot, stamp
from visionguard.heldout_changes import decode
from visionguard.heldout_resources import Budget


def notification(name, action=3):
    raw = name.encode("utf-16-le")
    return struct.pack("<III", 0, action, len(raw)) + raw


@pytest.mark.parametrize("name", ["../escape", "C:\\escape", "\\escape", "a:b", "a\0b"])
def test_bad_notification_name_refused(name):
    with pytest.raises(ValueError):
        decode(notification(name))


@pytest.mark.parametrize(
    "raw", [b"", b"x" * 12, notification("a")[:-1], notification("a") + b"trailing"]
)
def test_bad_notification_buffer_refused(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_rename_events_and_actions():
    assert decode(notification("a\\b", 5)) == {"a/b": {5}}


def settle(ledger, expected):
    deadline = time.monotonic() + 5
    while True:
        ledger.refresh()
        if ledger.used == expected:
            return
        assert time.monotonic() < deadline, "Synthetic notification not delivered"
        time.sleep(0.01)


def test_native_unregistered_add_modify_delete_rename_and_nested(tmp_path):
    old = tmp_path / "old.bin"
    old.write_bytes(b"old")
    ledger = Ledger(tmp_path)
    try:
        assert ledger.used == 3
        old.write_bytes(b"changed-size")
        added = tmp_path / "nested" / "deeper" / "new.bin"
        added.parent.mkdir(parents=True)
        added.write_bytes(b"added")
        settle(ledger, 17)
        assert ledger.files["old.bin"][0] == 12
        old.rename(tmp_path / "renamed.bin")
        deadline = time.monotonic() + 5
        while "old.bin" in ledger.files or "renamed.bin" not in ledger.files:
            ledger.refresh()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert ledger.used == 17
        assert "old.bin" not in ledger.files and "renamed.bin" in ledger.files
        added.unlink()
        settle(ledger, 12)
        (tmp_path / "renamed.bin").unlink()
        settle(ledger, 0)
        assert ledger.files == {}
    finally:
        ledger.close()


@pytest.mark.skipif(os.name != "nt", reason="Native Windows notification coverage")
def test_reader_continues_during_foreground_work_and_is_joined(tmp_path):
    ledger = Ledger(tmp_path)
    reader = ledger.observer.worker
    try:
        # Deliberately do not poll between writes: emulate a foreground native
        # operation while the observer continuously consumes kernel events.
        for i in range(1000):
            (tmp_path / f"synthetic-{i:04d}.bin").write_bytes(b"synthetic")
        settle(ledger, 9000)
        assert len(ledger.files) == 1000 and reader.is_alive()
    finally:
        ledger.close()
    assert not reader.is_alive()


@pytest.mark.skipif(os.name != "nt", reason="Native Windows bounded queue")
def test_reader_backlog_bound_is_not_silently_discarded(tmp_path):
    ledger = Ledger(tmp_path)
    try:
        with pytest.raises(ValueError, match="backlog overflow"):
            ledger.observer._enqueue({f"manufactured-{i}": {1} for i in range(8193)})
    finally:
        ledger.close()


def test_no_double_count_registered_writes(tmp_path):
    ledger = Ledger(tmp_path)
    try:
        path = tmp_path / "new.bin"
        path.write_bytes(b"1234")
        ledger.record(path)
        settle(ledger, 4)
        ledger.refresh(full=True)
        assert ledger.used == 4
        path.write_bytes(b"12345678")
        ledger.record(path)
        ledger.refresh(full=True)
        assert ledger.used == 8
    finally:
        ledger.close()


def test_same_size_change_and_deletion_detected_portably(tmp_path):
    path = tmp_path / "same.bin"
    path.write_bytes(b"old")
    ledger = Ledger(tmp_path)
    try:
        before = ledger.files["same.bin"]
        path.write_bytes(b"new")
        os.utime(path, ns=(before[1] + 1000000, before[1] + 1000000))
        ledger.refresh(full=True)
        assert ledger.files["same.bin"] != before
        assert ledger.changes["changed"] == 1 and ledger.used == 3
        path.unlink()
        ledger.refresh(full=True)
        assert ledger.changes["deleted"] == 1 and ledger.used == 0
    finally:
        ledger.close()


def test_notification_loss_fail_closed_without_scan(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path)
    if ledger.observer is not None:
        ledger.observer.close()

    class Broken:
        def drain(self):
            raise ValueError("Output change notification overflow: human review")

        def close(self):
            pass

    ledger.observer = Broken()
    monkeypatch.setattr(
        "visionguard.heldout_accounting.snapshot",
        lambda *a, **kw: pytest.fail("Silent full-scan fallback"),
    )
    try:
        with pytest.raises(ValueError, match="overflow"):
            ledger.refresh()
    finally:
        ledger.close()


def test_existing_directory_notification_does_not_rescan(tmp_path, monkeypatch):
    (tmp_path / "stage").mkdir()
    ledger = Ledger(tmp_path)
    if ledger.observer is not None:
        ledger.observer.close()
    ledger.observer = SimpleNamespace(drain=lambda: {"stage": {3}}, close=lambda: None)
    monkeypatch.setattr(
        "visionguard.heldout_accounting.snapshot",
        lambda *a, **kw: pytest.fail("Unnecessary directory rescan"),
    )
    try:
        ledger.refresh()
    finally:
        ledger.close()


def test_equal_stamps_do_not_hide_observed_modification(tmp_path):
    (tmp_path / "same.bin").write_bytes(b"same")
    ledger = Ledger(tmp_path)
    if ledger.observer is not None:
        ledger.observer.close()
    ledger.observer = SimpleNamespace(
        drain=lambda: {"same.bin": {3}}, close=lambda: None
    )
    try:
        ledger.refresh()
        assert ledger.used == 4 and ledger.changes["changed"] == 1
    finally:
        ledger.close()


@pytest.mark.parametrize("flag", [0x400, 0x1000, 0x40000, 0x400000])
def test_cloud_reparse_attributes_fail_before_contents(tmp_path, flag):
    state = tmp_path.stat()
    with pytest.raises(ValueError, match="Reparse/cloud"):
        stamp(SimpleNamespace(st_mode=state.st_mode, st_file_attributes=flag))


def test_snapshot_unsafe_file_type():
    with pytest.raises(ValueError, match="regular"):
        stamp(SimpleNamespace(st_mode=0, st_file_attributes=0))


def test_unexpected_addition_limit_and_fresh_resume_baseline(tmp_path):
    budget = Budget(tmp_path, limit=4, reserve=0)
    try:
        (tmp_path / "unexpected.bin").write_bytes(b"12345")
        budget.refresh(full=True)
    except ValueError as error:
        assert "budget exceeded" in str(error)
        assert budget.used == 5
    else:
        pytest.fail("Unexpected addition exceeded limit silently")
    finally:
        budget.close()
    resumed = Budget(tmp_path, limit=4, reserve=0)
    try:
        assert resumed.used == 5
        with pytest.raises(ValueError, match="budget exceeded"):
            resumed.before()
    finally:
        resumed.close()


def test_final_reconciliation_independent_of_observer(tmp_path):
    ledger = Ledger(tmp_path)
    try:
        (tmp_path / "unexpected.bin").write_bytes(b"123")
        ledger.refresh(full=True)
        assert ledger.used == sum(p.stat().st_size for p in tmp_path.iterdir())
        assert ledger.files == snapshot(tmp_path)
    finally:
        ledger.close()


def test_root_replacement_refused(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path)
    monkeypatch.setattr(ledger, "identity", ("different", "root"))
    try:
        with pytest.raises(ValueError, match="root identity"):
            ledger.refresh()
    finally:
        ledger.close()


def test_expensive_check_not_immediately_due_again(tmp_path, monkeypatch):
    import visionguard.heldout_runner as runner

    # Check begins at 100, provenance costs 65, accounting 5, publication 2.
    clock = iter([100, 165, 170, 172, 173, 231])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    rows = []
    monkeypatch.setattr(runner, "json_once", lambda path, row: rows.append(row))
    calls = []
    budget = SimpleNamespace(refresh=lambda: calls.append("account"))
    last = runner._periodic_checks(
        tmp_path, budget, lambda: calls.append("verify"), {}, 0
    )
    assert last == 172
    assert runner._periodic_checks(tmp_path, budget, lambda: None, {}, last) == last
    assert runner._periodic_checks(tmp_path, budget, lambda: None, {}, last) == last
    assert calls == ["verify", "account"]
    assert rows[0]["periodic_check"]["provenance_seconds"] == 65
    assert rows[0]["periodic_check"]["accounting_seconds"] == 5


def test_failed_check_does_not_publish_success(tmp_path, monkeypatch):
    import visionguard.heldout_runner as runner

    monkeypatch.setattr(runner.time, "monotonic", lambda: 100)
    monkeypatch.setattr(runner, "json_once", lambda *a: pytest.fail("False success"))
    budget = SimpleNamespace(refresh=lambda: (_ for _ in ()).throw(ValueError("lost")))
    with pytest.raises(ValueError, match="lost"):
        runner._periodic_checks(tmp_path, budget, lambda: None, {}, 0)


def test_accounting_failure_preserves_receipt_without_allowing_normal_writes(tmp_path):
    import json

    from visionguard.heldout_resources import ACTIVE_BUDGET, before_write
    from visionguard.heldout_stages import fail

    budget = Budget(tmp_path, limit=4096, reserve=0)
    token = ACTIVE_BUDGET.set(budget)
    original = ValueError("notification overflow")
    budget.accounting_error = original
    try:
        fail(tmp_path, original, 0)
        receipt = json.loads((tmp_path / "failure.json").read_text())
        assert receipt["status"] == "failed" and receipt["reason"] == str(original)
        with pytest.raises(ValueError, match="overflow"):
            before_write(1, tmp_path / "scientific.bin")
        assert budget.evidence_path is None and budget.accounting_error is original
    finally:
        ACTIVE_BUDGET.reset(token)
        budget.close()


def test_failure_evidence_never_bypasses_retention_limit(tmp_path):
    from visionguard.heldout_resources import ACTIVE_BUDGET
    from visionguard.heldout_stages import fail

    budget = Budget(tmp_path, limit=0, reserve=0)
    token = ACTIVE_BUDGET.set(budget)
    original = ValueError("notification overflow")
    budget.accounting_error = original
    try:
        fail(tmp_path, original)
        assert not (tmp_path / "failure.json").exists()
        assert any("budget exceeded" in note for note in original.__notes__)
    finally:
        ACTIVE_BUDGET.reset(token)
        budget.close()


def test_lost_accounting_stops_runner_preserves_lock_and_failure(tmp_path, monkeypatch):
    import json

    import visionguard.heldout_runner as runner
    from visionguard.heldout_admission import make_artificial

    class LostBudget(Budget):
        calls = 0

        def before(self, additional=0):
            self.calls += 1
            if self.calls == 20 and self.evidence_path is None:
                self.accounting_error = ValueError("synthetic notification overflow")
            return super().before(additional)

    monkeypatch.setattr(runner, "Budget", LostBudget)
    permit = make_artificial(tmp_path / "inputs")
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="overflow"):
        runner.run_artificial(root, permit)
    assert (root / "writer.lock").exists() and not (root / "complete.json").exists()
    receipts = list(root.glob("invocation-*/failure.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed"
    original = receipts[0].read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_artificial(root, permit)
    assert receipts[0].read_bytes() == original


def test_scientific_calculations_and_publication_bytes_match_v3(tmp_path, monkeypatch):
    import importlib.util

    import visionguard.heldout_runner as runner
    from visionguard.heldout_admission import make_artificial
    from visionguard.visa_acquire import sha256_file

    archive = Path(__file__).resolve().parents[1] / (
        "reports/output-accounting-v4/predecessor/"
        "src__visionguard__heldout_resources.py.txt"
    )
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader("original_resources", str(archive))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    class OriginalBudget(module.Budget):
        accounting_error = None
        evidence_path = None

        def record(self, path):
            self.used += path.stat().st_size

        def refresh(self, *, full=False):
            super().refresh()

        def close(self):
            pass

        def summary(self):
            return {"retained_logical_bytes": self.used, "mode": "v3_test_adapter"}

    permit = make_artificial(tmp_path / "inputs")
    new = runner.run_artificial(tmp_path / "new", permit)
    monkeypatch.setattr(runner, "Budget", OriginalBudget)
    old = runner.run_artificial(tmp_path / "old", permit)
    assert new["cells"] == old["cells"] and new["aggregate"] == old["aggregate"]
    assert new["total_scope_calls"] == old["total_scope_calls"] == 288
    # No timing/owner metadata comparison: these are intentionally invocation-specific.
    for path in (tmp_path / "new").glob("*/seed-*/*/attempt-*/*"):
        if path.name in {"complete.json", "validated.json", "result.json"} or (
            path.suffix in {".tiff", ".png"}
        ):
            assert sha256_file(path) == sha256_file(
                tmp_path / "old" / path.relative_to(tmp_path / "new")
            )
