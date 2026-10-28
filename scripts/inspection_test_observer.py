"""Opt-in fixture-test observer; no native periodic dumps, frame locals or models.

The design reuses the prior reviewed Python-stack observer approach. Snapshots
are built in a separate call so truncated frames cannot stay owned by the
observer's sleeping thread. Fatal crash reporting remains pytest's responsibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path


def python_stacks(max_threads=64, max_depth=80):
    """Return bounded code locations only, releasing all captured frame owners."""
    current = sys._current_frames()
    frame = None
    try:
        result = []
        for ident, frame in list(current.items())[:max_threads]:
            stack = []
            for _ in range(max_depth):
                if frame is None:
                    break
                stack.append(
                    {
                        "file": frame.f_code.co_filename,
                        "function": frame.f_code.co_name,
                        "line": frame.f_lineno,
                    }
                )
                frame = frame.f_back
            result.append({"thread_ident": ident, "stack": stack})
        return result
    finally:
        frame = None
        current.clear()


class StackObserver:
    """Write bounded, append-only snapshots; never keep a frame between ticks."""

    def __init__(self, path: Path, interval=5.0, limit_bytes=8 * 1024**2):
        if interval <= 0 or limit_bytes <= 0:
            raise ValueError("Positive observer bounds required")
        self.path = path
        self.interval = interval
        self.limit = limit_bytes
        self.stop = threading.Event()
        self.failures = []
        self.snapshots = 0
        self.stream = path.open("xb", buffering=0)
        self.thread = threading.Thread(
            target=self._run, name="inspection-test-stack-observer", daemon=True
        )
        self.thread.start()

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                raw = (
                    json.dumps(
                        {
                            "utc": datetime.now(UTC).isoformat(),
                            "python_stacks_only": True,
                            "frame_locals_read": False,
                            "threads": python_stacks(),
                        }
                    )
                    + "\n"
                ).encode()
                if self.stream.tell() + len(raw) > self.limit:
                    self.failures.append("Stack journal cap reached")
                    return
                self.stream.write(raw)
                os.fsync(self.stream.fileno())
                self.snapshots += 1
            except Exception as error:
                self.failures.append(type(error).__name__ + ": " + str(error))
                return

    def close(self):
        self.stop.set()
        self.thread.join(2)
        stopped = not self.thread.is_alive()
        if stopped:
            self.stream.close()
        return {
            "snapshots": self.snapshots,
            "thread_stopped": stopped,
            "failures": self.failures.copy(),
            "frame_locals_read": False,
        }


class TestObserver:
    """Keep test outcomes durable even when a later native process fails."""

    __test__ = False

    def __init__(self, output: Path):
        self.output = output
        self.journal = (output / "test-events.jsonl").open("xb", buffering=0)
        self.stack = None

    def emit(self, value):
        self.journal.write((json.dumps(value) + "\n").encode())
        os.fsync(self.journal.fileno())

    def pytest_sessionstart(self, session):
        self.stack = StackObserver(self.output / "python-stacks.jsonl")
        self.emit({"event": "session_start", "periodic_native_dumps": False})

    def pytest_runtest_logreport(self, report):
        self.emit(
            {
                "event": "test_report",
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "duration_seconds": report.duration,
                "longrepr": str(report.longrepr) if report.failed else None,
            }
        )

    def pytest_sessionfinish(self, session, exitstatus):
        state = self.stack.close()
        self.emit({"event": "session_finish", "exitstatus": int(exitstatus), **state})
        self.journal.close()
        if not state["thread_stopped"] or state["failures"]:
            session.exitstatus = 3


def main(argv=None):
    import pytest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments, tests = parser.parse_known_args(argv)
    if tests[:1] == ["--"]:
        tests = tests[1:]
    arguments.output.mkdir(parents=True, exist_ok=False)
    return pytest.main(
        [*tests, "-o", "faulthandler_timeout=0"],
        plugins=[TestObserver(arguments.output)],
    )


if __name__ == "__main__":
    raise SystemExit(main())
