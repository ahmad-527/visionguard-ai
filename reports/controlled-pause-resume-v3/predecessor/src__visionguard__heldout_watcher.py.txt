"""Independent CPU watcher: creation identity, drive, hardware warnings, heartbeat.

Never repairs locks, restarts computation, changes settings or accesses inputs.
Unexpected owner disappearance emits immutable stop evidence; guard failure itself
is detected by runner.poll(). A watcher cannot rescue a broken output volume.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from visionguard.heldout_resources import Budget, windows_warnings
from visionguard.heldout_storage import json_once
from visionguard.visa_evaluator import require


def process_identity(pid):
    if os.name == "nt":
        script = (
            "$p=Get-Process -Id "
            + str(int(pid))
            + " -ErrorAction Stop; $p.StartTime.ToUniversalTime().Ticks"
        )
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


class Watcher:
    def start(self, root, owner):
        self.root = root
        self.directory = root / f"watcher-{owner['nonce']}"
        self.directory.mkdir()
        identity = process_identity(owner["pid"])
        with (
            (self.directory / "stdout.log").open("xb") as out,
            (self.directory / "stderr.log").open("xb") as err,
        ):
            kwargs = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt"
                else {}
            )
            self.child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "visionguard.heldout_watcher",
                    "--root",
                    str(root),
                    "--directory",
                    str(self.directory),
                    "--owner-pid",
                    str(owner["pid"]),
                    "--owner-identity",
                    identity,
                    "--started",
                    owner["started"],
                ],
                stdout=out,
                stderr=err,
                **kwargs,
            )
        deadline = time.monotonic() + 30
        while (
            not (self.directory / "ready.json").exists() and time.monotonic() < deadline
        ):
            self.check()
            time.sleep(0.1)
        require((self.directory / "ready.json").exists(), "Watcher readiness timeout")

    def check(self):
        require(
            self.child.poll() is None
            and not (self.root / "watcher-stop.json").exists(),
            "Independent watcher failed/requested STOP",
        )

    def finish(self, safe):
        if not hasattr(self, "child"):
            return
        if self.child.poll() is None:
            json_once(self.directory / "finish.json", {"safe_owner_exit": safe})
            self.child.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "directory"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--owner-pid", type=int, required=True)
    p.add_argument("--owner-identity", required=True)
    p.add_argument("--started", required=True)
    args = p.parse_args()
    budget = Budget(args.root)
    try:
        require(
            process_identity(args.owner_pid) == args.owner_identity,
            "Owner creation mismatch",
        )
        json_once(
            args.directory / "ready.json",
            {"owner_pid": args.owner_pid, "owner_identity": args.owner_identity},
        )
        last = 0
        while not (args.directory / "finish.json").exists():
            require(
                process_identity(args.owner_pid) == args.owner_identity,
                "Unexpected owner disappearance/identity change",
            )
            if time.monotonic() - last >= 60:
                budget.refresh()
                budget.before()
                events = windows_warnings(args.started)
                require(
                    not events, "Hardware/filesystem warning: human review required"
                )
                json_once(
                    args.directory / f"heartbeat-{uuid4().hex}.json",
                    {
                        "monotonic": time.monotonic(),
                        "events": events,
                        "owner_identity": args.owner_identity,
                    },
                )
                last = time.monotonic()
            time.sleep(1)
    except BaseException as exc:
        json_once(
            args.root / "watcher-stop.json",
            {
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "actual_exit_code": "UNKNOWN if owner disappeared",
                "owner_pid": args.owner_pid,
                "owner_identity": args.owner_identity,
            },
        )
        raise


if __name__ == "__main__":
    main()
