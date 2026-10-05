"""Metadata-only retained-byte ledger, independently owned by runner and watcher.

Windows: initial/final reconciliation plus bounded change notifications. Other
hosts: linear, safe full reconciliation (portable synthetic/CI fallback). Neither
mode reads file contents or treats accounting as scientific hash verification.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from visionguard.heldout_changes import WindowsChanges
from visionguard.heldout_paths import checked
from visionguard.visa_evaluator import require


def stamp(state):
    require(
        not stat.S_ISLNK(state.st_mode)
        and not (
            getattr(state, "st_file_attributes", 0)
            & (0x400 | 0x1000 | 0x40000 | 0x400000)
        ),
        "Reparse/cloud path refused",
    )
    require(
        stat.S_ISREG(state.st_mode) or stat.S_ISDIR(state.st_mode),
        "Not regular file/directory",
    )
    return (
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
        state.st_dev,
        state.st_ino,
    )


def snapshot(root: Path, subtree: Path | None = None, directories=None) -> dict:
    """Validate ancestors once per directory, not once per file; no follow links."""
    result = {}

    def visit(directory):
        checked(directory)
        if directories is not None:
            directories.add(directory.relative_to(root).as_posix())
        identity = stamp(directory.lstat())[3:]
        with os.scandir(directory) as entries:
            for entry in entries:
                state = entry.stat(follow_symlinks=False)
                value = stamp(state)
                path = Path(entry.path)
                if stat.S_ISDIR(state.st_mode):
                    visit(path)
                else:
                    result[path.relative_to(root).as_posix()] = value
        require(
            stamp(directory.lstat())[3:] == identity,
            "Output directory identity changed during accounting",
        )
        checked(directory)

    visit(subtree or root)
    return result


class Ledger:
    def __init__(self, root: Path):
        checked(root)
        self.root = root
        self.identity = stamp(root.lstat())[3:]
        # Arm before baseline so no notification gap exists across its scan.
        self.observer = WindowsChanges(root) if os.name == "nt" else None
        self.files = {}
        self.directories = set()
        self.used = 0
        self.changes = {"added": 0, "changed": 0, "deleted": 0}
        self.total_changes = self.changes.copy()
        try:
            self.reconcile()
            self.total_changes = {"added": 0, "changed": 0, "deleted": 0}
        except BaseException:
            self.close()
            raise

    def _replace(self, name, value):
        old = self.files.get(name)
        if old is not None:
            self.used -= old[0]
        if value is None:
            self.files.pop(name, None)
        else:
            self.files[name] = value
            self.used += value[0]
        if old != value:
            self._event(
                "added" if old is None else "deleted" if value is None else "changed"
            )

    def _event(self, kind):
        self.changes[kind] += 1
        self.total_changes[kind] += 1

    def _guard(self):
        checked(self.root)
        require(
            stamp(self.root.lstat())[3:] == self.identity,
            "Output root identity changed",
        )

    def reconcile(self):
        self._guard()
        self.directories = set()
        current = snapshot(self.root, directories=self.directories)
        for name in self.files.keys() - current.keys():
            self._replace(name, None)
        for name, value in current.items():
            self._replace(name, value)
        self._guard()

    def refresh(self, *, full=False):
        self._guard()
        self.changes = {"added": 0, "changed": 0, "deleted": 0}
        if full or self.observer is None:
            self.reconcile()
            return
        names = self.observer.drain()
        # Directory changes affect their whole subtree, not unrelated stages.
        covered = []
        for name in sorted(names, key=lambda p: (p.count("/"), p)):
            if any(name.startswith(prefix + "/") for prefix in covered):
                continue
            path = self.root / name
            checked(path, missing=True)
            try:
                state = path.lstat()
            except FileNotFoundError:
                self.directories = {
                    old
                    for old in self.directories
                    if old != name and not old.startswith(name + "/")
                }
                for old in tuple(self.files):
                    if old == name or old.startswith(name + "/"):
                        self._replace(old, None)
                continue
            value = stamp(state)
            if stat.S_ISDIR(state.st_mode):
                # Directory LastWrite changes carry no retained bytes. Child
                # changes have their own notifications; do not scan the stage
                # again merely because another image was durably published.
                if name in self.directories and names[name] <= {3}:
                    continue
                current = snapshot(self.root, path, self.directories)
                covered.append(name)
                for old in tuple(self.files):
                    if old.startswith(name + "/") and old not in current:
                        self._replace(old, None)
                for relative, item in current.items():
                    self._replace(relative, item)
            else:
                old = self.files.get(name)
                self._replace(name, value)
                # A same-size write with restored metadata is still an observed
                # mutation. Do not mislabel identical stamps as no notification.
                if old == value and 3 in names[name]:
                    self._event("changed")
        self._guard()

    def record(self, path: Path):
        """Replace earlier observations with final size; never double count."""
        checked(path)
        relative = path.relative_to(self.root).as_posix()
        state = path.lstat()
        require(stat.S_ISREG(state.st_mode), "Written output is not a regular file")
        self._replace(relative, stamp(state))

    def close(self):
        if self.observer is not None:
            self.observer.close()
