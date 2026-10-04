"""No aliases/cloud hydration; check only authorized or artificial trees."""

import os
from pathlib import Path, PurePosixPath

from visionguard.visa_evaluator import require


def checked(path: Path, *, missing: bool = False) -> Path:
    require(path.is_absolute(), "Absolute local path required")
    require(".." not in path.parts, "Traversal refused")
    for part in reversed((path, *path.parents)):
        try:
            state = part.lstat()
        except FileNotFoundError:
            require(missing, "Missing asset")
            continue
        require(
            not part.is_symlink()
            and not (
                getattr(state, "st_file_attributes", 0)
                & (0x400 | 0x1000 | 0x40000 | 0x400000)
            ),
            "Reparse/cloud path refused",
        )
    return path


def asset(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    require(
        not p.is_absolute()
        and str(p) == relative
        and not set(p.parts) & {".", ".."}
        and ":" not in relative
        and "\\" not in relative,
        "Unsafe relative path",
    )
    return checked(root.joinpath(*p.parts))


def inventory(root: Path) -> set[str]:
    checked(root)
    result = set()

    def visit(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                path = checked(Path(entry.path))
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                else:
                    require(entry.is_file(follow_symlinks=False), "Not regular file")
                    result.add(path.relative_to(root).as_posix())

    visit(root)
    return result
