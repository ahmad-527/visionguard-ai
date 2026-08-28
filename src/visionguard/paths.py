"""Host-independent validation for portable repository-relative paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def portable_relative_path(value: str) -> Path:
    """Return an OS-native path after validating both POSIX and Windows forms.

    A path is portable and relative only when neither path grammar gives it an
    absolute/rooted anchor and neither grammar contains a parent-traversal
    component. Windows parts are used for the returned path so either slash
    convention has identical semantics on every host.
    """

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.anchor:
        raise ValueError("path must not be absolute or rooted")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("path must not contain parent traversal")
    return Path(*windows_path.parts)
