"""Hosted volume selection uses measured capacity, never a synthetic reserve."""

from collections import namedtuple
from pathlib import Path

import pytest
from ci_project_environment import select_volume

Usage = namedtuple("Usage", "total used free")


def test_windows_small_workspace_uses_measured_large_volume():
    sizes = {Path("small"): 8 * 2**30, Path("large"): 50 * 2**30}
    selected, observed = select_volume(
        list(sizes), probe=lambda path: Usage(100 * 2**30, 0, sizes[path])
    )
    assert selected == Path("large")
    assert [row["free"] for row in observed] == [8 * 2**30, 50 * 2**30]


def test_no_volume_can_bypass_capacity_refusal():
    with pytest.raises(RuntimeError, match="Insufficient actual"):
        select_volume([Path("small")], probe=lambda _: Usage(30 * 2**30, 0, 10 * 2**30))
