"""Materialise the explicit 0.2.0 source-only recipe outside historical sources."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Successor output must be new; preserve previous artifacts")
    destination.mkdir(parents=True)
    for prefix, package in [
        ("", "visionguard"),
        ("apps/inspection/", "visionguard_inspection"),
    ]:
        target = destination / prefix
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            ROOT / prefix / "src" / package,
            target / "src" / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in ["README.md", "LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md"]:
            shutil.copyfile(
                ROOT / (prefix + "README.md" if name == "README.md" else name),
                target / name,
            )
        shutil.copytree(
            ROOT / "metadata/license-texts", target / "metadata/license-texts"
        )
        shutil.copytree(
            ROOT / "supplemental-license-texts", target / "supplemental-license-texts"
        )
        recipe = "core" if not prefix else "app"
        shutil.copyfile(
            ROOT / f"packaging/inspection-v2/{recipe}-pyproject.toml",
            target / "pyproject.toml",
        )
    # This already reviewed presentation-only correction is confined to the build
    # derivative; executable scientific code and historical source stay intact.
    p = destination / "src/visionguard/efficientad.py"
    raw = p.read_bytes().replace(b"VisionGuard-owned", b"VisionGuard integration")
    p.write_bytes(raw)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--dist", type=Path)
    a = p.parse_args()
    source = prepare(a.source)
    if a.dist:
        a.dist.mkdir(parents=True, exist_ok=False)
        for path in [source, source / "apps/inspection"]:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--outdir",
                    str(a.dist.resolve()),
                    str(path),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
