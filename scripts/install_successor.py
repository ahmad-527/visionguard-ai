"""Resolve the v2 recipe in a clean project environment; never global pipx."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from successor_build import ROOT, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument(
        "--profile", choices=["ordinary", "application", "native"], required=True
    )
    a = p.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Clean project venv required")
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if "include-system-site-packages = false" not in cfg.read_text():
        raise RuntimeError("System-site-package inheritance prohibited")
    source = prepare(a.source)
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-c",
        str(ROOT / "requirements/ml-cpu-constraints-v2.txt"),
        str(source) + "[analysis,dev]",
    ]
    if a.profile != "ordinary":
        args.append(str(source / "apps/inspection") + "[test]")
    if a.profile == "native":
        args.extend(["-r", str(ROOT / "requirements/ml-cpu-successor-v2.txt")])
    # The explicit source argument already binds the selected project workspace;
    # do not infer receipt placement from GitHub's reserved RUNNER_* variables.
    report = a.source.parent / f"project-dependencies-{a.profile}.json"
    if report.exists():
        raise RuntimeError("Dependency receipt must be new; preserve prior evidence")
    print(json.dumps({"dependency_report_path": str(report.resolve())}), flush=True)
    args.extend(["--report", str(report)])
    subprocess.run(args, check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
