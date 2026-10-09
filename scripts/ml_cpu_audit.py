"""Unfiltered fresh public dependency scans, separate from the CPU runtime.

Empty reports are not security clearance: the manually verified upstream PT2
finding remains unresolved. Scanner failures/findings retain their raw exit codes
and fail this check; nothing is ignored, repaired or auto-upgraded in the runtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import sysconfig
import venv
from datetime import UTC, datetime
from pathlib import Path


def execute(command: list[str], root: Path, label: str) -> int:
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(command, cwd=root, capture_output=True, check=False)
    (root / f"{label}.log").write_bytes(result.stdout + result.stderr)
    record = {
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "exit_code": result.returncode,
        "no_advisory_suppressions": True,
        "not_security_certification": True,
    }
    with (root / f"{label}-command.json").open("x", encoding="utf-8") as target:
        json.dump(record, target, indent=2)
    print(json.dumps(record), flush=True)
    print((result.stdout + result.stderr).decode("utf-8", errors="replace"), flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tool = root / "scanner"
    venv.create(tool, with_pip=True)
    python = tool / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    status = execute(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "pip==26.2.1",
            "setuptools==84.0.0",
            "pip-audit==2.10.1",
        ],
        root,
        "scanner-provision",
    )
    if status:
        raise SystemExit(status)
    common = [
        str(python),
        "-m",
        "pip_audit",
        "--progress-spinner",
        "off",
        "--timeout",
        "15",
        "-f",
        "json",
    ]
    for service in ("pypi", "osv"):
        result = execute(
            [
                *common,
                "--path",
                sysconfig.get_path("purelib"),
                "--skip-editable",
                "-s",
                service,
                "--cache-dir",
                str(root / f"cache-{service}"),
                "-o",
                str(root / f"audit-{service}.json"),
            ],
            root,
            f"audit-{service}",
        )
        status = status or result
    # Public base version query is additional coverage, not CPU wheel attestation.
    requirements = root / "normalized-torch.txt"
    requirements.write_text("torch==2.13.0\ntorchvision==0.28.0\n", encoding="utf-8")
    result = execute(
        [
            *common,
            "-r",
            str(requirements),
            "--no-deps",
            "--disable-pip",
            "--cache-dir",
            str(root / "cache-normalized"),
            "-o",
            str(root / "audit-normalized-torch.json"),
        ],
        root,
        "audit-normalized-torch",
    )
    raise SystemExit(status or result)


if __name__ == "__main__":
    main()
