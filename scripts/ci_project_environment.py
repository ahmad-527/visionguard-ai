"""Free hosted CI only: record capacity and isolate project Python dependencies."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def select_volume(candidates, probe=shutil.disk_usage):
    """Require real capacity on a candidate volume; never alter Budget guards."""
    observations = [{"path": str(path), **probe(path)._asdict()} for path in candidates]
    admitted = [row for row in observations if row["free"] >= 35 * 2**30]
    if not admitted:
        raise RuntimeError("Insufficient actual runner capacity; reserve unchanged")
    return Path(max(admitted, key=lambda row: row["free"])["path"]), observations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    a = p.parse_args()
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
    ):
        raise RuntimeError(
            "Only disposable free GitHub-hosted runner preparation is allowed"
        )
    values = {
        "before": shutil.disk_usage(Path.cwd())._asdict(),
        "removed_sdk_paths": [],
        "production_safety_reserve_bytes": 20 * 2**30,
    }
    # Only unused SDK installations on this ephemeral Linux runner may be removed.
    # No repository/cache/evidence path is computed or passed to deletion.
    if sys.platform == "linux":
        for name in [
            "/usr/local/lib/android",
            "/usr/share/dotnet",
            "/opt/ghc",
            "/usr/local/share/boost",
            "/usr/share/swift",
        ]:
            if shutil.disk_usage(Path.cwd()).free >= 40 * 2**30:
                break
            if Path(name).exists():
                subprocess.run(["sudo", "rm", "-rf", "--", name], check=True)
                values["removed_sdk_paths"].append(name)

    candidates = [a.output.parent]
    if os.name == "nt":
        local = Path(os.environ["LOCALAPPDATA"]) / "Temp"
        if local.is_dir():
            candidates.append(local)
        candidates.append(Path(os.environ["SYSTEMDRIVE"] + "/"))
    values["candidate_volumes"] = [
        {"path": str(path), **shutil.disk_usage(path)._asdict()} for path in candidates
    ]
    print(json.dumps(values), flush=True)
    parent, volumes = select_volume(candidates)
    workspace = Path(tempfile.mkdtemp(prefix="visionguard-ci-", dir=parent))
    values["candidate_volumes"] = volumes
    values["selected_workspace"] = str(workspace)
    values["after"] = shutil.disk_usage(workspace)._asdict()
    a.output = workspace / "project-python"
    a.observations.parent.mkdir(parents=True, exist_ok=True)
    with a.observations.open("x") as out:
        json.dump(values, out, indent=2)
    print(json.dumps(values), flush=True)
    if values["after"]["free"] < 35 * 2**30:
        raise RuntimeError(
            "Insufficient actual runner capacity; reserve will not be weakened"
        )
    if a.output.exists():
        raise RuntimeError("Clean project environment must be new")
    venv.create(a.output, with_pip=True, system_site_packages=False)
    python = a.output / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Parent pip launches into only the new project venv; no global package repair.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-cache-dir",
            "pip==26.2.1",
            "setuptools==84.2.1",
            "hatchling==1.32.4",
            "build==1.6.1",
        ],
        check=True,
    )
    with Path(os.environ["GITHUB_ENV"]).open("a") as out:
        out.write("PIP_NO_CACHE_DIR=1\n")
        for key in ["VISIONGUARD_PROJECT_TEMP", "TEMP", "TMP", "TMPDIR"]:
            out.write(key + "=" + str(workspace) + "\n")
    with Path(os.environ["GITHUB_PATH"]).open("a") as out:
        out.write(str(python.parent.resolve()) + "\n")


if __name__ == "__main__":
    main()
