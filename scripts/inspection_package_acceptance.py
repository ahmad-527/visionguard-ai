"""Install both distributions into separate venvs and test outside the checkout.

Requires prebuilt artifacts in VG_DIST; creates unique VG_ENVS parent, refuses
pre-existing environment directories. No ML extras, artifact or dataset loading.
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

distribution = Path(os.environ["VG_DIST"]).resolve()
environments = Path(os.environ["VG_ENVS"]).resolve()
smoke = Path(__file__).with_name("inspection_install_smoke.py").resolve()
artifacts = list(distribution.glob("*.whl")) + list(distribution.glob("*.tar.gz"))
assert len(artifacts) == 4, "Exactly core/app wheel and sdist required"
environments.mkdir(parents=True, exist_ok=False)
for kind, suffix in (("wheel", ".whl"), ("sdist", ".gz")):
    selected = [artifact for artifact in artifacts if artifact.suffix == suffix]
    assert len(selected) == 2
    environment = environments / kind
    venv.create(environment, with_pip=True)
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    # Fresh venv bootstrap versions depend on the interpreter image. Pin reviewed
    # advisory-clean tools here, never upgrade the frozen ML/evaluation environment.
    subprocess.run(
        [str(python), "-m", "pip", "install", "pip==26.2.1", "setuptools==84.0.0"],
        cwd=environments,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            *[str(artifact) for artifact in selected],
            "httpx>=0.27,<1",
        ],
        cwd=environments,
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], cwd=environments, check=True)
    entrypoint = python.parent / (
        "visionguard-inspect.exe" if sys.platform == "win32" else "visionguard-inspect"
    )
    subprocess.run([str(entrypoint), "--help"], cwd=environments, check=True)
    subprocess.run([str(python), "-I", str(smoke)], cwd=environments, check=True)
