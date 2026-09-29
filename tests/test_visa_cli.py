"""CLI isolation checks work without datasets, CUDA, or installed model packages."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "visa_acquire",
        "visa_audit",
        "visa_execution",
        "visa_protocol",
        "visa_smoke",
        "visa_dispatcher",
        "visa_development_worker",
    ],
)
def test_help_never_loads_models_or_data(module):
    result = subprocess.run(
        [sys.executable, "-m", f"visionguard.{module}", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_lightweight_import_has_no_model_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import visionguard.visa_smoke; "
            "import visionguard.visa_development; import visionguard.visa_execution; "
            "import visionguard.visa_dispatcher; "
            "assert 'torch' not in sys.modules; assert 'anomalib' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_runtime_denies_sealed_file_access(tmp_path):
    development = tmp_path / "development"
    sealed = tmp_path / "sealed"
    development.mkdir()
    sealed.mkdir()
    (sealed / "synthetic.txt").write_text("synthetic, not dataset content")
    code = (
        "import sys; from pathlib import Path; "
        "from visionguard.visa_smoke import _deny_sealed_access; "
        "_deny_sealed_access(Path(sys.argv[1])); "
        "(Path(sys.argv[1]).parent/'sealed'/'synthetic.txt').read_text()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(development)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert "attempted sealed-data access" in result.stderr
