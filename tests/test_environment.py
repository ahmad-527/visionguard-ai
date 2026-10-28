from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from visionguard.environment import capture_environment, environment_json
from visionguard.experiment import ReproducibilityConfig
from visionguard.reproducibility import configure_reproducibility


@pytest.mark.parametrize("package_version", ["0.1.0", "0.2.0"])
def test_environment_capture_distinguishes_configured_values(
    monkeypatch: pytest.MonkeyPatch,
    package_version: str,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(
                metadata={"Name": "visionguard-ai"}, version=package_version
            )
        ],
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    capture = capture_environment("cpu")

    assert capture["schema_version"] == 1
    assert capture["configured"] == {
        "device_policy": "cpu",
        "hf_hub_offline": True,
    }
    assert capture["python"]["status"] == "detected"
    assert capture["resolved_packages"]["visionguard-ai"] == package_version
    assert "executable" not in json.loads(environment_json("cpu"))["python"]["value"]


def test_reproducibility_always_configures_python_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    report = configure_reproducibility(
        ReproducibilityConfig(
            seed=17, deterministic_algorithms=True, cudnn_benchmark=False
        )
    )

    assert report["python"] == {"status": "configured", "seed": 17}
    assert report["numpy"]["status"] in {"configured", "unavailable"}
    assert report["torch"]["status"] in {"configured", "unavailable"}
    assert report["cublas_workspace_config"] == {
        "status": "configured",
        "value": ":4096:8",
    }
    assert report["limitations"]


def test_environment_module_does_not_require_torch() -> None:
    importlib.import_module("visionguard.environment")
