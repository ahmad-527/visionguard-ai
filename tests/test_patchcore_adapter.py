from __future__ import annotations

import importlib.metadata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import visionguard.models.patchcore as patchcore_module
from visionguard.experiment import load_experiment_config
from visionguard.models.patchcore import AnomalibPatchCoreAdapter, ModelDependencyError


def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    config = load_experiment_config(
        Path("configs/experiments/patchcore-smoke.example.yaml")
    )

    with pytest.raises(ModelDependencyError, match="lightweight audit"):
        AnomalibPatchCoreAdapter(config)


def test_mismatched_anomalib_version_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "2.5.1")
    config = load_experiment_config(
        Path("configs/experiments/patchcore-smoke.example.yaml")
    )

    with pytest.raises(ModelDependencyError, match="but found"):
        AnomalibPatchCoreAdapter(config)


def test_cpu_policy_is_passed_through_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakePatchcore:
        @staticmethod
        def configure_pre_processor(**kwargs: object) -> object:
            calls["preprocessor"] = kwargs
            return object()

        def __init__(self, **kwargs: object) -> None:
            calls["model"] = kwargs

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            calls["engine"] = kwargs

        def fit(self, *, model: object, datamodule: object) -> None:
            calls["fit"] = (model, datamodule)

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "2.6.0")
    monkeypatch.setattr(
        patchcore_module,
        "_import_symbol",
        lambda module, _symbol: (
            FakePatchcore if module == "anomalib.models" else FakeEngine
        ),
    )
    config = load_experiment_config(
        Path("configs/experiments/patchcore-smoke.example.yaml")
    )
    config = replace(config, device_policy="cpu")

    adapter = AnomalibPatchCoreAdapter(config)
    adapter.fit(SimpleNamespace())

    assert calls["engine"] == {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
    }
    assert calls["model"]["post_processor"] is False  # type: ignore[index]
