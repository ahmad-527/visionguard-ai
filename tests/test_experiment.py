from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from visionguard.boundaries import DataBoundaryError, DataBoundaryPolicy, SplitRole
from visionguard.config import ConfigurationError
from visionguard.experiment import load_experiment_config

EXAMPLE = Path("configs/experiments/patchcore-smoke.example.yaml")


def write_changed(tmp_path: Path, change: callable) -> Path:  # type: ignore[type-arg]
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_repository_patchcore_config_loads() -> None:
    config = load_experiment_config(EXAMPLE)

    assert config.experiment_id == "patchcore_phase2a_smoke"
    assert config.dataset.categories == ("can",)
    assert config.dataset.evaluation_split is None
    assert config.model.backbone == "wide_resnet50_2.racm_in1k"
    assert config.image_threshold.quantile == 0.995


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = write_changed(
        tmp_path, lambda data: data["experiment"].update({"surprise": True})
    )

    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/private/file.yaml",
        "C:/private/file.yaml",
        r"C:\private\file.yaml",
        "C:private/file.yaml",
        r"\server\share\file.yaml",
        r"\\server\share\file.yaml",
        "//server/share/file.yaml",
        "../file.yaml",
        "configs/../../file.yaml",
        r"..\file.yaml",
        r"configs\..\..\file.yaml",
    ],
)
def test_non_relative_dataset_paths_are_rejected(
    tmp_path: Path, invalid_path: str
) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["dataset"].update({"config": invalid_path}),
    )

    with pytest.raises(ConfigurationError, match="repository-relative"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("field", "invalid_path"),
    [
        ("dataset.audit_report", "//server/share/audit.json"),
        ("output_dir", r"outputs\..\..\private"),
    ],
)
def test_all_configured_repository_paths_use_portable_validation(
    tmp_path: Path, field: str, invalid_path: str
) -> None:
    def change(data: Any) -> None:
        experiment = data["experiment"]
        if field == "dataset.audit_report":
            experiment["dataset"]["audit_report"] = invalid_path
        else:
            experiment[field] = invalid_path

    with pytest.raises(ConfigurationError, match="repository-relative"):
        load_experiment_config(write_changed(tmp_path, change))


def test_repository_paths_allow_relative_paths_with_either_slash_convention(
    tmp_path: Path,
) -> None:
    def change(data: Any) -> None:
        experiment = data["experiment"]
        experiment["dataset"]["config"] = r"configs\datasets\mvtec_ad_2.yaml"
        experiment["dataset"]["audit_report"] = "audit-reports/example.json"
        experiment["output_dir"] = r"outputs\run-001"

    config = load_experiment_config(write_changed(tmp_path, change))

    assert config.dataset.config_path == Path("configs/datasets/mvtec_ad_2.yaml")
    assert config.dataset.audit_report == Path("audit-reports/example.json")
    assert config.output_dir == Path("outputs/run-001")


def test_private_split_is_rejected(tmp_path: Path) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["dataset"].update(
            {"evaluation_split": "test_private", "configuration_frozen": True}
        ),
    )

    with pytest.raises(DataBoundaryError, match="Private"):
        load_experiment_config(path)


def test_public_test_requires_frozen_configuration(tmp_path: Path) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["dataset"].update(
            {"evaluation_split": "test_public", "configuration_frozen": False}
        ),
    )

    with pytest.raises(DataBoundaryError, match="freeze"):
        load_experiment_config(path)


def test_weight_alias_drift_is_rejected(tmp_path: Path) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["model"].update(
            {"backbone": "wide_resnet50_2"}
        ),
    )

    with pytest.raises(ConfigurationError, match="weight_id"):
        load_experiment_config(path)


def test_invalid_coreset_ratio_is_rejected(tmp_path: Path) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["model"].update({"coreset_sampling_ratio": 0}),
    )

    with pytest.raises(ConfigurationError, match="coreset_sampling_ratio"):
        load_experiment_config(path)


def test_incompatible_determinism_settings_are_rejected(tmp_path: Path) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["reproducibility"].update(
            {"cudnn_benchmark": True}
        ),
    )

    with pytest.raises(ConfigurationError, match="cudnn_benchmark"):
        load_experiment_config(path)


def test_center_crop_can_be_disabled_for_border_preserving_protocol(
    tmp_path: Path,
) -> None:
    path = write_changed(
        tmp_path,
        lambda data: data["experiment"]["preprocessing"].update({"center_crop": None}),
    )

    assert load_experiment_config(path).preprocessing.center_crop is None


def test_boundary_policy_rejects_validation_training() -> None:
    with pytest.raises(DataBoundaryError, match="cannot be used"):
        DataBoundaryPolicy().authorize("validation", SplitRole.TRAIN)


def test_boundary_policy_has_no_local_final_evaluation() -> None:
    with pytest.raises(DataBoundaryError, match="private-server"):
        DataBoundaryPolicy().authorize("test_public", SplitRole.FINAL_EVALUATION)
