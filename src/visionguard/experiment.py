"""Typed, validated Phase 2A experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from visionguard.boundaries import DataBoundaryPolicy, SplitRole
from visionguard.config import ConfigurationError
from visionguard.paths import portable_relative_path

DevicePolicy = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class DatasetExperimentConfig:
    """Dataset identity and approved split use for an experiment."""

    config_path: Path
    audit_report: Path
    categories: tuple[str, ...]
    train_split: str
    calibration_split: str
    evaluation_split: str | None
    configuration_frozen: bool


@dataclass(frozen=True)
class PatchCoreConfig:
    """Explicit PatchCore and pretrained-backbone configuration."""

    implementation: str
    implementation_version: str
    backbone: str
    layers: tuple[str, ...]
    pretrained: bool
    weight_id: str
    weight_source: str
    weight_revision: str
    coreset_sampling_ratio: float
    num_neighbors: int


@dataclass(frozen=True)
class PreprocessingConfig:
    """Deterministic input preprocessing contract."""

    resize: tuple[int, int]
    center_crop: tuple[int, int]
    normalization: str
    augmentation: str


@dataclass(frozen=True)
class ThresholdConfig:
    """One normal-only threshold calibration rule."""

    method: str
    quantile: float
    minimum_samples: int


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Configured seed and PyTorch backend policy."""

    seed: int
    deterministic_algorithms: bool
    cudnn_benchmark: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration required before a Phase 2A run."""

    schema_version: int
    experiment_id: str
    require_clean_git: bool
    dataset: DatasetExperimentConfig
    model: PatchCoreConfig
    preprocessing: PreprocessingConfig
    reproducibility: ReproducibilityConfig
    device_policy: DevicePolicy
    image_threshold: ThresholdConfig
    pixel_threshold: ThresholdConfig
    output_dir: Path


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _strict_keys(data: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigurationError(
            f"{location} contains unknown keys: {', '.join(sorted(unknown))}"
        )


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value.strip()


def _relative_path(value: Any, location: str) -> Path:
    raw_path = _nonempty_string(value, location)
    try:
        return portable_relative_path(raw_path)
    except ValueError as exc:
        raise ConfigurationError(
            f"{location} must be a repository-relative path without '..'"
        ) from exc


def _string_list(value: Any, location: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ConfigurationError(f"{location} must be a non-empty list of strings")
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{location} must not contain duplicates")
    return result


def _image_size(value: Any, location: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and item > 0 for item in value)
    ):
        raise ConfigurationError(f"{location} must contain two positive integers")
    return value[0], value[1]


def _threshold(value: Any, location: str) -> ThresholdConfig:
    data = _mapping(value, location)
    _strict_keys(data, {"method", "quantile", "minimum_samples"}, location)
    method = _nonempty_string(data.get("method"), f"{location}.method")
    if method != "empirical_quantile":
        raise ConfigurationError(
            f"{location}.method must be 'empirical_quantile' in Phase 2A"
        )
    quantile = data.get("quantile")
    if not isinstance(quantile, (int, float)) or isinstance(quantile, bool):
        raise ConfigurationError(f"{location}.quantile must be numeric")
    if not 0.0 < float(quantile) < 1.0:
        raise ConfigurationError(f"{location}.quantile must be between 0 and 1")
    minimum_samples = data.get("minimum_samples")
    if not isinstance(minimum_samples, int) or minimum_samples < 1:
        raise ConfigurationError(f"{location}.minimum_samples must be positive")
    return ThresholdConfig(method, float(quantile), minimum_samples)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and strictly validate a Phase 2A experiment YAML file."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to load {path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    _strict_keys(root, {"schema_version", "experiment"}, "configuration")
    if root.get("schema_version") != 1:
        raise ConfigurationError("schema_version must equal 1")
    experiment = _mapping(root.get("experiment"), "experiment")
    _strict_keys(
        experiment,
        {
            "id",
            "git",
            "dataset",
            "model",
            "preprocessing",
            "reproducibility",
            "device_policy",
            "thresholds",
            "output_dir",
        },
        "experiment",
    )

    git = _mapping(experiment.get("git"), "experiment.git")
    _strict_keys(git, {"require_clean"}, "experiment.git")
    require_clean = git.get("require_clean")
    if not isinstance(require_clean, bool):
        raise ConfigurationError("experiment.git.require_clean must be boolean")

    dataset_data = _mapping(experiment.get("dataset"), "experiment.dataset")
    _strict_keys(
        dataset_data,
        {
            "config",
            "audit_report",
            "categories",
            "train_split",
            "calibration_split",
            "evaluation_split",
            "configuration_frozen",
        },
        "experiment.dataset",
    )
    configuration_frozen = dataset_data.get("configuration_frozen")
    if not isinstance(configuration_frozen, bool):
        raise ConfigurationError(
            "experiment.dataset.configuration_frozen must be boolean"
        )
    evaluation_split = dataset_data.get("evaluation_split")
    if evaluation_split is not None:
        evaluation_split = _nonempty_string(
            evaluation_split, "experiment.dataset.evaluation_split"
        )
    dataset = DatasetExperimentConfig(
        config_path=_relative_path(
            dataset_data.get("config"), "experiment.dataset.config"
        ),
        audit_report=_relative_path(
            dataset_data.get("audit_report"), "experiment.dataset.audit_report"
        ),
        categories=_string_list(
            dataset_data.get("categories"), "experiment.dataset.categories"
        ),
        train_split=_nonempty_string(
            dataset_data.get("train_split"), "experiment.dataset.train_split"
        ),
        calibration_split=_nonempty_string(
            dataset_data.get("calibration_split"),
            "experiment.dataset.calibration_split",
        ),
        evaluation_split=evaluation_split,
        configuration_frozen=configuration_frozen,
    )
    policy = DataBoundaryPolicy()
    policy.authorize(dataset.train_split, SplitRole.TRAIN)
    policy.authorize(dataset.calibration_split, SplitRole.CALIBRATION)
    if evaluation_split is not None:
        policy.authorize(
            evaluation_split,
            SplitRole.PRELIMINARY_EVALUATION,
            configuration_frozen=configuration_frozen,
        )

    model_data = _mapping(experiment.get("model"), "experiment.model")
    _strict_keys(
        model_data,
        {
            "name",
            "implementation",
            "implementation_version",
            "backbone",
            "layers",
            "pretrained",
            "weight_id",
            "weight_source",
            "weight_revision",
            "coreset_sampling_ratio",
            "num_neighbors",
        },
        "experiment.model",
    )
    if model_data.get("name") != "patchcore":
        raise ConfigurationError("experiment.model.name must be 'patchcore'")
    if model_data.get("implementation") != "anomalib":
        raise ConfigurationError("experiment.model.implementation must be 'anomalib'")
    pretrained = model_data.get("pretrained")
    if pretrained is not True:
        raise ConfigurationError("Phase 2A PatchCore requires pretrained: true")
    ratio = model_data.get("coreset_sampling_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise ConfigurationError(
            "experiment.model.coreset_sampling_ratio must be numeric"
        )
    if not 0.0 < float(ratio) <= 1.0:
        raise ConfigurationError(
            "experiment.model.coreset_sampling_ratio must be in (0, 1]"
        )
    neighbors = model_data.get("num_neighbors")
    if not isinstance(neighbors, int) or neighbors < 1:
        raise ConfigurationError("experiment.model.num_neighbors must be positive")
    model = PatchCoreConfig(
        implementation="anomalib",
        implementation_version=_nonempty_string(
            model_data.get("implementation_version"),
            "experiment.model.implementation_version",
        ),
        backbone=_nonempty_string(
            model_data.get("backbone"), "experiment.model.backbone"
        ),
        layers=_string_list(model_data.get("layers"), "experiment.model.layers"),
        pretrained=True,
        weight_id=_nonempty_string(
            model_data.get("weight_id"), "experiment.model.weight_id"
        ),
        weight_source=_nonempty_string(
            model_data.get("weight_source"), "experiment.model.weight_source"
        ),
        weight_revision=_nonempty_string(
            model_data.get("weight_revision"), "experiment.model.weight_revision"
        ),
        coreset_sampling_ratio=float(ratio),
        num_neighbors=neighbors,
    )
    if model.implementation_version != "2.6.0":
        raise ConfigurationError("Phase 2A pins anomalib implementation_version 2.6.0")
    if model.backbone != model.weight_id:
        raise ConfigurationError(
            "experiment.model.backbone must equal the explicit weight_id to prevent "
            "default-weight drift"
        )
    if len(model.weight_revision) != 40 or any(
        character not in "0123456789abcdef" for character in model.weight_revision
    ):
        raise ConfigurationError(
            "experiment.model.weight_revision must be a lowercase 40-character "
            "Git commit"
        )

    preprocessing_data = _mapping(
        experiment.get("preprocessing"), "experiment.preprocessing"
    )
    _strict_keys(
        preprocessing_data,
        {"resize", "center_crop", "normalization", "augmentation"},
        "experiment.preprocessing",
    )
    preprocessing = PreprocessingConfig(
        resize=_image_size(
            preprocessing_data.get("resize"), "experiment.preprocessing.resize"
        ),
        center_crop=_image_size(
            preprocessing_data.get("center_crop"),
            "experiment.preprocessing.center_crop",
        ),
        normalization=_nonempty_string(
            preprocessing_data.get("normalization"),
            "experiment.preprocessing.normalization",
        ),
        augmentation=_nonempty_string(
            preprocessing_data.get("augmentation"),
            "experiment.preprocessing.augmentation",
        ),
    )
    if preprocessing.normalization != "imagenet":
        raise ConfigurationError("Phase 2A normalization must be 'imagenet'")
    if preprocessing.augmentation != "none":
        raise ConfigurationError("Phase 2A augmentation must be 'none'")
    if any(
        crop > resize
        for crop, resize in zip(
            preprocessing.center_crop, preprocessing.resize, strict=True
        )
    ):
        raise ConfigurationError("center_crop must not exceed resize")

    reproducibility_data = _mapping(
        experiment.get("reproducibility"), "experiment.reproducibility"
    )
    _strict_keys(
        reproducibility_data,
        {"seed", "deterministic_algorithms", "cudnn_benchmark"},
        "experiment.reproducibility",
    )
    seed = reproducibility_data.get("seed")
    deterministic = reproducibility_data.get("deterministic_algorithms")
    cudnn_benchmark = reproducibility_data.get("cudnn_benchmark")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("experiment.reproducibility.seed must be non-negative")
    if not isinstance(deterministic, bool) or not isinstance(cudnn_benchmark, bool):
        raise ConfigurationError(
            "deterministic_algorithms and cudnn_benchmark must be boolean"
        )
    if deterministic and cudnn_benchmark:
        raise ConfigurationError(
            "cudnn_benchmark must be false when deterministic_algorithms is true"
        )
    reproducibility = ReproducibilityConfig(seed, deterministic, cudnn_benchmark)

    device_policy = experiment.get("device_policy")
    if device_policy not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("experiment.device_policy must be auto, cpu, or cuda")
    thresholds = _mapping(experiment.get("thresholds"), "experiment.thresholds")
    _strict_keys(thresholds, {"image", "pixel"}, "experiment.thresholds")
    return ExperimentConfig(
        schema_version=1,
        experiment_id=_nonempty_string(experiment.get("id"), "experiment.id"),
        require_clean_git=require_clean,
        dataset=dataset,
        model=model,
        preprocessing=preprocessing,
        reproducibility=reproducibility,
        device_policy=device_policy,
        image_threshold=_threshold(thresholds.get("image"), "thresholds.image"),
        pixel_threshold=_threshold(thresholds.get("pixel"), "thresholds.pixel"),
        output_dir=_relative_path(
            experiment.get("output_dir"), "experiment.output_dir"
        ),
    )
