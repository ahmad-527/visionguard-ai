"""VisionGuard-owned orchestration for a limited Phase 2A smoke run."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from visionguard.artifacts import (
    ArtifactError,
    capture_git_state,
    dataset_audit_identity,
    new_experiment_artifact,
    sha256_file,
    write_artifact,
)
from visionguard.boundaries import DataBoundaryPolicy, SplitRole
from visionguard.calibration import empirical_quantile
from visionguard.config import ConfigurationError, load_dataset_config
from visionguard.environment import capture_environment
from visionguard.experiment import ExperimentConfig, load_experiment_config
from visionguard.models.patchcore import AnomalibPatchCoreAdapter, ModelDependencyError
from visionguard.reproducibility import configure_reproducibility

LOGGER = logging.getLogger(__name__)


class SmokeRunError(RuntimeError):
    """Raised when a Phase 2A smoke-run prerequisite or execution fails."""


def _portable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _portable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_portable(item) for item in value]
    return value


def _validate_dataset_root(
    dataset_root: Path, config: ExperimentConfig, repository: Path
) -> None:
    dataset_config = load_dataset_config(repository / config.dataset.config_path)
    unknown = set(config.dataset.categories) - set(dataset_config.categories)
    if unknown:
        raise SmokeRunError(
            f"Experiment contains unknown categories: {', '.join(sorted(unknown))}"
        )
    if len(config.dataset.categories) != 1:
        raise SmokeRunError("Phase 2A smoke runs require exactly one category")
    category_root = dataset_root / config.dataset.categories[0]
    required = (
        category_root / config.dataset.train_split / "good",
        category_root / config.dataset.calibration_split / "good",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SmokeRunError(
            "Dataset root lacks required train/validation normal directories"
        )


def build_smoke_datamodule(
    dataset_root: Path,
    config: ExperimentConfig,
    *,
    repository: Path | None = None,
) -> Any:
    """Build an Anomalib Folder module exposing train and validation normals only."""

    _validate_dataset_root(dataset_root, config, repository or Path.cwd())
    DataBoundaryPolicy().authorize(config.dataset.train_split, SplitRole.TRAIN)
    DataBoundaryPolicy().authorize(
        config.dataset.calibration_split, SplitRole.CALIBRATION
    )
    try:
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
    except ImportError as exc:
        raise ModelDependencyError(
            "Anomalib Folder data module is unavailable"
        ) from exc
    category = config.dataset.categories[0]
    return Folder(
        name=f"mvtec_ad_2_{category}_phase2a_smoke",
        root=dataset_root / category,
        normal_dir=Path(config.dataset.train_split) / "good",
        normal_test_dir=Path(config.dataset.calibration_split) / "good",
        abnormal_dir=None,
        mask_dir=None,
        normal_split_ratio=0.0,
        extensions=(".png",),
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=config.reproducibility.seed,
    )


def _start_resources(device_policy: str) -> dict[str, Any]:
    state: dict[str, Any] = {"started_ns": time.perf_counter_ns()}
    try:
        import torch
    except ImportError:
        state["cuda"] = False
    else:
        state["cuda"] = torch.cuda.is_available() and device_policy != "cpu"
        if state["cuda"]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    return state


def _finish_resources(state: dict[str, Any]) -> dict[str, Any]:
    if state["cuda"]:
        import torch

        torch.cuda.synchronize()
        cuda_memory: dict[str, Any] = {
            "status": "measured",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    else:
        cuda_memory = {"status": "unavailable", "reason": "CUDA was not active"}
    return {
        "wall_time_seconds": {
            "status": "measured",
            "value": (time.perf_counter_ns() - state["started_ns"]) / 1_000_000_000,
            "clock": "time.perf_counter_ns",
        },
        "cuda_memory": cuda_memory,
        "host_peak_memory": {
            "status": "unavailable",
            "reason": "No cross-platform process peak-RSS instrument is implemented",
        },
    }


def _weight_identity(expected_revision: str) -> dict[str, Any]:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError as exc:
        raise SmokeRunError("huggingface_hub cache inspection is unavailable") from exc
    cache = scan_cache_dir()
    matching = [
        repository
        for repository in cache.repos
        if repository.repo_id == "timm/wide_resnet50_2.racm_in1k"
    ]
    if len(matching) != 1:
        raise SmokeRunError("Pinned backbone repository was not found in the Hub cache")
    candidates = []
    for revision in matching[0].revisions:
        if revision.commit_hash != expected_revision:
            continue
        for cached_file in revision.files:
            if cached_file.file_name == "model.safetensors":
                candidates.append((revision.commit_hash, Path(cached_file.file_path)))
    if len(candidates) != 1:
        raise SmokeRunError(
            "Expected exactly one cached safetensors revision for the backbone"
        )
    revision, file_path = candidates[0]
    return {
        "id": "timm/wide_resnet50_2.racm_in1k",
        "source": "https://huggingface.co/timm/wide_resnet50_2.racm_in1k",
        "revision": revision,
        "file_name": file_path.name,
        "sha256": sha256_file(file_path),
        "redistributed": False,
    }


def _store_anomaly_map(
    *, output_dir: Path, sample_id: str, anomaly_map: Any
) -> dict[str, Any]:
    """Persist one generated map outside Git and return its portable identity."""

    import numpy as np

    portable_id = sample_id.replace("/", "__").replace("\\", "__")
    relative_path = Path("anomaly-maps") / f"{portable_id}.npy"
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            np.save(stream, anomaly_map.numpy(), allow_pickle=False)
    except FileExistsError as exc:
        raise SmokeRunError(
            f"Refusing to overwrite anomaly map: {destination}"
        ) from exc
    return {
        "status": "generated",
        "path": relative_path.as_posix(),
        "sha256": sha256_file(destination),
        "shape": list(anomaly_map.shape),
        "finite": bool(anomaly_map.isfinite().all().item()),
    }


def run_smoke(*, repository: Path, dataset_root: Path, config_path: Path) -> Path:
    """Execute one non-benchmark category using only train/validation normals."""

    repository = repository.resolve()
    dataset_root = dataset_root.resolve()
    config_path = config_path.resolve()
    config = load_experiment_config(config_path)
    git = capture_git_state(repository)
    if config.require_clean_git and git["dirty"]:
        raise SmokeRunError("Experiment configuration requires a clean Git tree")
    audit_path = repository / config.dataset.audit_report
    dataset_identity = dataset_audit_identity(audit_path, expected_root=dataset_root)
    _validate_dataset_root(dataset_root, config, repository)
    reproducibility = configure_reproducibility(config.reproducibility)
    artifact = new_experiment_artifact(
        experiment_id=config.experiment_id,
        git=git,
        dataset=dataset_identity,
        configuration=_portable(config),
        environment=capture_environment(config.device_policy),
        reproducibility=reproducibility,
    )
    output_dir = repository / config.output_dir
    artifact_path = output_dir / "experiment-artifact.json"
    resources = _start_resources(config.device_policy)
    try:
        datamodule = build_smoke_datamodule(dataset_root, config, repository=repository)
        adapter = AnomalibPatchCoreAdapter(config)
        adapter.fit(datamodule)
        artifact["model_state"] = {
            "memory_bank": adapter.memory_bank_identity(),
        }
        predictions = adapter.predict(datamodule)
        artifact["weights"] = [_weight_identity(config.model.weight_revision)]
        artifact_predictions = []
        for prediction in predictions:
            sample_id = Path(prediction.sample_id).relative_to(dataset_root).as_posix()
            artifact_predictions.append(
                {
                    "sample_id": sample_id,
                    "anomaly_score": prediction.anomaly_score,
                    "anomaly_map": _store_anomaly_map(
                        output_dir=output_dir,
                        sample_id=sample_id,
                        anomaly_map=prediction.anomaly_map,
                    ),
                }
            )
        artifact["predictions"] = artifact_predictions
        if not predictions:
            raise SmokeRunError("PatchCore returned no validation-normal predictions")
        if any(
            not math.isfinite(prediction.anomaly_score) for prediction in predictions
        ):
            raise SmokeRunError("PatchCore returned a non-finite anomaly score")
        image_scores = [prediction.anomaly_score for prediction in predictions]
        pixel_scores = [
            float(value)
            for prediction in predictions
            for value in prediction.anomaly_map.reshape(-1).tolist()
        ]
        artifact["thresholds"] = {
            "image": {
                "value": empirical_quantile(image_scores, config.image_threshold),
                "method": config.image_threshold.method,
                "quantile": config.image_threshold.quantile,
                "sample_count": len(image_scores),
                "calibration_split": config.dataset.calibration_split,
                "normal_only": True,
            },
            "pixel": {
                "value": empirical_quantile(pixel_scores, config.pixel_threshold),
                "method": config.pixel_threshold.method,
                "quantile": config.pixel_threshold.quantile,
                "sample_count": len(pixel_scores),
                "calibration_split": config.dataset.calibration_split,
                "normal_only": True,
            },
        }
        artifact["status"] = "completed"
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        artifact["resources"] = _finish_resources(resources)
        write_artifact(artifact_path, artifact)
    return artifact_path


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately narrow Phase 2A smoke-run CLI."""

    parser = argparse.ArgumentParser(
        description="Run one non-benchmark PatchCore engineering smoke test."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke workflow, preserving a failure artifact when execution begins."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        artifact = run_smoke(
            repository=args.repository,
            dataset_root=args.dataset_root,
            config_path=args.config,
        )
    except (
        ArtifactError,
        ConfigurationError,
        ModelDependencyError,
        SmokeRunError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:
        LOGGER.error("Smoke run failed with %s: %s", type(exc).__name__, exc)
        return 1
    LOGGER.info("Smoke run completed; artifact: %s", artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
