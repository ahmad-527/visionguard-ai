"""Resumable execution of the frozen MVTec AD 2 public benchmark matrix."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.artifacts import (
    ArtifactError,
    capture_git_state,
    dataset_audit_identity,
    new_benchmark_artifact,
    sha256_file,
    validate_artifact,
    write_artifact,
)
from visionguard.benchmark_metrics import (
    BinaryCountAccumulator,
    Float16AuProAccumulator,
)
from visionguard.boundaries import DataBoundaryPolicy, SplitRole
from visionguard.calibration import highest_order_statistic
from visionguard.environment import capture_environment
from visionguard.experiment import (
    DatasetExperimentConfig,
    ExperimentConfig,
    PatchCoreConfig,
    PreprocessingConfig,
    ReproducibilityConfig,
    ThresholdConfig,
)
from visionguard.metrics import MetricResult, binary_auroc, binary_f1
from visionguard.models.patchcore import AnomalibPatchCoreAdapter, ModelDependencyError
from visionguard.preprocessing import restore_anomaly_map
from visionguard.protocol import (
    OFFICIAL_CATEGORIES,
    PROTOCOL_SEEDS,
    BenchmarkGateInputs,
    ProtocolError,
    authorize_public_benchmark,
    load_protocol,
    protocol_fingerprint,
)
from visionguard.reproducibility import configure_reproducibility
from visionguard.runner import _finish_resources, _start_resources, _weight_identity

LOGGER = logging.getLogger(__name__)
METRIC_IMPLEMENTATION = {
    "au_pro": {
        "id": "visionguard.float16_histogram_au_pro.v1",
        "official_reference": "MVTec AD evaluation utility 1.0",
        "fpr_limit": 0.05,
        "ground_truth_connectivity": 8,
        "ties": "grouped by exact float16 score",
    },
    "image_auroc": {"id": "visionguard.mann_whitney_tie_aware.v1"},
    "image_f1": {"id": "visionguard.binary_f1.v1"},
    "pixel_f1": {"id": "visionguard.streaming_binary_f1.v1"},
}


class BenchmarkRunError(RuntimeError):
    """Raised when a public benchmark prerequisite or run is invalid."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _portable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _portable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def benchmark_cells() -> tuple[tuple[str, int], ...]:
    """Return the immutable category-major 24-cell execution order."""

    return tuple(
        (category, seed) for category in OFFICIAL_CATEGORIES for seed in PROTOCOL_SEEDS
    )


def pending_cells(
    completed: Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    """Return frozen matrix cells not already completed, preserving order."""

    completed_set = set(completed)
    unknown = completed_set - set(benchmark_cells())
    if unknown:
        raise BenchmarkRunError(f"Resume state contains unknown cells: {unknown}")
    return tuple(cell for cell in benchmark_cells() if cell not in completed_set)


def execute_pending_cells(
    completed: Iterable[tuple[str, int]],
    execute: Callable[[str, int], None],
) -> tuple[tuple[str, int], ...]:
    """Execute each pending cell once; failures propagate for durable resume."""

    cells = pending_cells(completed)
    for category, seed in cells:
        execute(category, seed)
    return cells


def _resolved_versions(environment: Mapping[str, Any]) -> dict[str, str]:
    packages = environment.get("packages")
    if not isinstance(packages, dict):
        raise BenchmarkRunError("Environment capture is missing package versions")
    versions: dict[str, str] = {}
    for name in ("anomalib", "timm", "torch", "torchvision"):
        entry = packages.get(name)
        if not isinstance(entry, dict) or entry.get("status") != "detected":
            raise BenchmarkRunError(f"Required package {name} is unavailable")
        versions[name] = str(entry.get("value"))
    return versions


def _experiment_config(
    *, protocol: Mapping[str, Any], category: str, seed: int, output_dir: Path
) -> ExperimentConfig:
    model = protocol["model"]
    preprocessing = protocol["preprocessing"]
    return ExperimentConfig(
        schema_version=1,
        experiment_id=f"phase2c-{category}-seed-{seed}",
        require_clean_git=True,
        dataset=DatasetExperimentConfig(
            config_path=Path("configs/datasets/mvtec_ad_2.yaml"),
            audit_report=Path("audit-reports/runtime-supplied.json"),
            categories=(category,),
            train_split=str(protocol["dataset"]["train_split"]),
            calibration_split=str(protocol["dataset"]["calibration_split"]),
            evaluation_split=str(
                protocol["dataset"]["authorized_public_evaluation_split"]
            ),
            configuration_frozen=True,
        ),
        model=PatchCoreConfig(
            implementation=str(model["implementation"]),
            implementation_version=str(model["implementation_version"]),
            backbone=str(model["backbone"]),
            layers=tuple(model["layers"]),
            pretrained=bool(model["pretrained"]),
            weight_id=str(model["backbone"]),
            weight_source=str(model["weight_source"]),
            weight_revision=str(model["weight_revision"]),
            coreset_sampling_ratio=float(model["coreset_sampling_ratio"]),
            num_neighbors=int(model["num_neighbors"]),
        ),
        preprocessing=PreprocessingConfig(
            resize=tuple(preprocessing["resize"]),
            center_crop=preprocessing["center_crop"],
            normalization="imagenet",
            augmentation=str(preprocessing["augmentation"]),
        ),
        reproducibility=ReproducibilityConfig(
            seed=seed,
            deterministic_algorithms=bool(
                protocol["reproducibility"]["deterministic_algorithms"]
            ),
            cudnn_benchmark=bool(protocol["reproducibility"]["cudnn_benchmark"]),
        ),
        device_policy="cuda",
        image_threshold=ThresholdConfig("highest_order_statistic", 0.5, 19),
        pixel_threshold=ThresholdConfig("highest_order_statistic", 0.5, 19),
        output_dir=output_dir,
    )


def _folder_datamodule(
    *, dataset_root: Path, category: str, seed: int, evaluation: bool
) -> Any:
    try:
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
    except ImportError as exc:  # pragma: no cover - ML environment contract
        raise ModelDependencyError(
            "Anomalib Folder data module is unavailable"
        ) from exc
    category_root = dataset_root / category
    common = {
        "name": f"mvtec_ad_2_{category}_{'public' if evaluation else 'calibration'}",
        "root": category_root,
        "normal_dir": Path("train") / "good",
        "normal_split_ratio": 0.0,
        "extensions": (".png",),
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_workers": 0,
        "test_split_mode": TestSplitMode.FROM_DIR,
        "seed": seed,
    }
    if evaluation:
        return Folder(
            **common,
            normal_test_dir=Path("test_public") / "good",
            abnormal_dir=Path("test_public") / "bad",
            mask_dir=Path("test_public") / "ground_truth" / "bad",
            val_split_mode=ValSplitMode.NONE,
        )
    return Folder(
        **common,
        normal_test_dir=Path("validation") / "good",
        abnormal_dir=None,
        mask_dir=None,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
    )


def _original_shape(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - base dependency contract
        raise BenchmarkRunError("Image inspection requires Pillow") from exc
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _calibrate(
    *, adapter: AnomalibPatchCoreAdapter, datamodule: Any, dataset_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = adapter.predict(datamodule)
    if not predictions:
        raise BenchmarkRunError("Calibration returned no validation-normal predictions")
    inputs: list[dict[str, Any]] = []
    image_scores: list[float] = []
    pixel_maxima: list[float] = []
    for prediction in predictions:
        sample_path = Path(prediction.sample_id).resolve()
        sample_id = sample_path.relative_to(dataset_root).as_posix()
        restored = restore_anomaly_map(
            prediction.anomaly_map, _original_shape(sample_path)
        )
        score = float(prediction.anomaly_score)
        pixel_maximum = float(restored.max().item())
        if not math.isfinite(score) or not math.isfinite(pixel_maximum):
            raise BenchmarkRunError("Calibration produced a non-finite score")
        image_scores.append(score)
        pixel_maxima.append(pixel_maximum)
        inputs.append(
            {
                "sample_id": sample_id,
                "image_anomaly_score": score,
                "pixel_maximum": pixel_maximum,
                "restored_map_sha256": _tensor_sha256(restored),
                "restored_map_shape": list(restored.shape),
            }
        )
    image = highest_order_statistic(image_scores, minimum_samples=19)
    pixel = highest_order_statistic(pixel_maxima, minimum_samples=19)
    calibration = {
        "normal_only": True,
        "split": "validation",
        "comparison": "score_strictly_greater_than_threshold",
        "inputs": inputs,
        "image": dataclasses.asdict(image),
        "pixel": dataclasses.asdict(pixel),
    }
    return calibration, {"image": image.threshold, "pixel": pixel.threshold}


def _ground_truth(sample_path: Path) -> tuple[int, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - ML environment contract
        raise BenchmarkRunError(
            "Ground-truth loading requires NumPy and Pillow"
        ) from exc
    if sample_path.parent.name == "good":
        height, width = _original_shape(sample_path)
        return 0, np.zeros((height, width), dtype=np.uint8)
    if sample_path.parent.name != "bad":
        raise BenchmarkRunError(f"Unexpected public-test condition: {sample_path}")
    mask_path = (
        sample_path.parent.parent
        / "ground_truth"
        / "bad"
        / f"{sample_path.stem}_mask{sample_path.suffix}"
    )
    if not mask_path.is_file():
        raise BenchmarkRunError(f"Missing public-test mask: {mask_path.name}")
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or not np.all(np.isin(mask, (0, 255))):
        raise BenchmarkRunError(f"Public-test mask is not binary: {mask_path.name}")
    return 1, (mask > 0).astype(np.uint8, copy=False)


def _write_maps(
    *,
    run_dir: Path,
    sample_id: str,
    restored: Any,
    pixel_threshold: float,
) -> tuple[dict[str, Any], Any, Any]:
    try:
        import numpy as np
        import tifffile
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - ML environment contract
        raise BenchmarkRunError(
            "Map persistence requires NumPy, tifffile, and Pillow"
        ) from exc
    float32_map = restored.detach().cpu().numpy().astype(np.float32, copy=False)
    if not bool(np.isfinite(float32_map).all()):
        raise BenchmarkRunError("Evaluation produced a non-finite anomaly map")
    official_map = float32_map.astype(np.float16)
    if not bool(np.isfinite(official_map).all()):
        raise BenchmarkRunError("Float16 conversion produced a non-finite anomaly map")
    thresholded = float32_map > pixel_threshold
    relative_source = Path(sample_id)
    relative_tiff = (
        Path("anomaly-images") / relative_source.parent / f"{relative_source.stem}.tiff"
    )
    relative_png = (
        Path("anomaly-images-thresholded")
        / relative_source.parent
        / f"{relative_source.stem}.png"
    )
    tiff_path = run_dir / relative_tiff
    png_path = run_dir / relative_png
    tiff_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if tiff_path.exists() or png_path.exists():
        raise BenchmarkRunError("Refusing to overwrite an existing anomaly map")
    tifffile.imwrite(tiff_path, official_map, photometric="minisblack")
    Image.fromarray((thresholded.astype(np.uint8) * 255), mode="L").save(png_path)
    identity = {
        "status": "generated",
        "path": relative_tiff.as_posix(),
        "sha256": sha256_file(tiff_path),
        "shape": list(official_map.shape),
        "dtype": "float16",
        "coordinate_space": "original_image_height_width",
        "finite": True,
        "thresholded_path": relative_png.as_posix(),
        "thresholded_sha256": sha256_file(png_path),
    }
    return identity, official_map, thresholded


def _metric_payload(result: MetricResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _evaluate(
    *,
    adapter: AnomalibPatchCoreAdapter,
    datamodule: Any,
    dataset_root: Path,
    run_dir: Path,
    thresholds: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = adapter.predict(datamodule)
    if not predictions:
        raise BenchmarkRunError("Public evaluation returned no predictions")
    image_labels: list[int] = []
    image_scores: list[float] = []
    image_decisions: list[int] = []
    pixel_f1 = BinaryCountAccumulator()
    au_pro = Float16AuProAccumulator()
    records: list[dict[str, Any]] = []
    for prediction in predictions:
        sample_path = Path(prediction.sample_id).resolve()
        sample_id = sample_path.relative_to(dataset_root).as_posix()
        label, mask = _ground_truth(sample_path)
        restored = restore_anomaly_map(prediction.anomaly_map, tuple(mask.shape))
        map_identity, official_map, thresholded = _write_maps(
            run_dir=run_dir,
            sample_id=sample_id,
            restored=restored,
            pixel_threshold=float(thresholds["pixel"]),
        )
        score = float(prediction.anomaly_score)
        if not math.isfinite(score):
            raise BenchmarkRunError("Public evaluation produced a non-finite score")
        decision = int(score > thresholds["image"])
        image_labels.append(label)
        image_scores.append(score)
        image_decisions.append(decision)
        pixel_f1.update(mask, thresholded)
        au_pro.update(mask, official_map)
        records.append(
            {
                "sample_id": sample_id,
                "label": label,
                "anomaly_score": score,
                "image_prediction": decision,
                "anomaly_map": map_identity,
            }
        )
    metrics = {
        "au_pro_0.05": _metric_payload(au_pro.result(fpr_limit=0.05)),
        "pixel_f1": _metric_payload(pixel_f1.result(level="pixel")),
        "image_f1": _metric_payload(
            binary_f1(image_labels, image_decisions, level="image")
        ),
        "image_auroc": _metric_payload(
            binary_auroc(image_labels, image_scores, level="image")
        ),
    }
    undefined = [
        name for name, value in metrics.items() if value["status"] != "defined"
    ]
    if undefined:
        raise BenchmarkRunError(
            "Required category metrics are undefined: " + ", ".join(undefined)
        )
    return records, metrics


def run_cell(
    *,
    repository: Path,
    dataset_root: Path,
    audit_report: Path,
    protocol_document: dict[str, Any],
    category: str,
    seed: int,
    run_dir: Path,
    explicit_benchmark_mode: bool,
) -> Path:
    """Execute one clean, protocol-gated category/seed benchmark cell."""

    repository = repository.resolve()
    dataset_root = dataset_root.resolve()
    git = capture_git_state(repository)
    dataset = dataset_audit_identity(audit_report, expected_root=dataset_root)
    environment = capture_environment("cuda")
    weight = _weight_identity(
        str(protocol_document["protocol"]["model"]["weight_revision"])
    )
    authorize_public_benchmark(
        protocol_document,
        BenchmarkGateInputs(
            explicit_benchmark_mode=explicit_benchmark_mode,
            evaluation_split="test_public",
            git_dirty=bool(git["dirty"]),
            dataset_audit_status=str(dataset["status"]),
            weight_sha256=str(weight["sha256"]),
            resolved_versions=_resolved_versions(environment),
            categories=OFFICIAL_CATEGORIES,
            seeds=PROTOCOL_SEEDS,
            recorded_fingerprint=protocol_fingerprint(protocol_document),
        ),
    )
    DataBoundaryPolicy().authorize("train", SplitRole.TRAIN)
    DataBoundaryPolicy().authorize("validation", SplitRole.CALIBRATION)
    DataBoundaryPolicy().authorize(
        "test_public", SplitRole.PRELIMINARY_EVALUATION, configuration_frozen=True
    )
    if category not in OFFICIAL_CATEGORIES or seed not in PROTOCOL_SEEDS:
        raise BenchmarkRunError("Cell is outside the frozen category/seed matrix")

    config = _experiment_config(
        protocol=protocol_document["protocol"],
        category=category,
        seed=seed,
        output_dir=run_dir,
    )
    reproducibility = configure_reproducibility(config.reproducibility)
    artifact = new_benchmark_artifact(
        protocol_document=protocol_document,
        experiment_id=config.experiment_id,
        git=git,
        dataset=dataset,
        category=category,
        seed=seed,
        environment=environment,
        weight=weight,
        calibration={"normal_only": True, "status": "pending"},
    )
    artifact["configuration"] = _portable(config)
    artifact["reproducibility"] = reproducibility
    artifact["evaluation_split"] = "test_public"
    artifact["model_state"] = {}
    artifact_path = run_dir / "benchmark-artifact.json"
    resources = _start_resources(config.device_policy)
    try:
        calibration_module = _folder_datamodule(
            dataset_root=dataset_root, category=category, seed=seed, evaluation=False
        )
        adapter = AnomalibPatchCoreAdapter(config)
        adapter.fit(calibration_module)
        artifact["model_state"] = {"memory_bank": adapter.memory_bank_identity()}
        calibration, thresholds = _calibrate(
            adapter=adapter, datamodule=calibration_module, dataset_root=dataset_root
        )
        artifact["calibration"] = calibration
        artifact["thresholds"] = thresholds
        public_module = _folder_datamodule(
            dataset_root=dataset_root, category=category, seed=seed, evaluation=True
        )
        artifact["predictions"], artifact["category_metrics"] = _evaluate(
            adapter=adapter,
            datamodule=public_module,
            dataset_root=dataset_root,
            run_dir=run_dir,
            thresholds=thresholds,
        )
        artifact["metric_implementation"] = METRIC_IMPLEMENTATION
        artifact["status"] = "completed"
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["failures"].append(
            {"type": type(exc).__name__, "message": str(exc), "recorded_at": _now()}
        )
        raise
    finally:
        artifact["resources"] = _finish_resources(resources)
        write_artifact(artifact_path, artifact)
    return artifact_path


def _artifact_metric(artifact: Mapping[str, Any], name: str) -> float:
    metrics = artifact.get("category_metrics")
    if not isinstance(metrics, dict):
        raise BenchmarkRunError("Completed artifact is missing category metrics")
    metric = metrics.get(name)
    if not isinstance(metric, dict) or metric.get("status") != "defined":
        raise BenchmarkRunError(f"Completed artifact has undefined metric {name}")
    value = metric.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BenchmarkRunError(f"Completed artifact has invalid metric {name}")
    return float(value)


def _load_completed_artifacts(
    manifest: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    cells = manifest.get("cells")
    if not isinstance(cells, dict):
        raise BenchmarkRunError("Manifest cells must be a mapping")
    for category, seed in benchmark_cells():
        entry = cells.get(f"{category}:{seed}")
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            raise BenchmarkRunError("Cannot aggregate an incomplete benchmark matrix")
        relative_path = entry.get("artifact_path")
        if not isinstance(relative_path, str):
            raise BenchmarkRunError("Completed manifest cell lacks an artifact path")
        artifact_path = output_root / relative_path
        if sha256_file(artifact_path) != entry.get("artifact_sha256"):
            raise BenchmarkRunError("Benchmark artifact hash differs from the manifest")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        validate_artifact(artifact)
        if artifact.get("status") != "completed":
            raise BenchmarkRunError("Manifest references a non-completed artifact")
        artifacts.append(artifact)
    return artifacts


def aggregate_benchmark(
    manifest: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Aggregate all 24 successful artifacts under the frozen mean/SD policy."""

    artifacts = _load_completed_artifacts(manifest, output_root)
    metric_names = ("au_pro_0.05", "pixel_f1", "image_f1", "image_auroc")
    per_category: dict[str, Any] = {}
    for category in OFFICIAL_CATEGORIES:
        category_artifacts = [a for a in artifacts if a["category"] == category]
        seed_values: dict[str, Any] = {}
        summary: dict[str, Any] = {}
        for name in metric_names:
            values = [
                _artifact_metric(artifact, name) for artifact in category_artifacts
            ]
            seed_values[name] = {
                str(artifact["seed"]): value
                for artifact, value in zip(category_artifacts, values, strict=True)
            }
            summary[name] = {
                "mean": statistics.fmean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "count": len(values),
            }
        per_category[category] = {"per_seed": seed_values, "across_seeds": summary}
    overall = {
        name: {
            "unweighted_category_mean": statistics.fmean(
                per_category[category]["across_seeds"][name]["mean"]
                for category in OFFICIAL_CATEGORIES
            ),
            "category_count": len(OFFICIAL_CATEGORIES),
        }
        for name in metric_names
    }
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "protocol_id": manifest["protocol_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "benchmark_git_commit": manifest["benchmark_git_commit"],
        "dataset_audit_sha256": manifest["dataset_audit_sha256"],
        "weight_sha256": manifest["weight_sha256"],
        "run_count": len(artifacts),
        "failed_attempt_count": sum(
            len(entry.get("failed_attempts", []))
            for entry in manifest["cells"].values()
        ),
        "aggregation": {
            "seeds": "unweighted_mean_and_sample_standard_deviation",
            "categories": "unweighted_arithmetic_mean",
            "best_seed_selection": False,
        },
        "per_category": per_category,
        "overall": overall,
    }


def _new_manifest(
    *,
    protocol_document: Mapping[str, Any],
    git: Mapping[str, Any],
    dataset: Mapping[str, Any],
    weight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "running",
        "protocol_id": protocol_document["protocol"]["id"],
        "protocol_fingerprint": protocol_fingerprint(protocol_document),
        "benchmark_git_commit": git["commit"],
        "dataset_audit_sha256": dataset["sha256"],
        "weight_sha256": weight["sha256"],
        "matrix": {
            "categories": list(OFFICIAL_CATEGORIES),
            "seeds": list(PROTOCOL_SEEDS),
            "expected_run_count": len(benchmark_cells()),
        },
        "cells": {
            f"{category}:{seed}": {
                "category": category,
                "seed": seed,
                "status": "pending",
                "failed_attempts": [],
            }
            for category, seed in benchmark_cells()
        },
    }


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_document: Mapping[str, Any],
    git: Mapping[str, Any],
    dataset: Mapping[str, Any],
    weight: Mapping[str, Any],
) -> None:
    expected = {
        "protocol_id": protocol_document["protocol"]["id"],
        "protocol_fingerprint": protocol_fingerprint(protocol_document),
        "benchmark_git_commit": git["commit"],
        "dataset_audit_sha256": dataset["sha256"],
        "weight_sha256": weight["sha256"],
    }
    differences = [key for key, value in expected.items() if manifest.get(key) != value]
    if differences:
        raise BenchmarkRunError(
            "Resume manifest identity drift: " + ", ".join(sorted(differences))
        )


def run_matrix(
    *,
    repository: Path,
    dataset_root: Path,
    audit_report: Path,
    protocol_path: Path,
    output_root: Path,
    explicit_benchmark_mode: bool,
    resume: bool,
) -> Path:
    """Run or resume the complete frozen 24-cell public benchmark."""

    repository = repository.resolve()
    dataset_root = dataset_root.resolve()
    audit_report = audit_report.resolve()
    output_root = output_root.resolve()
    protocol_document = load_protocol(protocol_path.resolve())
    git = capture_git_state(repository)
    dataset = dataset_audit_identity(audit_report, expected_root=dataset_root)
    environment = capture_environment("cuda")
    weight = _weight_identity(
        str(protocol_document["protocol"]["model"]["weight_revision"])
    )
    authorize_public_benchmark(
        protocol_document,
        BenchmarkGateInputs(
            explicit_benchmark_mode=explicit_benchmark_mode,
            evaluation_split="test_public",
            git_dirty=bool(git["dirty"]),
            dataset_audit_status=str(dataset["status"]),
            weight_sha256=str(weight["sha256"]),
            resolved_versions=_resolved_versions(environment),
            categories=OFFICIAL_CATEGORIES,
            seeds=PROTOCOL_SEEDS,
            recorded_fingerprint=protocol_fingerprint(protocol_document),
        ),
    )
    manifest_path = output_root / "benchmark-manifest.json"
    if manifest_path.exists():
        if not resume:
            raise BenchmarkRunError(
                "Benchmark manifest exists; pass --resume to preserve prior evidence"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(
            manifest,
            protocol_document=protocol_document,
            git=git,
            dataset=dataset,
            weight=weight,
        )
    else:
        manifest = _new_manifest(
            protocol_document=protocol_document,
            git=git,
            dataset=dataset,
            weight=weight,
        )
        _json_atomic(manifest_path, manifest)

    completed = {
        (entry["category"], entry["seed"])
        for entry in manifest["cells"].values()
        if entry.get("status") == "completed"
    }
    for category, seed in pending_cells(completed):
        key = f"{category}:{seed}"
        entry = manifest["cells"][key]
        attempt_number = len(entry.get("failed_attempts", [])) + 1
        relative_run_dir = (
            Path("runs") / category / f"seed-{seed}" / f"attempt-{attempt_number}"
        )
        run_dir = output_root / relative_run_dir
        entry["status"] = "running"
        entry["started_at"] = _now()
        entry["attempt"] = attempt_number
        manifest["updated_at"] = _now()
        _json_atomic(manifest_path, manifest)
        LOGGER.info(
            "Starting benchmark cell %s seed %d (attempt %d)",
            category,
            seed,
            attempt_number,
        )
        try:
            artifact_path = run_cell(
                repository=repository,
                dataset_root=dataset_root,
                audit_report=audit_report,
                protocol_document=protocol_document,
                category=category,
                seed=seed,
                run_dir=run_dir,
                explicit_benchmark_mode=explicit_benchmark_mode,
            )
        except Exception as exc:
            failure = {
                "attempt": attempt_number,
                "recorded_at": _now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "run_dir": relative_run_dir.as_posix(),
            }
            artifact_path = run_dir / "benchmark-artifact.json"
            if artifact_path.is_file():
                failure["artifact_path"] = artifact_path.relative_to(
                    output_root
                ).as_posix()
                failure["artifact_sha256"] = sha256_file(artifact_path)
            entry.setdefault("failed_attempts", []).append(failure)
            entry["status"] = "failed"
            entry["finished_at"] = _now()
            manifest["status"] = "interrupted"
            manifest["updated_at"] = _now()
            _json_atomic(manifest_path, manifest)
            raise
        relative_artifact = artifact_path.relative_to(output_root).as_posix()
        entry.update(
            {
                "status": "completed",
                "finished_at": _now(),
                "artifact_path": relative_artifact,
                "artifact_sha256": sha256_file(artifact_path),
            }
        )
        manifest["updated_at"] = _now()
        _json_atomic(manifest_path, manifest)

    manifest["status"] = "completed"
    manifest["completed_at"] = _now()
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    summary = aggregate_benchmark(manifest, output_root)
    summary_path = output_root / "benchmark-summary.json"
    _json_atomic(summary_path, summary)
    manifest["summary_path"] = summary_path.relative_to(output_root).as_posix()
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately explicit public benchmark CLI."""

    parser = argparse.ArgumentParser(
        description="Execute or resume the frozen 24-run PatchCore public benchmark."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/protocols/patchcore-mvtecad2-v1.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/phase2c-public-benchmark"),
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Explicitly authorize the frozen test_public benchmark gate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only cells not already recorded as completed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public benchmark while preserving durable failure evidence."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        summary = run_matrix(
            repository=args.repository,
            dataset_root=args.dataset_root,
            audit_report=args.audit_report,
            protocol_path=args.protocol,
            output_root=args.output_root,
            explicit_benchmark_mode=args.benchmark_mode,
            resume=args.resume,
        )
    except (
        ArtifactError,
        BenchmarkRunError,
        ModelDependencyError,
        ProtocolError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Benchmark interrupted by %s", type(exc).__name__)
        return 1
    LOGGER.info("All 24 public benchmark cells completed; summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
