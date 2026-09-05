"""Durable execution of the frozen 24-cell EfficientAD public benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from visionguard.artifacts import capture_git_state, dataset_audit_identity, sha256_file
from visionguard.benchmark import (
    METRIC_IMPLEMENTATION,
    _ground_truth,
    _json_atomic,
    _metric_payload,
    _now,
    _resolved_versions,
    _write_maps,
    benchmark_cells,
)
from visionguard.benchmark_metrics import (
    BinaryCountAccumulator,
    Float16AuProAccumulator,
)
from visionguard.calibration import highest_order_statistic
from visionguard.efficientad import (
    canonical_checkpoint_sha256,
    restore_efficientad_map,
    verify_file_identity,
)
from visionguard.efficientad_artifacts import validate_efficientad_artifact
from visionguard.efficientad_protocol import (
    IMAGENETTE_ARCHIVE_SHA256,
    TEACHER_SMALL_SHA256,
    EfficientAdGateInputs,
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
    validate_future_benchmark_prerequisites,
)
from visionguard.environment import capture_environment
from visionguard.experiment import ReproducibilityConfig
from visionguard.metrics import binary_auroc, binary_f1
from visionguard.paths import portable_relative_path
from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS
from visionguard.reproducibility import configure_reproducibility

LOGGER = logging.getLogger(__name__)
CHECKPOINT_INTERVAL = 1000
MANIFEST_SCHEMA_VERSION = 2


class EfficientAdBenchmarkError(RuntimeError):
    """Raised when benchmark execution or durable evidence is invalid."""


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cell_key(category: str, seed: int) -> str:
    return f"{category}:{seed}"


def _attempt_dir(output_root: Path, category: str, seed: int, attempt: int) -> Path:
    return output_root / "runs" / category / f"seed-{seed}" / f"attempt-{attempt}"


def _checkpoint_relative(category: str, seed: int, attempt: int) -> Path:
    return (
        Path("runs")
        / category
        / f"seed-{seed}"
        / f"attempt-{attempt}"
        / "latest-checkpoint.pt"
    )


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise EfficientAdBenchmarkError("Checkpointing requires PyTorch") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return sha256_file(path)


def _load_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise EfficientAdBenchmarkError("Checkpoint loading requires PyTorch") from exc
    if not path.is_file():
        raise EfficientAdBenchmarkError("Referenced training checkpoint is missing")
    if sha256_file(path) != expected_sha256:
        raise EfficientAdBenchmarkError("Training checkpoint SHA-256 mismatch")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise EfficientAdBenchmarkError("Training checkpoint is corrupt") from exc
    if not isinstance(value, dict):
        raise EfficientAdBenchmarkError("Training checkpoint payload is invalid")
    return value


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _image_paths(root: Path, category: str, split: str, condition: str) -> list[Path]:
    paths = sorted((root / category / split / condition).glob("*.png"))
    if not paths:
        raise EfficientAdBenchmarkError(
            f"No {category}/{split}/{condition} images found"
        )
    return paths


def _penalty_paths(root: Path) -> list[Path]:
    extensions = {".jpeg", ".jpg", ".png"}
    paths = sorted(
        path
        for path in (root / "train").rglob("*")
        if path.suffix.lower() in extensions
    )
    if not paths:
        raise EfficientAdBenchmarkError("ImageNette train images are missing")
    return paths


def _new_stream_state(length: int, seed: int) -> dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=generator).tolist()
    return {"order": order, "position": 0, "generator_state": generator.get_state()}


def _next_stream_index(stream: dict[str, Any], length: int) -> int:
    import torch

    if stream["position"] >= len(stream["order"]):
        generator = torch.Generator()
        generator.set_state(stream["generator_state"])
        stream["order"] = torch.randperm(length, generator=generator).tolist()
        stream["position"] = 0
        stream["generator_state"] = generator.get_state()
    index = int(stream["order"][stream["position"]])
    stream["position"] += 1
    return index


def _load_transforms() -> tuple[Any, Any]:
    from torchvision.transforms.v2 import (
        CenterCrop,
        Compose,
        RandomGrayscale,
        Resize,
        ToTensor,
    )

    normal = Compose([Resize((256, 256), antialias=True), ToTensor()])
    penalty = Compose(
        [
            Resize((512, 512)),
            RandomGrayscale(p=0.3),
            CenterCrop((256, 256)),
            ToTensor(),
        ]
    )
    return normal, penalty


def _load_image(path: Path, transform: Any) -> tuple[Any, tuple[int, int]]:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        shape = (rgb.height, rgb.width)
        return transform(rgb).unsqueeze(0), shape


def _teacher_statistics(
    model: Any, paths: Sequence[Path], transform: Any, device: Any
) -> None:
    import torch

    channel_sum = torch.zeros(384, device=device)
    channel_sum_sqr = torch.zeros(384, device=device)
    count = 0
    with torch.no_grad():
        for path in paths:
            image, _ = _load_image(path, transform)
            features = model.teacher(image.to(device))
            count += features[:, 0].numel()
            channel_sum += features.sum(dim=(0, 2, 3))
            channel_sum_sqr += (features**2).sum(dim=(0, 2, 3))
    mean = channel_sum / count
    std = torch.sqrt(channel_sum_sqr / count - mean**2)
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise EfficientAdBenchmarkError("Teacher statistics are non-finite")
    model.mean_std.update(
        {"mean": mean[None, :, None, None], "std": std[None, :, None, None]}
    )


def _build_training(
    *, teacher_weight: Path, device: Any, protocol: Mapping[str, Any]
) -> tuple[Any, Any, Any]:
    import torch
    from anomalib.models.image.efficient_ad.torch_model import (
        EfficientAdModel,
        EfficientAdModelSize,
    )

    model = EfficientAdModel(
        teacher_out_channels=384,
        model_size=EfficientAdModelSize.S,
        padding=False,
        pad_maps=True,
    ).to(device)
    model.teacher.load_state_dict(
        torch.load(teacher_weight, map_location=device, weights_only=True)
    )
    model.teacher.eval()
    optimizer = torch.optim.Adam(
        list(model.student.parameters()) + list(model.ae.parameters()),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(
            protocol["training"]["scheduler_step_fraction"]
            * protocol["training"]["max_steps"]
        ),
        gamma=float(protocol["training"]["scheduler_gamma"]),
    )
    return model, optimizer, scheduler


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    active_seconds: float,
    train_stream: Mapping[str, Any],
    penalty_stream: Mapping[str, Any],
    identity: Mapping[str, Any],
    progress: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": dict(identity),
        "step": step,
        "active_training_seconds": active_seconds,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng": _rng_state(),
        "train_stream": dict(train_stream),
        "penalty_stream": dict(penalty_stream),
        "progress": [dict(record) for record in progress],
    }


def _validate_checkpoint_identity(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    actual = checkpoint.get("identity")
    if not isinstance(actual, dict) or actual != dict(expected):
        raise EfficientAdBenchmarkError("Training checkpoint identity mismatch")
    step = checkpoint.get("step")
    if not isinstance(step, int) or not 0 <= step <= 70000:
        raise EfficientAdBenchmarkError("Training checkpoint step is invalid")


def _persist_checkpoint(
    *,
    checkpoint_path: Path,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    payload: Mapping[str, Any],
    output_root: Path,
) -> None:
    checkpoint_sha = _atomic_torch_save(checkpoint_path, payload)
    entry["latest_valid_checkpoint"] = {
        "path": checkpoint_path.relative_to(output_root).as_posix(),
        "sha256": checkpoint_sha,
        "step": payload["step"],
        "recorded_at": _now(),
    }
    entry["status"] = "training"
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)


def _train(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    train_paths: Sequence[Path],
    penalty_paths: Sequence[Path],
    normal_transform: Any,
    penalty_transform: Any,
    device: Any,
    protocol: Mapping[str, Any],
    identity: Mapping[str, Any],
    entry: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    checkpoint: Mapping[str, Any] | None,
    stop_after_step: int | None = None,
) -> tuple[float, list[dict[str, float | int]]]:
    import torch

    if checkpoint is None:
        _teacher_statistics(model, train_paths, normal_transform, device)
        step = 0
        active_seconds = 0.0
        train_stream = _new_stream_state(len(train_paths), int(identity["seed"]))
        penalty_stream = _new_stream_state(
            len(penalty_paths), int(identity["seed"]) + 1
        )
        progress: list[dict[str, float | int]] = []
        _persist_checkpoint(
            checkpoint_path=checkpoint_path,
            entry=entry,
            manifest=manifest,
            manifest_path=manifest_path,
            payload=_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                active_seconds=active_seconds,
                train_stream=train_stream,
                penalty_stream=penalty_stream,
                identity=identity,
                progress=progress,
            ),
            output_root=output_root,
        )
    else:
        _validate_checkpoint_identity(checkpoint, identity)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        _restore_rng_state(checkpoint["rng"])
        step = int(checkpoint["step"])
        active_seconds = float(checkpoint["active_training_seconds"])
        train_stream = dict(checkpoint["train_stream"])
        penalty_stream = dict(checkpoint["penalty_stream"])
        stored_progress = checkpoint.get("progress", [])
        if not isinstance(stored_progress, list):
            raise EfficientAdBenchmarkError("Checkpoint progress history is invalid")
        progress = [dict(record) for record in stored_progress]
        if not checkpoint_path.exists():
            _persist_checkpoint(
                checkpoint_path=checkpoint_path,
                entry=entry,
                manifest=manifest,
                manifest_path=manifest_path,
                payload=_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    active_seconds=active_seconds,
                    train_stream=train_stream,
                    penalty_stream=penalty_stream,
                    identity=identity,
                    progress=progress,
                ),
                output_root=output_root,
            )

    max_steps = int(protocol["training"]["max_steps"])
    segment_started = time.perf_counter()
    model.train()
    while step < max_steps:
        train_index = _next_stream_index(train_stream, len(train_paths))
        penalty_index = _next_stream_index(penalty_stream, len(penalty_paths))
        image, _ = _load_image(train_paths[train_index], normal_transform)
        penalty, _ = _load_image(penalty_paths[penalty_index], penalty_transform)
        optimizer.zero_grad(set_to_none=True)
        losses = model(image.to(device), batch_imagenet=penalty.to(device))
        loss = sum(losses)
        if not bool(torch.isfinite(loss)):
            raise EfficientAdBenchmarkError("Training produced NaN or infinity")
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        if step % CHECKPOINT_INTERVAL == 0 or step in (max_steps, stop_after_step):
            torch.cuda.synchronize()
            active_seconds += time.perf_counter() - segment_started
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "active_training_seconds": active_seconds,
            }
            progress.append(record)
            LOGGER.info(
                "%s seed %s step %d/%d lr %.8g active %.1fs",
                identity["category"],
                identity["seed"],
                step,
                max_steps,
                record["learning_rate"],
                active_seconds,
            )
            _persist_checkpoint(
                checkpoint_path=checkpoint_path,
                entry=entry,
                manifest=manifest,
                manifest_path=manifest_path,
                payload=_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    active_seconds=active_seconds,
                    train_stream=train_stream,
                    penalty_stream=penalty_stream,
                    identity=identity,
                    progress=progress,
                ),
                output_root=output_root,
            )
            segment_started = time.perf_counter()
            if step == stop_after_step:
                raise KeyboardInterrupt
    return active_seconds, progress


def _native_quantiles(
    model: Any, validation_paths: Sequence[Path], transform: Any, device: Any
) -> dict[str, float]:
    import torch
    from anomalib.models.image.efficient_ad.torch_model import reduce_tensor_elems

    maps_st: list[Any] = []
    maps_ae: list[Any] = []
    model.eval()
    with torch.no_grad():
        for path in validation_paths:
            image, _ = _load_image(path, transform)
            st, ae = model.get_maps(image.to(device), normalize=False)
            maps_st.append(st)
            maps_ae.append(ae)
    st_flat = reduce_tensor_elems(torch.cat(maps_st))
    ae_flat = reduce_tensor_elems(torch.cat(maps_ae))
    tensors = {
        "qa_st": torch.quantile(st_flat, 0.9),
        "qb_st": torch.quantile(st_flat, 0.995),
        "qa_ae": torch.quantile(ae_flat, 0.9),
        "qb_ae": torch.quantile(ae_flat, 0.995),
    }
    if any(not bool(torch.isfinite(value)) for value in tensors.values()):
        raise EfficientAdBenchmarkError("Native normalization is non-finite")
    model.quantiles.update(tensors)
    return {key: float(value.detach().cpu()) for key, value in tensors.items()}


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _calibrate(
    *,
    model: Any,
    validation_paths: Sequence[Path],
    transform: Any,
    device: Any,
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    image_scores: list[float] = []
    pixel_maxima: list[float] = []
    inputs: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for path in validation_paths:
            image, original_shape = _load_image(path, transform)
            output = model(image.to(device))
            restored = restore_efficientad_map(output.anomaly_map[0, 0], original_shape)
            score = float(output.pred_score[0].cpu())
            pixel_maximum = float(restored.max().cpu())
            if not math.isfinite(score) or not math.isfinite(pixel_maximum):
                raise EfficientAdBenchmarkError(
                    "Calibration produced non-finite scores"
                )
            image_scores.append(score)
            pixel_maxima.append(pixel_maximum)
            inputs.append(
                {
                    "sample_id": path.relative_to(dataset_root).as_posix(),
                    "image_anomaly_score": score,
                    "pixel_maximum": pixel_maximum,
                    "restored_map_sha256": _tensor_sha256(restored),
                    "restored_map_shape": list(restored.shape),
                }
            )
    image = highest_order_statistic(image_scores, minimum_samples=19)
    pixel = highest_order_statistic(pixel_maxima, minimum_samples=19)
    return (
        {
            "normal_only": True,
            "split": "validation",
            "comparison": "score_strictly_greater_than_threshold",
            "inputs": inputs,
            "image": dataclasses.asdict(image),
            "pixel": dataclasses.asdict(pixel),
        },
        {"image": image.threshold, "pixel": pixel.threshold},
    )


def _evaluate(
    *,
    model: Any,
    public_paths: Sequence[Path],
    transform: Any,
    device: Any,
    dataset_root: Path,
    run_dir: Path,
    thresholds: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    image_labels: list[int] = []
    image_scores: list[float] = []
    image_decisions: list[int] = []
    pixel_f1 = BinaryCountAccumulator()
    au_pro = Float16AuProAccumulator()
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for path in public_paths:
            label, mask = _ground_truth(path)
            image, _ = _load_image(path, transform)
            output = model(image.to(device))
            restored = restore_efficientad_map(
                output.anomaly_map[0, 0], tuple(mask.shape)
            )
            sample_id = path.relative_to(dataset_root).as_posix()
            map_identity, official_map, thresholded = _write_maps(
                run_dir=run_dir,
                sample_id=sample_id,
                restored=restored,
                pixel_threshold=float(thresholds["pixel"]),
            )
            score = float(output.pred_score[0].cpu())
            if not math.isfinite(score):
                raise EfficientAdBenchmarkError("Public score is non-finite")
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
    if any(metric["status"] != "defined" for metric in metrics.values()):
        raise EfficientAdBenchmarkError("A frozen public metric is undefined")
    return records, metrics


def _checkpoint_identity(
    *, manifest: Mapping[str, Any], category: str, seed: int
) -> dict[str, Any]:
    return {
        "protocol_id": manifest["protocol_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "benchmark_git_commit": manifest["benchmark_git_commit"],
        "dataset_audit_sha256": manifest["dataset_audit_sha256"],
        "teacher_weight_sha256": manifest["teacher_weight_sha256"],
        "teacher_archive_sha256": manifest["teacher_archive_sha256"],
        "imagenette_archive_sha256": manifest["imagenette_archive_sha256"],
        "environment_sha256": manifest["environment_sha256"],
        "category": category,
        "seed": seed,
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    validate_efficientad_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise EfficientAdBenchmarkError(
            "Refusing to overwrite benchmark artifact"
        ) from exc


def _verify_completed_cell(
    entry: Mapping[str, Any], output_root: Path, manifest: Mapping[str, Any]
) -> None:
    try:
        artifact_relative = portable_relative_path(str(entry["artifact_path"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise EfficientAdBenchmarkError("Completed artifact path is invalid") from exc
    artifact_path = output_root / artifact_relative
    if not artifact_path.is_file() or sha256_file(artifact_path) != entry.get(
        "artifact_sha256"
    ):
        raise EfficientAdBenchmarkError("Completed artifact identity is invalid")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_efficientad_artifact(artifact)
    if artifact.get("status") != "completed":
        raise EfficientAdBenchmarkError("Completed cell references incomplete artifact")
    expected_identity = {
        "category": entry.get("category"),
        "seed": entry.get("seed"),
        "protocol_id": manifest.get("protocol_id"),
        "protocol_fingerprint": manifest.get("protocol_fingerprint"),
    }
    if any(artifact.get(key) != value for key, value in expected_identity.items()):
        raise EfficientAdBenchmarkError("Completed artifact cell identity is invalid")
    if (
        artifact.get("git", {}).get("commit") != manifest.get("benchmark_git_commit")
        or artifact.get("dataset", {}).get("sha256")
        != manifest.get("dataset_audit_sha256")
        or artifact.get("auxiliary_data", {}).get("archive_sha256")
        != manifest.get("imagenette_archive_sha256")
        or artifact.get("weights", [{}])[0].get("sha256")
        != manifest.get("teacher_weight_sha256")
        or artifact.get("weights", [{}])[0].get("archive_sha256")
        != manifest.get("teacher_archive_sha256")
        or artifact.get("environment_sha256") != manifest.get("environment_sha256")
        or _canonical_json_sha256(artifact.get("environment", {}))
        != manifest.get("environment_sha256")
    ):
        raise EfficientAdBenchmarkError("Completed artifact provenance is invalid")
    model_state = artifact.get("model_state")
    if not isinstance(model_state, dict):
        raise EfficientAdBenchmarkError("Completed model-state identity is invalid")
    try:
        checkpoint_relative = portable_relative_path(
            str(model_state["checkpoint_path"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EfficientAdBenchmarkError("Completed checkpoint path is invalid") from exc
    checkpoint_path = output_root / checkpoint_relative
    checkpoint_sha = str(model_state.get("checkpoint_sha256", ""))
    if (
        checkpoint_sha != entry.get("checkpoint_sha256")
        or not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != checkpoint_sha
    ):
        raise EfficientAdBenchmarkError("Completed checkpoint identity is invalid")
    run_dir = artifact_path.parent
    for prediction in artifact["predictions"]:
        anomaly_map = prediction["anomaly_map"]
        for key, hash_key in (
            ("path", "sha256"),
            ("thresholded_path", "thresholded_sha256"),
        ):
            try:
                map_path = run_dir / portable_relative_path(anomaly_map[key])
            except (TypeError, ValueError) as exc:
                raise EfficientAdBenchmarkError(
                    "Completed anomaly-map path is invalid"
                ) from exc
            if not map_path.is_file() or sha256_file(map_path) != anomaly_map[hash_key]:
                raise EfficientAdBenchmarkError(
                    "Completed anomaly-map identity is invalid"
                )


def _new_manifest(
    *,
    document: Mapping[str, Any],
    git: Mapping[str, Any],
    dataset: Mapping[str, Any],
    environment: Mapping[str, Any],
    teacher_sha: str,
    teacher_archive_sha: str,
    imagenette_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "running",
        "protocol_id": document["protocol"]["id"],
        "protocol_fingerprint": efficientad_protocol_fingerprint(document),
        "benchmark_git_commit": git["commit"],
        "dataset_audit_sha256": dataset["sha256"],
        "teacher_weight_sha256": teacher_sha,
        "teacher_archive_sha256": teacher_archive_sha,
        "imagenette_archive_sha256": imagenette_sha,
        "environment": environment,
        "environment_sha256": _canonical_json_sha256(environment),
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
        "matrix": {
            "categories": list(OFFICIAL_CATEGORIES),
            "seeds": list(PROTOCOL_SEEDS),
            "expected_run_count": 24,
            "order": "category_major",
        },
        "cells": {
            _cell_key(category, seed): {
                "category": category,
                "seed": seed,
                "status": "pending",
                "attempts": [],
                "interruption_history": [],
                "failure_history": [],
            }
            for category, seed in benchmark_cells()
        },
    }


def _validate_manifest(
    manifest: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EfficientAdBenchmarkError("Manifest schema is invalid")
    drift = [key for key, value in expected.items() if manifest.get(key) != value]
    if drift:
        raise EfficientAdBenchmarkError(
            "Resume manifest identity drift: " + ", ".join(drift)
        )
    cells = manifest.get("cells")
    if not isinstance(cells, dict) or set(cells) != {
        _cell_key(category, seed) for category, seed in benchmark_cells()
    }:
        raise EfficientAdBenchmarkError(
            "Manifest matrix is incomplete or contains unknown cells"
        )
    statuses = {
        "pending",
        "training",
        "trained",
        "calibration",
        "evaluating",
        "completed",
        "failed",
        "interrupted",
        "invalid",
    }
    for category, seed in benchmark_cells():
        entry = cells[_cell_key(category, seed)]
        if (
            not isinstance(entry, dict)
            or entry.get("category") != category
            or entry.get("seed") != seed
            or entry.get("status") not in statuses
            or not isinstance(entry.get("attempts"), list)
            or not isinstance(entry.get("interruption_history"), list)
            or not isinstance(entry.get("failure_history"), list)
        ):
            raise EfficientAdBenchmarkError("Manifest cell integrity is invalid")


def _artifact_metric(artifact: Mapping[str, Any], name: str) -> float:
    metric = artifact["category_metrics"].get(name)
    if not isinstance(metric, dict) or metric.get("status") != "defined":
        raise EfficientAdBenchmarkError(f"Metric {name} is undefined")
    value = float(metric["value"])
    if not math.isfinite(value):
        raise EfficientAdBenchmarkError(f"Metric {name} is non-finite")
    return value


def aggregate_benchmark(
    manifest: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Refuse partial matrices and aggregate all 24 cells without selection."""

    artifacts: list[dict[str, Any]] = []
    for category, seed in benchmark_cells():
        entry = manifest["cells"][_cell_key(category, seed)]
        if entry.get("status") != "completed":
            raise EfficientAdBenchmarkError(
                "Cannot aggregate incomplete benchmark matrix"
            )
        _verify_completed_cell(entry, output_root, manifest)
        path = output_root / entry["artifact_path"]
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    names = ("au_pro_0.05", "pixel_f1", "image_f1", "image_auroc")
    per_category: dict[str, Any] = {}
    for category in OFFICIAL_CATEGORIES:
        selected = [value for value in artifacts if value["category"] == category]
        per_seed: dict[str, Any] = {}
        across: dict[str, Any] = {}
        for name in names:
            values = [_artifact_metric(artifact, name) for artifact in selected]
            per_seed[name] = {
                str(artifact["seed"]): value
                for artifact, value in zip(selected, values, strict=True)
            }
            across[name] = {
                "mean": statistics.fmean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "count": 3,
            }
        per_category[category] = {"per_seed": per_seed, "across_seeds": across}
    overall = {
        name: {
            "unweighted_category_mean": statistics.fmean(
                per_category[category]["across_seeds"][name]["mean"]
                for category in OFFICIAL_CATEGORIES
            ),
            "category_count": 8,
        }
        for name in names
    }
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "protocol_id": manifest["protocol_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "benchmark_git_commit": manifest["benchmark_git_commit"],
        "dataset_audit_sha256": manifest["dataset_audit_sha256"],
        "teacher_weight_sha256": manifest["teacher_weight_sha256"],
        "imagenette_archive_sha256": manifest["imagenette_archive_sha256"],
        "environment_sha256": manifest["environment_sha256"],
        "run_count": 24,
        "interrupted_attempt_count": sum(
            len(entry["interruption_history"]) for entry in manifest["cells"].values()
        ),
        "failed_attempt_count": sum(
            len(entry["failure_history"]) for entry in manifest["cells"].values()
        ),
        "aggregation": {
            "seeds": "unweighted_mean_and_sample_standard_deviation",
            "categories": "unweighted_arithmetic_mean",
            "selection": False,
        },
        "per_category": per_category,
        "overall": overall,
    }


def _run_cell(
    *,
    repository: Path,
    dataset_root: Path,
    document: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    category: str,
    seed: int,
    teacher_weight: Path,
    imagenette_root: Path,
    resume_checkpoint: Mapping[str, Any] | None,
    attempt: int,
) -> Path:
    import torch

    protocol = document["protocol"]
    device = torch.device("cuda")
    normal_transform, penalty_transform = _load_transforms()
    train_paths = _image_paths(dataset_root, category, "train", "good")
    validation_paths = _image_paths(dataset_root, category, "validation", "good")
    penalties = _penalty_paths(imagenette_root)
    model, optimizer, scheduler = _build_training(
        teacher_weight=teacher_weight, device=device, protocol=protocol
    )
    run_dir = _attempt_dir(output_root, category, seed, attempt)
    checkpoint_path = output_root / _checkpoint_relative(category, seed, attempt)
    entry = manifest["cells"][_cell_key(category, seed)]
    identity = _checkpoint_identity(manifest=manifest, category=category, seed=seed)
    active_seconds, progress = _train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_paths=train_paths,
        penalty_paths=penalties,
        normal_transform=normal_transform,
        penalty_transform=penalty_transform,
        device=device,
        protocol=protocol,
        identity=identity,
        entry=entry,
        manifest=manifest,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        checkpoint=resume_checkpoint,
    )
    entry["status"] = "trained"
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    checkpoint_sha = sha256_file(checkpoint_path)
    final_model_sha = canonical_checkpoint_sha256(model.state_dict())
    entry["status"] = "calibration"
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    native_quantiles = _native_quantiles(
        model, validation_paths, normal_transform, device
    )
    calibration, thresholds = _calibrate(
        model=model,
        validation_paths=validation_paths,
        transform=normal_transform,
        device=device,
        dataset_root=dataset_root,
    )
    calibration["native_map_normalization_quantiles"] = native_quantiles
    entry["status"] = "evaluating"
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    public_paths = sorted(
        _image_paths(dataset_root, category, "test_public", "good")
        + _image_paths(dataset_root, category, "test_public", "bad")
    )
    predictions, metrics = _evaluate(
        model=model,
        public_paths=public_paths,
        transform=normal_transform,
        device=device,
        dataset_root=dataset_root,
        run_dir=run_dir,
        thresholds=thresholds,
    )
    git = capture_git_state(repository)
    artifact_completed_at = _now()
    artifact_attempts = [dict(value) for value in entry["attempts"]]
    artifact_attempts[-1].update(
        {"status": "completed", "completed_at": artifact_completed_at}
    )
    artifact = {
        "artifact_schema_version": 3,
        "protocol_id": protocol["id"],
        "protocol_fingerprint": efficientad_protocol_fingerprint(document),
        "protocol_snapshot": protocol,
        "experiment_id": f"phase3b-{category}-seed-{seed}",
        "run_kind": "phase3b_protocol_authorized_public_benchmark",
        "benchmark_claim": True,
        "evaluation_split": "test_public",
        "git": git,
        "dataset": {
            "status": "passed",
            "sha256": manifest["dataset_audit_sha256"],
            "name": "mvtec_ad_2",
        },
        "category": category,
        "seed": seed,
        "implementation": {
            "name": "anomalib",
            "version": "2.6.0",
            "revision": protocol["model"]["implementation_revision"],
        },
        "model": protocol["model"],
        "training": {
            **protocol["training"],
            "final_optimization_step": 70000,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "active_training_seconds": active_seconds,
            "progress_checkpoints": progress,
        },
        "preprocessing": protocol["preprocessing"],
        "auxiliary_data": {
            **protocol["auxiliary_data"],
            "archive_sha256": manifest["imagenette_archive_sha256"],
        },
        "environment": manifest["environment"],
        "environment_sha256": manifest["environment_sha256"],
        "reproducibility": protocol["reproducibility"],
        "weights": [
            {
                "component": "teacher_pdn_small",
                "sha256": manifest["teacher_weight_sha256"],
                "archive_sha256": manifest["teacher_archive_sha256"],
            }
        ],
        "calibration": calibration,
        "thresholds": thresholds,
        "model_state": {
            "checkpoint_sha256": checkpoint_sha,
            "canonical_model_sha256": final_model_sha,
            "checkpoint_path": checkpoint_path.relative_to(output_root).as_posix(),
        },
        "predictions": predictions,
        "metrics": [],
        "metric_implementation": METRIC_IMPLEMENTATION,
        "category_metrics": metrics,
        "aggregation": "unweighted_arithmetic_mean",
        "resources": {
            "active_training_seconds": active_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "formal_inference_measurement": False,
        },
        "warnings": [
            "Training time is operational, excludes stopped-process intervals, "
            "and may include OS sleep within a checkpoint interval.",
            "No formal inference-latency claim was measured.",
        ],
        "failures": [dict(value) for value in entry["failure_history"]],
        "execution_history": {
            "attempts": artifact_attempts,
            "interruptions": [dict(value) for value in entry["interruption_history"]],
        },
        "completed_at": artifact_completed_at,
        "status": "completed",
    }
    artifact_path = run_dir / "benchmark-artifact.json"
    _write_artifact(artifact_path, artifact)
    return artifact_path


def run_matrix(
    *,
    repository: Path,
    dataset_root: Path,
    audit_report: Path,
    protocol_path: Path,
    teacher_weight: Path,
    teacher_archive: Path,
    imagenette_root: Path,
    imagenette_archive: Path,
    output_root: Path,
    explicit_benchmark_mode: bool,
    resume: bool,
) -> Path:
    """Execute or safely resume the category-major 24-cell matrix."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repository = repository.resolve()
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    document = load_efficientad_protocol(protocol_path.resolve())
    git = capture_git_state(repository)
    dataset = dataset_audit_identity(audit_report.resolve(), expected_root=dataset_root)
    environment = capture_environment("cuda")
    teacher_sha = verify_file_identity(
        teacher_weight.resolve(), TEACHER_SMALL_SHA256, "teacher"
    )
    teacher_archive_sha = verify_file_identity(
        teacher_archive.resolve(),
        str(document["protocol"]["model"]["teacher_weight_archive_sha256"]),
        "teacher archive",
    )
    imagenette_sha = verify_file_identity(
        imagenette_archive.resolve(), IMAGENETTE_ARCHIVE_SHA256, "ImageNette archive"
    )
    if not imagenette_root.resolve().joinpath("train").is_dir():
        raise EfficientAdBenchmarkError(
            "ImageNette extracted train directory is missing"
        )
    versions = _resolved_versions(environment)
    lightning_entry = environment["resolved_packages"].get("lightning")
    if not lightning_entry:
        raise EfficientAdBenchmarkError("Lightning version is unavailable")
    versions["lightning"] = str(lightning_entry)
    validate_future_benchmark_prerequisites(
        document,
        EfficientAdGateInputs(
            explicit_benchmark_mode=explicit_benchmark_mode,
            evaluation_split="test_public",
            git_dirty=bool(git["dirty"]),
            dataset_audit_status=str(dataset["status"]),
            teacher_weight_sha256=teacher_sha,
            auxiliary_archive_sha256=imagenette_sha,
            resolved_versions=versions,
            categories=OFFICIAL_CATEGORIES,
            seeds=PROTOCOL_SEEDS,
            recorded_fingerprint=efficientad_protocol_fingerprint(document),
        ),
    )
    manifest_path = output_root / "benchmark-manifest.json"
    expected = {
        "protocol_id": document["protocol"]["id"],
        "protocol_fingerprint": efficientad_protocol_fingerprint(document),
        "benchmark_git_commit": git["commit"],
        "dataset_audit_sha256": dataset["sha256"],
        "teacher_weight_sha256": teacher_sha,
        "teacher_archive_sha256": teacher_archive_sha,
        "imagenette_archive_sha256": imagenette_sha,
        "environment_sha256": _canonical_json_sha256(environment),
    }
    if manifest_path.exists():
        if not resume:
            raise EfficientAdBenchmarkError("Manifest exists; use --resume")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EfficientAdBenchmarkError(
                "Manifest is unreadable or partially written"
            ) from exc
        _validate_manifest(manifest, expected)
    else:
        if resume:
            raise EfficientAdBenchmarkError("Cannot resume without a manifest")
        if output_root.exists() and any(output_root.iterdir()):
            raise EfficientAdBenchmarkError(
                "Stale non-empty output directory has no manifest"
            )
        manifest = _new_manifest(
            document=document,
            git=git,
            dataset=dataset,
            environment=environment,
            teacher_sha=teacher_sha,
            teacher_archive_sha=teacher_archive_sha,
            imagenette_sha=imagenette_sha,
        )
        _json_atomic(manifest_path, manifest)

    for category, seed in benchmark_cells():
        key = _cell_key(category, seed)
        entry = manifest["cells"][key]
        if entry["status"] == "completed":
            _verify_completed_cell(entry, output_root, manifest)
            LOGGER.info("Verified and skipped completed cell %s", key)
            continue
        checkpoint: Mapping[str, Any] | None = None
        prior_status = entry["status"]
        if prior_status == "invalid":
            raise EfficientAdBenchmarkError(
                f"Cell {key} is invalid and cannot be resumed"
            )
        if prior_status in {
            "training",
            "trained",
            "calibration",
            "evaluating",
            "interrupted",
        }:
            latest = entry.get("latest_valid_checkpoint")
            if not isinstance(latest, dict):
                raise EfficientAdBenchmarkError(
                    "Incomplete cell lacks a valid checkpoint reference"
                )
            checkpoint = _load_checkpoint(
                output_root / latest["path"], str(latest["sha256"])
            )
            interrupted_stage = prior_status
            if prior_status == "interrupted" and entry["interruption_history"]:
                interrupted_stage = str(
                    entry["interruption_history"][-1].get("stage", "training")
                )
            if prior_status != "interrupted":
                entry["interruption_history"].append(
                    {
                        "recorded_at": _now(),
                        "reason": "resume_after_external_or_unclean_interruption",
                        "checkpoint_step": checkpoint["step"],
                        "stage": interrupted_stage,
                    }
                )
            if int(checkpoint["step"]) == 70000 and interrupted_stage in {
                "trained",
                "calibration",
                "evaluating",
            }:
                attempt = len(entry["attempts"]) + 1
                entry["attempts"].append(
                    {
                        "attempt": attempt,
                        "started_at": _now(),
                        "status": "running",
                        "recovery_from_attempt": entry["attempts"][-1]["attempt"],
                        "recovery_checkpoint_step": 70000,
                    }
                )
            else:
                attempt = int(entry["attempts"][-1]["attempt"])
        else:
            attempt = len(entry["attempts"]) + 1
            entry["attempts"].append(
                {"attempt": attempt, "started_at": _now(), "status": "running"}
            )
        entry["status"] = "training"
        manifest["status"] = "running"
        manifest["updated_at"] = _now()
        _json_atomic(manifest_path, manifest)
        reproducibility = configure_reproducibility(
            ReproducibilityConfig(
                seed=seed, deterministic_algorithms=True, cudnn_benchmark=False
            )
        )
        del reproducibility
        try:
            artifact_path = _run_cell(
                repository=repository,
                dataset_root=dataset_root,
                document=document,
                manifest=manifest,
                manifest_path=manifest_path,
                output_root=output_root,
                category=category,
                seed=seed,
                teacher_weight=teacher_weight.resolve(),
                imagenette_root=imagenette_root.resolve(),
                resume_checkpoint=checkpoint,
                attempt=attempt,
            )
        except KeyboardInterrupt:
            interrupted_from = str(entry["status"])
            entry["status"] = "interrupted"
            entry["interruption_history"].append(
                {
                    "recorded_at": _now(),
                    "reason": "KeyboardInterrupt",
                    "checkpoint_step": entry.get("latest_valid_checkpoint", {}).get(
                        "step"
                    ),
                    "stage": interrupted_from,
                }
            )
            entry["attempts"][-1]["status"] = "interrupted"
            manifest["status"] = "interrupted"
            manifest["updated_at"] = _now()
            _json_atomic(manifest_path, manifest)
            raise
        except Exception as exc:
            entry["status"] = "failed"
            failure = {
                "recorded_at": _now(),
                "attempt": attempt,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            entry["failure_history"].append(failure)
            entry["attempts"][-1].update({"status": "failed", "failure": failure})
            manifest["status"] = "failed"
            manifest["updated_at"] = _now()
            _json_atomic(manifest_path, manifest)
            raise
        relative_artifact = artifact_path.relative_to(output_root).as_posix()
        entry.update(
            {
                "status": "completed",
                "artifact_path": relative_artifact,
                "artifact_sha256": sha256_file(artifact_path),
                "checkpoint_sha256": entry["latest_valid_checkpoint"]["sha256"],
                "completed_at": _now(),
            }
        )
        entry["attempts"][-1].update({"status": "completed", "completed_at": _now()})
        manifest["updated_at"] = _now()
        _json_atomic(manifest_path, manifest)

    manifest["status"] = "completed"
    manifest["completed_at"] = _now()
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    summary = aggregate_benchmark(manifest, output_root)
    summary_path = output_root / "benchmark-summary.json"
    _json_atomic(summary_path, summary)
    manifest["summary_path"] = "benchmark-summary.json"
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["updated_at"] = _now()
    _json_atomic(manifest_path, manifest)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit, interruption-safe Phase 3B CLI."""

    parser = argparse.ArgumentParser(
        description="Execute or resume the frozen EfficientAD public benchmark."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/protocols/efficientad-mvtecad2-v1.yaml"),
    )
    parser.add_argument("--teacher-weight", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--imagenette-root", type=Path, required=True)
    parser.add_argument("--imagenette-archive", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/phase3b-efficientad-public-benchmark"),
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with durable external-interruption classification."""

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
            teacher_weight=args.teacher_weight,
            teacher_archive=args.teacher_archive,
            imagenette_root=args.imagenette_root,
            imagenette_archive=args.imagenette_archive,
            output_root=args.output_root,
            explicit_benchmark_mode=args.benchmark_mode,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Benchmark interrupted; resume with --resume")
        return 130
    except Exception as exc:
        LOGGER.exception("EfficientAD benchmark stopped: %s", exc)
        return 1
    LOGGER.info("All 24 cells completed: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
