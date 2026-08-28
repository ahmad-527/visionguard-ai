"""VisionGuard-owned contracts around Anomalib EfficientAD."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from visionguard.artifacts import sha256_file
from visionguard.calibration import highest_order_statistic, per_image_pixel_maxima
from visionguard.preprocessing import restore_anomaly_map


class EfficientAdError(ValueError):
    """Raised when EfficientAD inputs or measured outputs violate the protocol."""


def verify_file_identity(path: Path, expected_sha256: str, component: str) -> str:
    """Require an existing regular file with the exact predeclared SHA-256."""

    if len(expected_sha256) != 64:
        raise EfficientAdError(f"{component} expected SHA-256 is malformed")
    if not path.is_file():
        raise EfficientAdError(f"{component} file is missing: {path}")
    measured = sha256_file(path)
    if measured != expected_sha256.lower():
        raise EfficientAdError(f"{component} SHA-256 mismatch")
    return measured


def validate_finite_scores(values: Sequence[float], location: str) -> list[float]:
    """Convert scores to floats while rejecting NaN and infinity."""

    converted = [float(value) for value in values]
    if not converted or any(not math.isfinite(value) for value in converted):
        raise EfficientAdError(f"{location} must contain finite scores")
    return converted


def calibrate_efficientad_thresholds(
    image_scores: Sequence[float],
    restored_maps: Sequence[Sequence[float]],
    *,
    minimum_samples: int,
) -> dict[str, dict[str, float | int | str]]:
    """Calibrate operational decisions from validation-normal images only."""

    images = highest_order_statistic(
        validate_finite_scores(image_scores, "image calibration"),
        minimum_samples=minimum_samples,
    )
    pixels = highest_order_statistic(
        per_image_pixel_maxima(restored_maps), minimum_samples=minimum_samples
    )
    return {
        "image": images.__dict__,
        "pixel": pixels.__dict__,
    }


def canonical_checkpoint_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash ordered tensor names, dtypes, shapes, and raw CPU values."""

    try:
        import torch
    except ImportError as exc:
        raise EfficientAdError(
            "Checkpoint identity requires NumPy and PyTorch"
        ) from exc
    digest = hashlib.sha256()
    if not state_dict:
        raise EfficientAdError("Checkpoint state cannot be empty")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise EfficientAdError("Checkpoint state must map names to tensors")
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise EfficientAdError("Checkpoint contains NaN or infinity")
        header = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode()
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(value.numel().to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def restore_efficientad_map(
    anomaly_map: Any, original_height_width: tuple[int, int]
) -> Any:
    """Restore a finite EfficientAD map to decoded original coordinates."""

    return restore_anomaly_map(anomaly_map, original_height_width)
