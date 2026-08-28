"""Frozen preprocessing helpers at the model/original-coordinate boundary."""

from __future__ import annotations

from typing import Any


class PreprocessingError(ValueError):
    """Raised when an image or anomaly map violates the frozen shape contract."""


def restore_anomaly_map(
    anomaly_map: Any, original_height_width: tuple[int, int]
) -> Any:
    """Bilinearly resize one finite 2-D/CHW score map to original coordinates."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise PreprocessingError("Anomaly-map restoration requires PyTorch") from exc
    if len(original_height_width) != 2 or any(
        not isinstance(value, int) or value <= 0 for value in original_height_width
    ):
        raise PreprocessingError("Original image dimensions must be positive integers")
    if not isinstance(anomaly_map, torch.Tensor) or anomaly_map.ndim not in {2, 3}:
        raise PreprocessingError("Anomaly map must be a 2-D or CHW torch tensor")
    if not bool(torch.isfinite(anomaly_map).all().item()):
        raise PreprocessingError("Anomaly map must contain only finite scores")
    original_ndim = anomaly_map.ndim
    batched = (
        anomaly_map.unsqueeze(0).unsqueeze(0)
        if original_ndim == 2
        else anomaly_map.unsqueeze(0)
    )
    if batched.shape[1] != 1:
        raise PreprocessingError("Anomaly map must have exactly one channel")
    restored = functional.interpolate(
        batched,
        size=original_height_width,
        mode="bilinear",
        align_corners=False,
    )
    return restored[0, 0] if original_ndim == 2 else restored[0]
