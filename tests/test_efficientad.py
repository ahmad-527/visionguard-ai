from __future__ import annotations

from pathlib import Path

import pytest

from visionguard.efficientad import (
    EfficientAdError,
    calibrate_efficientad_thresholds,
    canonical_checkpoint_sha256,
    restore_efficientad_map,
    validate_finite_scores,
    verify_file_identity,
)

torch = pytest.importorskip("torch")


def test_weight_identity_is_verified(tmp_path: Path) -> None:
    weight = tmp_path / "teacher.pth"
    weight.write_bytes(b"reviewed")

    import hashlib

    expected = hashlib.sha256(b"reviewed").hexdigest()
    assert verify_file_identity(weight, expected, "teacher") == expected
    with pytest.raises(EfficientAdError, match="mismatch"):
        verify_file_identity(weight, "0" * 64, "teacher")


def test_checkpoint_identity_is_order_independent_and_sensitive() -> None:
    first = {"b": torch.tensor(2.0), "a": torch.tensor([1.0])}
    reordered = {"a": torch.tensor([1.0]), "b": torch.tensor(2.0)}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor(3.0)}

    assert canonical_checkpoint_sha256(first) == canonical_checkpoint_sha256(reordered)
    assert canonical_checkpoint_sha256(first) != canonical_checkpoint_sha256(changed)


def test_checkpoint_and_scores_reject_nonfinite_values() -> None:
    with pytest.raises(EfficientAdError, match="NaN or infinity"):
        canonical_checkpoint_sha256({"bad": torch.tensor([float("nan")])})
    with pytest.raises(EfficientAdError, match="finite"):
        validate_finite_scores([1.0, float("inf")], "validation")


def test_map_restoration_preserves_original_coordinates() -> None:
    restored = restore_efficientad_map(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), (5, 7))

    assert restored.shape == (5, 7)
    assert bool(torch.isfinite(restored).all())


def test_normal_only_calibration_uses_per_image_map_maxima() -> None:
    thresholds = calibrate_efficientad_thresholds(
        [float(index) for index in range(19)],
        [[float(index), -1.0] for index in range(19)],
        minimum_samples=19,
    )

    assert thresholds["image"]["threshold"] == 18.0
    assert thresholds["pixel"]["threshold"] == 18.0
    assert thresholds["pixel"]["sample_count"] == 19
