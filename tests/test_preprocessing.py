from __future__ import annotations

import pytest

from visionguard.preprocessing import PreprocessingError, restore_anomaly_map

torch = pytest.importorskip("torch")


def test_anomaly_map_is_restored_to_original_image_coordinates() -> None:
    anomaly_map = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    restored = restore_anomaly_map(anomaly_map, (5, 7))

    assert restored.shape == (5, 7)
    assert bool(torch.isfinite(restored).all())


def test_anomaly_map_restoration_rejects_invalid_shapes_and_scores() -> None:
    with pytest.raises(PreprocessingError, match="one channel"):
        restore_anomaly_map(torch.zeros(2, 2, 2), (4, 4))
    with pytest.raises(PreprocessingError, match="finite"):
        restore_anomaly_map(torch.tensor([[float("nan")]]), (4, 4))
