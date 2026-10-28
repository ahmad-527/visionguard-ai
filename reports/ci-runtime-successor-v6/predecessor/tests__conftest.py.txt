from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionguard.config import DatasetConfig, MaskConfig, SplitConfig


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        name="synthetic",
        version="test-only",
        categories=("widget",),
        image_extensions=(".png",),
        splits={
            "train": SplitConfig(layout="classified", conditions=("good",)),
            "validation": SplitConfig(layout="classified", conditions=("good",)),
            "test_public": SplitConfig(
                layout="classified",
                conditions=("good", "bad"),
                mask=MaskConfig(
                    image_condition="bad", directory="ground_truth", suffix="_mask"
                ),
            ),
            "test_private": SplitConfig(layout="flat"),
            "test_private_mixed": SplitConfig(layout="flat"),
        },
    )


def write_image(path: Path, value: int, size: tuple[int, int] = (4, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    values = {
        "train/good/000.png": 10,
        "validation/good/000.png": 20,
        "test_public/good/000.png": 30,
        "test_public/bad/000.png": 40,
        "test_public/ground_truth/bad/000_mask.png": 255,
        "test_private/000_regular.png": 50,
        "test_private_mixed/000_mixed.png": 60,
    }
    for relative, value in values.items():
        write_image(tmp_path / "widget" / relative, value)
    return tmp_path
