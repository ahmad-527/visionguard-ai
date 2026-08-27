from __future__ import annotations

from pathlib import Path

import pytest

from visionguard.config import ConfigurationError, load_dataset_config


def test_repository_mvtec_config_loads() -> None:
    config = load_dataset_config(Path("configs/datasets/mvtec_ad_2.yaml"))

    assert config.name == "mvtec_ad_2"
    assert len(config.categories) == 8
    assert config.splits["test_public"].mask is not None


def test_invalid_extension_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """dataset:
  name: sample
  version: one
  categories: [widget]
  image_extensions: [png]
  splits:
    train:
      layout: classified
      conditions: [good]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="must start"):
        load_dataset_config(config_path)
