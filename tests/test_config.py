from __future__ import annotations

from pathlib import Path

import pytest

from visionguard.config import ConfigurationError, load_dataset_config


def test_repository_mvtec_config_loads() -> None:
    config = load_dataset_config(Path("configs/datasets/mvtec_ad_2.yaml"))

    assert config.name == "mvtec_ad_2"
    assert len(config.categories) == 8
    assert config.splits["test_public"].mask is not None
    assert config.required_root_files == ("license.txt", "readme.txt")
    assert config.expected_hash_overlaps[0].splits == frozenset(
        {"test_private", "test_private_mixed"}
    )


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


def test_expected_overlap_requires_known_distinct_splits(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-overlap.yaml"
    config_path.write_text(
        """dataset:
  name: sample
  version: one
  categories: [widget]
  image_extensions: [.png]
  expected_hash_overlaps:
    - splits: [train, unknown]
      reason: paired views
  splits:
    train:
      layout: classified
      conditions: [good]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown splits"):
        load_dataset_config(config_path)


def test_expected_overlap_requires_documented_reason(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-reason.yaml"
    config_path.write_text(
        """dataset:
  name: sample
  version: one
  categories: [widget]
  image_extensions: [.png]
  expected_hash_overlaps:
    - splits: [train, validation]
  splits:
    train:
      layout: classified
      conditions: [good]
    validation:
      layout: classified
      conditions: [good]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reason must be"):
        load_dataset_config(config_path)


def test_root_file_path_traversal_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "path-traversal.yaml"
    config_path.write_text(
        """dataset:
  name: sample
  version: one
  categories: [widget]
  image_extensions: [.png]
  required_root_files: [../license.txt]
  splits:
    train:
      layout: classified
      conditions: [good]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="single relative path components"):
        load_dataset_config(config_path)
