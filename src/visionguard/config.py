"""Typed loading and validation for dataset audit configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Layout = Literal["classified", "flat"]


class ConfigurationError(ValueError):
    """Raised when an audit configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class MaskConfig:
    """Relationship between anomalous samples and their segmentation masks."""

    image_condition: str
    directory: str
    suffix: str = "_mask"


@dataclass(frozen=True)
class SplitConfig:
    """Expected organization of one official dataset split."""

    layout: Layout
    conditions: tuple[str, ...] = ()
    mask: MaskConfig | None = None


@dataclass(frozen=True)
class ExpectedHashOverlap:
    """Documented dataset relationship where exact split overlap is intentional."""

    splits: frozenset[str]
    reason: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset structure contract used by the filesystem auditor."""

    name: str
    version: str
    categories: tuple[str, ...]
    image_extensions: tuple[str, ...]
    splits: dict[str, SplitConfig]
    required_root_files: tuple[str, ...] = ()
    expected_hash_overlaps: tuple[ExpectedHashOverlap, ...] = ()


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _strings(
    value: Any, location: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{location} must be a list of strings")
    result = tuple(value)
    if not allow_empty and not result:
        raise ConfigurationError(f"{location} must not be empty")
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{location} must not contain duplicates")
    return result


def _path_names(
    value: Any, location: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = _strings(value, location, allow_empty=allow_empty)
    if any(name in {".", ".."} or "/" in name or "\\" in name for name in result):
        raise ConfigurationError(
            f"{location} entries must be single relative path components"
        )
    return result


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load a dataset audit configuration from YAML and validate its schema."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to load {path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    dataset = _mapping(root.get("dataset"), "dataset")
    name = dataset.get("name")
    version = dataset.get("version")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("dataset.name must be a non-empty string")
    if not isinstance(version, str) or not version.strip():
        raise ConfigurationError("dataset.version must be a non-empty string")

    categories = _path_names(dataset.get("categories"), "dataset.categories")
    extensions = tuple(
        extension.lower()
        for extension in _strings(
            dataset.get("image_extensions"), "dataset.image_extensions"
        )
    )
    if any(not extension.startswith(".") for extension in extensions):
        raise ConfigurationError("dataset.image_extensions entries must start with '.'")

    split_values = _mapping(dataset.get("splits"), "dataset.splits")
    if not split_values:
        raise ConfigurationError("dataset.splits must not be empty")
    splits: dict[str, SplitConfig] = {}
    for split_name, split_value in split_values.items():
        if (
            not isinstance(split_name, str)
            or not split_name
            or split_name in {".", ".."}
            or "/" in split_name
            or "\\" in split_name
        ):
            raise ConfigurationError(
                "dataset.splits keys must be single relative path components"
            )
        split = _mapping(split_value, f"dataset.splits.{split_name}")
        layout = split.get("layout")
        if layout not in {"classified", "flat"}:
            raise ConfigurationError(
                f"dataset.splits.{split_name}.layout must be 'classified' or 'flat'"
            )
        conditions = _path_names(
            split.get("conditions", []),
            f"dataset.splits.{split_name}.conditions",
            allow_empty=layout == "flat",
        )
        if layout == "flat" and conditions:
            raise ConfigurationError(
                f"dataset.splits.{split_name}.conditions must be empty for flat layout"
            )

        mask_value = split.get("mask")
        mask = None
        if mask_value is not None:
            mask_data = _mapping(mask_value, f"dataset.splits.{split_name}.mask")
            required = ("image_condition", "directory")
            if any(not isinstance(mask_data.get(key), str) for key in required):
                raise ConfigurationError(
                    f"dataset.splits.{split_name}.mask requires string "
                    "image_condition and directory values"
                )
            suffix = mask_data.get("suffix", "_mask")
            if not isinstance(suffix, str):
                raise ConfigurationError(
                    f"dataset.splits.{split_name}.mask.suffix must be a string"
                )
            mask = MaskConfig(
                image_condition=mask_data["image_condition"],
                directory=mask_data["directory"],
                suffix=suffix,
            )
            _path_names(
                [mask.directory],
                f"dataset.splits.{split_name}.mask.directory",
            )
            if layout != "classified" or mask.image_condition not in conditions:
                raise ConfigurationError(
                    f"dataset.splits.{split_name}.mask image_condition must name a "
                    "configured classified condition"
                )

        splits[split_name] = SplitConfig(
            layout=layout, conditions=conditions, mask=mask
        )

    required_root_files = _path_names(
        dataset.get("required_root_files", []),
        "dataset.required_root_files",
        allow_empty=True,
    )
    overlap_values = dataset.get("expected_hash_overlaps", [])
    if not isinstance(overlap_values, list):
        raise ConfigurationError("dataset.expected_hash_overlaps must be a list")
    expected_hash_overlaps: list[ExpectedHashOverlap] = []
    seen_overlap_pairs: set[frozenset[str]] = set()
    for index, overlap_value in enumerate(overlap_values):
        location = f"dataset.expected_hash_overlaps[{index}]"
        overlap = _mapping(overlap_value, location)
        overlap_splits = _strings(overlap.get("splits"), f"{location}.splits")
        if len(overlap_splits) != 2:
            raise ConfigurationError(f"{location}.splits must contain exactly 2 splits")
        unknown_splits = set(overlap_splits) - set(splits)
        if unknown_splits:
            raise ConfigurationError(
                f"{location}.splits contains unknown splits: "
                f"{', '.join(sorted(unknown_splits))}"
            )
        reason = overlap.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigurationError(f"{location}.reason must be a non-empty string")
        split_pair = frozenset(overlap_splits)
        if split_pair in seen_overlap_pairs:
            raise ConfigurationError(
                f"{location}.splits duplicates an earlier expected overlap"
            )
        seen_overlap_pairs.add(split_pair)
        expected_hash_overlaps.append(
            ExpectedHashOverlap(splits=split_pair, reason=reason.strip())
        )

    return DatasetConfig(
        name=name,
        version=version,
        categories=categories,
        image_extensions=extensions,
        splits=splits,
        required_root_files=required_root_files,
        expected_hash_overlaps=tuple(expected_hash_overlaps),
    )
