"""Read-only dataset structure, image, annotation, and leakage auditing."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from visionguard.config import DatasetConfig, MaskConfig, SplitConfig

LOGGER = logging.getLogger(__name__)
Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class AuditIssue:
    """One actionable problem found during inspection."""

    severity: Severity
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ImageRecord:
    """Machine-readable metadata measured from an actual image file."""

    path: str
    category: str
    split: str
    condition: str | None
    kind: Literal["image", "mask"]
    sha256: str
    width: int | None
    height: int | None
    mode: str | None


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _issue(
    issues: list[AuditIssue],
    severity: Severity,
    code: str,
    message: str,
    path: Path | None,
    root: Path,
) -> None:
    issues.append(
        AuditIssue(
            severity=severity,
            code=code,
            message=message,
            path=_relative(path, root) if path is not None else None,
        )
    )


def _image_paths(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(
    path: Path,
    *,
    root: Path,
    category: str,
    split: str,
    condition: str | None,
    kind: Literal["image", "mask"],
    issues: list[AuditIssue],
) -> ImageRecord | None:
    try:
        digest = _sha256(path)
    except OSError as exc:
        _issue(
            issues,
            "error",
            "unreadable_file",
            f"File cannot be read: {exc}",
            path,
            root,
        )
        return None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _issue(
            issues,
            "error",
            "unreadable_image",
            f"Image cannot be decoded: {exc}",
            path,
            root,
        )
        return ImageRecord(
            path=_relative(path, root),
            category=category,
            split=split,
            condition=condition,
            kind=kind,
            sha256=digest,
            width=None,
            height=None,
            mode=None,
        )
    return ImageRecord(
        path=_relative(path, root),
        category=category,
        split=split,
        condition=condition,
        kind=kind,
        sha256=digest,
        width=width,
        height=height,
        mode=mode,
    )


def _check_directory_contents(
    directory: Path,
    allowed_directories: set[str],
    issues: list[AuditIssue],
    root: Path,
    *,
    allowed_files: bool = False,
) -> None:
    if not directory.is_dir():
        _issue(
            issues,
            "error",
            "missing_directory",
            "Required directory is missing",
            directory,
            root,
        )
        return
    unexpected = sorted(
        item
        for item in directory.iterdir()
        if item.is_dir() and item.name not in allowed_directories
    )
    for item in unexpected:
        _issue(
            issues,
            "warning",
            "unexpected_directory",
            "Directory is not part of the configured dataset structure",
            item,
            root,
        )
    if not allowed_files:
        for item in sorted(path for path in directory.iterdir() if path.is_file()):
            _issue(
                issues,
                "warning",
                "unexpected_file",
                "File is not expected at this level of the dataset structure",
                item,
                root,
            )


def _validate_masks(
    *,
    category_dir: Path,
    split_dir: Path,
    split_name: str,
    mask: MaskConfig,
    extensions: tuple[str, ...],
    root: Path,
    records: list[ImageRecord],
    issues: list[AuditIssue],
) -> None:
    image_dir = split_dir / mask.image_condition
    mask_dir = split_dir / mask.directory / mask.image_condition
    if not mask_dir.is_dir():
        _issue(
            issues,
            "error",
            "missing_mask_directory",
            "Required segmentation-mask directory is missing",
            mask_dir,
            root,
        )
        return

    image_paths = _image_paths(image_dir, extensions)
    mask_paths = _image_paths(mask_dir, extensions)
    masks_by_name = {path.name: path for path in mask_paths}
    expected_names: set[str] = set()
    for image_path in image_paths:
        expected_name = f"{image_path.stem}{mask.suffix}{image_path.suffix}"
        expected_names.add(expected_name)
        mask_path = masks_by_name.get(expected_name)
        if mask_path is None:
            _issue(
                issues,
                "error",
                "missing_mask",
                f"No mask found for anomalous image {image_path.name}",
                image_path,
                root,
            )
            continue
        record = _inspect_image(
            mask_path,
            root=root,
            category=category_dir.name,
            split=split_name,
            condition=mask.image_condition,
            kind="mask",
            issues=issues,
        )
        if record is not None:
            records.append(record)
            image_record = next(
                (
                    candidate
                    for candidate in records
                    if candidate.path == _relative(image_path, root)
                ),
                None,
            )
            if (
                image_record
                and None not in (record.width, record.height)
                and None not in (image_record.width, image_record.height)
                and (record.width, record.height)
                != (
                    image_record.width,
                    image_record.height,
                )
            ):
                _issue(
                    issues,
                    "error",
                    "mask_size_mismatch",
                    "Mask dimensions do not match the corresponding image",
                    mask_path,
                    root,
                )

    for orphan in sorted(set(masks_by_name) - expected_names):
        _issue(
            issues,
            "error",
            "orphan_mask",
            "Mask has no corresponding anomalous image",
            masks_by_name[orphan],
            root,
        )


def _audit_split(
    *,
    category_dir: Path,
    split_name: str,
    split: SplitConfig,
    config: DatasetConfig,
    root: Path,
    records: list[ImageRecord],
    issues: list[AuditIssue],
) -> None:
    split_dir = category_dir / split_name
    allowed = set(split.conditions)
    if split.mask:
        allowed.add(split.mask.directory)
    _check_directory_contents(
        split_dir,
        allowed,
        issues,
        root,
        allowed_files=split.layout == "flat",
    )
    if not split_dir.is_dir():
        return

    locations: Iterable[tuple[str | None, Path]]
    if split.layout == "flat":
        locations = ((None, split_dir),)
    else:
        locations = tuple(
            (condition, split_dir / condition) for condition in split.conditions
        )

    for condition, directory in locations:
        if not directory.is_dir():
            _issue(
                issues,
                "error",
                "missing_directory",
                "Required condition directory is missing",
                directory,
                root,
            )
            continue
        paths = _image_paths(directory, config.image_extensions)
        for unsupported in sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() not in config.image_extensions
        ):
            _issue(
                issues,
                "warning",
                "unsupported_file",
                "File extension is not configured as an image format",
                unsupported,
                root,
            )
        if not paths:
            _issue(
                issues,
                "warning",
                "empty_image_directory",
                "No configured image files were found",
                directory,
                root,
            )
        for path in paths:
            record = _inspect_image(
                path,
                root=root,
                category=category_dir.name,
                split=split_name,
                condition=condition,
                kind="image",
                issues=issues,
            )
            if record is not None:
                records.append(record)

    if split.mask:
        _validate_masks(
            category_dir=category_dir,
            split_dir=split_dir,
            split_name=split_name,
            mask=split.mask,
            extensions=config.image_extensions,
            root=root,
            records=records,
            issues=issues,
        )


def _duplicates_and_leakage(
    records: list[ImageRecord], issues: list[AuditIssue]
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.kind == "image":
            by_hash[record.sha256].append(record)

    groups: list[dict[str, Any]] = []
    for digest, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        paths = sorted(record.path for record in group)
        splits = sorted({record.split for record in group})
        categories = sorted({record.category for record in group})
        groups.append(
            {
                "sha256": digest,
                "paths": paths,
                "splits": splits,
                "categories": categories,
            }
        )
        if len(splits) > 1:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="split_hash_overlap",
                    message=(
                        f"Identical image content crosses splits: {', '.join(splits)}"
                    ),
                    path=paths[0],
                )
            )
        else:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="duplicate_image",
                    message=(
                        "Identical image content appears more than once in one split"
                    ),
                    path=paths[0],
                )
            )
    return groups


def audit_dataset(root: Path, config: DatasetConfig) -> dict[str, Any]:
    """Inspect ``root`` without modifying it and return a JSON-compatible report."""

    root = root.expanduser().resolve()
    issues: list[AuditIssue] = []
    records: list[ImageRecord] = []
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    expected_categories = set(config.categories)
    actual_categories = {path.name for path in root.iterdir() if path.is_dir()}
    for category in sorted(expected_categories - actual_categories):
        _issue(
            issues,
            "error",
            "missing_category",
            "Configured dataset category is missing",
            root / category,
            root,
        )
    for category in sorted(actual_categories - expected_categories):
        _issue(
            issues,
            "warning",
            "unexpected_category",
            "Directory is not a configured dataset category",
            root / category,
            root,
        )

    for category in config.categories:
        category_dir = root / category
        if not category_dir.is_dir():
            continue
        _check_directory_contents(category_dir, set(config.splits), issues, root)
        for split_name, split in config.splits.items():
            _audit_split(
                category_dir=category_dir,
                split_name=split_name,
                split=split,
                config=config,
                root=root,
                records=records,
                issues=issues,
            )

    duplicate_groups = _duplicates_and_leakage(records, issues)
    issues.sort(key=lambda item: (item.severity, item.code, item.path or ""))
    records.sort(key=lambda item: item.path)
    counts_by_split = Counter(
        record.split for record in records if record.kind == "image"
    )
    counts_by_category = Counter(
        record.category for record in records if record.kind == "image"
    )
    issue_counts = Counter(issue.severity for issue in issues)
    LOGGER.info(
        "Audited %d images and %d masks under %s",
        sum(record.kind == "image" for record in records),
        sum(record.kind == "mask" for record in records),
        root,
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {"name": config.name, "version": config.version},
        "root": str(root),
        "summary": {
            "status": "failed" if issue_counts["error"] else "passed",
            "image_count": sum(record.kind == "image" for record in records),
            "mask_count": sum(record.kind == "mask" for record in records),
            "error_count": issue_counts["error"],
            "warning_count": issue_counts["warning"],
            "counts_by_split": dict(sorted(counts_by_split.items())),
            "counts_by_category": dict(sorted(counts_by_category.items())),
            "duplicate_group_count": len(duplicate_groups),
        },
        "issues": [asdict(issue) for issue in issues],
        "duplicates": duplicate_groups,
        "files": [asdict(record) for record in records],
    }
