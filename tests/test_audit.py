from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from conftest import write_image
from visionguard.audit import audit_dataset
from visionguard.config import DatasetConfig, ExpectedHashOverlap


def issue_codes(report: dict[str, object]) -> set[str]:
    issues = report["issues"]
    assert isinstance(issues, list)
    return {str(issue["code"]) for issue in issues}


def test_valid_dataset_passes_with_measured_counts(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    report = audit_dataset(valid_dataset, dataset_config)

    assert report["summary"] == {
        "status": "passed",
        "image_count": 6,
        "mask_count": 1,
        "error_count": 0,
        "warning_count": 0,
        "counts_by_split": {
            "test_private": 1,
            "test_private_mixed": 1,
            "test_public": 2,
            "train": 1,
            "validation": 1,
        },
        "counts_by_category": {"widget": 6},
        "duplicate_group_count": 0,
        "expected_overlap_group_count": 0,
        "unexpected_overlap_group_count": 0,
        "within_split_duplicate_group_count": 0,
    }
    assert len(report["files"]) == 7


def test_unreadable_image_is_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    (valid_dataset / "widget/train/good/000.png").write_bytes(b"not an image")

    report = audit_dataset(valid_dataset, dataset_config)

    assert "unreadable_image" in issue_codes(report)
    assert report["summary"]["status"] == "failed"


def test_missing_and_orphan_masks_are_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    mask = valid_dataset / "widget/test_public/ground_truth/bad/000_mask.png"
    mask.unlink()
    write_image(mask.with_name("orphan_mask.png"), 255)

    report = audit_dataset(valid_dataset, dataset_config)

    assert {"missing_mask", "orphan_mask"} <= issue_codes(report)


def test_mask_dimension_mismatch_is_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    mask = valid_dataset / "widget/test_public/ground_truth/bad/000_mask.png"
    Image.new("L", (2, 2), color=255).save(mask)

    report = audit_dataset(valid_dataset, dataset_config)

    assert "mask_size_mismatch" in issue_codes(report)


def test_duplicate_within_split_is_a_warning(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    source = valid_dataset / "widget/train/good/000.png"
    shutil.copyfile(source, source.with_name("001.png"))

    report = audit_dataset(valid_dataset, dataset_config)

    assert "duplicate_image" in issue_codes(report)
    assert report["summary"]["status"] == "passed"


def test_hash_overlap_across_splits_is_an_error(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    shutil.copyfile(
        valid_dataset / "widget/train/good/000.png",
        valid_dataset / "widget/validation/good/000.png",
    )

    report = audit_dataset(valid_dataset, dataset_config)

    assert "split_hash_overlap" in issue_codes(report)
    assert report["summary"]["status"] == "failed"


def test_configured_cross_split_overlap_is_classified_without_error(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    shutil.copyfile(
        valid_dataset / "widget/test_private/000_regular.png",
        valid_dataset / "widget/test_private_mixed/000_mixed.png",
    )
    configured = replace(
        dataset_config,
        expected_hash_overlaps=(
            ExpectedHashOverlap(
                splits=frozenset({"test_private", "test_private_mixed"}),
                reason="Synthetic paired evaluation views",
            ),
        ),
    )

    report = audit_dataset(valid_dataset, configured)

    assert report["summary"]["status"] == "passed"
    assert report["summary"]["expected_overlap_group_count"] == 1
    assert report["summary"]["unexpected_overlap_group_count"] == 0
    assert report["duplicates"][0]["classification"] == ("expected_cross_split_overlap")
    assert report["duplicates"][0]["policy_reason"] == (
        "Synthetic paired evaluation views"
    )


def test_configured_overlap_does_not_hide_extra_duplicate(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    source = valid_dataset / "widget/test_private/000_regular.png"
    shutil.copyfile(source, source.with_name("001_regular.png"))
    shutil.copyfile(source, valid_dataset / "widget/test_private_mixed/000_mixed.png")
    configured = replace(
        dataset_config,
        expected_hash_overlaps=(
            ExpectedHashOverlap(
                splits=frozenset({"test_private", "test_private_mixed"}),
                reason="Synthetic paired evaluation views",
            ),
        ),
    )

    report = audit_dataset(valid_dataset, configured)

    assert report["summary"]["status"] == "failed"
    assert report["summary"]["unexpected_overlap_group_count"] == 1
    assert "split_hash_overlap" in issue_codes(report)


def test_missing_structure_and_unexpected_category_are_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    (valid_dataset / "widget/validation/good/000.png").unlink()
    (valid_dataset / "widget/validation/good").rmdir()
    (valid_dataset / "extra").mkdir()

    report = audit_dataset(valid_dataset, dataset_config)

    assert {"missing_directory", "unexpected_category"} <= issue_codes(report)


def test_unexpected_file_at_structure_level_is_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    (valid_dataset / "widget/notes.txt").write_text("local note", encoding="utf-8")

    report = audit_dataset(valid_dataset, dataset_config)

    assert "unexpected_file" in issue_codes(report)


def test_missing_required_root_file_is_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    configured = replace(dataset_config, required_root_files=("license.txt",))

    report = audit_dataset(valid_dataset, configured)

    assert "missing_root_file" in issue_codes(report)


def test_image_symlink_is_rejected_without_following_it(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    target = valid_dataset / "outside.png"
    write_image(target, 70)
    link = valid_dataset / "widget/train/good/linked.png"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")

    report = audit_dataset(valid_dataset, dataset_config)

    assert "symlink_not_allowed" in issue_codes(report)
    assert not any(file["path"].endswith("linked.png") for file in report["files"])


def test_unexpected_root_file_is_reported(
    valid_dataset: Path, dataset_config: DatasetConfig
) -> None:
    (valid_dataset / "archive-note.txt").write_text("note", encoding="utf-8")

    report = audit_dataset(valid_dataset, dataset_config)

    assert "unexpected_root_file" in issue_codes(report)
