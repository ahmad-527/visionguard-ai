"""Synthetic dataset integrity tests; no real test outcomes or models."""

import csv
import hashlib
import io
import tarfile

import pytest
from PIL import Image

from visionguard.visa import VisaSample, allocate_normals, parse_split, safe_asset
from visionguard.visa_acquire import (
    VisaIntegrityError,
    extract_verified_archive,
    safe_member_path,
)
from visionguard.visa_audit import audit_samples, inspect_asset


def split_bytes(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["object", "split", "label", "image", "mask"])
    writer.writerows(rows)
    return out.getvalue().encode()


def synthetic_rows():
    rows = [
        ["candle", "train", "normal", f"candle/Data/Images/Normal/{i:03}.png", ""]
        for i in range(20)
    ]
    rows += [
        ["candle", "test", "normal", "candle/Data/Images/Normal/test.png", ""],
        [
            "candle",
            "test",
            "anomaly",
            "candle/Data/Images/Anomaly/test.png",
            "candle/Data/Masks/Anomaly/test.png",
        ],
    ]
    return rows


def parse_synthetic(rows):
    raw = split_bytes(rows)
    return parse_split(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(), categories=("candle",)
    )


@pytest.fixture
def dataset(tmp_path):
    samples = parse_synthetic(synthetic_rows())
    for i, sample in enumerate(samples):
        path = tmp_path / sample.image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), (i, i * 2, 80)).save(path)
        if sample.mask:
            mask = tmp_path / sample.mask
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (4, 3), 1).save(mask)
    return tmp_path, samples


def test_official_schema_and_stable_lexical_order():
    samples = parse_synthetic(synthetic_rows())
    assert len(samples) == 22
    assert [s.sample_id for s in samples] == sorted(s.sample_id for s in samples)
    assert all(s.label == "normal" for s in samples if s.split == "train")


def test_split_hash_mismatch():
    with pytest.raises(VisaIntegrityError, match="SHA-256"):
        parse_split(split_bytes(synthetic_rows()))


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "train_anomaly",
        "missing_mask",
        "normal_mask",
        "wrong_category",
        "wrong_mask_id",
        "extra_field",
    ],
)
def test_bad_split_rows(change):
    rows = synthetic_rows()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "train_anomaly":
        rows[0][2] = "anomaly"
    elif change == "missing_mask":
        rows[-1][4] = ""
    elif change == "normal_mask":
        rows[0][4] = rows[-1][4]
    elif change == "wrong_category":
        rows[0][0] = "unknown"
    elif change == "wrong_mask_id":
        rows[-1][4] = "candle/Data/Masks/Anomaly/other.png"
    else:
        rows[0].append("extra")
    with pytest.raises(VisaIntegrityError):
        parse_synthetic(rows)


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/root/x",
        "C:/data/x",
        "x\\y",
        "x/../y",
        "x//y",
        "x/NUL.png",
        "x/file. ",
    ],
)
def test_malformed_paths(path):
    with pytest.raises(VisaIntegrityError):
        safe_member_path(path)
    rows = synthetic_rows()
    rows[0][3] = path
    with pytest.raises(VisaIntegrityError):
        parse_synthetic(rows)


def test_allocation_reproducible_disjoint_and_no_test():
    samples = parse_synthetic(synthetic_rows())
    roles = allocate_normals(samples, minimum=1)
    assert roles == allocate_normals(tuple(reversed(samples)), minimum=1)
    assert len(roles) == 20
    assert list(roles.values()).count("calibration") == 2
    assert all(s.sample_id not in roles for s in samples if s.split == "test")
    ranks = sorted(
        (
            hashlib.sha256(
                f"visionguard-visa-development-v1\0candle\0{s.sample_id}".encode()
            ).hexdigest(),
            s.sample_id,
        )
        for s in samples
        if s.split == "train"
    )
    assert {s for s, r in roles.items() if r == "calibration"} == {
        s for _, s in ranks[:2]
    }
    with pytest.raises(VisaIntegrityError, match="Inadequate"):
        allocate_normals(samples)


def test_valid_inventory_audit(dataset):
    root, samples = dataset
    summary, inventory = audit_samples(root, samples, {"candle": {0, 1}}, minimum=1)
    assert summary["status"] == "passed"
    assert summary["decoded_image_count"] == 22
    assert summary["decoded_mask_count"] == 1
    assert summary["duplicates"]["decoded"]["train_test_overlap_groups"] == 0
    assert all("image_identity" in row for row in inventory)
    assert summary["test_performance_evaluated"] is False
    assert summary["final_test_lock"] == "closed"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_image",
        "decode",
        "missing_mask",
        "shape",
        "semantics",
        "empty_mask",
        "orphan",
    ],
)
def test_asset_failure_blocks_audit(dataset, failure):
    root, samples = dataset
    anomaly = next(s for s in samples if s.mask)
    path = root / samples[0].image
    mask = root / anomaly.mask
    if failure == "missing_image":
        path.unlink()
    elif failure == "decode":
        path.write_bytes(b"not_an_image")
    elif failure == "missing_mask":
        mask.unlink()
    elif failure == "shape":
        Image.new("L", (2, 2), 1).save(mask)
    elif failure == "semantics":
        Image.new("L", (4, 3), 255).save(mask)
    elif failure == "empty_mask":
        Image.new("L", (4, 3), 0).save(mask)
    else:
        Image.new("L", (4, 3), 1).save(mask.with_name("orphan.png"))
    summary, _ = audit_samples(root, samples, {"candle": {0, 1}}, minimum=1)
    assert summary["status"] == "failed"


@pytest.mark.parametrize("reencode", [False, True])
def test_exact_train_test_overlap_blocks(dataset, reencode):
    root, samples = dataset
    train = next(s for s in samples if s.split == "train")
    test = next(s for s in samples if s.split == "test" and s.label == "normal")
    source, target = root / train.image, root / test.image
    if reencode:
        with Image.open(source) as image:
            image.save(target, compress_level=0)
    else:
        target.write_bytes(source.read_bytes())
    summary, _ = audit_samples(root, samples, {"candle": {0, 1}}, minimum=1)
    assert summary["status"] == "failed"
    assert summary["duplicates"]["decoded"]["train_test_overlap_groups"] == 1
    assert summary["duplicates"]["file"]["train_test_overlap_groups"] == (
        0 if reencode else 1
    )


def test_fit_calibration_content_overlap_blocks(dataset):
    root, samples = dataset
    roles = allocate_normals(samples, minimum=1)
    fit = next(s for s, role in roles.items() if role == "fit")
    calibration = next(s for s, role in roles.items() if role == "calibration")
    (root / calibration).write_bytes((root / fit).read_bytes())
    summary, _ = audit_samples(root, samples, {"candle": {0, 1}}, minimum=1)
    assert summary["duplicates"]["decoded"]["fit_calibration_overlap_groups"] == 1
    assert summary["status"] == "failed"


def test_missing_asset_and_non_single_channel_mask(tmp_path):
    with pytest.raises(FileNotFoundError):
        safe_asset(tmp_path, "absent.png")
    mask = tmp_path / "mask.png"
    Image.new("RGB", (2, 2), (1, 1, 1)).save(mask)
    with pytest.raises(VisaIntegrityError):
        inspect_asset(mask, mask=True, allowed_values={0, 1})


def test_archive_special_member_rejected_before_extraction(tmp_path):
    archive = tmp_path / "synthetic.tar"
    with tarfile.open(archive, "w") as tar:
        item = tarfile.TarInfo("link")
        item.type = tarfile.SYMTYPE
        item.linkname = "../outside"
        tar.addfile(item)
    target = tmp_path / "sealed"
    with pytest.raises(VisaIntegrityError, match="special"):
        extract_verified_archive(archive, target)
    assert not target.exists()


def test_existing_partial_extraction_not_silently_reused(tmp_path):
    target = tmp_path / "sealed"
    target.mkdir()
    with pytest.raises(VisaIntegrityError, match="completion receipt"):
        extract_verified_archive(tmp_path / "missing.tar", target)


def test_bad_normal_population_rejected():
    samples = tuple(
        VisaSample("candle", "train", "anomaly", f"x{i}", None) for i in range(20)
    )
    with pytest.raises(VisaIntegrityError, match="normal-only"):
        allocate_normals(samples, minimum=1)
