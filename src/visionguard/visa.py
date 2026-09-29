"""Portable VisA records and deterministic development-only allocation."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, safe_member_path

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
SPLIT_SHA256 = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
ALLOCATION_SALT = "visionguard-visa-development-v1"


@dataclass(frozen=True)
class VisaSample:
    """Relative paths from official metadata; no machine-local roots."""

    category: str
    split: str
    label: str
    image: str
    mask: str | None

    @property
    def sample_id(self) -> str:
        return self.image


def parse_split(
    data: bytes,
    *,
    expected_sha256: str = SPLIT_SHA256,
    categories: tuple[str, ...] = CATEGORIES,
) -> tuple[VisaSample, ...]:
    """Fail on hash/schema/row/path errors; fixtures supply their own identity."""
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise VisaIntegrityError("Official split SHA-256 mismatch")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    if reader.fieldnames != ["object", "split", "label", "image", "mask"]:
        raise VisaIntegrityError("Official split schema mismatch")
    samples = []
    seen = set()
    masks = set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise VisaIntegrityError("Malformed split row")
        category, split, label = row["object"], row["split"], row["label"]
        if (
            category not in categories
            or split not in ("train", "test")
            or label not in ("normal", "anomaly")
            or (split == "train" and label != "normal")
        ):
            raise VisaIntegrityError("Split membership inconsistent")
        image = row["image"]
        image_path = safe_member_path(image)
        expected_prefix = (
            category,
            "Data",
            "Images",
            "Normal" if label == "normal" else "Anomaly",
        )
        if (
            image_path.parts[:4] != expected_prefix
            or len(image_path.parts) != 5
            or str(image_path) != image
        ):
            raise VisaIntegrityError("Unexpected image path structure")
        if image.casefold() in seen:
            raise VisaIntegrityError("Duplicate row or image identity")
        seen.add(image.casefold())
        mask = row["mask"] or None
        if label == "anomaly":
            if mask is None:
                raise VisaIntegrityError("Missing anomalous mask reference")
            mask_path = safe_member_path(mask)
            if (
                mask_path.parts[:4] != (category, "Data", "Masks", "Anomaly")
                or len(mask_path.parts) != 5
                or str(mask_path) != mask
                or mask_path.stem != image_path.stem
            ):
                raise VisaIntegrityError(
                    "Unexpected mask path or image/mask identity mismatch"
                )
            if mask.casefold() in masks:
                raise VisaIntegrityError("Duplicate mask reference")
            masks.add(mask.casefold())
        elif mask is not None:
            raise VisaIntegrityError("Normal image unexpectedly references a mask")
        samples.append(VisaSample(category, split, label, image, mask))
    if {s.category for s in samples} != set(categories):
        raise VisaIntegrityError("Incomplete category scope")
    for category in categories:
        if {(s.split, s.label) for s in samples if s.category == category} != {
            ("train", "normal"),
            ("test", "normal"),
            ("test", "anomaly"),
        }:
            raise VisaIntegrityError("Required official split classes absent")
    return tuple(sorted(samples, key=lambda s: s.sample_id))


def allocate_normals(
    samples: tuple[VisaSample, ...], *, minimum: int = 19
) -> dict[str, str]:
    """Frozen floor(n/10) calibration rule shared by both models and all seeds."""
    if type(minimum) is not int or minimum < 1:
        raise VisaIntegrityError("Invalid calibration minimum")
    roles = {}
    for category in sorted({s.category for s in samples}):
        normals = [s for s in samples if s.category == category and s.split == "train"]
        if any(s.label != "normal" for s in normals):
            raise VisaIntegrityError("Fitting population must be normal-only")
        count = len(normals) // 10
        if count < minimum:
            raise VisaIntegrityError(f"Inadequate calibration count in {category}")
        ordered = sorted(
            normals,
            key=lambda s: (
                hashlib.sha256(
                    f"{ALLOCATION_SALT}\0{category}\0{s.sample_id}".encode()
                ).hexdigest(),
                s.sample_id,
            ),
        )
        for index, sample in enumerate(ordered):
            if sample.sample_id in roles:
                raise VisaIntegrityError("Duplicate development identity")
            roles[sample.sample_id] = "calibration" if index < count else "fit"
    return dict(sorted(roles.items()))


def safe_asset(root: Path, relative: str) -> Path:
    """Reject symlinks/junction escapes instead of following external assets."""
    parts = safe_member_path(relative).parts
    root = root.resolve(strict=True)
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise VisaIntegrityError("Linked dataset assets are forbidden")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise VisaIntegrityError("Dataset asset escapes root or is not a regular file")
    return resolved


def sample_record(sample: VisaSample, inspected: dict, role: str) -> dict:
    """Stable adapter inventory with original dimensions and cryptographic identity."""
    return {**asdict(sample), "sample_id": sample.sample_id, "role": role, **inspected}
