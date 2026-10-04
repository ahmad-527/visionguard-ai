"""Capability-scoped synthetic admission prototype; no real-test loader.

Only trees newly manufactured by this module can be admitted. A caller cannot
assert that an existing test directory is synthetic. Official split parsing is
reused with a fixture-specific SHA; actual official split admission is gated out.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from visionguard.visa import CATEGORIES, parse_split
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import SyntheticSample, require, validate_sample
from visionguard.visa_evaluator_synthetic import fixture
from visionguard.visa_protocol import canonical_fingerprint


@dataclass(frozen=True)
class FixturePermit:
    """In-process identity capability, not a human execution authorization."""

    nonce: str


@dataclass(frozen=True)
class AdmittedFrame:
    sample: SyntheticSample
    rgb: np.ndarray


_ISSUED: dict[str, tuple[FixturePermit, Path, dict]] = {}


def make_fixture(parent: Path) -> FixturePermit:
    """Create an isolated artificial tree; never adopt existing input files."""
    root = parent / f"synthetic-input-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["object", "split", "label", "image", "mask"])
    files = {}
    for category in CATEGORIES:
        training = f"{category}/Data/Images/Normal/synthetic-train.png"
        target = root / training
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((2, 2, 3), 201, dtype=np.uint8)).save(target)
        files[training] = sha256_file(target)
        writer.writerow([category, "train", "normal", training, ""])
        samples = fixture(category)[0]
        for i, sample in enumerate(samples):
            label = "anomaly" if sample.label else "normal"
            image = f"{category}/Data/Images/{label.title()}/synthetic-{i}.png"
            path = root / image
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb = np.full((2, 2, 3), i * 40, dtype=np.uint8)
            Image.fromarray(rgb).save(path)
            files[image] = sha256_file(path)
            mask = ""
            if sample.label:
                mask = f"{category}/Data/Masks/Anomaly/synthetic-{i}.png"
                target = root / mask
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(sample.mask * 255).save(target)
                files[mask] = sha256_file(target)
            writer.writerow([category, "test", label, image, mask])
    split = buffer.getvalue().encode()
    (root / "synthetic-split.csv").write_bytes(split)
    permit = FixturePermit(uuid4().hex)
    origin = {
        "evidence_class": "manufactured_synthetic_tree",
        "split_sha256": hashlib.sha256(split).hexdigest(),
        "files": files,
        "category_counts": dict.fromkeys(CATEGORIES, 4),
        "dataset_audit_identity": "synthetic-known-answer-audit-v1",
    }
    _ISSUED[permit.nonce] = (permit, root, origin)
    return permit


def _origin(permit: FixturePermit) -> tuple[Path, dict]:
    require(type(permit) is FixturePermit, "Issued synthetic capability required")
    record = _ISSUED.get(permit.nonce)
    require(record is not None and record[0] is permit, "Unissued input capability")
    return record[1], record[2]


def fixture_origin(permit: FixturePermit) -> dict:
    """Stable source/audit binding, without admitting a caller path."""
    _, origin = _origin(permit)
    return {k: v for k, v in origin.items() if k != "files"} | {
        "file_inventory_sha256": canonical_fingerprint(origin["files"])
    }


def admit_fixture(permit: FixturePermit) -> dict[str, list[AdmittedFrame]]:
    """Validate exact source, category/split/count, hashes, masks and dimensions.

    No arbitrary root argument. Input order is lexicographic POSIX image path.
    Every symlink/reparse component is rejected before asset content access.
    """
    root, origin = _origin(permit)

    def asset(relative: str) -> Path:
        selected = root / relative
        for path in (
            root,
            *list(selected.parents)[: len(Path(relative).parts) - 1],
            selected,
        ):
            state = path.lstat()
            require(
                not path.is_symlink()
                and not (
                    getattr(state, "st_file_attributes", 0)
                    & (0x400 | 0x1000 | 0x40000 | 0x400000)
                ),
                "Reparse input refused",
            )
        require(selected.resolve().is_relative_to(root.resolve()), "Escaping input")
        return selected

    split_path = asset("synthetic-split.csv")
    inventory = set()

    def visit(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                state = entry.stat(follow_symlinks=False)
                require(
                    not entry.is_symlink()
                    and not (
                        getattr(state, "st_file_attributes", 0)
                        & (0x400 | 0x1000 | 0x40000 | 0x400000)
                    ),
                    "Reparse/cloud input refused",
                )
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path))
                else:
                    inventory.add(Path(entry.path).relative_to(root).as_posix())

    visit(root)
    require(
        inventory == set(origin["files"]) | {"synthetic-split.csv"},
        "Unexpected input inventory",
    )
    records = parse_split(
        split_path.read_bytes(), expected_sha256=origin["split_sha256"]
    )
    paths = {r.image for r in records} | {r.mask for r in records if r.mask}
    require(paths == set(origin["files"]), "Missing/substituted source membership")
    result: dict[str, list[AdmittedFrame]] = {c: [] for c in CATEGORIES}
    image_hashes = set()
    training_hashes = set()
    for row in records:
        if row.split == "train":
            digest = sha256_file(asset(row.image))
            require(digest == origin["files"][row.image], "Training fixture corruption")
            training_hashes.add((row.category, digest))
    for row in sorted(records, key=lambda r: (r.category, r.image)):
        if row.split != "test":
            continue
        path = asset(row.image)
        require(sha256_file(path) == origin["files"][row.image], "Image corruption")
        # Fixture RGB bytes may repeat across categories; identities are scoped.
        duplicate_key = (row.category, origin["files"][row.image])
        require(duplicate_key not in training_hashes, "Synthetic train/test overlap")
        require(duplicate_key not in image_hashes, "Duplicate category image")
        image_hashes.add(duplicate_key)
        with Image.open(path) as image:
            require(image.mode == "RGB", "RGB input required")
            rgb = np.asarray(image).copy()
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        if row.mask:
            path = asset(row.mask)
            require(sha256_file(path) == origin["files"][row.mask], "Mask corruption")
            with Image.open(path) as image:
                array = np.asarray(image).copy()
            require(
                array.ndim == 2 and array.shape == rgb.shape[:2],
                "Mask dimensions differ",
            )
            require(
                array.dtype == np.uint8 and bool(np.isin(array, (0, 255)).all()),
                "Invalid original mask values",
            )
            mask = (array > 0).astype(np.uint8)
        index = Path(row.image).stem.removeprefix("synthetic-")
        sample = SyntheticSample(
            f"synthetic:{row.category}:{index}",
            row.category,
            int(row.label == "anomaly"),
            mask,
        )
        validate_sample(sample)
        result[row.category].append(AdmittedFrame(sample, rgb))
    require(
        {c: len(r) for c, r in result.items()} == origin["category_counts"],
        "Incomplete source counts",
    )
    return result
