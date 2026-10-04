"""Actual official admission, invoked only AFTER independently issued permission.

Tests mint new artificial trees. No existing root can become an artificial tree.
No metadata from real test membership is loaded during engineering acceptance.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from visionguard.heldout_authorization import Permission, check_permission
from visionguard.heldout_contract import context
from visionguard.heldout_metrics import Sample, validate_sample
from visionguard.heldout_paths import asset, inventory
from visionguard.visa import CATEGORIES, parse_split
from visionguard.visa_acquire import SOURCE_COMMIT, sha256_file
from visionguard.visa_audit import inspect_asset, mask_semantics
from visionguard.visa_evaluator import require
from visionguard.visa_evaluator_storage import canonical_bytes
from visionguard.visa_protocol import canonical_fingerprint


@dataclass(frozen=True)
class Frame:
    sample: Sample
    rgb: np.ndarray


@dataclass(frozen=True)
class Inputs:
    root: Path
    rows: dict
    identities: dict
    semantics: dict
    origin: dict
    artificial: bool

    def frames(self, category: str):
        for row in self.rows[category]:
            recorded = self.identities[row.image]
            path = asset(self.root / "sealed", row.image)
            require(
                inspect_asset(path) == recorded["image_identity"],
                "Image changed after admission",
            )
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            if row.mask:
                path = asset(self.root / "sealed", row.mask)
                require(
                    inspect_asset(
                        path, mask=True, allowed_values=self.semantics[category]
                    )
                    == recorded["mask_identity"],
                    "Mask changed after admission",
                )
                with Image.open(path) as image:
                    mask = (np.asarray(image) > 0).astype(np.uint8)
            sample = Sample(row.image, category, int(row.label == "anomaly"), mask)
            validate_sample(sample)
            yield Frame(sample, rgb)


def _admit(root: Path, audit: dict, artificial: bool) -> Inputs:
    """Internal shared verifier; caller cannot set independent production anchors."""

    def bound(relative, digest):
        path = asset(root, relative)
        require(sha256_file(path) == digest, "Pinned metadata corruption")
        return path

    acquisition = json.loads(
        bound("acquisition.json", audit["acquisition_sha256"]).read_text()
    )
    require(
        acquisition["source"]["commit"] == SOURCE_COMMIT
        and acquisition["archive"]["observed_sha256"] == audit["archive_sha256"]
        and acquisition["extraction"]["status"] == "completed",
        "Official origin mismatch",
    )
    for relative, identity in acquisition["source"]["files"].items():
        bound("source/" + relative, identity["sha256"])
    split = bound("source/split_csv/1cls.csv", audit["official_split_sha256"])
    rows = parse_split(
        split.read_bytes(), expected_sha256=audit["official_split_sha256"]
    )
    records = json.loads(bound("inventory.json", audit["inventory_sha256"]).read_text())
    require(records["schema_version"] == 1, "Inventory schema mismatch")
    identities = {r["image"]: r for r in records["records"]}
    require(
        len(identities)
        == len(records["records"])
        == len(rows)
        == audit["sample_count"],
        "Exact membership count mismatch",
    )
    counts = {
        c: dict(Counter(f"{r.split}_{r.label}" for r in rows if r.category == c))
        for c in CATEGORIES
    }
    require(counts == audit["category_counts"], "Category/class/split count mismatch")
    expected = {r.image for r in rows} | {r.mask for r in rows if r.mask}
    require(
        inventory(root / "sealed") == expected, "Missing/unexpected sealed membership"
    )
    semantics = mask_semantics(asset(root, "source/utils/id2class.py"))
    train = {
        kind: {
            identities[r.image]["image_identity"][kind]
            for r in rows
            if r.split == "train"
        }
        for kind in ("sha256", "decoded_sha256")
    }
    seen = {kind: set() for kind in train}
    for row in rows:
        recorded = identities[row.image]
        require(
            recorded["sample_id"] == row.image
            and recorded["category"] == row.category
            and recorded["split"] == row.split
            and recorded["label"] == row.label
            and recorded["mask"] == row.mask,
            "Inventory relationship mismatch",
        )
        if row.split != "test":
            continue  # Training identities: independently SHA-bound inventory.
        observed = inspect_asset(asset(root / "sealed", row.image))
        require(
            observed == recorded["image_identity"], "Image identity/dimensions mismatch"
        )
        for kind in seen:
            require(
                observed[kind] not in seen[kind] and observed[kind] not in train[kind],
                "Duplicate/overlap rejected",
            )
            seen[kind].add(observed[kind])
        if row.mask:
            observed_mask = inspect_asset(
                asset(root / "sealed", row.mask),
                mask=True,
                allowed_values=semantics[row.category],
            )
            require(
                observed_mask == recorded["mask_identity"]
                and (observed_mask["width"], observed_mask["height"])
                == (observed["width"], observed["height"]),
                "Mask identity/dimensions mismatch",
            )
        else:
            require(
                recorded["mask_identity"] is None, "Normal mask relationship mismatch"
            )
    selected = {
        c: tuple(r for r in rows if r.category == c and r.split == "test")
        for c in CATEGORIES
    }
    origin = {
        "evidence_class": "artificial_real_id_acceptance"
        if artificial
        else "heldout_historical_independence_unverified",
        "audit_identity": canonical_fingerprint(audit),
        "split_sha256": audit["official_split_sha256"],
        "inventory_sha256": audit["inventory_sha256"],
        "membership": {c: [r.image for r in rr] for c, rr in selected.items()},
    }
    return Inputs(root, selected, identities, semantics, origin, artificial)


def admit(repository: Path, permission: Permission) -> Inputs:
    check_permission(permission)  # BEFORE Path/source-root operations, even stat.
    return _admit(permission.source_root, context(repository)["audit"], False)


@dataclass(frozen=True)
class ArtificialPermit:
    nonce: str


_ARTIFICIAL = {}


def make_artificial(parent: Path) -> ArtificialPermit:
    """Four known-answer images/category + training row, all freshly generated."""
    from visionguard.visa_evaluator_synthetic import fixture

    root = parent / f"artificial-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    records = []
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["object", "split", "label", "image", "mask"])
    semantics = {c: {0: "normal", 2: "artificial_defect"} for c in CATEGORIES}
    for ci, c in enumerate(CATEGORIES):
        for i in range(5):
            label = 0 if i == 4 else fixture(c)[0][i].label
            split = "train" if i == 4 else "test"
            relative = (
                f"{c}/Data/Images/{'Anomaly' if label else 'Normal'}/artificial-{i}.png"
            )
            target = root / "sealed" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Globally distinct encoded/decoded RGB, including training.
            rgb = np.full((2, 2, 3), i * 40, dtype=np.uint8)
            rgb[0, 0, 0] = ci
            Image.fromarray(rgb).save(target)
            mask_relative = None
            mask_identity = None
            if label:
                mask_relative = f"{c}/Data/Masks/Anomaly/artificial-{i}.png"
                target_mask = root / "sealed" / mask_relative
                target_mask.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(fixture(c)[0][i].mask * 2).save(target_mask)
                mask_identity = inspect_asset(
                    target_mask, mask=True, allowed_values={0, 2}
                )
            writer.writerow(
                [
                    c,
                    split,
                    "anomaly" if label else "normal",
                    relative,
                    mask_relative or "",
                ]
            )
            records.append(
                {
                    "category": c,
                    "split": split,
                    "label": "anomaly" if label else "normal",
                    "image": relative,
                    "mask": mask_relative,
                    "sample_id": relative,
                    "role": "sealed_test" if split == "test" else "fit",
                    "image_identity": inspect_asset(target),
                    "mask_identity": mask_identity,
                }
            )
    source = root / "source"
    (source / "split_csv").mkdir(parents=True)
    (source / "utils").mkdir()
    (source / "split_csv/1cls.csv").write_bytes(buffer.getvalue().encode())
    (source / "utils/id2class.py").write_text("id2class_map = " + repr(semantics))
    (root / "inventory.json").write_bytes(
        canonical_bytes({"schema_version": 1, "records": records})
    )
    acquisition = {
        "source": {
            "commit": SOURCE_COMMIT,
            "files": {
                p.relative_to(source).as_posix(): {"sha256": sha256_file(p)}
                for p in source.rglob("*")
                if p.is_file()
            },
        },
        "archive": {"observed_sha256": "0" * 64},
        "extraction": {"status": "completed"},
    }
    (root / "acquisition.json").write_bytes(canonical_bytes(acquisition))
    audit = {
        "archive_sha256": "0" * 64,
        "acquisition_sha256": sha256_file(root / "acquisition.json"),
        "inventory_sha256": sha256_file(root / "inventory.json"),
        "official_split_sha256": sha256_file(source / "split_csv/1cls.csv"),
        "sample_count": 60,
        "category_counts": {
            c: {"train_normal": 1, "test_normal": 2, "test_anomaly": 2}
            for c in CATEGORIES
        },
    }
    permit = ArtificialPermit(uuid4().hex)
    _ARTIFICIAL[permit.nonce] = (permit, root, audit)
    return permit


def admit_artificial(permit: ArtificialPermit) -> Inputs:
    require(
        type(permit) is ArtificialPermit
        and permit.nonce in _ARTIFICIAL
        and _ARTIFICIAL[permit.nonce][0] is permit,
        "Unissued artificial capability",
    )
    _, root, audit = _ARTIFICIAL[permit.nonce]
    return _admit(root, audit, True)
