"""VisA integrity audit only: no models, test performance, or qualitative outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from visionguard.visa import (
    CATEGORIES,
    allocate_normals,
    parse_split,
    safe_asset,
    sample_record,
)
from visionguard.visa_acquire import (
    ARCHIVE_SHA256,
    SOURCE_COMMIT,
    VisaIntegrityError,
    atomic_json,
    sha256_file,
)


def inspect_asset(
    path: Path, *, mask: bool = False, allowed_values: set[int] | None = None
) -> dict:
    """Decode without rendering; hash encoded bytes and normalized decoded content."""
    before = path.stat()
    file_hash = sha256_file(path)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1 or min(image.size) < 1:
            raise VisaIntegrityError("Invalid image frame count or dimensions")
        image.load()
        width, height = image.size
        if mask:
            if image.mode not in ("L", "P"):
                raise VisaIntegrityError(
                    "Mask must be a single-channel integer label image"
                )
            pixels = image.tobytes()
            values = set(pixels)
            if (
                not values - {0}
                or allowed_values is None
                or not values <= allowed_values
            ):
                raise VisaIntegrityError("Mask has empty or invalid anomaly semantics")
            mode = b"LABEL8"
        else:
            pixels = image.convert("RGB").tobytes()
            mode = b"RGB8"
    digest = hashlib.sha256(
        mode + struct.pack(">II", width, height) + pixels
    ).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise VisaIntegrityError("Asset changed during audit")
    return {
        "sha256": file_hash,
        "decoded_sha256": digest,
        "width": width,
        "height": height,
        "size_bytes": after.st_size,
    }


def mask_semantics(source: Path) -> dict[str, set[int]]:
    """Parse the official constant as data; never execute downloaded code."""
    module = ast.parse(source.read_text())
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "id2class_map" for t in node.targets
        )
    ]
    if len(assignments) != 1:
        raise VisaIntegrityError("Official mask semantics unavailable")
    mapping = ast.literal_eval(assignments[0].value)
    if set(mapping) != set(CATEGORIES):
        raise VisaIntegrityError("Mask semantics category mismatch")
    return {
        category: {int(key) for key in values} for category, values in mapping.items()
    }


def audit_samples(
    root: Path, samples: tuple, semantics: dict[str, set[int]], *, minimum: int = 19
) -> tuple[dict, list[dict]]:
    """Full decode/inventory/overlap pass; public summary excludes sample IDs."""
    issues = []
    records = []
    try:
        roles = allocate_normals(samples, minimum=minimum)
    except VisaIntegrityError as exc:
        roles = {}
        issues.append({"code": "allocation_failure", "detail": str(exc)})
    expected_images = {s.image for s in samples}
    expected_masks = {s.mask for s in samples if s.mask}
    actual_images = set()
    actual_masks = set()
    # Inventory every asset under the declared category Data trees, not arbitrary
    # paths supplied by metadata. Metadata files outside these trees are allowed.
    for category in sorted({s.category for s in samples}):
        for kind, destination in (("Images", actual_images), ("Masks", actual_masks)):
            directory = root / category / "Data" / kind
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_symlink():
                        issues.append({"code": "linked_asset"})
                    elif path.is_file():
                        destination.add(path.relative_to(root).as_posix())
    for name, actual, expected in (
        ("images", actual_images, expected_images),
        ("masks", actual_masks, expected_masks),
    ):
        if actual != expected:
            issues.append(
                {
                    "code": f"{name}_inventory_mismatch",
                    "missing": len(expected - actual),
                    "orphan": len(actual - expected),
                }
            )
    groups = {
        key: defaultdict(list)
        for key in ("file", "decoded", "mask_file", "mask_decoded")
    }
    counts = defaultdict(Counter)
    for sample in samples:
        counts[sample.category][f"{sample.split}_{sample.label}"] += 1
        try:
            image = inspect_asset(safe_asset(root, sample.image))
            mask = None
            if sample.mask:
                mask = inspect_asset(
                    safe_asset(root, sample.mask),
                    mask=True,
                    allowed_values=semantics[sample.category],
                )
                if (image["width"], image["height"]) != (mask["width"], mask["height"]):
                    raise VisaIntegrityError("Image/mask shape mismatch")
            role = roles.get(
                sample.sample_id,
                "sealed_test" if sample.split == "test" else "unallocated",
            )
            record = sample_record(
                sample, {"image_identity": image, "mask_identity": mask}, role
            )
            records.append(record)
            groups["file"][image["sha256"]].append(record)
            groups["decoded"][image["decoded_sha256"]].append(record)
            if mask:
                groups["mask_file"][mask["sha256"]].append(record)
                groups["mask_decoded"][mask["decoded_sha256"]].append(record)
        except (OSError, ValueError, KeyError) as exc:
            # IDs remain in ignored diagnostic inventory only; no raw local paths.
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "integrity_error": str(exc)
                    if isinstance(exc, VisaIntegrityError)
                    else "asset_unreadable_or_missing",
                }
            )
            issues.append({"code": "asset_integrity_error"})
    duplicate_summary = {}
    for identity, grouped in groups.items():
        duplicates = [group for group in grouped.values() if len(group) > 1]
        if identity.startswith("mask"):
            duplicate_summary[identity] = {
                "duplicate_groups": len(duplicates),
                "affected_samples": sum(map(len, duplicates)),
            }
            continue
        overlap = [
            g for g in duplicates if {r["split"] for r in g} == {"train", "test"}
        ]
        fit_cal = [
            g for g in duplicates if {"fit", "calibration"} <= {r["role"] for r in g}
        ]
        duplicate_summary[identity] = {
            "within_train_groups": sum(
                sum(r["split"] == "train" for r in g) > 1 for g in duplicates
            ),
            "within_test_groups": sum(
                sum(r["split"] == "test" for r in g) > 1 for g in duplicates
            ),
            "train_test_overlap_groups": len(overlap),
            "fit_calibration_overlap_groups": len(fit_cal),
            "train_test_overlap_group_sha256": sorted(
                hashlib.sha256(
                    "\n".join(sorted(r["sample_id"] for r in g)).encode()
                ).hexdigest()
                for g in overlap
            ),
        }
        if overlap:
            issues.append(
                {
                    "code": "train_test_exact_overlap",
                    "identity": identity,
                    "groups": len(overlap),
                }
            )
        if fit_cal:
            issues.append(
                {
                    "code": "fit_calibration_exact_overlap",
                    "identity": identity,
                    "groups": len(fit_cal),
                }
            )
    allocation_counts = {
        category: dict(
            Counter(
                roles.get(s.sample_id, "unallocated")
                for s in samples
                if s.category == category and s.split == "train"
            )
        )
        for category in sorted(counts)
    }
    return {
        "schema_version": 1,
        "status": "failed" if issues else "passed",
        "issues": issues,
        "sample_count": len(samples),
        "category_count": len(counts),
        "category_counts": dict(counts),
        "allocation_counts": allocation_counts,
        "duplicates": duplicate_summary,
        "decoded_image_count": sum("image_identity" in r for r in records),
        "decoded_mask_count": sum(bool(r.get("mask_identity")) for r in records),
        "test_performance_evaluated": False,
        "final_test_lock": "closed",
    }, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
        help="Ignored private integrity inventory",
    )
    args = parser.parse_args(argv)
    acquisition = json.loads(args.acquisition.read_text())
    if (
        acquisition["archive"]["observed_sha256"] != ARCHIVE_SHA256
        or acquisition["source"]["commit"] != SOURCE_COMMIT
        or acquisition["extraction"]["status"] != "completed"
    ):
        raise VisaIntegrityError("Acquisition identity is not verified")
    source = args.local_root / "source"
    for relative, identity in acquisition["source"]["files"].items():
        if sha256_file(safe_asset(source, relative)) != identity["sha256"]:
            raise VisaIntegrityError("Official source metadata changed")
    samples = parse_split((source / "split_csv/1cls.csv").read_bytes())
    summary, inventory = audit_samples(
        args.local_root / "sealed",
        samples,
        mask_semantics(source / "utils/id2class.py"),
    )
    atomic_json(args.inventory, {"schema_version": 1, "records": inventory})
    summary["inventory_sha256"] = sha256_file(args.inventory)
    summary["acquisition_sha256"] = sha256_file(args.acquisition)
    summary["official_split_sha256"] = acquisition["source"]["files"][
        "split_csv/1cls.csv"
    ]["sha256"]
    summary["archive_sha256"] = ARCHIVE_SHA256
    atomic_json(args.output, summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "sample_count",
                    "category_count",
                    "issues",
                    "decoded_image_count",
                    "decoded_mask_count",
                    "final_test_lock",
                )
            }
        )
    )
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
