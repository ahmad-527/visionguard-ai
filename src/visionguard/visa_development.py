"""Development manifest adapter: training never reads official test metadata."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from visionguard.visa import CATEGORIES, safe_asset
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


def materialize_development(
    sealed: Path,
    destination: Path,
    *,
    audit_path: Path,
    inventory_path: Path,
    membership_path: Path,
) -> str:
    """Create normal-only copies once, after a passed full audit.

    The ignored complete audit inventory remains separate; the portable membership
    output contains only permitted training normals, their allocation and hashes.
    """
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "passed"
        or sha256_file(inventory_path) != audit["inventory_sha256"]
    ):
        raise VisaIntegrityError("Passed audit and exact inventory required")
    records = json.loads(inventory_path.read_text())["records"]
    development = []
    for record in records:
        if record.get("split") != "train":
            continue
        if record["label"] != "normal" or record["role"] not in ("fit", "calibration"):
            raise VisaIntegrityError("Invalid development record")
        relative = f"{record['category']}/{record['role']}/{Path(record['image']).name}"
        development.append(
            {
                "sample_id": record["sample_id"],
                "category": record["category"],
                "role": record["role"],
                "path": relative,
                "image_identity": record["image_identity"],
            }
        )
    development.sort(key=lambda r: r["sample_id"])
    manifest = {
        "schema_version": 1,
        "dataset": "VisA",
        "audit_sha256": sha256_file(audit_path),
        "allocation_id": "visa-normal-sha256-90-10-v1",
        "records": development,
    }
    if destination.exists():
        raise VisaIntegrityError(
            "Existing development root: verify or inspect, never overwrite"
        )
    destination.mkdir(parents=True)
    for record in development:
        source = safe_asset(sealed, record["sample_id"])
        if sha256_file(source) != record["image_identity"]["sha256"]:
            raise VisaIntegrityError("Development source changed since audit")
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != record["image_identity"]["sha256"]:
            raise VisaIntegrityError("Development copy identity mismatch")
    atomic_json(membership_path, manifest)
    digest = sha256_file(membership_path)
    atomic_json(destination / "development-seal.json", {"membership_sha256": digest})
    return digest


class DevelopmentDataset:
    """Only fit/calibration normal paths; missing hashes or roles fail closed."""

    def __init__(self, root: Path, manifest: Path, *, expected_sha256: str) -> None:
        if sha256_file(manifest) != expected_sha256:
            raise VisaIntegrityError("Development membership identity mismatch")
        seal = json.loads((root / "development-seal.json").read_text())
        if seal != {"membership_sha256": expected_sha256}:
            raise VisaIntegrityError("Development root seal mismatch")
        document = json.loads(manifest.read_text())
        records = document["records"]
        if not records or [r["sample_id"] for r in records] != sorted(
            {r["sample_id"] for r in records}
        ):
            raise VisaIntegrityError(
                "Development membership missing, duplicated, or unordered"
            )
        for record in records:
            if (
                record["category"] not in CATEGORIES
                or record["role"] not in ("fit", "calibration")
                or record["path"].split("/")[:2] != [record["category"], record["role"]]
                or record["sample_id"].split("/")[:4]
                != [record["category"], "Data", "Images", "Normal"]
                or any(k in record for k in ("label", "mask", "mask_identity"))
            ):
                raise VisaIntegrityError(
                    "Test or label/mask metadata cannot enter development adapter"
                )
        self.root = root
        self.records = tuple(records)

    def paths(
        self, category: str, role: str, *, limit: int | None = None
    ) -> tuple[Path, ...]:
        if category not in CATEGORIES or role not in ("fit", "calibration"):
            raise VisaIntegrityError("Development adapter permits fit/calibration only")
        records = [
            r for r in self.records if r["category"] == category and r["role"] == role
        ]
        if limit is not None:
            if type(limit) is not int or limit < 1 or len(records) < limit:
                raise VisaIntegrityError("Invalid engineering subset size")
            records = records[:limit]
        if not records:
            raise VisaIntegrityError("Development role is empty")
        paths = []
        for record in records:
            path = safe_asset(self.root, record["path"])
            if sha256_file(path) != record["image_identity"]["sha256"]:
                raise VisaIntegrityError("Development image hash mismatch")
            paths.append(path)
        return tuple(paths)
