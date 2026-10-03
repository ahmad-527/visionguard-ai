"""Immutable synthetic artifact receipts and exact reducer replay, no test loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from visionguard.visa_acquire import atomic_json, sha256_file
from visionguard.visa_evaluator import (
    ModelSpec,
    Prediction,
    SyntheticSample,
    evaluate_synthetic_cell,
    require,
)


def canonical_bytes(document: dict) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def publish_synthetic(
    directory: Path,
    samples: list[SyntheticSample],
    pc: list[Prediction],
    ea: list[Prediction],
    specs: tuple[ModelSpec, ModelSpec],
) -> dict:
    """Validate before writes. Never overwrite an attempt, even a partial one.

    Completion manifest written LAST; failed output stays visibly partial. Native
    float32 binary decisions are saved before continuous float16 quantization.
    Ground truth here is manufactured synthetic data, not an external asset.
    """
    import tifffile
    from PIL import Image

    reduced = evaluate_synthetic_cell(samples, pc, ea, specs)
    require(not directory.exists(), "Immutable output attempt already exists")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "evidence_class": "synthetic_engineering_only",
        "specs": [asdict(s) for s in specs],
        "images": [],
        "result_sha256": hashlib.sha256(canonical_bytes(reduced)).hexdigest(),
    }
    for index, (sample, p, e) in enumerate(zip(samples, pc, ea, strict=True)):
        mask_path = directory / f"{index:06d}-truth.png"
        Image.fromarray(sample.mask.astype(np.uint8)).save(mask_path)
        row = {
            "sample_id": sample.sample_id,
            "category": sample.category,
            "label": sample.label,
            "mask": {"path": mask_path.name, "sha256": sha256_file(mask_path)},
            "predictions": {},
        }
        for prediction in (p, e):
            stem = f"{index:06d}-{prediction.spec.model}"
            continuous, binary = directory / f"{stem}.tiff", directory / f"{stem}.png"
            tifffile.imwrite(
                continuous, prediction.restored_map.astype(np.float16), compression=None
            )
            thresholded = (
                prediction.restored_map > prediction.spec.pixel_threshold
                if prediction.thresholded_map is None
                else prediction.thresholded_map
            )
            Image.fromarray(thresholded.astype(np.uint8)).save(binary)
            row["predictions"][prediction.spec.model] = {
                "score": prediction.score,
                "continuous": {
                    "path": continuous.name,
                    "sha256": sha256_file(continuous),
                },
                "binary": {"path": binary.name, "sha256": sha256_file(binary)},
            }
        manifest["images"].append(row)
    # All file hashes present before this atomic completion record is published.
    atomic_json(directory / "synthetic-manifest.json", manifest)
    return manifest


def replay_synthetic(
    directory: Path,
    expected_manifest_sha256: str,
    expected_specs: tuple[ModelSpec, ModelSpec],
    expected_membership_sha256: str,
) -> dict:
    """Read only SHA-bound synthetic bundle, never accept caller test paths.

    This is an explicit synthetic replay API; NOT a real-dataset evaluation CLI.
    Completion manifest must identify synthetic evidence before any map is opened.
    Expected origin/membership comes from the producer, never supplied-as-expected
    authorization; the closed test gate has no route into this reader.
    """
    import tifffile
    from PIL import Image

    path = directory / "synthetic-manifest.json"
    require(
        sha256_file(path) == expected_manifest_sha256, "Synthetic manifest corruption"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(
        manifest["evidence_class"] == "synthetic_engineering_only",
        "Not synthetic evidence",
    )
    require(
        manifest["specs"] == [asdict(s) for s in expected_specs],
        "Wrong replay models/seeds/thresholds",
    )

    def asset(record: dict) -> Path:
        relative = Path(record["path"])
        require(
            len(relative.parts) == 1 and not relative.is_absolute(),
            "Escaping synthetic artifact path",
        )
        selected = directory / relative
        require(
            selected.resolve().parent == directory.resolve(),
            "Escaping synthetic symlink",
        )
        require(
            sha256_file(selected) == record["sha256"], "Synthetic artifact corruption"
        )
        return selected

    samples, pcs, eas = [], [], []
    ids = [r["sample_id"] for r in manifest["images"]]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    require(digest == expected_membership_sha256, "Replay membership/order mismatch")
    for row in manifest["images"]:
        require(
            row["sample_id"].startswith(f"synthetic:{row['category']}:")
            and "/" not in row["sample_id"]
            and "\\" not in row["sample_id"],
            "Not synthetic ID",
        )
        with Image.open(asset(row["mask"])) as image:
            mask = np.asarray(image).copy()
        sample = SyntheticSample(row["sample_id"], row["category"], row["label"], mask)
        samples.append(sample)
        for spec, target in zip(expected_specs, (pcs, eas), strict=True):
            record = row["predictions"][spec.model]
            continuous = tifffile.imread(asset(record["continuous"]))
            with Image.open(asset(record["binary"])) as image:
                binary = np.asarray(image).copy()
            require(
                binary.dtype == np.uint8 and bool(np.isin(binary, (0, 1)).all()),
                "Invalid saved binary mask",
            )
            target.append(
                Prediction(
                    sample.sample_id,
                    spec,
                    record["score"],
                    continuous,
                    binary.astype(bool),
                )
            )
    result = evaluate_synthetic_cell(samples, pcs, eas, expected_specs)
    require(
        hashlib.sha256(canonical_bytes(result)).hexdigest()
        == manifest["result_sha256"],
        "Replay result differs",
    )
    return result
