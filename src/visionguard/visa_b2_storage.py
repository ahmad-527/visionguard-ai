"""Exclusive immutable synthetic output publication and independent replay.

This module has no test-root interface. Failure leaves partial attempts intact;
completion is written once, last. There is no replace-live-file retry loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict
from itertools import tee
from pathlib import Path

import numpy as np
from PIL import Image

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import (
    ModelSpec,
    Prediction,
    SyntheticSample,
    evaluate_synthetic_cell,
    require,
    validate_prediction,
    validate_sample,
)
from visionguard.visa_evaluator_storage import canonical_bytes


def write_once(path: Path, content: bytes) -> dict:
    """O_EXCL + fsync + exact readback; a partial file is never overwritten."""
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(content).hexdigest()
    require(sha256_file(path) == digest, "Publication rehash failed")
    return {"path": path.name, "bytes": len(content), "sha256": digest}


def json_once(path: Path, payload: dict) -> dict:
    return write_once(path, canonical_bytes(payload))


def _png(path: Path, array: np.ndarray) -> dict:
    import io

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return write_once(path, buffer.getvalue())


def publish_cell(
    directory: Path,
    records: Iterable[tuple[SyntheticSample, Prediction, Prediction]],
    specs: tuple[ModelSpec, ModelSpec],
    origin: dict,
    *,
    interrupt_after: int | None = None,
) -> dict:
    """Stream one paired image; retain float16 maps and native binary decisions."""
    import tifffile

    directory.mkdir(parents=True, exist_ok=False)
    origin_record = json_once(
        directory / "origin.json",
        {"specs": [asdict(s) for s in specs], "source": origin},
    )
    images = []

    def producer():
        for i, (sample, pc, ea) in enumerate(records):
            validate_sample(sample)
            row = {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "shape": list(sample.mask.shape),
                "predictions": {},
            }
            row["mask"] = _png(directory / f"{i:06d}-truth.png", sample.mask)
            for prediction, spec in zip((pc, ea), specs, strict=True):
                validate_prediction(sample, prediction, spec)
                stem = f"{i:06d}-{spec.model}"
                target = directory / f"{stem}.tiff"
                with target.open("xb") as handle:
                    tifffile.imwrite(
                        handle,
                        prediction.restored_map.astype(np.float16),
                        compression=None,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                binary = (
                    prediction.restored_map > spec.pixel_threshold
                    if prediction.thresholded_map is None
                    else prediction.thresholded_map
                )
                row["predictions"][spec.model] = {
                    "score": prediction.score,
                    "continuous": {
                        "path": target.name,
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                    },
                    "binary": _png(directory / f"{stem}.png", binary.astype(np.uint8)),
                }
            images.append(row)
            if interrupt_after is not None and len(images) == interrupt_after:
                raise InterruptedError("Declared synthetic interruption")
            yield sample, pc, ea

    streams = tee(producer(), 3)
    try:
        result = evaluate_synthetic_cell(
            (r[0] for r in streams[0]),
            (r[1] for r in streams[1]),
            (r[2] for r in streams[2]),
            specs,
        )
        result_record = json_once(directory / "result.json", result)
        manifest = {
            "schema_version": 1,
            "evidence_class": "synthetic_engineering_only",
            "specs": [asdict(s) for s in specs],
            "origin": origin,
            "origin_record": origin_record,
            "images": images,
            "result": result_record,
            "membership_sha256": result["membership_sha256"],
        }
        receipt = json_once(directory / "complete.json", manifest)
        replay_cell(directory, receipt["sha256"], specs, origin)
        return receipt
    except BaseException as exc:
        # This file describes failure, never completion. No guessed returncode.
        try:
            json_once(
                directory / "failure.json",
                {
                    "status": "interrupted"
                    if isinstance(exc, InterruptedError)
                    else "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "winerror": getattr(exc, "winerror", None),
                    "images_written": len(images),
                },
            )
        except OSError as receipt_error:
            exc.add_note(f"Failure receipt publication also failed: {receipt_error}")
        raise


def replay_cell(
    directory: Path, expected_sha: str, specs: tuple[ModelSpec, ModelSpec], origin: dict
) -> dict:
    """Independently SHA-verify every retained file, then stream the same reducer."""
    import tifffile

    require(
        sha256_file(directory / "complete.json") == expected_sha,
        "Completion corruption",
    )
    manifest = json.loads((directory / "complete.json").read_text())
    require(
        manifest["evidence_class"] == "synthetic_engineering_only",
        "Not synthetic evidence",
    )
    require(
        manifest["specs"] == [asdict(s) for s in specs]
        and manifest["origin"] == origin,
        "Wrong origin/models/membership",
    )

    def asset(row):
        relative = Path(row["path"])
        require(
            len(relative.parts) == 1
            and relative.name not in (".", "..")
            and not relative.is_absolute(),
            "Escaping artifact",
        )
        path = directory / relative
        state = path.lstat()
        require(
            not (getattr(state, "st_file_attributes", 0) & 0x400)
            and not path.is_symlink(),
            "Reparse artifact",
        )
        require(
            state.st_size == row["bytes"] and sha256_file(path) == row["sha256"],
            "Artifact corruption",
        )
        return path

    origin_path = asset(manifest["origin_record"])
    require(
        json.loads(origin_path.read_text())
        == {"specs": [asdict(s) for s in specs], "source": origin},
        "Origin receipt changed",
    )

    def producer():
        for row in manifest["images"]:
            with Image.open(asset(row["mask"])) as image:
                mask = np.asarray(image).copy()
            sample = SyntheticSample(
                row["sample_id"], specs[0].category, row["label"], mask
            )
            require(list(mask.shape) == row["shape"], "Original dimensions changed")
            predictions = []
            for spec in specs:
                record = row["predictions"][spec.model]
                array = tifffile.imread(asset(record["continuous"]))
                require(array.dtype == np.float16, "Wrong continuous export dtype")
                with Image.open(asset(record["binary"])) as image:
                    binary = np.asarray(image).copy()
                require(
                    binary.dtype == np.uint8 and bool(np.isin(binary, (0, 1)).all()),
                    "Invalid binary map",
                )
                predictions.append(
                    Prediction(
                        sample.sample_id,
                        spec,
                        record["score"],
                        array,
                        binary.astype(bool),
                    )
                )
            yield sample, *predictions

    streams = tee(producer(), 3)
    result = evaluate_synthetic_cell(
        (r[0] for r in streams[0]),
        (r[1] for r in streams[1]),
        (r[2] for r in streams[2]),
        specs,
    )
    require(
        result["membership_sha256"] == manifest["membership_sha256"],
        "Reordered/missing source",
    )
    require(
        hashlib.sha256(canonical_bytes(result)).hexdigest()
        == manifest["result"]["sha256"],
        "Metric replay differs",
    )
    asset(manifest["result"])
    return result
