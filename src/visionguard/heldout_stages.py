"""Immutable model stages; paired replay references them, never duplicates maps."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from visionguard.heldout_metrics import (
    Prediction,
    Sample,
    evaluate_cell,
    validate_prediction,
    validate_sample,
)
from visionguard.heldout_paths import asset, checked
from visionguard.heldout_resources import account_write, before_write
from visionguard.heldout_storage import _png, json_once
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_evaluator_storage import canonical_bytes


def fail(directory, exc, completed_calls=None):
    try:
        json_once(
            directory / "failure.json",
            {
                "status": "interrupted"
                if isinstance(exc, InterruptedError)
                else "failed",
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "winerror": getattr(exc, "winerror", None),
                "completed_inference_calls": completed_calls,
            },
        )
    except OSError as secondary:
        exc.add_note(f"Failure receipt publication failed: {secondary}")


def publish_stage(
    directory,
    frames,
    spec,
    origin,
    backend,
    *,
    interrupt_after=None,
    check=lambda: None,
):
    import tifffile

    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    calls = 0
    try:
        json_once(directory / "origin.json", {"spec": asdict(spec), "source": origin})
        from visionguard.heldout_backend import RealFrame

        for i, frame in enumerate(frames):
            check()
            sample = frame.sample
            validate_sample(sample)
            prediction = backend.predict(
                RealFrame(sample.sample_id, sample.category, frame.rgb)
            )
            calls += 1
            validate_prediction(sample, prediction, spec)
            require(
                prediction.restored_map.dtype == np.float32,
                "Native float32 map required",
            )
            target = directory / f"{i:06d}.tiff"
            before_write(prediction.restored_map.size * 2 + 4096)
            with target.open("xb") as handle:
                tifffile.imwrite(
                    handle, prediction.restored_map.astype(np.float16), compression=None
                )
                handle.flush()
                os.fsync(handle.fileno())
            account_write(target.stat().st_size)
            row = {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "shape": list(sample.mask.shape),
                "score": prediction.score,
                "continuous": {
                    "path": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                },
                "binary": _png(
                    directory / f"{i:06d}-binary.png",
                    (prediction.restored_map > spec.pixel_threshold).astype(np.uint8),
                ),
                "truth": _png(directory / f"{i:06d}-truth.png", sample.mask),
            }
            rows.append(row)
            if interrupt_after == len(rows):
                raise InterruptedError("Declared artificial model-stage interruption")
        require(
            [r["sample_id"] for r in rows] == origin["membership"][spec.category],
            "Model stage membership differs",
        )
        complete = json_once(
            directory / "complete.json",
            {"spec": asdict(spec), "origin": origin, "images": rows},
        )
        verify_stage(directory, complete["sha256"], spec, origin)
        json_once(directory / "validated.json", complete)
        return complete
    except BaseException as exc:
        fail(directory, exc, calls)
        raise


def _manifest(directory, sha, spec, origin):
    checked(directory)
    require(
        sha256_file(asset(directory, "complete.json")) == sha,
        "Stage completion corruption",
    )
    doc = json.loads((directory / "complete.json").read_text())
    require(
        doc["spec"] == asdict(spec) and doc["origin"] == origin,
        "Stage identity mismatch",
    )
    require(
        json.loads(asset(directory, "origin.json").read_text())
        == {"spec": asdict(spec), "source": origin},
        "Stage origin differs",
    )
    require(
        [r["sample_id"] for r in doc["images"]] == origin["membership"][spec.category],
        "Missing/reordered stage image",
    )
    return doc


def read_stage(directory, sha, spec, origin):
    import tifffile

    doc = _manifest(directory, sha, spec, origin)

    def bound(row):
        require(len(Path(row["path"]).parts) == 1, "Escaping stage asset")
        path = asset(directory, row["path"])
        require(
            path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
            "Stage artifact corruption",
        )
        return path

    for row in doc["images"]:
        array = tifffile.imread(bound(row["continuous"]))
        require(array.dtype == np.float16, "Continuous dtype changed")
        with Image.open(bound(row["truth"])) as image:
            mask = np.asarray(image).copy()
        with Image.open(bound(row["binary"])) as image:
            binary = np.asarray(image).copy()
        require(
            binary.dtype == np.uint8 and bool(np.isin(binary, (0, 1)).all()),
            "Invalid binary evidence",
        )
        require(list(mask.shape) == row["shape"], "Original dimensions changed")
        sample = Sample(row["sample_id"], spec.category, row["label"], mask)
        pred = Prediction(
            row["sample_id"], spec, row["score"], array, binary.astype(bool)
        )
        validate_sample(sample)
        validate_prediction(sample, pred, spec)
        yield sample, pred


def verify_stage(directory, sha, spec, origin):
    return sum(1 for _ in read_stage(directory, sha, spec, origin))


def paired_result(stages, specs, origin):
    from itertools import tee, zip_longest

    def pairs():
        for pc, ea in zip_longest(
            *(
                read_stage(path, sha, spec, stage_origin)
                for (path, sha, stage_origin), spec in zip(stages, specs, strict=True)
            )
        ):
            require(pc is not None and ea is not None, "Unpaired stage")
            s, p = pc
            e, q = ea
            require(
                s.sample_id == e.sample_id
                and s.label == e.label
                and np.array_equal(s.mask, e.mask),
                "Paired truth differs",
            )
            yield s, p, q

    streams = tee(pairs(), 3)
    return evaluate_cell(
        (r[0] for r in streams[0]),
        (r[1] for r in streams[1]),
        (r[2] for r in streams[2]),
        specs,
    )


def publish_pair(directory, stages, specs, origin):
    directory.mkdir(parents=True, exist_ok=False)
    try:
        result = paired_result(stages, specs, origin)
        require(
            canonical_bytes(result)
            == canonical_bytes(paired_result(stages, specs, origin)),
            "Independent paired replay differs",
        )
        result_receipt = json_once(directory / "result.json", result)
        receipt = json_once(
            directory / "complete.json",
            {
                "stages": [
                    {"path": str(p.relative_to(directory.parent.parent)), "sha256": sha}
                    for p, sha, _ in stages
                ],
                "specs": [asdict(s) for s in specs],
                "origin": origin,
                "result": result_receipt,
            },
        )
        json_once(directory / "validated.json", receipt)
        return result, receipt
    except BaseException as exc:
        fail(directory, exc)
        raise
