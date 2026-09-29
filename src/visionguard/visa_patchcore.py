"""Exact PatchCore development fitting with durable chunks and one CUDA embedding.

The installed feature extractor, projection, coreset algorithm and scoring are
unchanged. Only intermediate storage/residency is replaced. No test loader.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from visionguard.calibration import highest_order_statistic
from visionguard.embedding_journal import EmbeddingJournal
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


def tensor_hash(tensor) -> str:
    """Hash exact logical tensor bytes in bounded row blocks, not a full host copy."""
    value = tensor.detach()
    digest = hashlib.sha256(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    rows = value.reshape(-1) if value.ndim == 0 else value
    for start in range(0, len(rows), 256):
        digest.update(rows[start : start + 256].contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def durable_torch_save(path: Path, payload: dict) -> str:
    """Never overwrite a completed chunk/checkpoint; fsync before rename."""
    import torch

    if path.exists():
        raise VisaIntegrityError("Immutable checkpoint already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256_file(path)


def build_patchcore(science: dict, weight: Path, device):
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    from safetensors.torch import load_file
    from torchvision.transforms.v2 import Compose, Normalize, Resize, ToTensor

    expected = "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"
    if sha256_file(weight) != expected:
        raise VisaIntegrityError("PatchCore pretrained weight identity mismatch")
    settings = science["model"]
    model = PatchcoreModel(
        backbone=settings["backbone"],
        layers=settings["layers"],
        pre_trained=True,
        num_neighbors=settings["num_neighbors"],
    ).to(device)
    frozen = load_file(str(weight), device="cpu")
    for key, value in model.feature_extractor.feature_extractor.state_dict().items():
        if key not in frozen or not torch.equal(value.cpu(), frozen[key]):
            raise VisaIntegrityError("Loaded backbone differs from frozen weight")
    pre = science["preprocessing"]
    transform = Compose(
        [
            ToTensor(),
            Resize(tuple(pre["resize"]), antialias=True),
            Normalize(pre["normalization_mean"], pre["normalization_std"]),
        ]
    )
    return model, transform


def load_chunk(journal: EmbeddingJournal, record: dict) -> dict:
    from visionguard.efficientad_benchmark import _load_checkpoint

    payload = _load_checkpoint(journal.root / record["path"], record["sha256"])
    if (
        payload["identity"] != journal.identity
        or payload["sample_id"] != record["sample_id"]
        or payload["next_index"] != record["index"] + 1
        or payload["attempt"] != record["attempt"]
        or list(payload["embedding"].shape) != record["shape"]
        or str(payload["embedding"].dtype) != record["dtype"]
        or tensor_hash(payload["embedding"]) != record["tensor_sha256"]
    ):
        raise VisaIntegrityError("Embedding tensor payload identity mismatch")
    return payload


def fit_patchcore(
    model,
    transform,
    paths: tuple[Path, ...],
    journal: EmbeddingJournal,
    document: dict,
    *,
    device,
    ratio: float,
    reference: bool = False,
    exit_after: int | None = None,
    stop_before_coreset: bool = False,
) -> dict:
    """Exit 75 is an intentional hard process stop after a durable chunk receipt."""
    import torch
    from anomalib.models.components.sampling import KCenterGreedy

    from visionguard.efficientad_benchmark import (
        _load_image,
        _restore_rng_state,
        _rng_state,
    )

    if reference and document["next_index"]:
        raise VisaIntegrityError("Reference is uninterrupted only")
    if len(paths) != len(journal.samples):
        raise VisaIntegrityError("Embedding input membership length mismatch")
    if document["chunks"]:
        last = load_chunk(journal, document["chunks"][-1])
        _restore_rng_state(last["rng"])
        del last
    model.train()
    started = time.perf_counter()
    for index in range(document["next_index"], len(paths)):
        image, _ = _load_image(paths[index], transform)
        embedding = model(image.to(device))
        if (
            list(embedding.shape) != [1024, 1536]
            or embedding.dtype != torch.float32
            or not bool(torch.isfinite(embedding).all())
        ):
            raise VisaIntegrityError(
                "Frozen embedding shape/dtype/finite contract differs"
            )
        path = journal.directory / "chunks" / f"{index:06}.pt"
        payload = {
            "identity": journal.identity,
            "attempt": journal.relative,
            "sample_id": journal.samples[index],
            "next_index": index + 1,
            "embedding": embedding.detach().contiguous().cpu(),
            "rng": _rng_state(),
        }
        durable_torch_save(path, payload)
        document = journal.append(document, path, tensor_sha256=tensor_hash(embedding))
        if not reference:
            model.embedding_store.clear()
        del payload, embedding, image
        if exit_after == index + 1:
            os._exit(75)
    if stop_before_coreset:
        os._exit(75)
    extraction_seconds = time.perf_counter() - started
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    coreset_started = time.perf_counter()
    if reference:
        # Independent native aggregation, not the replacement reconstruction.
        ordered = torch.vstack(model.embedding_store)
        ordered_hash = tensor_hash(ordered)
        del ordered
    else:
        model.memory_bank = torch.empty(
            (len(paths) * 1024, 1536), dtype=torch.float32, device=device
        )
        for record in document["chunks"]:
            payload = load_chunk(journal, record)
            start = record["index"] * 1024
            model.memory_bank[start : start + 1024].copy_(payload["embedding"])
            del payload
        ordered_hash = tensor_hash(model.memory_bank)
    indices = []
    original_select = KCenterGreedy.select_coreset_idxs

    def record_indices(sampler):
        result = original_select(sampler)
        indices.extend(result)
        return result

    # Observation only: return the original indices without modifying algorithm/RNG.
    with patch.object(KCenterGreedy, "select_coreset_idxs", record_indices):
        if reference:
            model.subsample_embedding(sampling_ratio=ratio)
        else:
            sampler = KCenterGreedy(embedding=model.memory_bank, sampling_ratio=ratio)
            model.memory_bank = sampler.sample_coreset()
            del sampler
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {
        "ordered_embeddings_sha256": ordered_hash,
        "embedding_shape": [len(paths) * 1024, 1536],
        "dtype": "torch.float32",
        "embedding_bytes": len(paths) * 1024 * 1536 * 4,
        "coreset_indices": indices,
        "memory_bank_sha256": tensor_hash(model.memory_bank),
        "memory_bank_shape": list(model.memory_bank.shape),
        "extraction_seconds_this_process": extraction_seconds,
        "coreset_and_reconstruction_seconds": time.perf_counter() - coreset_started,
        "coreset_completed": True,
    }


def calibrate_development(model, transform, paths, device, directory: Path) -> dict:
    """Exact scores/maps and frozen maximum thresholds on calibration normals only."""
    import numpy as np
    import tifffile
    import torch
    from PIL import Image

    from visionguard.efficientad_benchmark import _load_image
    from visionguard.preprocessing import restore_anomaly_map

    model.eval()
    rows = []
    first = None
    with torch.no_grad():
        for index, path in enumerate(paths):
            image, shape = _load_image(path, transform)
            prediction = model(image.to(device))
            restored = restore_anomaly_map(prediction.anomaly_map[0, 0], shape)
            if not bool(torch.isfinite(restored).all()):
                raise VisaIntegrityError("Nonfinite calibration map")
            rows.append(
                {
                    "normal_id": path.name,
                    "score": float(prediction.pred_score[0]),
                    "pixel_maximum": float(restored.max()),
                    "map_sha256": tensor_hash(restored),
                    "shape": list(shape),
                }
            )
            if index == 0:
                first = restored.cpu().numpy()
    image = highest_order_statistic([r["score"] for r in rows], minimum_samples=19)
    pixel = highest_order_statistic(
        [r["pixel_maximum"] for r in rows], minimum_samples=19
    )
    if first is None or np.abs(first).max() > np.finfo(np.float16).max:
        raise VisaIntegrityError("Calibration map cannot be stored as float16")
    tifffile.imwrite(directory / "calibration-normal.tiff", first.astype(np.float16))
    Image.fromarray((first > pixel.threshold).astype(np.uint8) * 255).save(
        directory / "calibration-normal-thresholded.png"
    )
    result = {"image": asdict(image), "pixel": asdict(pixel), "inputs": rows}
    atomic_json(directory / "calibration.json", result)
    return result
