"""Fixed, tiny VisA engineering smoke on development normals, never final test."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from visionguard.artifacts import capture_git_state
from visionguard.calibration import highest_order_statistic
from visionguard.triage import triage_same_seed
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_development import DevelopmentDataset
from visionguard.visa_protocol import (
    load_visa_protocol,
    verify_audit,
    verify_implementation,
)


def _deny_sealed_access(development: Path):
    """Runtime defense: deny filesystem opens of the sibling sealed dataset."""
    sealed = (development.parent / "sealed").resolve()

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path == sealed or path.is_relative_to(sealed):
                raise VisaIntegrityError(
                    "Engineering smoke attempted sealed-data access"
                )

    sys.addaudithook(audit)


def _calibrate_normals(model, paths, transform, device, output: Path) -> dict:
    """Calibration only. No labels, binary metrics, AUROC, or AU-PRO are computed."""
    import torch

    from visionguard.efficientad_benchmark import _load_image
    from visionguard.preprocessing import restore_anomaly_map

    inputs = []
    model.eval()
    with torch.no_grad():
        for index, path in enumerate(paths):
            image, shape = _load_image(path, transform)
            prediction = model(image.to(device))
            restored = restore_anomaly_map(prediction.anomaly_map[0, 0], shape)
            score = float(prediction.pred_score[0].cpu())
            maximum = float(restored.max().cpu())
            inputs.append(
                {
                    "ordinal": index,
                    "normal_id": path.name,
                    "score": score,
                    "pixel_maximum": maximum,
                    "shape": list(shape),
                }
            )
    image = highest_order_statistic([r["score"] for r in inputs], minimum_samples=19)
    pixel = highest_order_statistic(
        [r["pixel_maximum"] for r in inputs], minimum_samples=19
    )
    # Exercise artifact writing on one calibration normal only, without panels.
    with torch.no_grad():
        tensor, shape = _load_image(paths[0], transform)
        prediction = model(tensor.to(device))
        restored = restore_anomaly_map(prediction.anomaly_map[0, 0], shape)
    # Preserve continuous float16 and strict threshold binary maps separately.
    import numpy as np
    import tifffile
    from PIL import Image

    continuous = output / "calibration-normal.tiff"
    thresholded = output / "calibration-normal-thresholded.png"
    array = restored.detach().cpu().numpy()
    if not np.isfinite(array).all() or np.abs(array).max() > np.finfo(np.float16).max:
        raise VisaIntegrityError("Map cannot be stored as finite float16")
    tifffile.imwrite(continuous, array.astype(np.float16), photometric="minisblack")
    Image.fromarray((array > pixel.threshold).astype(np.uint8) * 255).save(thresholded)
    return {
        "normal_count": len(inputs),
        "image": asdict(image),
        "pixel": asdict(pixel),
        "inputs": inputs,
        "maps": {p.name: sha256_file(p) for p in (continuous, thresholded)},
    }


def run_smoke(args: argparse.Namespace) -> dict:
    """Two optimization steps; interruption after step one is intentional."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    from torchvision.transforms.v2 import Compose, Normalize, Resize, ToTensor

    from visionguard.efficientad import (
        canonical_checkpoint_sha256,
        verify_file_identity,
    )
    from visionguard.efficientad_benchmark import (
        _atomic_torch_save,
        _build_training,
        _load_checkpoint,
        _load_image,
        _load_transforms,
        _native_quantiles,
        _penalty_paths,
        _restore_rng_state,
        _rng_state,
        _train,
    )
    from visionguard.efficientad_protocol import (
        IMAGENETTE_ARCHIVE_SHA256,
        TEACHER_SMALL_SHA256,
    )
    from visionguard.experiment import ReproducibilityConfig
    from visionguard.reproducibility import configure_reproducibility

    if not torch.cuda.is_available():
        raise VisaIntegrityError("CUDA unavailable; engineering smoke not run")
    protocols = {}
    for model in ("patchcore", "efficientad"):
        protocols[model] = load_visa_protocol(
            args.repository / f"configs/protocols/{model}-visa-v1.yaml",
            expected_fingerprint=args.fingerprints[model],
        )["protocol"]
    for protocol in protocols.values():
        verify_audit(args.audit, protocol["dataset"]["audit_sha256"])
        verify_implementation(args.repository, protocol)
    membership = protocols["patchcore"]["dataset"]["membership_sha256"]
    if protocols["efficientad"]["dataset"]["membership_sha256"] != membership:
        raise VisaIntegrityError("Models must share identical development membership")
    dataset = DevelopmentDataset(
        args.development, args.membership, expected_sha256=membership
    )
    _deny_sealed_access(args.development)
    fit = dataset.paths("candle", "fit", limit=4)
    calibration = dataset.paths("candle", "calibration", limit=19)
    verify_file_identity(
        args.patchcore_weight,
        "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000",
        "PatchCore",
    )
    verify_file_identity(args.teacher_weight, TEACHER_SMALL_SHA256, "teacher")
    verify_file_identity(
        args.imagenette_archive, IMAGENETTE_ARCHIVE_SHA256, "ImageNette"
    )
    for protocol in protocols.values():
        for package, expected in protocol["scientific"]["dependencies"].items():
            if package not in ("python", "cuda_runtime") and (
                importlib.metadata.version(package).split("+")[0] != str(expected)
            ):
                raise VisaIntegrityError(f"Dependency mismatch: {package}")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "purpose": "ENGINEERING_ONLY_NOT_BENCHMARK_PERFORMANCE",
        "category": "candle",
        "seed": 42,
        "protocol_fingerprints": args.fingerprints,
        "membership_sha256": membership,
        "audit_sha256": sha256_file(args.audit),
        "git": capture_git_state(args.repository),
        "fit_samples": 4,
        "calibration_samples": 19,
        "optimization_steps": 2,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("anomalib", "timm", "torchvision", "numpy", "Pillow")
            },
        },
    }
    atomic_json(args.output / "origin.json", identity)
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "identity": identity,
        "test_performance_evaluated": False,
        "final_test_lock": "closed",
        "status": "running",
        "models": {},
    }
    atomic_json(args.output / "report.json", report)
    try:
        device = torch.device("cuda")
        configure_reproducibility(ReproducibilityConfig(42, True, False))
        torch.cuda.reset_peak_memory_stats()
        pc_dir = args.output / "patchcore"
        pc_dir.mkdir()
        settings = protocols["patchcore"]["scientific"]["model"]
        model = PatchcoreModel(
            backbone=settings["backbone"],
            layers=settings["layers"],
            pre_trained=True,
            num_neighbors=settings["num_neighbors"],
        ).to(device)
        from safetensors.torch import load_file

        frozen_weights = load_file(str(args.patchcore_weight), device="cpu")
        for (
            key,
            value,
        ) in model.feature_extractor.feature_extractor.state_dict().items():
            if key not in frozen_weights or not torch.equal(
                value.cpu(), frozen_weights[key]
            ):
                raise VisaIntegrityError(f"Loaded PatchCore weight differs: {key}")
        del frozen_weights
        transform = Compose(
            [
                ToTensor(),
                Resize((256, 256), antialias=True),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        model.train()
        pc_started = time.perf_counter()
        for index, path in enumerate(fit):
            tensor, _ = _load_image(path, transform)
            model(tensor.to(device))
            if index == 1:
                checkpoint = pc_dir / "embedding-checkpoint.pt"
                digest = _atomic_torch_save(
                    checkpoint,
                    {
                        "identity": identity,
                        "next_index": 2,
                        "embeddings": [e.cpu() for e in model.embedding_store],
                        "rng": _rng_state(),
                    },
                )
                loaded = _load_checkpoint(checkpoint, digest)
                if loaded["identity"] != identity or loaded["next_index"] != 2:
                    raise VisaIntegrityError("PatchCore engineering resume mismatch")
                model.embedding_store = [e.to(device) for e in loaded["embeddings"]]
                _restore_rng_state(loaded["rng"])
        model.subsample_embedding(float(settings["coreset_sampling_ratio"]))
        torch.cuda.synchronize()
        pc_fit_seconds = time.perf_counter() - pc_started
        pc_calibration = _calibrate_normals(
            model, calibration, transform, device, pc_dir
        )
        pc_digest = _atomic_torch_save(
            pc_dir / "final-memory-bank.pt",
            {
                "identity": identity,
                "memory_bank": model.memory_bank,
                "pretrained_weight_sha256": sha256_file(args.patchcore_weight),
            },
        )
        report["models"]["patchcore"] = {
            "fit_seconds": pc_fit_seconds,
            "checkpoint_sha256": pc_digest,
            "canonical_tensor_sha256": canonical_checkpoint_sha256(model.state_dict()),
            "resume": "in_process_embedding_checkpoint_roundtrip_passed",
            "calibration": pc_calibration,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        configure_reproducibility(ReproducibilityConfig(42, True, False))
        torch.cuda.reset_peak_memory_stats()
        ead_dir = args.output / "efficientad"
        ead_dir.mkdir()
        scientific = protocols["efficientad"]["scientific"]
        model, optimizer, scheduler = _build_training(
            teacher_weight=args.teacher_weight, device=device, protocol=scientific
        )
        normal_transform, penalty_transform = _load_transforms()
        penalty_paths = _penalty_paths(args.imagenette_root)
        entry = {}
        manifest = {"identity": identity, "entry": entry}
        checkpoint_path = ead_dir / "training-checkpoint.pt"
        common = dict(
            train_paths=fit,
            penalty_paths=penalty_paths,
            normal_transform=normal_transform,
            penalty_transform=penalty_transform,
            device=device,
            protocol=scientific,
            identity=identity,
            entry=entry,
            manifest=manifest,
            manifest_path=ead_dir / "training-state.json",
            checkpoint_path=checkpoint_path,
            output_root=args.output,
        )
        interruptions = []
        for stop_step in (1, 2):
            checkpoint = (
                None
                if stop_step == 1
                else _load_checkpoint(
                    checkpoint_path, entry["latest_valid_checkpoint"]["sha256"]
                )
            )
            if checkpoint is not None:
                # Reconstruct before restoring checkpoint and RNG streams.
                del model, optimizer, scheduler
                gc.collect()
                torch.cuda.empty_cache()
                model, optimizer, scheduler = _build_training(
                    teacher_weight=args.teacher_weight,
                    device=device,
                    protocol=scientific,
                )
            try:
                _train(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint=checkpoint,
                    stop_after_step=stop_step,
                    **common,
                )
            except KeyboardInterrupt:
                if entry["latest_valid_checkpoint"]["step"] != stop_step:
                    raise
                interruptions.append(
                    {"reason": "intentional_engineering_stop", "step": stop_step}
                )
        quantiles = _native_quantiles(model, fit, normal_transform, device)
        ead_calibration = _calibrate_normals(
            model, calibration, normal_transform, device, ead_dir
        )
        final_digest = _atomic_torch_save(
            ead_dir / "final-engineering-model.pt",
            {
                "identity": identity,
                "model_state": model.state_dict(),
                "step": 2,
                "not_scientific_final_checkpoint": True,
            },
        )
        report["models"]["efficientad"] = {
            "resume": "model_optimizer_scheduler_rng_stream_restore_passed",
            "intentional_interruptions": interruptions,
            "quantiles_fit_role": "fit",
            "quantiles": quantiles,
            "calibration": ead_calibration,
            "checkpoint_sha256": final_digest,
            "canonical_tensor_sha256": canonical_checkpoint_sha256(model.state_dict()),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "training_state": entry,
        }
        compatible = triage_same_seed(
            pc_calibration["inputs"][0]["score"],
            pc_calibration["image"]["threshold"],
            ead_calibration["inputs"][0]["score"],
            ead_calibration["image"]["threshold"],
            patchcore_seed=42,
            efficientad_seed=42,
        )
        report["triage_interface_check"] = {
            "same_calibration_normal": True,
            "valid_three_way_output": compatible.triage_decision.value
            in {"PASS", "REVIEW", "REJECT"},
        }
        report["status"] = "passed"
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure_type=type(exc).__name__,
        )
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(args.output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "development",
        "membership",
        "audit",
        "freeze-record",
        "patchcore-weight",
        "teacher-weight",
        "imagenette-root",
        "imagenette-archive",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    args.fingerprints = json.loads(args.freeze_record.read_text())[
        "protocol_fingerprints"
    ]
    report = run_smoke(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "test_performance_evaluated": False,
                "final_test_lock": "closed",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
