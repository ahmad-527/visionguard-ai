"""GPU engineering smoke/repeatability runner with no evaluation-split path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from visionguard.artifacts import capture_git_state, dataset_audit_identity
from visionguard.efficientad import (
    calibrate_efficientad_thresholds,
    canonical_checkpoint_sha256,
    restore_efficientad_map,
    verify_file_identity,
)
from visionguard.efficientad_artifacts import validate_efficientad_artifact
from visionguard.efficientad_protocol import (
    IMAGENETTE_ARCHIVE_SHA256,
    TEACHER_SMALL_SHA256,
    authorize_engineering_split,
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
)
from visionguard.environment import capture_environment
from visionguard.experiment import ReproducibilityConfig
from visionguard.reproducibility import configure_reproducibility


class EfficientAdSmokeError(RuntimeError):
    """Raised when a non-benchmark smoke run cannot satisfy its contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short EfficientAD GPU engineering check using only one "
            "category's train/good and validation/good data."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-weight", type=Path, required=True)
    parser.add_argument("--imagenette-root", type=Path, required=True)
    parser.add_argument("--imagenette-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=["can"], default="can")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], default=42)
    parser.add_argument("--steps", type=int, choices=range(1, 11), default=2)
    return parser


def _image_paths(root: Path, category: str, split: str) -> list[Path]:
    authorize_engineering_split(split)
    paths = sorted((root / category / split / "good").glob("*.png"))
    if not paths:
        raise EfficientAdSmokeError(f"No {category} {split}/good images found")
    return paths


def _map_sha256(array: Any) -> str:
    contiguous = array.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a deliberately short, single-category non-benchmark run."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from anomalib.models.image.efficient_ad.torch_model import (
            EfficientAdModel,
            EfficientAdModelSize,
            reduce_tensor_elems,
        )
        from PIL import Image
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        from torchvision.transforms.v2 import (
            CenterCrop,
            Compose,
            RandomGrayscale,
            Resize,
            ToTensor,
        )
    except ImportError as exc:
        raise EfficientAdSmokeError(
            "EfficientAD smoke dependencies are missing"
        ) from exc
    if not torch.cuda.is_available():
        raise EfficientAdSmokeError("EfficientAD engineering smoke requires CUDA")

    repository = Path.cwd().resolve()
    document = load_efficientad_protocol(args.protocol)
    protocol = document["protocol"]
    verify_file_identity(args.teacher_weight, TEACHER_SMALL_SHA256, "teacher")
    verify_file_identity(
        args.imagenette_archive, IMAGENETTE_ARCHIVE_SHA256, "ImageNette archive"
    )
    if not (args.imagenette_root / "train").is_dir():
        raise EfficientAdSmokeError("ImageNette train directory is missing")
    train_paths = _image_paths(args.dataset_root, args.category, "train")
    validation_paths = _image_paths(args.dataset_root, args.category, "validation")
    dataset_identity = dataset_audit_identity(
        args.audit_report, expected_root=args.dataset_root
    )
    reproducibility = configure_reproducibility(
        ReproducibilityConfig(
            seed=args.seed, deterministic_algorithms=True, cudnn_benchmark=False
        )
    )
    device = torch.device("cuda")
    transform = Compose([Resize((256, 256), antialias=True), ToTensor()])

    def load_image(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            size = (rgb.height, rgb.width)
            return transform(rgb).unsqueeze(0), size

    model = EfficientAdModel(
        teacher_out_channels=384,
        model_size=EfficientAdModelSize.S,
        padding=False,
        pad_maps=True,
    ).to(device)
    state = torch.load(args.teacher_weight, map_location=device, weights_only=True)
    model.teacher.load_state_dict(state)
    model.teacher.eval()

    # Anomalib computes channel statistics over every normal training image.
    channel_sum = torch.zeros(384, device=device)
    channel_sum_sqr = torch.zeros(384, device=device)
    count = 0
    with torch.no_grad():
        for path in train_paths:
            image, _ = load_image(path)
            features = model.teacher(image.to(device))
            count += features[:, 0].numel()
            channel_sum += features.sum(dim=(0, 2, 3))
            channel_sum_sqr += (features**2).sum(dim=(0, 2, 3))
    mean = channel_sum / count
    std = torch.sqrt(channel_sum_sqr / count - mean**2)
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise EfficientAdSmokeError("Teacher channel statistics are invalid")
    model.mean_std.update(
        {"mean": mean[None, :, None, None], "std": std[None, :, None, None]}
    )

    penalty_transform = Compose(
        [
            Resize((512, 512)),
            RandomGrayscale(p=0.3),
            CenterCrop((256, 256)),
            ToTensor(),
        ]
    )
    penalty_data = ImageFolder(
        args.imagenette_root / "train", transform=penalty_transform
    )
    generator = torch.Generator().manual_seed(args.seed)
    penalty_loader = DataLoader(
        penalty_data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    penalty_iterator = iter(penalty_loader)
    train_order = torch.randperm(
        len(train_paths), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    optimizer = torch.optim.Adam(
        list(model.student.parameters()) + list(model.ae.parameters()),
        lr=0.0001,
        weight_decay=0.00001,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(0.95 * protocol["training"]["max_steps"]),
        gamma=0.1,
    )
    loss_trajectory: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        image, _ = load_image(train_paths[train_order[step % len(train_paths)]])
        penalty = next(penalty_iterator)[0]
        optimizer.zero_grad(set_to_none=True)
        losses = model(image.to(device), batch_imagenet=penalty.to(device))
        loss = sum(losses)
        if not bool(torch.isfinite(loss)):
            raise EfficientAdSmokeError("Training produced NaN or infinity")
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_trajectory.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started

    # Native EfficientAD map normalization is fit on validation-normal maps only.
    model.eval()
    maps_st: list[torch.Tensor] = []
    maps_ae: list[torch.Tensor] = []
    with torch.no_grad():
        for path in validation_paths:
            image, _ = load_image(path)
            st, ae = model.get_maps(image.to(device), normalize=False)
            maps_st.append(st)
            maps_ae.append(ae)
    st_flat = reduce_tensor_elems(torch.cat(maps_st))
    ae_flat = reduce_tensor_elems(torch.cat(maps_ae))
    model.quantiles.update(
        {
            "qa_st": torch.quantile(st_flat, 0.9),
            "qb_st": torch.quantile(st_flat, 0.995),
            "qa_ae": torch.quantile(ae_flat, 0.9),
            "qb_ae": torch.quantile(ae_flat, 0.995),
        }
    )

    predictions: list[dict[str, Any]] = []
    image_scores: list[float] = []
    restored_values: list[list[float]] = []
    with torch.no_grad():
        for path in validation_paths:
            image, original_size = load_image(path)
            output = model(image.to(device))
            restored = restore_efficientad_map(output.anomaly_map[0, 0], original_size)
            if not bool(torch.isfinite(restored).all()):
                raise EfficientAdSmokeError("Validation map contains NaN or infinity")
            score = float(output.pred_score[0].cpu())
            image_scores.append(score)
            restored_values.append([float(restored.max().cpu())])
            predictions.append(
                {
                    "sample_id": path.relative_to(args.dataset_root).as_posix(),
                    "anomaly_score": score,
                    "anomaly_map": {
                        "shape": list(restored.shape),
                        "dtype": "float32",
                        "sha256": _map_sha256(restored),
                    },
                }
            )
    thresholds = calibrate_efficientad_thresholds(
        image_scores, restored_values, minimum_samples=19
    )
    checkpoint_sha = canonical_checkpoint_sha256(model.state_dict())
    artifact: dict[str, Any] = {
        "artifact_schema_version": 3,
        "protocol_id": protocol["id"],
        "protocol_fingerprint": efficientad_protocol_fingerprint(document),
        "protocol_snapshot": protocol,
        "experiment_id": f"phase3a-{args.category}-{args.seed}-{args.steps}steps",
        "run_kind": "phase3a_engineering_non_benchmark",
        "benchmark_claim": False,
        "evaluation_split": None,
        "git": capture_git_state(repository),
        "dataset": dataset_identity,
        "category": args.category,
        "seed": args.seed,
        "implementation": {"name": "anomalib", "version": "2.6.0"},
        "model": protocol["model"],
        "training": {
            "engineering_steps": args.steps,
            "benchmark_steps": protocol["training"]["max_steps"],
            "loss_trajectory": loss_trajectory,
        },
        "preprocessing": protocol["preprocessing"],
        "auxiliary_data": {
            "archive_sha256": IMAGENETTE_ARCHIVE_SHA256,
            "dataset": "imagenette2",
        },
        "environment": capture_environment("cuda"),
        "reproducibility": reproducibility,
        "weights": [{"component": "teacher_pdn_small", "sha256": TEACHER_SMALL_SHA256}],
        "calibration": {
            "normal_only": True,
            "split": "validation",
            "native_quantiles": {
                key: float(value.detach().cpu())
                for key, value in model.quantiles.items()
            },
        },
        "thresholds": thresholds,
        "model_state": {"checkpoint_sha256": checkpoint_sha},
        "predictions": predictions,
        "metrics": [],
        "resources": {
            "training_wall_seconds": training_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "warnings": [
            "Short engineering run; not the frozen 70,000-step benchmark recipe.",
            "No performance metric or cross-platform determinism claim is authorized.",
        ],
        "failures": [],
        "status": "completed",
    }
    validate_efficientad_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parser().parse_args(argv)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
