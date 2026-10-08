"""CPU development-state adapter with explicit, hash-bound application permission.

No dataset loader, training, checkpoint export, download, or held-out admission.
Existing evaluation permission is not repurposed as application permission.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from visionguard.inspection_contract import (
    InspectionError,
    InspectionInput,
    ModelManifest,
    ModelOutput,
)
from visionguard_inspection.registry import Registration, read_bound


def approved_registration(root: Path, item: dict) -> Registration:
    """Require permission before opening any model/preprocessing/calibration file.

    Permission is a human-provided, operator-pinned use attestation, not a signing
    mechanism or proof that an arbitrary JSON author is human. Operators must
    establish that authority independently; the application cannot issue it.
    """
    approval = json.loads(read_bound(root, item["application_permission"]))
    required = {
        "schema_version": 1,
        "scope": "application-development-inference",
        "source_role": "development",
        "model_id": item["model_id"],
        "model": item["model"],
        "category": item["category"],
        "seed": item["seed"],
        "artifact_sha256": item["artifact"]["sha256"],
        "science_sha256": item["science"]["sha256"],
        "calibration_sha256": item["calibration"]["sha256"],
        "canonical_state_sha256": item["canonical_state_sha256"],
        "device": "cpu",
        "input_scope": "generated-or-development-non-held-out",
    }
    if any(approval.get(k) != v for k, v in required.items()):
        raise InspectionError("Application permission scope/binding mismatch")
    if (
        not isinstance(approval.get("human_attestation"), str)
        or not approval["human_attestation"].strip()
    ):
        raise InspectionError("Human application-use attestation is required")
    expiry = datetime.fromisoformat(approval["expires_at_utc"])
    if expiry.utcoffset() is None or datetime.now(UTC) >= expiry:
        raise InspectionError("Application permission is expired/invalid")
    if item["model"] not in {"patchcore", "efficientad"}:
        raise InspectionError("Unsupported native architecture")
    science = json.loads(read_bound(root, item["science"]))
    validate_science(item["model"], science)
    calibration = json.loads(read_bound(root, item["calibration"]))
    manifest = ModelManifest(
        item["model_id"],
        item["artifact"]["sha256"],
        item["science"]["sha256"],
        calibration["image"]["threshold"],
    )
    pixel = calibration["pixel"]["threshold"]
    ModelManifest(
        item["model_id"], manifest.artifact_sha256, manifest.preprocessing_sha256, pixel
    )

    def validate_use():
        # Eligibility is rechecked before each request, not only the first load.
        current = json.loads(read_bound(root, item["application_permission"]))
        if current != approval or datetime.now(UTC) >= expiry:
            raise InspectionError("Application permission changed/expired")
        if json.loads(read_bound(root, item["science"])) != science:
            raise InspectionError("Preprocessing changed")
        if json.loads(read_bound(root, item["calibration"])) != calibration:
            raise InspectionError("Calibration changed")

    def factory():
        validate_use()
        for package, expected in {
            "torch": "2.9.1",
            "torchvision": "0.24.1",
            "anomalib": "2.6.0",
        }.items():
            if version(package).split("+")[0] != expected:
                raise InspectionError(
                    "Native package version differs from approved adapter"
                )
        raw = read_bound(root, item["artifact"], limit=1024 * 1024 * 1024)
        import torch

        # No weights_only=False fallback, custom globals or arbitrary pickle execution.
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        from visionguard.efficientad import canonical_checkpoint_sha256
        from visionguard.visa_evaluator import ModelSpec
        from visionguard.visa_evaluator_backend import native_from_state

        state = payload["model_state"]
        identity = payload.get("identity", {})
        if any(identity.get(k) != item[k] for k in ("model", "category", "seed")):
            raise InspectionError("Checkpoint model/category/seed binding mismatch")
        if canonical_checkpoint_sha256(state) != item["canonical_state_sha256"]:
            raise InspectionError("Canonical native state identity mismatch")
        if payload.get("canonical_model_sha256") != item["canonical_state_sha256"]:
            raise InspectionError("Checkpoint declared canonical identity mismatch")
        spec = ModelSpec(
            item["model"],
            item["category"],
            item["seed"],
            manifest.artifact_sha256,
            manifest.threshold,
            pixel,
        )
        native = native_from_state(spec, state, science, device="cpu")
        return NativeInspectionBackend(manifest, native)

    return Registration(
        manifest,
        factory,
        "native-development",
        "Authorized development-state CPU inference",
        pixel,
        validate_use,
    )


def validate_science(model: str, science: dict) -> None:
    """Refuse metadata that the existing fixed native adapter cannot implement."""
    pre = science.get("preprocessing", {})
    expected = {
        "color_mode": "RGB",
        "resize": [256, 256],
        "resize_interpolation": "bilinear",
        "resize_antialias": True,
        "aspect_ratio": "distort_to_exact_size",
        "center_crop": None,
        "tensor_layout": "CHW",
        "tensor_dtype": "float32",
        "tensor_range": [0.0, 1.0],
        "anomaly_map_output_coordinates": "original_image_height_width",
    }
    if model == "patchcore":
        expected.update(
            {
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "augmentation": "none",
                "anomaly_map_interpolation": "bilinear",
                "anomaly_map_align_corners": False,
            }
        )
        settings = {
            "implementation": "anomalib",
            "implementation_version": "2.6.0",
            "algorithm": "patchcore",
            "backbone": "wide_resnet50_2.racm_in1k",
            "layers": ["layer2", "layer3"],
            "num_neighbors": 9,
        }
    else:
        expected.update(
            {
                "model_internal_normalization_mean": [0.485, 0.456, 0.406],
                "model_internal_normalization_std": [0.229, 0.224, 0.225],
                "inference_augmentation": "none",
                "anomaly_map_restoration_interpolation": "bilinear",
                "anomaly_map_restoration_align_corners": False,
            }
        )
        settings = {
            "implementation": "anomalib",
            "implementation_version": "2.6.0",
            "algorithm": "efficient_ad",
            "variant": "pdn_small",
            "teacher_out_channels": 384,
            "padding": False,
            "pad_maps": True,
        }
    if any(k not in pre or pre[k] != v for k, v in expected.items()) or any(
        science.get("model", {}).get(k) != v for k, v in settings.items()
    ):
        raise InspectionError(
            "Preprocessing/architecture differs from supported frozen adapter"
        )


class NativeInspectionBackend:
    def __init__(self, manifest, native):
        self.manifest, self.native = manifest, native

    def predict(self, image: InspectionInput) -> ModelOutput:
        import torch

        from visionguard.preprocessing import restore_anomaly_map

        with Image.open(io.BytesIO(image.image_bytes)) as decoded:
            rgb = decoded.convert("RGB")
        with torch.inference_mode():
            tensor = self.native.transform(rgb).unsqueeze(0).to("cpu")
            result = self.native.model(tensor)
            score = float(result.pred_score[0])
            restored = restore_anomaly_map(
                result.anomaly_map[0, 0], (image.height, image.width)
            )
            rows = tuple(
                tuple(float(v) for v in row) for row in restored.cpu().tolist()
            )
        return ModelOutput(
            image.image_sha256,
            self.manifest.artifact_sha256,
            self.manifest.preprocessing_sha256,
            score,
            rows,
        )

    def close(self):
        self.native = None
