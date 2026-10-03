"""Read-only frozen native backends, with synthetic-only in-memory prediction.

No dataset loader, optimizer, calibration or download. Real test execution must
enter the permanently closed public gate, NOT this synthetic engineering API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import (
    ModelSpec,
    Prediction,
    finite_scalar,
    model_specs,
    require,
)
from visionguard.visa_evaluator_artifacts import verify_published
from visionguard.visa_matrix_validation import load, validate_cell


@dataclass(frozen=True)
class SyntheticFrame:
    """Label-free artificial RGB pixels; deliberately no path or truth mask."""

    sample_id: str
    category: str
    rgb: np.ndarray

    def validate(self, spec: ModelSpec) -> None:
        require(
            type(self.sample_id) is str
            and self.sample_id.startswith(f"synthetic:{spec.category}:"),
            "Synthetic frame ID required",
        )
        require(
            "/" not in self.sample_id and "\\" not in self.sample_id, "Not a file path"
        )
        require(self.category == spec.category, "Wrong frame category")
        require(
            type(self.rgb) is np.ndarray
            and self.rgb.dtype == np.uint8
            and self.rgb.ndim == 3
            and self.rgb.shape[2] == 3
            and self.rgb.shape[0] > 0
            and self.rgb.shape[1] > 0,
            "Nonempty uint8 RGB required",
        )


class FrozenBackend:
    """One native model in eval/no-grad, own transform and original restoration."""

    def __init__(
        self, spec: ModelSpec, model: Any, transform: Any, device: str
    ) -> None:
        self.spec, self.model, self.transform, self.device = (
            spec,
            model,
            transform,
            device,
        )
        model.eval()
        model.requires_grad_(False)

    def predict_synthetic(self, frame: SyntheticFrame) -> Prediction:
        frame.validate(self.spec)  # Before image/tensor conversion or backend call.
        import torch
        from PIL import Image

        from visionguard.preprocessing import restore_anomaly_map

        with torch.no_grad():
            tensor = (
                self.transform(Image.fromarray(frame.rgb)).unsqueeze(0).to(self.device)
            )
            output = self.model(tensor)
            score = finite_scalar(float(output.pred_score[0]), "native image score")
            restored = restore_anomaly_map(
                output.anomaly_map[0, 0], frame.rgb.shape[:2]
            )
            array = restored.detach().cpu().numpy()
        require(
            array.dtype == np.float32 and bool(np.isfinite(array).all()),
            "Invalid native restored map",
        )
        require(
            bool((np.abs(array) <= np.finfo(np.float16).max).all()),
            "Float16 export overflow",
        )
        return Prediction(frame.sample_id, self.spec, score, array)


def native_from_state(
    spec: ModelSpec, state: dict, science: dict, device: str = "cpu"
) -> FrozenBackend:
    """Strictly restore final state; constructor initialization is fully replaced.

    PatchCore pre_trained=False prevents network/cache downloads: every backbone
    parameter and dynamic memory bank is loaded from the SHA-verified FINAL state.
    Exact canonical state equality is checked after loading, not assumed.
    """
    import torch
    from torchvision.transforms.v2 import Compose, Normalize, Resize, ToTensor

    from visionguard.efficientad import canonical_checkpoint_sha256

    if spec.model == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        settings, pre = science["model"], science["preprocessing"]
        model = PatchcoreModel(
            backbone=settings["backbone"],
            layers=settings["layers"],
            pre_trained=False,
            num_neighbors=settings["num_neighbors"],
        )
        transform = Compose(
            [
                ToTensor(),
                Resize(tuple(pre["resize"]), antialias=True),
                Normalize(pre["normalization_mean"], pre["normalization_std"]),
            ]
        )
    else:
        from anomalib.models.image.efficient_ad.torch_model import (
            EfficientAdModel,
            EfficientAdModelSize,
        )

        model = EfficientAdModel(
            teacher_out_channels=384,
            model_size=EfficientAdModelSize.S,
            padding=False,
            pad_maps=True,
        )
        transform = Compose([Resize((256, 256), antialias=True), ToTensor()])
    model.load_state_dict(state, strict=True)
    require(
        canonical_checkpoint_sha256(model.state_dict())
        == canonical_checkpoint_sha256(state),
        "Loaded model differs bitwise",
    )
    model = model.to(torch.device(device))
    return FrozenBackend(spec, model, transform, device)


def load_development_backend(
    repository: Path,
    development_root: Path,
    model: str,
    category: str,
    seed: int,
    device: str = "cpu",
) -> FrozenBackend:
    """Only declared development checkpoints; provenance independently verified.

    No caller-provided threshold, hash, checkpoint path, test root or label.
    Native semantic validation precedes deserialization/construction. No inference
    is performed here. B1 tests use synthetic state/fixtures, not real model scores.
    """
    from importlib.metadata import distribution

    from visionguard.efficientad_benchmark import _load_checkpoint
    from visionguard.visa_engineering import environment_identity
    from visionguard.visa_evaluator_protocol import verify_readiness_freeze
    from visionguard.visa_guard import MEMBERSHIP, verified_context

    repository, development_root = repository.resolve(), development_root.resolve()
    readiness = verify_readiness_freeze(repository)
    native_sources = readiness["document"]["native_source_sha256"]
    installed = distribution("anomalib")
    for relative, digest in native_sources.items():
        require(
            sha256_file(installed.locate_file(relative)) == digest,
            "Native implementation source drift",
        )
    freeze = verify_published(repository)
    require(
        environment_identity() == freeze["execution_contract"]["environment"],
        "Frozen backend environment mismatch",
    )
    for relative, digest in freeze["execution_contract"]["source_hashes"].items():
        require(
            sha256_file(repository / relative) == digest, "Frozen backend source drift"
        )
    specs = model_specs(freeze)
    key = f"{model}:{category}:{seed}"
    require(type(seed) is int and key in specs, "No alternate or missing cell")
    matrix_path = development_root / "matrix-manifest.json"
    require(
        sha256_file(matrix_path) == freeze["matrix_manifest_sha256"],
        "Matrix hash mismatch",
    )
    matrix = load(matrix_path)
    membership_path = (
        repository / "reports/phase4c-visa-readiness/development-membership.json"
    )
    require(sha256_file(membership_path) == MEMBERSHIP, "Membership corruption")
    result = validate_cell(
        development_root / model,
        model,
        category,
        seed,
        matrix["identities"][model],
        load(membership_path),
    )
    record = result["files"]["final_model"]
    require(
        record["sha256"] == specs[key].model_sha256
        and result["canonical_model_sha256"]
        == freeze["cells"][key]["canonical_model_sha256"],
        "Wrong final model",
    )
    payload = _load_checkpoint(
        development_root / model / record["path"], record["sha256"]
    )
    context = verified_context(repository, model)
    return native_from_state(
        specs[key], payload["model_state"], context["scientific"], device
    )
