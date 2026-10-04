"""Separate genuine-ID native interface; frozen restoration remains unchanged."""

from dataclasses import dataclass

import numpy as np

from visionguard.heldout_metrics import Prediction
from visionguard.visa_evaluator import ModelSpec, finite_scalar, require
from visionguard.visa_evaluator_backend import FrozenBackend, load_development_backend


@dataclass(frozen=True)
class RealFrame:
    sample_id: str
    category: str
    rgb: np.ndarray

    def validate(self, spec: ModelSpec) -> None:
        from pathlib import PurePosixPath

        path = PurePosixPath(self.sample_id)
        require(
            str(path) == self.sample_id
            and len(path.parts) == 5
            and path.parts[:3] == (spec.category, "Data", "Images")
            and path.parts[3] in ("Normal", "Anomaly")
            and ":" not in self.sample_id
            and "\\" not in self.sample_id
            and not set(path.parts) & {".", ".."},
            "Genuine original ID required",
        )
        require(self.category == spec.category, "Wrong frame category")
        require(
            type(self.rgb) is np.ndarray
            and self.rgb.dtype == np.uint8
            and self.rgb.ndim == 3
            and self.rgb.shape[2] == 3
            and min(self.rgb.shape[:2]) > 0,
            "Nonempty uint8 RGB required",
        )


class NativeBackend(FrozenBackend):
    def state_digest(self):
        from visionguard.efficientad import canonical_checkpoint_sha256

        return canonical_checkpoint_sha256(self.model.state_dict())

    def predict(self, frame: RealFrame) -> Prediction:
        frame.validate(self.spec)
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


def load_backend(repository, development_root, spec, device="cuda"):
    frozen = load_development_backend(
        repository, development_root, spec.model, spec.category, spec.seed, device
    )
    require(frozen.spec == spec, "Frozen model identity mismatch")
    return NativeBackend(frozen.spec, frozen.model, frozen.transform, frozen.device)
