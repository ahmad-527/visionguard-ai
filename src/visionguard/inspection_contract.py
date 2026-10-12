"""Model-free application boundary; no training, calibration, or artifact loading."""

from __future__ import annotations

import hashlib
import io
import math
import re
import warnings
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
QUALIFICATION = (
    "Held-out VisA evaluation with historical access independence unverified."
)


class InspectionError(ValueError):
    """Invalid image, model identity, or inference response; never a normal result."""


def _sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise InspectionError("Identity must be a lowercase SHA-256 digest")


def _finite(value: float) -> None:
    try:
        valid = type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise InspectionError("Score and threshold must be finite numeric scalars")


@dataclass(frozen=True)
class ModelManifest:
    """Server-approved identities, not user-selected paths or proof of payload validity.

    A future loader must verify the actual artifact and preprocessing bytes before
    registering a model. Thresholds are provided by prior approved calibration;
    this boundary does not compute or adjust them.
    """

    model_id: str
    artifact_sha256: str
    preprocessing_sha256: str
    threshold: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.model_id) is None
        ):
            raise InspectionError("Model ID must be a bounded registry identifier")
        _sha256(self.artifact_sha256)
        _sha256(self.preprocessing_sha256)
        _finite(self.threshold)


@dataclass(frozen=True)
class InspectionInput:
    """Immutable encoded image, digest and original-coordinate dimensions."""

    image_bytes: bytes
    image_sha256: str
    width: int
    height: int


def prepare_image(payload: bytes) -> InspectionInput:
    """Verify bounded, single-frame PNG/JPEG input without writing uploaded data.

    Image format is decoded rather than trusted from a filename/content type.
    Color normalization belongs to the verified model-specific preprocessing.
    """
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_IMAGE_BYTES:
        raise InspectionError("Image must contain 1 to 10 MiB encoded bytes")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                if image.format not in {"PNG", "JPEG"}:
                    raise InspectionError("Only PNG and JPEG are supported")
                if width * height > MAX_IMAGE_PIXELS:
                    raise InspectionError("Image exceeds the pixel limit")
                if getattr(image, "n_frames", 1) != 1:
                    raise InspectionError("Animated or multi-frame images are refused")
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()  # Refuse truncated/corrupt payloads before inference.
                # Reading PNG EXIF can load pixels: do this after the verify pass.
                # Refuse browser auto-rotation/mirroring without altering model pixels.
                if image.getexif().get(274, 1) != 1:
                    raise InspectionError("Non-upright EXIF orientation is unsupported")
    except InspectionError:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:
        raise InspectionError("Image decoding failed") from error
    return InspectionInput(payload, hashlib.sha256(payload).hexdigest(), width, height)


@dataclass(frozen=True)
class ModelOutput:
    """One model's score and immutable map in original image coordinates."""

    image_sha256: str
    artifact_sha256: str
    preprocessing_sha256: str
    score: float
    anomaly_map: tuple[tuple[float, ...], ...]


class InferenceBackend(Protocol):
    """Dependency-injected inference only; no fit/resume/evaluation surface."""

    def predict(self, image: InspectionInput) -> ModelOutput: ...


def validate_output(
    image: InspectionInput, manifest: ModelManifest, output: ModelOutput
) -> str:
    """Validate binding, map and strict threshold; do not fuse model scales.

    Return NORMAL/ANOMALOUS, not a product acceptance or safety guarantee.
    Three-way triage requires separately verified paired model evidence.
    """
    if (
        output.image_sha256 != image.image_sha256
        or output.artifact_sha256 != manifest.artifact_sha256
        or output.preprocessing_sha256 != manifest.preprocessing_sha256
    ):
        raise InspectionError("Inference identities do not match the request/registry")
    _finite(output.score)
    if type(output.anomaly_map) is not tuple or len(output.anomaly_map) != image.height:
        raise InspectionError("Map must have original image height")
    for row in output.anomaly_map:
        if type(row) is not tuple or len(row) != image.width:
            raise InspectionError("Map must have original image width")
        for value in row:
            _finite(value)
    return "ANOMALOUS" if output.score > manifest.threshold else "NORMAL"
