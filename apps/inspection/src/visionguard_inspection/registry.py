"""Server-owned model registry; clients select IDs, never paths or calibration."""

from __future__ import annotations

import hashlib
import io
import json
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from PIL import Image, ImageStat

from visionguard.inspection_contract import (
    InferenceBackend,
    InspectionError,
    InspectionInput,
    ModelManifest,
    ModelOutput,
)
from visionguard.paths import portable_relative_path


@dataclass(frozen=True)
class Registration:
    manifest: ModelManifest
    factory: Callable[[], InferenceBackend]
    mode: str
    description: str
    pixel_threshold: float
    validate_use: Callable[[], None] = lambda: None

    def public(self) -> dict:
        return {
            "model_id": self.manifest.model_id,
            "artifact_sha256": self.manifest.artifact_sha256,
            "preprocessing_sha256": self.manifest.preprocessing_sha256,
            "threshold": self.manifest.threshold,
            "pixel_threshold": self.pixel_threshold,
            "mode": self.mode,
            "description": self.description,
        }


class Registry:
    def __init__(self, registrations: tuple[Registration, ...] = ()):
        items = {entry.manifest.model_id: entry for entry in registrations}
        if len(items) != len(registrations):
            raise InspectionError("Duplicate registry model IDs")
        self.entries = MappingProxyType(items)

    def select(self, model_id: str) -> Registration:
        try:
            return self.entries[model_id]
        except KeyError as error:
            raise InspectionError("Model is not in the server registry") from error


def read_bound(root: Path, record: dict, *, limit: int = 1024 * 1024) -> bytes:
    """Read exact bounded bytes, refusing links/reparse points and traversal.

    The same bytes are hashed and then consumed: no second open for deserialization.
    The artifact root must be operator-owned and immutable for the server session.
    """
    if not root.is_absolute():
        raise InspectionError("Artifact root must be absolute")
    relative = portable_relative_path(record["path"])
    if not relative.parts or any(":" in part for part in relative.parts):
        raise InspectionError("Empty paths and alternate streams are refused")
    path = root / relative
    expected = record["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise InspectionError("Missing file SHA-256")
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise InspectionError("Linked/reparse registry assets are refused")
    if not stat.S_ISREG(path.stat().st_mode):
        raise InspectionError("Registry assets must be regular files")
    if path.stat().st_size > limit:
        raise InspectionError("Registry asset exceeds its size limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
        raise InspectionError("Registry asset identity/size mismatch")
    return raw


def load_registry(path: Path, expected_sha256: str) -> Registry:
    """Load pinned server configuration; native approval precedes artifact access."""
    if not path.is_absolute():
        raise InspectionError("Registry path must be absolute")
    raw = read_bound(path.parent, {"path": path.name, "sha256": expected_sha256})
    document = json.loads(raw)
    if set(document) != {"schema_version", "models"} or document["schema_version"] != 1:
        raise InspectionError("Unsupported registry schema")
    from visionguard_inspection.native import approved_registration

    if not isinstance(document["models"], list) or len(document["models"]) > 72:
        raise InspectionError("Registry requires at most 72 explicit entries")
    return Registry(
        tuple(approved_registration(path.parent, item) for item in document["models"])
    )


class ManufacturedBackend:
    """Deterministic UI/API fixture, explicitly not trained/native model inference."""

    def __init__(self, manifest: ModelManifest):
        self.manifest = manifest

    def predict(self, image: InspectionInput) -> ModelOutput:
        with Image.open(io.BytesIO(image.image_bytes)) as decoded:
            grey = decoded.convert("L")
            score = ImageStat.Stat(grey).mean[0] / 255
            values = list(grey.tobytes())
        rows = tuple(
            tuple(v / 255 for v in values[start : start + image.width])
            for start in range(0, len(values), image.width)
        )
        return ModelOutput(
            image.image_sha256,
            self.manifest.artifact_sha256,
            self.manifest.preprocessing_sha256,
            score,
            rows,
        )


def manufactured_registry() -> Registry:
    manifest = ModelManifest(
        "manufactured-demo",
        hashlib.sha256(
            b"manufactured intensity fixture v1; not a checkpoint"
        ).hexdigest(),
        hashlib.sha256(b"Pillow L intensity / 255; no resize").hexdigest(),
        0.5,
    )
    return Registry(
        (
            Registration(
                manifest,
                lambda: ManufacturedBackend(manifest),
                "manufactured",
                "Manufactured intensity fixture — NOT native ML",
                0.5,
            ),
        )
    )
