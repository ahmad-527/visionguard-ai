"""Typed VisionGuard adapter around Anomalib's public PatchCore API."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Protocol

from visionguard.experiment import ExperimentConfig


class ModelDependencyError(RuntimeError):
    """Raised when the optional, pinned ML stack is unavailable or incompatible."""


@dataclass(frozen=True)
class PatchCorePrediction:
    """Framework-neutral anomaly output consumed by VisionGuard artifacts."""

    sample_id: str
    anomaly_score: float
    anomaly_map: Any


class ExperimentEngine(Protocol):
    """Small engine surface required by the adapter."""

    def fit(self, *, model: Any, datamodule: Any) -> Any: ...

    def predict(self, *, model: Any, datamodule: Any) -> list[Any] | None: ...


def _import_symbol(module: str, symbol: str) -> Any:
    try:
        return getattr(importlib.import_module(module), symbol)
    except (ImportError, AttributeError) as exc:
        raise ModelDependencyError(
            "PatchCore requires the optional ML dependencies. Install the pinned "
            "hardware-specific stack documented in docs/patchcore-dependencies.md."
        ) from exc


class AnomalibPatchCoreAdapter:
    """Isolate VisionGuard orchestration and artifacts from Anomalib internals."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        try:
            installed = importlib.metadata.version("anomalib")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ModelDependencyError(
                "Anomalib is not installed; the lightweight audit environment "
                "remains usable"
            ) from exc
        if installed != config.model.implementation_version:
            raise ModelDependencyError(
                f"Configured Anomalib {config.model.implementation_version}, "
                f"but found {installed}"
            )
        patchcore = _import_symbol("anomalib.models", "Patchcore")
        preprocessor = patchcore.configure_pre_processor(
            image_size=config.preprocessing.resize,
            center_crop_size=config.preprocessing.center_crop,
        )
        self.model = patchcore(
            backbone=config.model.backbone,
            layers=config.model.layers,
            pre_trained=config.model.pretrained,
            coreset_sampling_ratio=config.model.coreset_sampling_ratio,
            num_neighbors=config.model.num_neighbors,
            pre_processor=preprocessor,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        self.engine: ExperimentEngine | None = None

    def fit(self, datamodule: Any) -> None:
        """Build PatchCore's memory bank from an already boundary-checked module."""

        engine_type = _import_symbol("anomalib.engine", "Engine")
        accelerator = (
            "auto"
            if self.config.device_policy == "auto"
            else ("gpu" if self.config.device_policy == "cuda" else "cpu")
        )
        self.engine = engine_type(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        self.engine.fit(model=self.model, datamodule=datamodule)

    def predict(self, datamodule: Any) -> tuple[PatchCorePrediction, ...]:
        """Return framework-neutral image scores and anomaly maps."""

        if self.engine is None:
            raise RuntimeError("PatchCore must be fit before prediction")
        batches = self.engine.predict(model=self.model, datamodule=datamodule)
        if batches is None:
            return ()
        predictions: list[PatchCorePrediction] = []
        for batch in batches:
            paths = getattr(batch, "image_path", None)
            scores = getattr(batch, "pred_score", None)
            maps = getattr(batch, "anomaly_map", None)
            if paths is None or scores is None or maps is None:
                raise RuntimeError(
                    "Anomalib prediction batch lacks image_path, pred_score, "
                    "or anomaly_map"
                )
            for sample_path, score, anomaly_map in zip(
                paths, scores, maps, strict=True
            ):
                numeric_score = float(score.detach().cpu().item())
                predictions.append(
                    PatchCorePrediction(
                        sample_id=str(sample_path),
                        anomaly_score=numeric_score,
                        anomaly_map=anomaly_map.detach().cpu(),
                    )
                )
        return tuple(predictions)
