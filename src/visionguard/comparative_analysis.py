"""Model-free comparison of frozen PatchCore and EfficientAD evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from visionguard.analysis_metrics import (
    AnalysisMetricError,
    Float16PixelAnalysisAccumulator,
    classification_metrics,
    describe_scores,
    localization_diagnostics,
    score_distributions,
)
from visionguard.artifacts import sha256_file, validate_artifact
from visionguard.calibration import CalibrationError, highest_order_statistic
from visionguard.efficientad_artifacts import validate_efficientad_artifact
from visionguard.efficientad_protocol import (
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
)
from visionguard.paths import portable_relative_path
from visionguard.protocol import (
    OFFICIAL_CATEGORIES,
    PROTOCOL_SEEDS,
    load_protocol,
    protocol_fingerprint,
)

ANALYSIS_SCHEMA_VERSION = 1
PATCHCORE = "patchcore"
EFFICIENTAD = "efficientad"
MODEL_ORDER = (PATCHCORE, EFFICIENTAD)
DISAGREEMENT_ORDER = (
    "both_correct",
    "patchcore_only_correct",
    "efficientad_only_correct",
    "both_wrong",
)
PANEL_SELECTION_SEED = 42
IMAGE_METRIC_PATHS = (
    ("sensitivity", "sensitivity"),
    ("specificity", "specificity"),
    ("precision", "precision"),
    ("image_f1", "image_f1"),
    ("image_auroc", "image_auroc"),
)
PIXEL_METRIC_PATHS = (
    ("pixel_precision", "pixel_precision"),
    ("pixel_sensitivity", "pixel_sensitivity"),
    ("pixel_specificity", "pixel_specificity"),
    ("pixel_f1", "pixel_f1"),
    ("au_pro_0.05", "au_pro_0.05"),
    ("pixel_auroc_diagnostic", "pixel_auroc_diagnostic"),
)
CAPTURE_VARIANT = re.compile(r"_(overexposed|underexposed|regular|shift_[123])$")
LOGGER = logging.getLogger(__name__)


class ComparativeAnalysisError(RuntimeError):
    """Raised when frozen evidence cannot be validated or compared."""


@dataclass(frozen=True)
class EvidenceSpec:
    """Read-only locations and validators for one frozen model evidence set."""

    name: str
    committed_manifest_path: Path
    evidence_root: Path
    protocol_path: Path
    protocol_loader: Callable[[Path], dict[str, Any]]
    fingerprint: Callable[[Mapping[str, Any]], str]
    artifact_validator: Callable[[dict[str, Any]], None]


@dataclass
class EvidenceBundle:
    """Validated manifest, artifacts, and provenance for one model."""

    spec: EvidenceSpec
    committed_manifest: dict[str, Any]
    local_manifest: dict[str, Any]
    artifacts: dict[tuple[str, int], dict[str, Any]]
    artifact_paths: dict[tuple[str, int], Path]
    frozen_summary: dict[str, Any]
    provenance: dict[str, Any]


def _read_json(path: Path, location: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparativeAnalysisError(f"Unable to read {location}: {path}") from exc
    if not isinstance(value, dict):
        raise ComparativeAnalysisError(f"{location} must contain a JSON object")
    return value


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _analysis_environment() -> dict[str, str]:
    packages = ("numpy", "scipy", "tifffile", "Pillow", "PyYAML")
    return {
        "python": platform.python_version(),
        **{name.lower(): importlib.metadata.version(name) for name in packages},
    }


def _analysis_implementation(repository_root: Path) -> dict[str, Any]:
    relative_paths = (
        "src/visionguard/analysis_metrics.py",
        "src/visionguard/artifacts.py",
        "src/visionguard/benchmark_metrics.py",
        "src/visionguard/calibration.py",
        "src/visionguard/comparative_analysis.py",
        "src/visionguard/efficientad_artifacts.py",
        "src/visionguard/efficientad_protocol.py",
        "src/visionguard/metrics.py",
        "src/visionguard/paths.py",
        "src/visionguard/protocol.py",
    )
    safe_directory = f"safe.directory={repository_root.resolve().as_posix()}"
    head = subprocess.run(
        ["git", "-c", safe_directory, "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git",
            "-c",
            safe_directory,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0 or status.returncode != 0:
        raise ComparativeAnalysisError("Unable to identify the analysis implementation")
    return {
        "id": "phase4a-comparative-failure-analysis-v1",
        "git_commit": head.stdout.strip(),
        "git_tracked_worktree_dirty_at_analysis_start": bool(status.stdout.strip()),
        "source_sha256": {
            relative: sha256_file(repository_root / relative)
            for relative in relative_paths
        },
    }


def _safe_join(root: Path, relative: str, location: str) -> Path:
    try:
        portable = portable_relative_path(relative)
    except (TypeError, ValueError) as exc:
        raise ComparativeAnalysisError(f"{location} is not a portable path") from exc
    if not portable.parts or portable == Path("."):
        raise ComparativeAnalysisError(f"{location} must not be empty")
    resolved_root = root.resolve()
    resolved = (resolved_root / portable).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ComparativeAnalysisError(f"{location} escapes its evidence root")
    return resolved


def _public_sample_parts(sample_id: str, category: str) -> tuple[str, str, str, str]:
    if "\\" in sample_id:
        raise ComparativeAnalysisError("Public sample IDs must use POSIX separators")
    path = PurePosixPath(sample_id)
    if path.is_absolute() or path.as_posix() != sample_id or len(path.parts) != 4:
        raise ComparativeAnalysisError(f"Malformed public sample ID: {sample_id}")
    sample_category, split, condition, filename = path.parts
    if (
        sample_category != category
        or split != "test_public"
        or condition not in {"good", "bad"}
        or PurePosixPath(filename).suffix.lower() != ".png"
    ):
        raise ComparativeAnalysisError(
            f"Sample ID is outside the declared public cell: {sample_id}"
        )
    return sample_category, split, condition, filename


def _manifest_differences(
    committed: Any, local: Any, path: tuple[str, ...] = ()
) -> list[str]:
    """Find differences, allowing only explicit committed path redactions."""

    if committed == local:
        return []
    if isinstance(committed, dict) and isinstance(local, dict):
        if set(committed) != set(local):
            raise ComparativeAnalysisError(
                f"Local manifest keys differ at {'/'.join(path) or '<root>'}"
            )
        differences: list[str] = []
        for key in committed:
            differences.extend(
                _manifest_differences(committed[key], local[key], (*path, str(key)))
            )
        return differences
    if isinstance(committed, list) and isinstance(local, list):
        if len(committed) != len(local):
            raise ComparativeAnalysisError(
                f"Local manifest list length differs at {'/'.join(path)}"
            )
        differences = []
        for index, (committed_item, local_item) in enumerate(
            zip(committed, local, strict=True)
        ):
            differences.extend(
                _manifest_differences(committed_item, local_item, (*path, str(index)))
            )
        return differences
    if isinstance(committed, str) and isinstance(local, str):
        marker = "<local-dataset-root>"
        if marker in committed:
            prefix, suffix = committed.split(marker, maxsplit=1)
            if local.startswith(prefix) and local.endswith(suffix):
                middle = (
                    local[len(prefix) : len(local) - len(suffix)]
                    if suffix
                    else local[len(prefix) :]
                )
                if PureWindowsPath(middle).anchor:
                    return ["/".join(path)]
    raise ComparativeAnalysisError(
        f"Local evidence differs from committed manifest at {'/'.join(path)}"
    )


def _git_commit_status(repository_root: Path, commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ComparativeAnalysisError("Benchmark implementation commit is invalid")
    safe_directory = f"safe.directory={repository_root.resolve().as_posix()}"
    verify = subprocess.run(
        ["git", "-c", safe_directory, "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        raise ComparativeAnalysisError(
            f"Benchmark implementation commit is unavailable: {commit}"
        )
    ancestor = subprocess.run(
        ["git", "-c", safe_directory, "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode not in {0, 1}:
        raise ComparativeAnalysisError("Unable to inspect implementation reachability")
    return {
        "commit": commit,
        "git_object_verified": True,
        "ancestor_of_analysis_head": ancestor.returncode == 0,
        "note": (
            "A false ancestor value is expected when benchmark work was squash-merged; "
            "artifact and protocol bindings remain mandatory."
        ),
    }


def _summary_validation(
    committed_manifest_path: Path,
    evidence_root: Path,
    local_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_relative = local_manifest.get("summary_path")
    summary_sha = local_manifest.get("summary_sha256")
    if not isinstance(summary_relative, str) or not isinstance(summary_sha, str):
        raise ComparativeAnalysisError("Completed manifest lacks summary identity")
    local_summary_path = _safe_join(
        evidence_root, summary_relative, "manifest summary path"
    )
    if (
        not local_summary_path.is_file()
        or sha256_file(local_summary_path) != summary_sha
    ):
        raise ComparativeAnalysisError("Local benchmark summary hash is invalid")
    committed_summary_path = committed_manifest_path.with_name(local_summary_path.name)
    committed_summary = _read_json(committed_summary_path, "committed summary")
    local_summary = _read_json(local_summary_path, "local summary")
    if committed_summary != local_summary:
        raise ComparativeAnalysisError(
            "Local summary is not semantically identical to the committed summary"
        )
    return (
        {
            "local_summary_sha256": summary_sha,
            "committed_summary_sha256": sha256_file(committed_summary_path),
            "semantic_match": True,
        },
        local_summary,
    )


def _validate_protocol(
    spec: EvidenceSpec, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    document = spec.protocol_loader(spec.protocol_path)
    fingerprint = spec.fingerprint(document)
    if fingerprint != manifest.get("protocol_fingerprint"):
        raise ComparativeAnalysisError(
            f"{spec.name} protocol fingerprint differs from its manifest"
        )
    if document.get("protocol", {}).get("id") != manifest.get("protocol_id"):
        raise ComparativeAnalysisError(
            f"{spec.name} protocol ID differs from its manifest"
        )
    return document


def _validate_efficientad_provenance(
    artifact: Mapping[str, Any], manifest: Mapping[str, Any], evidence_root: Path
) -> None:
    weights = artifact.get("weights")
    if not isinstance(weights, list) or not weights:
        raise ComparativeAnalysisError("EfficientAD artifact lacks weight provenance")
    identity_checks = (
        (artifact.get("environment_sha256"), manifest.get("environment_sha256")),
        (
            _canonical_mapping_sha256(artifact.get("environment", {})),
            manifest.get("environment_sha256"),
        ),
        (
            artifact.get("dataset", {}).get("sha256"),
            manifest.get("dataset_audit_sha256"),
        ),
        (
            artifact.get("auxiliary_data", {}).get("archive_sha256"),
            manifest.get("imagenette_archive_sha256"),
        ),
        (weights[0].get("sha256"), manifest.get("teacher_weight_sha256")),
        (weights[0].get("archive_sha256"), manifest.get("teacher_archive_sha256")),
    )
    if any(actual != expected for actual, expected in identity_checks):
        raise ComparativeAnalysisError("EfficientAD artifact provenance has drifted")
    checkpoint_relative = artifact.get("model_state", {}).get("checkpoint_path")
    checkpoint_sha = artifact.get("model_state", {}).get("checkpoint_sha256")
    if not isinstance(checkpoint_relative, str) or not isinstance(checkpoint_sha, str):
        raise ComparativeAnalysisError("EfficientAD checkpoint identity is missing")
    checkpoint_path = _safe_join(
        evidence_root, checkpoint_relative, "EfficientAD checkpoint path"
    )
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha:
        raise ComparativeAnalysisError("EfficientAD checkpoint hash is invalid")


def _validate_patchcore_provenance(
    artifact: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if artifact.get("dataset", {}).get("sha256") != manifest.get(
        "dataset_audit_sha256"
    ) or artifact.get("weight", {}).get("sha256") != manifest.get("weight_sha256"):
        raise ComparativeAnalysisError("PatchCore artifact provenance has drifted")


def _validate_prediction(
    prediction: Mapping[str, Any], category: str, image_threshold: float
) -> str:
    sample_id = prediction.get("sample_id")
    if not isinstance(sample_id, str):
        raise ComparativeAnalysisError("Prediction sample ID must be a string")
    _, _, condition, _ = _public_sample_parts(sample_id, category)
    expected_label = int(condition == "bad")
    if prediction.get("label") != expected_label:
        raise ComparativeAnalysisError(f"Public label differs from path: {sample_id}")
    score = prediction.get("anomaly_score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ComparativeAnalysisError(f"Prediction score is invalid: {sample_id}")
    expected_decision = int(float(score) > image_threshold)
    if prediction.get("image_prediction") != expected_decision:
        raise ComparativeAnalysisError(
            f"Stored image decision does not use the frozen threshold: {sample_id}"
        )
    return sample_id


def _validate_calibration_contract(
    artifact: Mapping[str, Any], category: str
) -> list[str]:
    calibration = artifact.get("calibration")
    if not isinstance(calibration, dict):
        raise ComparativeAnalysisError("Calibration evidence must be a mapping")
    inputs = calibration.get("inputs")
    if not isinstance(inputs, list):
        raise ComparativeAnalysisError("Calibration inputs must be a list")
    sample_ids: list[str] = []
    image_scores: list[float] = []
    pixel_maxima: list[float] = []
    for item in inputs:
        if not isinstance(item, dict):
            raise ComparativeAnalysisError("Calibration input must be a mapping")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or "\\" in sample_id:
            raise ComparativeAnalysisError("Calibration sample ID is invalid")
        path = PurePosixPath(sample_id)
        if (
            path.is_absolute()
            or path.as_posix() != sample_id
            or len(path.parts) != 4
            or path.parts[:3] != (category, "validation", "good")
            or path.suffix.lower() != ".png"
        ):
            raise ComparativeAnalysisError(
                f"Calibration input is not validation-normal: {sample_id}"
            )
        image_score = item.get("image_anomaly_score")
        pixel_maximum = item.get("pixel_maximum")
        if (
            not isinstance(image_score, (int, float))
            or not isinstance(pixel_maximum, (int, float))
            or not math.isfinite(float(image_score))
            or not math.isfinite(float(pixel_maximum))
        ):
            raise ComparativeAnalysisError("Calibration scores must be finite")
        restored_hash = item.get("restored_map_sha256")
        restored_shape = item.get("restored_map_shape")
        if (
            not isinstance(restored_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", restored_hash)
            or not isinstance(restored_shape, list)
            or len(restored_shape) != 2
            or any(not isinstance(value, int) or value <= 0 for value in restored_shape)
        ):
            raise ComparativeAnalysisError(
                "Calibration restored-map identity is invalid"
            )
        sample_ids.append(sample_id)
        image_scores.append(float(image_score))
        pixel_maxima.append(float(pixel_maximum))
    if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ComparativeAnalysisError(
            "Calibration sample IDs must be unique lexical order"
        )
    try:
        expected_image = asdict(
            highest_order_statistic(image_scores, minimum_samples=19)
        )
        expected_pixel = asdict(
            highest_order_statistic(pixel_maxima, minimum_samples=19)
        )
    except CalibrationError as exc:
        raise ComparativeAnalysisError(
            "Calibration sample count is insufficient"
        ) from exc
    if (
        calibration.get("image") != expected_image
        or calibration.get("pixel") != expected_pixel
    ):
        raise ComparativeAnalysisError(
            "Stored thresholds do not recompute from validation-normal inputs"
        )
    return sample_ids


def _load_evidence(spec: EvidenceSpec, repository_root: Path) -> EvidenceBundle:
    committed_manifest = _read_json(
        spec.committed_manifest_path, f"committed {spec.name} manifest"
    )
    local_manifest_path = spec.evidence_root / "benchmark-manifest.json"
    local_manifest = _read_json(local_manifest_path, f"local {spec.name} manifest")
    redactions = _manifest_differences(committed_manifest, local_manifest)
    if committed_manifest.get("status") != "completed":
        raise ComparativeAnalysisError(f"{spec.name} manifest is not completed")
    protocol_document = _validate_protocol(spec, committed_manifest)
    matrix = committed_manifest.get("matrix")
    if not isinstance(matrix, dict):
        raise ComparativeAnalysisError(f"{spec.name} manifest matrix is missing")
    if (
        tuple(matrix.get("categories", ())) != OFFICIAL_CATEGORIES
        or tuple(matrix.get("seeds", ())) != PROTOCOL_SEEDS
        or matrix.get("expected_run_count")
        != len(OFFICIAL_CATEGORIES) * len(PROTOCOL_SEEDS)
    ):
        raise ComparativeAnalysisError(f"{spec.name} category/seed matrix has drifted")
    cells = committed_manifest.get("cells")
    if not isinstance(cells, dict):
        raise ComparativeAnalysisError(f"{spec.name} manifest cells are missing")
    expected_keys = {
        f"{category}:{seed}"
        for category in OFFICIAL_CATEGORIES
        for seed in PROTOCOL_SEEDS
    }
    if set(cells) != expected_keys:
        raise ComparativeAnalysisError(f"{spec.name} manifest cell set has drifted")

    benchmark_commit = committed_manifest.get("benchmark_git_commit")
    if not isinstance(benchmark_commit, str):
        raise ComparativeAnalysisError(f"{spec.name} implementation SHA is missing")
    git_status = _git_commit_status(repository_root, benchmark_commit)
    artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_paths: dict[tuple[str, int], Path] = {}
    prediction_count = 0
    checkpoint_count = 0
    for category in OFFICIAL_CATEGORIES:
        for seed in PROTOCOL_SEEDS:
            entry = cells[f"{category}:{seed}"]
            if (
                not isinstance(entry, dict)
                or entry.get("status") != "completed"
                or entry.get("category") != category
                or entry.get("seed") != seed
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} cell is incomplete or misidentified: "
                    f"{category}:{seed}"
                )
            artifact_relative = entry.get("artifact_path")
            artifact_sha = entry.get("artifact_sha256")
            if not isinstance(artifact_relative, str) or not isinstance(
                artifact_sha, str
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} cell lacks artifact identity: {category}:{seed}"
                )
            artifact_path = _safe_join(
                spec.evidence_root, artifact_relative, f"{spec.name} artifact path"
            )
            if (
                not artifact_path.is_file()
                or sha256_file(artifact_path) != artifact_sha
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} artifact hash is invalid: {category}:{seed}"
                )
            artifact = _read_json(artifact_path, f"{spec.name} artifact")
            try:
                spec.artifact_validator(artifact)
            except ValueError as exc:
                raise ComparativeAnalysisError(
                    f"{spec.name} artifact schema is invalid: {category}:{seed}"
                ) from exc
            if (
                artifact.get("status") != "completed"
                or artifact.get("category") != category
                or artifact.get("seed") != seed
                or artifact.get("protocol_id") != committed_manifest.get("protocol_id")
                or artifact.get("protocol_fingerprint")
                != committed_manifest.get("protocol_fingerprint")
                or artifact.get("protocol_snapshot")
                != protocol_document.get("protocol")
                or artifact.get("git", {}).get("commit") != benchmark_commit
                or artifact.get("git", {}).get("dirty") is not False
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} artifact identity is invalid: {category}:{seed}"
                )
            if spec.name == EFFICIENTAD:
                _validate_efficientad_provenance(
                    artifact, committed_manifest, spec.evidence_root
                )
                if artifact.get("model_state", {}).get(
                    "checkpoint_sha256"
                ) != entry.get("checkpoint_sha256"):
                    raise ComparativeAnalysisError(
                        "EfficientAD checkpoint differs from manifest: "
                        f"{category}:{seed}"
                    )
                checkpoint_count += 1
            else:
                _validate_patchcore_provenance(artifact, committed_manifest)
            thresholds = artifact.get("thresholds")
            if not isinstance(thresholds, dict) or any(
                not isinstance(thresholds.get(name), (int, float))
                or not math.isfinite(float(thresholds[name]))
                for name in ("image", "pixel")
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} frozen thresholds are invalid: {category}:{seed}"
                )
            calibration = artifact.get("calibration")
            if (
                not isinstance(calibration, dict)
                or calibration.get("normal_only") is not True
                or calibration.get("split") != "validation"
                or calibration.get("comparison")
                != "score_strictly_greater_than_threshold"
                or calibration.get("image", {}).get("threshold") != thresholds["image"]
                or calibration.get("pixel", {}).get("threshold") != thresholds["pixel"]
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} calibration provenance is invalid: {category}:{seed}"
                )
            _validate_calibration_contract(artifact, category)
            predictions = artifact.get("predictions")
            if not isinstance(predictions, list) or not predictions:
                raise ComparativeAnalysisError(
                    f"{spec.name} predictions are missing: {category}:{seed}"
                )
            sample_ids = [
                _validate_prediction(prediction, category, float(thresholds["image"]))
                for prediction in predictions
            ]
            if sample_ids != sorted(sample_ids) or len(sample_ids) != len(
                set(sample_ids)
            ):
                raise ComparativeAnalysisError(
                    f"{spec.name} prediction order is not unique lexical order: "
                    f"{category}:{seed}"
                )
            prediction_count += len(predictions)
            artifacts[(category, seed)] = artifact
            artifact_paths[(category, seed)] = artifact_path

    summary_status, frozen_summary = _summary_validation(
        spec.committed_manifest_path, spec.evidence_root, local_manifest
    )
    return EvidenceBundle(
        spec=spec,
        committed_manifest=committed_manifest,
        local_manifest=local_manifest,
        artifacts=artifacts,
        artifact_paths=artifact_paths,
        frozen_summary=frozen_summary,
        provenance={
            "committed_manifest_sha256": sha256_file(spec.committed_manifest_path),
            "local_manifest_sha256": sha256_file(local_manifest_path),
            "allowed_manifest_redactions": redactions,
            "protocol_id": committed_manifest["protocol_id"],
            "protocol_fingerprint": committed_manifest["protocol_fingerprint"],
            "implementation": git_status,
            "dataset_audit_sha256": committed_manifest["dataset_audit_sha256"],
            "artifact_count": len(artifacts),
            "checkpoint_hashes_verified": checkpoint_count,
            "prediction_count": prediction_count,
            "summary": summary_status,
        },
    )


def _load_and_verify_public_audit(
    audit_report_path: Path,
    dataset_root: Path,
    expected_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, ...]], dict[str, Any]]:
    if sha256_file(audit_report_path) != expected_sha256:
        raise ComparativeAnalysisError("Dataset audit hash differs from the manifests")
    audit = _read_json(audit_report_path, "dataset audit")
    summary = audit.get("summary")
    if (
        audit.get("schema_version") != 2
        or not isinstance(summary, dict)
        or summary.get("status") != "passed"
        or summary.get("error_count") != 0
        or summary.get("warning_count") != 0
    ):
        raise ComparativeAnalysisError("Dataset audit is not a clean schema-v2 pass")
    files = audit.get("files")
    if not isinstance(files, list):
        raise ComparativeAnalysisError("Dataset audit lacks a file inventory")
    public_index: dict[str, dict[str, Any]] = {}
    validation_ids: defaultdict[str, list[str]] = defaultdict(list)
    for entry in files:
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("split") == "validation"
            and entry.get("kind") == "image"
            and entry.get("condition") == "good"
            and isinstance(entry.get("path"), str)
        ):
            validation_ids[str(entry.get("category"))].append(entry["path"])
        if entry.get("split") != "test_public":
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ComparativeAnalysisError("Public audit entry has an invalid path")
        if relative in public_index:
            raise ComparativeAnalysisError(
                f"Dataset audit repeats public path: {relative}"
            )
        public_index[relative] = entry
    image_count = sum(entry.get("kind") == "image" for entry in public_index.values())
    mask_count = sum(entry.get("kind") == "mask" for entry in public_index.values())
    if image_count != summary.get("counts_by_split", {}).get("test_public"):
        raise ComparativeAnalysisError("Public audit image count is inconsistent")
    if mask_count != summary.get("mask_count"):
        raise ComparativeAnalysisError("Public audit mask count is inconsistent")

    verified = Counter()
    for relative, entry in sorted(public_index.items()):
        path = _safe_join(dataset_root, relative, "audited public asset")
        expected_hash = entry.get("sha256")
        if (
            not path.is_file()
            or not isinstance(expected_hash, str)
            or sha256_file(path) != expected_hash
        ):
            raise ComparativeAnalysisError(
                f"Audited public asset hash is invalid: {relative}"
            )
        verified[str(entry.get("kind"))] += 1
    frozen_validation_ids = {
        category: tuple(sorted(validation_ids[category]))
        for category in OFFICIAL_CATEGORIES
    }
    if any(not values for values in frozen_validation_ids.values()):
        raise ComparativeAnalysisError(
            "Dataset audit lacks a validation-normal inventory"
        )
    return (
        public_index,
        frozen_validation_ids,
        {
            "sha256": expected_sha256,
            "status": "passed",
            "schema_version": 2,
            "public_images_verified": verified["image"],
            "public_masks_verified": verified["mask"],
            "private_assets_opened": 0,
            "summary": {
                "image_count": summary.get("image_count"),
                "mask_count": summary.get("mask_count"),
                "error_count": summary.get("error_count"),
                "warning_count": summary.get("warning_count"),
                "unexpected_overlap_group_count": summary.get(
                    "unexpected_overlap_group_count"
                ),
            },
        },
    )


def _expected_mask_id(sample_id: str) -> str | None:
    path = PurePosixPath(sample_id)
    if path.parts[2] == "good":
        return None
    return PurePosixPath(
        path.parts[0],
        path.parts[1],
        "ground_truth",
        "bad",
        f"{path.stem}_mask{path.suffix}",
    ).as_posix()


def _load_public_mask(
    sample_id: str,
    shape: tuple[int, int],
    dataset_root: Path,
    audit_index: Mapping[str, Mapping[str, Any]],
) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - analysis environment contract
        raise ComparativeAnalysisError(
            "Public-mask analysis requires NumPy and Pillow"
        ) from exc
    mask_id = _expected_mask_id(sample_id)
    if mask_id is None:
        return np.zeros(shape, dtype=np.uint8)
    audit_entry = audit_index.get(mask_id)
    if not isinstance(audit_entry, Mapping) or audit_entry.get("kind") != "mask":
        raise ComparativeAnalysisError(f"Public mask is absent from audit: {mask_id}")
    mask_path = _safe_join(dataset_root, mask_id, "public mask")
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if (
        mask.ndim != 2
        or tuple(mask.shape) != shape
        or not bool(np.isin(mask, (0, 255)).all())
    ):
        raise ComparativeAnalysisError(f"Public mask is malformed: {mask_id}")
    return (mask > 0).astype(np.uint8, copy=False)


def _read_continuous_map(
    run_dir: Path,
    prediction: Mapping[str, Any],
    verification_counts: Counter[str],
) -> tuple[Any, Path]:
    try:
        import numpy as np
        import tifffile
    except ImportError as exc:  # pragma: no cover - analysis environment contract
        raise ComparativeAnalysisError(
            "Continuous-map analysis requires NumPy and tifffile"
        ) from exc
    identity = prediction.get("anomaly_map")
    if not isinstance(identity, Mapping):
        raise ComparativeAnalysisError("Prediction lacks anomaly-map identity")
    relative = identity.get("path")
    expected_hash = identity.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ComparativeAnalysisError("Continuous-map identity is malformed")
    path = _safe_join(run_dir, relative, "continuous anomaly-map path")
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ComparativeAnalysisError(
            f"Continuous anomaly-map hash is invalid: {prediction.get('sample_id')}"
        )
    verification_counts["continuous_maps"] += 1
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise ComparativeAnalysisError(
                f"Continuous map is not single-page: {prediction.get('sample_id')}"
            )
        page = document.pages[0]
        photometric = getattr(page.photometric, "name", str(page.photometric))
        if photometric != "MINISBLACK":
            raise ComparativeAnalysisError(
                f"Continuous map photometric is invalid: {prediction.get('sample_id')}"
            )
        array = page.asarray()
    expected_shape = identity.get("shape")
    if (
        array.ndim != 2
        or array.dtype != np.float16
        or list(array.shape) != expected_shape
        or identity.get("dtype") != "float16"
        or identity.get("coordinate_space") != "original_image_height_width"
        or identity.get("finite") is not True
        or not bool(np.isfinite(array).all())
    ):
        raise ComparativeAnalysisError(
            f"Continuous anomaly-map schema is invalid: {prediction.get('sample_id')}"
        )
    return array, path


def _read_thresholded_map(
    run_dir: Path,
    prediction: Mapping[str, Any],
    expected_shape: tuple[int, int],
    verification_counts: Counter[str],
) -> tuple[Any, Path]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - analysis environment contract
        raise ComparativeAnalysisError(
            "Thresholded-map analysis requires NumPy and Pillow"
        ) from exc
    identity = prediction["anomaly_map"]
    relative = identity.get("thresholded_path")
    expected_hash = identity.get("thresholded_sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ComparativeAnalysisError("Thresholded-map identity is malformed")
    path = _safe_join(run_dir, relative, "thresholded anomaly-map path")
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ComparativeAnalysisError(
            f"Thresholded anomaly-map hash is invalid: {prediction.get('sample_id')}"
        )
    verification_counts["thresholded_maps"] += 1
    with Image.open(path) as image:
        if image.mode != "L":
            raise ComparativeAnalysisError(
                f"Thresholded map is not grayscale: {prediction.get('sample_id')}"
            )
        array = np.asarray(image)
    if (
        array.ndim != 2
        or tuple(array.shape) != expected_shape
        or not bool(np.isin(array, (0, 255)).all())
    ):
        raise ComparativeAnalysisError(
            f"Thresholded anomaly-map schema is invalid: {prediction.get('sample_id')}"
        )
    return (array > 0), path


def disagreement_bucket(
    label: int, patchcore_prediction: int, efficientad_prediction: int
) -> str:
    """Return one of the four exhaustive image-decision disagreement classes."""

    patchcore_correct = patchcore_prediction == label
    efficientad_correct = efficientad_prediction == label
    if patchcore_correct and efficientad_correct:
        return "both_correct"
    if patchcore_correct:
        return "patchcore_only_correct"
    if efficientad_correct:
        return "efficientad_only_correct"
    return "both_wrong"


def pair_prediction_records(
    patchcore: Sequence[Mapping[str, Any]],
    efficientad: Sequence[Mapping[str, Any]],
    *,
    category: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Join exact ordered public predictions without heuristic matching."""

    if len(patchcore) != len(efficientad):
        raise ComparativeAnalysisError(
            f"Prediction counts differ for {category}:{seed}"
        )
    records: list[dict[str, Any]] = []
    for ordinal, (patch, efficient) in enumerate(
        zip(patchcore, efficientad, strict=True)
    ):
        patch_identity = (patch.get("sample_id"), patch.get("label"))
        efficient_identity = (efficient.get("sample_id"), efficient.get("label"))
        if patch_identity != efficient_identity:
            raise ComparativeAnalysisError(
                f"Prediction identity/order differs for {category}:{seed}:{ordinal}"
            )
        label = int(patch["label"])
        patch_prediction = int(patch["image_prediction"])
        efficient_prediction = int(efficient["image_prediction"])
        records.append(
            {
                "category": category,
                "seed": seed,
                "ordinal": ordinal,
                "sample_id": str(patch["sample_id"]),
                "label": label,
                "disagreement": disagreement_bucket(
                    label, patch_prediction, efficient_prediction
                ),
                PATCHCORE: {
                    "anomaly_score": float(patch["anomaly_score"]),
                    "image_prediction": patch_prediction,
                    "correct": patch_prediction == label,
                },
                EFFICIENTAD: {
                    "anomaly_score": float(efficient["anomaly_score"]),
                    "image_prediction": efficient_prediction,
                    "correct": efficient_prediction == label,
                },
            }
        )
    return records


def _selection_hash(record: Mapping[str, Any]) -> str:
    value = (
        f"{record['category']}\0{record['seed']}\0"
        f"{record['label']}\0{record['sample_id']}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_panel_examples(
    records: Sequence[Mapping[str, Any]], *, seed: int = PANEL_SELECTION_SEED
) -> list[dict[str, Any]]:
    """Select panels without using scores, map content, or visual inspection."""

    candidates = [record for record in records if record.get("seed") == seed]
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}

    def add_candidate(candidate: Mapping[str, Any], reason: str) -> None:
        key = (
            str(candidate["category"]),
            int(candidate["seed"]),
            str(candidate["sample_id"]),
        )
        if key not in selected:
            selected[key] = {
                "category": key[0],
                "seed": key[1],
                "sample_id": key[2],
                "label": int(candidate["label"]),
                "disagreement": str(candidate["disagreement"]),
                "selection_sha256": _selection_hash(candidate),
                "reasons": [],
            }
        selected[key]["reasons"].append(reason)

    for category in OFFICIAL_CATEGORIES:
        for label in (0, 1):
            stratum = [
                record
                for record in candidates
                if record.get("category") == category and record.get("label") == label
            ]
            if not stratum:
                raise ComparativeAnalysisError(
                    f"Panel selection stratum is empty: {category}:label-{label}"
                )
            add_candidate(
                min(
                    stratum,
                    key=lambda record: (_selection_hash(record), record["sample_id"]),
                ),
                f"minimum_sha256_for_category={category},label={label},seed={seed}",
            )
    for disagreement in DISAGREEMENT_ORDER:
        stratum = [
            record
            for record in candidates
            if record.get("disagreement") == disagreement
        ]
        if stratum:
            add_candidate(
                min(
                    stratum,
                    key=lambda record: (_selection_hash(record), record["sample_id"]),
                ),
                f"minimum_sha256_for_disagreement={disagreement},seed={seed}",
            )
    order = {category: index for index, category in enumerate(OFFICIAL_CATEGORIES)}
    return sorted(
        selected.values(),
        key=lambda item: (order[item["category"]], item["label"], item["sample_id"]),
    )


def _calibration_diagnostics(artifact: Mapping[str, Any]) -> dict[str, Any]:
    calibration = artifact["calibration"]
    inputs = calibration.get("inputs")
    if not isinstance(inputs, list) or len(inputs) < 2:
        raise ComparativeAnalysisError("Calibration inputs are incomplete")
    pixel_values: list[tuple[float, str]] = []
    image_values: list[tuple[float, str]] = []
    for item in inputs:
        if not isinstance(item, dict):
            raise ComparativeAnalysisError("Calibration input must be a mapping")
        sample_id = item.get("sample_id")
        pixel_maximum = item.get("pixel_maximum")
        image_score = item.get("image_anomaly_score")
        if (
            not isinstance(sample_id, str)
            or not isinstance(pixel_maximum, (int, float))
            or not isinstance(image_score, (int, float))
            or not math.isfinite(float(pixel_maximum))
            or not math.isfinite(float(image_score))
        ):
            raise ComparativeAnalysisError("Calibration input values are invalid")
        pixel_values.append((float(pixel_maximum), sample_id))
        image_values.append((float(image_score), sample_id))
    ordered_pixel = sorted(pixel_values, key=lambda item: (item[0], item[1]))
    ordered_image = sorted(image_values, key=lambda item: (item[0], item[1]))

    def diagnostic(values: list[tuple[float, str]], threshold: float) -> dict[str, Any]:
        raw = [value for value, _ in values]
        maximum, maximum_sample_id = values[-1]
        second_maximum = values[-2][0]
        median = describe_scores(raw)["median"]
        ratio_to_second = (
            maximum / second_maximum if maximum > 0.0 and second_maximum > 0.0 else None
        )
        ratio_to_median = maximum / median if maximum > 0.0 and median > 0.0 else None
        return {
            "threshold": threshold,
            "threshold_equals_observed_maximum": threshold == maximum,
            "maximum_sample_id": maximum_sample_id,
            "maximum": maximum,
            "second_maximum": second_maximum,
            "maximum_minus_second_maximum": maximum - second_maximum,
            "maximum_to_second_maximum_ratio": ratio_to_second,
            "maximum_to_median_ratio": ratio_to_median,
            "distribution": describe_scores(raw),
        }

    result = {
        "normal_only": True,
        "split": "validation",
        "comparison": "score_strictly_greater_than_threshold",
        "image": diagnostic(ordered_image, float(artifact["thresholds"]["image"])),
        "pixel": diagnostic(ordered_pixel, float(artifact["thresholds"]["pixel"])),
    }
    quantiles = calibration.get("native_map_normalization_quantiles")
    if isinstance(quantiles, dict):
        required = ("qa_st", "qb_st", "qa_ae", "qb_ae")
        if any(not isinstance(quantiles.get(key), (int, float)) for key in required):
            raise ComparativeAnalysisError(
                "EfficientAD normalization quantiles are incomplete"
            )
        student_teacher_span = float(quantiles["qb_st"]) - float(quantiles["qa_st"])
        autoencoder_span = float(quantiles["qb_ae"]) - float(quantiles["qa_ae"])
        if student_teacher_span <= 0.0 or autoencoder_span <= 0.0:
            raise ComparativeAnalysisError(
                "EfficientAD normalization quantile spans must be positive"
            )
        result["native_map_normalization"] = {
            **{key: float(quantiles[key]) for key in required},
            "student_teacher_span": student_teacher_span,
            "student_teacher_scale_over_span": 0.1 / student_teacher_span,
            "autoencoder_span": autoencoder_span,
            "autoencoder_scale_over_span": 0.1 / autoencoder_span,
            "causal_limit": (
                "Only the combined normalized map is stored; branch-level cause "
                "cannot be identified from frozen evidence."
            ),
        }
    return result


def _capture_variant(sample_id: str) -> str:
    match = CAPTURE_VARIANT.search(PurePosixPath(sample_id).stem)
    return match.group(1) if match else "unclassified"


def _metric_value(payload: Mapping[str, Any], location: str) -> float:
    if payload.get("status") != "defined" or not isinstance(
        payload.get("value"), (int, float)
    ):
        raise ComparativeAnalysisError(f"Analysis metric is undefined: {location}")
    value = float(payload["value"])
    if not math.isfinite(value):
        raise ComparativeAnalysisError(f"Analysis metric is non-finite: {location}")
    return value


def _assert_recorded_metric(
    artifact: Mapping[str, Any], name: str, measured: Mapping[str, Any]
) -> None:
    recorded = artifact.get("category_metrics", {}).get(name)
    if not isinstance(recorded, dict):
        raise ComparativeAnalysisError(f"Artifact lacks recorded metric: {name}")
    measured_value = _metric_value(measured, f"measured {name}")
    recorded_value = _metric_value(recorded, f"recorded {name}")
    if not math.isclose(measured_value, recorded_value, rel_tol=0.0, abs_tol=1e-12):
        raise ComparativeAnalysisError(
            f"Recomputed {name} differs from the frozen artifact"
        )


def _seed_summary(
    payloads: Sequence[Mapping[str, Any]], *, seeds: Sequence[int]
) -> dict[str, Any]:
    if len(payloads) != len(seeds) or len(payloads) != len(PROTOCOL_SEEDS):
        raise ComparativeAnalysisError("Seed aggregation requires all frozen seeds")
    undefined = [
        seed
        for seed, payload in zip(seeds, payloads, strict=True)
        if payload.get("status") != "defined"
    ]
    if undefined:
        return {
            "status": "undefined",
            "mean": None,
            "sample_standard_deviation": None,
            "count": 0,
            "undefined_seeds": undefined,
            "reason": "At least one frozen seed metric is undefined",
        }
    values = [_metric_value(payload, "seed aggregation") for payload in payloads]
    return {
        "status": "defined",
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "count": len(values),
        "undefined_seeds": [],
        "reason": None,
    }


def _aggregate_model_results(
    cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    per_category: dict[str, Any] = {}
    for category in OFFICIAL_CATEGORIES:
        category_cells = cells.get(category)
        if not isinstance(category_cells, Mapping):
            raise ComparativeAnalysisError(f"Missing category analysis: {category}")
        metrics: dict[str, Any] = {}
        for output_name, source_name in (*IMAGE_METRIC_PATHS, *PIXEL_METRIC_PATHS):
            payloads: list[Mapping[str, Any]] = []
            per_seed: dict[str, Any] = {}
            for seed in PROTOCOL_SEEDS:
                cell = category_cells.get(str(seed))
                if not isinstance(cell, Mapping):
                    raise ComparativeAnalysisError(
                        f"Missing seed analysis: {category}:{seed}"
                    )
                section_name = (
                    "image_metrics"
                    if (output_name, source_name) in IMAGE_METRIC_PATHS
                    else "pixel_metrics"
                )
                section = cell[section_name]
                payload = section[source_name]
                payloads.append(payload)
                per_seed[str(seed)] = {
                    "status": payload.get("status"),
                    "value": payload.get("value"),
                    "reason": payload.get("reason"),
                }
            metrics[output_name] = {
                "per_seed": per_seed,
                "across_seeds": _seed_summary(payloads, seeds=PROTOCOL_SEEDS),
            }
        per_category[category] = metrics
    overall: dict[str, Any] = {}
    for name, _ in (*IMAGE_METRIC_PATHS, *PIXEL_METRIC_PATHS):
        undefined_categories = [
            category
            for category in OFFICIAL_CATEGORIES
            if per_category[category][name]["across_seeds"]["status"] != "defined"
        ]
        if undefined_categories:
            overall[name] = {
                "status": "undefined",
                "unweighted_category_mean": None,
                "category_count": 0,
                "undefined_categories": undefined_categories,
                "reason": "At least one category metric is undefined",
            }
        else:
            overall[name] = {
                "status": "defined",
                "unweighted_category_mean": statistics.fmean(
                    per_category[category][name]["across_seeds"]["mean"]
                    for category in OFFICIAL_CATEGORIES
                ),
                "category_count": len(OFFICIAL_CATEGORIES),
                "undefined_categories": [],
                "reason": None,
            }
    return {
        "aggregation": {
            "seeds": "unweighted_mean_and_sample_standard_deviation",
            "categories": "unweighted_arithmetic_mean",
            "best_seed_selection": False,
        },
        "per_category": per_category,
        "overall": overall,
    }


def _assert_summary_number(actual: Any, expected: Any, location: str) -> None:
    if (
        not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
        or not math.isfinite(float(actual))
        or not math.isfinite(float(expected))
        or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ComparativeAnalysisError(
            f"Recomputed aggregate differs from frozen benchmark summary: {location}"
        )


def _validate_frozen_summary_aggregates(
    bundle: EvidenceBundle, aggregate: Mapping[str, Any]
) -> dict[str, Any]:
    """Cross-check every overlapping frozen summary metric against recomputation."""

    frozen = bundle.frozen_summary
    identity_fields = (
        "benchmark_git_commit",
        "dataset_audit_sha256",
        "protocol_fingerprint",
        "protocol_id",
        "weight_sha256",
        "environment_sha256",
        "imagenette_archive_sha256",
        "teacher_weight_sha256",
    )
    shared_identity_fields = [
        field
        for field in identity_fields
        if field in frozen or field in bundle.committed_manifest
    ]
    if any(
        field not in frozen
        or field not in bundle.committed_manifest
        or frozen[field] != bundle.committed_manifest[field]
        for field in shared_identity_fields
    ) or frozen.get("run_count") != len(bundle.artifacts):
        raise ComparativeAnalysisError(
            f"{bundle.spec.name} frozen summary provenance is inconsistent"
        )
    frozen_per_category = frozen.get("per_category")
    frozen_overall = frozen.get("overall")
    if not isinstance(frozen_per_category, dict) or not isinstance(
        frozen_overall, dict
    ):
        raise ComparativeAnalysisError(
            f"{bundle.spec.name} frozen summary aggregation is missing"
        )
    frozen_metrics = ("image_f1", "image_auroc", "pixel_f1", "au_pro_0.05")
    comparison_count = 0
    for category in OFFICIAL_CATEGORIES:
        frozen_category = frozen_per_category.get(category)
        if not isinstance(frozen_category, dict):
            raise ComparativeAnalysisError(
                f"{bundle.spec.name} frozen summary lacks category {category}"
            )
        frozen_per_seed = frozen_category.get("per_seed")
        frozen_across = frozen_category.get("across_seeds")
        if not isinstance(frozen_per_seed, dict) or not isinstance(frozen_across, dict):
            raise ComparativeAnalysisError(
                f"{bundle.spec.name} frozen category aggregate is invalid: {category}"
            )
        for metric in frozen_metrics:
            frozen_seed_metric = frozen_per_seed.get(metric)
            frozen_across_metric = frozen_across.get(metric)
            computed_metric = aggregate["per_category"][category][metric]
            if not isinstance(frozen_seed_metric, dict) or not isinstance(
                frozen_across_metric, dict
            ):
                raise ComparativeAnalysisError(
                    f"{bundle.spec.name} frozen summary lacks {category}:{metric}"
                )
            for seed in PROTOCOL_SEEDS:
                computed_seed = computed_metric["per_seed"][str(seed)]
                if computed_seed.get("status") != "defined":
                    raise ComparativeAnalysisError(
                        f"Frozen summary metric unexpectedly became undefined: "
                        f"{bundle.spec.name}:{category}:{seed}:{metric}"
                    )
                _assert_summary_number(
                    computed_seed["value"],
                    frozen_seed_metric.get(str(seed)),
                    f"{bundle.spec.name}:{category}:{seed}:{metric}",
                )
                comparison_count += 1
            computed_across = computed_metric["across_seeds"]
            for key in ("mean", "sample_standard_deviation", "count"):
                _assert_summary_number(
                    computed_across.get(key),
                    frozen_across_metric.get(key),
                    f"{bundle.spec.name}:{category}:{metric}:{key}",
                )
                comparison_count += 1
    for metric in frozen_metrics:
        frozen_metric = frozen_overall.get(metric)
        computed_metric = aggregate["overall"][metric]
        if not isinstance(frozen_metric, dict):
            raise ComparativeAnalysisError(
                f"{bundle.spec.name} frozen overall summary lacks {metric}"
            )
        _assert_summary_number(
            computed_metric.get("unweighted_category_mean"),
            frozen_metric.get("unweighted_category_mean"),
            f"{bundle.spec.name}:overall:{metric}",
        )
        _assert_summary_number(
            computed_metric.get("category_count"),
            frozen_metric.get("category_count"),
            f"{bundle.spec.name}:overall:{metric}:category_count",
        )
        comparison_count += 2
    return {
        "status": "passed",
        "metrics": list(frozen_metrics),
        "identity_fields": shared_identity_fields,
        "run_count": len(bundle.artifacts),
        "numeric_comparisons": comparison_count,
    }


def _aggregate_disagreements(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_category: dict[str, Any] = {}
    overall = Counter(record["disagreement"] for record in records)
    for category in OFFICIAL_CATEGORIES:
        category_records = [
            record for record in records if record["category"] == category
        ]
        per_seed: dict[str, Any] = {}
        for seed in PROTOCOL_SEEDS:
            counts = Counter(
                record["disagreement"]
                for record in category_records
                if record["seed"] == seed
            )
            per_seed[str(seed)] = {name: counts[name] for name in DISAGREEMENT_ORDER}
        category_counts = Counter(record["disagreement"] for record in category_records)
        per_category[category] = {
            "per_seed": per_seed,
            "all_seed_observations": {
                name: category_counts[name] for name in DISAGREEMENT_ORDER
            },
        }
    return {
        "per_category": per_category,
        "all_category_seed_observations": {
            name: overall[name] for name in DISAGREEMENT_ORDER
        },
        "unit_note": (
            "Counts retain each category/seed prediction; the same public image "
            "appears once per frozen seed."
        ),
    }


def _continuous_palette() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - analysis environment contract
        raise ComparativeAnalysisError("Panel rendering requires NumPy") from exc
    anchors = np.asarray(
        [
            (0, 0, 4),
            (87, 15, 109),
            (187, 55, 84),
            (249, 142, 8),
            (252, 255, 164),
        ],
        dtype=np.float64,
    )
    positions = np.linspace(0.0, len(anchors) - 1, 256)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weights = (positions - lower)[:, None]
    return np.rint(anchors[lower] * (1.0 - weights) + anchors[upper] * weights).astype(
        np.uint8
    )


def _heatmap_image(array: Any) -> Any:
    import numpy as np
    from PIL import Image

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum > minimum:
        normalized = (array.astype(np.float32) - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros(array.shape, dtype=np.float32)
    indexes = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(_continuous_palette()[indexes], mode="RGB")


def _binary_image(array: Any, *, foreground: tuple[int, int, int]) -> Any:
    import numpy as np
    from PIL import Image

    binary = np.asarray(array).astype(bool, copy=False)
    rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb[binary] = foreground
    return Image.fromarray(rgb, mode="RGB")


def _fit_tile(image: Any, size: tuple[int, int], *, nearest: bool) -> Any:
    from PIL import Image

    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    copy = image.convert("RGB")
    copy.thumbnail(size, resampling)
    tile = Image.new("RGB", size, "black")
    offset = ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2)
    tile.paste(copy, offset)
    return tile


def _render_panel(
    *,
    output_path: Path,
    sample_id: str,
    dataset_root: Path,
    mask: Any,
    patch_continuous: Any,
    patch_thresholded: Any,
    efficient_continuous: Any,
    efficient_thresholded: Any,
    patch_threshold: float,
    efficient_threshold: float,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise ComparativeAnalysisError("Panel rendering requires Pillow") from exc
    image_path = _safe_join(dataset_root, sample_id, "public panel image")
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    tiles = (
        ("Original image", original, False),
        ("Ground-truth mask", _binary_image(mask, foreground=(255, 255, 255)), True),
        (
            f"PatchCore continuous; T={patch_threshold:.6g}",
            _heatmap_image(patch_continuous),
            False,
        ),
        (
            "PatchCore frozen binary",
            _binary_image(patch_thresholded, foreground=(255, 80, 80)),
            True,
        ),
        (
            f"EfficientAD continuous; T={efficient_threshold:.6g}",
            _heatmap_image(efficient_continuous),
            False,
        ),
        (
            "EfficientAD frozen binary",
            _binary_image(efficient_thresholded, foreground=(255, 80, 80)),
            True,
        ),
    )
    tile_size = (256, 192)
    header_height = 28
    title_height = 34
    canvas = Image.new(
        "RGB",
        (tile_size[0] * len(tiles), title_height + header_height + tile_size[1]),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), sample_id, fill="black", font=font)
    for index, (label, tile_image, nearest) in enumerate(tiles):
        x = index * tile_size[0]
        draw.text((x + 5, title_height + 8), label, fill="black", font=font)
        canvas.paste(
            _fit_tile(tile_image, tile_size, nearest=nearest),
            (x, title_height + header_height),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False, compress_level=9)


def _source_history(bundle: EvidenceBundle) -> dict[str, Any]:
    cells = bundle.committed_manifest["cells"]
    histories: dict[str, Any] = {}
    failure_count = 0
    interruption_count = 0
    for category in OFFICIAL_CATEGORIES:
        for seed in PROTOCOL_SEEDS:
            key = f"{category}:{seed}"
            entry = cells[key]
            failures = entry.get("failure_history", entry.get("failed_attempts", []))
            interruptions = entry.get("interruption_history", [])
            if failures or interruptions:
                histories[key] = {
                    "failures": failures,
                    "interruptions": interruptions,
                }
            failure_count += len(failures)
            interruption_count += len(interruptions)
    return {
        "failed_attempt_count": failure_count,
        "interruption_count": interruption_count,
        "nonempty_histories": histories,
        "negative_findings_preserved": True,
    }


def _protect_input_roots(
    *,
    report_output: Path,
    panel_output: Path,
    protected_roots: Sequence[Path],
) -> None:
    report_resolved = report_output.resolve()
    panel_resolved = panel_output.resolve()
    if report_resolved == panel_resolved:
        raise ComparativeAnalysisError("Report and panel outputs must be distinct")
    for output in (report_resolved, panel_resolved):
        for protected in protected_roots:
            protected_resolved = protected.resolve()
            if output == protected_resolved or output.is_relative_to(
                protected_resolved
            ):
                raise ComparativeAnalysisError(
                    "Analysis outputs must not be placed inside read-only "
                    "evidence roots"
                )


def _variant_payload(state: Mapping[str, Counter[str]]) -> dict[str, Any]:
    fields = (
        "image_count",
        "normal_image_count",
        "anomalous_image_count",
        "image_predicted_positive_count",
        "predicted_positive_pixels",
        "true_positive_pixels",
        "false_positive_pixels",
        "anomalous_images_with_mask_overlap",
    )
    return {
        variant: {field: counts[field] for field in fields}
        for variant, counts in sorted(state.items())
    }


def _threshold_diagnostics(
    *,
    artifact: Mapping[str, Any],
    pixel_metrics: Mapping[str, Any],
    sample_diagnostics: Sequence[Mapping[str, Any]],
    flag_counts: Counter[str],
    variant_state: Mapping[str, Counter[str]],
) -> dict[str, Any]:
    threshold = float(artifact["thresholds"]["pixel"])
    anomalous = [item for item in sample_diagnostics if item["label"] == 1]
    all_maxima = [float(item["continuous_maximum"]) for item in sample_diagnostics]
    anomalous_maxima = [float(item["continuous_maximum"]) for item in anomalous]
    anomaly_pixel_distribution = pixel_metrics["continuous_score_distributions"][
        "ground_truth_anomaly_pixels"
    ]
    mismatch_pixels = sum(
        int(item["float16_threshold_comparison"]["mismatched_pixels"])
        for item in sample_diagnostics
    )
    return {
        "frozen_pixel_threshold": threshold,
        "calibration": _calibration_diagnostics(artifact),
        "public_continuous_map_maximum": max(all_maxima),
        "public_anomalous_image_map_maximum": max(anomalous_maxima),
        "no_public_map_score_strictly_exceeds_threshold": threshold >= max(all_maxima),
        "no_ground_truth_anomaly_pixel_score_strictly_exceeds_threshold": (
            anomaly_pixel_distribution.get("status") == "defined"
            and threshold >= float(anomaly_pixel_distribution["maximum"])
        ),
        "anomalous_image_count": len(anomalous),
        "anomalous_images_with_any_thresholded_pixel": sum(
            int(item["predicted_positive_pixels"] > 0) for item in anomalous
        ),
        "anomalous_images_with_mask_overlap": sum(
            int(item["confusion"]["true_positive"] > 0) for item in anomalous
        ),
        "anomalous_images_with_float16_max_above_threshold": sum(
            int(item["continuous_maximum"] > threshold) for item in anomalous
        ),
        "float16_vs_stored_thresholded_map": {
            "mismatched_pixels": mismatch_pixels,
            "images_with_mismatch": sum(
                int(item["float16_threshold_comparison"]["mismatched_pixels"] > 0)
                for item in sample_diagnostics
            ),
            "stored_positive_float16_not_strictly_above": sum(
                int(
                    item["float16_threshold_comparison"][
                        "stored_positive_float16_not_strictly_above"
                    ]
                )
                for item in sample_diagnostics
            ),
            "float16_strictly_above_stored_negative": sum(
                int(
                    item["float16_threshold_comparison"][
                        "float16_strictly_above_stored_negative"
                    ]
                )
                for item in sample_diagnostics
            ),
            "authoritative_thresholded_evidence": "stored_png",
            "cause": (
                "Stored PNGs were created from pre-cast float32 maps; continuous "
                "evidence was subsequently stored as float16. Pixel confusion and "
                "F1 therefore use the stored PNG rather than re-thresholded float16."
            ),
        },
        "localization_flag_counts": {
            name: flag_counts[name]
            for name in (
                "missed_anomaly",
                "under_localization",
                "over_localization",
                "diffuse_false_positive_map",
                "threshold_collapse",
                "constant_continuous_map",
            )
        },
        "capture_variant_diagnostics": _variant_payload(variant_state),
    }


def _comparative_deltas(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def delta(
        efficient: Mapping[str, Any], patch: Mapping[str, Any], key: str
    ) -> dict[str, Any]:
        if efficient.get("status") != "defined" or patch.get("status") != "defined":
            return {
                "status": "undefined",
                "value": None,
                "reason": "At least one model aggregate is undefined",
            }
        return {
            "status": "defined",
            "value": float(efficient[key]) - float(patch[key]),
            "reason": None,
        }

    overall: dict[str, Any] = {}
    per_category: dict[str, Any] = {category: {} for category in OFFICIAL_CATEGORIES}
    metric_names = [name for name, _ in (*IMAGE_METRIC_PATHS, *PIXEL_METRIC_PATHS)]
    for name in metric_names:
        category_seed_deltas: dict[str, list[dict[str, Any]]] = {
            str(seed): [] for seed in PROTOCOL_SEEDS
        }
        for category in OFFICIAL_CATEGORIES:
            per_seed: dict[str, Any] = {}
            seed_payloads: list[dict[str, Any]] = []
            for seed in PROTOCOL_SEEDS:
                seed_delta = delta(
                    aggregates[EFFICIENTAD]["per_category"][category][name]["per_seed"][
                        str(seed)
                    ],
                    aggregates[PATCHCORE]["per_category"][category][name]["per_seed"][
                        str(seed)
                    ],
                    "value",
                )
                per_seed[str(seed)] = seed_delta
                seed_payloads.append(seed_delta)
                category_seed_deltas[str(seed)].append(seed_delta)
            per_category[category][name] = {
                "per_seed": per_seed,
                "paired_across_seeds": _seed_summary(
                    seed_payloads, seeds=PROTOCOL_SEEDS
                ),
            }

        per_seed_category_means: dict[str, Any] = {}
        overall_seed_payloads: list[dict[str, Any]] = []
        for seed in PROTOCOL_SEEDS:
            seed_deltas = category_seed_deltas[str(seed)]
            undefined_categories = [
                category
                for category, payload in zip(
                    OFFICIAL_CATEGORIES, seed_deltas, strict=True
                )
                if payload["status"] != "defined"
            ]
            if undefined_categories:
                seed_mean = {
                    "status": "undefined",
                    "value": None,
                    "reason": "At least one category paired delta is undefined",
                    "undefined_categories": undefined_categories,
                }
            else:
                seed_mean = {
                    "status": "defined",
                    "value": statistics.fmean(
                        float(payload["value"]) for payload in seed_deltas
                    ),
                    "reason": None,
                    "undefined_categories": [],
                }
            per_seed_category_means[str(seed)] = seed_mean
            overall_seed_payloads.append(seed_mean)
        overall[name] = {
            "difference_of_model_overall_means": delta(
                aggregates[EFFICIENTAD]["overall"][name],
                aggregates[PATCHCORE]["overall"][name],
                "unweighted_category_mean",
            ),
            "per_seed_unweighted_category_mean": per_seed_category_means,
            "paired_across_seeds": _seed_summary(
                overall_seed_payloads, seeds=PROTOCOL_SEEDS
            ),
        }
    return {
        "direction": "efficientad_minus_patchcore",
        "overall": overall,
        "per_category": per_category,
    }


def _targeted_efficientad_diagnostics(
    cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in ("can", "walnuts"):
        per_seed: dict[str, Any] = {}
        for seed in PROTOCOL_SEEDS:
            cell = cells[category][str(seed)]
            threshold = cell["threshold_diagnostics"]
            pixel = cell["pixel_metrics"]
            per_seed[str(seed)] = {
                "au_pro_0.05": _metric_value(
                    pixel["au_pro_0.05"], f"{category}:{seed}:au_pro"
                ),
                "pixel_f1": _metric_value(
                    pixel["pixel_f1"], f"{category}:{seed}:pixel_f1"
                ),
                "pixel_auroc_diagnostic": _metric_value(
                    pixel["pixel_auroc_diagnostic"],
                    f"{category}:{seed}:pixel_auroc",
                ),
                "pixel_confusion": pixel["confusion"],
                "pixel_threshold": threshold["frozen_pixel_threshold"],
                "calibration_maximum_sample_id": threshold["calibration"]["pixel"][
                    "maximum_sample_id"
                ],
                "calibration_maximum_to_second_maximum_ratio": threshold["calibration"][
                    "pixel"
                ]["maximum_to_second_maximum_ratio"],
                "calibration_maximum_to_median_ratio": threshold["calibration"][
                    "pixel"
                ]["maximum_to_median_ratio"],
                "anomalous_image_count": threshold["anomalous_image_count"],
                "anomalous_images_with_any_thresholded_pixel": threshold[
                    "anomalous_images_with_any_thresholded_pixel"
                ],
                "anomalous_images_with_mask_overlap": threshold[
                    "anomalous_images_with_mask_overlap"
                ],
                (
                    "no_ground_truth_anomaly_pixel_score_strictly_exceeds_threshold"
                ): threshold[
                    "no_ground_truth_anomaly_pixel_score_strictly_exceeds_threshold"
                ],
                "float16_vs_stored_thresholded_map": threshold[
                    "float16_vs_stored_thresholded_map"
                ],
                "native_map_normalization": threshold["calibration"].get(
                    "native_map_normalization"
                ),
            }
        result[category] = {"per_seed": per_seed}
    result["identifiability_limit"] = (
        "Frozen artifacts preserve only each model's combined continuous map. "
        "EfficientAD student-teacher and autoencoder component maps are absent, "
        "so branch-level normalization causality cannot be identified without "
        "forbidden reevaluation."
    )
    return result


def _json_bytes(value: Any, *, indent: int | None) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def run_comparative_analysis(
    *,
    repository_root: Path,
    dataset_root: Path,
    audit_report_path: Path,
    patchcore_manifest_path: Path,
    patchcore_evidence_root: Path,
    efficientad_manifest_path: Path,
    efficientad_evidence_root: Path,
    report_output: Path,
    panel_output: Path,
) -> dict[str, Any]:
    """Validate and compare existing frozen public evidence without model code."""

    _protect_input_roots(
        report_output=report_output,
        panel_output=panel_output,
        protected_roots=(
            dataset_root,
            patchcore_evidence_root,
            efficientad_evidence_root,
            patchcore_manifest_path.parent,
            efficientad_manifest_path.parent,
            audit_report_path.parent,
        ),
    )
    analysis_environment_start = _analysis_environment()
    analysis_implementation_start = _analysis_implementation(repository_root)
    bundles = {
        PATCHCORE: _load_evidence(
            EvidenceSpec(
                name=PATCHCORE,
                committed_manifest_path=patchcore_manifest_path,
                evidence_root=patchcore_evidence_root,
                protocol_path=repository_root
                / "configs/protocols/patchcore-mvtecad2-v1.yaml",
                protocol_loader=load_protocol,
                fingerprint=protocol_fingerprint,
                artifact_validator=validate_artifact,
            ),
            repository_root,
        ),
        EFFICIENTAD: _load_evidence(
            EvidenceSpec(
                name=EFFICIENTAD,
                committed_manifest_path=efficientad_manifest_path,
                evidence_root=efficientad_evidence_root,
                protocol_path=repository_root
                / "configs/protocols/efficientad-mvtecad2-v1.yaml",
                protocol_loader=load_efficientad_protocol,
                fingerprint=efficientad_protocol_fingerprint,
                artifact_validator=validate_efficientad_artifact,
            ),
            repository_root,
        ),
    }
    expected_audit = bundles[PATCHCORE].committed_manifest["dataset_audit_sha256"]
    if (
        bundles[EFFICIENTAD].committed_manifest["dataset_audit_sha256"]
        != expected_audit
    ):
        raise ComparativeAnalysisError(
            "PatchCore and EfficientAD use different dataset audits"
        )
    audit_index, audited_validation_ids, audit_provenance = (
        _load_and_verify_public_audit(audit_report_path, dataset_root, expected_audit)
    )

    for model in MODEL_ORDER:
        for category in OFFICIAL_CATEGORIES:
            expected_calibration_ids = audited_validation_ids[category]
            for seed in PROTOCOL_SEEDS:
                actual_calibration_ids = tuple(
                    _validate_calibration_contract(
                        bundles[model].artifacts[(category, seed)], category
                    )
                )
                if actual_calibration_ids != expected_calibration_ids:
                    raise ComparativeAnalysisError(
                        "Frozen calibration inputs do not match the exact audited "
                        f"validation-normal inventory: {model}:{category}:{seed}"
                    )

    paired_records: list[dict[str, Any]] = []
    records_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    audited_images_by_category = {
        category: tuple(
            sorted(
                path
                for path, entry in audit_index.items()
                if entry.get("kind") == "image" and entry.get("category") == category
            )
        )
        for category in OFFICIAL_CATEGORIES
    }
    for category in OFFICIAL_CATEGORIES:
        for seed in PROTOCOL_SEEDS:
            records = pair_prediction_records(
                bundles[PATCHCORE].artifacts[(category, seed)]["predictions"],
                bundles[EFFICIENTAD].artifacts[(category, seed)]["predictions"],
                category=category,
                seed=seed,
            )
            if (
                tuple(record["sample_id"] for record in records)
                != (audited_images_by_category[category])
            ):
                raise ComparativeAnalysisError(
                    "Frozen predictions do not cover the exact audited public image "
                    f"inventory: {category}:{seed}"
                )
            records_by_cell[(category, seed)] = records
            paired_records.extend(records)
    audited_images = {
        path for path, entry in audit_index.items() if entry.get("kind") == "image"
    }
    analyzed_images = {
        record["sample_id"]
        for record in paired_records
        if record["seed"] == PANEL_SELECTION_SEED
    }
    if audited_images != analyzed_images:
        raise ComparativeAnalysisError(
            "Frozen predictions do not cover the exact audited public image set"
        )
    audited_masks = {
        path for path, entry in audit_index.items() if entry.get("kind") == "mask"
    }
    expected_masks = {
        mask_id
        for sample_id in analyzed_images
        if (mask_id := _expected_mask_id(sample_id)) is not None
    }
    if audited_masks != expected_masks:
        raise ComparativeAnalysisError(
            "Frozen predictions do not cover the exact audited public mask set"
        )
    selections = select_panel_examples(paired_records)
    selection_lookup = {
        (item["category"], item["seed"], item["sample_id"]): item for item in selections
    }

    verification_counts: Counter[str] = Counter()
    model_cells: dict[str, dict[str, dict[str, Any]]] = {
        model: {category: {} for category in OFFICIAL_CATEGORIES}
        for model in MODEL_ORDER
    }
    panel_entries: list[dict[str, Any]] = []
    analyzed_records: list[dict[str, Any]] = []
    for category in OFFICIAL_CATEGORIES:
        for seed in PROTOCOL_SEEDS:
            artifacts = {
                model: bundles[model].artifacts[(category, seed)]
                for model in MODEL_ORDER
            }
            predictions = {
                model: artifacts[model]["predictions"] for model in MODEL_ORDER
            }
            image_metrics: dict[str, Any] = {}
            pixel_accumulators = {
                model: Float16PixelAnalysisAccumulator() for model in MODEL_ORDER
            }
            per_model_sample_diagnostics: dict[str, list[dict[str, Any]]] = {
                model: [] for model in MODEL_ORDER
            }
            flag_counts = {model: Counter() for model in MODEL_ORDER}
            variants: dict[str, defaultdict[str, Counter[str]]] = {
                model: defaultdict(Counter) for model in MODEL_ORDER
            }
            for model in MODEL_ORDER:
                labels = [int(item["label"]) for item in predictions[model]]
                decisions = [
                    int(item["image_prediction"]) for item in predictions[model]
                ]
                scores = [float(item["anomaly_score"]) for item in predictions[model]]
                measured = classification_metrics(labels, decisions, scores)
                _assert_recorded_metric(
                    artifacts[model], "image_f1", measured["image_f1"]
                )
                _assert_recorded_metric(
                    artifacts[model], "image_auroc", measured["image_auroc"]
                )
                measured["anomaly_score_distributions"] = score_distributions(
                    labels, scores
                )
                image_metrics[model] = measured

            for pair in records_by_cell[(category, seed)]:
                ordinal = int(pair["ordinal"])
                sample_id = str(pair["sample_id"])
                image_entry = audit_index.get(sample_id)
                if (
                    not isinstance(image_entry, Mapping)
                    or image_entry.get("kind") != "image"
                ):
                    raise ComparativeAnalysisError(
                        f"Public sample is absent from the audit: {sample_id}"
                    )
                continuous: dict[str, Any] = {}
                thresholded: dict[str, Any] = {}
                for model in MODEL_ORDER:
                    run_dir = bundles[model].artifact_paths[(category, seed)].parent
                    continuous[model], _ = _read_continuous_map(
                        run_dir, predictions[model][ordinal], verification_counts
                    )
                    thresholded[model], _ = _read_thresholded_map(
                        run_dir,
                        predictions[model][ordinal],
                        tuple(continuous[model].shape),
                        verification_counts,
                    )
                shape = tuple(continuous[PATCHCORE].shape)
                if (
                    tuple(continuous[EFFICIENTAD].shape) != shape
                    or image_entry.get("height") != shape[0]
                    or image_entry.get("width") != shape[1]
                ):
                    raise ComparativeAnalysisError(
                        f"Model/audit coordinate shapes differ: {sample_id}"
                    )
                mask = _load_public_mask(sample_id, shape, dataset_root, audit_index)
                analyzed = dict(pair)
                for model in MODEL_ORDER:
                    sample_counts = pixel_accumulators[model].update(
                        mask, continuous[model], thresholded[model]
                    )
                    diagnostic = localization_diagnostics(
                        mask, continuous[model], thresholded[model]
                    )
                    pixel_threshold = float(artifacts[model]["thresholds"]["pixel"])
                    float16_thresholded = continuous[model] > pixel_threshold
                    stored_thresholded = thresholded[model].astype(bool, copy=False)
                    diagnostic["float16_threshold_comparison"] = {
                        "mismatched_pixels": int(
                            (float16_thresholded != stored_thresholded).sum()
                        ),
                        "stored_positive_float16_not_strictly_above": int(
                            (stored_thresholded & ~float16_thresholded).sum()
                        ),
                        "float16_strictly_above_stored_negative": int(
                            (float16_thresholded & ~stored_thresholded).sum()
                        ),
                    }
                    if diagnostic["confusion"] != sample_counts.as_dict():
                        raise ComparativeAnalysisError(
                            f"Pixel confusion calculations disagree: {sample_id}"
                        )
                    diagnostic["label"] = int(pair["label"])
                    per_model_sample_diagnostics[model].append(diagnostic)
                    flag_counts[model].update(diagnostic["flags"])
                    variant = variants[model][_capture_variant(sample_id)]
                    variant["image_count"] += 1
                    variant["normal_image_count"] += int(pair["label"] == 0)
                    variant["anomalous_image_count"] += int(pair["label"] == 1)
                    variant["image_predicted_positive_count"] += int(
                        analyzed[model]["image_prediction"] == 1
                    )
                    variant["predicted_positive_pixels"] += diagnostic[
                        "predicted_positive_pixels"
                    ]
                    variant["true_positive_pixels"] += diagnostic["confusion"][
                        "true_positive"
                    ]
                    variant["false_positive_pixels"] += diagnostic["confusion"][
                        "false_positive"
                    ]
                    variant["anomalous_images_with_mask_overlap"] += int(
                        pair["label"] == 1
                        and diagnostic["confusion"]["true_positive"] > 0
                    )
                    analyzed[model]["localization"] = diagnostic
                analyzed_records.append(analyzed)

                selection = selection_lookup.get((category, seed, sample_id))
                if selection is not None:
                    filename = (
                        f"{category}__seed-{seed}__label-{pair['label']}__"
                        f"{selection['selection_sha256'][:12]}.png"
                    )
                    output_path = panel_output / filename
                    _render_panel(
                        output_path=output_path,
                        sample_id=sample_id,
                        dataset_root=dataset_root,
                        mask=mask,
                        patch_continuous=continuous[PATCHCORE],
                        patch_thresholded=thresholded[PATCHCORE],
                        efficient_continuous=continuous[EFFICIENTAD],
                        efficient_thresholded=thresholded[EFFICIENTAD],
                        patch_threshold=float(
                            artifacts[PATCHCORE]["thresholds"]["pixel"]
                        ),
                        efficient_threshold=float(
                            artifacts[EFFICIENTAD]["thresholds"]["pixel"]
                        ),
                    )
                    mask_id = _expected_mask_id(sample_id)
                    panel_entries.append(
                        {
                            **selection,
                            "panel_filename": filename,
                            "panel_sha256": sha256_file(output_path),
                            "source_image_sha256": image_entry["sha256"],
                            "ground_truth_mask": (
                                {
                                    "sample_id": mask_id,
                                    "sha256": audit_index[mask_id]["sha256"],
                                    "generated_zero_mask": False,
                                }
                                if mask_id is not None
                                else {
                                    "sample_id": None,
                                    "sha256": None,
                                    "generated_zero_mask": True,
                                }
                            ),
                            "patchcore_continuous_sha256": predictions[PATCHCORE][
                                ordinal
                            ]["anomaly_map"]["sha256"],
                            "patchcore_thresholded_sha256": predictions[PATCHCORE][
                                ordinal
                            ]["anomaly_map"]["thresholded_sha256"],
                            "efficientad_continuous_sha256": predictions[EFFICIENTAD][
                                ordinal
                            ]["anomaly_map"]["sha256"],
                            "efficientad_thresholded_sha256": predictions[EFFICIENTAD][
                                ordinal
                            ]["anomaly_map"]["thresholded_sha256"],
                            "display_scaling": (
                                "Each continuous map is independently min-max scaled "
                                "for display only; numeric analysis uses stored values."
                            ),
                        }
                    )

            for model in MODEL_ORDER:
                pixel_metrics = pixel_accumulators[model].result()
                _assert_recorded_metric(
                    artifacts[model], "pixel_f1", pixel_metrics["pixel_f1"]
                )
                _assert_recorded_metric(
                    artifacts[model], "au_pro_0.05", pixel_metrics["au_pro_0.05"]
                )
                model_cells[model][category][str(seed)] = {
                    "thresholds": {
                        "image": float(artifacts[model]["thresholds"]["image"]),
                        "pixel": float(artifacts[model]["thresholds"]["pixel"]),
                    },
                    "image_metrics": image_metrics[model],
                    "pixel_metrics": pixel_metrics,
                    "threshold_diagnostics": _threshold_diagnostics(
                        artifact=artifacts[model],
                        pixel_metrics=pixel_metrics,
                        sample_diagnostics=per_model_sample_diagnostics[model],
                        flag_counts=flag_counts[model],
                        variant_state=variants[model],
                    ),
                    "recorded_frozen_metrics_recomputed": {
                        "image_f1": True,
                        "image_auroc": True,
                        "pixel_f1": True,
                        "au_pro_0.05": True,
                    },
                }

    expected_map_count = sum(
        bundle.provenance["prediction_count"] for bundle in bundles.values()
    )
    if (
        verification_counts["continuous_maps"] != expected_map_count
        or verification_counts["thresholded_maps"] != expected_map_count
    ):
        raise ComparativeAnalysisError("Not every referenced map hash was verified")
    if len(panel_entries) != len(selections):
        raise ComparativeAnalysisError("Not every deterministic panel was rendered")

    aggregates = {
        model: _aggregate_model_results(model_cells[model]) for model in MODEL_ORDER
    }
    frozen_summary_reproduction = {
        model: _validate_frozen_summary_aggregates(bundles[model], aggregates[model])
        for model in MODEL_ORDER
    }
    if _analysis_environment() != analysis_environment_start:
        raise ComparativeAnalysisError(
            "Analysis environment changed while evidence was being processed"
        )
    if _analysis_implementation(repository_root) != analysis_implementation_start:
        raise ComparativeAnalysisError(
            "Analysis implementation changed while evidence was being processed"
        )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": "phase4a-comparative-failure-analysis-v1",
        "scope": {
            "evaluation_split": "test_public",
            "models": list(MODEL_ORDER),
            "new_training": False,
            "new_inference": False,
            "threshold_tuning": False,
            "private_image_or_label_assets_opened": False,
            "input_artifacts_mutated": False,
        },
        "metric_families": {
            "ranking": [
                "image_auroc",
                "au_pro_0.05",
                "pixel_auroc_diagnostic",
            ],
            "frozen_validation_threshold_dependent": [
                "image_confusion",
                "sensitivity",
                "specificity",
                "precision",
                "image_f1",
                "pixel_confusion",
                "pixel_precision",
                "pixel_sensitivity",
                "pixel_specificity",
                "pixel_f1",
            ],
            "diagnostic_metric_note": (
                "Pixel AUROC is a Phase 4A diagnostic, not a frozen selection metric."
            ),
        },
        "provenance": {
            "analysis_environment": analysis_environment_start,
            "analysis_implementation": analysis_implementation_start,
            "dataset_audit": audit_provenance,
            PATCHCORE: {
                **bundles[PATCHCORE].provenance,
                "source_history": _source_history(bundles[PATCHCORE]),
            },
            EFFICIENTAD: {
                **bundles[EFFICIENTAD].provenance,
                "source_history": _source_history(bundles[EFFICIENTAD]),
            },
            "map_hashes_verified": {
                "continuous": verification_counts["continuous_maps"],
                "thresholded": verification_counts["thresholded_maps"],
            },
            "paired_prediction_count": len(analyzed_records),
            "pairing_identity": "exact_category_seed_order_sample_id_and_label",
            "audit_inventory_matches": {
                "public_model_pairs": len(OFFICIAL_CATEGORIES) * len(PROTOCOL_SEEDS),
                "validation_calibration_cells": len(MODEL_ORDER)
                * len(OFFICIAL_CATEGORIES)
                * len(PROTOCOL_SEEDS),
            },
            "frozen_summary_reproduction": frozen_summary_reproduction,
        },
        "failure_taxonomy": {
            "multi_label": True,
            "missed_anomaly": "ground-truth anomaly pixels exist and pixel TP equals 0",
            "under_localization": (
                "ground-truth anomaly pixels exist, TP is positive, and FN exceeds TP"
            ),
            "over_localization": (
                "ground-truth anomaly pixels exist, TP is positive, and FP exceeds TP"
            ),
            "diffuse_false_positive_map": (
                "false-positive pixels occur in every one of four fixed image quadrants"
            ),
            "threshold_collapse": (
                "a nonconstant continuous map becomes empty on an anomalous image "
                "or becomes full-frame under the stored frozen binary map"
            ),
            "constant_continuous_map": (
                "continuous map is constant and has zero overlap on an anomalous image"
            ),
            "threshold_source": "stored validation-normal-only frozen thresholded PNG",
        },
        "panel_selection": {
            "seed": PANEL_SELECTION_SEED,
            "rule": (
                "At seed 42, select the minimum SHA-256 key per category and true "
                "image label, plus the minimum key per nonempty global disagreement "
                "class; deduplicate while retaining every reason."
            ),
            "selection_inputs": (
                "Strata use category, fixed seed, public image label, and model "
                "disagreement; the within-stratum SHA-256 tie-break key uses category, "
                "seed, label, and sample_id. Scores and pixel metrics are excluded."
            ),
            "panel_count": len(panel_entries),
            "panels_committed": False,
            "license_and_size_policy": (
                "Panels remain under ignored outputs; only hashes and selection "
                "metadata are committed."
            ),
        },
        "cells": model_cells,
        "aggregates": aggregates,
        "comparative_deltas": _comparative_deltas(aggregates),
        "disagreements": _aggregate_disagreements(analyzed_records),
        "targeted_efficientad_investigation": _targeted_efficientad_diagnostics(
            model_cells[EFFICIENTAD]
        ),
    }
    panel_index = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": summary["analysis_id"],
        "panels_committed": False,
        "panel_output_role": "ignored_local_artifact",
        "selection_rule": summary["panel_selection"],
        "panels": panel_entries,
    }
    report_output.mkdir(parents=True, exist_ok=True)
    report_contents = {
        "per-image-analysis.jsonl": b"".join(
            _json_bytes(record, indent=None) for record in analyzed_records
        ),
        "panel-index.json": _json_bytes(panel_index, indent=2),
        "analysis-summary.json": _json_bytes(summary, indent=2),
    }
    for filename, content in report_contents.items():
        _write_file(report_output / filename, content)
    analysis_manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": summary["analysis_id"],
        "artifacts": {
            filename: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for filename, content in sorted(report_contents.items())
        },
        "per_image_record_count": len(analyzed_records),
        "panel_record_count": len(panel_entries),
        "input_manifest_sha256": {
            model: bundles[model].provenance["committed_manifest_sha256"]
            for model in MODEL_ORDER
        },
    }
    _write_file(
        report_output / "analysis-manifest.json",
        _json_bytes(analysis_manifest, indent=2),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit, read-only-evidence Phase 4A command contract."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare existing frozen PatchCore and EfficientAD public "
            "evidence without training, inference, or threshold changes."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Audited MVTec AD 2 root")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--patchcore-manifest",
        type=Path,
        default=Path("reports/phase2c-public-benchmark/benchmark-manifest.json"),
    )
    parser.add_argument(
        "--patchcore-evidence",
        type=Path,
        default=Path("outputs/phase2c-public-benchmark"),
    )
    parser.add_argument(
        "--efficientad-manifest",
        type=Path,
        default=Path(
            "reports/phase3b-efficientad-public-benchmark/benchmark-manifest.json"
        ),
    )
    parser.add_argument(
        "--efficientad-evidence",
        type=Path,
        default=Path("outputs/phase3b-efficientad-public-benchmark"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/phase4a-comparative-failure-analysis"),
    )
    parser.add_argument(
        "--panel-output",
        type=Path,
        default=Path("outputs/phase4a-comparative-failure-analysis/panels"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 4A, failing closed before reports if provenance is invalid."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_comparative_analysis(
            repository_root=Path.cwd(),
            dataset_root=args.dataset_root,
            audit_report_path=args.audit_report,
            patchcore_manifest_path=args.patchcore_manifest,
            patchcore_evidence_root=args.patchcore_evidence,
            efficientad_manifest_path=args.efficientad_manifest,
            efficientad_evidence_root=args.efficientad_evidence,
            report_output=args.report_output,
            panel_output=args.panel_output,
        )
    except (ComparativeAnalysisError, AnalysisMetricError) as exc:
        LOGGER.error("Phase 4A analysis failed: %s", exc)
        return 2
    LOGGER.info("Phase 4A analysis completed: %s", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
