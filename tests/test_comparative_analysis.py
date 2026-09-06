from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import visionguard.comparative_analysis as comparative_analysis
from visionguard.analysis_metrics import (
    Float16PixelAnalysisAccumulator,
    classification_metrics,
    localization_diagnostics,
    score_distributions,
)
from visionguard.artifacts import sha256_file
from visionguard.calibration import highest_order_statistic
from visionguard.comparative_analysis import (
    DISAGREEMENT_ORDER,
    EFFICIENTAD,
    PANEL_SELECTION_SEED,
    PATCHCORE,
    ComparativeAnalysisError,
    EvidenceBundle,
    EvidenceSpec,
    _load_and_verify_public_audit,
    _load_evidence,
    _manifest_differences,
    _protect_input_roots,
    _read_continuous_map,
    _read_thresholded_map,
    _safe_join,
    _threshold_diagnostics,
    _validate_calibration_contract,
    disagreement_bucket,
    pair_prediction_records,
    run_comparative_analysis,
    select_panel_examples,
)
from visionguard.metrics import au_pro, binary_f1
from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS


def _value(result: dict[str, object], name: str) -> float:
    metric = result[name]
    assert isinstance(metric, dict)
    assert metric["status"] == "defined"
    value = metric["value"]
    assert isinstance(value, float)
    return value


def test_image_confusion_rates_f1_and_tie_aware_auroc() -> None:
    result = classification_metrics(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 0],
        scores=[0.1, 0.4, 0.35, 0.8],
    )

    assert result["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert _value(result, "sensitivity") == 0.5
    assert _value(result, "specificity") == 0.5
    assert _value(result, "precision") == 0.5
    assert _value(result, "image_f1") == 0.5
    assert _value(result, "image_auroc") == 0.75


def test_score_distributions_keep_normal_and_anomalous_images_separate() -> None:
    result = score_distributions(
        labels=[0, 1, 0, 1],
        scores=[0.0, 10.0, 4.0, 14.0],
    )

    normal = result["normal_public_images"]
    anomalous = result["anomalous_public_images"]
    assert normal == {
        "count": 2,
        "minimum": 0.0,
        "q1": 1.0,
        "median": 2.0,
        "q3": 3.0,
        "maximum": 4.0,
        "mean": 2.0,
        "population_standard_deviation": 2.0,
        "quantile_method": "linear_index_n_minus_1",
    }
    assert anomalous == {
        "count": 2,
        "minimum": 10.0,
        "q1": 11.0,
        "median": 12.0,
        "q3": 13.0,
        "maximum": 14.0,
        "mean": 12.0,
        "population_standard_deviation": 2.0,
        "quantile_method": "linear_index_n_minus_1",
    }


def _calibration_payload(category: str) -> dict[str, object]:
    inputs = [
        {
            "sample_id": f"{category}/validation/good/{index:03d}_regular.png",
            "image_anomaly_score": (index + 1) / 38,
            "pixel_maximum": (index + 1) / 38,
            "restored_map_sha256": f"{index + 1:064x}",
            "restored_map_shape": [2, 2],
        }
        for index in range(19)
    ]
    scores = [float(item["image_anomaly_score"]) for item in inputs]
    maxima = [float(item["pixel_maximum"]) for item in inputs]
    return {
        "normal_only": True,
        "split": "validation",
        "comparison": "score_strictly_greater_than_threshold",
        "inputs": inputs,
        "image": asdict(highest_order_statistic(scores, minimum_samples=19)),
        "pixel": asdict(highest_order_statistic(maxima, minimum_samples=19)),
    }


def test_calibration_contract_recomputes_thresholds_and_preserves_exact_ids() -> None:
    calibration = _calibration_payload("can")
    artifact = {"calibration": calibration}

    sample_ids = _validate_calibration_contract(artifact, "can")

    assert sample_ids == [
        f"can/validation/good/{index:03d}_regular.png" for index in range(19)
    ]
    assert calibration["image"] == calibration["pixel"]
    assert calibration["image"]["threshold"] == 0.5  # type: ignore[index]


def test_calibration_contract_rejects_non_validation_sample_id() -> None:
    calibration = _calibration_payload("can")
    calibration["inputs"][0]["sample_id"] = (  # type: ignore[index]
        "can/test_public/good/000_regular.png"
    )

    with pytest.raises(ComparativeAnalysisError, match="not validation-normal"):
        _validate_calibration_contract({"calibration": calibration}, "can")


def test_calibration_contract_rejects_threshold_recomputation_mismatch() -> None:
    calibration = _calibration_payload("can")
    calibration["image"]["threshold"] = 0.49  # type: ignore[index]

    with pytest.raises(ComparativeAnalysisError, match="do not recompute"):
        _validate_calibration_contract({"calibration": calibration}, "can")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _minimal_audit_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    dataset_root = tmp_path / "dataset"
    public_assets = {
        "can/test_public/good/000_regular.png": b"public-good",
        "can/test_public/bad/001_regular.png": b"public-bad",
        "can/test_public/ground_truth/bad/001_regular_mask.png": b"public-mask",
    }
    for relative, content in public_assets.items():
        path = dataset_root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    files: list[dict[str, object]] = [
        {
            "path": f"{category}/validation/good/000_regular.png",
            "category": category,
            "split": "validation",
            "condition": "good",
            "kind": "image",
            "sha256": "f" * 64,
        }
        for category in OFFICIAL_CATEGORIES
    ]
    for relative in public_assets:
        is_mask = "/ground_truth/" in relative
        path = dataset_root / Path(relative)
        files.append(
            {
                "path": relative,
                "category": "can",
                "split": "test_public",
                "condition": "bad" if "/bad/" in relative else "good",
                "kind": "mask" if is_mask else "image",
                "sha256": sha256_file(path),
                "height": 2,
                "width": 2,
            }
        )
    audit = {
        "schema_version": 2,
        "summary": {
            "status": "passed",
            "error_count": 0,
            "warning_count": 0,
            "image_count": len(OFFICIAL_CATEGORIES) + 2,
            "mask_count": 1,
            "counts_by_split": {"test_public": 2},
            "unexpected_overlap_group_count": 0,
        },
        "files": files,
    }
    report_path = tmp_path / "audit" / "mvtec-ad-2-audit.json"
    _write_json(report_path, audit)
    return dataset_root, report_path, audit


def test_audit_verification_covers_public_assets_and_validation_inventory(
    tmp_path: Path,
) -> None:
    dataset_root, report_path, _ = _minimal_audit_fixture(tmp_path)

    public_index, validation_ids, provenance = _load_and_verify_public_audit(
        report_path, dataset_root, sha256_file(report_path)
    )

    assert set(public_index) == {
        "can/test_public/good/000_regular.png",
        "can/test_public/bad/001_regular.png",
        "can/test_public/ground_truth/bad/001_regular_mask.png",
    }
    assert validation_ids == {
        category: (f"{category}/validation/good/000_regular.png",)
        for category in OFFICIAL_CATEGORIES
    }
    assert provenance["public_images_verified"] == 2
    assert provenance["public_masks_verified"] == 1
    assert provenance["private_assets_opened"] == 0


def test_audit_verification_rejects_report_and_public_asset_hash_drift(
    tmp_path: Path,
) -> None:
    dataset_root, report_path, _ = _minimal_audit_fixture(tmp_path)

    with pytest.raises(ComparativeAnalysisError, match="audit hash differs"):
        _load_and_verify_public_audit(report_path, dataset_root, "0" * 64)

    public_path = dataset_root / "can/test_public/good/000_regular.png"
    public_path.write_bytes(b"modified-after-audit")
    with pytest.raises(ComparativeAnalysisError, match="asset hash is invalid"):
        _load_and_verify_public_audit(
            report_path, dataset_root, sha256_file(report_path)
        )


def test_audit_verification_rejects_schema_and_validation_inventory_drift(
    tmp_path: Path,
) -> None:
    dataset_root, report_path, audit = _minimal_audit_fixture(tmp_path)
    malformed = deepcopy(audit)
    malformed["schema_version"] = 1
    _write_json(report_path, malformed)

    with pytest.raises(ComparativeAnalysisError, match="clean schema-v2 pass"):
        _load_and_verify_public_audit(
            report_path, dataset_root, sha256_file(report_path)
        )

    missing_validation = deepcopy(audit)
    missing_validation["files"] = [
        entry
        for entry in missing_validation["files"]
        if not (
            entry.get("category") == "walnuts" and entry.get("split") == "validation"
        )
    ]
    _write_json(report_path, missing_validation)
    with pytest.raises(ComparativeAnalysisError, match="validation-normal inventory"):
        _load_and_verify_public_audit(
            report_path, dataset_root, sha256_file(report_path)
        )


def _minimal_patchcore_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setattr(comparative_analysis, "OFFICIAL_CATEGORIES", ("can",))
    monkeypatch.setattr(comparative_analysis, "PROTOCOL_SEEDS", (42,))
    monkeypatch.setattr(
        comparative_analysis,
        "_git_commit_status",
        lambda _root, commit: {"commit": commit, "git_object_verified": True},
    )
    monkeypatch.setattr(
        comparative_analysis,
        "_summary_validation",
        lambda *_args: ({"semantic_match": True}, {"status": "completed"}),
    )

    evidence_root = tmp_path / "local-evidence"
    artifact_path = evidence_root / "runs" / "can-42" / "artifact.json"
    calibration = _calibration_payload("can")
    artifact: dict[str, Any] = {
        "status": "completed",
        "category": "can",
        "seed": 42,
        "protocol_id": "unit-protocol",
        "protocol_fingerprint": "1" * 64,
        "protocol_snapshot": {"id": "unit-protocol"},
        "git": {"commit": "2" * 40, "dirty": False},
        "dataset": {"sha256": "3" * 64},
        "weight": {"sha256": "4" * 64},
        "thresholds": {"image": 0.5, "pixel": 0.5},
        "calibration": calibration,
        "predictions": [
            {
                "sample_id": "can/test_public/good/000_regular.png",
                "label": 0,
                "anomaly_score": 0.1,
                "image_prediction": 0,
            }
        ],
    }
    _write_json(artifact_path, artifact)
    manifest: dict[str, Any] = {
        "status": "completed",
        "protocol_id": "unit-protocol",
        "protocol_fingerprint": "1" * 64,
        "benchmark_git_commit": "2" * 40,
        "dataset_audit_sha256": "3" * 64,
        "weight_sha256": "4" * 64,
        "matrix": {
            "categories": ["can"],
            "seeds": [42],
            "expected_run_count": 1,
        },
        "cells": {
            "can:42": {
                "status": "completed",
                "category": "can",
                "seed": 42,
                "artifact_path": "runs/can-42/artifact.json",
                "artifact_sha256": sha256_file(artifact_path),
            }
        },
    }
    committed_manifest_path = tmp_path / "committed" / "benchmark-manifest.json"
    local_manifest_path = evidence_root / "benchmark-manifest.json"
    _write_json(committed_manifest_path, manifest)
    _write_json(local_manifest_path, manifest)
    spec = EvidenceSpec(
        name=PATCHCORE,
        committed_manifest_path=committed_manifest_path,
        evidence_root=evidence_root,
        protocol_path=tmp_path / "protocol.yaml",
        protocol_loader=lambda _path: {"protocol": {"id": "unit-protocol"}},
        fingerprint=lambda _document: "1" * 64,
        artifact_validator=lambda _artifact: None,
    )
    return {
        "artifact": artifact,
        "artifact_path": artifact_path,
        "committed_manifest_path": committed_manifest_path,
        "evidence_root": evidence_root,
        "local_manifest_path": local_manifest_path,
        "manifest": manifest,
        "spec": spec,
    }


def _rewrite_evidence_artifact(fixture: dict[str, Any]) -> None:
    _write_json(fixture["artifact_path"], fixture["artifact"])
    fixture["manifest"]["cells"]["can:42"]["artifact_sha256"] = sha256_file(
        fixture["artifact_path"]
    )
    _write_json(fixture["committed_manifest_path"], fixture["manifest"])
    _write_json(fixture["local_manifest_path"], fixture["manifest"])


def test_load_evidence_accepts_bound_manifest_artifact_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _minimal_patchcore_evidence(tmp_path, monkeypatch)

    bundle = _load_evidence(fixture["spec"], tmp_path)

    assert set(bundle.artifacts) == {("can", 42)}
    assert bundle.provenance["artifact_count"] == 1
    assert bundle.provenance["prediction_count"] == 1


def test_load_evidence_rejects_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _minimal_patchcore_evidence(tmp_path, monkeypatch)
    local_manifest = deepcopy(fixture["manifest"])
    local_manifest["protocol_id"] = "drifted-protocol"
    _write_json(fixture["local_manifest_path"], local_manifest)

    with pytest.raises(ComparativeAnalysisError, match="Local evidence differs"):
        _load_evidence(fixture["spec"], tmp_path)


def test_load_evidence_rejects_artifact_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _minimal_patchcore_evidence(tmp_path, monkeypatch)
    fixture["artifact_path"].write_text("{}", encoding="utf-8")

    with pytest.raises(ComparativeAnalysisError, match="artifact hash is invalid"):
        _load_evidence(fixture["spec"], tmp_path)


def test_load_evidence_rejects_artifact_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _minimal_patchcore_evidence(tmp_path, monkeypatch)
    fixture["artifact"]["dataset"]["sha256"] = "5" * 64
    _rewrite_evidence_artifact(fixture)

    with pytest.raises(ComparativeAnalysisError, match="provenance has drifted"):
        _load_evidence(fixture["spec"], tmp_path)


def test_map_readers_verify_continuous_and_thresholded_hashes(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from PIL import Image

    continuous_path = tmp_path / "maps" / "continuous.tiff"
    thresholded_path = tmp_path / "maps" / "thresholded.png"
    continuous_path.parent.mkdir(parents=True)
    tifffile.imwrite(
        continuous_path,
        np.array([[0.1, 0.9], [0.2, 0.3]], dtype=np.float16),
    )
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8), mode="L").save(
        thresholded_path
    )
    prediction: dict[str, Any] = {
        "sample_id": "can/test_public/bad/000_regular.png",
        "anomaly_map": {
            "path": "maps/continuous.tiff",
            "sha256": sha256_file(continuous_path),
            "shape": [2, 2],
            "dtype": "float16",
            "coordinate_space": "original_image_height_width",
            "finite": True,
            "thresholded_path": "maps/thresholded.png",
            "thresholded_sha256": sha256_file(thresholded_path),
        },
    }
    counts: Counter[str] = Counter()

    continuous, _ = _read_continuous_map(tmp_path, prediction, counts)
    thresholded, _ = _read_thresholded_map(
        tmp_path, prediction, continuous.shape, counts
    )

    assert continuous.dtype == np.float16
    assert thresholded.tolist() == [[False, True], [False, False]]
    assert counts == {"continuous_maps": 1, "thresholded_maps": 1}

    prediction["anomaly_map"]["sha256"] = "0" * 64
    with pytest.raises(ComparativeAnalysisError, match="map hash is invalid"):
        _read_continuous_map(tmp_path, prediction, Counter())
    prediction["anomaly_map"]["sha256"] = sha256_file(continuous_path)
    prediction["anomaly_map"]["thresholded_sha256"] = "0" * 64
    with pytest.raises(ComparativeAnalysisError, match="map hash is invalid"):
        _read_thresholded_map(tmp_path, prediction, (2, 2), Counter())


def test_float16_pixel_accumulator_matches_frozen_pixel_metrics() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    labels = [
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
        np.array([[0, 0], [1, 1]], dtype=np.uint8),
    ]
    scores = [
        np.array([[0.9, 0.2], [0.1, 0.0]], dtype=np.float16),
        np.array([[0.8, 0.3], [0.7, 0.4]], dtype=np.float16),
    ]
    predictions = [(score > np.float16(0.5)).astype(np.uint8) for score in scores]
    accumulator = Float16PixelAnalysisAccumulator()

    for label, score, prediction in zip(labels, scores, predictions, strict=True):
        accumulator.update(label, score, prediction)
    result = accumulator.result()

    flat_labels = np.concatenate([label.reshape(-1) for label in labels]).tolist()
    flat_predictions = np.concatenate(
        [prediction.reshape(-1) for prediction in predictions]
    ).tolist()
    frozen_f1 = binary_f1(flat_labels, flat_predictions, level="pixel")
    frozen_au_pro = au_pro(
        [label.tolist() for label in labels],
        [score.astype(float).tolist() for score in scores],
        fpr_limit=0.05,
    )

    assert result["confusion"] == {
        "true_positive": 2,
        "false_positive": 1,
        "true_negative": 4,
        "false_negative": 1,
    }
    assert _value(result, "pixel_precision") == pytest.approx(2 / 3)
    assert _value(result, "pixel_sensitivity") == pytest.approx(2 / 3)
    assert _value(result, "pixel_specificity") == pytest.approx(4 / 5)
    assert _value(result, "pixel_f1") == frozen_f1.value
    assert _value(result, "au_pro_0.05") == pytest.approx(
        frozen_au_pro.value, abs=1e-12
    )
    assert _value(result, "pixel_auroc_diagnostic") == pytest.approx(13 / 15)


def test_float16_pixel_auroc_treats_signed_zero_as_one_tie_group() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    accumulator = Float16PixelAnalysisAccumulator()
    accumulator.update(
        np.array([[1, 0], [1, 0]], dtype=np.uint8),
        np.array([[-0.0, 0.0], [-1.0, 1.0]], dtype=np.float16),
        np.zeros((2, 2), dtype=np.uint8),
    )

    result = accumulator.result()

    assert _value(result, "pixel_auroc_diagnostic") == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("labels", "predictions", "expected_flags"),
    [
        (
            [[1, 1], [1, 0]],
            [[1, 0], [0, 0]],
            ["under_localization"],
        ),
        (
            [[1, 0], [0, 0]],
            [[1, 1], [1, 0]],
            ["over_localization"],
        ),
        (
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
            ["missed_anomaly", "threshold_collapse"],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
            ],
            ["over_localization", "diffuse_false_positive_map"],
        ),
    ],
    ids=("under", "over", "missed-and-collapse", "diffuse"),
)
def test_localization_flags_are_assigned_by_documented_rules(
    labels: list[list[int]],
    predictions: list[list[int]],
    expected_flags: list[str],
) -> None:
    np = pytest.importorskip("numpy")
    label_array = np.asarray(labels, dtype=np.uint8)
    prediction_array = np.asarray(predictions, dtype=np.uint8)
    scores = np.arange(label_array.size, dtype=np.float16).reshape(label_array.shape)

    result = localization_diagnostics(label_array, scores, prediction_array)

    assert result["flags"] == expected_flags


@pytest.mark.parametrize(
    ("labels", "scores", "predictions", "message"),
    [
        ([[2, 0]], [[0.1, 0.2]], [[0, 0]], "labels must be binary"),
        ([[1, 0]], [[0.1, 0.2]], [[-1, 0]], "predictions must be binary"),
        ([[1, 0]], [[float("nan"), 0.2]], [[0, 0]], "scores must be finite"),
    ],
)
def test_localization_diagnostics_reject_malformed_raw_inputs(
    labels: list[list[int]],
    scores: list[list[float]],
    predictions: list[list[int]],
    message: str,
) -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(ValueError, match=message):
        localization_diagnostics(
            np.asarray(labels), np.asarray(scores), np.asarray(predictions)
        )


def test_threshold_diagnostics_preserve_float16_vs_stored_map_mismatches() -> None:
    diagnostics = _threshold_diagnostics(
        artifact={
            "thresholds": {"image": 0.5, "pixel": 0.5},
            "calibration": _calibration_payload("can"),
        },
        pixel_metrics={
            "continuous_score_distributions": {
                "ground_truth_anomaly_pixels": {
                    "status": "defined",
                    "maximum": 0.5,
                }
            }
        },
        sample_diagnostics=[
            {
                "label": 1,
                "continuous_maximum": 0.5,
                "predicted_positive_pixels": 1,
                "confusion": {"true_positive": 0},
                "float16_threshold_comparison": {
                    "mismatched_pixels": 2,
                    "stored_positive_float16_not_strictly_above": 2,
                    "float16_strictly_above_stored_negative": 0,
                },
            },
            {
                "label": 0,
                "continuous_maximum": 0.8,
                "predicted_positive_pixels": 0,
                "confusion": {"true_positive": 0},
                "float16_threshold_comparison": {
                    "mismatched_pixels": 1,
                    "stored_positive_float16_not_strictly_above": 0,
                    "float16_strictly_above_stored_negative": 1,
                },
            },
        ],
        flag_counts=Counter(),
        variant_state={"regular": Counter()},
    )

    assert diagnostics["float16_vs_stored_thresholded_map"] == {
        "mismatched_pixels": 3,
        "images_with_mismatch": 2,
        "stored_positive_float16_not_strictly_above": 2,
        "float16_strictly_above_stored_negative": 1,
        "authoritative_thresholded_evidence": "stored_png",
        "cause": (
            "Stored PNGs were created from pre-cast float32 maps; continuous "
            "evidence was subsequently stored as float16. Pixel confusion and "
            "F1 therefore use the stored PNG rather than re-thresholded float16."
        ),
    }
    assert (
        diagnostics["no_ground_truth_anomaly_pixel_score_strictly_exceeds_threshold"]
        is True
    )


@pytest.mark.parametrize(
    ("label", "patchcore_prediction", "efficientad_prediction", "expected"),
    [
        (1, 1, 1, "both_correct"),
        (1, 1, 0, "patchcore_only_correct"),
        (1, 0, 1, "efficientad_only_correct"),
        (1, 0, 0, "both_wrong"),
        (0, 0, 0, "both_correct"),
        (0, 0, 1, "patchcore_only_correct"),
        (0, 1, 0, "efficientad_only_correct"),
        (0, 1, 1, "both_wrong"),
    ],
)
def test_disagreement_classes_are_exhaustive(
    label: int,
    patchcore_prediction: int,
    efficientad_prediction: int,
    expected: str,
) -> None:
    assert (
        disagreement_bucket(label, patchcore_prediction, efficientad_prediction)
        == expected
    )


def _prediction(
    sample_id: str, *, label: int, image_prediction: int, anomaly_score: float
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "label": label,
        "image_prediction": image_prediction,
        "anomaly_score": anomaly_score,
    }


def test_pairing_preserves_exact_prediction_order_and_identifiers() -> None:
    patchcore = [
        _prediction(
            "can/test_public/good/000.png",
            label=0,
            image_prediction=0,
            anomaly_score=0.1,
        ),
        _prediction(
            "can/test_public/bad/001.png",
            label=1,
            image_prediction=1,
            anomaly_score=0.9,
        ),
    ]
    efficientad = [
        _prediction(
            "can/test_public/good/000.png",
            label=0,
            image_prediction=1,
            anomaly_score=0.8,
        ),
        _prediction(
            "can/test_public/bad/001.png",
            label=1,
            image_prediction=0,
            anomaly_score=0.2,
        ),
    ]

    result = pair_prediction_records(patchcore, efficientad, category="can", seed=42)

    assert [record["sample_id"] for record in result] == [
        "can/test_public/good/000.png",
        "can/test_public/bad/001.png",
    ]
    assert [record["ordinal"] for record in result] == [0, 1]
    assert [record["disagreement"] for record in result] == [
        "patchcore_only_correct",
        "patchcore_only_correct",
    ]


def test_pairing_fails_closed_when_prediction_order_differs() -> None:
    patchcore = [
        _prediction("first.png", label=0, image_prediction=0, anomaly_score=0.1),
        _prediction("second.png", label=1, image_prediction=1, anomaly_score=0.9),
    ]
    efficientad = [
        _prediction("second.png", label=1, image_prediction=1, anomaly_score=0.8),
        _prediction("first.png", label=0, image_prediction=0, anomaly_score=0.2),
    ]

    with pytest.raises(ComparativeAnalysisError, match="identity/order differs"):
        pair_prediction_records(patchcore, efficientad, category="can", seed=42)


def _panel_candidates() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for category_index, category in enumerate(OFFICIAL_CATEGORIES):
        for label in (0, 1):
            condition = "bad" if label else "good"
            for index in range(2):
                disagreement = DISAGREEMENT_ORDER[
                    (category_index + label + index) % len(DISAGREEMENT_ORDER)
                ]
                records.append(
                    {
                        "category": category,
                        "seed": PANEL_SELECTION_SEED,
                        "ordinal": index,
                        "sample_id": (
                            f"{category}/test_public/{condition}/{index:03d}.png"
                        ),
                        "label": label,
                        "disagreement": disagreement,
                        PATCHCORE: {
                            "anomaly_score": float(category_index + index),
                            "image_prediction": label,
                        },
                        EFFICIENTAD: {
                            "anomaly_score": float(100 - category_index - index),
                            "image_prediction": label,
                        },
                    }
                )
    return records


def test_panel_selection_is_deterministic_and_score_independent() -> None:
    records = _panel_candidates()
    baseline = select_panel_examples(records)
    changed_scores = deepcopy(records)
    for index, record in enumerate(changed_scores):
        patchcore = record[PATCHCORE]
        efficientad = record[EFFICIENTAD]
        assert isinstance(patchcore, dict)
        assert isinstance(efficientad, dict)
        patchcore["anomaly_score"] = float(10_000 + index)
        efficientad["anomaly_score"] = float(-10_000 - index)

    selected_after_score_change = select_panel_examples(list(reversed(changed_scores)))

    assert selected_after_score_change == baseline
    assert {selection["category"] for selection in baseline} == set(OFFICIAL_CATEGORIES)
    assert all("anomaly_score" not in selection for selection in baseline)


def test_manifest_comparison_accepts_only_absolute_local_root_redaction() -> None:
    committed = {
        "status": "completed",
        "message": (
            "missing '<local-dataset-root>\\rice\\train\\good\\193_regular.png'"
        ),
    }
    local = {
        "status": "completed",
        "message": (
            "missing 'D:\\datasets\\mvtec-ad-2\\rice\\train\\good\\193_regular.png'"
        ),
    }

    assert _manifest_differences(committed, local) == ["message"]


@pytest.mark.parametrize(
    "local_message",
    [
        "missing 'datasets\\mvtec-ad-2\\rice\\train\\good\\193_regular.png'",
        "a different unredacted failure message",
    ],
)
def test_manifest_comparison_rejects_unapproved_differences(
    local_message: str,
) -> None:
    committed = {
        "message": (
            "missing '<local-dataset-root>\\rice\\train\\good\\193_regular.png'"
        )
    }

    with pytest.raises(ComparativeAnalysisError, match="Local evidence differs"):
        _manifest_differences(committed, {"message": local_message})


def test_safe_join_rejects_absolute_traversal_and_empty_paths(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"

    assert (
        _safe_join(evidence_root, "cell/artifact.json", "artifact")
        == (evidence_root / "cell" / "artifact.json").resolve()
    )
    for unsafe in ("../outside.json", "C:\\outside.json", ""):
        with pytest.raises(ComparativeAnalysisError):
            _safe_join(evidence_root, unsafe, "artifact")


def test_output_roots_cannot_overlap_each_other_or_protected_inputs(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "frozen-evidence"
    report_output = tmp_path / "reports" / "phase4a"
    panel_output = tmp_path / "outputs" / "phase4a"

    _protect_input_roots(
        report_output=report_output,
        panel_output=panel_output,
        protected_roots=(evidence_root,),
    )
    with pytest.raises(ComparativeAnalysisError, match="must be distinct"):
        _protect_input_roots(
            report_output=report_output,
            panel_output=report_output,
            protected_roots=(evidence_root,),
        )
    with pytest.raises(ComparativeAnalysisError, match="read-only evidence roots"):
        _protect_input_roots(
            report_output=evidence_root / "analysis",
            panel_output=panel_output,
            protected_roots=(evidence_root,),
        )


def _perfect_frozen_summary(
    *, model: str, audit_sha256: str, benchmark_commit: str
) -> dict[str, Any]:
    metric_names = ("image_f1", "image_auroc", "pixel_f1", "au_pro_0.05")
    per_category: dict[str, Any] = {}
    for category in OFFICIAL_CATEGORIES:
        per_category[category] = {
            "per_seed": {
                metric: {str(seed): 1.0 for seed in PROTOCOL_SEEDS}
                for metric in metric_names
            },
            "across_seeds": {
                metric: {
                    "mean": 1.0,
                    "sample_standard_deviation": 0.0,
                    "count": len(PROTOCOL_SEEDS),
                }
                for metric in metric_names
            },
        }
    return {
        "model": model,
        "benchmark_git_commit": benchmark_commit,
        "dataset_audit_sha256": audit_sha256,
        "run_count": len(OFFICIAL_CATEGORIES) * len(PROTOCOL_SEEDS),
        "per_category": per_category,
        "overall": {
            metric: {
                "unweighted_category_mean": 1.0,
                "category_count": len(OFFICIAL_CATEGORIES),
            }
            for metric in metric_names
        },
    }


def _orchestration_bundle(
    *, model: str, audit_sha256: str, root: Path
) -> EvidenceBundle:
    benchmark_commit = "a" * 40
    artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_paths: dict[tuple[str, int], Path] = {}
    cells: dict[str, dict[str, object]] = {}
    for category in OFFICIAL_CATEGORIES:
        calibration = _calibration_payload(category)
        for seed in PROTOCOL_SEEDS:
            predictions = []
            for label, condition, score in (
                (1, "bad", 0.9),
                (0, "good", 0.1),
            ):
                predictions.append(
                    {
                        "sample_id": (
                            f"{category}/test_public/{condition}/000_regular.png"
                        ),
                        "label": label,
                        "anomaly_score": score,
                        "image_prediction": label,
                        "anomaly_map": {
                            "sha256": f"{seed + label:064x}",
                            "thresholded_sha256": f"{seed + label + 1:064x}",
                        },
                    }
                )
            artifacts[(category, seed)] = {
                "thresholds": {"image": 0.5, "pixel": 0.5},
                "calibration": deepcopy(calibration),
                "predictions": predictions,
                "category_metrics": {
                    metric: {"status": "defined", "value": 1.0}
                    for metric in (
                        "image_f1",
                        "image_auroc",
                        "pixel_f1",
                        "au_pro_0.05",
                    )
                },
            }
            artifact_paths[(category, seed)] = (
                root / model / category / f"seed-{seed}" / "artifact.json"
            )
            cells[f"{category}:{seed}"] = {
                "failure_history": [],
                "interruption_history": [],
            }
    spec = EvidenceSpec(
        name=model,
        committed_manifest_path=root / f"{model}-committed.json",
        evidence_root=root / model,
        protocol_path=root / f"{model}-protocol.yaml",
        protocol_loader=lambda _path: {},
        fingerprint=lambda _document: "unused",
        artifact_validator=lambda _artifact: None,
    )
    manifest = {
        "benchmark_git_commit": benchmark_commit,
        "dataset_audit_sha256": audit_sha256,
        "cells": cells,
    }
    return EvidenceBundle(
        spec=spec,
        committed_manifest=manifest,
        local_manifest=deepcopy(manifest),
        artifacts=artifacts,
        artifact_paths=artifact_paths,
        frozen_summary=_perfect_frozen_summary(
            model=model,
            audit_sha256=audit_sha256,
            benchmark_commit=benchmark_commit,
        ),
        provenance={
            "artifact_count": len(artifacts),
            "prediction_count": len(artifacts) * 2,
            "integrity_status": "passed",
            "committed_manifest_sha256": ("4" * 64 if model == PATCHCORE else "5" * 64),
        },
    )


def _orchestration_audit(
    audit_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, ...]], dict[str, Any]]:
    audit_index: dict[str, dict[str, Any]] = {}
    validation_ids: dict[str, tuple[str, ...]] = {}
    for category in OFFICIAL_CATEGORIES:
        good_id = f"{category}/test_public/good/000_regular.png"
        bad_id = f"{category}/test_public/bad/000_regular.png"
        mask_id = f"{category}/test_public/ground_truth/bad/000_regular_mask.png"
        audit_index[good_id] = {
            "category": category,
            "kind": "image",
            "height": 2,
            "width": 2,
            "sha256": "1" * 64,
        }
        audit_index[bad_id] = {
            "category": category,
            "kind": "image",
            "height": 2,
            "width": 2,
            "sha256": "2" * 64,
        }
        audit_index[mask_id] = {
            "category": category,
            "kind": "mask",
            "height": 2,
            "width": 2,
            "sha256": "3" * 64,
        }
        validation_ids[category] = tuple(
            f"{category}/validation/good/{index:03d}_regular.png" for index in range(19)
        )
    return (
        audit_index,
        validation_ids,
        {
            "sha256": audit_sha256,
            "status": "passed",
            "public_images_verified": len(OFFICIAL_CATEGORIES) * 2,
            "public_masks_verified": len(OFFICIAL_CATEGORIES),
            "private_assets_opened": 0,
        },
    )


def _orchestration_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "repository_root": Path.cwd(),
        "dataset_root": tmp_path / "dataset",
        "audit_report_path": tmp_path / "audit" / "audit.json",
        "patchcore_manifest_path": tmp_path / "committed-patch" / "manifest.json",
        "patchcore_evidence_root": tmp_path / "evidence-patch",
        "efficientad_manifest_path": tmp_path / "committed-efficient" / "manifest.json",
        "efficientad_evidence_root": tmp_path / "evidence-efficient",
        "report_output": tmp_path / "reports" / "phase4a",
        "panel_output": tmp_path / "outputs" / "phase4a" / "panels",
    }


def test_run_comparative_analysis_writes_deterministic_provenance_bound_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    audit_sha256 = "9" * 64
    bundles = {
        model: _orchestration_bundle(
            model=model, audit_sha256=audit_sha256, root=tmp_path / "bundles"
        )
        for model in (PATCHCORE, EFFICIENTAD)
    }
    monkeypatch.setattr(
        comparative_analysis,
        "_load_evidence",
        lambda spec, _root: bundles[spec.name],
    )
    monkeypatch.setattr(
        comparative_analysis,
        "_load_and_verify_public_audit",
        lambda _path, _root, expected: _orchestration_audit(expected),
    )

    def read_continuous(
        _run_dir: Path,
        prediction: dict[str, Any],
        verification_counts: Counter[str],
    ) -> tuple[Any, Path]:
        verification_counts["continuous_maps"] += 1
        if "/bad/" in prediction["sample_id"]:
            value = np.array([[0.9, 0.1], [0.1, 0.1]], dtype=np.float16)
        else:
            value = np.full((2, 2), 0.1, dtype=np.float16)
        return value, Path("unused.tiff")

    def read_thresholded(
        _run_dir: Path,
        prediction: dict[str, Any],
        _shape: tuple[int, int],
        verification_counts: Counter[str],
    ) -> tuple[Any, Path]:
        verification_counts["thresholded_maps"] += 1
        value = np.zeros((2, 2), dtype=bool)
        if "/bad/" in prediction["sample_id"]:
            value[0, 0] = True
        return value, Path("unused.png")

    def load_mask(
        sample_id: str,
        _shape: tuple[int, int],
        _dataset_root: Path,
        _audit_index: dict[str, dict[str, Any]],
    ) -> Any:
        value = np.zeros((2, 2), dtype=np.uint8)
        if "/bad/" in sample_id:
            value[0, 0] = 1
        return value

    def render_panel(*, output_path: Path, sample_id: str, **_kwargs: Any) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(sample_id.encode("utf-8"))

    monkeypatch.setattr(comparative_analysis, "_read_continuous_map", read_continuous)
    monkeypatch.setattr(comparative_analysis, "_read_thresholded_map", read_thresholded)
    monkeypatch.setattr(comparative_analysis, "_load_public_mask", load_mask)
    monkeypatch.setattr(comparative_analysis, "_render_panel", render_panel)
    paths = _orchestration_paths(tmp_path)

    first = run_comparative_analysis(**paths)
    report_names = (
        "analysis-manifest.json",
        "analysis-summary.json",
        "panel-index.json",
        "per-image-analysis.jsonl",
    )
    first_report_bytes = {
        name: (paths["report_output"] / name).read_bytes() for name in report_names
    }
    second = run_comparative_analysis(**paths)

    assert second == first
    assert {
        name: (paths["report_output"] / name).read_bytes() for name in report_names
    } == first_report_bytes
    assert first["scope"]["new_training"] is False
    assert first["scope"]["new_inference"] is False
    assert first["scope"]["threshold_tuning"] is False
    assert first["provenance"]["map_hashes_verified"] == {
        "continuous": 96,
        "thresholded": 96,
    }
    assert first["provenance"]["frozen_summary_reproduction"] == {
        PATCHCORE: {
            "status": "passed",
            "metrics": ["image_f1", "image_auroc", "pixel_f1", "au_pro_0.05"],
            "identity_fields": ["benchmark_git_commit", "dataset_audit_sha256"],
            "run_count": 24,
            "numeric_comparisons": 200,
        },
        EFFICIENTAD: {
            "status": "passed",
            "metrics": ["image_f1", "image_auroc", "pixel_f1", "au_pro_0.05"],
            "identity_fields": ["benchmark_git_commit", "dataset_audit_sha256"],
            "run_count": 24,
            "numeric_comparisons": 200,
        },
    }


def test_run_comparative_analysis_stops_on_cross_model_audit_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles = {
        PATCHCORE: _orchestration_bundle(
            model=PATCHCORE, audit_sha256="1" * 64, root=tmp_path / "bundles"
        ),
        EFFICIENTAD: _orchestration_bundle(
            model=EFFICIENTAD, audit_sha256="2" * 64, root=tmp_path / "bundles"
        ),
    }
    monkeypatch.setattr(
        comparative_analysis,
        "_load_evidence",
        lambda spec, _root: bundles[spec.name],
    )

    def unexpected_audit_read(*_args: Any) -> Any:
        raise AssertionError("audit must not be read after provenance mismatch")

    monkeypatch.setattr(
        comparative_analysis,
        "_load_and_verify_public_audit",
        unexpected_audit_read,
    )
    paths = _orchestration_paths(tmp_path)

    with pytest.raises(ComparativeAnalysisError, match="different dataset audits"):
        run_comparative_analysis(**paths)

    assert not paths["report_output"].exists()
    assert not paths["panel_output"].exists()
