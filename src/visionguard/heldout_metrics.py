"""Separately reviewed real-ID interface; numerical reducers copied from frozen B1.

No B1 global is modified and no real ID is renamed synthetic. Numerical function
bodies are AST-equivalence tested against B1, except the evidence-interface tag.
Input provenance and authorization are enforced by heldout admission/runner.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

import numpy as np

from visionguard.analysis_metrics import (
    Float16PixelAnalysisAccumulator,
    classification_metrics,
    describe_scores,
)
from visionguard.triage import ALLOWED_SEEDS, triage_same_seed
from visionguard.triage_metrics import calculate_triage_metrics
from visionguard.visa import CATEGORIES
from visionguard.visa_evaluator import ModelSpec, finite_scalar, require


@dataclass(frozen=True)
class Sample:
    sample_id: str
    category: str
    label: int
    mask: np.ndarray


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    spec: ModelSpec
    score: float
    restored_map: np.ndarray
    thresholded_map: np.ndarray | None = None


def validate_sample(sample: Sample) -> None:
    require(type(sample) is Sample, "Real-ID sample interface required")
    require(type(sample.sample_id) is str, "String identifier required")
    path = PurePosixPath(sample.sample_id)
    require(
        not path.is_absolute()
        and str(path) == sample.sample_id
        and "\\" not in sample.sample_id
        and ":" not in sample.sample_id
        and not set(path.parts) & {".", ".."},
        "Unsafe genuine ID",
    )
    require(
        len(path.parts) == 5
        and path.parts[:3] == (sample.category, "Data", "Images")
        and path.parts[3] in ("Normal", "Anomaly"),
        "Wrong ID/category membership",
    )
    require(
        sample.category in CATEGORIES
        and type(sample.label) is int
        and sample.label in (0, 1),
        "Invalid category/label",
    )
    require(
        path.parts[3] == ("Anomaly" if sample.label else "Normal"), "ID/label conflict"
    )
    require(
        type(sample.mask) is np.ndarray
        and sample.mask.ndim == 2
        and sample.mask.size > 0
        and sample.mask.dtype in (np.dtype("uint8"), np.dtype("bool")),
        "Nonempty original binary mask required",
    )
    require(
        bool(np.isin(sample.mask, (0, 1)).all())
        and bool(sample.mask.any()) == bool(sample.label),
        "Mask/label conflict",
    )


def validate_prediction(
    sample: Sample, prediction: Prediction, spec: ModelSpec
) -> None:
    require(
        type(prediction) is Prediction and prediction.spec == spec,
        "Wrong model/cell/hash/threshold",
    )
    require(prediction.sample_id == sample.sample_id, "Missing/reordered image")
    require(spec.category == sample.category, "Category mismatch")
    finite_scalar(prediction.score, "image score")
    array = prediction.restored_map
    require(
        type(array) is np.ndarray
        and array.dtype in (np.dtype("float32"), np.dtype("float16"))
        and array.ndim == 2
        and array.shape == sample.mask.shape,
        "Original-coordinate float map/mask dimensions mismatch",
    )
    if array.dtype == np.float16:
        binary = prediction.thresholded_map
        require(
            type(binary) is np.ndarray
            and binary.dtype == np.bool_
            and binary.shape == array.shape,
            "Replay needs preserved float32-derived binary map",
        )
    else:
        require(
            prediction.thresholded_map is None,
            "Cannot override native float32 threshold semantics",
        )
    require(bool(np.isfinite(array).all()), "Nonfinite map")
    require(bool((np.abs(array) <= np.finfo(np.float16).max).all()), "Float16 overflow")


def evaluate_cell(
    samples: Iterable[Sample],
    patchcore: Iterable[Prediction],
    efficientad: Iterable[Prediction],
    specs: tuple[ModelSpec, ModelSpec],
) -> dict:
    """Pure streamed paired reducer, no file paths, labels in decisions or inference.

    Predictions are validated before use; no unmatched item is silently dropped.
    Memory is bounded by one paired image plus score/decision rows and fixed
    float16 histograms. Sample identifiers remain in canonical supplied order.
    """
    from itertools import zip_longest

    pc, ea = specs
    require(
        pc.model == "patchcore" and ea.model == "efficientad", "Model order mismatch"
    )
    require(
        pc.category == ea.category and pc.seed == ea.seed, "Cross-category/seed pair"
    )
    acc = {m: Float16PixelAnalysisAccumulator() for m in ("patchcore", "efficientad")}
    scores: dict[str, list[float]] = {m: [] for m in acc}
    labels, triages, rows = [], [], []
    seen = set()
    for sample, p, e in zip_longest(samples, patchcore, efficientad):
        require(
            sample is not None and p is not None and e is not None,
            "Missing image/prediction",
        )
        validate_sample(sample)
        require(sample.sample_id not in seen, "Duplicate image identifier")
        seen.add(sample.sample_id)
        for prediction, spec in ((p, pc), (e, ea)):
            validate_prediction(sample, prediction, spec)
            scores[spec.model].append(float(prediction.score))
            # Match Phase4D-A: float32 strict thresholded map, float16 ranking map.
            binary = (
                prediction.restored_map > spec.pixel_threshold
                if prediction.thresholded_map is None
                else prediction.thresholded_map
            )
            acc[spec.model].update(
                sample.mask, prediction.restored_map.astype(np.float16), binary
            )
        outcome = triage_same_seed(
            p.score,
            pc.image_threshold,
            e.score,
            ea.image_threshold,
            patchcore_seed=pc.seed,
            efficientad_seed=ea.seed,
        )
        labels.append(sample.label)
        triages.append(outcome)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "original_shape": list(sample.mask.shape),
                "patchcore_score": float(p.score),
                "efficientad_score": float(e.score),
                "patchcore_anomalous": outcome.patchcore_decision,
                "efficientad_anomalous": outcome.efficientad_decision,
                "triage": outcome.triage_decision.value,
            }
        )
    require(bool(rows), "Empty cell cannot be completed")
    models = {}
    for spec in specs:
        model = spec.model
        models[model] = {
            "model_sha256": spec.model_sha256,
            "image_threshold": spec.image_threshold,
            "pixel_threshold": spec.pixel_threshold,
            "image": classification_metrics(
                labels,
                [int(s > spec.image_threshold) for s in scores[model]],
                scores[model],
            ),
            "pixel": acc[model].result(),
            "image_score_distributions": {
                name: describe_scores(
                    [
                        score
                        for label, score in zip(labels, scores[model], strict=True)
                        if label == value
                    ]
                )
                if value in labels
                else {"count": 0, "status": "undefined", "reason": "class_absent"}
                for name, value in (("normal_images", 0), ("anomalous_images", 1))
            },
        }
    reduced = calculate_triage_metrics(triages, [bool(label) for label in labels])
    triage_payload = asdict(reduced)
    triage_payload["rates"] = {
        name: {
            **asdict(rate),
            "value": rate.value,
            "undefined_reason": rate.undefined_reason,
        }
        for name, rate in reduced.rates.items()
    }
    return {
        "schema_version": 1,
        "evidence_class": "real_id_interface",
        "category": pc.category,
        "seed": pc.seed,
        "sample_count": len(rows),
        "membership_sha256": hashlib.sha256(
            json.dumps([r["sample_id"] for r in rows], separators=(",", ":")).encode()
        ).hexdigest(),
        "models": models,
        "triage": triage_payload,
        "images": rows,
    }


def summarize(values: list[float | None]) -> dict:
    """Strict macro: any undefined group makes macro undefined, never omitted."""
    defined = [v for v in values if v is not None]
    for value in defined:
        finite_scalar(value, "aggregate metric")
    return {
        "group_count": len(values),
        "defined_group_count": len(defined),
        "mean": statistics.fmean(defined)
        if len(defined) == len(values) and values
        else None,
        "sample_std": statistics.stdev(defined)
        if len(defined) == len(values) and len(values) > 1
        else None,
        "undefined_reason": None
        if len(defined) == len(values) and values
        else "undefined_or_missing_group",
    }


def aggregate_matrix(cells: Iterable[dict]) -> dict:
    """Exact 36 same-seed pairs; category macro then descriptive three-seed stats.

    No pooling raw scores across categories/model scales/seeds. Pooled binary
    confusion and triage counts are secondary, never pooled ranking scores.
    """
    indexed = {}
    for cell in cells:
        key = (cell["category"], cell["seed"])
        require(type(cell["seed"]) is int, "Integer seed required")
        require(key not in indexed, "Duplicate category/seed cell")
        require(
            cell["evidence_class"] == "real_id_interface",
            "Controlled real-ID evidence interface required",
        )
        indexed[key] = cell
    expected = {(c, s) for c in CATEGORIES for s in ALLOWED_SEEDS}
    require(set(indexed) == expected, "Full 36 paired cells required; no substitutions")
    for category in CATEGORIES:
        reference = indexed[category, 42]
        for seed in ALLOWED_SEEDS:
            cell = indexed[category, seed]
            rows = cell["images"]
            ids = [row["sample_id"] for row in rows]
            require(
                len(rows) == cell["sample_count"] and len(set(ids)) == len(ids),
                "Invalid image membership/count",
            )
            digest = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            require(
                digest == cell["membership_sha256"], "Cell membership hash mismatch"
            )
            require(
                cell["membership_sha256"] == reference["membership_sha256"]
                and [(r["sample_id"], r["label"], r["original_shape"]) for r in rows]
                == [
                    (r["sample_id"], r["label"], r["original_shape"])
                    for r in reference["images"]
                ],
                "Seeds must use identical ordered membership/truth/dimensions",
            )
            require(
                cell["triage"]["sample_count"] == len(rows)
                and sum(cell["triage"]["decision_counts"].values()) == len(rows),
                "Triage dropped samples",
            )
    output = {
        "schema_version": 1,
        "evidence_class": "real_id_interface",
        "seeds": {},
    }
    for seed in ALLOWED_SEEDS:
        group = [indexed[c, seed] for c in CATEGORIES]
        model_results, pooled_models = {}, {}
        metric_fields = (
            ("image", "image_auroc"),
            ("image", "image_f1"),
            ("image", "sensitivity"),
            ("image", "specificity"),
            ("image", "precision"),
            ("pixel", "au_pro_0.05"),
            ("pixel", "pixel_f1"),
            ("pixel", "pixel_precision"),
            ("pixel", "pixel_sensitivity"),
            ("pixel", "pixel_specificity"),
            ("pixel", "pixel_auroc_diagnostic"),
        )
        for model in ("patchcore", "efficientad"):
            model_results[model] = {
                metric: summarize(
                    [cell["models"][model][level][metric]["value"] for cell in group]
                )
                for level, metric in metric_fields
            }
            pooled_models[model] = {
                level: pooled_confusion(
                    [cell["models"][model][level]["confusion"] for cell in group]
                )
                for level in ("image", "pixel")
            }
        triage_results = {
            name: summarize([cell["triage"]["rates"][name]["value"] for cell in group])
            for name in group[0]["triage"]["rates"]
        }
        pooled_rates = {}
        for name in group[0]["triage"]["rates"]:
            numerator = sum(
                cell["triage"]["rates"][name]["numerator"] for cell in group
            )
            denominator = sum(
                cell["triage"]["rates"][name]["denominator"] for cell in group
            )
            pooled_rates[name] = {
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator if denominator else None,
                "undefined_reason": None if denominator else "zero_denominator",
            }
        deltas = {
            metric: summarize(
                [
                    None
                    if any(
                        cell["models"][m][level][metric]["value"] is None
                        for m in model_results
                    )
                    else cell["models"]["patchcore"][level][metric]["value"]
                    - cell["models"]["efficientad"][level][metric]["value"]
                    for cell in group
                ]
            )
            for level, metric in metric_fields
        }
        pooled_triage_counts = {
            field: sum_nested_counts([cell["triage"][field] for cell in group])
            for field in (
                "decision_counts",
                "class_decision_counts",
                "disagreement_counts",
                "class_disagreement_counts",
            )
        }
        output["seeds"][str(seed)] = {
            "category_macro": model_results,
            "triage_category_macro": triage_results,
            "triage_pooled_rates": pooled_rates,
            "triage_pooled_counts": pooled_triage_counts,
            "model_pooled_confusion": pooled_models,
            "paired_patchcore_minus_efficientad": deltas,
        }
    output["descriptive_across_three_seeds"] = {
        model: {
            metric: summarize(
                [
                    output["seeds"][str(s)]["category_macro"][model][metric]["mean"]
                    for s in ALLOWED_SEEDS
                ]
            )
            for metric in output["seeds"]["42"]["category_macro"][model]
        }
        for model in ("patchcore", "efficientad")
    }
    output["triage_descriptive_across_three_seeds"] = {
        name: summarize(
            [
                output["seeds"][str(s)]["triage_category_macro"][name]["mean"]
                for s in ALLOWED_SEEDS
            ]
        )
        for name in output["seeds"]["42"]["triage_category_macro"]
    }
    output["paired_descriptive_across_three_seeds"] = {
        name: summarize(
            [
                output["seeds"][str(s)]["paired_patchcore_minus_efficientad"][name][
                    "mean"
                ]
                for s in ALLOWED_SEEDS
            ]
        )
        for name in output["seeds"]["42"]["paired_patchcore_minus_efficientad"]
    }
    return output


def sum_nested_counts(documents: list[dict]) -> dict:
    """Sum identically shaped count trees; no decisions or classes are dropped."""
    keys = set(documents[0])
    require(all(set(d) == keys for d in documents), "Count schema mismatch")
    result = {}
    for key in sorted(keys):
        values = [d[key] for d in documents]
        if all(type(v) is dict for v in values):
            result[key] = sum_nested_counts(values)
        else:
            require(all(type(v) is int and v >= 0 for v in values), "Invalid count")
            result[key] = sum(values)
    return result


def pooled_confusion(documents: list[dict]) -> dict:
    """Micro binary rates from counts only, NOT scores or pooled ranking."""
    counts = sum_nested_counts(documents)
    tp, fp, tn, fn = (
        counts[k]
        for k in ("true_positive", "false_positive", "true_negative", "false_negative")
    )
    pairs = {
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "precision": (tp, tp + fp),
        "f1": (2 * tp, 2 * tp + fp + fn),
    }
    return {
        "confusion": counts,
        "rates": {
            k: {
                "numerator": n,
                "denominator": d,
                "value": n / d if d else None,
                "undefined_reason": None if d else "zero_denominator",
            }
            for k, (n, d) in pairs.items()
        },
    }
