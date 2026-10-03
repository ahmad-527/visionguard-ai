"""Built-in artificial fixtures only; no external image, label or prediction I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visionguard.visa import CATEGORIES
from visionguard.visa_evaluator import (
    ModelSpec,
    Prediction,
    SyntheticSample,
    aggregate_matrix,
    evaluate_synthetic_cell,
)
from visionguard.visa_evaluator_protocol import verify_readiness_freeze


def fixture(category: str = "candle", seed: int = 42) -> tuple:
    """Four 2x2 samples, all four binary combinations, including equality.

    Thresholds and hashes below are fabricated synthetic constants, NEVER
    real calibration/model identities. No development outputs enter these tests.
    """
    pc = ModelSpec("patchcore", category, seed, "a" * 64, 2.0, 2.0)
    ea = ModelSpec("efficientad", category, seed, "b" * 64, 2.0, 2.0)
    samples, pcs, eas = [], [], []
    for i, (label, pscore, escore) in enumerate(
        ((0, 1.0, 1.0), (0, 2.0, 3.0), (1, 3.0, 2.0), (1, 4.0, 4.0))
    ):
        mask = np.zeros((2, 2), dtype=np.uint8)
        if label:
            mask[0, 0] = 1
        sample = SyntheticSample(f"synthetic:{category}:{i}", category, label, mask)
        pmap = mask.astype(np.float32) * 3
        emap = mask.astype(np.float32) * (3 if i == 3 else 0)
        samples.append(sample)
        pcs.append(Prediction(sample.sample_id, pc, pscore, pmap))
        eas.append(Prediction(sample.sample_id, ea, escore, emap))
    return samples, pcs, eas, (pc, ea)


def synthetic_report() -> dict:
    """All 36 canonical paired cells with identical intentionally artificial data."""
    cells = [
        evaluate_synthetic_cell(*fixture(c, s))
        for c in CATEGORIES
        for s in (42, 123, 2026)
    ]
    return {
        "evidence_class": "synthetic_engineering_only",
        "real_test_asset_access": False,
        "real_test_performance_evaluations": 0,
        "final_test_lock": "closed",
        "phase4d_b2_started": False,
        "cells": cells,
        "aggregate": aggregate_matrix(cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    snapshot = verify_readiness_freeze(args.repository)
    report = synthetic_report()
    report["evaluator_implementation_fingerprint"] = snapshot["fingerprint"]
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
