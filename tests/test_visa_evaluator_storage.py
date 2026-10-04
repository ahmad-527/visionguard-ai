"""Synthetic retention/replay faults, never real dataset files."""

from dataclasses import replace

import numpy as np
import pytest

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import EvaluationError, evaluate_synthetic_cell
from visionguard.visa_evaluator_storage import (
    canonical_bytes,
    publish_synthetic,
    replay_synthetic,
)
from visionguard.visa_evaluator_synthetic import fixture


def test_exact_replay_preserves_prequantization_binary_and_immutable_output(tmp_path):
    s, p, e, specs = fixture()
    p[2] = replace(p[2], restored_map=np.full((2, 2), 2.0001, dtype=np.float32))
    expected = evaluate_synthetic_cell(s, p, e, specs)
    target = tmp_path / "artificial"
    publish_synthetic(target, s, p, e, specs)
    digest = sha256_file(target / "synthetic-manifest.json")
    for _ in range(2):
        actual = replay_synthetic(target, digest, specs, expected["membership_sha256"])
        assert canonical_bytes(actual) == canonical_bytes(expected)
    with pytest.raises(EvaluationError, match="Immutable"):
        publish_synthetic(target, s, p, e, specs)
    with pytest.raises(EvaluationError, match="membership"):
        replay_synthetic(target, digest, specs, "0" * 64)
    with pytest.raises(EvaluationError, match="models"):
        replay_synthetic(
            target,
            digest,
            (specs[0], replace(specs[1], seed=123)),
            expected["membership_sha256"],
        )
    (target / "000000-patchcore.tiff").write_bytes(b"corrupt synthetic map")
    with pytest.raises(EvaluationError, match="corruption"):
        replay_synthetic(target, digest, specs, expected["membership_sha256"])


def test_invalid_input_no_partial_output(tmp_path):
    s, p, e, specs = fixture()
    with pytest.raises(EvaluationError):
        publish_synthetic(tmp_path / "none", s, p[:-1], e, specs)
    assert not (tmp_path / "none").exists()
