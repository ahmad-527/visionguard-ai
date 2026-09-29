"""Model-free image triage; no calibration, I/O, labels, or category routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

ALLOWED_SEEDS = (42, 123, 2026)


class TriageError(ValueError):
    """Missing, malformed, or incompatible triage evidence."""


class Decision(Enum):
    """Three distinct outcomes, deliberately without implicit binary conversion."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"

    def __bool__(self) -> bool:
        raise TypeError("A triage decision is not binary; compare explicit outcomes")


@dataclass(frozen=True)
class TriageResult:
    """Each model's anomalous flag plus the consistent three-way decision."""

    patchcore_decision: bool
    efficientad_decision: bool
    triage_decision: Decision

    def __post_init__(self) -> None:
        if (
            type(self.patchcore_decision) is not bool
            or type(self.efficientad_decision) is not bool
        ):
            raise TriageError("Model decisions must be explicit boolean flags")
        expected = _decision(self.patchcore_decision, self.efficientad_decision)
        if self.triage_decision is not expected:
            raise TriageError("Triage outcome conflicts with the frozen truth table")

    def __bool__(self) -> bool:
        raise TypeError("A triage result is not a binary prediction")


def _decision(patchcore: bool, efficientad: bool) -> Decision:
    if patchcore != efficientad:
        return Decision.REVIEW
    return Decision.REJECT if patchcore else Decision.PASS


def _finite_scalar(value: object, name: str) -> None:
    # Exact built-in types prevent string, bool, array, and custom coercions.
    # Preserve integers without lossy float conversion, including large integers.
    if type(value) not in (int, float):
        raise TriageError(f"{name} must be a finite built-in int or float")
    if type(value) is float and not math.isfinite(value):
        raise TriageError(f"{name} must be finite")


def triage(
    patchcore_score: float,
    patchcore_threshold: float,
    efficientad_score: float,
    efficientad_threshold: float,
) -> TriageResult:
    """Apply strict score > own threshold; equality is normal for each model.

    Inputs must already be frozen, paired evidence on each model's own scale.
    Finite negative values are valid. This primitive cannot certify provenance;
    callers must validate sample identity and use ``triage_same_seed`` for runs.
    """
    for name, value in (
        ("patchcore_score", patchcore_score),
        ("patchcore_threshold", patchcore_threshold),
        ("efficientad_score", efficientad_score),
        ("efficientad_threshold", efficientad_threshold),
    ):
        _finite_scalar(value, name)
    patchcore = patchcore_score > patchcore_threshold
    efficientad = efficientad_score > efficientad_threshold
    return TriageResult(patchcore, efficientad, _decision(patchcore, efficientad))


def validate_seed_pair(patchcore_seed: int, efficientad_seed: int) -> int:
    """Reject missing, unapproved, coerced, or cross-seed pairs."""
    if any(
        type(seed) is not int or seed not in ALLOWED_SEEDS
        for seed in (patchcore_seed, efficientad_seed)
    ):
        raise TriageError(f"Both seeds must be integers in {ALLOWED_SEEDS}")
    if patchcore_seed != efficientad_seed:
        raise TriageError("Both model outputs must have the same seed")
    return patchcore_seed


def triage_same_seed(
    patchcore_score: float,
    patchcore_threshold: float,
    efficientad_score: float,
    efficientad_threshold: float,
    *,
    patchcore_seed: int,
    efficientad_seed: int,
) -> TriageResult:
    """Seed-checked entry point; no missing-seed default or best-seed selection."""
    validate_seed_pair(patchcore_seed, efficientad_seed)
    return triage(
        patchcore_score, patchcore_threshold, efficientad_score, efficientad_threshold
    )
