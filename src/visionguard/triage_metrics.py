"""Predeclared three-way metrics. Phase 4B uses synthetic fixtures only.

This pure reducer has no dataset loader or evaluation runner. A future authorized
caller must provide one explicit category/seed or pooled-category/seed group.
REVIEW never receives a binary correctness label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from visionguard.triage import Decision, TriageError, TriageResult


@dataclass(frozen=True)
class Rate:
    """Auditable numerator/denominator; undefined rates retain an explicit reason."""

    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    @property
    def undefined_reason(self) -> str | None:
        return None if self.denominator else "zero_denominator"


@dataclass(frozen=True)
class TriageMetrics:
    """Counts by true class, including both disagreement directions."""

    sample_count: int
    decision_counts: dict[str, int]
    class_decision_counts: dict[str, dict[str, int]]
    disagreement_counts: dict[str, int]
    class_disagreement_counts: dict[str, dict[str, int]]
    rates: dict[str, Rate]


def calculate_triage_metrics(
    results: Sequence[TriageResult], anomalous_labels: Sequence[bool]
) -> TriageMetrics:
    """Reduce paired outcomes and explicit class labels, preserving abstentions.

    This module is separate from decision logic. Length mismatches and non-bool
    labels fail closed; empty inputs return zero counts and undefined rates.
    """
    if len(results) != len(anomalous_labels):
        raise TriageError("Results and labels must have identical lengths")
    counts = {
        label: dict.fromkeys((d.value for d in Decision), 0)
        for label in ("normal", "anomalous")
    }
    directions = (
        "patchcore_anomalous_efficientad_normal",
        "patchcore_normal_efficientad_anomalous",
    )
    disagreements = {label: dict.fromkeys(directions, 0) for label in counts}
    for result, label in zip(results, anomalous_labels, strict=True):
        if type(result) is not TriageResult or type(label) is not bool:
            raise TriageError("Expected TriageResult and explicit boolean class label")
        result.__post_init__()
        class_name = "anomalous" if label else "normal"
        counts[class_name][result.triage_decision.value] += 1
        if result.triage_decision is Decision.REVIEW:
            direction = directions[0 if result.patchcore_decision else 1]
            disagreements[class_name][direction] += 1
    total = {d.value: sum(c[d.value] for c in counts.values()) for d in Decision}
    normal, anomaly = counts["normal"], counts["anomalous"]
    n_normal, n_anomaly = sum(normal.values()), sum(anomaly.values())
    n_samples = n_normal + n_anomaly
    automatic = total["PASS"] + total["REJECT"]
    rates = {
        "review_rate": Rate(total["REVIEW"], n_samples),
        "automatic_decision_coverage": Rate(automatic, n_samples),
        "selective_accuracy": Rate(normal["PASS"] + anomaly["REJECT"], automatic),
        "anomaly_pass_through_rate": Rate(anomaly["PASS"], n_anomaly),
        "normal_reject_rate": Rate(normal["REJECT"], n_normal),
        "anomaly_review_capture": Rate(anomaly["REVIEW"], n_anomaly),
        "normal_review_rate": Rate(normal["REVIEW"], n_normal),
        "reject_precision": Rate(anomaly["REJECT"], total["REJECT"]),
        "pass_negative_predictive_value": Rate(normal["PASS"], total["PASS"]),
    }
    return TriageMetrics(
        n_samples,
        total,
        counts,
        {d: sum(c[d] for c in disagreements.values()) for d in directions},
        disagreements,
        rates,
    )
