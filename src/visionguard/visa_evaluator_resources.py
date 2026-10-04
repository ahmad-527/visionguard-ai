"""Capacity assumptions from committed aggregate counts, NEVER test enumeration."""

from __future__ import annotations

from visionguard.visa import CATEGORIES
from visionguard.visa_evaluator import require

SAFETY_FLOOR = 20 * 1024**3
RUN_BUDGET = 128 * 1024**3


def estimate(counts: dict, height: int = 1536, width: int = 1536) -> dict:
    """Uncompressed upper scenario, not measured test dimensions or latency.

    Six model/seed passes. TIFF float16=2 bytes/pixel, binary PNG pessimistically
    budgeted as uint8=1; 64 KiB per pair of files and 1 MiB per model-cell for
    rows/hash manifests. Inputs/models already exist, not copied. Scratch plus
    reducer/object overhead requires the separate 20 GiB safety reserve.
    """
    require(set(counts) == set(CATEGORIES), "Exact category counts required")
    require(
        type(height) is int and type(width) is int and min(height, width) > 0,
        "Positive dimension assumption required",
    )
    require(
        all(type(n) is int and n > 0 for n in counts.values()),
        "Positive counts required",
    )
    images = sum(counts.values())
    predictions = images * 6
    map_bytes = predictions * height * width * 3
    overhead = predictions * 65536 + 72 * 1024**2
    retained = map_bytes + overhead
    require(retained <= RUN_BUDGET, "Assumed outputs exceed frozen retention budget")
    return {
        "evidence_class": "calculated_capacity_scenario_not_inference_measurement",
        "counts_source": "previously_committed_aggregate_audit_only",
        "sample_count": images,
        "model_image_calls": predictions,
        "assumed_maximum_height": height,
        "assumed_maximum_width": width,
        "dimensions_measured": False,
        "map_payload_bytes": map_bytes,
        "file_and_manifest_allowance_bytes": overhead,
        "estimated_retained_bytes": retained,
        "run_budget_bytes": RUN_BUDGET,
        "safety_floor_bytes": SAFETY_FLOOR,
        "required_free_bytes": RUN_BUDGET + SAFETY_FLOOR,
        "latency_measured": False,
        "gpu_workloads_at_once": 1,
        "future_preflight": (
            "separate_B2_authorization_required_actual_dimensions_"
            "and_disk_budget_checked_before_scoring"
        ),
    }


def check_capacity(free_bytes: int, estimate_document: dict) -> None:
    """Fail before output writes; do not free disk or inspect input assets."""
    require(
        type(free_bytes) is int
        and free_bytes >= estimate_document["required_free_bytes"],
        "Insufficient output budget plus safety floor",
    )
