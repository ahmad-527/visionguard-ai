import copy
from pathlib import Path

import pytest

from visionguard.heldout_contract import build_freeze, context
from visionguard.visa_protocol import canonical_fingerprint

REPO = Path(__file__).resolve().parents[1]


def test_original_anchors_unchanged():
    ctx = context(REPO)
    assert (
        ctx["activation"]["b1_fingerprint"]
        == "0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49"
    )
    assert (
        ctx["activation"]["b2_readiness_fingerprint"]
        == "1b44805040f211fdf75bd54561bae1b61f0fe2912d25ff734c355ff1202ba6cc"
    )
    assert len(ctx["specs"]) == 72 and sum(ctx["counts"].values()) == 2162
    assert ctx["audit"]["final_test_lock"] == "closed"


@pytest.mark.parametrize(
    "field",
    [
        "authorization",
        "admission",
        "inference",
        "metrics",
        "publication",
        "restart",
        "monitoring",
        "retention",
        "scope",
        "run_budget_bytes",
        "safety_reserve_bytes",
        "human_access_history",
        "independent_reservation",
    ],
)
def test_scientific_operational_changes_alter_fingerprint(field):
    frozen = build_freeze(REPO)
    changed = copy.deepcopy(frozen["document"])
    changed["contract"][field] = "ALTERED"
    assert canonical_fingerprint(changed) != frozen["fingerprint"]


def test_source_changes_alter_fingerprint():
    frozen = build_freeze(REPO)
    changed = copy.deepcopy(frozen["document"])
    key = next(iter(changed["source_sha256"]))
    changed["source_sha256"][key] = "0" * 64
    assert canonical_fingerprint(changed) != frozen["fingerprint"]
