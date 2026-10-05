import copy
from pathlib import Path

import pytest

from visionguard.accounting_successor import build_freeze, verify_freeze
from visionguard.operational_successor import verify_freeze as verify_previous
from visionguard.visa_protocol import canonical_fingerprint

REPO = Path(__file__).resolve().parents[1]


def test_v4_chain_preserves_exact_predecessor_and_scientific_inputs():
    old = verify_previous(REPO)
    new = verify_freeze(REPO)
    assert new == build_freeze(REPO)
    assert new["document"]["schema_version"] == 4
    assert new["fingerprint"] != old["fingerprint"]
    for field in ("contract", "models", "environment"):
        assert new["document"][field] == old["document"][field]
    assert len(new["document"]["models"]) == 72


@pytest.mark.parametrize(
    "field",
    [
        "operational_protocol",
        "contract",
        "models",
        "source_sha256",
        "historical_fingerprints",
    ],
)
def test_v4_changes_cannot_reuse_authorization_fingerprint(field):
    original = build_freeze(REPO)
    changed = copy.deepcopy(original["document"])
    changed[field] = "altered"
    assert canonical_fingerprint(changed) != original["fingerprint"]
