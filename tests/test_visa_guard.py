"""Arbitrary self-asserted identities cannot authorize final test."""

from pathlib import Path

import pytest

from visionguard.visa_acquire import VisaIntegrityError
from visionguard.visa_guard import AUDIT, FROZEN, reject_test_request, verified_context

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "fingerprint,audit,reason",
    [
        ("arbitrary", "arbitrary", "fingerprint"),
        (FROZEN["patchcore"], "arbitrary", "audit"),
        (FROZEN["patchcore"], AUDIT, "not authorized"),
    ],
)
def test_canonical_comparison_stays_closed(fingerprint, audit, reason):
    with pytest.raises(VisaIntegrityError, match=reason):
        reject_test_request(
            ROOT,
            "patchcore",
            supplied_fingerprint=fingerprint,
            supplied_audit=audit,
            confirmed=True,
        )


def test_repository_canonical_sources_still_verify():
    assert verified_context(ROOT, "patchcore")["id"] == "patchcore-visa-v1"
    assert verified_context(ROOT, "efficientad")["id"] == "efficientad-visa-v1"
