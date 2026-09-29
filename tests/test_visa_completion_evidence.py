"""Verify published engineering receipts without models, data or GPU execution."""

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_guard import AUDIT, FROZEN, MEMBERSHIP
from visionguard.visa_protocol import canonical_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_completion_is_hash_bound_engineering_not_test_performance():
    summary = json.loads(
        (ROOT / "reports/phase4c-visa-readiness/completion-summary.json").read_text()
    )
    assert summary["final_test_lock"] == "closed"
    assert summary["test_performance_evaluated"] is False
    assert summary["phase4d_started"] is False
    assert summary["human_attestation"] == "PENDING HUMAN ATTESTATION"
    assert summary["generator_sha256"] == sha256_file(
        ROOT / "scripts/phase4c_completion_evidence.py"
    )
    equivalence = summary["accepted_equivalence"]
    assert equivalence["status"] == "exact"
    assert all(all(checks.values()) for checks in equivalence["checks"].values())
    identities = [equivalence["implementation_identity"]] + [
        result["identity"] for result in summary["dispatcher"]["models"].values()
    ]
    for identity in identities:
        assert identity["audit_sha256"] == AUDIT
        assert identity["membership_sha256"] == MEMBERSHIP
        assert identity["protocol_fingerprint"] == FROZEN[identity["model"]]
        assert (
            canonical_fingerprint(identity["source_hashes"])
            == identity["source_fingerprint"]
        )
        for name, digest in identity["source_hashes"].items():
            assert sha256_file(ROOT / name) == digest
    assert summary["largest_fit"]["fit_only"] is True
    assert summary["largest_fit"]["fit"]["coreset_completed"] is True
    assert summary["preserved_failure"]["status"] == "failed"
    assert summary["membership_review"]["representation_changed"] is False


def test_published_restart_receipts_preserve_distinct_process_and_attempts():
    summary = json.loads(
        (ROOT / "reports/phase4c-visa-readiness/completion-summary.json").read_text()
    )
    processes = summary["accepted_equivalence"]["processes"]
    assert processes["interrupted"]["returncode"] == 75
    assert processes["resumed"]["returncode"] == 0
    assert processes["interrupted"]["pid"] != processes["resumed"]["pid"]
    for model, attempts in (("patchcore", 5), ("efficientad", 3)):
        result = summary["dispatcher"]["models"][model]
        assert result["canonical_model_and_calibration_exact"] is True
        assert result["uninterrupted"]["attempt_count"] == 1
        assert result["restarted"]["attempt_count"] == attempts
        assert result["restarted"]["attempt_statuses"] == [
            *(["interrupted"] * (attempts - 1)),
            "development_complete",
        ]
