"""Independently verified B2 readiness contract, not execution permission."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import model_specs, require
from visionguard.visa_evaluator_artifacts import verify_published
from visionguard.visa_evaluator_protocol import verify_readiness_freeze
from visionguard.visa_protocol import canonical_fingerprint, verify_audit

PROTOCOL = "configs/protocols/visa-b2-readiness-v1.json"
FREEZE = "reports/phase4d-b2-readiness/implementation-freeze.json"
B1 = "0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49"


def context(repository: Path) -> dict:
    """Only canonical committed evidence; never touch a dataset root."""
    b1 = verify_readiness_freeze(repository)
    require(b1["fingerprint"] == B1, "B1 fingerprint changed")
    protocol = json.loads((repository / PROTOCOL).read_text())
    published = verify_published(repository)
    require(protocol["b1_fingerprint"] == B1, "Wrong B1 anchor")
    require(
        protocol["development_freeze_sha256"]
        == b1["document"]["protocol"]["development_freeze_sha256"],
        "Wrong development anchor",
    )
    require(
        protocol["dataset_audit_sha256"]
        == b1["document"]["protocol"]["dataset_audit_sha256"],
        "Wrong audit anchor",
    )
    require(
        not protocol["real_execution_enabled"], "Readiness cannot open real execution"
    )
    audit_path = repository / "reports/phase4c-visa-readiness/audit-summary.json"
    verify_audit(audit_path, protocol["dataset_audit_sha256"])
    audit = json.loads(audit_path.read_text())
    counts = {
        c: r["test_normal"] + r["test_anomaly"]
        for c, r in audit["category_counts"].items()
    }
    require(
        sum(counts.values()) == 2162 and sum(counts.values()) * 6 == 12972,
        "Scope count mismatch",
    )
    require(
        protocol["scope"]
        == {
            "categories": 12,
            "seeds": [42, 123, 2026],
            "models": 72,
            "pairs": 36,
            "images": 2162,
            "model_image_calls": 12972,
        },
        "Scope altered",
    )
    return {
        "protocol": protocol,
        "counts": counts,
        "specs": model_specs(published),
        "published": published,
    }


def build_freeze(repository: Path) -> dict:
    ctx = context(repository)
    paths = [
        p.relative_to(repository).as_posix()
        for pattern in (
            "src/visionguard/visa_b2*.py",
            "tests/test_visa_b2*.py",
            "scripts/phase4d_b2*.py",
        )
        for p in repository.glob(pattern)
    ]
    paths += [
        PROTOCOL,
        ".github/workflows/b2-readiness.yml",
        "configs/schemas/visa-b2-authorization-intent-v1.json",
    ]
    document = {
        "schema_version": 1,
        "protocol": ctx["protocol"],
        "b1_fingerprint": B1,
        "source_sha256": {p: sha256_file(repository / p) for p in sorted(paths)},
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository: Path) -> dict:
    stored = json.loads((repository / FREEZE).read_text())
    require(stored == build_freeze(repository), "B2 readiness fingerprint/source drift")
    return stored
