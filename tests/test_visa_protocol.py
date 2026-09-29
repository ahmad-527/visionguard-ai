"""Protocol provenance and mutation checks; no real model outcomes."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_protocol import (
    build_visa_protocol,
    canonical_fingerprint,
    load_visa_protocol,
    main,
    verify_audit,
    verify_implementation,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/phase4c-visa-readiness"
FREEZE = json.loads((REPORTS / "protocol-freeze.json").read_text())


@pytest.fixture(params=["patchcore", "efficientad"])
def protocol(request):
    return load_visa_protocol(
        ROOT / f"configs/protocols/{request.param}-visa-v1.yaml",
        expected_fingerprint=FREEZE["protocol_fingerprints"][request.param],
    )


def test_reproduce_freeze_and_implementation(protocol):
    p = protocol["protocol"]
    model = p["id"].split("-")[0]
    generated = build_visa_protocol(
        ROOT,
        model,
        acquisition=json.loads((REPORTS / "acquisition.json").read_text()),
        audit=json.loads((REPORTS / "audit-summary.json").read_text()),
        audit_sha256=FREEZE["audit_sha256"],
        membership_sha256=FREEZE["membership_sha256"],
    )
    assert generated == protocol
    verify_implementation(ROOT, p)
    assert main(["--repository", str(ROOT)]) == 0


def test_yaml_comments_whitespace_key_order_do_not_change_identity(protocol):
    encoded = "# non-scientific comment\n\n" + yaml.safe_dump(protocol, sort_keys=True)
    assert canonical_fingerprint(yaml.safe_load(encoded)) == canonical_fingerprint(
        protocol
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("dataset", "archive_sha256"), "0" * 64),
        (("dataset", "official_split_sha256"), "0" * 64),
        (("dataset", "membership_sha256"), "0" * 64),
        (("dataset", "audit_sha256"), "0" * 64),
        (("dataset", "categories"), ["candle"]),
        (("dataset", "allocation", "salt"), "changed"),
        (("dataset", "test_sequestration"), "opened"),
        (("seeds",), [42]),
        (("scientific", "calibration", "comparison"), ">="),
        (("scientific", "preprocessing", "resize"), [128, 128]),
        (("scientific", "model", "algorithm"), "different"),
        (("triage", "same_seed_only"), False),
        (("triage", "metrics"), {}),
        (("artifacts", "schema_version"), 2),
        (("failure_policy",), "continue"),
    ],
)
def test_scientific_changes_mutate_fingerprint(protocol, path, value):
    changed = deepcopy(protocol)
    target = changed["protocol"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert canonical_fingerprint(changed) != canonical_fingerprint(protocol)


def test_normal_only_strict_semantics_and_unchanged_steps(protocol):
    p = protocol["protocol"]
    science = p["scientific"]
    assert science["calibration"]["normal_only"] is True
    assert science["calibration"]["split"] == "calibration"
    assert (
        science["calibration"]["comparison"] == "score_strictly_greater_than_threshold"
    )
    if p["id"] == "efficientad-visa-v1":
        assert science["training"]["max_steps"] == 70000
        assert science["internal_map_normalization"]["source_split"] == "fit"


def test_wrong_fingerprint_and_source_fail_closed(protocol, tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    with pytest.raises(VisaIntegrityError, match="fingerprint"):
        load_visa_protocol(path, expected_fingerprint="0" * 64)
    changed = deepcopy(protocol["protocol"])
    name = next(iter(changed["implementation_source_sha256"]))
    changed["implementation_source_sha256"][name] = "0" * 64
    with pytest.raises(VisaIntegrityError, match="source mismatch"):
        verify_implementation(ROOT, changed)


def test_audit_hash_and_status_both_required(tmp_path):
    path = tmp_path / "synthetic-audit.json"
    atomic_json(path, {"status": "failed"})
    with pytest.raises(VisaIntegrityError, match="identity"):
        verify_audit(path, "0" * 64)
    with pytest.raises(VisaIntegrityError, match="passed"):
        verify_audit(path, sha256_file(path))


def test_scientific_builder_rejects_failed_audit():
    with pytest.raises(VisaIntegrityError, match="complete passed"):
        build_visa_protocol(
            ROOT,
            "patchcore",
            acquisition={},
            audit={"status": "failed"},
            audit_sha256="a" * 64,
            membership_sha256="b" * 64,
        )


def test_nonfinite_fingerprint_forbidden():
    with pytest.raises(ValueError):
        canonical_fingerprint({"value": float("nan")})
