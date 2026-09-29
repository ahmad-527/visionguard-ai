"""Freeze/provenance contract tests; repository metadata, never a hybrid replay."""

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from visionguard.triage import ALLOWED_SEEDS, triage
from visionguard.triage_metrics import calculate_triage_metrics
from visionguard.triage_protocol import (
    EXPECTED_TRIAGE_FINGERPRINT,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    TriageProtocolError,
    load_triage_protocol,
    main,
    triage_protocol_fingerprint,
    verify_repository_anchors,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fingerprint_independent_canonical_calculation():
    document = load_triage_protocol(ROOT / PROTOCOL_PATH)
    canonical = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_TRIAGE_FINGERPRINT
    assert document["protocol"]["id"] == PROTOCOL_ID
    assert tuple(document["protocol"]["pairing"]["seeds"]) == ALLOWED_SEEDS
    for row in document["protocol"]["decision"]["truth_table"]:
        result = triage(
            int(row["patchcore_anomalous"]), 0, int(row["efficientad_anomalous"]), 0
        )
        assert result.triage_decision.value == row["outcome"]
    assert set(document["protocol"]["metrics"]["rates"]) == set(
        calculate_triage_metrics([], []).rates
    )


def test_mapping_order_comments_and_whitespace_do_not_change_identity(tmp_path):
    document = load_triage_protocol(ROOT / PROTOCOL_PATH)
    reversed_root = dict(reversed(list(document.items())))
    path = tmp_path / "protocol.yaml"
    path.write_text(
        "# harmless presentation change\n\n"
        + yaml.safe_dump(reversed_root, sort_keys=False)
    )
    assert (
        triage_protocol_fingerprint(load_triage_protocol(path))
        == EXPECTED_TRIAGE_FINGERPRINT
    )


@pytest.mark.parametrize(
    "keys,value",
    [
        (("id",), "different-v2"),
        (("version",), 2),
        (("decision", "anomalous_comparison"), "greater_or_equal"),
        (("decision", "truth_table", 0, "outcome"), "REVIEW"),
        (("decision", "review_to_binary_conversion"), "normal"),
        (("inputs", "cross_model_normalization"), "min_max"),
        (("inputs", "category_specific_routing"), "allowed"),
        (("localization", "fusion"), "union"),
        (("pairing", "seeds"), [42]),
        (("pairing", "same_seed_required"), False),
        (("metrics", "rates", "anomaly_pass_through_rate"), "P_A/N"),
        (("metrics", "zero_denominator"), "zero"),
        (("metrics", "reporting", "per_seed_macro"), "weighted"),
        (("evaluation", "mvtec_test_public", "independent"), True),
        (
            ("evaluation", "mvtec_test_public", "mandatory_label"),
            "independent validation",
        ),
        (("evaluation", "visa", "test"), "development_allowed"),
        (("evaluation", "mvtec_private", "authorization"), "automatic"),
        (("evaluation", "phase4c_authorized"), True),
        (("evidence", "patchcore", "implementation_sha"), "0" * 40),
        (("evidence", "efficientad", "protocol_fingerprint"), "0" * 64),
        (("evidence", "dataset_audit_sha256"), "0" * 64),
        (("evidence", "phase4a_reviewed_merge"), "0" * 40),
        (("fail_closed", "missing_evidence"), "PASS"),
        (("interpretation", "single_metric_winner_claim"), "allowed"),
    ],
)
def test_scientific_changes_change_fingerprint_and_are_rejected(tmp_path, keys, value):
    document = deepcopy(load_triage_protocol(ROOT / PROTOCOL_PATH))
    node = document["protocol"]
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    assert triage_protocol_fingerprint(document) != EXPECTED_TRIAGE_FINGERPRINT
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(TriageProtocolError, match="frozen fingerprint"):
        load_triage_protocol(path)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        "schema_version: true\nprotocol: {}",
        "schema_version: 1.0\nprotocol: {}",
        "schema_version: 1\nprotocol: {a: 1, a: 2}",
        "schema_version: 1\nprotocol: {threshold: .nan}",
        "schema_version: 1\nprotocol: &a {recursive: *a}",
        "schema_version: 1\nprotocol: {date: 2026-09-29}",
        "schema_version: 1\nprotocol: {[a, b]: invalid}",
        "schema_version: 1\nprotocol: {}\nextra: ignored",
    ],
)
def test_malformed_ambiguous_or_noncanonical_protocol_fails(tmp_path, raw):
    path = tmp_path / "invalid.yaml"
    path.write_text(raw)
    with pytest.raises(TriageProtocolError):
        load_triage_protocol(path)


def test_missing_protocol_fails(tmp_path):
    with pytest.raises(TriageProtocolError):
        load_triage_protocol(tmp_path / "missing.yaml")


def test_repository_evidence_anchors_and_cli(capsys):
    verified = verify_repository_anchors(ROOT)
    assert len(verified) == 4
    assert main(["--repository", str(ROOT)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["fingerprint"] == EXPECTED_TRIAGE_FINGERPRINT
    assert output["performance_evaluation_performed"] is False


@pytest.fixture
def evidence_copy(tmp_path):
    paths = [
        PROTOCOL_PATH,
        Path("configs/protocols/patchcore-mvtecad2-v1.yaml"),
        Path("configs/protocols/efficientad-mvtecad2-v1.yaml"),
        Path("reports/phase2c-public-benchmark/benchmark-manifest.json"),
        Path("reports/phase3b-efficientad-public-benchmark/benchmark-manifest.json"),
        Path("reports/phase4a-comparative-failure-analysis/analysis-manifest.json"),
        Path("reports/phase4a-comparative-failure-analysis/analysis-summary.json"),
    ]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


@pytest.mark.parametrize(
    "relative",
    [
        "configs/protocols/patchcore-mvtecad2-v1.yaml",
        "configs/protocols/efficientad-mvtecad2-v1.yaml",
        "reports/phase2c-public-benchmark/benchmark-manifest.json",
        "reports/phase3b-efficientad-public-benchmark/benchmark-manifest.json",
        "reports/phase4a-comparative-failure-analysis/analysis-summary.json",
    ],
)
def test_changed_or_missing_anchor_rejected(evidence_copy, relative):
    path = evidence_copy / relative
    path.write_text("{}")
    with pytest.raises(TriageProtocolError):
        verify_repository_anchors(evidence_copy)
    path.unlink()
    with pytest.raises(TriageProtocolError):
        verify_repository_anchors(evidence_copy)


def test_phase4a_manifest_binding_rejected(evidence_copy):
    path = (
        evidence_copy
        / "reports/phase4a-comparative-failure-analysis/analysis-manifest.json"
    )
    document = json.loads(path.read_text())
    document["input_manifest_sha256"]["patchcore"] = "0" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(TriageProtocolError, match="provenance disagrees"):
        verify_repository_anchors(evidence_copy)


def test_cli_failure_has_nonzero_exit(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--repository", str(tmp_path)])
    assert exc.value.code == 2
