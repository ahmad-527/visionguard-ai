"""VisA protocol construction from pre-data choices and passed integrity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

from visionguard.efficientad_protocol import load_efficientad_protocol
from visionguard.protocol import load_protocol
from visionguard.triage_protocol import (
    EXPECTED_TRIAGE_FINGERPRINT,
    load_triage_protocol,
)
from visionguard.visa import CATEGORIES, SPLIT_SHA256
from visionguard.visa_acquire import (
    ARCHIVE_SHA256,
    VisaIntegrityError,
    sha256_file,
)

IMPLEMENTATION_FILES = (
    "src/visionguard/visa.py",
    "src/visionguard/visa_development.py",
    "src/visionguard/visa_smoke.py",
    "src/visionguard/efficientad_benchmark.py",
    "src/visionguard/calibration.py",
    "src/visionguard/preprocessing.py",
    "src/visionguard/metrics.py",
    "src/visionguard/benchmark_metrics.py",
    "src/visionguard/triage.py",
    "src/visionguard/triage_metrics.py",
)


def canonical_fingerprint(document: dict) -> str:
    """Scientific JSON identity independent of YAML comments and key ordering."""
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def build_visa_protocol(
    repository: Path,
    model: str,
    *,
    acquisition: dict,
    audit: dict,
    audit_sha256: str,
    membership_sha256: str,
) -> dict:
    """Freeze dataset bindings only after a passed audit; never infer observations."""
    if model not in ("patchcore", "efficientad"):
        raise VisaIntegrityError("Unknown model")
    if audit.get("status") != "passed" or audit.get("sample_count") != 10821:
        raise VisaIntegrityError("A complete passed official VisA audit is required")
    if (
        acquisition["archive"]["observed_sha256"] != ARCHIVE_SHA256
        or acquisition["source"]["files"]["split_csv/1cls.csv"]["sha256"]
        != SPLIT_SHA256
    ):
        raise VisaIntegrityError("VisA release/split identity mismatch")
    for digest in (audit_sha256, membership_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VisaIntegrityError("Evidence SHA-256 required")
    parent_path = repository / f"configs/protocols/{model}-mvtecad2-v1.yaml"
    parent = (load_protocol if model == "patchcore" else load_efficientad_protocol)(
        parent_path
    )
    triage = load_triage_protocol(
        repository / "configs/protocols/dual-model-triage-v1.yaml"
    )
    design = yaml.safe_load(
        (repository / "configs/protocols/visa-predata-design-v1.yaml").read_text()
    )
    scientific = deepcopy(parent["protocol"])
    for key in (
        "id",
        "status",
        "phase3a_public_evaluation_lock",
        "dataset",
        "resources",
    ):
        scientific.pop(key, None)
    scientific["calibration"]["split"] = "calibration"
    scientific["metrics"]["official_thresholded"]["threshold_source"] = (
        "calibration_normal_only"
    )
    if model == "efficientad":
        scientific["internal_map_normalization"]["source_split"] = "fit"
    return {
        "schema_version": 1,
        "protocol": {
            "id": f"{model}-visa-v1",
            "status": "frozen_for_review",
            "predata_design_fingerprint": canonical_fingerprint(design),
            "parent_protocol_fingerprint": design["design"][model][
                "parent_fingerprint"
            ],
            "scientific": scientific,
            "implementation_source_sha256": {
                name: sha256_file(repository / name) for name in IMPLEMENTATION_FILES
            },
            "dataset": {
                "release": "VisA_20220922.tar",
                "archive_sha256": ARCHIVE_SHA256,
                "official_split_sha256": SPLIT_SHA256,
                "categories": list(CATEGORIES),
                "source": deepcopy(acquisition["source"]),
                "audit_sha256": audit_sha256,
                "membership_sha256": membership_sha256,
                "allocation": deepcopy(design["design"]["allocation"]),
                "development_roles": ["fit", "calibration"],
                "sealed_test_role": "sealed_test",
                "test_sequestration": "separate_root_no_training_label_or_mask_access",
            },
            "seeds": [42, 123, 2026],
            "triage": {
                "id": triage["protocol"]["id"],
                "fingerprint": EXPECTED_TRIAGE_FINGERPRINT,
                "same_seed_only": True,
                "metrics": deepcopy(triage["protocol"]["metrics"]),
            },
            "model_metric_interpretation": (
                "prespecified_visionguard_metrics_not_official_visa_leaderboard_claim"
            ),
            "before_test_changes": deepcopy(design["design"][model]),
            "artifacts": {
                "schema_version": 1,
                "predictions": "lexical_portable_sample_id",
                "maps": (
                    "original_coordinate_float16_tiff_and_frozen_binary_png_separate_models"
                ),
                "image_score": "finite_float_own_model_scale",
                "thresholds": "normal_calibration_only",
                "checkpoint": "file_and_canonical_tensor_sha256",
                "attempts": "immutable",
                "provenance": [
                    "implementation_commit",
                    "source_hashes",
                    "environment",
                    "hardware",
                    "protocol_fingerprint",
                    "audit_sha256",
                    "membership_sha256",
                    "category",
                    "seed",
                ],
            },
            "failure_policy": (
                "stop_on_identity_integrity_nonfinite_or_hardware_failure_preserve_attempt"
            ),
            "final_test_authorization": (
                "closed_by_default_requires_separate_reviewed_execution_authorization"
            ),
            "partial_matrix_complete_claim": "forbidden",
        },
    }


def load_visa_protocol(path: Path, *, expected_fingerprint: str) -> dict:
    """The expected fingerprint must come from reviewed committed freeze metadata."""
    document = yaml.safe_load(path.read_text())
    if canonical_fingerprint(document) != expected_fingerprint:
        raise VisaIntegrityError("VisA protocol fingerprint mismatch")
    protocol = document["protocol"]
    if protocol["id"] not in ("patchcore-visa-v1", "efficientad-visa-v1"):
        raise VisaIntegrityError("Unexpected VisA protocol ID")
    if (
        protocol["dataset"]["archive_sha256"] != ARCHIVE_SHA256
        or protocol["dataset"]["official_split_sha256"] != SPLIT_SHA256
        or protocol["dataset"]["categories"] != list(CATEGORIES)
        or protocol["seeds"] != [42, 123, 2026]
        or protocol["triage"]["fingerprint"] != EXPECTED_TRIAGE_FINGERPRINT
    ):
        raise VisaIntegrityError("Frozen VisA scientific identity mismatch")
    return document


def verify_audit(path: Path, expected_sha256: str) -> dict:
    """Binding an audit hash is mandatory; status alone cannot authorize anything."""
    if sha256_file(path) != expected_sha256:
        raise VisaIntegrityError("Dataset audit identity mismatch")
    audit = json.loads(path.read_text())
    if audit.get("status") != "passed" or audit.get("archive_sha256") != ARCHIVE_SHA256:
        raise VisaIntegrityError("Audit is not a passed official release audit")
    return audit


def verify_implementation(repository: Path, protocol: dict) -> None:
    """Prevent running a changed adapter/training/metric implementation unnoticed."""
    for name, expected in protocol["implementation_source_sha256"].items():
        if sha256_file(repository / name) != expected:
            raise VisaIntegrityError(f"Implementation source mismatch: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce reviewed VisA fingerprints")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    reports = args.repository / "reports/phase4c-visa-readiness"
    freeze = json.loads((reports / "protocol-freeze.json").read_text())
    for model, expected in freeze["protocol_fingerprints"].items():
        document = load_visa_protocol(
            args.repository / f"configs/protocols/{model}-visa-v1.yaml",
            expected_fingerprint=expected,
        )
        protocol = document["protocol"]
        verify_audit(
            reports / "audit-summary.json", protocol["dataset"]["audit_sha256"]
        )
        if (
            sha256_file(reports / "development-membership.json")
            != (protocol["dataset"]["membership_sha256"])
        ):
            raise VisaIntegrityError("Membership binding mismatch")
        verify_implementation(args.repository, protocol)
    print(json.dumps(freeze["protocol_fingerprints"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
