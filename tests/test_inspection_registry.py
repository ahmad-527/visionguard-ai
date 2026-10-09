"""Manufactured permission/identity fixtures; never real application authorization."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from visionguard.inspection_contract import InspectionError

pytest.importorskip("visionguard_inspection")
from visionguard_inspection.native import approved_registration, validate_science
from visionguard_inspection.registry import (
    Registry,
    load_registry,
    manufactured_registry,
    read_bound,
)
from visionguard_inspection.runtime import MODEL_VERSIONS, PROFILE_ID


@pytest.fixture(autouse=True)
def manufactured_runtime_metadata(monkeypatch):
    """These are permission/registry unit fixtures, not native runtime acceptance."""
    monkeypatch.setattr(
        "visionguard_inspection.runtime.version", lambda name: MODEL_VERSIONS[name]
    )


def bound(root, name, document):
    raw = json.dumps(document).encode()
    (root / name).write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def fixture_entry(root, **approval_overrides):
    science = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/protocols/patchcore-visa-v1.yaml"
        ).read_text()
    )["protocol"]["scientific"]
    item = {
        "model_id": "fixture",
        "model": "patchcore",
        "category": "candle",
        "seed": 42,
        "canonical_state_sha256": "c" * 64,
        "artifact": {"path": "NEVER-OPEN.pt", "sha256": "a" * 64},
        "science": bound(root, "science.json", science),
        "calibration": bound(
            root,
            "calibration.json",
            {"image": {"threshold": 0.123}, "pixel": {"threshold": 0.456}},
        ),
    }
    approval = {
        "schema_version": 1,
        "scope": "application-development-inference",
        "source_role": "development",
        "model_id": item["model_id"],
        "model": item["model"],
        "category": item["category"],
        "seed": item["seed"],
        "canonical_state_sha256": item["canonical_state_sha256"],
        "artifact_sha256": "a" * 64,
        "science_sha256": item["science"]["sha256"],
        "calibration_sha256": item["calibration"]["sha256"],
        "device": "cpu",
        "input_scope": "generated-or-development-non-held-out",
        "runtime_profile_id": PROFILE_ID,
        "runtime_versions": dict(MODEL_VERSIONS),
        "human_attestation": "MANUFACTURED UNIT FIXTURE ONLY",
        "expires_at_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        **approval_overrides,
    }
    item["application_permission"] = bound(root, "permission.json", approval)
    return item


def test_registry_immutable_duplicate_ids_and_no_path_selection():
    registry = manufactured_registry()
    entry = next(iter(registry.entries.values()))
    with pytest.raises(TypeError):
        registry.entries["new"] = entry
    with pytest.raises(InspectionError, match="Duplicate"):
        Registry((entry, replace(entry)))
    with pytest.raises(InspectionError):
        registry.select("../model.pt")


@pytest.mark.parametrize(
    "path", ["../model.pt", "C:/model.pt", "/model.pt", "file:stream", ""]
)
def test_nonportable_paths_refused(tmp_path, path):
    with pytest.raises(ValueError):
        read_bound(tmp_path, {"path": path, "sha256": "a" * 64})


def test_hash_and_size_and_absolute_root(tmp_path):
    record = bound(tmp_path, "file.json", {})
    with pytest.raises(InspectionError):
        read_bound(tmp_path, {**record, "sha256": "0" * 64})
    with pytest.raises(InspectionError):
        read_bound(tmp_path, record, limit=1)
    with pytest.raises(InspectionError):
        read_bound(Path("relative"), record)
    (tmp_path / "directory").mkdir()
    with pytest.raises(InspectionError, match="regular files"):
        read_bound(tmp_path, {"path": "directory", "sha256": "a" * 64})


@pytest.mark.parametrize(
    "overrides",
    [
        {"scope": "historical-evaluation"},
        {"device": "cuda"},
        {"human_attestation": ""},
        {"expires_at_utc": "2000-01-01T00:00:00+00:00"},
    ],
)
def test_permission_refused_before_artifact_open(tmp_path, overrides):
    item = fixture_entry(tmp_path, **overrides)
    with pytest.raises(InspectionError):
        approved_registration(tmp_path, item)
    assert not (tmp_path / "NEVER-OPEN.pt").exists()


def test_existing_thresholds_unchanged_and_permission_rechecked(tmp_path):
    item = fixture_entry(tmp_path)
    entry = approved_registration(tmp_path, item)
    assert entry.manifest.threshold == 0.123 and entry.pixel_threshold == 0.456
    entry.validate_use()
    (tmp_path / "permission.json").write_bytes(b"{}")
    with pytest.raises(InspectionError):
        entry.validate_use()


@pytest.mark.parametrize("changed", ["science", "calibration"])
def test_changed_pinned_configuration_is_refused_before_checkpoint_use(
    tmp_path, changed
):
    item = fixture_entry(tmp_path)
    entry = approved_registration(tmp_path, item)
    (tmp_path / item[changed]["path"]).write_bytes(b"{}")
    with pytest.raises(InspectionError, match="identity/size mismatch"):
        entry.validate_use()
    assert not (tmp_path / "NEVER-OPEN.pt").exists()


def test_permission_expires_after_registration_without_loading_any_artifact(
    tmp_path, monkeypatch
):
    item = fixture_entry(tmp_path)
    entry = approved_registration(tmp_path, item)

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(UTC) + timedelta(hours=2)

    monkeypatch.setattr("visionguard_inspection.native.datetime", Later)
    with pytest.raises(InspectionError, match="expired"):
        entry.validate_use()
    assert not (tmp_path / "NEVER-OPEN.pt").exists()


@pytest.mark.parametrize("present", [False, True])
def test_missing_or_mismatched_checkpoint_rejected_before_deserialization(
    tmp_path, monkeypatch, present
):
    item = fixture_entry(tmp_path)
    entry = approved_registration(tmp_path, item)
    monkeypatch.setattr(
        "visionguard_inspection.runtime.version",
        lambda package: MODEL_VERSIONS[package],
    )
    if present:
        (tmp_path / "NEVER-OPEN.pt").write_bytes(b"manufactured non-checkpoint bytes")
    with pytest.raises((InspectionError, FileNotFoundError)):
        entry.factory()


def test_missing_permission_before_any_other_asset(tmp_path):
    item = fixture_entry(tmp_path)
    (tmp_path / "permission.json").unlink()
    (tmp_path / "science.json").unlink()
    with pytest.raises(FileNotFoundError, match="permission"):
        approved_registration(tmp_path, item)


def test_pinned_empty_registry_and_size_schema(tmp_path):
    record = bound(tmp_path, "registry.json", {"schema_version": 1, "models": []})
    assert not load_registry(tmp_path / record["path"], record["sha256"]).entries
    record = bound(
        tmp_path, "registry.json", {"schema_version": 1, "models": [{}] * 73}
    )
    with pytest.raises(InspectionError):
        load_registry(tmp_path / record["path"], record["sha256"])


@pytest.mark.parametrize("model", ["patchcore", "efficientad"])
def test_native_preprocessing_contract_rejects_unimplemented_variants(model):
    path = (
        Path(__file__).resolve().parents[1] / f"configs/protocols/{model}-visa-v1.yaml"
    )
    science = yaml.safe_load(path.read_text())["protocol"]["scientific"]
    validate_science(model, science)
    science["preprocessing"]["resize_antialias"] = False
    with pytest.raises(InspectionError):
        validate_science(model, science)
