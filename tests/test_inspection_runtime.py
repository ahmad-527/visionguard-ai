"""Manufactured metadata admission, not native execution or security clearance."""

from importlib.metadata import PackageNotFoundError

import pytest

from visionguard.inspection_contract import InspectionError

pytest.importorskip("visionguard_inspection")
from visionguard_inspection import runtime
from visionguard_inspection.native import approved_registration

from test_inspection_registry import fixture_entry


@pytest.mark.parametrize("package", runtime.MODEL_VERSIONS)
def test_exact_cpu_profile_refuses_missing_or_changed_versions(monkeypatch, package):
    def observed(name):
        if name == package:
            return "2.13.0" if name == "torch" else "0.0.0"
        return runtime.MODEL_VERSIONS[name]

    monkeypatch.setattr(runtime, "version", observed)
    with pytest.raises(InspectionError, match="runtime drift"):
        runtime.verify_runtime()

    def missing(name):
        if name == package:
            raise PackageNotFoundError(name)
        return runtime.MODEL_VERSIONS[name]

    monkeypatch.setattr(runtime, "version", missing)
    with pytest.raises(InspectionError, match="unavailable") as captured:
        runtime.verify_runtime()
    assert isinstance(captured.value.__cause__, PackageNotFoundError)


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime_profile_id": "historical-evaluation"},
        {"runtime_versions": {"torch": "2.9.1"}},
        {"runtime_versions": None},
    ],
)
def test_historical_permission_cannot_admit_successor_runtime(tmp_path, overrides):
    with pytest.raises(InspectionError, match="scope/binding mismatch"):
        approved_registration(tmp_path, fixture_entry(tmp_path, **overrides))
    assert not (tmp_path / "NEVER-OPEN.pt").exists()


def test_runtime_drift_is_rechecked_before_checkpoint_read(tmp_path, monkeypatch):
    registration = approved_registration(tmp_path, fixture_entry(tmp_path))
    monkeypatch.setattr(runtime, "version", lambda _: "UNREVIEWED")
    with pytest.raises(InspectionError, match="runtime drift"):
        registration.factory()
    assert not (tmp_path / "NEVER-OPEN.pt").exists()
