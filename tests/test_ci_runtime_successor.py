"""Current-vs-historical source separation, without model/data access."""

import json
import shutil
from pathlib import Path

import pytest

from visionguard import ci_runtime_successor as current
from visionguard.cleanup_successor import verify_freeze as historical_v5
from visionguard.heldout_resources import Budget

ROOT = Path(__file__).resolve().parents[1]


def test_current_binding_is_distinct_and_preserves_science_and_closed_lock():
    new = current.verify_freeze(ROOT)
    old = historical_v5(ROOT)
    assert new["document"]["schema_version"] == 6
    assert new["document"]["historical_fingerprint_v5"] == old["fingerprint"]
    assert new["fingerprint"] != old["fingerprint"]
    assert new["document"]["evaluation_lock"] == "CLOSED"
    assert new["document"]["real_execution_authorized"] is False
    for field in ["contract", "models", "environment"]:
        assert new["document"][field] == old["document"][field]


@pytest.fixture
def manufactured_repository(tmp_path):
    mapping = current.current_bindings(ROOT)["document"]
    names = set(mapping["current_source_sha256"]) | {current.MAP, current.FREEZE}
    names.update(row["archive"] for row in mapping["historical_archives"].values())
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize("kind", ["current", "historical", "mapping", "freeze"])
def test_tampered_current_or_historical_evidence_is_never_accepted(
    manufactured_repository, kind
):
    root = manufactured_repository
    mapping = current.current_bindings(root)["document"]
    if kind == "current":
        path = root / "tests/test_heldout_admission.py"
    elif kind == "historical":
        path = (
            root / mapping["historical_archives"][".github/workflows/ci.yml"]["archive"]
        )
    else:
        path = root / (current.MAP if kind == "mapping" else current.FREEZE)
    if path.suffix == ".json":
        content = json.loads(path.read_bytes())
        content["fingerprint"] = "0" * 64
        path.write_text(json.dumps(content))
    else:
        path.write_bytes(path.read_bytes() + b"\nmanufactured corruption\n")
    with pytest.raises(ValueError, match="drift"):
        current.verify_freeze(root)


def test_actual_insufficient_space_still_refuses_before_writing(tmp_path, monkeypatch):
    from visionguard import heldout_resources

    class EmptyDisk:
        free = 0

    budget = Budget(tmp_path)
    assert budget.reserve == 20 * 2**30
    monkeypatch.setattr(heldout_resources.shutil, "disk_usage", lambda _: EmptyDisk())
    with pytest.raises(ValueError, match="safety reserve exhausted"):
        budget.before(1)
    assert not (tmp_path / "writer.lock").exists()
