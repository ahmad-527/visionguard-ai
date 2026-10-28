import pytest

from visionguard import heldout_admission as admission
from visionguard.visa import CATEGORIES


def test_exact_artificial_admission(tmp_path):
    permit = admission.make_artificial(tmp_path)
    inputs = admission.admit_artificial(permit)
    assert set(inputs.rows) == set(CATEGORIES)
    assert sum(len(r) for r in inputs.rows.values()) == 48
    assert inputs.artificial
    for c in CATEGORIES:
        frames = list(inputs.frames(c))
        assert [f.sample.sample_id for f in frames] == sorted(
            inputs.origin["membership"][c]
        )
        assert sum(f.sample.label for f in frames) == 2
        assert all(set(f.sample.mask.ravel()) <= {0, 1} for f in frames)


def test_unissued_cannot_adopt_existing_directory(tmp_path):
    with pytest.raises(ValueError):
        admission.admit_artificial(admission.ArtificialPermit("x"))
    with pytest.raises(ValueError):
        admission.admit(tmp_path, object())


@pytest.mark.parametrize(
    "fault", ["extra", "missing", "corrupt", "split", "inventory", "mask", "symlink"]
)
def test_admission_fail_closed(tmp_path, fault):
    permit = admission.make_artificial(tmp_path)
    root = admission._ARTIFICIAL[permit.nonce][1]
    path = next((root / "sealed/candle/Data/Images/Normal").glob("*.png"))
    if fault == "extra":
        (root / "sealed/extra.txt").write_text("artificial")
    elif fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"bad")
    elif fault == "split":
        (root / "source/split_csv/1cls.csv").write_text("wrong")
    elif fault == "inventory":
        (root / "inventory.json").write_text("{}")
    elif fault == "mask":
        next((root / "sealed/candle/Data/Masks/Anomaly").glob("*.png")).write_bytes(
            b"bad"
        )
    else:
        other = tmp_path / "outside"
        other.mkdir()
        try:
            (root / "sealed/link").symlink_to(other, target_is_directory=True)
        except OSError:
            pytest.skip(
                "Symlink privilege unavailable; reparse stat unit tested separately"
            )
    with pytest.raises((ValueError, OSError, KeyError)):
        admission.admit_artificial(permit)


def test_no_toctou_between_admission_and_frames(tmp_path):
    permit = admission.make_artificial(tmp_path)
    inputs = admission.admit_artificial(permit)
    path = inputs.root / "sealed" / inputs.rows["candle"][0].image
    path.write_bytes(b"altered")
    with pytest.raises((ValueError, OSError)):
        list(inputs.frames("candle"))


def test_unsafe_paths(tmp_path):
    from visionguard.heldout_paths import asset

    for relative in ("../x", "C:/x", "a\\x", "/root/x", "a//x"):
        with pytest.raises(ValueError):
            asset(tmp_path, relative)
