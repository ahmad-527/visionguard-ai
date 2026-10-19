"""Container admission controls only; no trained bytes or exploit execution."""

import io
import zipfile

import pytest
from visionguard_inspection.state_archive import require_state_archive

from visionguard.inspection_contract import InspectionError


def archive(records, compression=zipfile.ZIP_STORED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as output:
        for name, data in records:
            output.writestr(name, data)
    raw = stream.getvalue()
    # ZipInfo normalizes the Windows separator on write; craft the actual name
    # bytes so this tests parser admission rather than writer normalization.
    separator = chr(92)
    for name, _ in records:
        if separator in name:
            raw = raw.replace(name.replace(separator, "/").encode(), name.encode())
    return raw


def state_records():
    # Deliberately not a valid pickle; admission must inspect metadata only.
    return [("archive/data.pkl", b"not deserialized"), ("archive/version", b"3\n")]


def test_ordinary_state_metadata_is_admitted_without_pickle_execution():
    require_state_archive(
        archive(
            [
                *state_records(),
                ("archive/data/0", b"fixture"),
                ("archive/.data/serialization_id", b"1"),
            ]
        )
    )


@pytest.mark.parametrize(
    "extra",
    [
        "archive/models/model.json",
        "archive/data/weights/model_weights_config.json",
        "archive/constants.pkl",
        "archive/code/model.py",
        "other/data/0",
        "archive/../data/0",
        "/archive/data/0",
        "archive/data\\0",
        "archive/data/x",
    ],
)
def test_export_jit_or_escaping_records_are_refused(extra):
    with pytest.raises(InspectionError):
        require_state_archive(archive([*state_records(), (extra, b"fixture")]))


@pytest.mark.parametrize("raw", [b"", b"pickle fixture", b"PK\x03\x04broken"])
def test_legacy_and_malformed_inputs_are_refused(raw):
    with pytest.raises(InspectionError):
        require_state_archive(raw)


def test_missing_and_compressed_state_are_refused():
    with pytest.raises(InspectionError, match="required records"):
        require_state_archive(archive([("archive/version", b"3\n")]))
    with pytest.raises(InspectionError, match="Unsupported"):
        require_state_archive(archive(state_records(), zipfile.ZIP_DEFLATED))


def test_duplicate_members_are_refused():
    with pytest.warns(UserWarning, match="Duplicate"):
        raw = archive([*state_records(), ("archive/data.pkl", b"duplicate")])
    with pytest.raises(InspectionError, match="Duplicate"):
        require_state_archive(raw)


def test_factory_refuses_export_container_before_torch_loader(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from visionguard_inspection import native
    from visionguard_inspection.runtime import MODEL_VERSIONS

    from test_inspection_registry import fixture_entry

    item = fixture_entry(tmp_path)
    raw = archive([*state_records(), ("archive/models/model.json", b"{}")])
    (tmp_path / item["artifact"]["path"]).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    item["artifact"]["sha256"] = digest
    permission = tmp_path / item["application_permission"]["path"]
    value = json.loads(permission.read_bytes())
    value["artifact_sha256"] = digest
    permission.write_bytes(json.dumps(value).encode())
    item["application_permission"]["sha256"] = hashlib.sha256(
        permission.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        "visionguard_inspection.runtime.version", lambda name: MODEL_VERSIONS[name]
    )
    calls = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(load=lambda *a, **kw: calls.append((a, kw))),
    )
    with pytest.raises(InspectionError, match="PT2/export"):
        native.approved_registration(tmp_path, item).factory()
    assert calls == []
