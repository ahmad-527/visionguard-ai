"""Synthetic durable-chunk fault injection; no models or benchmark data."""

import json

import pytest

from visionguard.embedding_journal import EmbeddingJournal
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


@pytest.fixture
def journal(tmp_path):
    identity = {
        "protocol": "synthetic",
        "audit": "a",
        "membership": "m",
        "code": "c",
        "environment": "e",
        "seed": 42,
    }
    store = EmbeddingJournal(tmp_path, tmp_path / "attempt-1", identity, ["a", "b"])
    doc = store.create()
    for index in range(2):
        path = store.directory / "chunks" / f"{index}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"synthetic chunk {index}".encode())
        doc = store.append(doc, path, tensor_sha256="a" * 64)
    return store


def test_resume_references_immutable_prior_attempt(journal):
    original = journal.path.read_bytes()
    second = EmbeddingJournal(
        journal.root, journal.root / "attempt-2", journal.identity, journal.samples
    )
    d = second.create(journal.receipt())
    assert second.validate()["next_index"] == 2
    assert d["chunks"] == journal.validate()["chunks"]
    assert journal.path.read_bytes() == original
    with pytest.raises(VisaIntegrityError, match="already exists"):
        second.create()


@pytest.mark.parametrize(
    "field", ["protocol", "audit", "membership", "code", "environment", "seed"]
)
def test_wrong_origin_fails_closed(journal, field):
    changed = {**journal.identity, field: "wrong"}
    second = EmbeddingJournal(
        journal.root, journal.root / "attempt-2", changed, journal.samples
    )
    with pytest.raises(VisaIntegrityError, match="identity"):
        second.create(journal.receipt())


@pytest.mark.parametrize(
    "fault", ["reordered", "missing", "corrupt", "index", "shape", "membership"]
)
def test_chunk_faults_rejected(journal, fault):
    d = json.loads(journal.path.read_text())
    if fault == "reordered":
        d["chunks"].reverse()
    elif fault == "missing":
        (journal.root / d["chunks"][0]["path"]).unlink()
    elif fault == "corrupt":
        (journal.root / d["chunks"][0]["path"]).write_bytes(b"corrupted synthetic")
    elif fault == "index":
        d["next_index"] = 1
    elif fault == "shape":
        d["chunks"][0]["shape"] = [1, 1]
    else:
        d["sample_ids"] = ["wrong"]
    with pytest.raises((VisaIntegrityError, FileNotFoundError)):
        journal.validate(d)


def test_uncommitted_partial_is_not_complete(journal):
    partial = journal.directory / "chunks" / "partial.pt.partial"
    partial.write_bytes(b"incomplete synthetic")
    assert len(journal.validate()["chunks"]) == 2


def test_journal_and_origin_tampering(journal):
    receipt = journal.receipt()
    atomic_json(journal.path, {})
    second = EmbeddingJournal(
        journal.root, journal.root / "attempt-2", journal.identity, journal.samples
    )
    with pytest.raises(VisaIntegrityError, match="hash"):
        second.create(receipt)
    assert sha256_file(journal.path) != receipt["sha256"]
