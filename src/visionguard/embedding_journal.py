"""Immutable ordered embedding chunks with atomic, hash-bound completion records."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from visionguard.visa import safe_asset
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


class EmbeddingJournal:
    """Tensor serialization is external; a chunk is committed only after its hash.

    Each attempt has its own journal and immutable origin. Resume imports verified
    references, never rewrites the old attempt or copies the full embedding set.
    The caller holds the execution-wide single-writer lock for its whole lifetime.
    """

    def __init__(
        self, root: Path, directory: Path, identity: dict, sample_ids: list[str]
    ):
        self.root = root.resolve()
        self.directory = directory.resolve()
        self.relative = self.directory.relative_to(self.root).as_posix()
        self.path = self.directory / "embeddings.json"
        self.identity = deepcopy(identity)
        self.samples = list(sample_ids)
        if not sample_ids or sample_ids != sorted(set(sample_ids)):
            raise VisaIntegrityError(
                "Embedding sample membership must be unique lexical order"
            )

    def create(self, resume: dict | None = None) -> dict:
        if self.path.exists() or (self.directory / "embedding-origin.json").exists():
            raise VisaIntegrityError("Attempt embedding origin already exists")
        prior = None
        if resume is not None:
            path = safe_asset(self.root, resume["path"])
            if sha256_file(path) != resume["sha256"]:
                raise VisaIntegrityError("Resume embedding journal hash mismatch")
            prior = self.validate(json.loads(path.read_text()))
        self.directory.mkdir(parents=True, exist_ok=True)
        origin = {
            "identity": self.identity,
            "sample_ids": self.samples,
            "attempt": self.relative,
            "parent": resume,
        }
        atomic_json(self.directory / "embedding-origin.json", origin)
        document = {
            **origin,
            "origin_sha256": sha256_file(self.directory / "embedding-origin.json"),
            "chunks": [] if prior is None else prior["chunks"],
            "next_index": 0 if prior is None else prior["next_index"],
        }
        atomic_json(self.path, document)
        return document

    def validate(self, document: dict | None = None) -> dict:
        d = json.loads(self.path.read_text()) if document is None else document
        if d["identity"] != self.identity or d["sample_ids"] != self.samples:
            raise VisaIntegrityError(
                "Embedding resume origin/membership identity mismatch"
            )
        origin_path = safe_asset(self.root, d["attempt"] + "/embedding-origin.json")
        if sha256_file(origin_path) != d["origin_sha256"]:
            raise VisaIntegrityError("Embedding origin hash mismatch")
        origin = json.loads(origin_path.read_text())
        if any(
            origin[k] != d[k] for k in ("identity", "sample_ids", "attempt", "parent")
        ):
            raise VisaIntegrityError("Embedding origin content mismatch")
        if d["next_index"] != len(d["chunks"]) or len(d["chunks"]) > len(self.samples):
            raise VisaIntegrityError("Embedding next index mismatch")
        for index, record in enumerate(d["chunks"]):
            if (
                record["index"] != index
                or record["sample_id"] != self.samples[index]
                or record["shape"] != [1024, 1536]
                or record["dtype"] != "torch.float32"
                or not record["path"].startswith(record["attempt"] + "/chunks/")
            ):
                raise VisaIntegrityError(
                    "Missing, reordered, or malformed embedding chunk"
                )
            path = safe_asset(self.root, record["path"])
            if sha256_file(path) != record["sha256"]:
                raise VisaIntegrityError("Embedding chunk corruption")
        return d

    def append(self, document: dict, path: Path, *, tensor_sha256: str) -> dict:
        index = document["next_index"]
        if index >= len(self.samples):
            raise VisaIntegrityError("Embedding sequence already complete")
        relative = path.resolve().relative_to(self.root).as_posix()
        if not relative.startswith(self.relative + "/chunks/"):
            raise VisaIntegrityError("Chunk must belong to current attempt")
        if relative in {r["path"] for r in document["chunks"]}:
            raise VisaIntegrityError("Chunk already committed")
        document["chunks"].append(
            {
                "index": index,
                "sample_id": self.samples[index],
                "path": relative,
                "attempt": self.relative,
                "sha256": sha256_file(path),
                "tensor_sha256": tensor_sha256,
                "shape": [1024, 1536],
                "dtype": "torch.float32",
                "size_bytes": path.stat().st_size,
            }
        )
        document["next_index"] = index + 1
        atomic_json(self.path, document)
        return document

    def receipt(self) -> dict:
        return {
            "path": self.path.relative_to(self.root).as_posix(),
            "sha256": sha256_file(self.path),
        }
