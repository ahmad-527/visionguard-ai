"""Distinct current CI/runtime binding; historical freezes use original blobs.

This is metadata verification, never scientific acceptance or execution authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from visionguard.visa_evaluator import require
from visionguard.visa_protocol import canonical_fingerprint

MAP = "reports/ci-runtime-successor-v6/source-bindings.json"
FREEZE = "reports/ci-runtime-successor-v6/implementation-freeze-v6.json"
BASE = "f13beb897e9223e07866f1f681c457cce6dcfec1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HistoricalRepository(os.PathLike):
    """Route only explicitly archived historical source paths, not current tests."""

    def __init__(self, root, mapping):
        self.root = root
        self.mapping = mapping

    def __fspath__(self):
        return str(self.root)

    def __truediv__(self, name):
        key = str(name).replace("\\", "/")
        item = self.mapping.get(key)
        return self.root / (item["archive"] if item else name)

    def glob(self, pattern):
        return self.root.glob(pattern)

    def resolve(self):
        return self.root.resolve()


def current_bindings(repository):
    root = Path(repository)
    record = json.loads((root / MAP).read_bytes())
    require(
        record["fingerprint"] == canonical_fingerprint(record["document"]),
        "CI successor binding drift",
    )
    doc = record["document"]
    require(
        doc["historical_base"] == BASE and doc["schema_version"] == 6,
        "CI successor historical anchor drift",
    )
    for name, expected in doc["current_source_sha256"].items():
        require(
            digest(root / name) == expected, "Current successor source drift: " + name
        )
    for name, item in doc["historical_archives"].items():
        raw = (root / item["archive"]).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\x00" + raw
        ).hexdigest()
        require(
            hashlib.sha256(raw).hexdigest() == item["sha256"]
            and blob == item["git_blob"],
            "Historical source blob drift: " + name,
        )
    return record


def historical_repository(repository):
    if isinstance(repository, HistoricalRepository):
        return repository
    root = Path(repository)
    if not (root / MAP).exists():
        return root
    return HistoricalRepository(
        root, current_bindings(root)["document"]["historical_archives"]
    )


def build_freeze(repository):
    root = Path(repository)
    binding = current_bindings(root)
    from visionguard.cleanup_successor import verify_freeze as historical_v5

    old = historical_v5(historical_repository(root))
    document = {
        "schema_version": 6,
        "evidence_class": "CURRENT_CI_RUNTIME_SUCCESSOR_NOT_HISTORICAL_ACCEPTANCE",
        "historical_base": BASE,
        "historical_fingerprint_v5": old["fingerprint"],
        "contract": old["document"]["contract"],
        "models": old["document"]["models"],
        "environment": old["document"]["environment"],
        "application_runtime_profile": "application-cpu-20261028-v2",
        "evaluation_lock": "CLOSED",
        "real_execution_authorized": False,
        "source_sha256": binding["document"]["current_source_sha256"],
        "source_binding_sha256": digest(root / MAP),
    }
    return {"document": document, "fingerprint": canonical_fingerprint(document)}


def verify_freeze(repository):
    root = Path(repository)
    saved = json.loads((root / FREEZE).read_bytes())
    require(
        saved == build_freeze(root), "Current CI/runtime successor fingerprint drift"
    )
    return saved
