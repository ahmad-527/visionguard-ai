"""Canonical readiness implementation fingerprint; never grants test access."""

from __future__ import annotations

import json
from pathlib import Path

from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import EvaluationError
from visionguard.visa_protocol import canonical_fingerprint

PROTOCOL_PATH = "configs/protocols/visa-heldout-evaluator-v1.json"
FREEZE_PATH = "reports/phase4d-b1-visa-evaluator-readiness/implementation-freeze.json"


def verify_readiness_freeze(repository: Path) -> dict:
    """Scientific rules AND executable source bytes are part of the fingerprint."""
    snapshot = json.loads((repository / FREEZE_PATH).read_text(encoding="utf-8"))
    document = snapshot["document"]
    if canonical_fingerprint(document) != snapshot["fingerprint"]:
        raise EvaluationError("Readiness implementation fingerprint mismatch")
    protocol = json.loads((repository / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if document["protocol"] != protocol:
        raise EvaluationError("Readiness scientific methodology drift")
    for relative, digest in document["source_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise EvaluationError("Escaping readiness source binding")
        if (repository / path).resolve().is_relative_to(repository.resolve()) is False:
            raise EvaluationError("Escaping readiness source symlink")
        if sha256_file(repository / relative) != digest:
            raise EvaluationError(f"Readiness implementation drift: {relative}")
    return snapshot
