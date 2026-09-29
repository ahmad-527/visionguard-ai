"""Canonical Phase 4B protocol identity and repository-only provenance checks.

No dataset, prediction, map, checkpoint, or trained model is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from visionguard.efficientad_protocol import (
    efficientad_protocol_fingerprint,
    load_efficientad_protocol,
)
from visionguard.protocol import load_protocol, protocol_fingerprint

PROTOCOL_ID = "visionguard-dual-model-triage-v1"
EXPECTED_TRIAGE_FINGERPRINT = (
    "94442ab3121bccd392e3805b6134710cc6e8c95e8f17b8eccda288f8b1bd672d"
)
PROTOCOL_PATH = Path("configs/protocols/dual-model-triage-v1.yaml")


class TriageProtocolError(ValueError):
    """The protocol or its repository evidence differs from the frozen contract."""


class _UniqueLoader(yaml.SafeLoader):
    """Reject ambiguous YAML mappings rather than silently keeping the last key."""


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if type(key) is not str or key in result:
            raise TriageProtocolError("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _json_value(value: Any) -> None:
    # v1 needs no floats: this also rejects NaN, YAML dates, aliases with cycles,
    # custom objects, and Python equality quirks such as True == 1.
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is list:
        for child in value:
            _json_value(child)
    elif type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _json_value(child)
    else:
        raise TriageProtocolError(
            "Protocol must contain JSON primitives without floats"
        )


def triage_protocol_fingerprint(document: dict[str, Any]) -> str:
    """SHA-256 of the ENTIRE canonical document, including schema version.

    UTF-8 JSON; ASCII escaping; sorted mapping keys; compact separators; no NaN.
    YAML whitespace, comments, and mapping order have no scientific identity.
    List order is significant. This calculates identity, not authorization.
    """
    try:
        _json_value(document)
        if type(document) is not dict or set(document) != {
            "schema_version",
            "protocol",
        }:
            raise TriageProtocolError("Expected schema_version and protocol only")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != 1
        ):
            raise TriageProtocolError("Expected integer schema version 1")
        if type(document["protocol"]) is not dict:
            raise TriageProtocolError("protocol must be a mapping")
        canonical = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    except (RecursionError, TypeError) as exc:
        raise TriageProtocolError("Invalid or cyclic protocol document") from exc


def load_triage_protocol(path: Path) -> dict[str, Any]:
    """Load the exact frozen v1 contract; edited scientific rules fail closed."""
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
        fingerprint = triage_protocol_fingerprint(document)
    except (OSError, yaml.YAMLError, RecursionError) as exc:
        raise TriageProtocolError(f"Unable to read triage protocol: {exc}") from exc
    if fingerprint != EXPECTED_TRIAGE_FINGERPRINT:
        raise TriageProtocolError("Triage protocol differs from its frozen fingerprint")
    return document


def verify_repository_anchors(repository: Path) -> dict[str, str]:
    """Verify committed evidence identities without replaying any sample outputs.

    Hash LF-normalized repository text, matching .gitattributes. This checks
    manifests and Phase 4A provenance, not the original ignored run artifacts.
    Git merge identity is separately verified when preparing the freeze.
    """
    document = load_triage_protocol(repository / PROTOCOL_PATH)
    evidence = document["protocol"]["evidence"]
    verified: dict[str, str] = {}

    def read_json(relative: str, expected: str | None = None) -> dict:
        path = repository / relative
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise TriageProtocolError(f"Repository evidence hash mismatch: {relative}")
        verified[relative] = digest
        return json.loads(raw)

    try:
        summary_path = (
            "reports/phase4a-comparative-failure-analysis/analysis-summary.json"
        )
        summary = read_json(summary_path, evidence["phase4a_summary_sha256"])
        analysis_manifest = read_json(
            "reports/phase4a-comparative-failure-analysis/analysis-manifest.json"
        )
        if (
            analysis_manifest["artifacts"]["analysis-summary.json"]["sha256"]
            != (evidence["phase4a_summary_sha256"])
        ):
            raise TriageProtocolError("Phase 4A summary binding disagrees")
        if (
            summary["provenance"]["dataset_audit"]["sha256"]
            != (evidence["dataset_audit_sha256"])
        ):
            raise TriageProtocolError("Phase 4A audit identity disagrees")
        sources = (
            (
                "patchcore",
                "phase2c-public-benchmark",
                load_protocol,
                protocol_fingerprint,
            ),
            (
                "efficientad",
                "phase3b-efficientad-public-benchmark",
                load_efficientad_protocol,
                efficientad_protocol_fingerprint,
            ),
        )
        for model, directory, loader, fingerprint in sources:
            anchor = evidence[model]
            source_protocol = loader(
                repository / "configs/protocols" / f"{anchor['protocol_id']}.yaml"
            )
            if fingerprint(source_protocol) != anchor["protocol_fingerprint"]:
                raise TriageProtocolError(f"{model} protocol disagrees")
            manifest = read_json(
                f"reports/{directory}/benchmark-manifest.json",
                anchor["manifest_sha256"],
            )
            expected = {
                "protocol_id": anchor["protocol_id"],
                "protocol_fingerprint": anchor["protocol_fingerprint"],
                "benchmark_git_commit": anchor["implementation_sha"],
                "dataset_audit_sha256": evidence["dataset_audit_sha256"],
                "status": "completed",
            }
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise TriageProtocolError(f"{model} manifest identity disagrees")
            provenance = summary["provenance"][model]
            if (
                provenance["implementation"]["commit"] != anchor["implementation_sha"]
                or provenance["protocol_id"] != anchor["protocol_id"]
                or provenance["protocol_fingerprint"] != anchor["protocol_fingerprint"]
                or provenance["dataset_audit_sha256"]
                != evidence["dataset_audit_sha256"]
                or analysis_manifest["input_manifest_sha256"][model]
                != anchor["manifest_sha256"]
            ):
                raise TriageProtocolError(f"{model} Phase 4A provenance disagrees")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, TriageProtocolError):
            raise
        raise TriageProtocolError(
            f"Unable to verify repository evidence: {exc}"
        ) from exc
    return verified


def main(argv: list[str] | None = None) -> int:
    """Read-only protocol/anchor verification CLI; there is no evaluation mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        verified = verify_repository_anchors(args.repository)
    except TriageProtocolError as exc:
        parser.exit(2, f"Protocol verification failed: {exc}\n")
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "fingerprint": EXPECTED_TRIAGE_FINGERPRINT,
                "repository_evidence_sha256": verified,
                "performance_evaluation_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
