"""Snapshot the isolated installed profile and public pip installation receipts.

Does not open model/data artifacts, inspect permissions, install or fix packages.
Output is external by explicit path; refuses to overwrite prior observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path


def snapshot(reports: list[Path]) -> dict:
    packages = {}
    for installed in distributions():
        name = installed.metadata["Name"]
        if name in {"visionguard-ai", "visionguard-inspection"}:
            continue  # local build/commit identity is recorded separately
        packages[name.lower().replace("_", "-")] = {
            "name": name,
            "version": installed.version,
            "license_expression": installed.metadata.get("License-Expression"),
            "declared_license": installed.metadata.get("License"),
            "license_classifiers": [
                v
                for v in installed.metadata.get_all("Classifier", [])
                if v.startswith("License ::")
            ],
            "requires_dist": installed.metadata.get_all("Requires-Dist", []),
        }
    receipts = []
    for path in reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        sources = []
        for entry in report["install"]:
            if entry.get("is_direct") or entry.get("is_editable"):
                continue  # no private machine paths in public dependency inventory
            info = entry.get("download_info", {})
            sources.append(
                {
                    "name": entry["metadata"]["name"],
                    "version": entry["metadata"]["version"],
                    "url": info.get("url"),
                    "hashes": info.get("archive_info", {}).get("hashes", {}),
                }
            )
        receipts.append(
            {
                "receipt_name": path.name,
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "public_package_sources": sources,
            }
        )
    return {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "evidence_class": "CPU_DEPENDENCY_INVENTORY_NOT_SECURITY_CERTIFICATION",
        "python": sys.version,
        "platform": platform.platform(),
        "installed_public_distributions": packages,
        "installation_receipts": receipts,
        "limitations": [
            "Package metadata is not license/legal clearance",
            "Hashes identify received artifacts, not assurance of safety",
            "Platform-specific wheels and source-build outputs may differ",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pip-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(snapshot(args.pip_report), target, indent=2, sort_keys=True)
        target.write("\n")


if __name__ == "__main__":
    main()
