"""Publish a portable engineering-only projection of the retained smoke receipt."""

import argparse
import json
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = args.attempt / "report.json"
    source = json.loads(path.read_text())
    if source["test_performance_evaluated"] is not False:
        raise VisaIntegrityError("Not a development-only smoke receipt")
    models = {}
    for model, evidence in source["models"].items():
        models[model] = {
            key: value
            for key, value in evidence.items()
            if key not in ("calibration", "quantiles", "training_state")
        }
        models[model]["calibration_normal_count"] = evidence["calibration"][
            "normal_count"
        ]
        models[model]["map_hashes"] = evidence["calibration"]["maps"]
        for name, digest in evidence["calibration"]["maps"].items():
            if sha256_file(args.attempt / model / name) != digest:
                raise VisaIntegrityError("Smoke map hash mismatch")
    artifact_files = {}
    for file in sorted(args.attempt.rglob("*")):
        if file.is_file():
            artifact_files[file.relative_to(args.attempt).as_posix()] = {
                "sha256": sha256_file(file),
                "size_bytes": file.stat().st_size,
            }
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "source_report_sha256": sha256_file(path),
            "identity": source["identity"],
            "status": source["status"],
            "models": models,
            "artifact_files": artifact_files,
            "triage_interface_check": source["triage_interface_check"],
            "elapsed_seconds": source["elapsed_seconds"],
            "test_performance_evaluated": False,
            "final_test_lock": "closed",
            "optional_1000_step_probe": "not_run_existing_history_used_for_capacity",
        },
    )


if __name__ == "__main__":
    main()
