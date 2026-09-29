"""Run declared normal-only equivalence/restart workers, then optional fit-only gate.

Any failed worker or unequal tensor/map/threshold stops the driver. No retry,
no tolerance selection, no test path. Logs and failed receipts stay ignored.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--largest-only", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable,
        "-m",
        "visionguard.visa_patchcore_acceptance",
        "--repository",
        str(args.repository.resolve()),
        "--development",
        str(args.development.resolve()),
        "--weight",
        str(args.weight.resolve()),
        "--root",
        str(args.root.resolve()),
    ]
    runs = [
        ("reference", "reference", []),
        ("safe", "safe", []),
        ("interrupted", "safe", ["--exit-after", "4"]),
        ("resumed", "safe", ["--resume-from", "interrupted"]),
    ]
    if args.largest_only:
        runs = [("largest", "largest", [])]
    for attempt, mode, extras in runs:
        log = args.root / f"{attempt}.log"
        with log.open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(
                [*base, "--attempt", attempt, "--mode", mode, *extras],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            code = child.wait()
        expected = 75 if attempt == "interrupted" else 0
        atomic_json(
            args.root / f"{attempt}-process.json",
            {
                "attempt": attempt,
                "pid": child.pid,
                "returncode": code,
                "expected_returncode": expected,
                "intentional_termination": expected == 75,
                "log_sha256": sha256_file(log),
                "test_performance_evaluated": False,
            },
        )
        print(json.dumps({"attempt": attempt, "returncode": code}), flush=True)
        if code != expected:
            raise VisaIntegrityError("Acceptance worker failed; no automatic retry")
    if args.largest_only:
        return
    reports = {
        name: json.loads((args.root / name / "report.json").read_text())
        for name in ("reference", "safe", "resumed")
    }
    baseline = reports["reference"]
    checks = {}
    for name in ("safe", "resumed"):
        candidate = reports[name]
        checks[name] = {
            key: baseline["fit"][key] == candidate["fit"][key]
            for key in (
                "ordered_embeddings_sha256",
                "embedding_shape",
                "dtype",
                "coreset_indices",
                "memory_bank_sha256",
                "memory_bank_shape",
            )
        }
        checks[name]["canonical_model"] = (
            baseline["canonical_model_sha256"] == candidate["canonical_model_sha256"]
        )
        checks[name]["calibration_scores_maps_thresholds"] = (
            baseline["calibration"] == candidate["calibration"]
        )
    status = (
        "exact" if all(all(row.values()) for row in checks.values()) else "FAILED_STOP"
    )
    implementation = dict(baseline["identity"])
    for key in ("category", "seed", "fit_sample_ids", "pretrained_weight_sha256"):
        implementation.pop(key)
    atomic_json(
        args.root / "equivalence.json",
        {
            "criterion": "exact_bitwise_no_tolerance",
            "status": status,
            "checks": checks,
            "implementation_identity": implementation,
            "test_performance_evaluated": False,
            "report_sha256": {
                name: sha256_file(args.root / name / "report.json") for name in reports
            },
            "interrupted_journal_sha256": sha256_file(
                args.root / "interrupted/embeddings.json"
            ),
        },
    )
    if status != "exact":
        raise VisaIntegrityError("Equivalence failed; STOP for human review")
    print(json.dumps({"equivalence": status, "new_process_resume": "exact"}))


if __name__ == "__main__":
    main()
