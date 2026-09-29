"""Project existing normal-only engineering receipts; never execute a model."""

import argparse
import json
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_guard import verified_context
from visionguard.visa_protocol import canonical_fingerprint


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise VisaIntegrityError(message)


def checked(path, expected):
    require(sha256_file(path) == expected, f"Receipt identity mismatch: {path.name}")
    return load(path)


def project_equivalence(root):
    evidence = load(root / "equivalence.json")
    require(evidence["status"] == "exact", "Equivalence has not passed")
    reports = {
        name: checked(root / name / "report.json", digest)
        for name, digest in evidence["report_sha256"].items()
    }
    baseline = reports["reference"]
    keys = (
        "ordered_embeddings_sha256",
        "embedding_shape",
        "dtype",
        "coreset_indices",
        "memory_bank_sha256",
        "memory_bank_shape",
    )
    for report in reports.values():
        require(report["status"] == "passed", "Failed reference/candidate")
        require(report["identity"] == baseline["identity"], "Origin mismatch")
        require(
            all(report["fit"][k] == baseline["fit"][k] for k in keys),
            "Tensor/coreset equivalence changed; STOP",
        )
        require(
            report["canonical_model_sha256"] == baseline["canonical_model_sha256"],
            "Model equivalence changed; STOP",
        )
        require(
            canonical_fingerprint(report["calibration"])
            == canonical_fingerprint(baseline["calibration"]),
            "Exact calibration JSON differs; no tolerance allowed",
        )
    processes = {
        name: load(root / f"{name}-process.json")
        for name in ("reference", "safe", "interrupted", "resumed")
    }
    require(processes["interrupted"]["returncode"] == 75, "Missing hard termination")
    require(processes["resumed"]["returncode"] == 0, "Resume failed")
    require(
        processes["interrupted"]["pid"] != processes["resumed"]["pid"],
        "Resume did not use a new process",
    )
    return {
        **evidence,
        "raw_equivalence_sha256": sha256_file(root / "equivalence.json"),
        "processes": processes,
        "fixture": {
            "category": "candle",
            "seed": 42,
            "fit_count": 8,
            "calibration_count": len(baseline["calibration"]["inputs"]),
        },
        "exact_results": {
            **{k: baseline["fit"][k] for k in keys if k != "coreset_indices"},
            "coreset_indices_sha256": canonical_fingerprint(
                baseline["fit"]["coreset_indices"]
            ),
            "canonical_model_sha256": baseline["canonical_model_sha256"],
            "calibration_json_sha256": canonical_fingerprint(baseline["calibration"]),
            "image_threshold": baseline["calibration"]["image"]["threshold"],
            "pixel_threshold": baseline["calibration"]["pixel"]["threshold"],
            "thresholds_are_engineering_only": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repository = args.repository.resolve()
    for model in ("patchcore", "efficientad"):
        verified_context(repository, model)
    old = repository / "outputs/phase4c-completion-patchcore"
    root = repository / "outputs/phase4c-completion-patchcore-io-hardened"
    dispatch = repository / "outputs/phase4c-dispatcher-acceptance"
    equivalence = project_equivalence(root)
    initial = project_equivalence(old)
    largest = load(root / "largest/report.json")
    require(
        largest["status"] == "passed" and largest["fit"]["coreset_completed"],
        "Largest fit incomplete",
    )
    require(
        largest["fit_only"] is True and "calibration" not in largest,
        "Largest acceptance must be fit-only",
    )
    identity = largest["identity"]
    for name, digest in equivalence["implementation_identity"]["source_hashes"].items():
        require(sha256_file(repository / name) == digest, "Acceptance code drift")
    for key, value in equivalence["implementation_identity"].items():
        require(identity[key] == value, "Largest fit differs from accepted origin")
    failure = load(old / "largest/report.json")
    journal = load(old / "largest/embeddings.json")
    dispatcher = load(dispatch / "acceptance.json")
    require(dispatcher["status"] == "exact", "Dispatcher acceptance incomplete")
    for model, result in dispatcher["models"].items():
        states, calibrations = [], []
        for kind in ("uninterrupted", "restarted"):
            record = result[kind]
            output = dispatch / f"{model}-{kind}" / model
            manifest = checked(output / "manifest.json", record["manifest_sha256"])
            state = checked(
                dispatch / record["directory"] / "worker-state.json",
                record["receipt_sha256"],
            )
            require(state["status"] == "development_complete", "Incomplete cell")
            for artifact in state["files"].values():
                require(
                    sha256_file(output / artifact["path"]) == artifact["sha256"],
                    "Completed dispatcher artifact corruption",
                )
            for name, digest in state["identity"]["source_hashes"].items():
                require(
                    sha256_file(repository / name) == digest, "Dispatcher code drift"
                )
            states.append(state)
            calibrations.append(
                load(dispatch / record["directory"] / "calibration.json")
            )
            record["attempt_statuses"] = [
                a["status"] for a in manifest["cells"]["candle:42"]["attempts"]
            ]
        require(
            states[0]["canonical_final_model_sha256"]
            == states[1]["canonical_final_model_sha256"],
            "Dispatcher model differs",
        )
        require(
            canonical_fingerprint(calibrations[0])
            == canonical_fingerprint(calibrations[1]),
            "Dispatcher calibration differs",
        )
        result["identity"] = states[0]["identity"]
        result["canonical_final_model_sha256"] = states[0][
            "canonical_final_model_sha256"
        ]
        result["calibration_json_sha256"] = canonical_fingerprint(calibrations[0])
    membership = (
        repository / "reports/phase4c-visa-readiness/development-membership.json"
    )
    summary = {
        "schema_version": 1,
        "purpose": "NORMAL_ONLY_PRETEST_ENGINEERING_NOT_PERFORMANCE",
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
        "phase4d_started": False,
        "human_attestation": "PENDING HUMAN ATTESTATION",
        "generator_sha256": sha256_file(Path(__file__)),
        "initial_equivalence": initial,
        "accepted_equivalence": equivalence,
        "preserved_failure": {
            **{k: v for k, v in failure.items() if k != "identity"},
            "implementation_commit": failure["identity"]["implementation_commit"],
            "raw_report_sha256": sha256_file(old / "largest/report.json"),
            "process": load(old / "largest-process.json"),
            "committed_embedding_chunks": journal["next_index"],
            "total_chunk_files": len(list((old / "largest/chunks").glob("*.pt"))),
            "diagnosis": "Windows journal replace PermissionError; no CUDA failure",
            "remedy": (
                "bounded identical-metadata publication retry; "
                "fresh equivalence and fit"
            ),
        },
        "largest_fit": {
            **{k: v for k, v in largest.items() if k not in ("identity", "fit")},
            "raw_report_sha256": sha256_file(root / "largest/report.json"),
            "implementation_commit": identity["implementation_commit"],
            "source_fingerprint": identity["source_fingerprint"],
            "category": identity["category"],
            "seed": identity["seed"],
            "fit_count": len(identity["fit_sample_ids"]),
            "fit": {k: v for k, v in largest["fit"].items() if k != "coreset_indices"},
            "coreset_indices_sha256": canonical_fingerprint(
                largest["fit"]["coreset_indices"]
            ),
            "memory_bank_bytes": largest["fit"]["memory_bank_shape"][0] * 1536 * 4,
            "process": load(root / "largest-process.json"),
            "resource_caveats": [
                "CUDA allocator reserved is not measured physical resident VRAM",
                "Host measurement is worker peak working set, not system-wide RAM",
                "Disk is attempt file bytes before report, not filesystem peak",
                "Wall time is worker body, excluding imports and provenance preflight",
            ],
        },
        "dispatcher": {
            **dispatcher,
            "raw_receipt_sha256": sha256_file(dispatch / "acceptance.json"),
        },
        "membership_review": {
            "sha256": sha256_file(membership),
            "bytes": membership.stat().st_size,
            "lines": len(membership.read_text().splitlines()),
            "representation_changed": False,
            "conclusion": (
                "Retain exact normal image hashes/dimensions; "
                "split allocation alone cannot supply these"
            ),
        },
    }
    atomic_json(
        repository / "reports/phase4c-visa-readiness/completion-summary.json", summary
    )


if __name__ == "__main__":
    main()
