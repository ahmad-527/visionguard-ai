"""Single-process worker for exact development-only PatchCore acceptance.

The driver launches fresh workers; an intentional exit never opens test data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from visionguard.embedding_journal import EmbeddingJournal
from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_development import DevelopmentDataset
from visionguard.visa_engineering import execution_identity, host_peak_bytes, load_json
from visionguard.visa_guard import MEMBERSHIP
from visionguard.visa_patchcore import (
    build_patchcore,
    calibrate_development,
    durable_torch_save,
    fit_patchcore,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--attempt", required=True)
    parser.add_argument(
        "--mode", choices=("reference", "safe", "largest"), required=True
    )
    parser.add_argument("--resume-from")
    parser.add_argument("--exit-after", type=int)
    parser.add_argument("--stop-before-coreset", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    from visionguard.efficientad import canonical_checkpoint_sha256
    from visionguard.experiment import ReproducibilityConfig
    from visionguard.reproducibility import configure_reproducibility
    from visionguard.visa_smoke import _deny_sealed_access

    repository = args.repository.resolve()
    args.root = args.root.resolve()
    if args.attempt not in ("reference", "safe", "interrupted", "resumed", "largest"):
        raise VisaIntegrityError("Unknown declared acceptance attempt")
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < 20 * 1024**3:
        raise VisaIntegrityError("Insufficient disk headroom")
    protocol, identity = execution_identity(
        repository,
        "patchcore",
        extra_sources=(
            "src/visionguard/visa_patchcore_acceptance.py",
            "scripts/phase4c_patchcore_acceptance.py",
        ),
    )
    reports = repository / "reports/phase4c-visa-readiness"
    dataset = DevelopmentDataset(
        args.development,
        reports / "development-membership.json",
        expected_sha256=MEMBERSHIP,
    )
    _deny_sealed_access(args.development)
    category, limit = "candle", 8
    if args.mode == "largest":
        accepted = load_json(args.root / "equivalence.json")
        if (
            accepted["status"] != "exact"
            or accepted["implementation_identity"] != identity
        ):
            raise VisaIntegrityError(
                "Exact equivalence at this implementation required"
            )
        counts = load_json(reports / "audit-summary.json")["allocation_counts"]
        category = sorted(counts, key=lambda c: (-counts[c]["fit"], c))[0]
        limit = None
    paths = dataset.paths(category, "fit", limit=limit)
    records = [
        r for r in dataset.records if r["category"] == category and r["role"] == "fit"
    ]
    ids = [r["sample_id"] for r in records[:limit]]
    identity = {
        **identity,
        "category": category,
        "seed": 42,
        "fit_sample_ids": ids,
        "pretrained_weight_sha256": sha256_file(args.weight),
    }
    directory = args.root / args.attempt
    directory.mkdir(exist_ok=False)
    atomic_json(directory / "origin.json", identity)
    journal = EmbeddingJournal(args.root, directory, identity, ids)
    resume = None
    if args.resume_from:
        if args.resume_from != "interrupted":
            raise VisaIntegrityError("Only declared interrupted attempt may resume")
        prior = args.root / args.resume_from / "embeddings.json"
        resume = {
            "path": prior.relative_to(args.root).as_posix(),
            "sha256": sha256_file(prior),
        }
    document = journal.create(resume)
    report = {
        "status": "running",
        "identity": identity,
        "pid": os.getpid(),
        "resume_parent": resume,
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
        "fit_only": args.mode == "largest",
    }
    atomic_json(directory / "report.json", report)
    started = time.perf_counter()
    try:
        configure_reproducibility(ReproducibilityConfig(42, True, False))
        torch.cuda.reset_peak_memory_stats()
        model, transform = build_patchcore(protocol["scientific"], args.weight, "cuda")
        report["fit"] = fit_patchcore(
            model,
            transform,
            paths,
            journal,
            document,
            device="cuda",
            ratio=0.01,
            reference=args.mode == "reference",
            exit_after=args.exit_after,
            stop_before_coreset=args.stop_before_coreset,
        )
        report["canonical_model_sha256"] = canonical_checkpoint_sha256(
            model.state_dict()
        )
        digest = durable_torch_save(
            directory / "final-model.pt",
            {
                "identity": identity,
                "model_state": model.state_dict(),
                "canonical_model_sha256": report["canonical_model_sha256"],
            },
        )
        report["final_checkpoint_sha256"] = digest
        if args.mode != "largest":
            calibration = dataset.paths(category, "calibration")
            report["calibration"] = calibrate_development(
                model, transform, calibration, "cuda", directory
            )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        report.update(
            wall_seconds=time.perf_counter() - started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_peak_working_set_bytes=host_peak_bytes(),
            attempt_disk_bytes=sum(
                p.stat().st_size for p in directory.rglob("*") if p.is_file()
            ),
        )
        atomic_json(directory / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
