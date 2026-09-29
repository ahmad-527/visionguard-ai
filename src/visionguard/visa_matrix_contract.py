"""Frozen operational contract; no training or final-test access."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from visionguard.artifacts import capture_git_state
from visionguard.visa import CATEGORIES
from visionguard.visa_acquire import atomic_json, sha256_file
from visionguard.visa_guard import AUDIT, FROZEN, MEMBERSHIP, verified_context
from visionguard.visa_matrix_validation import load, require
from visionguard.visa_protocol import canonical_fingerprint

MERGE = "7bd562d2e5642a3e919805cbcc633acc8a98b941"
CONTRACT = "configs/execution/visa-development-matrix-v1.yaml"
MODELS = ("patchcore", "efficientad")
SEEDS = (42, 123, 2026)


def order():
    return [f"{m}:{c}:{s}" for m in MODELS for c in CATEGORIES for s in SEEDS]


def git(repository, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repository.as_posix()}", *args],
        cwd=repository,
        text=True,
    ).strip()


def source_hashes(repository):
    paths = sorted((repository / "src/visionguard").glob("*.py"))
    paths += sorted((repository / "scripts").glob("*.py"))
    paths += sorted((repository / "configs/engineering").glob("*.yaml"))
    paths += [repository / "pyproject.toml"]
    return {p.relative_to(repository).as_posix(): sha256_file(p) for p in paths}


def load_contract(repository: Path, *, expected: str | None = None) -> tuple[dict, str]:
    snapshot = yaml.safe_load((repository / CONTRACT).read_text())
    document = snapshot["execution"]
    fingerprint = canonical_fingerprint(document)
    require(fingerprint == snapshot["fingerprint"], "Execution fingerprint mismatch")
    if expected is not None:
        require(fingerprint == expected, "Wrong requested execution contract")
    require(document["order"] == order(), "Matrix order changed")
    require(
        document["protocols"] == FROZEN
        and document["audit_sha256"] == AUDIT
        and document["membership_sha256"] == MEMBERSHIP,
        "Scientific identities differ",
    )
    require(
        document["phase4c_merge"] == MERGE
        and document["final_test_lock"] == "closed"
        and document["human_access_history"] == "UNKNOWN"
        and document["independent_reservation"] == "NOT ESTABLISHED",
        "Governance drift",
    )
    require(
        source_hashes(repository) == document["source_hashes"],
        "Frozen implementation changed; STOP MATRIX",
    )
    for model in MODELS:
        verified_context(repository, model)
    return document, fingerprint


def freeze_commit(repository):
    return git(repository, "log", "-1", "--format=%H", "--", CONTRACT)


def verify_execution(repository, expected):
    document, fingerprint = load_contract(repository, expected=expected)
    state = capture_git_state(repository)
    require(not state["dirty"], "Matrix requires clean frozen checkout")
    require(
        state["commit"] == freeze_commit(repository),
        "HEAD changed after execution freeze; STOP",
    )
    return document, fingerprint, state["commit"]


def create_contract(repository: Path, local: dict):
    from visionguard.triage_protocol import EXPECTED_TRIAGE_FINGERPRINT
    from visionguard.visa_engineering import environment_identity

    require(
        git(repository, "show", "-s", "--format=%P", MERGE).split()
        == [
            "c5bc3f8269a2b507c57a6663360c2846f90922da",
            "86a027b5543ea7db709b1be3091cd8c63b7cf881",
        ],
        "Wrong Phase 4C merge parents",
    )
    git(
        repository,
        "merge-base",
        "--is-ancestor",
        "c019f5515ee7692b42cda5bcc0595a4d3f00ec1e",
        MERGE,
    )
    protocols = {m: verified_context(repository, m) for m in MODELS}
    assets = {
        k: sha256_file(Path(local[k]))
        for k in ("weight", "teacher", "imagenette_archive")
    }
    require(
        assets["weight"]
        == "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000",
        "Wrong weight",
    )
    require(
        assets["teacher"]
        == protocols["efficientad"]["scientific"]["model"]["teacher_weight_sha256"]
        and assets["imagenette_archive"]
        == protocols["efficientad"]["scientific"]["auxiliary_data"]["archive_sha256"],
        "Wrong auxiliary assets",
    )
    auxiliary = Path(local["imagenette_root"])
    inventory = [
        {
            "path": p.relative_to(auxiliary).as_posix(),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted((auxiliary / "train").rglob("*"))
        if p.is_file()
    ]
    require(bool(inventory), "ImageNette training inventory empty")
    inventory_path = (
        repository / "outputs/phase4d-a-preflight/imagenette-inventory.json"
    )
    atomic_json(inventory_path, {"records": inventory})
    membership = load(
        repository / "reports/phase4c-visa-readiness/development-membership.json"
    )
    counts = {
        c: {
            role: sum(
                r["category"] == c and r["role"] == role for r in membership["records"]
            )
            for role in ("fit", "calibration")
        }
        for c in CATEGORIES
    }
    largest = max(v["fit"] for v in counts.values())
    # Three retained model copies/cell, one EAD optimizer checkpoint, one normal map.
    # Measured Phase 4C files bound the conservative rounded ceilings below.
    storage = {
        "free_bytes_at_freeze": shutil.disk_usage(repository).free,
        "dataset_archive_bytes_existing": 1929840640,
        "dataset_extracted_bytes_existing_audit": 1920559633,
        "development_copy_bytes": sum(
            r["image_identity"]["size_bytes"] for r in membership["records"]
        ),
        "largest_embedding_payload_bytes": largest * 1024 * 1536 * 4,
        "active_patchcore_budget_bytes": 8 * 1024**3,
        "retained_patchcore_budget_bytes": 36 * (3 * 160 * 1024**2 + 16 * 1024**2),
        "retained_efficientad_budget_bytes": 36
        * (3 * 40 * 1024**2 + 96 * 1024**2 + 16 * 1024**2),
        "interruption_extra_budget_bytes": 20 * 1024**3,
        "minimum_free_bytes": 20 * 1024**3,
        "compaction": (
            "only validated-completed PatchCore embedding chunks; "
            "retain full inventory and durable deletion receipt"
        ),
        "checks": (
            "before/after cells and validation/compaction; every 60 seconds during work"
        ),
    }
    storage["required_additional_budget_bytes"] = sum(
        storage[k]
        for k in (
            "active_patchcore_budget_bytes",
            "retained_patchcore_budget_bytes",
            "retained_efficientad_budget_bytes",
            "interruption_extra_budget_bytes",
            "minimum_free_bytes",
        )
    )
    require(
        storage["free_bytes_at_freeze"] > storage["required_additional_budget_bytes"],
        "Insufficient safe storage; STOP",
    )
    document = {
        "id": "visa-development-matrix-v1",
        "phase4c_merge": MERGE,
        "protocols": FROZEN,
        "protocol_ids": {m: protocols[m]["id"] for m in MODELS},
        "triage_fingerprint": EXPECTED_TRIAGE_FINGERPRINT,
        "audit_sha256": AUDIT,
        "membership_sha256": MEMBERSHIP,
        "source_hashes": source_hashes(repository),
        "environment": environment_identity(),
        "asset_sha256": assets,
        "auxiliary_inventory_sha256": sha256_file(inventory_path),
        "phase4c_acceptance_sha256": sha256_file(
            repository / "reports/phase4c-visa-readiness/completion-summary.json"
        ),
        "category_order": list(CATEGORIES),
        "seeds": list(SEEDS),
        "order": order(),
        "counts": counts,
        "checkpoint_policy": {
            "patchcore": "every image; original coreset restarts from exact chunks/RNG",
            "efficientad_interval_steps": 1000,
            "efficientad_final_step": 70000,
            "efficientad": (
                "one latest complete optimizer/scheduler/RNG/stream "
                "checkpoint per attempt"
            ),
        },
        "resume_policy": (
            "new immutable attempt; exact origin; only classified "
            "ordinary interruption auto-resumes; all failures stop"
        ),
        "storage": storage,
        "hardware_policy": (
            "one GPU worker; no power/clock changes; "
            "disk/CUDA/WHEA/instability stops; no hardware retry"
        ),
        "monitor_seconds": 60,
        "failure_policy": (
            "stop scheduling; preserve state; source change requires human review"
        ),
        "final_test_lock": "closed",
        "phase4d_b_authorized": False,
        "human_access_history": "UNKNOWN",
        "independent_reservation": "NOT ESTABLISHED",
    }
    return {"execution": document, "fingerprint": canonical_fingerprint(document)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--freeze-from-local", type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    if args.freeze_from_local:
        snapshot = create_contract(repository, load(args.freeze_from_local))
        # JSON is canonical machine-readable YAML and avoids emitter ambiguity.
        atomic_json(repository / CONTRACT, snapshot)
        print(snapshot["fingerprint"])
    else:
        _, fingerprint = load_contract(repository)
        print(json.dumps({"fingerprint": fingerprint, "final_test_lock": "closed"}))


if __name__ == "__main__":
    main()
