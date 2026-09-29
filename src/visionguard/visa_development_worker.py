"""Normal-only worker for the VisA dispatcher. No final-test path exists."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from visionguard.embedding_journal import EmbeddingJournal, publish_journal
from visionguard.visa import CATEGORIES, safe_asset
from visionguard.visa_acquire import VisaIntegrityError, sha256_file
from visionguard.visa_development import DevelopmentDataset
from visionguard.visa_guard import MEMBERSHIP, verified_context
from visionguard.visa_patchcore import (
    build_patchcore,
    calibrate_development,
    durable_torch_save,
    fit_patchcore,
)

STAGES = (
    "created",
    "fit",
    "normalization",
    "calibration",
    "artifact_write",
    "development_complete",
)


def validate_request_scope(request: dict) -> None:
    """Independently reject undeclared seeds, subsets, and step budgets."""
    model, category, seed = request["model"], request["category"], request["seed"]
    if (
        model not in ("patchcore", "efficientad")
        or category not in CATEGORIES
        or type(seed) is not int
        or seed not in (42, 123, 2026)
    ):
        raise VisaIntegrityError("Invalid worker category/seed/model")
    scope = request["identity"]["scope"]
    engineering = scope.get("purpose") == "ENGINEERING_ONLY"
    expected = {
        "purpose": "ENGINEERING_ONLY"
        if engineering
        else "NORMAL_ONLY_FULL_DEVELOPMENT",
        "fit_limit": (8 if model == "patchcore" else 4) if engineering else None,
        "calibration_limit": 19 if engineering and model == "efficientad" else None,
        "steps": 2 if engineering else 70000,
    }
    if scope != expected or (engineering and (category, seed) != ("candle", 42)):
        raise VisaIntegrityError(
            "Worker scope differs from frozen development contract"
        )
    if request["stop_at"] is not None and not engineering:
        raise VisaIntegrityError("Worker fault injection requires engineering scope")
    if any(key in request for key in ("test_root", "test_labels", "evaluate_test")):
        raise VisaIntegrityError("Test inputs forbidden")


def file_record(root: Path, path: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    safe_asset(root, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_receipt(root: Path, receipt: dict, identity: dict) -> dict:
    path = safe_asset(root, receipt["path"])
    if sha256_file(path) != receipt["sha256"]:
        raise VisaIntegrityError("Development receipt hash mismatch")
    document = json.loads(path.read_text())
    if (
        document["identity"] != identity
        or document["final_test_lock"] != "closed"
        or document["test_performance_evaluated"] is not False
        or document["stage"] not in STAGES
    ):
        raise VisaIntegrityError("Development resume identity/stage mismatch")
    for record in document["files"].values():
        if sha256_file(safe_asset(root, record["path"])) != record["sha256"]:
            raise VisaIntegrityError("Development artifact corruption")
    return document


def advance(directory: Path, state: dict, stage: str) -> None:
    if stage not in STAGES or STAGES.index(stage) < STAGES.index(state["stage"]):
        raise VisaIntegrityError("Development stage cannot regress")
    state["stage"] = stage
    state["stage_history"].append(stage)
    publish_journal(directory / "worker-state.json", state)


def run(request: dict) -> dict:
    validate_request_scope(request)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"

    from visionguard.efficientad import (
        canonical_checkpoint_sha256,
        verify_file_identity,
    )
    from visionguard.efficientad_benchmark import (
        _build_training,
        _load_checkpoint,
        _load_transforms,
        _native_quantiles,
        _penalty_paths,
        _restore_rng_state,
        _rng_state,
        _train,
    )
    from visionguard.experiment import ReproducibilityConfig
    from visionguard.reproducibility import configure_reproducibility
    from visionguard.visa_engineering import execution_identity
    from visionguard.visa_smoke import _deny_sealed_access

    repository, root, directory = (
        Path(request[k]).resolve() for k in ("repository", "root", "directory")
    )
    model_name, category, seed = (request[k] for k in ("model", "category", "seed"))
    relative = directory.relative_to(root).parts
    if (
        len(relative) != 4
        or relative[:3] != ("runs", category, f"seed-{seed}")
        or not relative[3].startswith("attempt-")
    ):
        raise VisaIntegrityError("Worker directory is not an immutable cell attempt")
    origin = json.loads((directory / "origin.json").read_text())
    if (
        origin["identity"] != request["identity"]
        or origin["category"] != category
        or origin["seed"] != seed
        or relative[3] != f"attempt-{origin['attempt']}"
    ):
        raise VisaIntegrityError("Worker immutable attempt origin mismatch")
    protocol = verified_context(repository, model_name)
    _, live_identity = execution_identity(
        repository,
        model_name,
        extra_sources=(
            "src/visionguard/visa_dispatcher.py",
            "src/visionguard/visa_development_worker.py",
            "configs/engineering/visa-dispatcher-acceptance-v1.yaml",
        ),
    )
    # Parent scope/assets are separately fixed; current code/environment must match.
    identity = request["identity"]
    if any(identity[k] != v for k, v in live_identity.items()):
        raise VisaIntegrityError("Worker code/environment differs from origin")
    cell_identity = {**identity, "category": category, "seed": seed}
    development = Path(request["development"])
    dataset = DevelopmentDataset(
        development,
        repository / "reports/phase4c-visa-readiness/development-membership.json",
        expected_sha256=MEMBERSHIP,
    )
    _deny_sealed_access(development)
    scope = identity["scope"]
    fit = dataset.paths(category, "fit", limit=scope["fit_limit"])
    calibration = dataset.paths(
        category, "calibration", limit=scope["calibration_limit"]
    )
    ids = [
        r["sample_id"]
        for r in dataset.records
        if r["category"] == category and r["role"] == "fit"
    ][: scope["fit_limit"]]
    previous = (
        None
        if request["resume"] is None
        else validate_receipt(root, request["resume"], cell_identity)
    )
    state = {
        "identity": cell_identity,
        "stage": "created",
        "stage_history": ["created"],
        "files": {},
        "resume_parent": request["resume"],
        "status": "running",
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
    }
    advance(directory, state, "fit")
    configure_reproducibility(ReproducibilityConfig(seed, True, False))
    science = protocol["scientific"]
    stop = request["stop_at"]

    def bind(name, path):
        state["files"][name] = file_record(root, path)
        publish_journal(directory / "worker-state.json", state)

    def stop_after(stage):
        if stop == f"after_{stage}":
            os._exit(75)

    prior_files = {} if previous is None else previous["files"]
    if model_name == "patchcore":
        model, transform = build_patchcore(science, Path(request["weight"]), "cuda")
    else:
        if (
            identity["teacher_sha256"] != science["model"]["teacher_weight_sha256"]
            or identity["imagenette_sha256"]
            != science["auxiliary_data"]["archive_sha256"]
        ):
            raise VisaIntegrityError("Worker assets differ from canonical protocol")
        verify_file_identity(
            Path(request["teacher"]), identity["teacher_sha256"], "teacher"
        )
        verify_file_identity(
            Path(request["imagenette_archive"]),
            identity["imagenette_sha256"],
            "ImageNette",
        )
        model, optimizer, scheduler = _build_training(
            teacher_weight=Path(request["teacher"]), device="cuda", protocol=science
        )
        transform, penalty_transform = _load_transforms()

    def load_model(record):
        payload = _load_checkpoint(root / record["path"], record["sha256"])
        if payload["identity"] != cell_identity:
            raise VisaIntegrityError("Model checkpoint origin mismatch")
        model.load_state_dict(payload["model_state"])
        _restore_rng_state(payload["rng"])
        if (
            canonical_checkpoint_sha256(model.state_dict())
            != payload["canonical_model_sha256"]
        ):
            raise VisaIntegrityError("Canonical model checkpoint mismatch")

    if "fit_model" in prior_files:
        load_model(prior_files["fit_model"])
        shutil.copyfile(
            root / prior_files["fit_model"]["path"], directory / "fit-model.pt"
        )
    elif model_name == "patchcore":
        journal = EmbeddingJournal(root, directory, cell_identity, ids)
        document = journal.create(prior_files.get("embedding_journal"))
        fit_evidence = fit_patchcore(
            model,
            transform,
            fit,
            journal,
            document,
            device="cuda",
            ratio=0.01,
            exit_after=4 if stop == "embedding:4" else None,
            stop_before_coreset=stop == "before_coreset",
        )
        publish_journal(directory / "fit-evidence.json", fit_evidence)
        bind("embedding_journal", journal.path)
        bind("fit_evidence", directory / "fit-evidence.json")
    else:
        entry = {}
        training_manifest = {"identity": cell_identity, "entry": entry}
        checkpoint = None
        if "training_checkpoint" in prior_files:
            r = prior_files["training_checkpoint"]
            checkpoint = _load_checkpoint(root / r["path"], r["sha256"])
        limit = 1 if stop == "step:1" else scope["steps"]
        try:
            _train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_paths=fit,
                penalty_paths=_penalty_paths(Path(request["imagenette_root"])),
                normal_transform=transform,
                penalty_transform=penalty_transform,
                device="cuda",
                protocol=science,
                identity=cell_identity,
                entry=entry,
                manifest=training_manifest,
                manifest_path=directory / "training-state.json",
                checkpoint_path=directory / "training-checkpoint.pt",
                output_root=root,
                checkpoint=checkpoint,
                stop_after_step=limit if limit < 70000 else None,
            )
        except KeyboardInterrupt:
            if entry.get("latest_valid_checkpoint", {}).get("step") != limit:
                raise
            bind("training_checkpoint", directory / "training-checkpoint.pt")
            if limit != scope["steps"]:
                os._exit(75)
        bind("training_checkpoint", directory / "training-checkpoint.pt")

    def save_model(path):
        durable_torch_save(
            path,
            {
                "identity": cell_identity,
                "model_state": model.state_dict(),
                "rng": _rng_state(),
                "canonical_model_sha256": canonical_checkpoint_sha256(
                    model.state_dict()
                ),
            },
        )

    if not (directory / "fit-model.pt").exists():
        save_model(directory / "fit-model.pt")
    bind("fit_model", directory / "fit-model.pt")
    stop_after("fit")
    advance(directory, state, "normalization")
    if "normalized_model" in prior_files:
        load_model(prior_files["normalized_model"])
        shutil.copyfile(
            root / prior_files["normalized_model"]["path"],
            directory / "normalized-model.pt",
        )
        if "normalization" in prior_files:
            shutil.copyfile(
                root / prior_files["normalization"]["path"],
                directory / "normalization.json",
            )
            bind("normalization", directory / "normalization.json")
    else:
        if model_name == "efficientad":
            quantiles = _native_quantiles(model, fit, transform, "cuda")
            publish_journal(
                directory / "normalization.json",
                {"source_role": "fit", "quantiles": quantiles},
            )
            bind("normalization", directory / "normalization.json")
        save_model(directory / "normalized-model.pt")
    bind("normalized_model", directory / "normalized-model.pt")
    stop_after("normalization")
    advance(directory, state, "calibration")
    if "calibration" in prior_files:
        for name in ("calibration", "continuous_map", "thresholded_map"):
            record = prior_files[name]
            target = directory / Path(record["path"]).name
            shutil.copyfile(root / record["path"], target)
    else:
        calibrate_development(model, transform, calibration, "cuda", directory)
    for name, filename in (
        ("calibration", "calibration.json"),
        ("continuous_map", "calibration-normal.tiff"),
        ("thresholded_map", "calibration-normal-thresholded.png"),
    ):
        bind(name, directory / filename)
    stop_after("calibration")
    advance(directory, state, "artifact_write")
    shutil.copyfile(directory / "normalized-model.pt", directory / "final-model.pt")
    bind("final_model", directory / "final-model.pt")
    state["canonical_final_model_sha256"] = canonical_checkpoint_sha256(
        model.state_dict()
    )
    state["status"] = "development_complete"
    advance(directory, state, "development_complete")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    run(json.loads(args.request.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
