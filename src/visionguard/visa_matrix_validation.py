"""Independent CPU-only validation/compaction of normal-development artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from visionguard.calibration import highest_order_statistic
from visionguard.embedding_journal import publish_journal
from visionguard.visa import safe_asset
from visionguard.visa_acquire import VisaIntegrityError, sha256_file
from visionguard.visa_dispatcher import execution_scope
from visionguard.visa_execution import ExecutionState
from visionguard.visa_protocol import canonical_fingerprint


def require(condition, message):
    if not condition:
        raise VisaIntegrityError(message)


def load(path):
    return json.loads(path.read_text())


def finite(value):
    """Recursively check floats/tensors without converting RNG integers."""
    import torch

    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Nonfinite model/state tensor")
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite artifact value")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def validate_calibration(calibration: dict, records: list[dict]) -> None:
    rows = calibration["inputs"]
    require(len(rows) == len(records) >= 19, "Wrong calibration count")
    for row, record in zip(rows, records, strict=True):
        require(
            row["normal_id"] == Path(record["path"]).name,
            "Wrong calibration membership/order",
        )
        shape = [record["image_identity"]["height"], record["image_identity"]["width"]]
        require(row["shape"] == shape, "Wrong calibration original shape")
        require(
            len(row["map_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["map_sha256"]),
            "Missing calibration map hash",
        )
        require(
            math.isfinite(row["score"]) and math.isfinite(row["pixel_maximum"]),
            "Nonfinite calibration values",
        )
    for name, field in (("image", "score"), ("pixel", "pixel_maximum")):
        expected = asdict(
            highest_order_statistic([r[field] for r in rows], minimum_samples=19)
        )
        require(calibration[name] == expected, "Threshold estimator/denominator drift")


def validate_cell(
    root: Path,
    model: str,
    category: str,
    seed: int,
    expected_identity: dict,
    membership: dict,
    *,
    engineering=False,
) -> dict:
    """Hash/semantic validation; does not instantiate a model or read an image."""
    from visionguard.efficientad import canonical_checkpoint_sha256
    from visionguard.efficientad_benchmark import _load_checkpoint
    from visionguard.visa_patchcore import tensor_hash

    state = ExecutionState(root, expected_identity)
    require(state.validate_completed(category, seed), "Cell not durably complete")
    require(
        expected_identity["scope"] == execution_scope(model, engineering),
        "Wrong execution scope",
    )
    identity = {**expected_identity, "category": category, "seed": seed}
    attempts = state.read()["cells"][f"{category}:{seed}"]["attempts"]
    files, receipts = {}, []
    for number, attempt in enumerate(attempts, 1):
        directory = safe_asset(root, attempt["directory"] + "/origin.json").parent
        origin = load(directory / "origin.json")
        require(
            origin
            == {
                "identity": expected_identity,
                "category": category,
                "seed": seed,
                "attempt": number,
            },
            "Attempt origin/order differs",
        )
        require(attempt["attempt"] == number, "Attempt sequence gap")
        require(
            attempt["status"] in ("failed", "interrupted", "development_complete"),
            "Nonterminal attempt in completed chain",
        )
        receipt_path = directory / "worker-state.json"
        if receipt_path.exists():
            receipt = load(receipt_path)
            require(receipt["identity"] == identity, "Worker origin differs")
            require(
                receipt["final_test_lock"] == "closed"
                and receipt["test_performance_evaluated"] is False,
                "Test boundary violated",
            )
            for name, record in receipt["files"].items():
                require(
                    sha256_file(safe_asset(root, record["path"])) == record["sha256"],
                    "Attempt artifact corruption",
                )
                files[name] = record
            receipts.append(
                {
                    "path": receipt_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(receipt_path),
                }
            )
    required = {"final_model", "normalized_model", "fit_model", "calibration"}
    required |= (
        {"embedding_journal", "fit_evidence"}
        if model == "patchcore"
        else {"training_checkpoint", "normalization"}
    )
    require(required <= files.keys(), "Missing final artifact contract")
    final_receipt = load(root / attempts[-1]["directory"] / "worker-state.json")

    def checkpoint(name):
        record = files[name]
        payload = _load_checkpoint(root / record["path"], record["sha256"])
        require(payload["identity"] == identity, "Checkpoint origin mismatch")
        finite(payload["model_state"])
        digest = canonical_checkpoint_sha256(payload["model_state"])
        require(digest == payload["canonical_model_sha256"], "Canonical model differs")
        return payload, digest

    final, digest = checkpoint("final_model")
    require(
        digest == final_receipt["canonical_final_model_sha256"], "Final receipt differs"
    )
    normalized, normalized_digest = checkpoint("normalized_model")
    require(normalized_digest == digest, "Final model not normalized final model")
    del normalized
    fitted, fit_digest = checkpoint("fit_model")
    del fitted
    fit_records = [
        r
        for r in membership["records"]
        if r["category"] == category and r["role"] == "fit"
    ]
    calibration_records = [
        r
        for r in membership["records"]
        if r["category"] == category and r["role"] == "calibration"
    ]
    if engineering:
        fit_records = fit_records[: 8 if model == "patchcore" else 4]
        if model == "efficientad":
            calibration_records = calibration_records[:19]
    calibration = load(root / files["calibration"]["path"])
    validate_calibration(calibration, calibration_records)
    result = {
        "model": model,
        "category": category,
        "seed": seed,
        "identity": identity,
        "attempts": attempts,
        "receipts": receipts,
        "files": files,
        "canonical_model_sha256": digest,
        "calibration": calibration,
        "fit_inventory_sha256": canonical_fingerprint(fit_records),
        "calibration_inventory_sha256": canonical_fingerprint(calibration_records),
        "fit_count": len(fit_records),
        "calibration_count": len(calibration_records),
        "final_test_lock": "closed",
        "test_performance_evaluated": False,
        "status": "validated",
        "engineering_only": engineering,
    }
    if model == "patchcore":
        evidence = load(root / files["fit_evidence"]["path"])
        journal = load(root / files["embedding_journal"]["path"])
        require(
            journal["identity"] == identity
            and journal["sample_ids"] == [r["sample_id"] for r in fit_records],
            "Embedding membership differs",
        )
        require(
            journal["next_index"] == len(fit_records) == len(journal["chunks"]),
            "Incomplete embedding sequence",
        )
        for index, chunk in enumerate(journal["chunks"]):
            require(
                chunk["index"] == index
                and chunk["sample_id"] == fit_records[index]["sample_id"],
                "Embedding sequence reordered",
            )
        bank = final["model_state"]["memory_bank"]
        require(
            evidence["coreset_completed"] is True
            and evidence["dtype"] == "torch.float32",
            "Incomplete/wrong coreset",
        )
        require(
            evidence["embedding_shape"] == [len(fit_records) * 1024, 1536],
            "Wrong full embedding shape",
        )
        require(
            list(bank.shape) == [int(len(fit_records) * 1024 * 0.01), 1536],
            "Wrong bank size",
        )
        require(
            tensor_hash(bank) == evidence["memory_bank_sha256"], "Bank tensor differs"
        )
        indices = evidence["coreset_indices"]
        require(
            len(indices) == len(bank)
            and all(
                type(i) is int and 0 <= i < len(fit_records) * 1024 for i in indices
            ),
            "Invalid coreset indices",
        )
        require(fit_digest == digest, "PatchCore model changed after fit")
        result["fit_evidence"] = evidence
        result["chunk_inventory_sha256"] = canonical_fingerprint(journal["chunks"])
        result["coreset_indices_sha256"] = canonical_fingerprint(indices)
        result["active_fit_seconds"] = (
            evidence["extraction_seconds_this_process"]
            + evidence["coreset_and_reconstruction_seconds"]
        )
        result["active_time_limit"] = (
            "successful fit segment; excludes calibration and earlier interrupted work"
        )
    else:
        record = files["training_checkpoint"]
        training = _load_checkpoint(root / record["path"], record["sha256"])
        expected_step = 2 if engineering else 70000
        require(
            training["identity"] == identity and training["step"] == expected_step,
            "Wrong final optimization step/origin",
        )
        require(
            training["scheduler_state"]["last_epoch"] == expected_step
            and training["scheduler_state"]["step_size"] == 66500,
            "Wrong scheduler state",
        )
        finite(training["optimizer_state"])
        require(
            canonical_checkpoint_sha256(training["model_state"]) == fit_digest,
            "Fit model differs from final-step checkpoint",
        )
        for name in ("python", "numpy", "torch_cpu", "torch_cuda"):
            require(name in training["rng"], "Incomplete RNG state")
        for name, size in (
            ("train_stream", len(fit_records)),
            ("penalty_stream", None),
        ):
            stream = training[name]
            require(
                {"order", "position", "generator_state"} <= stream.keys(),
                "Incomplete durable stream",
            )
            if size is not None:
                require(
                    len(stream["order"]) == size,
                    "Training stream membership length differs",
                )
        normalization = load(root / files["normalization"]["path"])
        require(
            normalization["source_role"] == "fit", "Normalization source is not fit"
        )
        for name in ("qa_st", "qb_st", "qa_ae", "qb_ae"):
            require(
                math.isfinite(normalization["quantiles"][name])
                and float(final["model_state"][f"quantiles.{name}"])
                == normalization["quantiles"][name],
                "Normalization evidence differs",
            )
        result["normalization"] = normalization
        result["final_optimization_step"] = expected_step
        result["active_training_seconds"] = training["active_training_seconds"]
        result["active_time_limit"] = (
            "native checkpoint training timer; may include OS sleep, "
            "excludes normalization/calibration"
        )
    return result


def compact_embeddings(root: Path, result: dict, receipt_path: Path) -> dict:
    """Only exact completed-cell .pt chunks; no directory deletion or final models."""
    require(
        result["model"] == "patchcore" and result["status"] == "validated",
        "Compaction requires independently validated PatchCore",
    )
    category, seed = result["category"], result["seed"]
    manifest = load(root / "manifest.json")
    require(
        manifest["cells"][f"{category}:{seed}"]["attempts"] == result["attempts"],
        "Completion manifest changed before compaction",
    )
    require(
        result["attempts"][-1]["status"] == "development_complete",
        "Active/incomplete cell cannot compact",
    )
    require(not (root / ".writer.lock").exists(), "Cell writer active")
    cell_root = (root / "runs" / category / f"seed-{seed}").resolve()
    require(cell_root.is_relative_to(root.resolve()), "Compaction outside model root")
    journal_record = result["files"]["embedding_journal"]
    require(
        sha256_file(root / journal_record["path"]) == journal_record["sha256"],
        "Compaction journal changed",
    )
    chunks = load(root / journal_record["path"])["chunks"]
    require(
        canonical_fingerprint(chunks) == result["chunk_inventory_sha256"],
        "Inventory differs",
    )
    binding = {
        "category": category,
        "seed": seed,
        "status": "prepared",
        "chunk_inventory_sha256": result["chunk_inventory_sha256"],
        "ordered_embeddings_sha256": result["fit_evidence"][
            "ordered_embeddings_sha256"
        ],
        "membership_sha256": result["identity"]["membership_sha256"],
        "coreset_indices_sha256": result["coreset_indices_sha256"],
        "memory_bank_sha256": result["fit_evidence"]["memory_bank_sha256"],
        "final_model_sha256": result["files"]["final_model"]["sha256"],
        "calibration_sha256": result["files"]["calibration"]["sha256"],
        "bytes": sum(r["size_bytes"] for r in chunks),
        "chunks": len(chunks),
        "recoverable": (
            "recompute exact embeddings from retained normals "
            "and frozen code; not trash"
        ),
    }
    prior = load(receipt_path) if receipt_path.exists() else None
    if prior:
        require(
            {**prior, "status": "prepared"} == binding, "Compaction receipt differs"
        )
    targets = []
    for chunk in chunks:
        path = (root / chunk["path"]).resolve()
        require(
            path.is_relative_to(cell_root)
            and path.parent.name == "chunks"
            and path.suffix == ".pt"
            and path.stem.isdigit()
            and not (root / chunk["path"]).is_symlink(),
            "Unsafe compaction target",
        )
        if path.exists():
            require(
                sha256_file(path) == chunk["sha256"]
                and path.stat().st_size == chunk["size_bytes"],
                "Chunk corruption before deletion",
            )
        else:
            require(
                prior is not None, "Missing chunk without durable compaction intent"
            )
        targets.append(path)
    if prior is None:
        publish_journal(receipt_path, binding)
    for path in targets:
        path.unlink(missing_ok=True)
    result = {**binding, "status": "completed"}
    publish_journal(receipt_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    request = load(args.request)
    result = validate_cell(
        Path(request["root"]),
        request["model"],
        request["category"],
        request["seed"],
        request["identity"],
        load(Path(request["membership"])),
        engineering=request.get("engineering", False),
    )
    publish_journal(Path(request["output"]), result)


if __name__ == "__main__":
    main()
