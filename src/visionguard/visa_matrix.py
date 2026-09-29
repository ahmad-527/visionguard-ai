"""Durable sequential normal-only matrix supervisor. No test evaluation route."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from visionguard.embedding_journal import publish_journal
from visionguard.visa_acquire import sha256_file
from visionguard.visa_dispatcher import execution_scope
from visionguard.visa_execution import exclusive_writer
from visionguard.visa_matrix_contract import MODELS, verify_execution
from visionguard.visa_matrix_validation import compact_embeddings, load, require


def now():
    return datetime.now(UTC).isoformat()


def file_bytes(directory):
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def counts(document):
    result = {}
    for model in MODELS:
        rows = [v for k, v in document["cells"].items() if k.startswith(model + ":")]
        result[model] = {
            s: sum(r["status"] == s for r in rows)
            for s in ("completed", "active", "interrupted", "failed", "pending")
        }
    result["completed_total"] = sum(v["completed"] for v in result.values())
    result["expected_total"] = 72
    return result


def disk_check(root, contract):
    free = shutil.disk_usage(root).free
    require(
        free >= contract["storage"]["minimum_free_bytes"], "Disk pressure; STOP MATRIX"
    )
    return free


def powershell(command):
    return subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        text=True,
        timeout=30,
    ).strip()


def hardware_observation(since: str):
    """Read-only Windows telemetry. Missing safety telemetry fails closed."""
    require(os.name == "nt", "This measured execution contract requires Windows")
    # Only agent-generated ISO timestamps enter this fixed script.
    datetime.fromisoformat(since)
    boot = powershell(
        "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime"
        ".ToUniversalTime().ToString('o')"
    )
    events = powershell(
        "$ErrorActionPreference='Stop'; try { "
        "@(Get-WinEvent -FilterHashtable @{LogName='System'; "
        f"StartTime=[datetime]::Parse('{since}'); "
        "ProviderName='Microsoft-Windows-WHEA-Logger'} | "
        "Select-Object Id,RecordId).Count } catch { "
        "if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') "
        "{ 0 } else { throw } }"
    )
    require(int(events) == 0, "WHEA event observed; STOP MATRIX")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    ).strip()
    return {"at": now(), "boot": boot, "gpu_csv": gpu, "whea_events_since_start": 0}


def command(repository, root, local, model, category, seed, resume):
    result = [
        sys.executable,
        "-m",
        "visionguard.visa_dispatcher",
        "--repository",
        str(repository),
        "--model",
        model,
        "--output",
        str(root),
        "--development",
        local["development"],
        "--confirm-full-development-fit",
        "--resume-cell" if resume else "--run-cell",
        category,
        str(seed),
    ]
    assets = (
        ("weight",)
        if model == "patchcore"
        else ("teacher", "imagenette_root", "imagenette_archive")
    )
    for name in assets:
        result.extend(["--" + name.replace("_", "-"), local[name]])
    return result


def expected_identity(repository, root, model, contract, commit):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionguard.visa_dispatcher",
            "--repository",
            str(repository),
            "--model",
            model,
            "--output",
            str(root),
            "--identity-only",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    identity = json.loads(result.stdout)
    require(
        identity["environment"] == contract["environment"]
        and identity["implementation_commit"] == commit,
        "Environment/implementation drift",
    )
    identity["scope"] = execution_scope(model, False)
    assets = contract["asset_sha256"]
    if model == "patchcore":
        identity["weight_sha256"] = assets["weight"]
    else:
        identity["teacher_sha256"] = assets["teacher"]
        identity["imagenette_sha256"] = assets["imagenette_archive"]
    return identity


def validate_subprocess(repository, root, model, category, seed, identity, output):
    request = {
        "root": str(root / model),
        "model": model,
        "category": category,
        "seed": seed,
        "identity": identity,
        "membership": str(
            repository / "reports/phase4c-visa-readiness/development-membership.json"
        ),
        "output": str(output),
    }
    request_path = output.with_suffix(".request.json")
    publish_journal(request_path, request)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "visionguard.visa_matrix_validation",
            "--request",
            str(request_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return load(output)


def terminate_tree(pid):
    """Stop only this supervisor's directly recorded child process tree."""
    require(type(pid) is int and pid > 0, "Invalid owned child PID")
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def check_assets(local, contract, repository):
    for name, digest in contract["asset_sha256"].items():
        require(sha256_file(Path(local[name])) == digest, "Asset identity changed")
    inventory = repository / "outputs/phase4d-a-preflight/imagenette-inventory.json"
    require(
        sha256_file(inventory) == contract["auxiliary_inventory_sha256"],
        "Auxiliary inventory changed",
    )
    for row in load(inventory)["records"]:
        path = Path(local["imagenette_root"]) / row["path"]
        require(sha256_file(path) == row["sha256"], "Extracted auxiliary data changed")


def run(repository, root, local, fingerprint):
    contract, fingerprint, commit = verify_execution(repository, fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    require(
        root.resolve().is_relative_to((repository / "outputs").resolve()),
        "Matrix output must be under ignored outputs",
    )
    path = root / "matrix-manifest.json"
    with exclusive_writer(root):
        document = (
            load(path)
            if path.exists()
            else {
                "execution_fingerprint": fingerprint,
                "implementation_commit": commit,
                "started_at": now(),
                "final_test_lock": "closed",
                "test_performance_evaluated": False,
                "human_access_history": "UNKNOWN",
                "independent_reservation": "NOT ESTABLISHED",
                "cells": {
                    k: {"status": "pending", "history": []} for k in contract["order"]
                },
                "status": "initialized",
                "completed_model_validations": {},
            }
        )
        require(
            document["execution_fingerprint"] == fingerprint
            and document["implementation_commit"] == commit,
            "Matrix origin mismatch",
        )
        require(
            document["status"] not in ("stopped", "running"),
            "Unclassified failure/abrupt supervisor interruption needs review",
        )
        # Do not trust alphabetical JSON key order to define the scientific schedule.
        require(
            set(document["cells"]) == set(contract["order"]), "Matrix coverage mismatch"
        )
        current_key = None
        child = None
        try:
            observation = hardware_observation(document["started_at"])
            if "boot" in document:
                require(
                    document["boot"] == observation["boot"], "Unexpected reboot; STOP"
                )
            document["boot"] = observation["boot"]
            document["supervisor"] = {"pid": os.getpid(), "started_at": now()}
            check_assets(local, contract, repository)
            identities = {}
            # Never initialize a CUDA context in this long-lived supervisor.
            with exclusive_writer(repository / "outputs/visa-development-gpu-lease"):
                for model in MODELS:
                    identities[model] = expected_identity(
                        repository, root, model, contract, commit
                    )
            document["identities"] = identities
            document["status"] = "running"
            publish_journal(path, document)
            for model in MODELS:
                for key in [k for k in contract["order"] if k.startswith(model + ":")]:
                    current_key = key
                    _, category, seed_text = key.split(":")
                    seed = int(seed_text)
                    row = document["cells"][key]
                    verify_execution(repository, fingerprint)
                    disk_check(root, contract)
                    if (root / "PAUSE_AFTER_CELL").exists():
                        document["status"] = "paused_between_cells"
                        publish_journal(path, document)
                        return
                    model_manifest = root / model / "manifest.json"
                    prior = (
                        []
                        if not model_manifest.exists()
                        else load(model_manifest)["cells"][f"{category}:{seed}"][
                            "attempts"
                        ]
                    )
                    if prior:
                        require(
                            load(model_manifest)["identity"] == identities[model],
                            "Model matrix origin mismatch",
                        )
                    while not prior or prior[-1]["status"] != "development_complete":
                        require(
                            not prior or prior[-1]["status"] == "interrupted",
                            "Unresolved failed/active attempt; STOP",
                        )
                        attempt = len(prior) + 1
                        log = (
                            root
                            / "supervisor-logs"
                            / f"{model}-{category}-{seed}-{attempt}.log"
                        )
                        log.parent.mkdir(exist_ok=True)
                        row["status"] = "active"
                        row["history"].append(
                            {"at": now(), "event": "launch", "attempt": attempt}
                        )
                        publish_journal(path, document)
                        started = time.perf_counter()
                        with log.open("x", encoding="utf-8") as stream:
                            child = subprocess.Popen(
                                command(
                                    repository,
                                    root,
                                    local,
                                    model,
                                    category,
                                    seed,
                                    bool(prior),
                                ),
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                            )
                            row["dispatcher_pid"] = child.pid
                            publish_journal(path, document)
                            while True:
                                try:
                                    code = child.wait(
                                        timeout=contract["monitor_seconds"]
                                    )
                                    break
                                except subprocess.TimeoutExpired:
                                    verify_execution(repository, fingerprint)
                                    free = disk_check(root, contract)
                                    observation = hardware_observation(
                                        document["started_at"]
                                    )
                                    require(
                                        observation["boot"] == document["boot"],
                                        "Boot identity changed",
                                    )
                                    with (root / "telemetry.jsonl").open(
                                        "a", encoding="utf-8"
                                    ) as telemetry:
                                        telemetry.write(
                                            json.dumps(
                                                {
                                                    **observation,
                                                    "cell": key,
                                                    "disk_free_bytes": free,
                                                }
                                            )
                                            + "\n"
                                        )
                                    document["heartbeat_at"] = now()
                                    publish_journal(path, document)
                        child = None
                        row["history"].append(
                            {
                                "at": now(),
                                "event": "process_exit",
                                "attempt": attempt,
                                "returncode": code,
                                "wall_seconds": time.perf_counter() - started,
                            }
                        )
                        require(code == 0, "Dispatcher failed; STOP before next cell")
                        prior = load(model_manifest)["cells"][f"{category}:{seed}"][
                            "attempts"
                        ]
                        row["status"] = (
                            "interrupted"
                            if prior[-1]["status"] == "interrupted"
                            else "active"
                        )
                        publish_journal(path, document)
                        disk_check(root, contract)
                    output = root / "validation" / f"{model}-{category}-{seed}.json"
                    validated = validate_subprocess(
                        repository,
                        root,
                        model,
                        category,
                        seed,
                        identities[model],
                        output,
                    )
                    row.update(
                        status="completed",
                        validation={
                            "path": output.relative_to(root).as_posix(),
                            "sha256": sha256_file(output),
                        },
                        validated_at=now(),
                    )
                    cell_directory = root / model / "runs" / category / f"seed-{seed}"
                    row.setdefault(
                        "storage_before_compaction_bytes", file_bytes(cell_directory)
                    )
                    # Atomic matrix completion precedes any allowed compaction.
                    document["counts"] = counts(document)
                    publish_journal(path, document)
                    disk_check(root, contract)
                    if model == "patchcore":
                        compaction_path = (
                            root / "compaction" / f"{category}-{seed}.json"
                        )
                        compact_embeddings(root / model, validated, compaction_path)
                        row["compaction"] = {
                            "path": compaction_path.relative_to(root).as_posix(),
                            "sha256": sha256_file(compaction_path),
                        }
                        publish_journal(path, document)
                    disk_check(root, contract)
                    row["retained_cell_bytes"] = file_bytes(cell_directory)
                    publish_journal(path, document)
                # Fresh CPU semantic/hash validation of all 36 cells before next model.
                validations = {}
                for key in [k for k in contract["order"] if k.startswith(model + ":")]:
                    _, category, seed_text = key.split(":")
                    output = (
                        root
                        / "independent-validation"
                        / f"{model}-{category}-{seed_text}.json"
                    )
                    validate_subprocess(
                        repository,
                        root,
                        model,
                        category,
                        int(seed_text),
                        identities[model],
                        output,
                    )
                    validations[key] = sha256_file(output)
                summary_path = root / f"{model}-development-summary.json"
                publish_journal(
                    summary_path,
                    {
                        "model": model,
                        "validated_cells": 36,
                        "validation_sha256": validations,
                        "execution_fingerprint": fingerprint,
                        "final_test_lock": "closed",
                    },
                )
                document["completed_model_validations"][model] = sha256_file(
                    summary_path
                )
                publish_journal(path, document)
            require(
                counts(document)["completed_total"] == 72,
                "Partial matrix cannot freeze",
            )
            document["status"] = "completed"
            document["finished_at"] = now()
            document["counts"] = counts(document)
            publish_journal(path, document)
            summaries = {
                k: load(root / v["validation"]["path"])
                for k, v in document["cells"].items()
            }
            freeze = {
                "execution_contract": contract,
                "execution_fingerprint": fingerprint,
                "matrix_manifest_sha256": sha256_file(path),
                "validated_cells": 72,
                "cells": summaries,
                "final_test_lock": "closed",
                "test_performance_evaluated": False,
                "human_access_history": "UNKNOWN",
                "independent_reservation": "NOT ESTABLISHED",
                "phase4d_b_started": False,
            }
            publish_journal(root / "development-freeze.json", freeze)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                terminate_tree(child.pid)
            document["status"] = "stopped"
            document["stop"] = {
                "at": now(),
                "type": type(exc).__name__,
                "reason": str(exc),
                "cell": current_key,
            }
            if current_key and document["cells"][current_key]["status"] != "completed":
                document["cells"][current_key]["status"] = "failed"
            document["counts"] = counts(document)
            publish_journal(path, document)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--local-config", type=Path)
    parser.add_argument("--execution-fingerprint")
    args = parser.parse_args(argv)
    if args.status:
        path = args.root / "matrix-manifest.json"
        print(
            json.dumps(
                {"status": "not_started", "completed": 0, "expected": 72}
                if not path.exists()
                else {
                    k: v
                    for k, v in load(path).items()
                    if k
                    in ("status", "counts", "stop", "heartbeat_at", "final_test_lock")
                }
            )
        )
        return
    require(
        args.local_config is not None and args.execution_fingerprint,
        "Explicit frozen contract and local assets required",
    )
    run(
        args.repository.resolve(),
        args.root.resolve(),
        load(args.local_config),
        args.execution_fingerprint,
    )


if __name__ == "__main__":
    main()
