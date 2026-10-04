"""Invocation-bound human pause controls. Only output receipts, never inputs.

REQUESTED is not PAUSED. SAFE_PAUSED additionally requires independent process
absence; inaccessible process metadata raises instead of assuming absence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from visionguard.heldout_paths import checked
from visionguard.heldout_storage import json_once
from visionguard.heldout_watcher import process_identity
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require
from visionguard.visa_evaluator_storage import canonical_bytes


class ControlledPause(InterruptedError):
    """A validated cooperative stop, never a successful scientific completion."""


def identifier(value: str) -> str:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value), "Invalid ID")
    return value


def read(path: Path) -> dict:
    return json.loads(checked(path).read_bytes())


def absent(pid: int, identity: str) -> bool:
    """PID reuse is not the original process. Inspection failures remain UNKNOWN."""
    require(type(pid) is int and pid > 0, "Invalid process PID")
    if os.name == "nt":
        script = (
            "try {$p=Get-Process -Id "
            + str(pid)
            + " -ErrorAction Stop; $p.StartTime.ToUniversalTime().Ticks} catch {"
            "if ($_.FullyQualifiedErrorId -like 'NoProcessFoundForGivenId,*') "
            "{'ABSENT'} else {throw}}"
        )
        actual = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        require(actual == "ABSENT" or actual.isdigit(), "Unresolved process inspection")
        return actual == "ABSENT" or actual != identity
    try:
        return process_identity(pid) != identity
    except FileNotFoundError:
        return True


def binding(owner: dict, owner_sha: str) -> dict:
    origin = owner["origin"]
    artificial = owner["artificial"]
    return {
        "invocation": owner["nonce"],
        "owner_sha256": owner_sha,
        "authorization_id": origin.get("authorization_id") if not artificial else None,
        "claim_sha256": origin.get("claim_sha256")
        if not artificial
        else hashlib.sha256(canonical_bytes(origin)).hexdigest(),
        "artificial": artificial,
    }


def owner_at(root: Path, nonce: str) -> tuple[Path, dict, dict]:
    directory = checked(root / ("invocation-" + identifier(nonce)))
    owner = read(directory / "owner.json")
    require(owner["nonce"] == nonce, "Invocation owner mismatch")
    return directory, owner, binding(owner, sha256_file(directory / "owner.json"))


def validate_claim(root: Path, owner: dict, bound: dict) -> None:
    if owner["artificial"]:
        return
    claim = (
        root.parent.parent
        / "authorization-claims"
        / (identifier(bound["authorization_id"]) + ".json")
    )
    require(
        sha256_file(checked(claim)) == bound["claim_sha256"]
        and read(claim)["run_root"] == str(root),
        "Missing/mismatched original claim",
    )


def request(root: Path, nonce: str) -> dict:
    directory, owner, bound = owner_at(root, nonce)
    validate_claim(root, owner, bound)
    require(read(root / "writer.lock") == owner, "Not the active invocation")
    require(
        process_identity(owner["pid"]) == owner["process_identity"], "Owner not active"
    )
    target = directory / "pause-request.json"
    if target.exists():
        old = read(target)
        validate_request(old, bound)
        return {"status": "REQUESTED", "request_sha256": sha256_file(target)}
    record = bound | {
        "requested_at": datetime.now(UTC).isoformat(),
        "requester_pid": os.getpid(),
        "requester_process_identity": process_identity(os.getpid()),
    }
    try:
        receipt = json_once(target, record)
    except FileExistsError:
        validate_request(read(target), bound)
        receipt = {"sha256": sha256_file(target)}
    return {"status": "REQUESTED", "request_sha256": receipt["sha256"]}


def validate_request(value: dict, bound: dict) -> None:
    require(
        set(value)
        == set(bound) | {"requested_at", "requester_pid", "requester_process_identity"}
        and {k: value[k] for k in bound} == bound,
        "Invalid/stale pause request binding",
    )
    require(
        type(value["requester_pid"]) is int
        and value["requester_pid"] > 0
        and type(value["requester_process_identity"]) is str
        and bool(value["requester_process_identity"]),
        "Invalid requester provenance",
    )
    when = datetime.fromisoformat(value["requested_at"])
    require(
        when.tzinfo is not None and when <= datetime.now(UTC), "Invalid request time"
    )


def checkpoint(directory: Path, owner: dict, label: str) -> None:
    path = directory / "pause-request.json"
    if not path.exists():
        return
    bound = binding(owner, sha256_file(directory / "owner.json"))
    validate_request(read(path), bound)
    validate_claim(directory.parent, owner, bound)
    json_once(
        directory / "pause-acknowledged.json",
        bound | {"request_sha256": sha256_file(path), "checkpoint": label},
    )
    raise ControlledPause("Human pause acknowledged at " + label)


def verify_safe(root: Path, nonce: str, *, expected_origin=None) -> dict:
    require(
        not (root / "watcher-stop.json").exists(),
        "Recorded watcher STOP: human review required before resume",
    )
    directory, owner, bound = owner_at(root, nonce)
    validate_claim(root, owner, bound)
    safe = read(directory / "safe-exit.json")
    require(
        not (directory / "cleanup-failure.json").exists(),
        "Cleanup failed: human review required",
    )
    require(
        safe["binding"] == bound and safe["origin"] == owner["origin"],
        "Safe-exit origin mismatch",
    )
    require(
        expected_origin is None or owner["origin"] == expected_origin,
        "Resume origin mismatch",
    )
    require(
        safe["status"] == "COOPERATIVE_STOP_CLEANUP_VERIFIED"
        and safe["gpu_allocated_bytes"] == 0
        and safe["writer_lock_released"] is True
        and sha256_file(directory / "failure.json") == safe["failure_sha256"]
        and read(directory / "failure.json")["status"] == "interrupted",
        "Invalid safe interruption evidence",
    )
    if safe["request_sha256"] is not None:
        validate_request(read(directory / "pause-request.json"), bound)
        ack = read(directory / "pause-acknowledged.json")
        require(
            sha256_file(directory / "pause-request.json") == safe["request_sha256"]
            and ack
            == bound
            | {
                "request_sha256": safe["request_sha256"],
                "checkpoint": safe["checkpoint"],
            }
            and sha256_file(directory / "pause-acknowledged.json")
            == safe["ack_sha256"],
            "Pause acknowledgement mismatch",
        )
    watcher = safe["watcher"]
    if not owner["artificial"]:
        require(
            type(watcher) is dict and watcher["exit_code"] == 0, "Watcher exit missing"
        )
    if watcher:
        stopped = root / ("watcher-" + nonce) / "stopped.json"
        require(
            sha256_file(checked(stopped)) == watcher["sha256"], "Watcher proof corrupt"
        )
        proof = read(stopped)
        require(
            proof["owner_nonce"] == nonce
            and proof["safe_owner_exit"] is True
            and proof["owner_pid"] == owner["pid"]
            and proof["owner_identity"] == owner["process_identity"]
            and sha256_file(root / ("watcher-" + nonce) / "finish.json")
            == proof["finish_sha256"],
            "Watcher origin mismatch",
        )
        require(
            absent(proof["pid"], proof["process_identity"]), "Watcher still running"
        )
    require(
        not (root / "writer.lock").exists(), "Writer lock present: no automatic cleanup"
    )
    require(
        absent(owner["pid"], owner["process_identity"]),
        "Runner exit not independently confirmed",
    )
    # A later invocation invalidates this receipt as a resume parent.
    for path in root.glob("invocation-*"):
        other = read(path / "owner.json")
        require(
            other["started"] <= owner["started"], "Later invocation requires review"
        )
    return safe


def status(root: Path, nonce: str) -> dict:
    directory, _, _ = owner_at(root, nonce)
    if (directory / "safe-exit.json").exists():
        verify_safe(root, nonce)
        return {
            "status": "SAFE_PAUSED",
            "safe_exit_sha256": sha256_file(directory / "safe-exit.json"),
        }
    if (directory / "pause-acknowledged.json").exists():
        return {"status": "ACKNOWLEDGED_NOT_SAFE"}
    if (directory / "pause-request.json").exists():
        return {"status": "REQUESTED_NOT_SAFE"}
    return {"status": "NO_SAFE_PAUSE_EVIDENCE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--active", action="store_true")
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--invocation")
    args = parser.parse_args()
    from visionguard.heldout_contract import context, verify_freeze

    verify_freeze(args.repository)
    ctx = context(args.repository)
    require(
        str(args.repository.absolute()) == ctx["activation"]["canonical_repository"],
        "Canonical repository required",
    )
    root = (
        args.repository
        / ctx["activation"]["output_relative"]
        / "runs"
        / identifier(args.authorization_id)
    )
    if args.active:
        owner = read(root / "writer.lock")
        result = {
            "invocation": owner["nonce"],
            "owner_pid": owner["pid"],
            "process_identity": owner["process_identity"],
        }
    else:
        _, owner, bound = owner_at(root, identifier(args.invocation))
        require(
            not owner["artificial"]
            and bound["authorization_id"] == args.authorization_id,
            "Wrong authorization ID",
        )
        result = (
            request(root, args.invocation)
            if args.request
            else status(root, args.invocation)
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
