"""Human-signed, protected-registry authorization. No approval/signing installer.

Trust model: local administrators can alter the machine and are outside the
evaluator's threat boundary. The ordinary agent/user must not be able to write
the independent registry. No fingerprint, CLI flag or caller registry is approval.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from visionguard.heldout_contract import context, verify_freeze
from visionguard.heldout_paths import checked
from visionguard.visa_acquire import sha256_file
from visionguard.visa_b2_storage import json_once
from visionguard.visa_evaluator import require
from visionguard.visa_evaluator_storage import canonical_bytes

_ISSUED = {}


@dataclass(frozen=True)
class Permission:
    authorization_id: str
    claim_sha256: str
    run_root: Path
    source_root: Path
    registry_sha256: str
    approval_sha256: str
    expected: dict
    expires: datetime


def _protected_registry() -> tuple[dict, str]:
    require(
        sys.platform == "win32", "Real execution requires reviewed Windows environment"
    )
    # Fixed machine-wide location; no caller/environment-variable trust override.
    script = r"""
$target='C:\ProgramData\VisionGuardAI\authorization\trust.json'
$ErrorActionPreference='Stop'
$id=[Security.Principal.WindowsIdentity]::GetCurrent()
$principal=New-Object Security.Principal.WindowsPrincipal($id)
if($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){
    throw 'Run evaluator without administrative write authority'
}
$protected=@('S-1-5-18','S-1-5-32-544')
foreach($ancestor in @('C:\','C:\ProgramData')){
  $item=Get-Item -LiteralPath $ancestor
  if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){
    throw 'Linked trust ancestor'
  }
  foreach($rule in (Get-Acl -LiteralPath $ancestor).Access){
    $sid=$rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier])
    $danger=([int]$rule.FileSystemRights -band 0xD0040) -ne 0
    if($rule.AccessControlType -eq 'Allow' -and $danger `
        -and $sid.Value -notin $protected){throw 'Replaceable trust ancestor'}
  }
}
$paths=@('C:\ProgramData\VisionGuardAI','C:\ProgramData\VisionGuardAI\authorization',$target)
foreach($p in $paths){
  $item=Get-Item -LiteralPath $p -ErrorAction Stop
  if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){
    throw 'Linked trust path'
  }
  $acl=Get-Acl -LiteralPath $p -ErrorAction Stop
  $account=[Security.Principal.NTAccount]$acl.Owner
  $owner=$account.Translate([Security.Principal.SecurityIdentifier]).Value
  if($owner -notin $protected){throw 'Untrusted registry owner'}
  foreach($rule in $acl.Access){
    $sid=$rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
    $write=([int]$rule.FileSystemRights -band 0xD0156) -ne 0
    if($rule.AccessControlType -eq 'Allow' -and $write -and $sid -notin $protected){
      throw 'Registry writable by ordinary principal'
    }
  }
}
[IO.File]::ReadAllText($target)
"""
    raw = subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        text=True,
        stderr=subprocess.PIPE,
    )
    document = json.loads(raw)
    return document, hashlib.sha256(canonical_bytes(document)).hexdigest()


def _signature_valid(public_key: dict, payload: bytes, signature: str) -> bool:
    """Use the platform cryptographic provider, not an evaluator signing key."""
    try:
        modulus = base64.b64decode(public_key["modulus"], validate=True)
        exponent = base64.b64decode(public_key["exponent"], validate=True)
        signed = base64.b64decode(signature, validate=True)
        require(
            int.from_bytes(modulus, "big").bit_length() >= 3072
            and int.from_bytes(exponent, "big") == 65537
            and len(signed) == len(modulus),
            "Invalid RSA parameters",
        )
        request = {
            "modulus": public_key["modulus"],
            "exponent": public_key["exponent"],
            "signature": signature,
            "payload": base64.b64encode(payload).decode(),
        }
        # All dynamic data travels via stdin JSON, not interpolation into shell code.
        script = r"""
$ErrorActionPreference='Stop'
$j=[Console]::In.ReadToEnd()|ConvertFrom-Json
$p=New-Object Security.Cryptography.RSAParameters
$p.Modulus=[Convert]::FromBase64String($j.modulus)
$p.Exponent=[Convert]::FromBase64String($j.exponent)
$r=[Security.Cryptography.RSA]::Create()
$r.ImportParameters($p)
$ok=$r.VerifyData([Convert]::FromBase64String($j.payload),
    [Convert]::FromBase64String($j.signature),
    [Security.Cryptography.HashAlgorithmName]::SHA256,
    [Security.Cryptography.RSASignaturePadding]::Pkcs1)
$r.Dispose()
if($ok){'true'}else{'false'}
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip() == "true"
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        return False


def _review_state(repository: Path, registry: dict) -> None:
    def git(*args):
        return subprocess.check_output(
            ["git", *args],
            cwd=repository,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).strip()

    require(not git("status", "--porcelain"), "Execution requires clean checkout")
    expected = registry["reviewed_merge"]
    require(
        git("rev-parse", "HEAD") == expected == git("rev-parse", "origin/main"),
        "Review merge/checkout mismatch",
    )
    parents = git("show", "-s", "--format=%P", expected).split()
    require(
        len(parents) == 2 and parents[1] == registry["reviewed_head"],
        "Regular reviewed two-parent merge required",
    )
    require(
        registry["reviewed_head"]
        != context(repository)["activation"]["base_regular_merge"],
        "Activation PR itself not merged",
    )
    git(
        "merge-base",
        "--is-ancestor",
        context(repository)["activation"]["base_regular_merge"],
        expected,
    )


def expected_approval(repository: Path, registry: dict) -> dict:
    ctx = context(repository)
    frozen = verify_freeze(repository)
    require(
        registry["activation_fingerprint"] == frozen["fingerprint"],
        "Registry activation mismatch",
    )
    output = repository.resolve() / ctx["activation"]["output_relative"]
    require(
        repository.resolve()
        == Path(ctx["activation"]["canonical_repository"]).absolute(),
        "Canonical reviewed D repository required",
    )
    require(output.drive.upper() == "D:", "Exact D output required")
    return {
        "schema_version": 1,
        "allow_real_test_access": True,
        "reviewed_merge": registry["reviewed_merge"],
        "reviewed_head": registry["reviewed_head"],
        "activation_fingerprint": frozen["fingerprint"],
        "b1_fingerprint": ctx["activation"]["b1_fingerprint"],
        "b2_readiness_fingerprint": ctx["activation"]["b2_readiness_fingerprint"],
        "development_freeze_sha256": ctx["activation"]["development_freeze_sha256"],
        "dataset_audit_sha256": ctx["protocol"]["dataset_audit_sha256"],
        "scope": ctx["activation"]["scope"],
        "output_destination": str(output),
        "source_root": registry["source_root"],
        "environment": frozen["document"]["environment"],
        "models": frozen["document"]["models"],
    }


def authorize(
    repository: Path, approval: dict | None, *, resume_claim: str | None = None
) -> Permission:
    """No test-root argument or path operation. Verify before immutable claim."""
    require(type(approval) is dict, "Final-test lock CLOSED: human approval absent")
    registry, registry_sha = _protected_registry()
    _review_state(repository, registry)
    expected = expected_approval(repository, registry)
    require(set(approval) == {"payload", "signature"}, "Malformed signed approval")
    payload = approval["payload"]
    require(
        type(payload) is dict
        and set(payload)
        == set(expected) | {"authorization_id", "not_before", "expires"},
        "Incomplete approval",
    )
    require(
        canonical_bytes({k: v for k, v in payload.items() if k in expected})
        == canonical_bytes(expected),
        "Approval scientific/scope/input/output identity mismatch",
    )
    identifier = payload["authorization_id"]
    require(
        type(identifier) is str
        and re.fullmatch("[0-9a-f]{32}", identifier) is not None,
        "Invalid authorization ID",
    )
    now = datetime.now(UTC)
    start = datetime.fromisoformat(payload["not_before"])
    expires = datetime.fromisoformat(payload["expires"])
    require(
        start.tzinfo is not None
        and expires.tzinfo is not None
        and start <= now < expires,
        "Expired/not-yet-valid approval",
    )
    require(
        _signature_valid(
            registry["public_key"], canonical_bytes(payload), approval["signature"]
        ),
        "Final-test lock CLOSED: invalid human signature",
    )
    from visionguard.visa_engineering import environment_identity

    require(
        environment_identity() == expected["environment"],
        "Actual frozen environment mismatch",
    )
    output = Path(expected["output_destination"])
    checked(output, missing=True)
    require(
        shutil.disk_usage(repository).free
        >= context(repository)["activation"]["minimum_free_bytes"],
        "Insufficient actual D capacity",
    )
    # Claim belongs to this one authorization ID/run. First write precedes admission.
    claim_root = output / "authorization-claims"
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_root / f"{identifier}.json"
    checked(claim_path, missing=True)
    approval_sha = hashlib.sha256(canonical_bytes(approval)).hexdigest()
    document = {
        "approval": approval,
        "registry_sha256": registry_sha,
        "approval_sha256": approval_sha,
        "run_root": str(output / "runs" / identifier),
    }
    if resume_claim is None:
        receipt = json_once(
            claim_path, document
        )  # O_EXCL: reused approval cannot new-run.
    else:
        require(
            sha256_file(claim_path) == resume_claim
            and json.loads(claim_path.read_text()) == document,
            "Same-run claim lineage differs",
        )
        require(
            not (output / "runs" / identifier / "complete.json").exists(),
            "Completed approval cannot be reused",
        )
        receipt = {"sha256": resume_claim}
    permit = Permission(
        identifier,
        receipt["sha256"],
        output / "runs" / identifier,
        Path(registry["source_root"]),
        registry_sha,
        approval_sha,
        expected,
        expires,
    )
    _ISSUED[identifier] = permit
    return permit


def check_permission(permission: Permission) -> None:
    require(
        type(permission) is Permission
        and _ISSUED.get(permission.authorization_id) is permission,
        "Unissued permission: test lock CLOSED",
    )
    require(datetime.now(UTC) < permission.expires, "Execution permission expired")
