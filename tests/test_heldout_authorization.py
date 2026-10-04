"""No real trust/approval is generated. Positive paths use mocked artificial trust."""

import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from visionguard import heldout_authorization as gate
from visionguard.heldout_runner import evaluate_real


class HostileRoot:
    def __getattribute__(self, name):
        raise AssertionError("Unauthorized test-root touch: " + name)


def test_missing_approval_before_any_root_operation(tmp_path):
    with pytest.raises(ValueError, match="CLOSED"):
        evaluate_real(tmp_path, test_root=HostileRoot())


def test_direct_real_matrix_requires_issued_permission():
    from visionguard.heldout_runner import run_matrix

    with pytest.raises(ValueError, match="CLOSED"):
        run_matrix(None, HostileRoot(), {}, {}, lambda *a: None, real=True)


@pytest.fixture
def artificial_trust(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    expected = {
        "schema_version": 1,
        "allow_real_test_access": True,
        "reviewed_merge": "a" * 40,
        "reviewed_head": "b" * 40,
        "activation_fingerprint": "c" * 64,
        "b1_fingerprint": "d" * 64,
        "b2_readiness_fingerprint": "e" * 64,
        "development_freeze_sha256": "f" * 64,
        "dataset_audit_sha256": "0" * 64,
        "scope": {"pairs": 36},
        "output_destination": str(tmp_path / "output"),
        "source_root": str(tmp_path / "NONEXISTENT-ARTIFICIAL-SOURCE"),
        "environment": {"artificial": True},
        "models": {"artificial-only": True},
    }
    approval = {
        "payload": expected
        | {
            "authorization_id": "1" * 32,
            "not_before": (now - timedelta(minutes=1)).isoformat(),
            "expires": (now + timedelta(minutes=10)).isoformat(),
        },
        "signature": "TEST MOCK NOT REAL HUMAN APPROVAL",
    }
    monkeypatch.setattr(
        gate,
        "_protected_registry",
        lambda: ({"public_key": {}, "source_root": expected["source_root"]}, "2" * 64),
    )
    monkeypatch.setattr(gate, "_review_state", lambda *a: None)
    monkeypatch.setattr(gate, "expected_approval", lambda *a: expected)
    monkeypatch.setattr(gate, "_signature_valid", lambda *a: True)
    monkeypatch.setattr(
        gate, "context", lambda *a: {"activation": {"minimum_free_bytes": 0}}
    )
    import visionguard.visa_engineering

    monkeypatch.setattr(
        visionguard.visa_engineering,
        "environment_identity",
        lambda: expected["environment"],
    )
    return tmp_path, expected, approval


@pytest.mark.parametrize(
    "field",
    [
        "allow_real_test_access",
        "reviewed_merge",
        "reviewed_head",
        "activation_fingerprint",
        "b1_fingerprint",
        "b2_readiness_fingerprint",
        "development_freeze_sha256",
        "dataset_audit_sha256",
        "scope",
        "output_destination",
        "source_root",
        "environment",
        "models",
    ],
)
def test_wrong_bound_fields_never_touch_test_root(artificial_trust, field):
    root, _, approval = artificial_trust
    changed = copy.deepcopy(approval)
    changed["payload"][field] = "self-asserted"
    with pytest.raises(ValueError):
        evaluate_real(root, changed, test_root=HostileRoot())
    assert not (root / "output").exists()


@pytest.mark.parametrize(
    "case",
    [
        "expired",
        "future",
        "badid",
        "signature",
        "extra",
        "missing",
        "environment",
        "review",
    ],
)
def test_other_denials(artificial_trust, monkeypatch, case):
    root, _, approval = artificial_trust
    changed = copy.deepcopy(approval)
    if case == "expired":
        changed["payload"]["expires"] = "2000-01-01T00:00:00+00:00"
    elif case == "future":
        changed["payload"]["not_before"] = "2100-01-01T00:00:00+00:00"
    elif case == "badid":
        changed["payload"]["authorization_id"] = "../../escape"
    elif case == "extra":
        changed["payload"]["extra"] = 1
    elif case == "missing":
        del changed["payload"]["models"]
    elif case == "signature":
        monkeypatch.setattr(gate, "_signature_valid", lambda *a: False)
    elif case == "review":

        def deny(*a):
            raise ValueError("Unmerged")

        monkeypatch.setattr(gate, "_review_state", deny)
    else:
        import visionguard.visa_engineering

        monkeypatch.setattr(
            visionguard.visa_engineering,
            "environment_identity",
            lambda: {"wrong": True},
        )
    with pytest.raises(ValueError):
        evaluate_real(root, changed, test_root=HostileRoot())
    assert not (root / "output").exists()


def test_artificial_single_use_same_run_only(artificial_trust):
    root, _, approval = artificial_trust
    permission = gate.authorize(root, approval)
    gate.check_permission(permission)
    with pytest.raises(FileExistsError):
        gate.authorize(root, approval)
    restored = gate.authorize(root, approval, resume_claim=permission.claim_sha256)
    assert (
        restored.run_root == permission.run_root
        and restored.claim_sha256 == permission.claim_sha256
    )
    with pytest.raises(ValueError):
        gate.check_permission(permission)
    with pytest.raises(ValueError):
        gate.authorize(root, approval, resume_claim="0" * 64)
    restored.run_root.mkdir(parents=True)
    (restored.run_root / "complete.json").write_text("{}")
    with pytest.raises(ValueError):
        gate.authorize(root, approval, resume_claim=permission.claim_sha256)


def test_unsigned_bad_parameters():
    assert not gate._signature_valid({}, b"artificial", "fake")


def test_protected_registry_not_caller_override():
    import inspect

    source = inspect.getsource(gate._protected_registry)
    assert "C:\\ProgramData\\VisionGuardAI\\authorization\\trust.json" in source
    assert (
        "Get-Acl" in source
        and "Run evaluator without administrative write authority" in source
    )
    assert len(inspect.signature(gate._protected_registry).parameters) == 0


@pytest.mark.skipif(
    __import__("os").name != "nt", reason="Windows platform RSA provider"
)
def test_platform_signature_with_ephemeral_synthetic_key():
    import subprocess

    # EPHEMERAL TEST KEY only, no human/private key file or trust registration.
    script = r"""
$ErrorActionPreference='Stop'
$r=New-Object Security.Cryptography.RSACryptoServiceProvider -ArgumentList 3072
$p=$r.ExportParameters($false)
$b=[Text.Encoding]::UTF8.GetBytes('ARTIFICIAL UNIT TEST')
$s=$r.SignData($b,[Security.Cryptography.HashAlgorithmName]::SHA256,
    [Security.Cryptography.RSASignaturePadding]::Pkcs1)
@{modulus=[Convert]::ToBase64String($p.Modulus)
exponent=[Convert]::ToBase64String($p.Exponent)
signature=[Convert]::ToBase64String($s)}|ConvertTo-Json -Compress
$r.Dispose()
"""
    d = json.loads(
        subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
        )
    )
    assert gate._signature_valid(d, b"ARTIFICIAL UNIT TEST", d["signature"])
    assert not gate._signature_valid(d, b"ALTERED ARTIFICIAL UNIT TEST", d["signature"])
