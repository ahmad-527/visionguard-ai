"""Manufactured ACL metadata only: no provisioning, key, claim or dataset access."""

import copy
import json
import stat
import subprocess
from types import SimpleNamespace

import pytest

from visionguard import heldout_authorization as gate
from visionguard import windows_trust_acl as acl


def rule(rights=0x1200A9, *, sid="S-1-5-32-545", inherited=False, propagation=0):
    return {
        "sid": sid,
        "rights": rights,
        "type": "Allow",
        "inheritance": 3 if propagation else 0,
        "propagation": propagation,
        "is_inherited": inherited,
    }


@pytest.fixture
def chain():
    return {
        "elevated": False,
        "paths": [
            {
                "path": p,
                "full_name": p,
                "owner": "S-1-5-18",
                "attributes": 18 if i < 4 else 32,
                "directory": i < 4,
                "dacl_present": True,
                "rules": [rule(), rule(0x1F01FF, sid="S-1-5-32-544")],
            }
            for i, p in enumerate(acl.TRUST_PATHS)
        ],
    }


def test_hidden_read_only_trusted_owners_and_inherited_reads(chain):
    chain["paths"][0]["owner"] = acl.TRUSTED_INSTALLER
    for row in chain["paths"]:
        row["rules"][0]["is_inherited"] = True
    acl.validate_chain(chain)
    acl.validate_chain(
        {"elevated": False, "paths": chain["paths"][:2]}, ancestors_only=True
    )


def test_native_root_inherit_only_old_false_rejection(chain):
    observed = rule(-536805376, sid="S-1-5-11", propagation=2)
    assert observed["rights"] & 0xD0040 == 0x10000  # independent old-mask answer
    chain["paths"][0]["rules"].append(observed)
    chain["paths"][0]["rules"].append(rule(4, sid="S-1-5-11"))
    chain["paths"][1]["rules"].append(rule(278))  # creation, not replacement
    acl.validate_chain(chain)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize(
    "rights", [0x10000, 0x40000, 0x80000, 0x40, 0x40000000, 0x10000000]
)
@pytest.mark.parametrize("inherited", [False, True])
def test_effective_danger_never_skipped(chain, index, rights, inherited):
    chain["paths"][index]["rules"].append(rule(rights, inherited=inherited))
    with pytest.raises(ValueError, match="Effective dangerous"):
        acl.validate_chain(chain)


@pytest.mark.parametrize("index", [2, 3, 4])
@pytest.mark.parametrize("rights", [2, 4, 0x10, 0x100])
def test_dedicated_write_rights_rejected(chain, index, rights):
    chain["paths"][index]["rules"].append(rule(rights, inherited=True))
    with pytest.raises(ValueError):
        acl.validate_chain(chain)


@pytest.mark.parametrize("propagation", [2, 3])
def test_inherit_only_does_not_hide_propagated_effective_child(chain, propagation):
    for index in [0, 1, 2]:
        chain["paths"][index]["rules"].append(rule(0x10000, propagation=propagation))
    acl.validate_chain(chain)
    chain["paths"][3]["rules"].append(rule(0x10000, inherited=True))
    with pytest.raises(ValueError, match="Effective dangerous"):
        acl.validate_chain(chain)


def test_deny_does_not_cancel_unsafe_allow(chain):
    deny = rule(0x1F01FF)
    deny["type"] = "Deny"
    chain["paths"][4]["rules"].extend([deny, rule(2)])
    with pytest.raises(ValueError):
        acl.validate_chain(chain)


@pytest.mark.parametrize("attributes", [0x400, 0x1000, 0x40000, 0x400000, -1, True])
@pytest.mark.parametrize("index", range(5))
def test_unsafe_metadata_before_read(chain, attributes, index):
    chain["paths"][index]["attributes"] = attributes
    with pytest.raises(ValueError):
        acl.validate_chain(chain)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sid", "unresolved-name"),
        ("rights", True),
        ("rights", 0x200),
        ("rights", 2**32),
        ("inheritance", 4),
        ("propagation", 4),
        ("is_inherited", 1),
        ("type", "CallbackAllow"),
    ],
)
def test_malformed_ace(chain, field, value):
    chain["paths"][2]["rules"][0][field] = value
    with pytest.raises(ValueError):
        acl.validate_chain(chain)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "reorder",
        "alias",
        "owner",
        "null",
        "empty",
        "file-kind",
        "flags",
        "file-inheritance",
        "elevated",
        "installer-child",
    ],
)
def test_chain_fail_closed(chain, case):
    row = chain["paths"][4]
    if case == "missing":
        chain["paths"].pop()
    elif case == "reorder":
        chain["paths"].reverse()
    elif case == "alias":
        row["full_name"] = "C:\\alias\\trust.json"
    elif case == "owner":
        row["owner"] = "S-1-5-32-545"
    elif case == "null":
        row["dacl_present"] = False
    elif case == "empty":
        row["rules"] = []
    elif case == "file-kind":
        row["directory"] = True
    elif case == "flags":
        row["rules"][0]["propagation"] = 2
    elif case == "file-inheritance":
        row["rules"][0]["inheritance"] = 1
    elif case == "elevated":
        chain["elevated"] = True
    else:
        chain["paths"][2]["owner"] = acl.TRUSTED_INSTALLER
    with pytest.raises(ValueError):
        acl.validate_chain(chain)


def test_collector_force_fixed_module_and_no_path_override(monkeypatch, chain):
    commands = []

    def collect(command, **kwargs):
        assert command[0] == acl.POWERSHELL
        commands.append(command[-1])
        assert kwargs["stderr"] == subprocess.PIPE
        return json.dumps(chain)

    monkeypatch.setattr(acl.sys, "platform", "win32")
    monkeypatch.setattr(acl.subprocess, "check_output", collect)
    assert acl.collect_chain() == chain
    acl.collect_chain(ancestors_only=True)
    assert "$paths=$paths[0..1]" not in commands[0]
    assert "$paths=$paths[0..1]" in commands[1]
    for command in commands:
        assert "Get-Item -LiteralPath $p -Force -ErrorAction Stop" in command
        assert "$PSHOME\\Modules\\Microsoft.PowerShell.Security" in command
        assert "Unsupported trust ACE" in command
        assert "Unrepresented trust ACE" in command
        assert "Get-ChildItem" not in command and "ReadAllText" not in command
        assert "heldout-visa-source" not in command


def test_collection_error_not_swallowed(monkeypatch):
    monkeypatch.setattr(acl.sys, "platform", "win32")

    def deny(*a, **kw):
        raise subprocess.CalledProcessError(
            1, "manufactured", stderr="Missing trust path"
        )

    monkeypatch.setattr(acl.subprocess, "check_output", deny)
    with pytest.raises(subprocess.CalledProcessError):
        acl.collect_chain()


def test_registry_unsafe_chain_denied_before_file_operation(monkeypatch, chain):
    monkeypatch.setattr(gate.sys, "platform", "win32")
    chain["paths"][4]["rules"].append(rule(2, inherited=True))
    monkeypatch.setattr(gate, "collect_chain", lambda: chain)

    def forbidden(*a, **kw):
        raise AssertionError("Registry must not be opened")

    monkeypatch.setattr(gate, "Path", forbidden)
    with pytest.raises(ValueError):
        gate._protected_registry()


@pytest.mark.parametrize(
    "case",
    ["valid", "hardlink", "notfile", "extra", "badidentity", "privatekey", "badjson"],
)
def test_registry_read_uses_only_validated_fixed_public_target(
    monkeypatch, chain, case
):
    document = {
        "reviewed_merge": "a" * 40,
        "reviewed_head": "b" * 40,
        "activation_fingerprint": "c" * 64,
        "source_root": "MANUFACTURED-NEVER-TOUCHED",
        "public_key": {"modulus": "public-placeholder", "exponent": "AQAB"},
    }
    calls = []
    if case == "extra":
        document["unexpected"] = True
    elif case == "badidentity":
        document["reviewed_merge"] = "arbitrary"
    elif case == "privatekey":
        document["public_key"]["private"] = "FORBIDDEN-PLACEHOLDER-NOT-A-KEY"

    class ManufacturedPath:
        def stat(self):
            return SimpleNamespace(
                st_mode=stat.S_IFDIR if case == "notfile" else stat.S_IFREG,
                st_nlink=2 if case == "hardlink" else 1,
            )

        def read_bytes(self):
            calls.append("public-read")
            return b"not-json" if case == "badjson" else json.dumps(document).encode()

    def path(value):
        assert value == acl.TRUST_PATHS[4]
        return ManufacturedPath()

    monkeypatch.setattr(gate.sys, "platform", "win32")
    monkeypatch.setattr(gate, "collect_chain", lambda: copy.deepcopy(chain))
    monkeypatch.setattr(gate, "Path", path)
    monkeypatch.setattr(gate, "checked", lambda p: p)
    if case == "valid":
        result, digest = gate._protected_registry()
        assert result == document and len(digest) == 64
    else:
        with pytest.raises(ValueError):
            gate._protected_registry()
    if case in ("hardlink", "notfile"):
        assert calls == []


def test_no_propagate_is_not_inherit_only(chain):
    chain["paths"][2]["rules"].append(rule(2, propagation=1, inherited=True))
    with pytest.raises(ValueError, match="Effective dangerous"):
        acl.validate_chain(chain)
