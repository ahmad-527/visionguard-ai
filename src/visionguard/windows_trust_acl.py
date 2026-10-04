"""Conservative actual-object ACL screening; no installer or access-check bypass.

InheritOnly is about this object, unlike IsInherited. Every actual descendant is
screened independently. Dangerous Allow grants are never cancelled by Deny ACEs.
This is intentionally stricter than a complete Windows token AccessCheck.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from visionguard.visa_evaluator import require

TRUST_PATHS = (
    "C:\\",
    "C:\\ProgramData",
    "C:\\ProgramData\\VisionGuardAI",
    "C:\\ProgramData\\VisionGuardAI\\authorization",
    "C:\\ProgramData\\VisionGuardAI\\authorization\\trust.json",
)
TRUSTED = frozenset({"S-1-5-18", "S-1-5-32-544"})
TRUSTED_INSTALLER = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
UNSAFE_ATTRIBUTES = 0x400 | 0x1000 | 0x40000 | 0x400000
ANCESTOR_DANGER = 0xD0040
DEDICATED_DANGER = 0xD0156
GENERIC_MUTATION = 0x10000000 | 0x40000000  # GENERIC_ALL / GENERIC_WRITE
KNOWN_RIGHTS = 0xF0000000 | 0x1F01FF
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

# Metadata only. Fixed literals, no environment/caller path override. -Force
# reveals hidden ProgramData but does not override access denial or hydrate data.
COLLECT_SCRIPT = r"""
$ErrorActionPreference='Stop'
$module="$PSHOME\Modules\Microsoft.PowerShell.Security"
Import-Module -Name "$module\Microsoft.PowerShell.Security.psd1" -ErrorAction Stop
$id=[Security.Principal.WindowsIdentity]::GetCurrent()
$principal=New-Object Security.Principal.WindowsPrincipal($id)
if($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){
  throw 'Run evaluator without administrative write authority'
}
$paths=@('C:\','C:\ProgramData','C:\ProgramData\VisionGuardAI',
  'C:\ProgramData\VisionGuardAI\authorization',
  'C:\ProgramData\VisionGuardAI\authorization\trust.json')
__ANCESTORS_ONLY__
$rows=@(foreach($p in $paths){
  $item=Get-Item -LiteralPath $p -Force -ErrorAction Stop
  $attributes=[int64]$item.Attributes
  if(($attributes -band 0x441400) -ne 0){throw 'Reparse/cloud trust path'}
  $acl=Get-Acl -LiteralPath $p -ErrorAction Stop
  $owner=$acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
  $binary=$acl.GetSecurityDescriptorBinaryForm()
  $raw=New-Object Security.AccessControl.RawSecurityDescriptor `
    -ArgumentList $binary,0
  if($null -eq $raw.DiscretionaryAcl){throw 'Null trust DACL'}
  foreach($ace in $raw.DiscretionaryAcl){
    if([int]$ace.AceType -notin @(0,1) -or (([int]$ace.AceFlags -band 0xE0) -ne 0)){
      throw 'Unsupported trust ACE'
    }
  }
  $rules=@(foreach($r in $acl.Access){
    @{sid=$r.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
      rights=[int64]$r.FileSystemRights;type=$r.AccessControlType.ToString()
      inheritance=[int]$r.InheritanceFlags;propagation=[int]$r.PropagationFlags
      is_inherited=[bool]$r.IsInherited}
  })
  if($rules.Count -ne $raw.DiscretionaryAcl.Count){throw 'Unrepresented trust ACE'}
  @{path=$p;full_name=$item.FullName;owner=$owner;attributes=$attributes
    directory=[bool]$item.PSIsContainer;rules=$rules;dacl_present=$true}
})
@{elevated=$false;paths=$rows}|ConvertTo-Json -Depth 8 -Compress
"""


def collect_chain(*, ancestors_only: bool = False) -> dict:
    """Read only fixed trust metadata; ancestor-only mode grants no permission."""
    require(sys.platform == "win32", "Windows trust metadata required")
    script = COLLECT_SCRIPT.replace(
        "__ANCESTORS_ONLY__", "$paths=$paths[0..1]" if ancestors_only else ""
    )
    raw = subprocess.check_output(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        text=True,
        stderr=subprocess.PIPE,
    )
    return json.loads(raw)


def validate_chain(snapshot: dict, *, ancestors_only: bool = False) -> None:
    """Reject any effective dangerous ordinary Allow on each actual object.

    The only privilege exceptions are SYSTEM/Administrators. TrustedInstaller may
    own Windows ancestors, never the dedicated registry hierarchy. Unknown rights,
    ACE flags/types, malformed metadata and incomplete chains are fail-closed.
    """
    require(type(snapshot) is dict, "Malformed trust metadata")
    require(snapshot.get("elevated") is False, "Administrative write authority")
    paths = snapshot.get("paths")
    expected = TRUST_PATHS[:2] if ancestors_only else TRUST_PATHS
    require(type(paths) is list and len(paths) == len(expected), "Missing trust path")
    for index, (row, path) in enumerate(zip(paths, expected, strict=True)):
        require(type(row) is dict, "Malformed trust path")
        require(
            row.get("path") == path
            and type(row.get("full_name")) is str
            and row["full_name"].casefold() == path.casefold(),
            "Trust path alias/substitution",
        )
        require(row.get("directory") is (index < 4), "Wrong trust object kind")
        attributes = row.get("attributes")
        require(
            type(attributes) is int
            and 0 <= attributes <= 0xFFFFFFFF
            and not attributes & UNSAFE_ATTRIBUTES,
            "Reparse/cloud trust path",
        )
        owners = TRUSTED | {TRUSTED_INSTALLER} if index < 2 else TRUSTED
        require(row.get("owner") in owners, "Untrusted registry owner")
        require(row.get("dacl_present") is True, "Null trust DACL")
        rules = row.get("rules")
        require(type(rules) is list and bool(rules), "Missing trust DACL rules")
        for rule in rules:
            require(type(rule) is dict, "Malformed trust ACE")
            sid = rule.get("sid")
            require(
                type(sid) is str and re.fullmatch(r"S-1-\d+(?:-\d+)+", sid),
                "Unresolved principal",
            )
            rights, inheritance, propagation = (
                rule.get("rights"),
                rule.get("inheritance"),
                rule.get("propagation"),
            )
            require(
                type(rights) is int
                and -(2**31) <= rights <= 0xFFFFFFFF
                and not (rights & 0xFFFFFFFF) & ~KNOWN_RIGHTS
                and type(inheritance) is int
                and 0 <= inheritance <= 3
                and type(propagation) is int
                and 0 <= propagation <= 3
                and type(rule.get("is_inherited")) is bool
                and rule.get("type") in ("Allow", "Deny"),
                "Unsupported trust ACE",
            )
            require(
                (not propagation or bool(inheritance))
                and (index != 4 or not (inheritance or propagation)),
                "Ambiguous inheritance flags",
            )
            # InheritOnly does not apply HERE, but a propagated descendant ACE
            # may be effective. IsInherited never causes an ACE to be skipped.
            if propagation & 2:
                continue
            danger = ANCESTOR_DANGER if index < 2 else DEDICATED_DANGER
            require(
                not (
                    rule["type"] == "Allow"
                    and sid not in TRUSTED
                    and (rights & 0xFFFFFFFF) & (danger | GENERIC_MUTATION)
                ),
                "Effective dangerous trust permission",
            )
