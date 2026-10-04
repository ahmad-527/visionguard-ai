# Stage B ACL compatibility review

Status: **engineering implementation frozen; DRAFT independent review only;
evaluation lock CLOSED**. CI verification is recorded in the PR and local handoff.
Issue: #27. Base regular merge: `91b5b45a82ccef042ad13535d2456ea7d6013885`.
Draft PR: [#28](https://github.com/ahmad-527/visionguard-ai/pull/28).
Branch: `codex/stage-b-acl-compatibility`.

## Amendment design established before source edits

Keep the predecessor activation freeze and original scientific protocol byte-identical.
Archive the exact predecessor authorization and contract source bytes under the
successor evidence directory, binding them to their existing frozen source hashes.
The successor verifier will validate the complete historical freeze, all unamended
current source/evidence, B1/B2, all 72 model specifications and frozen environment,
then independently bind the amended gate, contract dispatcher, new ACL predicate,
new tests/workflow and versioned amendment protocol into a new immutable freeze.
The runner's existing freeze-verification entry point must explicitly select this
successor; the predecessor fingerprint must never be represented as current code.
A future trust registry must name the new reviewed regular merge/head and successor
fingerprint. No compatibility fallback may authorize a predecessor approval.

The predicate will skip only ACEs marked InheritOnly for the object currently
examined, not all IsInherited ACEs. It will independently check the five actual
path components, including effective inherited outcomes on descendants. Dangerous
Allow rights remain fail-closed even in the presence of Deny entries. Missing,
ambiguous, linked/offline/cloud metadata and untrusted dedicated ownership are
rejected before any public registry read. Hidden metadata is queried with Force,
not through a permissions bypass. No Windows ACL or power change is planned.

## Preserved tooling-boundary incident

The stopped attempt recorded this exact command:

```powershell
rg --files D:\VisionGuardAI -g gh.exe -g '!heldout-visa-source/**' -g '!repository/outputs/**' -g '!repository/data/**'
```

The exact command-execution timestamp was not recorded in the original stop report.
The stop-record timestamp observed by the UTC clock was **2026-10-04 14:29:53 UTC**;
it must not be presented as a precise command-start time. The search returned no
matching tooling paths. Whether its exclusion prevented staged-source directory
enumeration is **UNKNOWN**. No zero-dataset-metadata-access claim is made for that
attempt; no retrospective dataset inspection will establish what happened. No
asset contents, hashing, decoding, inference or performance results were obtained.
The local stop report remains unchanged (SHA-256
`4036d602aaca69fc7e77a41c61b6e3a902286aae37a1a79208f6f3105506dbcd`).
The contributor explicitly acknowledged this uncertainty and authorized the engineering-only
continuation. Subsequent searches are confined to exact authorized repository,
Stage B receipt or individually identified tooling paths; no dataset operations
are authorized. Personal paths and raw operational snapshots stay out of Git.

## Operational history

The original Stage B report predates human-operated AC safeguards. The manual
original Turbo backup SHA-256 was independently verified as
`afe709c443f1ac84ae11e85f4ad6423e1d4670005e411095f0386a6818647ae9`.
The human reports successful AC-only application and unchanged DC settings. No
application/rollback script will be executed by the agent. Historical failed export,
hidden-directory diagnostic failure and previous scientific recovery history remain
preserved. Human access history UNKNOWN; independent reservation NOT ESTABLISHED.

## Root cause and retained security policy

The predecessor screened the rights bitmask without examining propagation flags.
The actual root ACL contains an Authenticated Users Allow ACE with signed rights
`-536805376`, inheritance flags `3` (containers and objects), propagation flags `2`
(`InheritOnly`), and `IsInherited=False`. The old ancestor mask `0xD0040` matches
`0x10000` (DELETE), incorrectly treating that ACE as an effective root permission.
An effective CreateDirectories-only grant (`4`) is not a root replacement grant.

`InheritOnly` means an ACE is not effective on its current object; `IsInherited`
only describes its origin. Inheritance can produce effective child grants. The
successor therefore checks every **actual** ancestor/dedicated directory/file,
not a hypothetical propagated ACL and not a blanket exception for inherited
permissions. This distinction follows [Microsoft's ACE inheritance semantics](https://learn.microsoft.com/en-us/windows/win32/secauthz/ace-inheritance).

| Object | Conservative rejection rule for ordinary Allow ACEs |
| --- | --- |
| C: root and ProgramData | effective DELETE, WRITE_DAC, WRITE_OWNER or DELETE_CHILD (`0xD0040`), or GENERIC_ALL/GENERIC_WRITE |
| Dedicated VisionGuardAI directory, authorization directory and trust.json | the above plus WriteData, AppendData, WriteExtendedAttributes and WriteAttributes (`0xD0156`), or generic mutation |

Only SYSTEM and Administrators are exempt from dangerous Allow screening.
TrustedInstaller may own the two Windows ancestors, but not the dedicated
registry hierarchy. Dangerous ordinary Allow is rejected even if a Deny might
cancel it; this is deliberately **stricter than Windows token AccessCheck**, not
a complete effective-token ACL evaluator. `NoPropagateInherit` alone is not
`InheritOnly`. Propagated effective ordinary permissions on any actual child fail.

The collector uses five fixed literal trust paths, the exact Windows PowerShell
executable and its own native security module. `Get-Item -LiteralPath ... -Force`
handles hidden ProgramData without changing permissions. Reparse/offline/recall
attributes, aliases, missing paths, untrusted owners, null/empty DACLs, unsupported
raw ACEs/flags, unknown rights, malformed metadata and multiple-link/non-regular
registries fail closed. There is no caller-supplied registry or path override.
Public registry content is read only after complete metadata screening. Its
schema has exactly reviewed_merge, reviewed_head, activation_fingerprint,
source_root and public_key; public_key has only modulus and exponent. No private
material is accepted. Source-root strings are not inspected by this reader.

Local administrators can alter the machine and remain outside the threat model.
The predicate does not protect against an administrator changing ACLs between
metadata collection and reading, or an already compromised interpreter/OS.
No ordinary-principal dangerous grant is excused to work around compatibility.

## Native read-only evidence and preserved negative findings

At **2026-10-04 15:14:45 UTC**, the non-elevated collector validated only the two
actual Windows ancestors. Hidden ProgramData metadata was successfully read.
The old predicate would reject one root InheritOnly ACE; the new ancestor
predicate passes. Native ancestor receipt SHA-256:
`4d2cbb2a54066986b151ceb9615105825998ec0255a75cd685766c2f5cc611dc`.
Raw ACL metadata remains in ignored local engineering outputs, not public Git.

A separate full-chain **metadata-only** probe stopped at missing
`C:\ProgramData\VisionGuardAI` with native Get-Item PathNotFound/return code 1.
It did not read a registry. Thus **production registry NOT VALIDATED**; no trust
hierarchy was installed. Ancestor-only diagnostic mode cannot issue permission.

An intermediate Python child-process check failed because its inherited module
search path selected conflicting PowerShell security type definitions
(`AuditToString`, `AccessToString`, `Sddl`, `Access`, `Group`, `Owner`, `Path`
already present). Selecting Windows PowerShell's own module corrected the local
collector without any module installation, machine policy or ACL change.
Intermediate shell quoting/path-glob errors and two Ruff long-line findings were
ordinary engineering failures, corrected before freezing; none involved dataset
access. A sandbox-owner Git archive attempt failed ownership checks; archive
extraction succeeded in the existing non-admin owner context, without changing
Git safe.directory or repository permissions.

## Immutable predecessor and explicit successor

| Identity | Value |
| --- | --- |
| Predecessor reviewed PR head | `58062446d7a6d7bded8515e2d884de04981c843f` |
| Predecessor activation fingerprint | `960811034e904c6f86a8e127ef0b149a67fb2ec5715e96333e813d461d4d431f` |
| Successor protocol | `visionguard-visa-authorization-amendment-v2` |
| **Successor activation fingerprint** | `8b30e6b923c8bb49f457b50860a0adcc62d6bf1031405e28e60ab1c28ddd00b0` |
| Successor freeze-file SHA-256 | `3b0ee1ae01afd6f475d99406361d4bfa09c679f9a0fc404a5da82e6a3071ad1f` |
| Unchanged B1 fingerprint | `0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49` |
| Unchanged B2 readiness fingerprint | `1b44805040f211fdf75bd54561bae1b61f0fe2912d25ff734c355ff1202ba6cc` |
| Unchanged 72-cell development freeze SHA-256 | `d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061` |

Original v1 freeze bytes remain unchanged. Exact predecessor authorization and
contract archives match respectively
`351c457a9528a87958e614908d002799caba4810a711d0f9b9df70db6a3e9c69`
and `57d0afc38a7f00631b3cf99a4a275a2c17720092a5d13de46cc7ba2acc4b3c5b`.
Every other original activation source and evidence hash verifies against its
unchanged current file. The successor binds both current amended files, those
archives, predecessor freeze, new ACL/verifier modules, tests, scripts, workflow
and protocol. Missing/drifted successor evidence never falls back to v1.

All 72 model specifications, seeds, categories, image/pixel thresholds, original
scientific contract and frozen environment compare exactly with the predecessor.
B1/B2 reproduction passed. No model, normalization, calibration, metric, triage,
dataset audit, membership or original scientific implementation was changed.
No actual model fitting/inference or binary model-artifact rewrite was performed.
Signing verification, expected-approval construction, authorize/claim and issued
permission functions retain exact AST identity with their archived versions.

Future approval must pin the successor fingerprint and a **new reviewed regular
merge and PR head**. The predecessor merge/head cannot authorize the amendment;
the predecessor merge must remain an ancestor. The fixed trust registry must be
provisioned by a separately authorized human administrator **after review and
regular merge**, not by this assignment. New private signing material stays under
human control. No real approval, authorization claim or permission was generated.

Reproduction (verification only):

```powershell
python scripts/security_amendment_freeze.py
python scripts/phase4d_b1_freeze.py
python scripts/phase4d_b2_freeze.py
python -m visionguard.heldout_runner --plan
```

The existing activation-verification entry point explicitly delegates to schema 2;
it does not report predecessor code as current. The separate successor creation
command writes the new freeze exclusively; it does not overwrite the v1 freeze.
Do not recreate either freeze during human review.

## Synthetic regression and artificial acceptance

The focused group passed **184 tests**, with one explicitly deselected existing
Windows test that would generate an ephemeral RSA key. No key was generated.
Coverage includes effective/inherited/InheritOnly/NoPropagate distinctions,
multi-level propagation, all dangerous rights, trusted owners, hidden metadata,
missing/reordered/aliased/linked/cloud paths, null/unsupported/malformed ACLs,
hard-linked registries, public-only schema, denial before reads, exact historical
archives, fingerprint mutation, scientific drift, no fallback and new-merge rules.
Existing hostile-root denials prove missing/wrong approval cannot inspect inputs.

The artificial 36-pair/72-stage acceptance used **48 manufactured images** and
known labels/maps only. All B1 numerical outputs and category/seed aggregates were
bitwise equal; per pair PASS=1, REVIEW=2, REJECT=1. Strict independent thresholds
and three-way REVIEW semantics were unchanged. The injected interruption left
one partial call and a preserved failure record. Resume made 284 calls; full
replay validated/skipped all 72 stages and made **zero new inference calls**.
Measured acceptance time was **43.0044 seconds**, retaining **1,836,339 bytes**;
these are synthetic engineering observations, not real-model speed/storage claims.
Acceptance receipt SHA-256:
`6279b0bdb5e2dedb0e9e3b8c0bf29ea8ca58eac364522b25533c5227a5e8dce2`.
Its failure receipt SHA-256:
`8e8da18c7e73fb5636959f91072c5322528dacf7952f49c0df11acc916c94ae2`.

Full applicable pytest passed **867 tests**, with **2 skips** for unavailable
Windows symlink privileges and **1 deliberate deselection** of key generation
in **334.59 seconds**. Ruff format (164 Python files), Ruff lint, pip check,
provenance, closed plan/CLI and staged hygiene checks passed. The staged scan
found no private-key/token/personal-path matches or large unintended artifacts.
The Windows key-generating test
is intentionally excluded locally; Ubuntu CI naturally skips that Windows-only
test. All three supported Python versions run the unchanged general and
controlled-activation workflows plus the new synthetic amendment workflow.

## Review boundary and remaining limitations

This fixes observed ancestor compatibility, not a complete production-readiness
claim. The actual dedicated hierarchy, public trust registry and future signed
approval still need separate human provisioning and independent validation.
The PR must remain **DRAFT and unmerged** for independent security/scientific
review. Eventual merge must be a **regular merge commit**, never squash/rebase.

The two failed and one interrupted Phase 4D-A attempts, approved migrations and
unknown operational causes remain preserved in the unchanged development evidence.
The original stopped tooling incident and metadata uncertainty remain visible.
No staged-source enumeration/stat/hash/read or real evaluation occurred in this
authorized continuation. No power setting, Windows ACL, key or trust installation
was changed. Human history **UNKNOWN**, reservation **NOT ESTABLISHED**; the default
future reporting label remains **held-out VisA evaluation with historical access
independence unverified**. Real admission/inference/evaluation is not authorized.

**Do not merge. A future regular merge and separately authorized human provisioning
and signing are required. STOP for independent review after engineering completion.**
