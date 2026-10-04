# Phase 4D-B2 controlled activation — implementation and artificial acceptance

**DRAFT — HUMAN REVIEW REQUIRED. REAL VisA TEST ACCESS NOT AUTHORIZED.**

Issue [#25](https://github.com/ahmad-527/visionguard-ai/issues/25), draft PR
[#26](https://github.com/ahmad-527/visionguard-ai/pull/26).

This layer implements a future execution route; it is not a held-out evaluation.
No test root was enumerated, statted, hashed, opened, decoded or scored. No real
approval, trust registry or authorization claim was created. Positive gate tests
use explicitly mocked artificial trust and an ephemeral unit-test key only.

Human historical access remains **UNKNOWN**; independent reservation remains
**NOT ESTABLISHED**. Every future performance report must be titled:

> Held-out VisA evaluation with historical access independence unverified.

There is no independent-confirmatory, production-readiness or generalization
claim. Artificial known-answer metrics are engineering checks, not benchmark
results. All categories, seeds, negative results and operational failures must
remain visible in any eventual separately authorized report.

## Starting point and immutable scientific inputs

PR #24 was verified merged by regular two-parent merge
`3ab2b3bc9135aa1c1c63aef78fd2d88c1552f1f7`; its second parent is the reviewed
`5464f17a0792e1aa8819281599a3f6415f698712`. Work branched from that merged main
in `D:\VisionGuardAI\repository`, not from either preserved C: checkout.

| Binding | Unchanged identity |
| --- | --- |
| B1 evaluator | `0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49` |
| B2 readiness | `1b44805040f211fdf75bd54561bae1b61f0fe2912d25ff734c355ff1202ba6cc` |
| Published 72-cell development freeze | `d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061` |
| D-A implementation | `cca5e47ec8c3f68adfa6c90e17a17770b48049d0` |
| D-A execution contract | `31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3` |
| Phase 4C audit | `06e227bb5d2cd26f38010b2c304c62f14f383a81c64f5b2e4c48f1019128f58f` |
| Fit/calibration membership | `10e7a6c898fb18fbd1b93a115a5a94f3e2ebcecb7a1796d99d8c1e6f3567ce0b` |
| Official split | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |

No B1/B2 frozen source, workflow, methodology, protocol, model, normalization,
threshold or membership file is edited. New modules use `heldout_*` names so
neither previous source freeze expands accidentally. Their source hashes,
configuration, all 72 model specifications, original environment and compact
test evidence are bound by the separate activation freeze.

The fresh read-only development verification checks all 72 final models and
their checkpoint, normalization, calibration and inventory identities against
the committed freeze. It preserves 72 completed attempts, two failed cashew
attempts and the interrupted pcb1 attempt. The original WinError 5 failures,
approved migrations, interrupted-process unknown cause/exit status and forensic
receipts are not repaired, cleaned or reinterpreted in this task. C: originals
remain untouched. Compact acceptance receipts record measured verification
duration and hashes; large artifacts remain ignored.

## Genuine IDs and official admission

`heldout_admission.admit` checks an issued permission **before** any source-root
operation. Only afterwards does it read the independently pinned committed audit
and compare acquisition, split and inventory hashes against it. Caller-supplied
hashes never become their own verification expectations.

The future human-approved source layout is explicit: `acquisition.json`,
`inventory.json` (the byte-identical original Phase 4C private integrity
inventory), `source/split_csv/1cls.csv`, `source/utils/id2class.py`, remaining
acquisition-bound source files, and `sealed/<category>/Data/{Images,Masks}`.
Source-tree compatibility has **not** been checked against real test assets.
Preparing or migrating that layout needs a separately approved, byte-preserving
operational plan if existing files use different locations. Do not regenerate
the inventory or infer expected hashes from the current contents to make it fit.

After permission, admission verifies acquisition/source identity, source-file
hashes, the official split, all 10,821 inventory records, all 2,162 test records,
committed category/class counts, complete asset membership and original image
and mask dimensions. It rejects missing/extra assets, duplicate encoded/RGB
decoded test images, and overlap with original training identities. Training
identities come from the independently SHA-bound original inventory; this is
not new training or calibration. Test images and masks are checked again before
each lazily loaded frame. Arrays are never all loaded into RAM simultaneously.

Every path component rejects traversal, symlinks, junctions, reparse points and
cloud/offline placeholders before content access. Genuine POSIX relative image
IDs remain unchanged and lexicographically ordered. No real ID is renamed
`synthetic:*`, and B1's original synthetic-only validator remains intact.

Official mask semantics are parsed as literal data from the acquisition-bound
`id2class.py`, never executed. Valid nonzero official defect class values become
binary foreground (`> 0`) in original coordinates. Ground truth is not resized.
The artificial admission fixture uses class value 2 deliberately: assuming that
every original VisA mask contains only 0/255 would be incorrect.

## Trust and single-use authorization

This is a trusted-local-execution mechanism, not a sandbox against compromised
administrators, the OS, a malicious Python interpreter or monkeypatched code.
No evaluator signing key, registry installer or approval generator is supplied.

A human administrator, **after review and regular merge**, must independently
provision `C:\ProgramData\VisionGuardAI\authorization\trust.json`. The registry
pins the actual reviewed merge, reviewed PR head, activation fingerprint,
approved source root and human public RSA key. Administrators/SYSTEM must own
and exclusively control writes to the registry hierarchy. Common ancestors
must not grant ordinary principals replacement/deletion rights. Reparse trust
paths are refused; elevated evaluator execution is refused. Unusual ACLs fail
closed and require human review, not automatic permissions changes.

The private signing key stays with the human outside the evaluator. An external
trusted signer signs the canonical payload using RSA PKCS#1 v1.5/SHA-256,
modulus at least 3072 bits and exponent 65537. Platform .NET cryptography verifies
it; the evaluator does not implement cryptographic arithmetic. Canonical bytes
are UTF-8 `json.dumps(sort_keys=True, ensure_ascii=True, allow_nan=False,
separators=(",", ":"))`, with no appended newline.

The signed document has exactly `payload` and `signature`; its payload binds
every value returned by `expected_approval` plus a 32-lowercase-hex authorization
ID, timezone-aware `not_before`, and `expires`. These include explicit true
`allow_real_test_access`, actual merge/head, activation/B1/B2/development/audit
identities, all model/threshold specifications, environment, scope and exact
input/output destinations. The independent protected registry is not accepted
from a CLI argument or environment variable.

Actual clean HEAD and `origin/main` must equal the registry's regular merge,
with the reviewed head as second parent and the reviewed readiness base in its
ancestry. Execution is restricted to the canonical D: repository. Source and
environment identities must match, and actual D: free space must meet 148 GiB.
Matching fingerprints alone, current feature HEAD or a self-asserted key cannot
authorize. Missing, early, expired, altered or mismatched approvals deny before
any operation on even a hostile supplied test-root object.

Before admission, exclusive publication claims the authorization ID once. A
restart needs the exact immutable claim SHA, identical signed approval/registry
lineage and same run destination. A completed claim cannot resume or start a
new run. A new run needs new human permission. Permission expiry is checked
again during execution. This assignment creates no genuine approval or claim;
the unmerged branch cannot satisfy the real gate.

## Execution, output schema and replay

Scope is exactly 12 categories in canonical order, seeds 42/123/2026, both
models, 36 pairs/72 model configurations and 12,972 effective model-image calls.
No missing-cell substitution, seed mixing, routing, score fusion, recalibration
or post-result tuning exists. One frozen model is restored and held on GPU at
a time. The new native prediction operation is AST-equivalent to B1 except its
frame-interface type/name; restoration, preprocessing and numerical operations
remain unchanged. Model state digests are checked before/after native inference.

Each immutable model stage publishes its origin/specification, image rows with
genuine ID/label/original shape/native score, original-coordinate float16 TIFF,
uint8 binary PNG and binary truth PNG. Every retained file has byte size and
SHA-256. **Float32 maps use strict `>` against the frozen pixel threshold before
float16 export.** Image decisions likewise use each model's own score and own
image threshold. No fused explanation mask is produced.

Model-stage completion is published last, followed only by an independently
validated receipt. Paired completion references stage receipts instead of
duplicating maps. Result JSON preserves all B1 fields: separate image/pixel
metrics, distributions, per-image decisions, membership SHA and triage. Pair
manifests bind model specs, stage paths/SHAs, origin and result SHA. Final matrix
JSON includes all 36 cells, aggregation, effective scope calls, new invocation
calls, verified skipped stages and the evidence class. No partial matrix receives
a final completion marker.

Independent reducers SHA-verify/decode retained evidence and reproduce each
pair twice before completion. Resumes revalidate completed model stages and
paired results before skipping inference. Partial attempts are never overwritten.
Only a recorded classified interruption may create a new immutable attempt with
its parent failure SHA; failed hardware/filesystem/corrupt/unclassified attempts
and stale locks require human review. A partial interrupted model stage restarts
that stage, not a completed other model. Additional fault-injection calls are
recorded separately from effective scope calls.

Writes are exclusive-create, fsynced and rehashed. Failure receipts retain error
type/message, WinError where present and completed inference calls where known.
Receipt-publication failure does not erase the original exception. Writer locks
are exclusive and normally removed only by the same normally unwinding owner;
unexpected process loss leaves the lock for review. No automatic lock cleanup,
filesystem/CUDA retry, checkpoint repair or settings modification exists.

## Metrics and interpretation frozen before access

The separate genuine-ID reducer is structurally and numerically equivalent to
B1. Ranking metrics are image AUROC, AU-PRO@0.05 and diagnostic pixel AUROC.
Threshold metrics use the original frozen thresholds: image/pixel confusion,
recall, specificity, precision and F1. Undefined denominators/classes remain
explicit null/undefined records; they are never coerced to zero or dropped.
Continuous pixel ranking/distributions use the same B1 float16 methodology;
binary maps retain their float32-derived decisions.

Three-way triage is unchanged:

| PatchCore | EfficientAD | Outcome |
| --- | --- | --- |
| Normal | Normal | PASS |
| Anomalous | Anomalous | REJECT |
| Normal | Anomalous | REVIEW |
| Anomalous | Normal | REVIEW |

REVIEW stays a third outcome. Reports preserve both disagreement directions,
class-specific counts, review rate, automatic coverage, selective accuracy,
anomaly pass-through, normal rejection, anomaly review capture, normal review,
reject precision and pass negative predictive value. REVIEW is never silently
classified as correct/incorrect or negative.

All categories/seeds are reported. Within-seed category macros and separately
labelled pooled confusion/triage counts are preserved. Three-seed means and
sample standard deviations are descriptive only. Paired PatchCore-minus-
EfficientAD deltas use the same category and seed. Ranking scores are not pooled
across category/model scales. There are no significance, seed-independence or
single-metric-winner claims. Deployment interpretation still needs an explicit
operational risk/workload preference.

## Artificial acceptance and negative findings

The complete manufactured tree contains four 2×2 test images plus one training
row/category. Both classes, all four binary combinations, strict threshold
equality and all three triage outcomes are represented. All 36 pairs use the
same genuine-shaped artificial IDs without bypassing B1 protections. Every
cell and aggregate exactly matches the independently retained B1 artificial
oracle; hand-counted tests separately verify confusion, AUROC, pixel F1,
coverage/selective accuracy and undefined-class behavior.

Durable acceptance intentionally interrupts EfficientAD candle:42 after one
artificial call, preserves that failure and the completed PatchCore stage,
resumes a new attempt, finishes 36 pairs, then performs a full 72-stage validated
restart with zero new inference. A near-threshold float32 pixel that rounds to
the threshold in float16 still retains its positive native binary decision.
Adversarial checks cover corrupt/missing/extra evidence, wrong IDs/specs/seeds,
invalid dimensions/masks/nonfinite values, wrong provenance, authorization
expiry/reuse, cloud/reparse guards, volume replacement, disk budget, multiple
writers and independent watcher detection of a minted idle process disappearing.

Further native smoke was bounded to one manufactured 384×512 RGB frame per
model, compared through both old and new interfaces (four calls). Both native
scores and restored maps matched bitwise; model states stayed identical.
Original-coordinate map publication and two independent paired replays passed.
The earlier PR #24 four-call measurements are historical engineering evidence
only, not held-out performance or a repeated scientific benchmark.

A first new smoke candidate stopped on the absolute-output-path guard because
the CLI passed a relative path. Its two PatchCore interface calls and partial
publication/failure remain ignored/local and SHA-bound in compact negative
evidence. The CLI path boundary was fixed in this new, unfrozen layer; a fresh
candidate then passed. No old scientific source, map, model or threshold was
changed. Initial unit failures were engineering mocks/annotation normalization
and the platform test-key constructor, not tolerated numerical disagreement.

The first durable acceptance oracle supplied B1 fixture-index order while the
new admission correctly supplied canonical original-path order. All cell metrics
and aggregate values were exact; only the order of per-image rows differed.
That negative candidate is retained. The reference now receives the same
canonical input order; no metric, tolerance, threshold or result was altered.

## Resource envelope and remaining operational uncertainty

The original 128 GiB total run budget plus 20 GiB free-space safety reserve is
retained. A grant checks at least 148 GiB on actual D:. Runtime checks volume
identity/connectivity, remaining space and cumulative retained bytes before
publication; periodic refresh includes watcher files and all failed attempts.
After authorized admission, original dimensions provide a conservative projected
payload before inference. It must fit without deleting failures. Large maps,
models, inputs and raw logs remain Git-ignored.

New native measurements: PatchCore allocator peak 223,997,952 bytes allocated /
243,269,632 reserved; EfficientAD 115,962,368 / 153,092,096. Overall process peak
working set reached 1,963,311,104 bytes. GPU readings were points at 45–46 °C,
not device-wide peaks. Restore times were 13.752 s and 1.783 s; two-interface
call durations were 0.389 s and 0.082 s. Cache/storage/environment effects mean
these are not comparative throughput claims or bounds on real evaluation time.
The historical HDD benchmark is not rerun here.

An independent CPU watcher binds PID **and creation identity**, records periodic
heartbeats and inspects Windows disk/USB/Ntfs/storage/WHEA/GPU warnings. Runner
polling detects guard loss. Inference observes RAM, allocator peaks and GPU
temperature points; an 85 °C conservative operational stop is not a model
hyperparameter or manufacturer-rated claim. CUDA, corruption, drive replacement,
publication failure or disappearance stop and preserve available evidence.
Monitoring is sampled: it cannot promise instantaneous USB/thermal detection or
write a durable D: receipt after D: physically disappears. Power settings,
Defender, OneDrive and hardware configuration are not changed.

Actual original dimensions, connected-region counts, whole-run latency, sustained
thermal behavior and worst-case metric RAM remain unknown until separately
authorized access. Point readings and small artificial fixtures do not establish
end-to-end feasibility. Keep AC power, lid open and stable external connectivity;
any operational change or relocation needs its own approved plan.

## Subsequent human-controlled procedure — NOT EXECUTED

1. Review this draft PR, code, frozen evidence and unresolved limits. Do not treat
   it as permission to evaluate. Prefer a regular merge; preserve all history.
2. Fetch reviewed main and verify the actual two-parent activation merge and PR
   head, clean canonical D: checkout, all three evaluator/development freezes,
   original environment, local artifacts and at least 148 GiB actual D: capacity.
3. Independently provision the protected public trust registry with the final
   reviewed merge/head/fingerprint and exact approved source layout. Do not use
   this feature branch, a placeholder merge or a caller-defined expected hash.
4. The human separately authorizes real test access/execution and signs the exact
   canonical payload described above using an external trusted signer. Choose a
   validity interval adequate for a run of currently unknown duration. Keep the
   private key away from agent tooling and the repository. No signer is provided
   here, and Codex must not create this permission for itself.
5. Only then, an ordinary non-admin process may invoke
   `python -m visionguard.heldout_runner --repository <canonical-D-repository>
   --human-approval <human-signed-document>`. No alternate test-root argument or
   bypass flag exists. Authorization claims precede all admission operations.
6. On a classified interruption, preserve the same approval and immutable claim
   SHA and use `--resume-claim-sha256 <verified-claim-SHA>`. Failed/unclassified
   processes, stale locks, hardware/publication errors or expired permission need
   human review; do not retry or remove locks automatically.
7. After all 36 pairs pass independent verification, publish compact SHA-bound
   reporting with the mandatory historical-independence limitation. Do not tune,
   choose a seed/model from one favorable metric or retrospectively relabel the
   test as independently reserved.

The exact implementation commit, activation fingerprint, full local checks,
fresh artifact verification and CI status are recorded in the adjacent compact
freeze/acceptance evidence and final PR handoff. **No merge or actual evaluation
is performed by this task. All real gates remain CLOSED.**

## Frozen review handoff

Implementation commit: `810d5f26016a7369edaf363bf3162f9068a03915`.
Activation protocol: `visionguard-visa-controlled-activation-v1`.
Activation fingerprint:
`960811034e904c6f86a8e127ef0b149a67fb2ec5715e96333e813d461d4d431f`.

`reports/phase4d-b2-controlled-activation/implementation-freeze.json` binds the
source/configuration/environment/model matrix and SHA-bound compact acceptance
evidence. `acceptance-evidence.json` records 710 passed / 2 skipped full local
tests, successful 36-pair artificial execution, 72 verified restart stages with
zero new calls, four successful native equivalence calls, and all preserved
negative findings. Ruff format/lint, pip check, CLI/import, prior-freeze and
focused secret/personal-path/large-file/whitespace checks passed.

Future registry `reviewed_head` must be the **final reviewed PR head**, including
this freeze/evidence handoff, not merely the earlier implementation commit.
`reviewed_merge` must be its later actual regular merge on main. Neither identity
is guessed or supplied as an executable permission in this branch. CI is checked
on the published final PR head and recorded separately; hosted CPU/artificial
jobs do not replace the local Windows/CUDA measurements.
