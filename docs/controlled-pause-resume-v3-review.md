# Controlled pause/resume v3 — engineering review

Status: v3 engineering implementation frozen; complete local verification passed.
Final-head CI is recorded on the DRAFT PR (no merge authorized).
Issue [#29](https://github.com/ahmad-527/visionguard-ai/issues/29),
DRAFT PR [#30](https://github.com/ahmad-527/visionguard-ai/pull/30),
branch `codex/controlled-pause-resume-v3`.
**DRAFT review only; final-test evaluation lock CLOSED. No real execution.**

Reviewed baseline main/origin/main:
`808319e29164d69a72d7de6286fc6b014bc546c3`, with regular merge parents
`91b5b45a82ccef042ad13535d2456ea7d6013885` and
`63e2787989cfe84e3363bfcfd21f37941620560d`. Clean main verified before edits.
V2 fingerprint `8b30e6b923c8bb49f457b50860a0adcc62d6bf1031405e28e60ab1c28ddd00b0`
and original v1, B1/B2 and all 72 model specifications reproduced exactly.

## Design established before source edits

One immutable request targets the current authorization ID, claim and writer
invocation nonce/creation identity. Requests and acknowledgements are different
from a final cleanup receipt. Requests are checked before stages, before each
inference call through the existing stage hook, between pairs and before summary.
Native calls are not asynchronously killed. A current call and atomic image
publication may finish before acknowledgement. A partial model stage is retained
as interrupted and restarted in a new attempt, never reused as complete.

Safe cleanup requires verified writer ownership, release of backend references
and CUDA allocator state, independent watcher acknowledgement and actual child
exit, and ownership-checked lock release. The runner records cleanup completion,
not its own subsequent OS death. A separate human status command checks the
recorded process creation identity independently; missing/blocked inspection is
not proof of safe pause. Hardware, watcher, publication or provenance failures
remain fail-closed and leave review evidence/locks rather than a PAUSED claim.

Resume requires the original signed approval, immutable claim hash and previous
invocation's validated safe-exit receipt BEFORE real admission. The signed window
is exactly 120 hours from not_before; no renewal or expiry bypass. The same origin,
model/threshold specifications and output location are required. Completed stages
are independently verified before skip; interrupted partial stages get new
numbered attempts. Old requests cannot stop a fresh invocation.

Archive exact changed predecessor source bytes. V3 independently verifies both
historical v1/v2 fingerprints and source maps against those archives and unchanged
files, and binds current operational changes separately. Preserve original freezes
and scientific contract. New public trust requires a new reviewed regular merge,
PR head and v3 fingerprint; installed v2 trust is not changed by this assignment.

No job scheduler, calendar automation, auto-renewal, stale-lock cleanup or retry
of failed/unclassified attempts will be added. The Wednesday 7 October class
deadline requires an early manual request and verified process/GPU clearance
before 14:00 Berlin; no hard pause-time guarantee can precede real execution.

## Preserved history and boundaries

The exact broad rg invocation, stop-clock observation, unknown dataset metadata
access and default-pytest scratch-location deviation remain in the unchanged
Stage B report and engineering evidence. No retrospective dataset inspection or
external scratch cleanup is authorized. All new pytest scratch is a verified-new
ignored repository-local directory. The existing public trust registry, iPhone
private key, power safeguards, ACLs and 72 models must not be accessed/modified by
this amendment. No real approval/claim, admission, inference or metrics run.

Human access history **UNKNOWN**, independent reservation **NOT ESTABLISHED**.
Future reporting: **Held-out VisA evaluation with historical access independence
unverified.** Historical training failures/migrations remain unchanged.

## Frozen identities and unchanged science

Current v3 fingerprint:
`0334a17b5b5c8f954e692c4caa47f93eaead0228acb6c45a082a75ca3fbc1c47`.
The fingerprint binds the new protocol, runner/control/watcher/gate, all affected
tests, dedicated Python 3.11–3.13 CI, and exact historical source archives.

| Evidence | Unchanged identity |
| --- | --- |
| V1 activation | `960811034e904c6f86a8e127ef0b149a67fb2ec5715e96333e813d461d4d431f` |
| V2 ACL successor | `8b30e6b923c8bb49f457b50860a0adcc62d6bf1031405e28e60ab1c28ddd00b0` |
| B1 implementation | `0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49` |
| B2 readiness | `1b44805040f211fdf75bd54561bae1b61f0fe2912d25ff734c355ff1202ba6cc` |
| 72-cell development publication | `d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061` |

All 72 specification dictionaries, image/pixel thresholds, normalization/weight
identities and environment compare exactly against the reviewed v2 freeze.
No model payload is read, retrained, recalibrated or modified in this amendment.
The original B1/B2 verifiers pass. Native backend, metrics, aggregation, triage,
sample membership, stage export format and inference mathematics are unchanged.
Scope stays 12 categories × seeds 42/123/2026 × two models: 36 pairs, 72 stages,
12,972 logical model-image calls. Restarting a partial stage repeats its already
executed calls; retained failure receipts expose those additional physical calls.
They must not be misreported as new samples or omitted from operational cost.

## Minimal operational implementation

`heldout_pause` reads only run-output receipts. Each invocation has an immutable
owner record (nonce, PID, creation identity, source origin and evidence class).
One immutable request binds this owner SHA, authorization ID and original claim.
Repeated matching requests return the same hash; malformed, mismatched, stale or
future-dated requests fail closed. Only the current invocation checks its own
request, so an old request cannot pause the next invocation.

The existing per-image stage `check` hook is retained. A small iterator wrapper
also checks before fetching the next image. Checks run before stage loading,
before each inference, after stages, around pair publication, and before summary.
Authorization expiry is checked at every real checkpoint. Genuine native OS
interruptions are not automatically classified as safe human pauses.

Acknowledgement raises a dedicated `InterruptedError` subclass. The existing
stage publisher records partial-stage failure and its exact completed-call count.
Completed-stage evidence is left byte-identical. No prediction is scheduled after
acknowledgement. Backend references and exception traceback references are
released; CUDA is synchronized, its cache emptied, and remaining allocated bytes
must equal zero. Driver/context memory can remain until OS process exit.

The watcher validates its runner's creation identity and its own launch lineage.
Windows venv launcher PID and actual Python worker PID can differ; both are
handled explicitly. An immutable finish request must match owner nonce; the
watcher publishes a SHA-bound stopped acknowledgement. The runner observes exit
code zero and confirms worker-process absence before releasing its exact-owned
writer lock. Watcher timeout/failure, residual allocations or owner change produce
cleanup-failure evidence and no safe pause. No forced termination or lock repair.

Final `safe-exit.json` means cooperative cleanup verified, not that the runner can
observe its own death. CLI exit 75 means intentional pause, not evaluation success.
Only a separate `--status` process may return `SAFE_PAUSED`, after checking runner
and watcher creation identities are absent, ownership lock is gone, and original
claim/request/acknowledgement/failure/cleanup evidence matches. Blocked inspection
is unresolved, never absence. An unclassified later invocation also blocks resume.

Resume verifies the original signed approval and same immutable claim, then the
previous safe invocation receipt BEFORE dataset admission. After admission, exact
origin is checked again; completed stages are independently hash/replay validated
before skipping. A partial stage receives the next numbered attempt and binds
the preserved parent failure SHA. Failed or unclassified attempts are not retried.

## Human commands after review and separate execution authorization

These commands are documentation only. None were run against a real authorization
ID. Replace each placeholder with the original human-approved value. Use another
PowerShell window while the evaluator runs:

```powershell
Set-Location 'D:\VisionGuardAI\repository'
$AuthId = '<original 32-hex authorization ID>'
$Invocation = '<active 32-hex invocation nonce>'
.\.venv\Scripts\python.exe -m visionguard.heldout_pause --active --authorization-id $AuthId
.\.venv\Scripts\python.exe -m visionguard.heldout_pause --request --authorization-id $AuthId --invocation $Invocation
.\.venv\Scripts\python.exe -m visionguard.heldout_pause --status --authorization-id $AuthId --invocation $Invocation
```

`--active` supplies the nonce and PID/creation identity; copy the returned nonce.
`REQUESTED_NOT_SAFE` and `ACKNOWLEDGED_NOT_SAFE` are not laptop-clearance statuses.
A validation error or missing receipt requires review, not a cleanup/retry command.

Resume the same run in the canonical checkout:

```powershell
$Approval = '<path to the ORIGINAL human-signed approval JSON>'
$ClaimSha = '<ORIGINAL immutable claim SHA-256>'
.\.venv\Scripts\python.exe -m visionguard.heldout_runner --human-approval $Approval --resume-claim-sha256 $ClaimSha --resume-invocation $Invocation
```

The signature window is **exactly 432,000 seconds / 120 hours** between signed
timezone-aware `not_before` and `expires`. The approval additionally binds
`authorization_validity_seconds: 432000`. Time begins at the original human
signing/validity start; there is no resume-time reset, renewal or expiry bypass.
The same approval, ID, claim, source/output binding and scientific lineage remain.

## Wednesday operational limit and clearance procedure

For Wednesday **7 October 2026, before 14:00 Berlin**, request early (for example
13:00, or earlier if observed model/verification times warrant it). This is a
manual recommendation, not a timer or a guaranteed one-hour response bound.
Possible 23:00 resumption remains within the original signed window or is refused.

No native inference/kernel is asynchronously killed. Worst-case acknowledgement
latency includes the current model load/state hash, current image fetch/native
prediction and immutable writes, or an in-progress stage verification/pair reducer.
Cleanup additionally includes synchronization, garbage collection and watcher
exit (a 10-second watcher wait limit; timeout means failure, not safe pause).
Before any real execution, native wall-time bounds are UNKNOWN. A stalled GPU,
storage or native operation can exceed any practical bound; do not promise the
laptop will be free merely because the request was recorded.

Before leaving for class:

1. Require the separate status command to report `SAFE_PAUSED`.
2. Inspect only the recorded runner/watcher PIDs and their creation identities.
   `Get-Process -Id <recorded PID>` not-found is distinct from access-denied/error;
   PID reuse is a different process, never something to kill.
3. Check `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`
   read-only. No evaluator-owned PID may remain. WDDM may omit some process memory;
   a global nonzero VRAM reading can include desktop/other apps and is not itself
   proof that the evaluator is active or inactive.
4. Confirm the runner terminal exited with intentional-pause code 75 and watcher
   observed exit zero. If inspection is blocked or another evaluator is present,
   clearance is unproven: stop for human review; do not delete locks or kill jobs.

## Future public-only trust update (not performed here)

After independent review, merge this PR by a **regular merge commit only**. Verify
the new two-parent merge has this PR's final head as its second parent and descends
from `808319...`; clean HEAD and origin/main must match that new merge.
A separately authorized human administrator may then update public-only trust
fields `reviewed_merge`, `reviewed_head`, `activation_fingerprint` to that NEW merge,
NEW final PR head and v3 fingerprint. Preserve independently approved `source_root`
and existing RSA public `modulus`/`exponent` and the reviewed protection policy.
No agent registry installer, private key access, key generation, signing, real
claim or permission/ACL change is included. Installed v2 registry was neither
read nor modified; its old fingerprint cannot authorize v3 execution.

## Verification and negative history

All tests use manufactured inputs, CPU workers or mocked authorization/CUDA.
The one legacy Windows RSA-key-generation test is explicitly deselected locally;
it is skipped on Linux CI. No RSA signing credentials were generated.
Completed/partial synthetic attempts and earlier failed batches remain in ignored
repository-local scratch; no scratch reuse or cleanup was performed.

- Batch 1: 8 passed, 81 setup errors, 1 deselected. The verified-new basetemp's
  parent was absent; no external/default scratch or affected test body was used.
- Batch 2: 86 passed, 3 failed, 1 deselected (454.65 seconds). Full synthetic
  equivalence and cross-process resume passed; watcher PID mismatch and two
  missing manufactured historical-evidence fixture files were exposed.
- Fast batch 3: 66 passed, 3 failed, 7 deselected, confirming the same defects.
- Fix batch 4: 22 passed, 6 deselected after actual worker/launcher binding and
  complete manufactured fixture evidence.
- Watcher batch 5: 1 failed, 22 deselected: a test-only Popen mock also intercepted
  PowerShell identity queries. No real event-log/input access occurred.
- Watcher batch 6: 1 passed, 22 deselected after limiting that mock to the newly
  created CPU watcher. Production start and safe-exit handshake passed.
- Preliminary v3 freeze `27f7305ca089b82a05d106fd875d649dcf3f948696019b47b207c578f3eb7168`
  is preserved separately (file SHA `8c1efbb0f98eca5b530964a7c3f13168de006894e1b2af4f32f5025b4b7b3ba4`).
  A missing import-separator blank line was caught by lint immediately before
  the full suite; only the test source formatting differs in the final v3 freeze.
  The old v1/v2/B1/B2 freezes were not regenerated or edited.
- Full batch 7: 895 passed, 1 failed, 2 Windows privilege skips, 1 RSA test
  deselected (414.80 seconds). The failing assertion counted a pair validation
  receipt as a model-stage receipt; scientific equivalence itself passed.
  The assertion was corrected, and the resume validator gained an explicit
  pre-admission refusal for any retained watcher STOP marker.
- Intermediate engineering freeze `395e2adb7be8ea7f34728a4794ff9d9354340871f7c9a3ef24be5e4b0f201f74`
  is preserved under `intermediate-freeze-v3.json`, not current authorization.
  Only `implementation-freeze-v3.json` and its current fingerprint are selected
  by the runner; preliminary/intermediate candidates never authorized execution.
- Targeted batch 8: 21 passed, 3 deselected after the receipt-count correction and
  STOP-marker gate test. No scientific module changed.

- Final full batch 9: **896 passed, 2 Windows privilege skips, 1 RSA-key test
  deselected**, zero failures, 427.85 seconds. Exact final-source v3 reproduction,
  Ruff format/lint, pip check, B1/B2 and historical source verification pass.
  Native import paths independently resolve to this D repository.
- Synthetic full-scope comparison: 36 pairs, 72 stages, 288 logical calls,
  one preserved partial call, exact scientific cells SHA
  `f8aae1b2bac4191a9c31746a08f2207737f88e48fe53d26e3eff9b201d7a9065`
  and aggregate SHA
  `3fe710a7f28d86aa6b7d9d1a356e15cb5d21bcf7a90c44b1f12d81cebeb1e3e3`.
  All validated prediction records and their map hashes match the uninterrupted
  run. Ten prior interrupted files remain byte-identical. Fresh CPU process pause
  exits 75; separately confirmed process clearance and fresh-process completion
  skip the previously completed model stage. GPU failure tests are mocks, not a
  native inference or hardware acceptance claim.
- Final-head CI must pass all five workflows on Python 3.11/3.12/3.13; the exact
  SHA and job outcomes are recorded in PR #30 after execution. The tracked receipt
  never claims an unobserved CI result. Secret/personal-path/large-file/whitespace
  checks pass; only compact text/config/source archives are staged.

Compact verified receipts are in `reports/controlled-pause-resume-v3/engineering-evidence.json`.
No real test assets or test-performance artifacts were accessed this amendment.
The earlier metadata-access uncertainty is NOT erased by this scoped statement.
Do not merge autonomously; stop at DRAFT independent review.
