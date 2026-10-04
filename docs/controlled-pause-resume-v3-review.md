# Controlled pause/resume v3 — engineering review

Status: design and engineering implementation in progress. Issue #29.
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

Implementation evidence, exact commands, limitations, synthetic equivalence,
full checks, successor identity and final-head CI will be added before handoff.
Do not merge; eventual regular merge only, never squash/rebase.
