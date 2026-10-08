# Application limitations and failure semantics

> Held-out VisA evaluation with historical access independence unverified.

This qualification remains visible in UI/API and release documentation. The app
does not certify historical independence, improve metrics or fix the historical
GPU allocation leak. The parent v5 work is diagnostics/truthful failure handling;
original exceptions, zero-allocation guard and retained-lock behavior are intact.
Completion publication and clean shutdown are separate facts.

The service accepts a single PNG/JPEG at a time: 10 MiB encoded, at most
4096×4096 decoded pixels, single frame. It checks decoded format rather than
filename, verifies and fully decodes before inference. Multipart bodies and
client model paths are not accepted. Request bytes stay in memory; no upload
output directory is created. This is not a secure memory-erasure guarantee.
Peak working memory includes decode, tensors, immutable float-map rows and PNG
encoding; a large permitted image can require substantial RAM. No production
memory-isolation or throughput claim is made.

Non-upright EXIF orientation is refused: browser auto-rotation/mirroring would
otherwise misalign the original preview with raw-pixel model maps. No silent
EXIF transformation changes approved model preprocessing. Use an upright image
with absent/identity orientation; normalization of real inputs needs review.

Default upload deadline is 15 seconds; inference response deadline is 30 seconds
(bounded configuration caps are 60/300 seconds). The browser abandons its wait at
60 seconds. One executor slot and a concurrency cap reject overlapping work.
A timed-out inference thread may still be running: the worker is quarantined,
new requests are refused, and the busy slot remains until actual work finishes.
No automatic restart, forceful cancellation, repair or quiet recovery exists.
Application shutdown drains active work off the event loop before closing its
backend; a permanently hung call can prevent graceful exit. Process isolation
is a future reviewed capability. Upload disconnects admit no inference. Cancelling
an HTTP task after inference admission quarantines the worker without cancelling
native execution; a TCP/browser disconnect is not proof the server task was cancelled.

Model residency is one immutable backend/identity pair. Switching first unpublishes
the old pair, then closes it, then constructs the replacement. Failed close or
construction quarantines the worker: no stale identity, fallback model or automatic
retry. Failed close retains its owner in `cleanup_pending_model_id`; initialization
failure leaves resource cleanup unverified. A prior transition failure makes
shutdown raise rather than claim clean cleanup; it is not retried during shutdown.
Operator review is required; this service cannot prove a failed constructor freed
all resources. Permission refusal before transition does not destroy a resident
model but prevents inference and makes it ineligible for readiness.

Decode/contract failure returns 422; unknown ID 404; unsupported body type 415;
upload limit 413; upload timeout 408; upload disconnect 499; busy 429; cross-origin/native input-role
refusal 403; inference exception 500; inference timeout 504; quarantined worker
503. Error bodies have `status: failed`, `decision: null`; tracebacks remain in
local structured logging. Failed backend transitions return 500 even when the
original constructor/close exception is an InspectionError. Invalid model outputs
return 422 and quarantine; input-validation failures alone do not quarantine.
Browser failures clear previous decision and images; readiness is rechecked before
re-enabling submission. Late success or failure for a changed selection is ignored.
Host-level/middleware/network failures might not have the JSON error shape;
the UI still clears the result. Changing selected file/model/input-role also clears it.

NORMAL means only **this verified model score is not strictly above its existing
threshold**, including equality. It is not proof of defect absence, a safety
guarantee or the paired-model PASS/REVIEW/REJECT policy. No universal winner is
selected. Original-coordinate grayscale heatmaps are per-image min/max display
normalization, **not calibrated probabilities or threshold masks**. Constant maps
have no contrast. CSS scales display only; canvas pixel dimensions remain exact.

Loopback binding, host checks, same-origin POST/client headers and restrictive
CSP reduce browser exposure. They do not authenticate local users or make public
deployment safe. TLS, multi-user access control, retention/audit policy, malware
hardening and production process isolation are missing. A dated dependency/advisory
inventory is review evidence, not a penetration test or a security certification;
see `v1-security-and-rights-inventory.md` and the external advisory reports.

Manufactured intensity tests verify integration only. Manufactured CPU tensor
tests verify adapter plumbing/identity rejection, not trained native model
acceptance. No development checkpoint has been opened for application serving;
no held-out asset, metric, map or prediction has been inspected in this task.
Linux checks use existing Ubuntu WSL, not a native Linux desktop/GPU deployment.
Broader Python/platform/ML/browser support needs the declared hosted matrix and
authorized native acceptance before release.
