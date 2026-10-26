# Operational and native-security disposition — restricted local CPU path

Historical investigation recorded 19 October 2026; admission corrections and
restricted disposition authorised 26 October 2026. This dated record does not
claim that upstream advisory status was refreshed after the original review.

This addresses the retained observer/GPU incidents and PT2 advisory within the
existing local CPU application scope. No trained artifact or scientific output
was loaded, no consumed authority reset, and no historical environment upgraded.

## Retained observer failures

The earlier startup diagnosis reproduced a Windows sandbox socketpair accept
stall without ML or instrumentation; the same prerequisite worked in the
loopback-enabled context. That explains an execution-context hazard, but the
original silent attempt lacked enough evidence to prove its exact blocked frame.

Import-only audit/profile probes and periodic native faulthandler dumps also
produced retained 0xC0000005 exits. Legacy unfiltered frame-locals instrumentation
had independently identified defects; later reviewed instrumentation filters
code/event identity before accessing only relevant locals and runs after imports.
A later complete ordinary test attempt also ended with a timed-dump AV. These
observations do not establish a CPython/PyTorch fault or a single historical root
cause. Passing CI does not erase those failures.

The new opt-in scripts/inspection_test_observer.py follows the already reviewed
bounded Python-stack approach for fixture checks. It disables only periodic
native timed dumps via faulthandler_timeout=0, retains pytest fatal reporting,
writes fsynced setup/call/teardown outcomes, bounds stack output, and reports
observer failures as failures. Stack capture reads code locations, never frame
locals. Capture occurs in a short-lived call, releasing raw frames even when the
depth is truncated; the observer's waiting thread cannot retain a model frame.
A weak-reference worker-lifetime regression verifies this manufactured ownership
property. This is a diagnostic mitigation, not an upstream AV fix.

## Historical GPU cleanup

The current-device zero-live-allocation check is unchanged. Reserved cache is
reported separately; empty_cache cannot free a still-owned tensor. Existing
manufactured probes distinguish model cycles, borrowed references, retained
exception tracebacks and reserved cache, and preserve the original cleanup error,
unsafe watcher exit and writer lock on failure. They were not weakened.

The retained historical incident has no sufficient native allocation-owner
backtrace to identify the original owner. Existing source/reference analysis
supports hypotheses, not a verified repair of that incident. Do not clear its
STOP/lock, assert safe original pause/resume or resume it. No new test can rewrite
the old exit evidence. A future GPU investigation would need a new synthetic,
owned-process CUDA allocation/owner capture in an isolated runtime; trained
replay is not justified by this evidence gap.

The v1 inspection adapter is CPU-only. Its drain/unpublish/failed-close quarantine
and new untrained ownership tests do not establish GPU application support or
repair the original GPU evaluation. That historical limitation remains explicit.

## PT2 reachability and current authoritative record

The [GitHub Advisory Database entry](https://github.com/advisories/GHSA-33x2-ppm4-v46v)
listed unknown affected and patched versions in the 19 October 2026 review. Its package-less unreviewed
metadata cannot support Dependabot alerts. The
[PyTorch fix proposal #176791](https://github.com/pytorch/pytorch/pull/176791)
was closed without merge at that review. It describes unsafe pickle fallback in export/PT2
deserialization. No patched Torch release was established by this review.
The repository-scoped security-advisory URL returned 404; the global advisory
and upstream proposal were accessible. The NVD response supplied no usable body.

The actual application path is registry approval -> same captured artifact bytes
-> require_state_archive -> torch.load(BytesIO, map_location="cpu",
weights_only=True) -> identity/canonical checks -> native_from_state with strict
restoration -> generated/development image. The installed reviewed Torch
serialization source routes ordinary ZIP states to the restricted unpickler;
its TorchScript branch refuses weights_only=True. No application call enters
torch.export.load, PT2 package loaders or weights_only=False. The historical
benchmark unsafe loader is not imported/invoked by this application factory.

The new metadata-only gate positively admits ordinary torch.save state ZIPs:
one safe root, required data.pkl/version, known state metadata and numeric tensor
records, at most 8192 entries, no encryption/compression, duplicates, traversal,
raw backslashes, export records or TorchScript code/constants. Raw orig_filename
values containing NUL are rejected before path normalization or Torch import.
For ZIP_STORED records, compressed and uncompressed sizes must agree. Each
declared size and the aggregate declared payload must not exceed len(raw), the
actual captured archive byte budget; no claimed payload is read or allocated.
It reads no pickle payload. Unsupported legacy state formats are refused without a fallback.
Raw orig_filename is checked because Windows ZipInfo normalizes separators.
Restricted loading, exact hash/permission, canonical identity and strict
restoration remain mandatory. Both native architectures are checked with newly
generated states, not previously trained artifacts.

This closes an application input-format ambiguity and prevents the supported
factory from accepting export/JIT containers. It does not fix Torch export,
prove all dependencies secure, authenticate a hostile local operator, or approve
untrusted model uploads. On 26 October 2026 the owner accepted only the documented
local, single-worker, CPU-only, independently pinned ordinary-state path with
these corrected controls and retained restrictions. This is not upstream
remediation, package-wide clearance, repaired historical crashes, GPU support,
permission to load a trained artifact or release/deployment approval.
No dependency version was changed on speculation.

## Acceptance and operating boundaries

The original bounded trained pass remains exact-scope evidence. Changed native.py
and the new format gate were not part of those source hashes. Their untrained
regressions establish the new controls and generated-state compatibility only;
they do not renew whole-candidate trained acceptance. Scientific preprocessing,
calibration, model implementation and dependency pins are unchanged.

Local manufactured mode is available without a model. Native operation needs a
new independently established exact-artifact application-use authority; the
permanently consumed test permission is never reusable. A further trained test
should be requested only if the final review identifies a concrete uncovered
acceptance claim, with that exact scope, after all independent work is complete.
There is no authority here to run such a test, deploy, tag or redistribute models.

Held-out VisA evaluation with historical access independence unverified.
