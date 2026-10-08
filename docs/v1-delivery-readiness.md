# VisionGuard v1.0 delivery readiness — isolated preparation

Audited source baseline: `f2cabde46774f9c093900e69b35d07c667c47121`.
Engineering adoption is draft PR #33, stacked on accounting successor PR #32.
This local application branch is not pushed, activated or deployed. The original
evaluation, STOP/lock, historical records and incident bundles are out of scope.

## Implemented, tested, missing

| Milestone | Implemented evidence | Test evidence in the source | Required before v1.0 |
| --- | --- | --- | --- |
| Results documentation | README links public MVTec reports, comparative analysis and triage; `docs/phase-4c-visa-independent-evaluation-readiness.md` records access uncertainty | Artifact/provenance/metric checks: `tests/test_artifacts.py`, `test_comparative_analysis.py`, `test_heldout_metrics.py`; these do not establish independent historical access | Reconcile stale README/strategy status with completed VisA evidence using an approved publication summary; retain provenance and shutdown qualification; no rerun/tuning |
| Inference API | PatchCore adapter (`models/patchcore.py`), EfficientAD utilities (`efficientad.py`), map restoration (`preprocessing.py`), calibration and triage primitives; guarded `heldout_backend.py` is an evaluation backend, not an application API | Component tests exist for adapters, preprocessing, EfficientAD, triage; no HTTP-route tests at baseline | Approved immutable model registry, verified safe artifact loading, image upload route, readiness/error contracts, concurrency/timeouts and bounded resource lifecycle |
| Inspection interface | README architecture plans an interactive inspection app; no UI/server implemented at baseline | No browser acceptance tests at baseline | Single-image upload, original-coordinate heatmap, explicit model identity/threshold, uncertainty/qualification, failures visibly distinct from NORMAL, accessible keyboard/error states |
| Installation/packaging | Hatchling wheel configuration, `visionguard-ai` version 0.1.0, optional pinned ML stack and CLI entrypoints in `pyproject.toml`; dataset/setup and ML dependency docs | CI uses editable `.[analysis,dev]` installs and CLI help/plan checks | Build and inspect wheel/sdist, install from wheel outside checkout, exercise CLI/config-resource behavior on clean OS/Python environments, package app extras and reproducible hardware instructions |
| End-to-end acceptance | Frozen synthetic scientific/operational pipeline tests and dedicated safeguards matrix | `tests/test_heldout_execution.py`, `test_visa_evaluator.py`, `test_heldout_cleanup.py`; these are not browser-to-model delivery acceptance | Generated/non-held-out image -> HTTP -> approved backend -> restored map -> browser display; cancellation, malformed input, out-of-memory and clean shutdown tests; controlled non-held-out native smoke separately |
| Release documentation | Dataset/engineering protocols and limitations exist | No release-install/upgrade acceptance workflow at baseline | v1 support matrix, limitations/model cards, changelog, installation/upgrade/rollback runbook, security and dependency inventory, source licensing decision; tag/release requires separate permission |

Baseline CI declares Ubuntu Python 3.11/3.12/3.13 and additional Ubuntu/Windows
safeguards matrices. Hosted results apply to the committed engineering candidate,
not this new application preparation. Optional ML tests may skip when the native
stack is absent; green lightweight CI is not native application acceptance.

The README still says VisA test performance/evaluation has not been performed.
That status requires an explicit, reviewable documentation update, not a rewrite
of pre-run freezes. No scientific numbers were read or copied for this audit.
The source-code license is still deferred in README and no LICENSE file exists.
License selection is a release-blocking human decision, not an implementation default.

The completed evaluation must retain this exact qualification:

> Held-out VisA evaluation with historical access independence unverified.

Completion publication and shutdown success are separate. The prior incident
does not become a clean shutdown because diagnostics are now implemented.

Current isolated preparation checks: 115 tests passed and two Torch-dependent
modules were skipped (Torch is deliberately absent from this lightweight runtime).
These are generated fixtures/component checks, not native model inference.
The seven tests inside those two modules also pass separately using the existing
Torch runtime read-only with isolated source imports and manufactured CPU tensors.
No completed-evaluation checkpoint, input, prediction or scientific result is used.
The untagged preparation wheel (including the new boundary module) builds and
installs in a fresh Windows/Python 3.11 virtual environment; isolated site-packages
import, audit/protocol CLI help and dependency consistency pass outside the source
checkout. This is not a baseline candidate-wheel result, a Linux install check,
or full application/ML installation acceptance. Configs and freeze/report records
are not wheel resources; guarded repository commands still require an explicit
reviewed source checkout. Failed ad-hoc source-import setup was retained separately
and corrected by using the explicit application source import path.

## Prepared first application slice

`inspection_contract.py` and its generated-image tests provide a model-free
boundary: decoded PNG/JPEG validation, bounded encoded bytes/pixels, immutable
image identity, registry identifiers (not client filesystem paths), explicit
artifact/preprocessing identities, finite scores, original-coordinate maps and
strict preexisting thresholds. It does not load a checkpoint, run inference,
calibrate, prove registry payload validity, expose HTTP or implement a UI.

Reversible defaults for the next slice:

- Local-first FastAPI service as already planned; loopback binding, no remote
  upload or external network exposure. One bounded inference request at a time.
- Server-managed verified registry; choose a model explicitly. Do not infer a
  universal winning model or average incompatible score scales.
- Plain browser interface served by the API; no separate frontend build needed
  for the first slice. Display NORMAL/ANOMALOUS as model decisions, not a product
  safety guarantee. Three-way triage requires verified paired evidence.
- 10 MiB encoded input and 4096 x 4096 decoded-pixel ceiling; PNG/JPEG only.
  No persistence of uploads by default. Versioned metadata, no client paths.
- Start with injected manufactured backends for route/browser tests. Native
  checkpoint loading and smoke tests use approved non-held-out fixtures only.

Delivery sequence: finish boundary -> HTTP/error/readiness slice -> inspection
UI -> wheel/clean-install acceptance -> native non-held-out acceptance -> results
and release documentation review. Registry/model artifact licensing, deployment
exposure, activation and release are not silently authorized by this plan.
