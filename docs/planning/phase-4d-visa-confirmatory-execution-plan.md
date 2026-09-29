# DRAFT — NOT AUTHORIZED FOR EXECUTION

## Phase 4D proposal: VisA confirmatory execution

This document is a review agenda, not permission to train a full matrix or open
the test seal. Phase 4C did not begin Phase 4D. The current code deliberately
refuses final-test execution even with matching confirmation flags.

### 1. Human decision gates before scheduling

1. Review/merge Phase 4C separately; preserve its pre-data freeze chronology.
   **DO NOT SQUASH-MERGE**: a regular merge must retain the distinct pre-data
   freeze commit `c019f5515ee7692b42cda5bcc0595a4d3f00ec1e` in main history.
2. A named custodian must attest to prior VisA test access by contributors,
   including any use of images/labels/predictions/results for scientific choices.
   If credible reservation is impossible, STOP: do not rename exposed evidence
   independent and do not silently substitute a split.
3. Review the resolved fit-memory engineering gate: preallocated exact chunks
   passed native-reference equivalence and the largest category (pcb3, 815
   fitting normals, seed 42) completed on the RTX 3070 Ti Laptop. Peak allocated
   CUDA was 7,519,790,080 bytes; allocator reserved was 8,646,557,696 bytes. The
   latter is not measured physical VRAM residency. Headroom, paging and other
   platform behavior are not established; larger hardware remains an option.
   No category reduction, smaller resize/coreset or rescue tuning is allowed.
4. Review the resolved development dispatcher/cross-process acceptance in the
   [completion record](../phase-4c-visa-independent-evaluation-readiness.md#completion-pass-measured-engineering-gates).
   Exact PatchCore and EfficientAD short staged restarts passed; no full matrix
   or final-test evaluator was run. A separate evaluator still requires review
   **before any test exposure**. If source
   hashes change, record a new reviewed freeze before test; do not simply ignore
   the mismatch. Scientific changes require a new protocol version/rationale.
5. Approve a written time/storage/retention budget and stable power/cooling plan.
   Reserve at least 100 GiB additional working space; retaining multiple full
   map copies or many attempts can require more. Use a location suitable for
   long-lived scientific data rather than relying on sync-client behavior.
6. Give explicit scoped authorization for full normal-only training and later
   one-time test evaluation. Private MVTec access/submission is a separate issue
   and remains prohibited. Define a decision-maker's risk/review-workload
   preferences before interpreting operational benefit, not after results.

Engineering evidence is available for gates 3–4, not automatic human approval.
Gate 2 remains **PENDING HUMAN ATTESTATION**; use the
[declaration template](../attestations/visa-test-access-history.md). No full
execution should be scheduled before these reviews and explicit authorization.

### 2. Fixed scientific inputs

Use the precise identities in
[`protocol-freeze.json`](../../reports/phase4c-visa-readiness/protocol-freeze.json)
and reproduce them with `visionguard-visa-protocol`. The source release is
VisA_20220922, pinned official one-class CSV, all categories in lexical order:
candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2,
pcb3, pcb4, pipe_fryum. Seeds: 42, 123, 2026. Within each model execute category
order, then seed order; pair identical category/seed and image ordering later.
There are 36 cells per model, 72 total; no best seed or early winner selection.

Development membership is the frozen SHA-ranked official training normals:
7,795 fit, 864 calibration. Use copied normal-only roots with exact membership
and image hashes. Never open official test CSV/labels/masks during ordinary fit
or calibration. Retain approved pretrained and auxiliary-data hashes. Reverify
changed assets, but reuse unchanged verified cache state rather than downloading
or repeatedly rehashing gigabytes without a reason.

PatchCore uses the frozen architecture/preprocessing/coreset/neighbors.
EfficientAD uses PDN-S and the unchanged 70,000-step schedule. Fit teacher
statistics and final native map normalization using fitting normals only.
No normalization fitting on calibration normals. Calibrate image/pixel maxima
on the full category calibration subset; reject inadequate/nonfinite evidence.
Keep separate model scales and thresholds. Freeze every cell's final checkpoint,
normalization and thresholds before opening test. Do not reuse the tiny smoke.

### 3. Engineering acceptance before scientific execution

- The normal-only `visionguard-visa-develop` dispatcher now wraps
  `DevelopmentDataset` and `ExecutionState`, binds immutable attempt origins and
  supports plan/status/run-cell/resume-cell. Its full 36-cell-per-model schedule
  is declared, not executed. It contains no final-test path.
- PatchCore cross-process acceptance is RESOLVED on the declared small fixture:
  frozen feature weights, exact embedding order/RNG, original coreset indices,
  final bank/model and all calibration-normal scores/maps/thresholds agree.
  Interrupted coreset work restarts from durable embeddings; partial greedy
  iterations themselves are not checkpointed. Largest fit acceptance also passed.
- EfficientAD dispatcher integration is RESOLVED on two-step normal-only smoke:
  optimizer/scheduler/RNG, training/ImageNette streams, fit normalization and
  calibration survived new processes with exact final model/calibration results.
  This does not establish full 70,000-step thermal or power-loss reliability.
- Validate stage progression, final-step checkpoint selection, artifact schema,
  same-seed pairing, deterministic lexical image order, nonfinite handling and
  original-coordinate map restoration. Never infer a missing field or threshold.
- Validate hardware memory headroom on fitting normals without evaluating test.
  Keep one GPU job active; do not use category difficulty to select probes.
- Keep provenance: code commit/source hashes, protocol fingerprint, environment,
  hardware, archive/split/audit/membership, category/seed and pretrained/auxiliary
  identity. A resume mismatch means STOP, not regenerate metadata to match.

### 4. Durable attempts and interruption operations

Each matrix cell owns immutable numbered attempt directories. Publish manifests
atomically only after durable checkpoint/artifact writes and their hashes. Record
active stage and reason separately for interruption versus failure. Resume into
a new attempt referencing the prior validated checkpoint; never erase failed
attempts. Preserve all preemption, disk, CUDA and validation failures.

Before skipping a completed cell, hash-validate final checkpoint, calibration
and every declared output. A file merely existing is insufficient. Do not use
best checkpoint or best seed. If a lock remains after a crash, inspect process
state and origin before human-approved lock recovery. Do not infer a stopped
process from missing logs or automatically delete stale locks.

On CUDA error, abnormal thermals, shutdown/WHEA-type symptoms or disk pressure,
stop scheduling work and preserve state. Do not automatically retry suspected
hardware failure. On a normal authorized interruption, validate all identities
and exact checkpoint bytes before continuation. Measure new wall-clock and
active time separately; do not turn pauses into model speed claims.

### 5. Sealed evaluation authorization and once-only access

A separate reviewed evaluator must require explicit independent-test confirmation,
exact protocol fingerprint and dataset-audit hash, plus an auditable human
authorization record. Existing Phase 4C flags do not unlock it. A custodian should
use OS account/filesystem isolation so training cannot read sealed test data.
Expected identities must be loaded independently from repository-verified frozen
configuration, never assigned from supplied values. Phase 4C now tests arbitrary
self-asserted fingerprints/audits and correct identities: all remain CLOSED.
The human attestation template cannot authorize execution by itself.

First validate that all 72 final development cells and thresholds exist, their
hashes/provenance agree, and the metric/artifact implementations are frozen. Only
then permit the evaluator to load the official test assets. No preliminary
category pilot or test feedback for tuning. Evaluate the frozen final model once
per cell on the exact official test set. If interrupted, resume durable test
artifact writing without changing settings and record any recomputation; do not
count partial matrices as confirmatory completion.

Keep per-image portable IDs/order, original shape, image score, threshold and
strict model decision; preserve float16 continuous TIFF and strict binary PNG
per model, map hashes, checkpoint/threshold identities, and aggregate metric
numerators/denominators. Compute thresholded maps before float16 quantization;
reject nonfinite or unrepresentable maps. No model-map fusion or postprocessing
rescue. Full test manifests remain ignored if licensing/privacy rules require;
publish sufficient compact hash-bound evidence for audit.

### 6. Prespecified reporting and claims

Report all category/seed cells and both model policies on identical samples:
image AUROC and AU-PRO@0.05 are ranking metrics; confusion counts, sensitivity,
specificity, precision, image/pixel F1 use frozen normal-calibrated thresholds.
Apply unchanged `visionguard-dual-model-triage-v1` only to same-seed pairs.
Both normal → PASS; both anomalous → REJECT; disagreement → REVIEW. Retain both
disagreement directions and every class-conditioned PASS/REVIEW/REJECT count.

Use the Phase 4B rate definitions and zero-denominator behavior unchanged. Report
category/seed tables, per-seed pooled counts/rates, unweighted category macros,
then mean/sample-SD across the three seeds. No pooling repeated seeds into an
inflated sample size, no silent exclusion of undefined categories, no partial
complete-matrix summary. Seed variability is not a generalization confidence
interval. Any inferential procedure must be reviewed before opening test.

Assess anomaly pass-through risk, false rejection, review workload, automatic
coverage and selective accuracy jointly. A lower automatic error rate achieved
by near-universal review is not alone evidence of operational superiority.
Preserve unfavorable outcomes and operational failures. No forced winner,
production-readiness claim, literature-as-local-result claim or private benchmark
claim. Once inspected, VisA outcomes become exposed for subsequent rule design;
later tuning needs different independent evidence.

### 7. Budget and scheduling decision

See [capacity evidence](../../reports/phase4c-visa-readiness/capacity-evidence.json).
Historical EfficientAD training suggests 87–245 active GPU hours for 36 cells,
with a median-based 125 hours; this is extrapolation, not measured VisA speed.
PatchCore prior whole-cell timing varies widely and new fit memory invalidates
straight-line prediction. Combined 4–16 continuous days is only a historical
sensitivity envelope, excluding pauses and new evaluation overhead; it must not
be advertised as a promise. Largest-fit worker host peak was 2,804,207,616 bytes;
this is not a full-workflow/system-wide peak. CPU evaluation time remains unmeasured.

Uncompressed test-map payload for both models/all seeds is ~54.4 GiB. Add at
least ~2.5 GiB EfficientAD resumable checkpoints, ~1.4 GiB PatchCore coresets,
pretrained assets, temporary embeddings, duplicate final/attempt checkpoints,
metric scratch and retained failures. Approve retention before removing anything.
Largest-fit temporary/durable attempt files measured 5,320,840,756 bytes before
the report; the failed Windows journal-publication attempt retains another
4,669,107,860 bytes. These are measured engineering file totals, not a replacement
for full-matrix storage budgeting. Membership evidence remains unchanged after
read-only review (~3.7 MiB; exact image hashes/dimensions aid normal-only checks).
If cost is unacceptable, STOP for a separately reviewed scope decision. This
plan neither drops categories nor reduces steps to fit an overnight window.

### 8. Required handoff before opening test

Provide the human reviewer with: access-history attestation; all scientific
fingerprints; all development checkpoint/threshold hashes; full matrix status;
hardware/memory proof; budget/retention decision; dispatcher/resume tests; clean
CI; failed-attempt history; and the exact proposed evaluation command and
authorization record. Until that review, the test lock stays CLOSED.
