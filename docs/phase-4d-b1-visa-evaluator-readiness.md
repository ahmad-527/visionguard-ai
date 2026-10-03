# Phase 4D-B1 — held-out VisA evaluator readiness

ENGINEERING ONLY — NO TEST ACCESS. B1 engineering implementation is frozen for
human review. This document does not authorize execution of Phase 4D-B2.

Protocol ID: `visionguard-heldout-visa-evaluator-v1`.
Evaluator implementation fingerprint:
`0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49`.
The canonical JSON fingerprint binds methodology, source bytes, tests, CI,
output schema, native environment and installed native implementation hashes.
It is distinct from the unchanged development execution fingerprint.

Base: reviewed Phase 4D-A regular merge
`ab6f47402e11ae55c835ba7323c681e995ef4082`. Frozen development implementation:
`cca5e47ec8c3f68adfa6c90e17a17770b48049d0`; execution fingerprint:
`31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3`.
Published 72-cell development freeze:
`d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061`.

All 72 development models and their frozen thresholds are immutable inputs.
B1 permits read-only development artifact verification and entirely synthetic
evaluation fixtures, not real test images, masks, labels, predictions or
performance artifacts. The final-test lock remains CLOSED. No training,
recalibration, model/threshold modification, routing optimization or seed
selection is permitted. A future B2 requires separate human authorization.

Human access history: UNKNOWN. Independent reservation: NOT ESTABLISHED.
Default future interpretation: **Held-out VisA evaluation with historical
access independence unverified**. No independent-confirmatory, production
readiness or proven-generalization claim is supported.

The [Phase 4D-A report](phase-4d-a-visa-development-matrix.md) and its immutable
recovery evidence retain two failed attempts, one interrupted attempt, approved
runtime copying/recoveries and unresolved file-lock/process-loss causes. B1
does not rewrite that history or reinterpret recovery success as a diagnosis.

## Verified immutable development inputs

The authoritative merge was checked as a regular two-parent merge, with the
reviewed development publication in its ancestry. The committed development
freeze's exact SHA-256 was reproduced. All 72 local completed cells passed
read-only CPU hash and semantic validation against that publication:
PatchCore 36/36 and EfficientAD 36/36. No trained model was instantiated during
the full inventory verification and no dataset root was visited.

Validation covered attempt origins/receipts, named artifact SHA bindings,
final/normalized/fit checkpoints, canonical tensor state hashes, exact
category/seed/implementation/environment identities, calibration inventory and
ordered normal score records, frozen image/pixel thresholds, and normalization.
PatchCore bank/coreset evidence and EfficientAD final 70,000-step optimizer,
scheduler, RNG and stream state were checked by the existing native validator.
The completed matrix hash was checked again after validation. Failed and
interrupted attempt history was not discarded. Unpublished failed-attempt
payloads remain forensic evidence, never eligible final-model inputs or an
alternative resume parent.

Underlying identities remain:

| Binding | SHA-256 / fingerprint |
| --- | --- |
| PatchCore VisA | `3ffcdc37cf3117d319da3e970383c6d0bdb52e2cd42dca87ad3161c5a9bc3191` |
| EfficientAD VisA | `78b27feeee044f560287a1ce45000800452344190e7b68003eefb3e2e853f1f9` |
| Dual-model triage | `94442ab3121bccd392e3805b6134710cc6e8c95e8f17b8eccda288f8b1bd672d` |
| Dataset audit | `06e227bb5d2cd26f38010b2c304c62f14f383a81c64f5b2e4c48f1019128f58f` |
| Development membership | `10e7a6c898fb18fbd1b93a115a5a94f3e2ebcecb7a1796d99d8c1e6f3567ce0b` |

The inventory receipt is
`reports/phase4d-b1-visa-evaluator-readiness/development-artifact-verification.json`,
SHA-256 `d47530a8ed69fd17e26edce01062337817dd03218c237db4ab506d5edd019554`.
It contains only compact declared model identities/statuses, not images or
model weights. The D-A freeze, all scientific source files named by its contract,
model files, calibrations, membership and recovery receipts were not modified.

## Read-only architecture and boundary

`visa_evaluator_artifacts` independently verifies canonical repository evidence;
`model_specs` admits exactly the 72 declared model/category/seed identities and
their own thresholds. It has no default model, fallback cell or best-seed path.
`visa_evaluator_backend` constructs only native frozen architectures and strictly
loads SHA-verified final state. Its environment/source guards refuse drift.
PatchCore construction disables pretrained downloading because the complete
verified backbone and dynamic bank are restored from final state; this does
not substitute weights. EfficientAD restores the teacher, student, autoencoder,
teacher mean/std and fitted quantiles, with no optimizer or calibration call.

One candle/42 checkpoint from each model was restored on CPU using the new
loader. Exact canonical state equality held before/after strict native loading;
eval mode and disabled gradients were checked. This was model restoration,
NOT inference. No pretrained asset was downloaded. Native library deprecation
warnings were preserved; preprocessing was not modernized or substituted.

Prediction interfaces accept explicitly synthetic, in-memory, label-free RGB
frames only. B1 tests use manufactured arrays and fake native backends, not
development model predictions. Metrics receive labels only after decisions.
There is no real VisA input loader or final-test execution command in B1.
The pure reducers and frozen native adapters are reusable by a future reviewed
data-admission layer; this deliberate closed boundary is not a claim that B2
has been implemented or authorized. A malicious caller who independently loads
real data and misrepresents it as synthetic is outside this trusted engineering
API; the phase restriction also applies to operators, not only type checks.

The public final-test API always rejects, including correct independently loaded
fingerprint/audit values, purported human authorization text and an authorization
environment variable. It does not resolve, stat, enumerate, hash or decode the
supplied test-root object. A later reviewed authorization layer must load its
expected identities independently; supplied identities are never their own
expected values. Missing repository evidence is a closed-gate error, not a bypass.

## Preregistered future procedure — not executed

Only after separate B2 authorization and review:

1. Verify the reviewed evaluator commit/fingerprint, canonical development freeze,
   exact native environment/source and all 72 model/calibration bindings before
   admitting inputs. Reserve an immutable attempt and record the authorization.
2. Admit the official held-out split through a separately reviewed access layer.
   Order each category by relative POSIX image path lexicographically. Preserve
   every official sample once per model/seed; seeds and models use identical
   identifiers, order, truth and original dimensions. No adaptive selection.
3. Execute categories in canonical order, seeds 42, 123, 2026, then PatchCore and
   EfficientAD. Keep one model workload on GPU at a time; unload between models.
   Use batch size 1, eval/no-grad and the exact original RGB/256 preprocessing
   and model-specific normalization. Do not restore training RNG/streams, train,
   fit normalization, recalibrate, optimize a router or replace a failed cell.
4. Preserve native image scores without rescaling. Restore each continuous map
   to original coordinates with bilinear interpolation, align_corners=False.
   Threshold that restored float32 map strictly above its own frozen pixel
   threshold BEFORE casting the continuous export to float16. Never derive a
   binary mask by re-thresholding the rounded export. Reject nonfinite values,
   float16 overflow, invalid truth or shape mismatch; never resize ground truth.
5. Stream saved paired evidence into the frozen reducers; join by exact
   category/seed/ordered membership. Preserve both model explanations separately.
   No heatmap/mask fusion, score averaging, morphology or threshold rescue.
6. Recompute reductions twice from the same saved rows, continuous maps and
   preserved binary maps. Require exact canonical JSON identity, not a tolerance
   invented after results. Do not rerun inference just to obtain nicer results.
7. Publish completion only when every required cell and artifact validates.
   Any missing/corrupt/mismatched input or failed cell stops the attempt; keep
   partial evidence explicitly incomplete. No silently dropped samples, partial
   overall claims, automatic hardware retries or threshold/category exceptions.

## Metric definitions and aggregation

Ranking metrics are image AUROC and AU-PRO@0.05. AUROC uses Mann–Whitney
positive/negative pairs with half credit for ties. AU-PRO uses 8-connected truth
regions, equal weighting per region, all background pixels for FPR, exact
float16 score tie groups, interpolation at FPR 0.05 and normalized trapezoidal
area. Supplemental pixel AUROC is a diagnostic, not a primary selection metric.
These are reused VisionGuard definitions, not optimized thresholds.

Threshold-dependent metrics preserve image and pixel TP/FP/TN/FN separately:
recall = TP/(TP+FN), specificity = TN/(TN+FP), precision = TP/(TP+FP),
F1 = 2TP/(2TP+FP+FN). Each model uses its own frozen image/pixel thresholds.
Equality is normal/background. Image score distributions by true class retain
count, min/max, mean, population standard deviation and linear-index n−1
quartiles on each model's own scale. No cross-model score normalization.

| PatchCore image flag | EfficientAD image flag | Triage |
| --- | --- | --- |
| normal | normal | PASS |
| normal | anomalous | REVIEW |
| anomalous | normal | REVIEW |
| anomalous | anomalous | REJECT |

Anomalous means `score > own_frozen_image_threshold`. REVIEW cannot be coerced
to binary truth/correctness or discarded. Predeclared rates, with numerators and
denominators retained, are:

| Rate | Numerator / denominator |
| --- | --- |
| Review rate | REVIEW / all images |
| Automatic coverage | (PASS+REJECT) / all images |
| Selective accuracy | (normal PASS + anomalous REJECT) / (PASS+REJECT) |
| Anomaly pass-through | anomalous PASS / anomalous images |
| Normal reject rate | normal REJECT / normal images |
| Anomaly review capture | anomalous REVIEW / anomalous images |
| Normal review rate | normal REVIEW / normal images |
| Reject precision | anomalous REJECT / REJECT |
| Pass NPV | normal PASS / PASS |

Report PASS/REVIEW/REJECT counts by class and both disagreement directions,
including their class-specific counts. Binary baselines retain their individual
FN/FP rates; triage safety and review workload must be interpreted together,
not as a weighted scalar or an ordinary binary F1.

Primary aggregation: compute each category/seed metric first, then unweighted
12-category means within each seed, then the three seed means and sample
standard deviations (ddof=1). Preserve all cell results and all seeds. Paired
PatchCore-minus-EfficientAD differences are computed within the same
category/seed/ordered membership BEFORE category and seed aggregation.
Secondary within-seed micro rates sum confusion or triage numerators/denominators
across categories. Never pool ranking scores across categories/models/seeds.
Triage macro rates and micro rates are both labeled, including their different
denominator interpretation.

Zero denominators and absent metric classes/regions produce null plus an explicit
reason, never zero. If any category is undefined, its macro and paired macro
are undefined; report defined-group counts, without quietly omitting categories.
Across-seed summaries follow the same rule. Repeated images across three seeds
are not independent observations. No p-values, significance claims or confidence
intervals are justified by this protocol's three descriptive seed replicates.
Deployment needs an external operational risk/cost preference; no arbitrary
acceptance target or hybrid superiority claim is frozen here.

## Output schemas, retention and resource gates

The machine-readable protocol is `configs/protocols/visa-heldout-evaluator-v1.json`.
The per-cell synthetic schema is `configs/schemas/visa-evaluator-report-v1.json`.
Any real-evidence admission/schema label extension requires separate B2 review;
synthetic evidence must never be relabeled as held-out performance.

Per-cell JSON has schema_version, evidence_class, category, integer seed,
sample_count, ordered-ID membership SHA-256, both model hashes/thresholds,
image/pixel metrics/distributions, triage count/rate trees and ordered image rows.
Rows have identifier, integer label, original [height,width], both native scores,
boolean model flags and PASS/REVIEW/REJECT. Metric objects retain value/status/
reason; triage rates retain integer numerator/denominator, value and undefined
reason. Aggregates retain each seed's category macro, micro counts/rates, paired
deltas and separately labeled descriptive three-seed summaries.

An immutable artifact manifest joins rows by index/identifier to continuous map,
binary map and truth hash bindings. The synthetic writer validates before writes,
refuses an existing attempt and atomically publishes its completion manifest last.
Replay checks all hashes, model/seed/threshold identity and ordered membership,
preserving binary maps rather than applying thresholds again. A future real run
must additionally record reviewed code/environment/hardware, authorization
receipt, official input membership, development freeze and stage/attempt history.

Keep all full-resolution float16 TIFF maps, binary PNGs, image rows and hash
inventories in ignored/local storage. Retain all samples and both models, not
visually chosen examples. Do not commit images, datasets, checkpoints or raw
private/machine-path records. Git receives compact schemas, summaries/counts,
hash bindings and provenance only. Original input datasets/models are reused,
not recopied; no compression savings or mid-run deletion is assumed.

Calculated scenario from previously committed aggregate audit counts only:
2,162 images × 2 models × 3 seeds = 12,972 model-image calls. Actual test
dimensions and latency were NOT inspected or measured. Assuming every image
fits within 1536×1536, continuous+binary map payloads cost 91,814,363,136 bytes;
64 KiB per map/binary pair and 1 MiB per model-cell add 925,630,464 bytes.
Total scenario retention = 92,739,993,600 bytes (about 86.37 GiB).

The frozen capacity gate requires 128 GiB output budget + 20 GiB reserve =
158,913,789,952 free bytes (148 GiB). Observed local free space at evidence
generation was 116,582,596,608 bytes, BELOW that gate. **B2 storage readiness
is unresolved**; do not delete evidence or silently reduce retention to proceed.
The 1536 scenario is not an observed maximum. Future authorized preflight must
check actual dimensions/estimates against this budget before scoring; oversize
or insufficient capacity stops execution for review.

Streaming reducer persistent pixel histograms are 4 MiB for a pair, plus one
original-resolution pair, connected-component temporaries and small score rows.
Native model allocations and transient library workspace are additional. The
existing 8 GiB GPU is known from development evidence, not an inference memory
acceptance test. End-to-end latency and peak inference VRAM/host RAM remain
unmeasured. No runtime or throughput claim is supported by B1.

## Verification and negative engineering history

Local full pytest: **581 passed, 1 skipped**. The skip is the existing Windows
symlink privilege limitation, not a benchmark omission. Ruff format/lint and
pip check passed. Tests cover all truth-table outcomes/equality, NaN/Inf,
ties, hand-calculated confusion/F1/AUROC/PRO and independent PRO reference,
missing/reordered/duplicate images, wrong seed/category/model/threshold,
invalid masks/dimensions, corruption/membership, absent classes/all REVIEW,
determinism, macro/micro pairing, repeatable map replay, immutable attempts,
resource refusal, fingerprint mutations and permanently closed authorization.

The first full run had one legacy D-A contract-test failure (580 passed,
1 failed, 1 skipped): its live source-inventory guard correctly refused later
B1 additions. The execution guard/source/science stayed unchanged. The test
now verifies every original bound file and the historical execution fingerprint,
and explicitly expects STOP MATRIX for post-freeze additions. The preliminary
B1 candidate fingerprint `fa56a2cebbd91884919bdd6b269993fa0963ac5c03654a551cfd5f1929d90466`
was superseded only for that test/freeze-builder binding change; its bytes are
preserved in ignored engineering-candidate storage, SHA-256
`2eae1b131b2067711b86bdd70e51b9daafd34c9ce3f6d9f38665fd60ebfbb3f4`.
Scientific methodology and synthetic numerical answers did not change.

CI verifies the frozen B1 fingerprint and synthetic CLI on Python 3.11/3.12/3.13,
without ML installation or real datasets. Native CPU restoration and full local
artifact checks are separate recorded evidence, not claimed as Linux ML tests.
CI completion/status is recorded on the draft PR's exact pushed head.
Changed-file known-secret-pattern and local absolute-path scans had no matches;
no new file exceeds 1 MiB and staged whitespace checks passed. These scoped
hygiene scans do not claim to be an exhaustive security audit.

Reproduce the safe engineering checks from the reviewed repository:

```powershell
python scripts/phase4d_b1_freeze.py
python -m visionguard.visa_evaluator_synthetic --repository .
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m pip check
```

`visa_evaluator_artifacts --development-artifacts` is an explicit, potentially
expensive read-only local development validation command, not a test evaluator.
Do not substitute a dataset directory. Native backends require the exact bound
ML environment; the synthetic/core/gate checks require only analysis/dev extras.

## Handoff and unresolved limits

B1 engineering evidence does not establish test performance, independence,
hybrid validation, production readiness or generalization. Historical access is
UNKNOWN and independent reservation NOT ESTABLISHED, regardless of verified
chronology/integrity/zero overlap. Unless credible historical independence is
later established, every future report must visibly retain the default limited
held-out title and this uncertainty.

Remaining B2 gates: explicit human authorization; reviewed real-input access/
attempt manifest integration without changing frozen metric/model/triage rules;
adequate storage and authorized resource preflight; reproducibility on the exact
native environment. Stop on scientific drift, corrupt/missing model evidence,
membership mismatch, failed authorization, unexpected test access or hardware
instability. Do not modify science or rescue failures from test outcomes.

All 72 development models are unchanged. Final-test lock CLOSED. Zero real test
performance evaluations. Phase 4D-B2 NOT STARTED. The issue/PR remains a review
handoff, not authorization to execute; the PR is not merged.
