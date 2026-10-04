# Phase 4D-B2 — final evaluation readiness

Status: **ENGINEERING REVIEW HANDOFF — REAL TEST LOCK CLOSED.**

[Issue #23](https://github.com/ahmad-527/visionguard-ai/issues/23) and
[PR #24](https://github.com/ahmad-527/visionguard-ai/pull/24).
This is synthetic integration/readiness, not held-out evaluation. No VisA test
image, mask, label, prediction, or performance artifact was enumerated, statted,
hashed, decoded, read, or scored. No training or calibration was performed.

Human historical access is **UNKNOWN**; independent reservation is **NOT
ESTABLISHED**. The eventual report must be titled **“Held-out VisA evaluation
with historical access independence unverified.”** Chronology, hashes and zero
overlap do not establish historical independence, confirmatory validity,
generalization, hybrid superiority, or production readiness.

## Verified frozen inputs

| Identity | Verified anchor |
| --- | --- |
| Reviewed B1 regular merge | `d335b91d10489b4c0811530bd4d69418f68fb2ff` |
| Phase 4D-A regular merge | `ab6f47402e11ae55c835ba7323c681e995ef4082` |
| Frozen development implementation | `cca5e47ec8c3f68adfa6c90e17a17770b48049d0` |
| Development execution fingerprint | `31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3` |
| Published 72-cell freeze SHA-256 | `d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061` |
| Unchanged B1 evaluator fingerprint | `0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49` |
| Dataset audit SHA-256 | `06e227bb5d2cd26f38010b2c304c62f14f383a81c64f5b2e4c48f1019128f58f` |
| Official split SHA-256 | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |

The canonical B2 snapshot and fingerprint are in
`reports/phase4d-b2-readiness/implementation-freeze.json`; reproduce with
`python scripts/phase4d_b2_freeze.py`. It binds the new source, tests, workflow,
authorization schema, complete readiness contract and B1 fingerprint. B1 source,
protocol, metric definitions, backend and permanently closed gate were not
changed. An unrelated new workflow provides B2-specific CI without changing
the B1-bound existing workflow.

The authoritative workspace/output volume is the migrated external NTFS drive.
Its 67-receipt migration index was independently SHA/size verified against
`80b786e5eaef37e4a56c33f7fc43c65dd8713830f0f848a5e1c6e7c31fbedac7`.
Native read-only artifact verification passed for all 72 complete models,
checkpoint bindings, normalization/calibration state, model/category/seed
membership, environment and immutable attempt history. The measured validation
took 375.75 seconds; it was not inference. Compact per-model hashes and retained
attempt statuses are in `development-artifact-verification.json`, bound to its
local raw receipt. Models and thresholds remain unchanged; originals on the
system disk were neither deleted nor reorganized.

The exact native environment is Python 3.11.6, Torch 2.9.1+cu126,
torchvision 0.24.1+cu126, Anomalib 2.6.0 and timm 1.0.28. Core CI does not
install/download model weights or restore models; Windows GPU smoke is separate
local evidence. The rebuilt virtual environment still relies on the installed
base Python on the system disk, so this is not a portable environment for a
different PC.

## Architecture and pre-access boundary

The separate `visa_b2_*` modules integrate:

- Independent repository-bound 72-model identities and committed audit counts.
- Capability-scoped artificial directory creation and admission, reusing the
  official split parser on an artificial CSV with its own pinned SHA.
- Read-only native B1 backend restoration/inference on synthetic frames.
- Unchanged B1 streamed metrics and 36-pair aggregation.
- Exclusive immutable output publication, SHA verification and restart.
- A separate draft authorization-intent validator; all real entry points deny.

Only trees newly manufactured by `make_fixture` can enter the readiness loader.
No caller can supply an existing root and merely label it synthetic. Admission
checks source-file inventory/SHA, CSV/category/split membership, missing/extra/
duplicate files, within-category train/test overlap, RGB mode, mask values,
original dimensions and label/mask consistency. Reparse/cloud paths are refused
before content reads. Fixture RGB templates deliberately repeat across
categories; this is not evidence of real dataset independence.

Future real admission is **design only** in this branch, not an executable
loader for existing VisA directories. It must use the trusted precommitted
archive/split/audit identities and exact per-category membership, preserve the
same images across both models and all three seeds, reject aliases/duplicates,
and validate image/mask/label/dimension relationships only **after** separate
human authorization. It must never disguise real IDs as synthetic IDs to bypass
the B1 reducer/backend guards. The first real admission and reviewed real-ID
adapter/activation are reserved for the separately authorized execution stage.

## Exact future scope and order

Committed aggregate audit metadata, not test-file inspection, establishes
2,162 distinct images (962 normal, 1,200 anomalous). Six model/seed passes require
**12,972 model-image calls**, 72 frozen models and 36 paired category/seed cells.
Canonical category order is `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`,
`macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, `pipe_fryum`;
then seeds 42, 123, 2026. Pair only identical category and seed. Within a cell,
use POSIX relative-image-path lexicographic order and retain identifiers/order.
No best seed, category dropping, missing-model substitution, threshold rescue,
score normalization, fusion, or learned routing is allowed.

Only one native GPU model is resident at a time. A future paired cell can retain
one architecture's outputs, unload it, process the other in exactly the same
order and independently bind both to the frozen model specifications. The
current known-answer matrix deliberately uses synthetic constant model hashes,
thresholds and scripted predictions, not 72 native model loads. Native smoke
separately establishes both real backend paths through the same publisher and
reducers; neither evidence class is a benchmark result.

## Metrics and reporting remain B1-frozen

Ranking metrics: image AUROC (ties handled by frozen pairwise ranking),
AU-PRO@0.05 and supplemental pixel AUROC. Pixel ranking uses retained float16
continuous maps; native float32 restored maps are thresholded **before** export,
and the derived binary maps are retained separately. No post-export threshold
retuning is allowed.

Threshold-dependent metrics: image and pixel confusion counts, recall,
specificity, precision and F1, using each model's own frozen threshold and strict
`score > threshold`. Equality is normal. Keep image/pixel metrics separate,
score distributions and both explanations; do not fuse maps or masks.

| PatchCore | EfficientAD | Triage |
| --- | --- | --- |
| Normal | Normal | PASS |
| Anomalous | Anomalous | REJECT |
| Normal | Anomalous | REVIEW |
| Anomalous | Normal | REVIEW |

REVIEW is neither a negative prediction nor a discarded observation. Report
review rate, automatic coverage, selective accuracy, anomaly pass-through,
normal rejection, anomaly review capture, normal review rate, reject precision,
pass NPV, three-way counts, both disagreement directions and class-conditioned
counts. Keep numerators/denominators and explicit undefined reasons.

Primary aggregation is equal-category macro within each seed, followed by
descriptive mean and sample standard deviation across the three seeds. Preserve
paired PatchCore-minus-EfficientAD category/seed differences. Pooled counts and
class rates are secondary; do not pool scores from incompatible model scales.
Three seeds repeat the same images and do not justify independent-trial,
significance, causal or pristine-holdout claims. Undefined category values are
not silently dropped from an otherwise complete macro summary.

## Synthetic integration and restart evidence

Artificial known-answer data has four 2×2 images per category; no manufactured
benchmark claims. Every pair yields 1 PASS, 2 REVIEW, 1 REJECT. Hand-calculated
PatchCore image counts are TP=2, FP=0, TN=2, FN=0; EfficientAD counts are 1 each.
Image AUROCs are 1 and 0.75; EfficientAD pixel F1 is 2/3 and pixel AUROC 0.75.
Review and automatic coverage are each 0.5, selective accuracy 1, and anomaly
pass-through 0. These are synthetic known answers only.

The entire 36-pair orchestration validates 144 paired image records / 288
scripted prediction records. Each successful attempt writes an origin, separate
truth PNG, two float16 TIFFs, two binary PNGs, scores/dimensions/hash records,
result and completion candidate. The origin is itself SHA-bound. A completion
candidate is **not operational completion** until independently rehashed and
metric-replayed with an immutable `validated.json` receipt. Overall output is
withheld until all 36 cells validate; no partial matrix is called complete.

Restart validates every retained asset and exact canonical metrics before
skipping completed cells. A declared interrupted attempt gets a new immutable
attempt with its parent-failure SHA; the failed/interrupted directory remains.
A new-process test reconstructs identical artificial inputs and skips the
already validated cell without recalculating predictions. A stale exclusive
writer lock, unclassified death, corruption, missing validation receipt,
hardware failure or publication failure stops for human review. There is no
automatic stale-lock deletion, guessed exit code, overwrite or silent retry.
Secondary failure-receipt errors cannot hide the original exception.

Tests cover missing/reordered/duplicate inputs, invalid masks/dimensions,
threshold equality, nonfinite predictions, wrong model/seed/origin, corrupt
maps/metadata, multiple writers, process restart, partial post-publication
validation and injected Windows-style permission failure.

## Native synthetic GPU smoke and HDD observations

Selection was frozen from development metadata: largest fitting membership,
then lexicographic category and seed 42, not test behavior. This selected
PatchCore `pcb3:42` (815 fitting normals; memory bank 8,345×1,536) and the matching
EfficientAD model. Two manufactured 384×512 RGB frames per model made exactly
four native inference calls, with original-coordinate maps restored to 384×512.

| Measured operation | PatchCore | EfficientAD |
| --- | ---: | ---: |
| Restoration including independent validation (seconds) | 107.026 | 4.112 |
| First synthetic inference (seconds) | 0.524317 | 0.118083 |
| Second synthetic inference (seconds) | 0.019314 | 0.056612 |
| Peak CUDA allocated (bytes) | 223,997,952 | 115,962,368 |
| Peak CUDA reserved (bytes) | 243,269,632 | 153,092,096 |

Measured process peak working set reached 1,981,050,880 bytes; this is not total
host-RAM attribution. Device point readings reached 1,679 MiB and 46°C, **not a
continuously measured device-wide peak**. CUDA allocator peaks above are measured
peaks. Both native model states were bitwise unchanged after inference. No new
hardware/storage/GPU warning was found in the bounded smoke interval. Upstream
deprecation warnings were preserved without changing frozen transforms.

Publication plus two reducer replays took 1.310 seconds in the initial bounded
smoke. After origin-binding hardening, already-retained synthetic maps and native
binary maps were independently rehashed and republished in a new directory;
exact metric identity passed with **no new inference**. Original smoke bytes and
receipt remain preserved, and compact evidence explicitly distinguishes this
post-smoke republication from native execution.

The previously verified external-drive synthetic benchmark measured a 512 MiB
unbuffered/write-through pass at 97.853 MiB/s write and 66.630 MiB/s read;
100/100 small atomic publications validated. It is SHA-bound in the compact
resource report, not repeated or represented as a current long-run reliability
test. This task additionally used exclusive-create/fsync/readback publication
on that drive. No recurrent WinError 5 occurred. Small bounded tests cannot
establish sustained reliability under hours of USB/HDD load.

SMART remained unavailable without additional privileges. Preserve stable USB,
AC power and an open lid; prior observations showed enabled USB suspend,
disk-idle timers and lid sleep. No Windows/OneDrive/Defender settings were changed.
The historical `nvlddmkm` event and process disappearance have unresolved causes;
this bounded success does not explain them. Stop, preserve state and seek review
on CUDA/OOM, new hardware events, disconnects, corruption or failed publication.

## Capacity, retention and runtime uncertainty

Measured free capacity was about 2,367 GiB on the actual external output volume,
well above 148 GiB (128 GiB run budget plus 20 GiB reserve). B1's unchanged
1,536×1,536 capacity scenario estimates 92,739,993,600 bytes retained, including
allowance. **Actual test dimensions have not been inspected.** Its old system-
disk observation is historical, not current D-volume capacity.

Before authorized scoring, verify actual admitted dimensions and deterministic
payload budget; reserve allowance includes failed attempts. Enforce capacity
before every future publication, retain the 20 GiB floor, and stop rather than
drop outputs if the 128 GiB budget would be exceeded. Keep large data/maps/models,
checkpoints and raw machine-specific logs in ignored local storage. Git retains
only compact canonical manifests, per-asset SHA/size bindings, definitions and
sanitized engineering evidence. No valuable old recovery copy is disposable
merely because an attempt failed.

For planning only, scaling the two measured synthetic latencies to 6,486 calls
per architecture yields approximately 8.21 minutes warm or 69.44 minutes if
every call resembled the first cold call. Applying the measured pair restoration
to all 36 pairs adds approximately 66.68 minutes. Thus those **explicit synthetic
extrapolations**, before output I/O and metrics, are about 75–136 minutes; they
are neither bounds nor measured real runtime. Full-resolution metric computation,
connected regions, multiple SHA/replay passes, filesystem overhead, fragmentation
and USB interruptions remain unmeasured. A defensible real end-to-end upper bound
is unknown. Do not promise completion in that scenario interval.

## Authorization and remaining execution gates

`expected_intent` independently loads canonical repository identities, actual
checkout HEAD, exact model matrix, audit, contract and derived output destination.
The schema is explicitly **DRAFT_NOT_AUTHORIZED**. Incorrect/missing/malformed/
reused intent, altered commit/hash/model matrix/output or inadequate capacity is
rejected. A matching SHA, caller flag, free-space assertion, human-looking text
or successful draft validation is **never permission**. Tests use a root object
that raises on any operation and demonstrate denial before any root access.

The future gate must authenticate a separately recorded human approval from a
trusted out-of-band ledger, compare its reviewed commit against actual HEAD and
source fingerprints, verify the full frozen scope/output binding and current
capacity, and exclusively claim a single-use authorization ID before admission.
A resume must stay in the same run/receipt chain, not reuse approval for a new
run. This branch does not create/consume an approval or implement an opening path.
The permanently closed B1 gate remains unchanged.

Remaining gates are **human scientific review**, a separately recorded real-
execution authorization and reviewed activation/real-ID admission adapter, actual
post-authorization dimension/capacity checks, and operational USB/power safeguards.
They must not be inferred from successful synthetic tests. No arbitrary tolerance
or test-dependent engineering/model rescue is authorized.

Both failed cashew attempts, the interrupted pcb1 attempt, approved migrations,
checkpoint/recovery receipts and unknown exit/cause history remain untouched.
All 72 development models are unchanged. Real VisA performance evaluations: zero.
Final-test lock CLOSED. Held-out execution NOT STARTED. PR NOT MERGED.

## Reproduction and review checks

```text
python scripts/phase4d_b1_freeze.py
python scripts/phase4d_b2_freeze.py
python -m visionguard.visa_b2_runner --plan
python -m visionguard.visa_b2_runner --synthetic --output <new-ignored-output>
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m pip check
```

`--synthetic` manufactures artificial inputs; the default runner rejects real
execution. Native smoke is an explicit bounded engineering command requiring
the exact native environment and ≥148 GiB, never a test-directory command.
Local check results and per-version CI observations are reported separately in
compact verification evidence and PR #24; neither substitutes for human approval.
