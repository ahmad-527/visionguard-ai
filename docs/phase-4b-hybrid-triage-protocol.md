# Phase 4B — dual-model triage and independent-evaluation protocol

Status: **frozen for scientific review; no hybrid performance evaluation**.
Issue: [#15](https://github.com/ahmad-527/visionguard-ai/issues/15).
Protocol ID: `visionguard-dual-model-triage-v1`, version 1.
Canonical document: [dual-model-triage-v1.yaml](../configs/protocols/dual-model-triage-v1.yaml).
SHA-256 fingerprint:
`94442ab3121bccd392e3805b6134710cc6e8c95e8f17b8eccda288f8b1bd672d`.

## Scientific boundary: development evidence is not independent evidence

**MVTec AD 2 `test_public` was fully inspected during Phases 2C, 3B, and 4A.
It cannot now independently validate a hybrid or routing rule designed after
Phase 4A.** Freezing a new rule does not restore that split's independence.
Public labels/results must not select, optimize, weight, calibrate, threshold,
or category-route this policy and then be reused as confirmatory evidence.

Any later authorized replay must carry the exact label **"retrospective
development analysis"** in its report, tables, and machine-readable metadata.
It may verify implementation and describe development review workload. It
cannot support generalization, model-selection, or production-performance
claims. No such replay is executed in Phase 4B.

[Phase 4A](phase-4a-comparative-failure-analysis.md) established complementary
behavior of frozen models and preserved substantial failures. It did not
validate a router, hybrid, fused localization map, or deployment policy. The
present hypothesis handles operational uncertainty: disagreement warrants
review. It is not an empirical claim that agreement is reliable or that a
reviewer can resolve every disagreement.

## Exact image decision rule

For each model independently, `anomalous = score > threshold`. Equality is
normal. Inputs are the frozen image anomaly score and that model's own frozen
validation-normal image threshold on the same score scale.

| PatchCore | EfficientAD | Triage |
|---|---|---|
| NORMAL | NORMAL | PASS |
| ANOMALOUS | ANOMALOUS | REJECT |
| NORMAL | ANOMALOUS | REVIEW |
| ANOMALOUS | NORMAL | REVIEW |

REVIEW is a first-class abstention, not a binary error or success. No automatic
override, default PASS for missing input, assumed perfect reviewer, or implicit
REVIEW-to-binary conversion is permitted. PASS is an operational assignment,
not a guarantee that the object is safe. Shared blind spots can produce PASS
on an anomaly or REJECT on a normal image.

There is no cross-model normalization, weighted average, logistic regression,
learned meta-model, threshold optimization, category route, best-seed choice,
or override based on Phase 4A. EfficientAD's existing *internal* normalization
remains part of its frozen upstream model protocol; the triage module does not
modify it. Existing per-run thresholds are supplied unchanged, including where
the upstream model was trained for a category. This does not introduce a new
category-specific triage threshold or a category-dependent branch.

The [decision module](../src/visionguard/triage.py) takes four explicit scalar
arguments and preserves both anomalous flags with a `Decision` enum. It accepts
finite built-in integers/floats, including negative values; strings, booleans,
arrays, missing values, NaN, and infinities fail. No coercion or lossy integer
conversion occurs. Enum/result truthiness raises an error so consumers must
compare explicit outcomes. The module has no I/O, model imports, label input,
category parameter, calibration logic, or dependency on Phase 4A records.

The scalar primitive cannot certify a caller's provenance. A future runner must
verify dataset identity, split, sample ID, category, seed, artifact hashes,
threshold provenance, and score/threshold scale before calling it. It must stop
on missing, duplicated, or mismatched records rather than aligning by position
alone. `triage_same_seed` requires both seeds and rejects any mismatch; allowed
seeds are **42, 123, 2026**. All three must be retained and reported. No cross-seed
pairing, majority vote across seeds, or best-seed selection is defined.

## Localization remains two separate explanations

For REVIEW and REJECT, preserve the PatchCore continuous map and frozen binary
map, and the EfficientAD continuous map and frozen binary map, with their
identifiers and hashes. A future UI may show them side by side. Phase 4B creates
no UI or map processor. Existing PASS evidence is not deleted.

No union, intersection, average, weighted heatmap, category-based map selection,
morphology, or threshold adjustment is authorized. Phase 4A's different failure
modes do not independently justify fusion. Image triage does not imply that
either localization is accurate. A localization fusion hypothesis would require
a separately reviewed protocol and new independent evidence.

## Frozen future metrics

Use a two-class by three-decision count table. Let `N` be all samples, `A`
anomalous samples, `G` normal samples; `P`, `V`, `R` denote PASS, REVIEW, REJECT.
A class suffix gives the corresponding count (for example, `P_A`). The metric
implementation is a separate [pure reducer](../src/visionguard/triage_metrics.py),
tested on clearly synthetic examples only. There is no dataset evaluation runner.

| Metric | Numerator / denominator | Interpretation |
|---|---|---|
| Review rate | `V / N` | Workload assigned to review |
| Automatic-decision coverage | `(P + R) / N` | Fraction receiving an automatic decision |
| Selective accuracy | `(P_G + R_A) / (P + R)` | Correctness conditional on automatic coverage |
| Anomaly pass-through rate | `P_A / A` | Key safety metric: anomalies assigned PASS |
| Normal reject rate | `R_G / G` | Normals automatically rejected |
| Anomaly review capture | `V_A / A` | Anomalies deferred to review, not proven resolved |
| Normal review rate | `V_G / G` | Normal-image review burden |
| Reject precision | `R_A / R` | Fraction of rejects that are anomalous |
| Pass negative predictive value | `P_G / P` | Fraction of passes that are normal |
| Decision counts | `P`, `V`, `R` | Preserve total and true-class counts |

Preserve both disagreement directions separately, overall and by true class:
PatchCore anomalous/EfficientAD normal and PatchCore normal/EfficientAD anomalous.
Their sum must equal REVIEW. Preserve every rate's integer numerator and
denominator. A zero denominator means `null` with reason `zero_denominator`,
never zero, one, NaN, or a silently dropped result. For empty groups counts are
zero and all rates undefined. If every sample is reviewed, coverage is zero and
selective accuracy is undefined, not perfect. No binary F1 is defined for triage.

Future reporting is fixed as follows:

1. Publish every category × seed cell, the true-class count table, both
   disagreement directions, and all nine rates. Include class support.
2. For each seed, publish a micro aggregate by pooling category counts before
   computing rates, and a macro aggregate as the unweighted mean of category
   rates. A macro rate is undefined if any required category rate is undefined;
   preserve the missing/undefined category list.
3. For each category and for each aggregate type, publish all three seed values,
   their arithmetic mean and sample standard deviation (`ddof=1`). If a required
   seed is missing/undefined, mark the summary undefined and name it. Do not pool
   repeated seed observations as if they were independent physical samples.
4. Publish both individual model policies on exactly the same samples and seed
   at their own frozen thresholds, with paired differences in relevant risk
   rates. No seed/category selection is allowed. Comparators have full automatic
   coverage; explicitly identify this difference from triage.
5. Report precision/NPV with observed prevalence and sample counts. A benchmark's
   class mixture is not deployment prevalence. Review resolution accuracy,
   review time, staffing capacity, and downstream outcomes remain unmeasured.

The reducer computes one caller-supplied group. This phase does not implement
the future grouping/label-joining runner. Its audited implementation and checks
for the complete category/seed matrix must precede any authorized evaluation.
Seed SD describes training variability, not an independent-sample confidence
interval. Any future inferential uncertainty procedure must be prespecified
before test outcomes are exposed; no significance claims are defined here.

## Success and failure interpretation

Evidence must characterize the joint trade-off among anomaly pass-through,
normal rejection, review workload, automatic coverage, and selective accuracy.
No real operational cost model or externally justified acceptance bound is
available. Consequently there is no weighted score, arbitrary numeric target,
or single-metric winner criterion. Deployment requires explicit risk and cost
preferences and measured review capacity/outcomes.

On the same paired inputs, PASS is the intersection of the two normal-decision
sets. Its anomaly pass-through count therefore cannot exceed either individual
policy's. Likewise, REJECT is a subset of each model's anomalous-decision set,
so normal rejection cannot increase. These are **structural subset properties**,
not measured improvements or independent validation. Strict reduction is not
guaranteed, and the cost is abstention. Selective accuracy, precision, and NPV
need not improve. Nearly universal review would satisfy low automatic error
counts while providing little operational value.

A useful future finding could combine lower pass-through with coverage that a
real operator finds useful. Failure or inconclusive outcomes include high
review workload, shared misses, poor conditional accuracy, unstable seed or
category behavior, insufficient class support, or inability to preserve test
independence. Preserve all outcomes and refrain from category rescue or tuning.

## Evidence anchors and verification scope

The working branch began at current reviewed main
`bc71bb3290e61417a7c461f369d2d520fb3ce217`, confirmed against origin and the merge
metadata of [PR #14](https://github.com/ahmad-527/visionguard-ai/pull/14).

| Model | Protocol | Fingerprint | Benchmark implementation |
|---|---|---|---|
| PatchCore | `patchcore-mvtecad2-v1` | `03f545ea23b1bd00206cb919aece6972502712aa9f981e8a3f11dbd1be1f0c2b` | `8848e8defb1f734a319168fd597b4252b606fff7` |
| EfficientAD | `efficientad-mvtecad2-v1` | `e9d6a66e7a52f2993e984ec20278c4ca4c710198cc466df15f947adff763f69f` | `9e477389530743f8a7cf4caa8c48214e5c63ec28` |

Common audit SHA-256:
`8c0f71f0a7dc81436b7bd3affed0ba7f97ea3844213d487c2d9886befa055a92`.
Both implementation Git objects exist. The existing protocol loaders recomputed
the fingerprints and accepted the frozen documents. Both committed benchmark
manifests match these IDs, implementation SHAs, audit identity, and the complete
8-category × 3-seed matrix. Their hashes also agree with Phase 4A provenance.
The new verifier hashes the reviewed Phase 4A summary and checks its provenance
and manifest bindings. These are repository evidence checks; Phase 4B does not
reopen datasets, maps, predictions, or model checkpoints or repeat Phase 4A's
artifact validation. Original frozen protocols and reports remain unchanged.

Run the read-only verification from the repository root:

```bash
python -m visionguard.triage_protocol --repository .
```

The triage fingerprint covers the **entire** YAML document after conversion to
UTF-8 JSON using ASCII escaping, sorted mapping keys, compact separators, and
no nonfinite values. Mapping order, comments, and whitespace do not matter;
list order does. Duplicate YAML keys, unsupported values, missing fields,
unknown additions, and content drift fail closed. The expected digest is pinned
in code and independently calculated in tests. Unlike the older model protocol
hashes, this digest also includes `schema_version`; those older protocols are
not changed. SHA-256 detects drift, not authorship or approval: changing the
digest alongside the protocol requires scientific review and a new identity
for a materially changed policy. Never silently redefine v1.

## Future evaluation routes

### A. Retrospective development replay

After separate authorization, an implementation could pair the existing frozen
public image scores and thresholds by the same seed and sample identity, apply
the unchanged truth table without new model inference, and calculate the frozen
metrics. It must use the label **retrospective development analysis**, preserve
all cells, and make no independent-validation or selection claim. Phase 4A
reports must not be relabeled as hybrid validation. Phase 4B performs no replay,
including no algebraic reconstruction of public triage performance from earlier
counts.

### B. Independent confirmatory route: VisA

The existing [dataset strategy](dataset-strategy.md) names VisA as the secondary
robustness benchmark. Official documentation describes 12 object subsets and a
one-class setup with normal-only training data, mixed normal/anomalous test
data, and anomaly masks. It distributes the `VisA_20220922.tar` release under
CC BY 4.0. These are source facts, not newly audited local dataset claims.
Sources: [official README at the inspected commit](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/README.md),
[AWS dataset registry](https://registry.opendata.aws/visa/).

The official preparation code defaults to `split_csv/1cls.csv`; its one-class
layout has training normals and separate test/ground-truth directories. It
converts nonzero anomaly-mask values to binary foreground. It supplies no
separate validation directory. The two-class alternatives expose anomalous
training information and are outside this plan.
Source: [official preparation code](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/utils/prepare_data.py).

**Assessment: conditionally feasible.** Licensing and documented structure do
not rule out clean normal-only training/calibration followed by a single
confirmatory evaluation. This is a design conclusion from documentation, not
proof that local VisA data are untouched. No VisA images, masks, split rows,
test predictions, or test outcomes were inspected in Phase 4B. Dataset access
history, exact bytes, duplicate audit, sample sufficiency, and test reservation
remain prerequisites. If prior access prevents a credible reservation, stop;
do not retrospectively declare VisA independent or substitute a convenient split.

A later separately authorized protocol must satisfy these gates in order:

1. Pin the official one-class split and release checksums and license records.
   A data custodian establishes prior access history and keeps test images,
   labels, masks, outcomes, and plots inaccessible to development. Original
   directory names/annotation files can reveal labels, so merely hiding a
   column is insufficient. Audit duplicate and cross-split relationships without
   revealing test outcomes to developers. Any independence concern stops work.
2. Before either model is fitted, reserve a deterministic disjoint holdout from
   **official training normals only**. Freeze its fraction, minimum sample
   count, ID/hash-based selection algorithm, salt, exact membership, and use
   across seeds. Apply the same rule to every category and retain all categories.
   Do not borrow test normals or choose a split after viewing results.
3. Freeze new VisA model protocols: architectures, pretrained/auxiliary assets,
   training schedule, preprocessing, all normalization, threshold estimators,
   dependencies, hardware record, and seeds 42/123/2026. Normalization fitted
   from data uses permitted development normals only. Separate normalization
   fitting from threshold calibration if claiming exchangeable order-statistic
   coverage; otherwise explicitly disclaim that guarantee. These subset sizes
   and details need review before training, not after test access.
4. Fit each model solely on its permitted training-normal subset and calibrate
   solely on its reserved normal calibration subset. Freeze per-model scores,
   thresholds, checkpoints, configuration hashes, and complete category/seed
   manifests before test evaluation. No anomalous training information, test
   feedback, category rescue, or best seed. VisA training creates **new** model
   artifacts and thresholds under new protocol IDs; it cannot reuse MVTec audit
   identity or claim that MVTec numerical thresholds transfer across datasets.
   The model-agnostic triage truth table and metric definitions stay v1.
5. Freeze the evaluator, complete test manifest, and paired comparison reporting
   before release of test outcomes. Evaluate every frozen seed pair once as a
   single confirmatory campaign. Commit/hash predictions before the label join;
   retain failures and all metrics. Viewing the results consumes this holdout
   for subsequent rule design. Later optimization needs a new independent set.

No future model settings or holdout recipe can be chosen using VisA test outcomes
or MVTec public hybrid performance. This Phase 4B document freezes the triage
rule and independence requirements; it is deliberately **not an executable VisA
benchmark authorization**. The proposed route tests robustness of the unchanged
triage rule after normal-only model adaptation, not zero-shot transfer of the
existing MVTec checkpoints or production performance.

### C. Conditional MVTec AD 2 private evaluation

Official MVTec documentation keeps private ground truth non-public and provides
an evaluation server. Its published workflow describes continuous anomaly maps
and segmentation thresholds, rather than a three-way image decision interface.
Sources: [MVTec dataset documentation](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2),
[authors' paper, evaluation-server workflow](https://link.springer.com/article/10.1007/s11263-026-02743-0).

We have not established that the official mechanism accepts paired image
decisions and returns all required class-conditioned triage counts/rates while
keeping labels hidden. Standard localization scores alone cannot validate this
triage policy. This route remains conditional on documented support for those
outputs, concealed private labels, and explicit future human authorization.
Do not encode REVIEW as a binary decision or a fused map to fit a submission
format. If support is unavailable, document that incompatibility and stop this
route. Phase 4B accesses no private data or submission endpoint and makes no
private performance claim. MVTec's non-commercial license restrictions remain.

Sources above were checked on 2026-09-29. The VisA repository reference was
pinned to `2a692ab575001cbde74d402d897a7286086c6199`; no dataset was downloaded.

## Stop conditions and Phase 4C boundary

Stop and report an anchor mismatch, required threshold change, public-label
tuning, category rescue, cross-seed pairing, missing/malformed evidence,
inability to represent REVIEW, unreservable independent data, required private
access, or a claim exceeding evidence. Scientific settings are not repaired by
changing the protocol hash or silently excluding troublesome categories/seeds.

**Phase 4C is not authorized by this freeze or by passing CI.** Human review
must choose and explicitly authorize the next activity. A later scope could
permit a labeled retrospective replay or preparation/review of VisA-specific
training and evaluation protocols. Neither authorization should be inferred
from the other. Training, inference, confirmatory test access, and private-server
submissions require their own explicit scope and satisfied prerequisites. No
future phase may change v1's rule while reporting results as confirmation of v1.

## Engineering verification

Tests use synthetic scalar inputs and synthetic class labels only. Repository
provenance tests read committed metadata; none apply triage to benchmark data.
Tests cover all truth-table rows, exact/adjacent thresholds, separate score
scales, NaN/Inf and malformed inputs, immutable deterministic results, absent
category/label input paths, same-seed enforcement, explicit REVIEW handling,
hand-calculated metrics, zero denominators, canonical identity, scientific
mutation rejection, source anchor mismatch, and read-only verification CLI.

Local verification: **345 passed, 1 skipped**, including **127 new synthetic
policy/metric and protocol/provenance tests**. The skip is the existing Windows
symlink-privilege test, not a triage test. Ruff format/lint, `pip check`, and the
read-only CLI/import checks pass. Diff, secret-pattern, and large-file checks
cover the eight intended changed files; no datasets or model artifacts are added.
The existing CI matrix runs Python 3.11, 3.12, and 3.13; its live PR checks are
the authoritative hosted status for the final commit. No new benchmark was
executed, and no frozen scientific evidence or old model protocol was edited.
