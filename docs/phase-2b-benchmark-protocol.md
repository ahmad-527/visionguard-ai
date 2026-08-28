# Phase 2B frozen PatchCore benchmark protocol

## Status and integrity boundary

Protocol `patchcore-mvtecad2-v1` is frozen before benchmark evaluation. Its
machine-readable source is
`configs/protocols/patchcore-mvtecad2-v1.yaml`; the canonical SHA-256
fingerprint is computed at runtime rather than copied into this document.

Phase 2B does not authorize a benchmark run. No `test_public` prediction or
metric, private label, private-server submission, or PatchCore benchmark result
was used to select this protocol. Phase 2C may perform the first public
evaluation only after this protocol is reviewed and merged. Private evaluation
requires a later, separate human authorization.

This document deliberately separates official requirements, VisionGuard
methodological choices, and engineering implementation details.

## A. Official MVTec AD 2 requirements and definitions

### Sources reviewed

The authoritative review used:

- the [MVTec AD 2 dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2);
- Heckler-Kram et al., [The MVTec AD 2 Dataset: Advanced Scenarios for
  Unsupervised Anomaly Detection](https://doi.org/10.1007/s11263-026-02743-0),
  especially Sections 3.2, 4.1.4, and 4.2;
- MVTec's official `MVTecAD2_public_code_utils` archive downloaded from the
  dataset page (reviewed archive SHA-256
  `fda9b379affbbde8b4d4fc1fe6ac52aaff981f347f3424e6b6de027457549f15`).

Anomalib is an implementation dependency, not the authority for the dataset's
evaluation protocol.

### Dataset and split roles

The complete category set is `can`, `fabric`, `fruit_jelly`, `rice`,
`sheet_metal`, `vial`, `wallplugs`, and `walnuts`. Training and validation
contain defect-free data only. `test_public` contains normal and anomalous
images with public pixel masks and is intended for local testing or initial
estimation. `test_private` and `test_private_mixed` expose images but not labels;
official evaluation is server-side. The mixed split depicts the same scenes as
the regular private split under a mixture of seen and unseen lighting.

MVTec AD 2 contains defects at image borders. In the paper's PatchCore setup,
center cropping is disabled for this reason. The paper generally resizes inputs
to 256×256 unless a method-specific exception is stated.

### Official metrics

The principal threshold-independent localization metric is per-region overlap
(PRO), averaged equally over four-connected ground-truth anomaly regions. The
PRO curve is integrated against false-positive rate only through 0.05 and
normalized by that range: AU-PRO\(_{0.05}\). Higher anomaly values mean more
anomalous pixels. Results are reported per category and as an unweighted mean
over categories.

Thresholded anomaly maps are evaluated with pixel-level F1. An image is rejected
when at least one pixel exceeds the segmentation threshold, and image-level F1
is then reported. These operational metrics are distinct from AU-PRO. The
paper's baseline threshold is the pooled validation-normal pixel mean plus three
standard deviations, reused as the image threshold. That formula is an official
paper baseline, not a mandatory server-side calibration rule.

### Submission contract

The official utility requires every private image for all eight categories and
both private splits. Continuous anomaly images are single-channel float16 TIFF
files in the original image dimensions. Optional thresholded maps are
single-channel PNG files containing only 0 for normal and 255 for anomalous.
The archive's checker validates exact names, counts, directories, types, and
binary values before compression. Continuous maps enable threshold-independent
evaluation; thresholded maps enable threshold-dependent evaluation.

The private labels and definitive evaluator are server-only. Local code cannot
claim equivalence to undisclosed server internals. Official server output must
be preserved verbatim with submission identity and time.

### Official resource-measurement reference

MVTec's reference procedure uses batch size 1 and float32, 1,000 GPU warm-up
passes followed by 1,000 timed passes, includes host-to-device transfer, and
stops after the anomaly image is returned to CPU. It reports mean, standard
deviation, minimum, and maximum runtime. GPU memory is PyTorch peak reserved
memory. PatchCore timing depends on memory-bank size; the paper averages real
test inference after warm-up rather than treating model initialization as
inference.

## B. VisionGuard methodological choices

### Frozen model

The first VisionGuard baseline remains deliberately simpler than the ensemble
used in MVTec's paper:

- Anomalib 2.6.0 PatchCore;
- `wide_resnet50_2.racm_in1k` only;
- `layer2` and `layer3` features;
- reviewed, revision-pinned pretrained weights;
- 3×3 stride-1 padding-1 average feature pooling;
- coreset sampling ratio 0.01;
- nine support neighbors for the weighted image score;
- Gaussian anomaly-map smoothing sigma 4;
- no augmentation.

The backbone, layers, coreset ratio, and neighbor count are the established
Anomalib/PatchCore defaults inherited by Phase 2A, not values selected using
MVTec AD 2 test outcomes. The explicit timm model identifier and revision are
VisionGuard provenance controls. This single-backbone baseline is not claimed
to reproduce MVTec's published ensemble.

### Frozen preprocessing

Images are decoded with Pillow through torchvision's default loader, which
converts them to RGB. This makes the dataset's gray-value categories three
channel by replication. Images are resized directly to 256×256 using bilinear
interpolation with antialiasing; native aspect ratio is not preserved. Center
cropping is disabled so border defects remain visible. Tensors are float32 CHW
in [0,1], then normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard
deviation `[0.229, 0.224, 0.225]`. No stochastic transform is permitted.

Patch scores are bilinearly upsampled to 256×256 with `align_corners=False`,
then smoothed by Anomalib's sigma-4 Gaussian operation. Before evaluation or
submission, the continuous map is bilinearly resized with
`align_corners=False` to the decoded original `(height, width)`. Thresholding
occurs only in original-image coordinates. The continuous float map remains the
source for ranking metrics and the private TIFF; visualization normalization is
forbidden on the metric path.

Changing decoding, color conversion, resizing, crop, normalization, smoothing,
or map restoration invalidates v1 and requires a new protocol identifier.

### Normal-only calibration

Image and pixel thresholds are separate. Both use the largest finite order
statistic from validation-normal data:

- the image threshold uses one PatchCore image score per validation image;
- the pixel threshold first reduces every validation map to its maximum pixel
  score, then uses the largest of those per-image maxima.

The decision rule is `score > threshold`; equality remains normal. This treats
the image as the sampling unit and does not pretend that spatially correlated
pixels are millions of independent observations. For `n` exchangeable normal
calibration images, the highest order statistic has marginal next-observation
coverage `n/(n+1)`. Artifacts record `n`, rank `n`, threshold, coverage, and the
hashes/identities of calibration inputs.

This is not an unconditional false-positive guarantee. Distribution shift,
including unseen private-mixed lighting, can violate exchangeability. Small
category validation sets limit attainable finite-sample coverage. The protocol
therefore publishes no claimed operational false-positive rate.

Alternatives were rejected before test evaluation:

- interpolated empirical quantiles add unsupported precision near a sparsely
  observed tail;
- pooled-pixel mean-plus-3σ ignores spatial dependence, assumes a useful tail
  summary, and is specifically problematic with PatchCore normalization in the
  MVTec AD 2 paper;
- EVT has too few independent image-level tail observations to justify its
  additional modeling choices.

### Ranking versus decisions

AU-PRO\(_{0.05}\) and supplemental image AUROC use continuous scores and never
use an operational threshold. Pixel and image F1 use only the predeclared
normal-only thresholds. F1 is never optimized using `test_public` or private
labels. Supplemental image AUROC must be labeled VisionGuard-local rather than
an official private-server metric.

Undefined categories are never silently removed. AU-PRO is undefined without
both normal pixels and at least one anomalous connected region. AUROC is
undefined without both image classes. F1 is undefined when neither labels nor
predictions contain a positive. An undefined result is recorded with its reason
and prevents an unqualified aggregate.

### Category and seed policy

Every official category is mandatory. Technical failure remains a visible
failed category and does not alter the denominator or silently create a
successful subset benchmark.

Seeds are exactly `42`, `123`, and `2026`. PatchCore's random projection and
random initial k-center make between-seed variation scientifically relevant.
Every seed is retained and reported. The predeclared summary is the unweighted
mean and sample standard deviation across seeds; no best-seed result is allowed.

### Failure policy

- A crash, OOM, non-finite value, missing image/map, or corrupted artifact marks
  the run invalid and remains recorded.
- Resolution, crop, batch size, coreset ratio, or another scientific field is
  never changed in-place to rescue a run.
- Dependency, weight, or protocol-fingerprint drift blocks execution.
- A scientifically necessary post-evaluation change creates a new protocol ID
  and restarts affected evaluation; v1 results are not pooled with it.
- No category or seed is dropped after outcomes are observed.

### Performance/resource protocol

Published inference timing follows MVTec's reference: float32, batch size 1,
1,000 warm-ups and 1,000 timed repetitions on GPU, synchronization through
returning the anomaly map to CPU, with mean, sample standard deviation, minimum,
and maximum. Model construction, weight loading, training feature extraction,
and coreset construction are reported separately. Peak allocated and peak
reserved CUDA memory are both captured and labeled. Device, driver, runtime,
image size, category, memory-bank size, and package environment are mandatory.
Phase 2A smoke timing is excluded from benchmark performance claims.

### Private evaluation policy

Private execution generates maps without labels, applies the frozen protocol,
runs the official checker, and records archive hash, protocol fingerprint,
submission identity, and submission time. Returned server artifacts are
preserved without transcription or modification. Submission and any resubmission
require human authorization; server feedback is never used for tuning.

## C. Engineering implementation details

### Protocol identity and gates

`visionguard.protocol` canonicalizes the machine-readable `protocol` mapping as
sorted, compact JSON and hashes it with SHA-256. Runtime dataset roots, audit
paths, output directories, and other machine-local context are outside that
mapping and cannot change the fingerprint. Any model, preprocessing,
calibration, metric, category, seed, loader, or dependency change does.

Public benchmark mode is not a default CLI path. Authorization requires an
explicit request, `test_public`, a clean Git tree, passing dataset audit,
verified weight SHA-256, exact categories and seeds, expected package versions,
and matching fingerprint. Private splits always fail this gate.

### Metric implementation

VisionGuard's independent AU-PRO implementation validates image counts, exact
2-D shapes, binary labels, finite scores, and score direction. It forms
four-connected ground-truth components, processes tied scores as one threshold
group, interpolates the curve at FPR 0.05, trapezoidally integrates, and divides
by 0.05. Synthetic tests cover perfect, reversed, tied/constant, no-anomaly,
all-anomaly, single-pixel, malformed-shape, NaN, and infinity cases.

Before Phase 2C publication, local outputs must also be checked against MVTec's
official public AU-PRO utility on synthetic/reference fixtures. A mismatch
blocks benchmark claims; VisionGuard will not relabel an approximation as
official.

### Benchmark artifact schema

Schema version 2 binds each category/seed run to protocol ID, fingerprint, and
snapshot; Git revision and clean state; dataset audit hash/status; exact model,
preprocessing, environment, and weight hash; calibration inputs and outputs;
thresholds; per-image scores; original-coordinate map path/shape/hash; metric
implementation identity; category metrics; aggregation; resources; warnings;
failures; and status. Validation rejects protocol, category, seed, Git, audit,
weight, score, or map-identity drift. Large maps and artifacts remain ignored
outside Git.

### Repeatability authorization

The Phase 2B repeatability check may use one category and only `train/good` plus
`validation/good`. Identical runs compare memory-bank SHA-256, ordered validation
scores, thresholds, and original-coordinate anomaly-map hashes. Exact equality
is reported only if observed. Otherwise maximum absolute/relative differences
and map differences are recorded with declared tolerances. It is engineering
evidence, not a benchmark result.

The completed same-machine outcome is recorded in
[`phase-2b-repeatability.md`](phase-2b-repeatability.md). The compared memory
bank, validation scores, approved order-statistic thresholds, and all stored map
hashes were exact across two runs; the record states the limited scope and does
not generalize this observation across environments.

## Protocol amendment rule

Any benchmark-relevant change after Phase 2C begins invalidates the affected
series. The replacement must use a new protocol identifier, a new fingerprint,
fresh review, and fresh runs. Historical failures and results remain traceable.
