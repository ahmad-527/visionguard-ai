# Phase 3A proposed frozen EfficientAD comparison protocol

## Status and integrity boundary

`efficientad-mvtecad2-v1` is proposed for review before any EfficientAD public
evaluation. Its machine-readable source is
`configs/protocols/efficientad-mvtecad2-v1.yaml` and its deterministic
fingerprint is
`e9d6a66e7a52f2993e984ec20278c4ca4c710198cc466df15f947adff763f69f`.
Machine-local dataset, cache, audit, and output paths are outside the hashed
mapping.

Phase 3A permits only MVTec AD 2 `train/good` and `validation/good` for
engineering. The code deliberately rejects `test_public`, both private split
names, and every other split. The future benchmark gate validates the intended
prerequisites but always ends in a Phase 3A lock error. EfficientAD public
evaluation requires this protocol to be reviewed and merged plus a distinct
Phase 3B authorization.

Historical PatchCore public results were not used to choose any EfficientAD
setting. Settings are uniform across all categories. In particular, no choice
responds to PatchCore's category outcomes, and Phase 3A produces no EfficientAD
public metric or comparison winner.

## A. Original EfficientAD methodology

The WACV 2024 paper defines a lightweight patch-description-network (PDN)
teacher, a two-head student, and an autoencoder. Local anomalies use the
teacher/student squared feature discrepancy; logical anomalies use the
autoencoder/student discrepancy. Their maps are normalized separately using
normal validation quantiles and averaged equally. The student loss keeps the
hardest 0.1% teacher/student feature discrepancies and adds a penalty on
natural ImageNet images. The autoencoder and second student head train on a
random brightness, contrast, or saturation transform.

The reference recipe uses 256×256 inputs, batch size one, Adam at `1e-4` with
weight decay `1e-5`, 70,000 optimization steps, and a 0.1 learning-rate factor
at 95% of training. The method defines small and medium PDNs; they are separate
model choices, not candidates to select after test evaluation.

The paper's latency values and benchmark results are literature evidence only.
They are not VisionGuard measurements.

## B. Maintained Anomalib 2.6.0 behavior

VisionGuard uses the PyPI wheel `anomalib==2.6.0` (wheel SHA-256
`0395d2e2ad859fb45b9c4544479639afe5d6aaada5e2aefc460bb65b638bd972`),
corresponding to tag `lib/v2.6.0` at
`3759687e76395c4d6d239552d3bf6d72e003da78`. The implementation is inspectable,
Apache-2.0, exposes teacher/student/autoencoder tensors and raw maps, and allows
VisionGuard to bypass its evaluator and operational postprocessor.

Anomalib's implementation:

- defaults to PDN-S with 384 teacher channels, `padding=false`, and padded maps;
- loads a frozen distilled teacher and initializes student/autoencoder under the
  configured PyTorch seed;
- computes teacher channel mean/std over all training-normal images;
- uses Adam and StepLR with the paper settings;
- uses the hard-feature 0.999 quantile plus a natural-image student penalty;
- applies one random brightness/contrast/saturation transform in `[0.8, 1.2]`;
- uses ImageNette rather than full ImageNet for its penalty stream, resized to
  512×512, random grayscale probability 0.3, then center-cropped to 256×256;
- obtains validation-normal 0.9 and 0.995 quantiles independently for
  student/teacher and student/autoencoder maps, scales each normalized map by
  0.1, and averages them equally;
- bilinearly upsamples maps to 256×256. VisionGuard then restores them
  bilinearly with `align_corners=false` to decoded original coordinates.

Anomalib documentation describes the teacher as an EfficientNet backbone, but
the reviewed source and weight keys implement a distilled PDN. The source and
loaded tensor contract, not that inaccurate prose description, are frozen.

## C. VisionGuard protocol decisions

### Primary variant

The sole primary baseline is **PDN-S**. It is Anomalib 2.6.0's default, directly
serves EfficientAD's efficiency objective, and is comfortably feasible on the
available RTX 3070 Ti Laptop GPU. PDN-M is rejected for v1 because it increases
compute and capacity without being needed to establish the first maintained
reference baseline. This choice was made before any EfficientAD public result;
PDN-M must not be run on `test_public` as an alternative and selected afterward.

### Preprocessing

Pillow/torchvision decoding converts every input to RGB. Images are resized
directly to 256×256 with bilinear antialiased interpolation, without preserving
aspect ratio, crop, or pad. Tensors are CHW float32 in `[0,1]`. ImageNet
mean/std normalization occurs inside PDN and autoencoder forwards, matching the
maintained source. Inference has no stochastic transform. Internal and restored
map interpolation are bilinear; restoration targets original `(height,width)`
and uses `align_corners=false`. Thresholding and external metrics consume only
the original-coordinate map.

This differs from PatchCore internally because the methods are specified on
their own authoritative recipes. Fairness comes from the common dataset,
original-coordinate outputs, external metrics, seeds, and aggregation—not from
forcing identical internal feature pipelines.

### Training

- optimizer: Adam, learning rate `1e-4`, weight decay `1e-5`;
- schedule: StepLR gamma 0.1 at step 66,500 (95% of 70,000);
- exactly 70,000 steps, batch size 1, no accumulation, no gradient clipping;
- float32 only; mixed precision disabled;
- frozen standardized 384-channel teacher; seeded default student/AE init;
- summed hard-feature/penalty, AE, and student/AE losses;
- shuffled training and penalty streams, worker count 0;
- final-step checkpoint; no early stopping or anomaly-label selection;
- CUDA device, deterministic algorithms requested, cuDNN benchmark disabled,
  cuBLAS workspace `:4096:8`.

An OOM or deterministic-operation error is a visible failed run. Resolution,
variant, batch definition, steps, auxiliary data, or precision is never changed
silently to rescue a benchmark cell.

### Penalty data and licenses

The v1 recipe requires fastai ImageNette 2 because omitting the penalty term
would materially change EfficientAD and Anomalib 2.6.0. The exact archive is
`imagenette2.tgz`, source
`https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz`, SHA-256
`6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd`.
ImageNette is an ImageNet subset. Although fastai's repository code is
Apache-2.0, ImageNet does not own the underlying image copyrights and provides
images for non-commercial research/education under its access terms.
VisionGuard therefore treats the archive and derived runs as non-commercial
research only and never redistributes it. This restriction is compatible with
the already non-commercial MVTec AD 2 project scope but is unsuitable for a
commercial product without separate legal review.

### Teacher provenance

The Anomalib project publishes
`efficientad_pretrained_weights.zip` from its EfficientAD release, archive
SHA-256
`c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a`.
The selected file is `pretrained_teacher_small.pth`, 10,779,695 bytes, SHA-256
`a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d`.
The unused medium file is identified but not selected: SHA-256
`f7356663c8e00ada12ae01fb8c8aad0a1de2f800f8eadf252a46d29bbdfdf718`.
The release belongs to the Apache-2.0 Anomalib project; the teacher is derived
from ImageNet distillation, so VisionGuard applies the same conservative
research-only restriction and does not redistribute either file. Runtime code
requires the selected file hash before training.

### Native normalization versus operational calibration

Native EfficientAD normalization and operational decisions remain distinct.
The required internal 0.9/0.995 quantiles are computed from validation-normal
raw component maps and stored in the checkpoint/artifact. After equal map
combination and original-coordinate restoration, the image threshold is the
highest finite validation-normal image score. The pixel threshold is the
highest finite order statistic of one restored-map maximum per validation
image. Decisions use strict `score > threshold`; equality is normal. This is
compatible with EfficientAD's higher-is-more-anomalous finite score semantics
and preserves images—not correlated pixels—as calibration units.

### External metrics and comparison

Phase 3B will use VisionGuard's same verified definitions as PatchCore:
AU-PRO at FPR 0.05 on official-format original-coordinate float16 maps, pixel
F1, image F1, and supplemental VisionGuard-local image AUROC. Ties, 8-connected
regions, per-category aggregation, and the unweighted mean/sample SD across
seeds 42, 123, and 2026 remain unchanged. Anomalib metrics and thresholds cannot
replace this evaluator. All eight categories and all seeds must be reported;
there is no best-seed selection.

## D. Deviations from the paper/reference

1. Maintained Anomalib 2.6.0 is used rather than recreating paper code.
2. ImageNette 2 replaces the full ImageNet penalty stream. This is Anomalib's
   explicit maintained behavior and makes the auxiliary identity tractable.
3. VisionGuard adds normal-only operational thresholds; the native quantiles
   remain model normalization, not decision thresholds.
4. VisionGuard restores maps to MVTec AD 2 original coordinates and uses its
   independently verified external metrics/artifacts.
5. Three predeclared comparison seeds replace any single-run reporting.

These deviations are fixed before evaluation. A change requires a new protocol
ID and fingerprint, not an in-place amendment after observing outcomes.

## E. Artifact, gate, failure, and resource contracts

Schema v3 extends `visionguard.artifacts` for non-benchmark EfficientAD evidence.
It binds protocol snapshot/fingerprint, Git and audit identity, implementation,
variant, exact training/preprocessing, auxiliary and teacher hashes,
environment, deterministic controls, native quantiles, operational calibration,
canonical checkpoint tensor hash, ordered validation predictions, restored-map
hashes, failures, warnings, and resources. It rejects benchmark claims,
evaluation splits, category/seed drift, non-finite values, and non-portable
paths during Phase 3A.

The future Phase 3B gate requires explicit benchmark mode, `test_public`, a
clean tree, passed dataset audit, exact categories/seeds/dependencies, exact
teacher and ImageNette identities, and the reviewed fingerprint. Even when all
are correct, the current gate refuses execution with the Phase 3A lock. Private
split names are rejected earlier.

Failures retained by the future orchestrator include OOM, NaN/Inf loss/score,
crash, corrupt checkpoint, missing/mismatched weight or auxiliary data, audit,
protocol, or dependency drift. No category is dropped and no scientific field
is silently changed.

Formal inference measurement is separately predeclared: float32 batch one,
1,000 warm-ups, 1,000 timed repetitions, CUDA synchronization, start before
host-to-device transfer, stop after map return to CPU, and report mean, sample
SD, min/max, peak allocated, and peak reserved memory. Training time is
separate. PatchCore operational benchmark wall times are not a formal latency
comparator; a fair speed study remains later work.

## Engineering smoke and repeatability evidence

On 2026-08-28, two identical short runs used `can` only, seed 42, the same
environment/assets, every `train/good` image for teacher statistics, two
optimization steps, and all 46 `validation/good` images for native normalization
and calibration. This deliberately short run validates plumbing, not the
70,000-step recipe and not model effectiveness.

The two runs were bitwise identical for the canonical final checkpoint
(`463c6e0c61c7f1b513af79964335d470e4895c9c81623922d1ec1987f551ec61`),
two-value loss trajectory, four native quantiles, ordered validation scores,
both operational threshold records, and all 46 restored-map hashes. Peak CUDA
allocation/reservation were 400,742,400/583,008,256 bytes in both runs. This is
same-machine, same-environment engineering evidence only; it does not establish
cross-platform determinism or formal memory/latency performance.

The first two attempts reached completed training/validation but failed during
artifact serialization (scalar checkpoint hashing, then a missing environment
function argument). The next identical pair completed, but diff review found
that the smoke-only scheduler had been scaled to two steps and its image
calibration used the restored-map maximum instead of EfficientAD's native image
score. Those artifacts were preserved as superseded engineering evidence, the
probe was corrected to retain the real 66,500-step schedule boundary and native
image score, and both final runs were repeated. None produced a benchmark
metric; the defects and superseded evidence are not hidden.

## Dependencies

No new direct package was required: the reviewed Phase 2 ML extra and common
Python 3.11/3.12 lock already contain Anomalib 2.6.0, Lightning 2.6.5, PyTorch
2.9.1, torchvision 0.24.1, and their exact resolved dependencies. CPU/CUDA
hardware manifests remain separate, and ordinary CI remains lightweight with
no GPU, network, dataset, weights, or auxiliary data.

## Authoritative sources reviewed

- [Original WACV 2024 EfficientAD paper](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)
- [Anomalib 2.6.0 EfficientAD source](https://github.com/open-edge-platform/anomalib/tree/lib/v2.6.0/src/anomalib/models/image/efficient_ad)
- [Anomalib pretrained-weight release](https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights)
- [Anomalib 2.6.0 PyPI release](https://pypi.org/project/anomalib/2.6.0/)
- [fastai ImageNette repository](https://github.com/fastai/imagenette)
- [ImageNet access/copyright statement](https://www.image-net.org/about)
- [Official MVTec AD 2 dataset and evaluation page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

## Remaining limitations

- The repeatability run is only a two-step engineering probe on one machine.
- The full 70,000-step, eight-category, three-seed compute cost is not measured.
- ImageNet-derived asset rights remain research-only under VisionGuard's
  conservative policy and require legal review for any commercial use.
- Formal inference timing and fair PatchCore/EfficientAD speed comparison have
  not been executed.
- The protocol remains proposed until human review and merge. Phase 3B has not
  started.
