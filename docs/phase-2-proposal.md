# Phase 2 proposal: baseline anomaly detection

## Status

This is a design proposal for human review. It contains no VisionGuard model
results, benchmark values, runtime measurements, or GPU measurements. No model
training was performed while preparing it.

## Proposed objective

Phase 2 should add one reproducible baseline at a time while preserving the
validated dataset boundaries. The recommended sequence is:

1. implement and verify a PatchCore baseline;
2. freeze its preprocessing, metric, and artifact contracts;
3. add EfficientAD as the first trained comparison method;
4. compare methods only after both pipelines pass independent correctness and
   leakage reviews.

PatchCore is the preferred first baseline because fitting consists primarily of
pretrained feature extraction, memory-bank construction, and coreset selection.
EfficientAD should follow because its teacher/student/autoencoder training loop,
teacher weights, penalty data option, and normalization introduce more
reproducibility and licensing decisions.

## Implementation-source decision

Two routes should be evaluated in a short dependency/licensing spike before
model code is committed:

- **Maintained Anomalib adapter:**
  [Anomalib](https://github.com/open-edge-platform/anomalib) currently includes
  both PatchCore and EfficientAD under Apache-2.0 and exposes common
  training/evaluation plumbing. This reduces algorithm reimplementation risk but
  adds Lightning, TorchMetrics, timm, OpenCV, scikit-image, scikit-learn, Kornia,
  and other dependencies.
- **Minimal project-native integration:** use the
  [official PatchCore reference repository](https://github.com/amazon-science/patchcore-inspection)
  as a behavior reference while writing a small typed adapter around PyTorch,
  torchvision/timm, and a reviewed nearest-neighbor implementation. This keeps
  the dependency surface smaller but requires substantially more correctness
  testing. EfficientAD would still need a separately reviewed authoritative
  implementation source; a repository that describes itself as unofficial must
  not silently become the scientific reference.

Recommendation: perform the dependency spike first and prefer the maintained
Anomalib adapter if its exact version, transitive dependencies, pretrained-weight
provenance, and MVTec AD 2 data contract can be pinned and reviewed. Do not add
the full ML stack to Phase 1 runtime dependencies.

## Dependency and environment implications

Create an isolated optional ML dependency group or lock file. It should include
an explicitly selected PyTorch/torchvision build, the chosen model integration,
and only required numerical/image packages. Record:

- Python, PyTorch, torchvision, CUDA/runtime, and driver versions;
- resolved package lock and installation source;
- model-library commit or release;
- pretrained-weight source, license, cryptographic hash, and loading policy;
- CPU/GPU device selection and determinism settings.

Do not download weights until their source and redistribution terms have been
reviewed. Never commit downloaded weights.

## GPU and resource plan

No minimum GPU claim should be made before measurement. PatchCore feature
extraction and nearest-neighbor inference can be designed to support CPU, but its
memory bank and MVTec AD 2 resolution may make GPU acceleration and bounded
chunking important. EfficientAD training is expected to benefit materially from
CUDA acceleration. The first dry run should therefore:

- inventory actual available hardware;
- run only a tiny non-benchmark smoke fixture first;
- measure peak memory and runtime with committed instrumentation;
- choose image resizing, tiling, batch size, and coreset settings from a written
  resource protocol, not from private-test feedback.

Measured resource values belong in machine-generated artifacts, never as manual
estimates in documentation.

## Experiment design

Each run should be driven by a validated configuration containing the Git SHA,
experiment ID, category, seed, model/backbone and weight identity, input size,
preprocessing, augmentation, coreset or training parameters, threshold method,
environment capture, and output directory.

Data use must remain fixed:

- train only on official `train` normal images;
- use only official `validation` normal images for normalization, calibration,
  early stopping, or model selection;
- use `test_public` only for preliminary post-freeze evaluation/debugging;
- keep both private test views out of tuning and submit them to the official
  server only for a final assessment.

Start with one category as an engineering smoke run only after the pipeline is
complete. A benchmark run should cover the predeclared category set and seeds;
failed runs must remain traceable.

## Evaluation contract

Metric definitions and implementations must be selected and tested before any
result is generated. Candidate outputs are:

- image-level anomaly score and a documented ranking metric;
- pixel-level anomaly map at original-image coordinates;
- official pixel-localization metric(s) required by MVTec AD 2;
- threshold-dependent image and pixel decisions when supported by the official
  evaluation path;
- measured runtime and memory only when the measurement protocol is active.

Metric implementations should have synthetic sanity tests for perfect,
inverted, constant, empty-positive, and shape-mismatch cases. Public/private
server output must remain clearly separated from local synthetic or public-test
results.

## Threshold calibration

Because validation data is defect-free, do not optimize a threshold against
public or private anomalous labels. Predeclare a normal-only calibration rule,
such as a fixed validation-score quantile or another justified tail model, and
record all parameters. Calibrate image and pixel thresholds independently when
their score distributions differ. Freeze thresholds before public-test
evaluation and never revise them from private-server feedback.

## Reproducibility and artifacts

Every run should emit immutable machine-readable artifacts for configuration,
environment, seed, Git state, dataset audit identity, checkpoint/weight hashes,
per-sample predictions, thresholds, metrics, failures, and measured resource
usage. Artifact schemas should be versioned and tested. Large predictions,
weights, and checkpoints remain outside Git, with documented retention and
integrity hashes.

## Human decisions required before implementation

1. Approve Anomalib versus a minimal project-native integration.
2. Approve the exact dependency/lock strategy and supported accelerator builds.
3. Review pretrained-weight and any auxiliary-data licenses.
4. Freeze input resizing/tiling and metric definitions.
5. Approve the normal-only threshold calibration rule.
6. Define when a run becomes benchmark-worthy and when private-server evaluation
   is allowed.
