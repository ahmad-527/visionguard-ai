# Phase 2A PatchCore engineering protocol

## Scope

Phase 2A establishes infrastructure and permits at most one category-level,
non-benchmark engineering smoke run after all automated checks pass. It does not
authorize parameter optimization, a full dataset benchmark, private-label access,
or a README performance claim.

## Data boundaries

Memory-bank construction uses only official `train/good`. Normalization and
threshold calibration use only official `validation/good`. `test_public` is
disabled in the checked-in smoke configuration. It can be enabled for later
preliminary evaluation only after configuration and calibration are explicitly
frozen. Both private test views are unavailable to local experiment roles; the
future final path will submit predictions without reading private labels.

Every real run requires a passing schema-version-2 audit report. The artifact
records its SHA-256 and portable identity fields, not its machine-specific root or
per-file local paths.

## Threshold decision

The provisional Phase 2A rule is a linearly interpolated empirical quantile of
validation-normal scores. Image and pixel populations use separate configurations
and produce separate thresholds. The checked-in engineering values are 0.995 for
image scores and 0.999 for pixel scores; they are predeclared smoke defaults, were
not selected from public/private anomaly outcomes, and are not yet approved as a
benchmark protocol.

An empirical quantile is transparent and distribution-free but unstable with a
small validation sample and cannot directly justify an operational false-alarm
rate outside the observed distribution. Candidate future alternatives include a
finite-sample order statistic with an explicit false-positive confidence bound
and a prevalidated peaks-over-threshold tail model. Selecting among them requires
human scientific review before benchmark use; Phase 2A must not compare them on
`test_public` to make that choice.

The smoke configuration requires at least 20 validation-normal image scores and
1,000 validation-normal pixel scores. These are input-sufficiency guards, not a
claim that the resulting thresholds have a particular confidence or operational
false-alarm guarantee.

## Metric contract

VisionGuard distinguishes image-ranking, pixel-localization, and thresholded
decision metrics. The initial independent implementation covers tie-aware binary
AUROC and binary F1, with pixel arrays validated and flattened only after exact
shape checks. AUROC is explicitly undefined without both classes; F1 is explicitly
undefined when labels and predictions contain no positives. Synthetic tests cover
perfect, inverted, constant, empty-positive, malformed, and shape-mismatch cases.

Official MVTec AD 2 outputs, including the definitive private-server evaluation
and any official localization aggregation, remain unimplemented and must not be
substituted with these local sanity metrics.

## Artifact and reproducibility contract

Artifact schema version 1 records experiment ID, UTC timestamp, Git commit/branch
and dirty state, dataset audit identity, full configuration, weight identity/hash,
seed controls, environment/device capture, thresholds, per-sample predictions,
metrics, runtime/resource measurements, warnings, failures, and completion status.
Phase 2A hard-codes `benchmark_claim: false` and refuses to overwrite an existing
artifact. Generated anomaly maps are retained as non-pickle NumPy arrays outside
Git; each prediction records a portable relative path, shape, finite-value check,
and SHA-256.

Environment fields distinguish detected, unavailable, and configured values.
Seed handling covers Python, NumPy when installed, PyTorch, CUDA seed-all behavior,
deterministic algorithms, and cuDNN settings. These controls improve repeatability
but do not claim bitwise determinism across devices, versions, or unsupported
operations.

Generated artifacts, predictions, model caches, weights, checkpoints, and run
outputs remain ignored and outside Git. Runtime and memory may be reported only
when written by instrumentation; missing measurements remain missing.
