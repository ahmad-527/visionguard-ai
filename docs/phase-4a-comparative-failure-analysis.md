# Phase 4A comparative failure analysis

## Status, scope, and authorization

Phase 4A is a model-free analysis of the already generated, frozen PatchCore
Phase 2C and EfficientAD Phase 3B MVTec AD 2 `test_public` evidence. It compares
the same eight categories and the same predeclared seeds 42, 123, and 2026. It
does not train, tune, recalibrate, or run inference for either model.

The analysis reads only the committed benchmark manifests, ignored local
artifacts, stored image scores and decisions, stored continuous anomaly maps,
stored thresholded maps, and audited public images and masks. The frozen
validation-normal thresholds are used exactly as recorded. Public labels are
never used to choose a replacement threshold, rescue a category, select a seed,
or alter either protocol.

This phase does not authorize:

- new training or inference;
- threshold optimization on `test_public`;
- category-specific tuning or post-processing;
- access to `test_private`, `test_private_mixed`, private labels, or the MVTec
  evaluation server;
- a private-performance, deployment-speed, or production-readiness claim;
- merging this work without review; or
- Phase 4B or another benchmark.

All negative results and recorded source-run failures remain part of the
evidence. If any required identity, artifact, prediction, map, image, or mask
cannot be validated, the analysis fails closed and no generated report may be
treated as valid.

## Frozen evidence and provenance gates

The two committed evidence anchors are:

- `reports/phase2c-public-benchmark/benchmark-manifest.json` for PatchCore; and
- `reports/phase3b-efficientad-public-benchmark/benchmark-manifest.json` for
  EfficientAD.

The corresponding full evidence remains outside Git under
`outputs/phase2c-public-benchmark/` and
`outputs/phase3b-efficientad-public-benchmark/`. The common schema-v2 dataset
audit is machine-local at `config/local/mvtec-ad-2-audit.json`. Its SHA-256 is
`8c0f71f0a7dc81436b7bd3affed0ba7f97ea3844213d487c2d9886befa055a92`.

For PatchCore, the read-only readiness audit confirmed:

- protocol `patchcore-mvtecad2-v1` and fingerprint
  `03f545ea23b1bd00206cb919aece6972502712aa9f981e8a3f11dbd1be1f0c2b`;
- benchmark implementation commit
  `8848e8defb1f734a319168fd597b4252b606fff7`;
- all 24 category/seed cells completed;
- 3,252 ordered public predictions, continuous maps, and thresholded maps;
- local manifest SHA-256
  `cceaf8614f550dffb3172f0d3b928e7a30192aa24f3bbf0555193ad2028d0d13`;
- local summary SHA-256
  `cef20151986bbfb554b59286b1b23c971d5027e6bd3dd2a1aa035fa0232e62c8`;
- committed manifest SHA-256
  `762c693616cbb02d9e215264c589497ef97cc07b29c1bf2ba2e661d6e7d2e171`;
- committed summary SHA-256
  `8a0adb2991a5cf47ae75b941828c45f39e5195c5f41e4b32a6d2b2300036cccf`;
  and
- successful re-hashing of every artifact, continuous map, and thresholded map.

For EfficientAD, the read-only readiness audit confirmed:

- protocol `efficientad-mvtecad2-v1` and fingerprint
  `e9d6a66e7a52f2993e984ec20278c4ca4c710198cc466df15f947adff763f69f`;
- benchmark implementation commit
  `9e477389530743f8a7cf4caa8c48214e5c63ec28`;
- all 24 category/seed cells completed;
- 3,252 ordered public predictions, continuous maps, and thresholded maps;
- raw local manifest SHA-256
  `c951d74724227eb589c9e4344f99928d8a05cfd0b33a4fbc01bbdf71f5212b05`;
- raw local summary SHA-256
  `61e51bd43ff0c7c41557d07438d4ec9782bf7b9f2ac12adf39a87e481f4b8a61`;
- LF-normalized committed sanitized-manifest SHA-256
  `e2ac8e3f050271458b9d1aa04afb3543f7792b38906a2e3e73c872cce8e8841a`;
- LF-normalized committed summary SHA-256
  `cf4bb72364cee1594d8c67e61d20d43d2729937a95541370b0322b8ee8503296`;
  and
- successful re-hashing of all artifacts, final checkpoints, continuous maps,
  and thresholded maps.

The sanitized EfficientAD manifest differs from the local manifest only in the
two documented replacements of the absolute dataset root in the preserved
`rice`, seed 123 failure message. Comparisons use semantic JSON plus this narrow
redaction rule; they do not mistake platform line endings for scientific drift.

Before calculating results, `visionguard-comparative-analysis` verifies all of
the following:

1. Both committed and local manifests are complete and differ only by an
   explicitly permitted absolute-root redaction.
2. Protocol IDs and deterministic fingerprints match the manifests.
3. The recorded benchmark Git objects exist. A squash merge may make a source
   commit a non-ancestor of the analysis branch, so reachability is recorded
   separately from artifact and protocol identity.
4. Both models use the exact same passed dataset-audit identity.
5. The category-major 8 x 3 matrix is complete, with no unknown, omitted, or
   selected cell.
6. Every per-cell artifact hash and schema is valid. EfficientAD checkpoint
   hashes and all available environment, teacher, archive, and auxiliary-data
   bindings must agree with its manifest.
7. Every calibration input is a unique, lexically ordered
   `validation/good` sample and exactly matches the audited validation-normal
   inventory. The highest-order-statistic image and pixel thresholds are
   recomputed from those stored inputs and must equal the recorded thresholds.
8. Predictions are unique and lexically ordered, their labels agree with their
   public paths, and their saved decisions equal `score > frozen threshold`.
9. PatchCore and EfficientAD predictions pair exactly by category, seed,
   ordinal, sample identifier, and label, and cover the complete audited public
   inventory.
10. All 1,084 public images and 705 public masks still match their audit hashes.
    Private assets are neither opened nor required.
11. Every continuous and thresholded map exists, matches its recorded hash, and
    has the audited original image shape. Continuous maps must be finite,
    single-page, MINISBLACK float16 TIFFs. Thresholded maps must be binary
    grayscale PNGs.
12. Recomputed image F1, image AUROC, pixel F1, and AU-PRO@0.05 match every
    frozen per-seed artifact and both frozen benchmark summaries.

`analysis-summary.json` records the analysis implementation Git state, exact
source-file hashes, package versions, all gate outcomes, source histories, and
verification counts. A partial output left by an interrupted or failed run is
not evidence and must not be committed.

## Deterministic analysis pipeline

For every category and seed, the pipeline processes the two artifacts in their
preserved lexical prediction order. It loads the audited public mask, or creates
the protocol-defined all-zero mask for a public-normal image, and reads each
model's stored continuous TIFF and authoritative frozen-threshold PNG. It never
loads a model or checkpoint for inference.

The compact outputs are:

- `reports/phase4a-comparative-failure-analysis/analysis-summary.json`:
  provenance, per-cell metrics and diagnostics, category/overall aggregates,
  paired deltas, disagreements, failure-taxonomy counts, and targeted
  EfficientAD findings;
- `reports/phase4a-comparative-failure-analysis/per-image-analysis.jsonl`:
  ordered, traceable image decisions and algorithmic localization indicators;
  and
- `reports/phase4a-comparative-failure-analysis/panel-index.json`:
  deterministic panel selections and hashes without redistributing dataset
  images; and
- `reports/phase4a-comparative-failure-analysis/analysis-manifest.json`:
  hashes and sizes for the three analysis artifacts above, with input-manifest
  identities and record counts.

The reports contain no trained parameters or dataset pixels. Large visual
panels remain ignored under
`outputs/phase4a-comparative-failure-analysis/panels/`.

### Metric definitions

The positive image class is anomalous. At each model's frozen, validation-
derived image threshold:

- true positive (TP): anomalous image predicted anomalous;
- false positive (FP): normal image predicted anomalous;
- true negative (TN): normal image predicted normal;
- false negative (FN): anomalous image predicted normal;
- sensitivity/recall: `TP / (TP + FN)`;
- specificity: `TN / (TN + FP)`;
- precision: `TP / (TP + FP)`; and
- image F1: `2 TP / (2 TP + FP + FN)`.

Pixel precision, sensitivity, specificity, and F1 use the analogous counts from
the stored thresholded PNG and public mask. The PNG is authoritative because it
was produced from the pre-cast float32 map during the frozen benchmark, whereas
the continuous evidence was subsequently stored as float16. The analysis
records any pixels for which re-thresholding the float16 TIFF would disagree,
but does not substitute that reconstructed decision.

Image AUROC is the frozen tie-aware VisionGuard ranking metric. AU-PRO@0.05 is
recomputed from the original-coordinate float16 TIFF with the frozen 8-connected
region, exact-float16-tie, and FPR-limit contract. Pixel AUROC is included only
as an explicitly labeled Phase 4A diagnostic of anomaly-pixel/background-pixel
ranking; it is not a frozen benchmark metric or a model-selection target.

Image anomaly-score distributions are kept separately for normal and anomalous
public images. Each records count, minimum, first quartile, median, third
quartile, maximum, mean, and population standard deviation. Quantiles use
linear interpolation at index `(n - 1) p`. Continuous pixel-score distributions
use the same rule over exact float16 histograms without materializing a selected
threshold.

Per-category values are the unweighted mean and sample standard deviation over
all three seeds. Overall values are the unweighted arithmetic mean of the eight
category means. No best seed, category weighting, or post-result exclusion is
used. Paired deltas are always EfficientAD minus PatchCore.

## Ranking metrics versus frozen-threshold metrics

These metric families answer different questions and must not be conflated.

| Family | Metrics | Interpretation |
| --- | --- | --- |
| Ranking | Image AUROC, AU-PRO@0.05 | Ordering/separation across thresholds under the frozen metric definitions |
| Supplemental diagnostic ranking | Pixel AUROC diagnostic | Pixel ordering only; not a frozen selection metric |
| Frozen image threshold | TP, FP, TN, FN, sensitivity, specificity, precision, image F1 | Decisions at the model's own validation-normal threshold |
| Frozen pixel threshold | Pixel confusion, precision, sensitivity, specificity, pixel F1 | Localization decisions in the saved benchmark PNG |

A favorable threshold-dependent F1 can coexist with weak ranking, poor
specificity, or class-prevalence effects. Conversely, useful continuous-map
ranking can coexist with a poorly scaled operational threshold. Model selection
therefore considers all metric families, seed stability, and traceable failure
patterns rather than one favorable number.

## Comparative results

The placeholders in this section must be replaced only from a successfully
generated and validated `analysis-summary.json`. Seed-level values remain in
the machine-readable report.

### Overall unweighted category means

| Metric | Family | PatchCore | EfficientAD | EfficientAD - PatchCore |
| --- | --- | ---: | ---: | ---: |
| Image AUROC | ranking | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| AU-PRO@0.05 | ranking | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Pixel AUROC diagnostic | diagnostic ranking | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Sensitivity | frozen image threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Specificity | frozen image threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Precision | frozen image threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Image F1 | frozen image threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Pixel precision | frozen pixel threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Pixel sensitivity | frozen pixel threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Pixel specificity | frozen pixel threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |
| Pixel F1 | frozen pixel threshold | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE | PENDING MACHINE EVIDENCE |

### Category-level ranking comparison

| Category | PatchCore image AUROC | EfficientAD image AUROC | PatchCore AU-PRO@0.05 | EfficientAD AU-PRO@0.05 |
| --- | ---: | ---: | ---: | ---: |
| can | PENDING | PENDING | PENDING | PENDING |
| fabric | PENDING | PENDING | PENDING | PENDING |
| fruit_jelly | PENDING | PENDING | PENDING | PENDING |
| rice | PENDING | PENDING | PENDING | PENDING |
| sheet_metal | PENDING | PENDING | PENDING | PENDING |
| vial | PENDING | PENDING | PENDING | PENDING |
| wallplugs | PENDING | PENDING | PENDING | PENDING |
| walnuts | PENDING | PENDING | PENDING | PENDING |

### Category-level frozen-threshold comparison

| Category | PatchCore image F1 | EfficientAD image F1 | PatchCore pixel F1 | EfficientAD pixel F1 |
| --- | ---: | ---: | ---: | ---: |
| can | PENDING | PENDING | PENDING | PENDING |
| fabric | PENDING | PENDING | PENDING | PENDING |
| fruit_jelly | PENDING | PENDING | PENDING | PENDING |
| rice | PENDING | PENDING | PENDING | PENDING |
| sheet_metal | PENDING | PENDING | PENDING | PENDING |
| vial | PENDING | PENDING | PENDING | PENDING |
| wallplugs | PENDING | PENDING | PENDING | PENDING |
| walnuts | PENDING | PENDING | PENDING | PENDING |

Category/seed confusion counts, sensitivity, specificity, precision, image F1,
image AUROC, AU-PRO@0.05, pixel confusion and rates, score distributions, and
threshold diagnostics are preserved without rounding in
`analysis-summary.json`.

## Image-decision disagreement

Every paired prediction receives exactly one image-level bucket:

- `both_correct`: both frozen decisions equal the public label;
- `patchcore_only_correct`: only PatchCore equals the label;
- `efficientad_only_correct`: only EfficientAD equals the label; or
- `both_wrong`: neither model equals the label.

Counts preserve every category/seed observation; the same public image appears
once for each predeclared seed. `per-image-analysis.jsonl` retains the category,
seed, ordinal, sample identifier, label, scores, decisions, correctness, and
bucket, so every aggregate and example remains traceable.

| Bucket | All category/seed observations |
| --- | ---: |
| Both correct | PENDING MACHINE EVIDENCE |
| PatchCore only correct | PENDING MACHINE EVIDENCE |
| EfficientAD only correct | PENDING MACHINE EVIDENCE |
| Both wrong | PENDING MACHINE EVIDENCE |

The category and seed breakdown is retained in the machine-readable summary.
Disagreement is descriptive evidence, not an authorization to construct or
evaluate a routing rule on the public labels.

## Algorithmic localization-failure taxonomy

Localization indicators are deterministic, non-exclusive, and descriptive.
They do not assign an undocumented visual or causal label:

- `missed_anomaly`: ground-truth anomaly pixels exist and pixel TP is zero;
- `under_localization`: ground-truth anomaly pixels exist, TP is positive, and
  FN exceeds TP;
- `over_localization`: ground-truth anomaly pixels exist, TP is positive, and
  FP exceeds TP;
- `diffuse_false_positive_map`: false-positive pixels occur in all four fixed
  image quadrants;
- `threshold_collapse`: the stored frozen binary map is empty on an anomalous
  image or full-frame, while its stored continuous float16 map is nonconstant;
  and
- `constant_continuous_map`: an anomalous image has a constant continuous map
  and zero overlap.

An image may carry multiple indicators. Counts and per-image TP/FP/TN/FN,
continuous extrema, predicted area, and label are preserved so future reviewers
can audit every assignment.

## Targeted EfficientAD localization investigation

The following findings come only from the validated Phase 3B scores,
calibration inputs, maps, and public masks. No alternative threshold was tested.

### Can

| Seed | Frozen pixel threshold | Image TP / FP / TN / FN | AU-PRO@0.05 | Pixel F1 | Pixel TP | Pixel FP | Pixel FN | Anomalous images with any mask overlap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.087390 | 72 / 57 / 15 / 18 | 0.010966 | 0.000100 | 1,027 | 20,493,342 | 14,321 | 11 / 90 |
| 123 | 0.786660 | 70 / 57 / 15 / 20 | 0.021389 | 0.000136 | 1,404 | 20,670,470 | 13,944 | 11 / 90 |
| 2026 | 0.856854 | 71 / 59 / 13 / 19 | 0.020294 | 0.000153 | 1,804 | 23,525,543 | 13,544 | 11 / 90 |

Pixel precision is approximately 0.0050% to 0.0077%, while pixel sensitivity is
6.69% to 11.75%. The stored maps mark roughly 20.5 to 23.5 million pixels but
overlap only 1,027 to 1,804 of 15,348 anomalous pixels.

The response is also strongly capture-condition dependent. For every seed, all
27 regular-lighting public images have empty frozen maps. Nearly every
underexposed, `shift_1`, `shift_2`, and `shift_3` image has a nonempty map
regardless of whether the image is normal or anomalous. All mask overlap is
confined to 11 anomalous `shift_1`/`shift_3` captures. Together with image AUROC
of approximately 0.474 to 0.501 and AU-PRO@0.05 below 0.022, this supports poor
normal/anomaly ranking and spatial localization plus diffuse, illumination-
conditioned false positives. A too-high frozen threshold is not the primary
measured explanation for `can`.

### Walnuts

| Seed | Frozen pixel threshold | Image TP / FP / TN / FN | AU-PRO@0.05 | Pixel F1 | Validation maximum / second maximum | Validation maximum / median | Anomalous images with any map | Anomalous images with mask overlap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 37.527328 | 16 / 5 / 55 / 74 | 0.185677 | 0.000000 | 57.20x | 357.30x | 16 / 90 | 0 / 90 |
| 123 | 38.623848 | 9 / 5 / 55 / 81 | 0.239468 | 0.000000 | 76.74x | 341.00x | 9 / 90 | 0 / 90 |
| 2026 | 27.541859 | 16 / 5 / 55 / 74 | 0.172354 | 0.000082 | 52.35x | 221.89x | 16 / 90 | 1 / 90 |

The same audited normal calibration image,
`walnuts/validation/good/018_regular.png`, sets the maximum image and pixel
threshold for all three seeds. Its pixel maximum is approximately 52 to 77
times the second-highest calibration maximum and 222 to 357 times the median.
This is a repeatable, measurable maximum-order-statistic threshold-collapse
mechanism rather than a seed-specific fluctuation.

At the frozen threshold, seeds 42 and 123 produce no true-positive mask pixel;
seed 2026 produces only 382. The few above-threshold public regions are almost
entirely outside the masks. However, AU-PRO@0.05 remains approximately 0.172 to
0.239, so the continuous maps retain limited ranking/localization signal across
the frozen threshold sweep. The evidence therefore supports both threshold-
scale collapse and high-score spatial mislocalization; it does not support the
claim that the continuous maps contain no signal.

EfficientAD stores only its final, equally combined normalized map. It does not
store the student/teacher and autoencoder component maps for public images.
Consequently, the evidence can measure native quantile spans, threshold scale,
score distributions, capture-condition behavior, and spatial overlap, but
cannot identify which component branch caused the extreme responses. Producing
those missing component maps would require forbidden reevaluation.

## Deterministic representative panels

Panels use seed 42 only. Selection first takes one normal and one anomalous
image per category by the smallest SHA-256 of
`category + NUL + seed + NUL + label + NUL + sample_id`. It then takes the
smallest hash from each nonempty global disagreement bucket and deduplicates
images while retaining every selection reason. Scores, metric values, map
content, and visual inspection are excluded from selection.

Each panel has this fixed order:

`original image | public ground-truth mask | PatchCore continuous map |
PatchCore frozen map | EfficientAD continuous map | EfficientAD frozen map`

Continuous maps are independently min-max scaled for display only; all numeric
analysis uses the stored values. The panel index records the selection hash,
source image/mask identities, both models' map hashes, frozen thresholds, and
panel hash. The PNG panels remain ignored because they contain licensed dataset
imagery and are large derived artifacts. Only the compact selection metadata is
committed.

Generated panel count: **PENDING MACHINE EVIDENCE**.

## Model-selection assessment

**PENDING HUMAN-REVIEWED SYNTHESIS OF THE VERIFIED MACHINE EVIDENCE.**

The final statement must be limited to one evidence-supported conclusion:

- prefer PatchCore;
- prefer EfficientAD;
- task-dependent routing or a hybrid merits further study; or
- the evidence is currently insufficient.

The assessment must consider ranking, frozen-threshold behavior, localization,
normal/anomalous score separation, category and seed stability, and failure
patterns together. No winner may be inferred from image F1 alone, from a single
category, or from informal panel appearance. The current evidence contains no
formal matched inference-speed measurement, so EfficientAD's intended
efficiency cannot be used as a locally measured advantage.

## Limitations

- This is public-split candidate evidence, not private-test performance.
- Three fixed seeds characterize only the declared same-machine experiments.
- Public lighting variants are traceable capture conditions, not additional
  independent defect scenes; condition-level diagnostics are descriptive.
- Each model has its own validation-normal score scale and threshold. Raw score
  magnitudes and threshold values are not directly comparable between models.
- Pixel AUROC is supplemental Phase 4A analysis and was not predeclared as a
  frozen benchmark-selection metric.
- The saved binary PNG, not a re-thresholded float16 TIFF, defines frozen pixel
  decisions. Quantization mismatches are recorded rather than silently changed.
- Panel heatmaps use independent display scaling and cannot support absolute
  cross-model score comparisons.
- EfficientAD component maps were not stored, which limits causal attribution
  of normalization failures.
- No new routing, ensemble, hybrid, calibration, or rescue strategy has been
  evaluated. Such work would require a separately reviewed phase.
- Source-run failures and interruptions remain preserved; operational durations
  do not constitute a fair speed benchmark.

## Reproduction

From the repository root, use the reviewed analysis dependencies and the exact
audited dataset root:

```powershell
python -m pip install -e ".[analysis,dev]"
visionguard-comparative-analysis <dataset-root> `
  --audit-report config/local/mvtec-ad-2-audit.json `
  --patchcore-manifest reports/phase2c-public-benchmark/benchmark-manifest.json `
  --patchcore-evidence outputs/phase2c-public-benchmark `
  --efficientad-manifest reports/phase3b-efficientad-public-benchmark/benchmark-manifest.json `
  --efficientad-evidence outputs/phase3b-efficientad-public-benchmark `
  --report-output reports/phase4a-comparative-failure-analysis `
  --panel-output outputs/phase4a-comparative-failure-analysis/panels
```

The command performs only read-only validation and analysis of its evidence
inputs. It writes compact reports and ignored panels to separate output roots.

Before proposing the pull request for review, run:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest
python -m pip check
```

Secret scanning, large-file hygiene, a complete diff review, and the repository
CI matrix are also required. Record the verified local and CI outcomes here:

- pytest: **PENDING VERIFIED RUN**;
- Ruff format/lint: **PENDING VERIFIED RUN**;
- pip check: **PENDING VERIFIED RUN**;
- secret and large-file hygiene: **PENDING VERIFIED RUN**; and
- CI: **PENDING VERIFIED RUN**.

Phase 4A ends with a focused pull request and human review. It must not be
merged automatically, and Phase 4B or another benchmark must not begin from
this report alone.
