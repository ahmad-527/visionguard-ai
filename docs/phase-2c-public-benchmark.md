# Phase 2C public PatchCore benchmark

## Scope and authorization

Phase 2C executes the first authorized evaluation of MVTec AD 2 `test_public`
under the merged frozen protocol `patchcore-mvtecad2-v1`. The complete matrix is
eight declared categories by three declared seeds, for 24 required runs. A
failed cell remains visible and prevents an unqualified aggregate.

This phase does not authorize access to private labels, a private-server
submission, protocol tuning, Phase 3 / EfficientAD work, or merging its pull
request. Public outcomes must not be used to change the frozen model,
preprocessing, calibration, metric, category, or seed choices.

## Execution command

The benchmark is deliberately unavailable through the smoke command. From the
pinned CUDA 12.6 environment and a clean committed worktree, run:

```powershell
visionguard-patchcore-benchmark <dataset-root> `
  --audit-report <passed-audit-report.json> `
  --protocol configs/protocols/patchcore-mvtecad2-v1.yaml `
  --output-root outputs/phase2c-public-benchmark `
  --benchmark-mode
```

If execution is interrupted, repeat the command with `--resume`. The manifest
binds resumed work to the same protocol fingerprint, Git commit, dataset-audit
hash, and pretrained-weight hash. Completed cells are validated and skipped;
failed attempts remain recorded and a new attempt directory is used.

## Per-cell pipeline

Each category/seed cell performs the following fixed sequence:

1. Re-run the public benchmark gate using the full frozen category and seed
   sets, `test_public`, clean Git state, passed audit identity, exact dependency
   versions, protocol fingerprint, and cached weight hash.
2. Configure deterministic Python, NumPy, PyTorch, cuDNN, and cuBLAS controls.
3. Fit PatchCore from that category's `train/good` images only.
4. Predict `validation/good`, restore each map bilinearly to its original image
   coordinates, and calibrate separate image and pixel highest-order-statistic
   thresholds. Pixel calibration uses one restored-map maximum per image.
5. Predict every `test_public/good` and `test_public/bad` image exactly once.
6. Restore maps to original coordinates. Continuous maps are converted to the
   official MVTec submission dtype, single-channel float16 TIFF. Thresholded
   maps are generated from the restored float32 score with the predeclared
   strict `score > threshold` comparison and stored as binary PNG.
7. Compute AU-PRO at FPR 0.05, pixel F1, image F1, and supplemental image AUROC.
8. Write an immutable schema-v2 artifact containing the protocol snapshot,
   environment, memory-bank identity, calibration inputs, thresholds, ordered
   predictions, map hashes, metrics, resources, warnings, and failures.

The protocol's `tensor_dtype: float32` describes model input tensors. MVTec's
official AD 2 submission checker separately requires continuous anomaly TIFFs
to be float16. Local AU-PRO is therefore evaluated on those exact float16 maps,
matching the score precision that a future private submission would contain.

## Exact streaming AU-PRO

The official reference sorts every original-resolution pixel, which has a high
peak-memory cost for AD 2 images. The VisionGuard implementation is an exact
re-expression for official float16 maps: it accumulates false-positive and
per-region-overlap changes in one bin for each of the 65,536 float16 bit
patterns, orders the used finite scores, groups ties, interpolates at FPR 0.05,
and applies trapezoidal integration. Eight-connected ground-truth regions and
the unweighted region mean are unchanged. Synthetic reference tests compare it
with the previously verified direct implementation.

This is an engineering implementation choice, not a change to the frozen
metric definition.

## Aggregation and reporting

The machine-generated summary retains each run. For each category and metric it
reports the unweighted mean and sample standard deviation across seeds 42, 123,
and 2026. The overall value is the unweighted arithmetic mean of the eight
category means. No seed or category is selected, dropped, or reweighted.

Execution wall time and peak CUDA allocation/reservation are operational
records only. They are not the separately frozen 1,000-warm-up/1,000-repetition
inference-speed benchmark and must not be published as model latency.

## Failure and evidence policy

- The manifest is updated before and after each cell.
- A crash, OOM, non-finite score, malformed mask, missing map, undefined metric,
  or artifact-integrity failure records a failed attempt and interrupts the
  process for diagnosis.
- Resume never overwrites a prior attempt or completed artifact.
- Scientific settings are never reduced or changed to rescue a run.
- Generated maps, artifacts, manifests, and summaries remain under ignored
  `outputs/`; dataset assets and pretrained weights remain outside Git.
- The final human-readable results section is added only from the completed,
  hash-bound machine summary and must state any failures or limitations.

## Results

### Completion and provenance

The authorized matrix completed on 2026-08-28 with all 24 category/seed cells
successful on their first attempt. The machine manifest records 3,252 ordered
public predictions and corresponding continuous/thresholded map identities.
Independent post-run validation re-hashed all 24 artifacts and all 3,252 map
pairs successfully.

- benchmark implementation Git commit:
  `8848e8defb1f734a319168fd597b4252b606fff7`;
- protocol fingerprint:
  `03f545ea23b1bd00206cb919aece6972502712aa9f981e8a3f11dbd1be1f0c2b`;
- dataset audit SHA-256:
  `8c0f71f0a7dc81436b7bd3affed0ba7f97ea3844213d487c2d9886befa055a92`;
- pretrained-weight SHA-256:
  `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000`;
- committed machine summary SHA-256:
  `cef20151986bbfb554b59286b1b23c971d5027e6bd3dd2a1aa035fa0232e62c8`;
- complete local evidence size: 27.584 GiB across 6,554 files;
- failed attempts: 0.

The exact machine-generated [manifest](../reports/phase2c-public-benchmark/benchmark-manifest.json)
and [summary](../reports/phase2c-public-benchmark/benchmark-summary.json) are
committed. Full per-image artifacts and maps remain ignored under `outputs/`
because of their size; their hashes are retained in the manifest and per-run
artifacts.

### Public candidate results

Values are fractions in `[0, 1]`. Each category entry is the predeclared
unweighted mean across seeds 42, 123, and 2026, followed by the sample standard
deviation. The overall row is the unweighted mean of the eight category means;
it does not select a seed or category.

| Category | AU-PRO@0.05 | Pixel F1 | Image F1 | Image AUROC |
| --- | ---: | ---: | ---: | ---: |
| can | 0.008988 ± 0.011756 | 0.000000 ± 0.000000 | 0.442763 ± 0.081228 | 0.477109 ± 0.020834 |
| fabric | 0.036633 ± 0.014266 | 0.112113 ± 0.009081 | 0.289170 ± 0.067848 | 0.631987 ± 0.061078 |
| fruit_jelly | 0.392454 ± 0.005442 | 0.165390 ± 0.101984 | 0.396079 ± 0.096298 | 0.841944 ± 0.032214 |
| rice | 0.132654 ± 0.002838 | 0.103237 ± 0.015816 | 0.210134 ± 0.018140 | 0.536772 ± 0.003704 |
| sheet_metal | 0.078845 ± 0.005252 | 0.329499 ± 0.031502 | 0.383967 ± 0.128668 | 0.693827 ± 0.041435 |
| vial | 0.471360 ± 0.028954 | 0.127071 ± 0.038786 | 0.533415 ± 0.102278 | 0.878277 ± 0.019134 |
| wallplugs | 0.088729 ± 0.019282 | 0.000000 ± 0.000000 | 0.072756 ± 0.012586 | 0.476975 ± 0.027939 |
| walnuts | 0.378767 ± 0.012053 | 0.477443 ± 0.011691 | 0.747128 ± 0.040877 | 0.815370 ± 0.008432 |
| **Unweighted category mean** | **0.198554** | **0.164344** | **0.384427** | **0.669033** |

AU-PRO@0.05, pixel F1, and image F1 are the protocol's official/local metrics.
Image AUROC is the explicitly labeled VisionGuard-local supplemental metric.
F1 uses only validation-normal thresholds and was not optimized against public
labels. In particular, the measured zero pixel F1 for `can` and `wallplugs` is
retained rather than hidden or recalibrated.

### Interpretation boundaries and execution note

These are preliminary public-split measurements of the frozen PatchCore
baseline, not private-server results and not evidence that a final VisionGuard
model is effective. Category variation and weak aggregate localization are
visible limitations. The public outcomes were not used to change the protocol.

The user requested a pause while `walnuts`, seed 42 was running. That process
was suspended and resumed in place. Its stored wall-clock duration
(`12,868.55` seconds) includes approximately 3.5 hours of suspension and is
invalid for runtime interpretation. No model inputs, state, scores, maps, or
metrics were changed by the pause. All execution wall times remain operational
records only; no inference-speed benchmark was performed or claimed.

Private labels were not accessed, no private archive was evaluated, and no
submission was made to the MVTec server. Private evaluation still requires
separate human authorization.
