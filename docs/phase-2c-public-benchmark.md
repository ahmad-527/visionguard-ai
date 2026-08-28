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

Not yet generated. This section is populated only after all 24 authorized cells
complete and their artifacts pass validation.
