# Phase 3B public EfficientAD benchmark

## Scope and authorization

Phase 3B executes the first authorized evaluation of EfficientAD on MVTec AD 2
`test_public` under the frozen protocol `efficientad-mvtecad2-v1`. The complete
matrix is eight declared categories by three declared seeds, for 24 required
cells. Public outcomes did not change the model, preprocessing, calibration,
metric, category, or seed choices.

This phase does not authorize access to private labels, a private-server
submission, protocol tuning, another model or phase, or merging its pull
request. The results below are public-split candidate measurements, not final
private-test performance.

## Execution and evidence

The interruption-safe CLI and recovery policy are documented in the
[Phase 3B execution contract](phase-3b-execution-contract.md). Each cell trained
the frozen PDN-S EfficientAD configuration for 70,000 optimization steps,
calibrated image and pixel thresholds from `validation/good` only, evaluated
every ordered `test_public/good` and `test_public/bad` item once, and wrote
original-coordinate continuous TIFF and thresholded PNG maps.

The machine summary retains all seeds. Category values are unweighted means and
sample standard deviations across seeds 42, 123, and 2026. Overall values are
unweighted arithmetic means of the eight category means. No seed or category
was selected, dropped, or reweighted.

## Results

### Completion and provenance

The authorized matrix completed on 2026-09-04 with all 24 category/seed cells
successful. The manifest records 3,252 ordered public predictions. An
independent post-run validation re-hashed all 24 artifacts, their final
checkpoints, and all 3,252 continuous/thresholded map pairs successfully.

- benchmark implementation Git commit:
  `9e477389530743f8a7cf4caa8c48214e5c63ec28`;
- protocol fingerprint:
  `e9d6a66e7a52f2993e984ec20278c4ca4c710198cc466df15f947adff763f69f`;
- dataset audit SHA-256:
  `8c0f71f0a7dc81436b7bd3affed0ba7f97ea3844213d487c2d9886befa055a92`;
- teacher weight SHA-256:
  `a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d`;
- teacher archive SHA-256:
  `c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a`;
- ImageNette archive SHA-256:
  `6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd`;
- environment SHA-256:
  `ff013d8d026eb5cdc7ed643b6f56bcbc48669680236c94639e8a8627281e3093`;
- local machine manifest SHA-256:
  `c951d74724227eb589c9e4344f99928d8a05cfd0b33a4fbc01bbdf71f5212b05`;
- local machine summary SHA-256:
  `61e51bd43ff0c7c41557d07438d4ec9782bf7b9f2ac12adf39a87e481f4b8a61`;
- committed path-sanitized, LF-normalized manifest SHA-256:
  `e2ac8e3f050271458b9d1aa04afb3543f7792b38906a2e3e73c872cce8e8841a`;
- committed LF-normalized summary SHA-256:
  `cf4bb72364cee1594d8c67e61d20d43d2729937a95541370b0322b8ee8503296`;
- complete local evidence size: 26.670 GiB across 6,555 files;
- failed attempts: 1;
- recorded interruptions: 5.

The exact machine-generated [summary](../reports/phase3b-efficientad-public-benchmark/benchmark-summary.json)
and a [path-sanitized manifest](../reports/phase3b-efficientad-public-benchmark/benchmark-manifest.json)
are committed. The committed manifest replaces two occurrences of the same
absolute local dataset path in the preserved failure message with
`<local-dataset-root>`; metrics, identities, statuses, and histories are
unchanged. The original manifest remains in ignored local evidence and is bound
by its hash above. Full per-image artifacts, checkpoints, and maps also remain
ignored under `outputs/` because of their size; their hashes remain bound into
the manifest and per-cell artifacts.

### Public candidate results

Values are fractions in `[0, 1]`. Each category entry is the predeclared
unweighted mean across the three seeds, followed by the sample standard
deviation. Image AUROC is the explicitly labeled VisionGuard-local supplemental
metric. F1 thresholds use validation-normal data only and were not optimized
against public labels.

| Category | AU-PRO@0.05 | Pixel F1 | Image F1 | Image AUROC |
| --- | ---: | ---: | ---: | ---: |
| can | 0.017550 +/- 0.005728 | 0.000130 +/- 0.000027 | 0.649383 +/- 0.007060 | 0.489763 +/- 0.014064 |
| fabric | 0.063512 +/- 0.001499 | 0.022091 +/- 0.000540 | 0.668475 +/- 0.033659 | 0.541077 +/- 0.015929 |
| fruit_jelly | 0.332994 +/- 0.007267 | 0.058645 +/- 0.029983 | 0.293306 +/- 0.036824 | 0.830000 +/- 0.023467 |
| rice | 0.020303 +/- 0.001408 | 0.004038 +/- 0.000177 | 0.725721 +/- 0.018722 | 0.528571 +/- 0.013164 |
| sheet_metal | 0.065037 +/- 0.003438 | 0.207488 +/- 0.017995 | 0.773431 +/- 0.019723 | 0.600617 +/- 0.014354 |
| vial | 0.678785 +/- 0.010372 | 0.070376 +/- 0.018644 | 0.566148 +/- 0.049186 | 0.811066 +/- 0.010714 |
| wallplugs | 0.031852 +/- 0.002399 | 0.006692 +/- 0.000787 | 0.602118 +/- 0.015714 | 0.530185 +/- 0.003208 |
| walnuts | 0.199166 +/- 0.035532 | 0.000027 +/- 0.000047 | 0.249884 +/- 0.066517 | 0.762407 +/- 0.005293 |
| **Unweighted category mean** | **0.176150** | **0.046186** | **0.566058** | **0.636711** |

The low aggregate localization scores and near-zero `walnuts` pixel F1 are
retained as measured. No post-result threshold change or category-specific
adjustment was made.

### Frozen PatchCore baseline comparison

The prior PatchCore public benchmark used the same eight categories, seeds, and
metric definitions. The table compares the two frozen overall aggregates; the
delta is EfficientAD minus PatchCore.

| Metric | PatchCore | EfficientAD | Delta |
| --- | ---: | ---: | ---: |
| AU-PRO@0.05 | 0.198554 | 0.176150 | -0.022404 |
| Pixel F1 | 0.164344 | 0.046186 | -0.118158 |
| Image F1 | 0.384427 | 0.566058 | +0.181632 |
| Image AUROC | 0.669033 | 0.636711 | -0.032322 |

EfficientAD improves the validation-calibrated image F1 aggregate while
measuring lower on aggregate AU-PRO, pixel F1, and image AUROC. These public
results do not establish a final model winner, and image F1 should not be read
as a ranking metric independent of its model-specific validation threshold.

### Failure, interruption, and timing record

One engineering failure is preserved for `rice`, seed 123, attempt 1: a local
training image was missing after the temporary dataset tree was disrupted. The
local dataset was restored and revalidated before a new immutable attempt
completed; no scientific setting changed.

Five interruptions are retained: four user-requested stops during training and
one external or unclean interruption while resuming `wallplugs`, seed 2026.
Each continuation used the latest valid 1,000-step checkpoint and the exact
committed `--resume` workflow. Earlier attempts and their histories remain in
the manifest.

Training durations are operational records only. Pauses, OS sleep, and resume
segments make them unsuitable for a speed comparison. The separately frozen
1,000-warm-up/1,000-timed-repetition inference benchmark was not run, so no
latency or throughput claim is made.

Private labels were not accessed, no private archive was evaluated, and no
submission was made to the MVTec server. Private evaluation still requires
separate human authorization.
