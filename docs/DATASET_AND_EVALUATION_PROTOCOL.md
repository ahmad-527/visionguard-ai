# Dataset and Evaluation Protocol

## Status
Phase 0 decision record. No benchmark results are claimed by this document.

## Dataset strategy

### Primary benchmark: MVTec AD 2
VisionGuard will use MVTec AD 2 as its primary research benchmark, subject to compliance with its dataset terms. The benchmark is attractive because it targets challenging industrial anomaly detection, provides pixel-level ground truth for public data, includes acquisition-condition/domain shifts, and supports evaluation against private test data through the official evaluation system.

**License constraint:** MVTec AD 2 is distributed under CC BY-NC-SA 4.0. VisionGuard therefore treats dataset-dependent work as non-commercial research/education unless separate permission is obtained. Dataset files must not be committed to this repository.

### Secondary external validation: VisA
VisA will be used as a secondary benchmark where practical. It provides a separate industrial anomaly dataset and is distributed under CC BY 4.0 by its maintainers. Using a second dataset helps test whether conclusions generalize beyond one benchmark.

### Legacy comparison: MVTec AD
The original MVTec AD may be used for comparison with established literature and tooling. It is not the primary target because MVTec AD 2 was designed to provide a more challenging successor benchmark.

## Split and leakage policy

1. Final test data is never used to select models, thresholds, augmentations, hyperparameters, or stopping criteria.
2. Validation data may be used for model selection and threshold calibration only when the benchmark protocol permits it.
3. Dataset-provided splits are preserved unless an experiment explicitly requires another split.
4. Any custom split must be deterministic, seeded, recorded, and justified.
5. Where file access permits, duplicate and near-duplicate checks must be performed across custom splits.
6. Preprocessing statistics that learn from data must be fit only on permitted training data.
7. Private benchmark results are reported exactly as returned by the official evaluator.

## Metric policy

Metrics will be chosen according to the benchmark task and official protocol. Candidate measures include image-level AUROC, pixel-level AUROC, average precision, F1 at a documented threshold, false-positive/false-negative measures, localization metrics supported by the benchmark, and inference performance.

A metric may appear in the README only if all of the following exist:

- exact dataset and category scope;
- model/configuration identifier;
- source-code Git commit;
- saved machine-readable metric artifact;
- documented threshold-selection method when applicable;
- environment/model metadata sufficient to reproduce the run;
- evaluation code or official evaluator provenance.

## Experiment artifact contract

Each publishable experiment should produce a directory or tracked run containing, at minimum:

```text
experiment_id/
├── config.yaml
├── metrics.json
├── model_metadata.json
├── environment.txt
├── training.log
└── predictions_or_evaluation_reference.*
```

Large weights, datasets, caches, and raw generated outputs must not be committed directly to Git unless explicitly approved and technically appropriate.

## Reproducibility

Runs must record random seeds where relevant. Deterministic behavior should be enabled where practical, but any performance cost or nondeterministic CUDA operation must be documented rather than hidden. Dependency versions and hardware information must be captured for benchmark runs.

## Reporting failures

VisionGuard will document meaningful failure modes and limitations. Poor-performing categories, domain-shift failures, false positives, and false negatives must not be omitted merely because they weaken the headline result.

## Dataset redistribution

The repository will contain dataset acquisition/validation instructions or scripts where permitted, not copies of restricted dataset assets. Attribution and license notices required by each dataset must be preserved.

## Decision review

This protocol must be reviewed if the dataset terms, official evaluation procedure, or project use changes. Commercial use of dataset-dependent components requires a separate licensing review.