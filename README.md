# VisionGuard AI

Production-grade deep learning platform for industrial visual anomaly detection, defect localization, benchmarking, and real-time quality inspection.

> **Project status:** Phase 2C — the merged frozen PatchCore protocol is ready
> for its first authorized public evaluation. No benchmark result is published
> until all required artifacts have been generated and reviewed.

## Mission

VisionGuard AI is being developed as an integrity-first computer vision project. The goal is not merely to train a model, but to build a reproducible end-to-end system covering data validation, anomaly detection, localization, experiment tracking, inference, API delivery, and an interactive inspection interface.

## Integrity principles

- No fabricated, estimated, or manually edited benchmark results.
- Every published metric must trace to code, configuration, dataset version, and a saved experiment artifact.
- Train/validation/test separation must be explicit and protected against leakage.
- Dataset licenses and attribution requirements must be respected.
- Failed and negative experiments may be retained when they improve scientific understanding.
- AI-assisted development is allowed, but architecture, evaluation, and published claims must remain reviewable and defensible.

## Planned system

```text
Image / Camera Input
        |
        v
Data Validation & Preprocessing
        |
        v
Anomaly Detection Model
        |
        +--> Image-level anomaly score
        |
        +--> Pixel-level anomaly map
        |
        v
Decision Layer
PASS / REVIEW / REJECT
        |
        +--> FastAPI inference service
        |
        +--> Interactive inspection application
```

## Current roadmap

1. Validate candidate industrial anomaly datasets and licensing.
2. Define the evaluation protocol and experiment schema.
3. Establish reproducible project structure and development rules.
4. Implement and verify a baseline model.
5. Compare stronger anomaly-detection approaches.
6. Add defect localization and decision calibration.
7. Build the inference API and user interface.
8. Add tests, CI, Docker, documentation, and reproducibility tooling.
9. Publish only verified benchmark results and documented limitations.

## Results

No benchmark results are published yet. This section will only contain values produced by committed code and traceable experiment artifacts.

## Development setup

Python 3.11 or newer is required. From a fresh clone, create an isolated
environment and install the package with its development tools:

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` on PowerShell or
`source .venv/bin/activate` on macOS/Linux), then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The default runtime dependency set remains intentionally small: Pillow performs
real image decoding and PyYAML loads explicit dataset contracts. Pytest and Ruff
are development-only dependencies. The PatchCore stack is isolated in the `ml`
optional group so dataset-audit users do not install a deep-learning framework.
See the [reviewed ML installation and provenance guide](docs/patchcore-dependencies.md)
before installing it; PyTorch and torchvision must come from the documented
official hardware-specific wheel index.

## Dataset audit

MVTec AD 2 must be obtained directly from MVTec and kept outside Git. See
[the local dataset setup and audit guide](docs/dataset-setup.md) for licensing,
placement, configuration, and command examples. Audit reports are generated from
the supplied filesystem; the repository contains no invented dataset statistics.
The [real-dataset validation record](docs/real-dataset-validation.md) documents
the Phase 1 full audit and its limitations. A separate
[Phase 2 proposal](docs/phase-2-proposal.md) is available for review; it contains
no model results and authorizes no training by itself. Phase 2A's enforced data,
threshold, metric, artifact, and reproducibility decisions are documented in the
[engineering protocol](docs/phase-2a-protocol.md). The checked-in PatchCore smoke
configuration is deliberately non-benchmark and does not enable `test_public`.
The [Phase 2B frozen benchmark protocol](docs/phase-2b-benchmark-protocol.md)
predeclares the public evaluation methodology. The
[Phase 2C execution contract](docs/phase-2c-public-benchmark.md) defines the
resumable 24-run workflow and evidence policy without authorizing private
evaluation.

## License

Source-code licensing will be selected after the Phase 0 dependency and dataset-license audit. Dataset assets will not be redistributed unless their licenses explicitly permit it.
