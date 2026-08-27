# VisionGuard AI — Agent Engineering Contract

This file defines mandatory rules for AI coding agents working in this repository.

## Mission

Build VisionGuard AI as a production-quality, reproducible deep-learning system for industrial visual anomaly detection and defect localization. Engineering quality and scientific integrity take priority over speed or impressive-looking results.

## Non-negotiable integrity rules

1. **Never fabricate results.** Never invent, estimate, interpolate, or manually improve metrics, logs, timings, model outputs, dataset statistics, or benchmark tables.
2. **Never present external benchmark numbers as our results.** Literature values must be clearly attributed and separated from locally generated results.
3. **Never tune on the final test set.** Validation is used for model selection and threshold calibration. Final/private test evaluation is reserved for final assessment.
4. **Never hide failed experiments.** Failed or negative results may be summarized or archived when they inform engineering decisions.
5. **Never manually edit generated metric artifacts to improve results.** Fix the code and rerun the experiment instead.
6. **Prevent data leakage.** Explicitly validate configured splits and implement duplicate/overlap checks where feasible.
7. **Respect licenses.** Do not commit or redistribute dataset images, annotations, pretrained weights, or third-party assets unless their licenses permit redistribution.
8. **Do not commit secrets.** Never commit API keys, passwords, access tokens, SSH keys, `.env` secrets, credentials, or private user information.

## Git workflow

- Do not force-push.
- Do not rewrite published history.
- Do not delete remote branches or tags without explicit approval.
- Do not change repository visibility, security settings, or access permissions.
- Prefer focused feature/experiment branches for substantive implementation work.
- Use descriptive conventional-style commits such as `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `exp:`, `perf:`, `ci:`.
- Before proposing a merge, inspect the diff and run the relevant tests, formatting, linting, and validation checks.
- Do not merge substantive changes into `main` merely because tests pass; they must also satisfy the scientific and architectural requirements of the task.

## Code quality

- Prefer maintainable modules over monolithic scripts or notebooks.
- Use type hints for public Python interfaces where practical.
- Add docstrings where behavior, assumptions, shapes, units, or side effects are not obvious.
- Avoid hard-coded machine-specific paths.
- Keep configuration separate from implementation where it improves reproducibility.
- Use structured logging instead of unnecessary `print` statements in application/library code.
- Raise informative errors rather than silently swallowing failures.
- Add tests for data validation, preprocessing, metrics, inference contracts, and API behavior as those components are introduced.
- Keep notebooks for exploration and analysis; production logic belongs in importable modules.

## ML experiment requirements

Every benchmark-worthy run must be traceable to:

- Git commit SHA;
- experiment/config identifier;
- dataset and subset/category;
- model architecture and pretrained weights;
- preprocessing and augmentation;
- random seed(s);
- threshold/calibration method;
- relevant environment/package versions;
- hardware where performance measurements depend on it;
- machine-generated metrics artifacts.

Do not place a number in the README Results section unless its provenance can be demonstrated.

## Dataset policy

Primary benchmark: **MVTec AD 2**.
Secondary robustness benchmark: **VisA**.

Read `docs/dataset-strategy.md` before implementing dataset ingestion or evaluation.

Large datasets must live outside Git history. Provide scripts/instructions for obtaining and validating them rather than committing the data.

## Architecture direction

The intended system will eventually include:

- dataset acquisition/validation tooling;
- reproducible preprocessing and configuration;
- baseline anomaly detection;
- comparative experiments with stronger methods;
- image-level anomaly scoring;
- pixel-level anomaly localization/heatmaps;
- calibrated PASS / REVIEW / REJECT decision logic;
- inference service/API;
- interactive inspection UI;
- tests and CI;
- containerized deployment;
- documented limitations and failure analysis.

Do not implement all layers prematurely. Work milestone by milestone and preserve a working, reviewable repository.

## AI-assisted development disclosure

AI assistance is permitted and expected in this project. It must not be used to conceal copied work, falsify authorship, fabricate experiments, or bypass licenses. Prefer original integration, documented design decisions, proper attribution, and verifiable experiments over re-creating another repository wholesale.

## Definition of done for a task

A task is not complete merely because code runs. Before declaring completion, confirm as applicable:

1. requirements are satisfied;
2. tests pass;
3. lint/format checks pass;
4. no secrets or large unintended artifacts are included;
5. relevant documentation is updated;
6. experimental claims are supported by artifacts;
7. known limitations or unresolved risks are reported;
8. the diff has been reviewed for unnecessary generated code or unrelated changes.

When uncertain about a scientific assumption, dataset license, benchmark protocol, destructive Git operation, or architectural decision with major downstream impact, stop and surface the uncertainty rather than guessing.
