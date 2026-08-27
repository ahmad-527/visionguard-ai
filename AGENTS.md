# VisionGuard AI — Agent Engineering Contract

These rules apply to Codex and any other AI coding agent working in this repository.

## Mission
Build a reproducible, production-minded industrial visual anomaly detection and quality-inspection platform. Optimize for technical integrity, clear engineering, scientific validity, maintainability, and demonstrable real-world usefulness — not for impressive-looking claims.

## Non-negotiable integrity rules

1. Never fabricate, estimate, invent, or manually improve experimental results.
2. Never claim that code, tests, training, evaluation, downloads, or benchmarks ran unless they actually ran successfully.
3. Never modify generated metric artifacts to improve results.
4. Never use final test data for hyperparameter selection, threshold tuning, augmentation selection, early stopping, or model choice.
5. Never hide failed experiments or known limitations when they materially affect a published conclusion.
6. Never copy another repository wholesale or remove attribution/license notices.
7. Prefer original integration and project-specific engineering over unnecessary duplication of library internals.
8. Respect dataset, model-weight, dependency, and source-code licenses.

## Git workflow

- Do not develop directly on `main`.
- Work on a clearly named feature, experiment, fix, or documentation branch.
- Keep commits focused and use descriptive conventional-style messages where practical.
- Before proposing a merge, inspect the diff and run applicable tests/lint/type checks.
- Never force-push published history unless the repository owner explicitly requests it.
- Never commit credentials, tokens, private keys, `.env` secrets, private data, restricted datasets, large model caches, or generated binary artifacts without explicit approval.
- Do not change repository visibility, security settings, branch protection, or licensing without owner approval.

## Engineering standards

- Python code should use clear module boundaries and type hints for public interfaces where practical.
- Prefer configuration over hard-coded experiment parameters.
- Use structured logging instead of scattered print statements in production paths.
- Validate external inputs and fail with actionable errors.
- Add tests for meaningful logic and regressions.
- Keep notebooks for exploration; production logic belongs in importable modules.
- Avoid giant scripts and unnecessary abstractions.
- Document non-obvious architecture decisions.
- Make GPU/CPU device behavior explicit and test graceful CPU fallback where appropriate.

## ML experiment standards

Every publishable run must be traceable to:

- dataset/version/category scope;
- code commit;
- configuration;
- random seed(s) where relevant;
- model/backbone/pretrained-weight identity;
- preprocessing and augmentation;
- threshold-selection procedure;
- environment/dependency information;
- hardware information for performance claims;
- machine-readable metrics;
- evaluation implementation or official-evaluator reference.

Do not put a metric in README.md merely because it appeared in terminal output.

## Data rules

- Follow `docs/DATASET_AND_EVALUATION_PROTOCOL.md`.
- Preserve official splits unless a documented experiment requires otherwise.
- Check custom splits for leakage/duplicates where possible.
- Never commit MVTec/VisA dataset assets unless their exact license and repository use explicitly permit that action and the owner approves it.
- Acquisition scripts/instructions must preserve required attribution and terms.

## Application standards

The final product must not be a notebook-only demo. It should expose a usable inspection workflow through an interface and a documented inference boundary/API. User-visible confidence/anomaly scores must be correctly defined; do not present arbitrary scores as calibrated probabilities.

## Security and privacy

Treat uploaded inspection images as untrusted input. Validate file type/size and avoid unsafe path handling. Never log secrets. Dependencies and containers should use sensible secure defaults.

## Agent behavior

Before a substantial task:
1. Inspect relevant repository files and existing decisions.
2. State or record a concise implementation plan when the task is non-trivial.
3. Implement the smallest coherent change that advances the task.
4. Run the appropriate verification.
5. Report exactly what changed, what ran, what failed, and what remains uncertain.

If a requested action conflicts with this contract or scientific integrity, stop and flag the conflict instead of silently complying.

## Definition of done

A feature is not done merely because code exists. It is done when applicable tests pass, documentation is updated, failure cases are considered, the change is reviewable, and claims are supported by evidence.