# Phase 3B EfficientAD execution contract

This document records the engineering and recovery policy fixed before the
first EfficientAD access to MVTec AD 2 `test_public`. It contains no benchmark
results. The scientific configuration remains the merged Phase 3A protocol
`efficientad-mvtecad2-v1`, fingerprint
`e9d6a66e7a52f2993e984ec20278c4ca4c710198cc466df15f947adff763f69f`.

## Authorization and boundaries

The benchmark CLI requires `--benchmark-mode`, an exact protocol fingerprint,
a clean Git tree, a passed dataset audit, the frozen package versions and
category/seed matrix, and the exact teacher and ImageNette archive hashes. Only
`train/good`, `validation/good`, and the authorized public image/mask split are
used. Private splits and the private evaluation server remain outside scope.

The implementation must be committed before execution. Its commit SHA, audit
identity, environment, teacher identity, and auxiliary-data identity are bound
into the manifest, checkpoints, and artifacts. Resume fails closed if any bound
identity changes.

## Durable series state

`visionguard-efficientad-benchmark` creates one atomic JSON manifest containing
all 24 category-major cells before training begins. A cell moves through
`pending`, `training`, `trained`, `calibration`, `evaluating`, and `completed`;
external interruptions and engineering/scientific failures are recorded
separately. Merely finding an output directory never counts as completion.

Manifest updates use write-then-replace semantics. Completed cells are skipped
only after revalidating the artifact, checkpoint, continuous TIFF maps, binary
PNG maps, and every recorded SHA-256 identity. Aggregation refuses a partial
matrix.

## Checkpoint and resume policy

Training checkpoints are written at step 0, every 1,000 completed optimization
steps, and the frozen final step 70,000. The 1,000-step interval is an
engineering durability choice fixed before public evaluation; it does not
alter optimizer updates or model selection.

Each atomic checkpoint preserves model, optimizer, scheduler, completed step,
Python/NumPy/PyTorch CPU and CUDA RNG states, deterministic train-image and
ImageNette stream positions, progress records, active-process time, and all
series identities. An interrupted training segment rolls back only to its
latest valid completed-step checkpoint. A pre-public engineering probe verified
that a short interrupted-and-resumed run produced the exact same final model
hash, loss checkpoints, learning rates, scheduler state, and final step as its
uninterrupted counterpart.

If interruption occurs after training while calibration or public map writing
is incomplete, the final training checkpoint is restored into a new immutable
attempt directory. Partial evidence remains in the prior attempt and is never
treated as complete. Missing, corrupt, hash-mismatched, or identity-mismatched
checkpoints stop the run rather than triggering a questionable continuation.

Ctrl+C is recorded as `interrupted`, not `failed`. Abrupt termination is
recognized from the durable in-progress state on the next `--resume`. A failed
cell remains visible and a later engineering-only retry starts a new attempt;
scientific settings are never changed to rescue a failure.

## Evidence and timing caveat

Every completed artifact contains validation-normal native quantiles,
validation-only operational thresholds, ordered public predictions, original-
coordinate map identities, all four frozen metrics, provenance, checkpoint
identity, attempts, interruptions, failures, and operational resource
observations. The final report is generated only after all 24 cells and their
hashes validate.

Training time is operational. Stopped-process intervals are excluded, while OS
sleep that occurs inside a checkpoint interval may be included. Consequently it
is not a formal speed measurement. No inference-efficiency claim is made unless
the separate frozen 1,000-warmup/1,000-timed-repetition procedure is actually
executed.
