# Phase 4D-A — normal-only execution contract

This phase authorizes the full **development** matrix, not held-out evaluation.
The final-test lock remains **CLOSED**. No Phase 4D-B evaluator is implemented.
Human access-history declaration is **UNKNOWN**; independent reservation is
**NOT ESTABLISHED**. Training/calibration do not resolve that uncertainty.

Base: reviewed regular merge `7bd562d2e5642a3e919805cbcc633acc8a98b941`, with
parents `c5bc3f8269a2b507c57a6663360c2846f90922da` and
`86a027b5543ea7db709b1be3091cd8c63b7cf881`. The pre-data freeze
`c019f5515ee7692b42cda5bcc0595a4d3f00ec1e` remains an ancestor.

## Freeze before execution

The [machine contract](../configs/execution/visa-development-matrix-v1.yaml)
binds exact scientific identities, source files, environment, assets, category/
seed/cell order, checkpoint and retention policies, disk budget and governance.
Its fingerprint is SHA-256 of sorted compact JSON of the `execution` object.
The containing commit must precede the first full cell. The supervisor requires
a clean checkout at that contract commit throughout execution. No source edits,
mixed implementations, automatic code fixes or scientific rescue during the run.
If a change becomes necessary, stop and request human review of a new boundary.

Scientific model/triage YAMLs and their fingerprints remain unchanged. All
reviewed fitting, normalization, calibration, worker and dispatcher sources are
unchanged. New code only supervises, independently validates and compacts
completed-cell temporary storage. Phase 4C hash-bound exactness and restart
evidence is reused, not expensively rerun. CPU-only preflight applies the new
validator to retained uninterrupted and restarted normal-only smoke artifacts.

## Fixed schedule and cell definition

PatchCore first, then EfficientAD; each uses candle, capsules, cashew,
chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum;
within each category seeds 42, 123, 2026. Exactly 36 cells per model, 72 total.
No runtime/threshold/failure-based reordering. A failed cell stops the schedule.

PatchCore uses all frozen fitting normals, exact durable float32 chunks,
unchanged coreset, and all category calibration normals. EfficientAD uses the
same frozen fitting membership, PDN-S/teacher/ImageNette, exactly 70,000 steps,
final-step selection, fit-only native normalization and all calibration normals.
Each threshold is the unchanged maximum-order statistic; strict `score > threshold`.
There are no labels, test samples or test metrics in this workflow.

Process exit does not establish matrix completion. A fresh CPU validation
process checks attempt origins/history, every referenced artifact hash, canonical
final/fit/normalized tensor state and finiteness, full ordered calibration IDs/
shapes/counts and exact threshold reproduction. PatchCore adds ordered embedding
identity, coreset indices, memory-bank shape/hash. EfficientAD adds final 70,000
step, scheduler epoch, optimizer finiteness, RNG/streams, final-step model
identity, normalization source and quantile/model agreement. Validation reads
only metadata and development checkpoints, never test assets or images.

The supervisor atomically records matrix completion only after validation.
After each model, fresh independent validation of all 36 cells precedes the
next model. A `development-freeze.json` can be emitted only at validated 72/72.
Partial status cannot be described as a complete scientific matrix.

## Durable supervision and restart

The long-lived supervisor holds no CUDA context. It launches the unchanged
single-cell dispatcher sequentially; identity inspection and validation run in
short-lived processes. The existing repository GPU lease permits one worker.
Atomic matrix metadata, per-attempt logs, immutable worker attempts and receipts
persist independently of the Codex conversation. One log per attempt and
one-minute telemetry avoid unbounded per-step model logging.

PatchCore commits each completed image chunk with exact sample order, RNG and
hashes; interrupted coreset computation restarts from completed embeddings with
the original RNG. EfficientAD retains the reviewed **1,000-step cadence**, initial
checkpoint and final step; one latest checkpoint per attempt includes optimizer,
scheduler, Python/NumPy/Torch/CUDA RNG and both streams. No best-checkpoint path.

Classified ordinary worker interruptions resume into a new immutable attempt
automatically, after existing checkpoint validation. Missing/corrupt identity,
failed validation, abnormal worker failure or hardware symptoms stop scheduling.
An abruptly lost supervisor, unexpected reboot or stale lock is ambiguous and
fails closed: inspect recorded PIDs/boot/origin before recovery; never infer that
processes stopped merely from absent logs or delete a lock automatically.

Create the exact ignored file `outputs/phase4d-a-visa-development-matrix/PAUSE_AFTER_CELL`
to request a safe between-cell pause; remove only that requested marker to resume.
This does not change code/science. Abrupt interruption may lose work since the
last valid native checkpoint, never silently change sample/RNG order.

## Resource and storage policy

The machine contract records current measured free space and conservative
additional budgets. Existing archive/extracted sizes come from the bound audit;
development-copy bytes come from normal membership only. No sealed-test walk,
decode or archive rehash is needed. Asset hashes and ImageNette training-file
inventory are bound; extracted auxiliary bytes are checked at supervisor startup.

Rounded ceilings: 8 GiB active PatchCore space; per completed PatchCore cell,
three 160 MiB model copies plus 16 MiB metadata/normal-map allowance; per
EfficientAD cell, three 40 MiB models, 96 MiB resumable checkpoint and 16 MiB
metadata/map allowance. Reserve another 20 GiB for interruptions and maintain
**20 GiB minimum free space**. This is a conservative planning budget, not a
model metric or permission to delete evidence under pressure.

Check free space before/after each cell and validation/compaction, plus every
60 seconds during computation (including reconstruction). Resource telemetry
records NVIDIA driver/model/temperature/used-memory observations, system boot,
WHEA event check and free disk. NVIDIA used memory is not Torch allocated/reserved
peak; those peaks are unavailable from the unchanged dispatcher. Retain Phase 4C
measured peaks as prior engineering evidence, not new cell measurements. No
overclocking, Windows power changes or synthetic stress tests. Any CUDA/OOM,
WHEA event, reboot, filesystem/evidence error or disk-pressure breach stops the
matrix; do not automatically retry hardware failures.

### Completed-cell compaction

Only independently validated, durably completed PatchCore cells may delete
their exact hash-verified embedding `.pt` chunks. A durable `prepared` receipt
precedes deletion; a `completed` receipt follows it. Keep the full immutable
chunk journal/origin, ordered-embedding hash, membership identity, coreset
indices/hash, bank/model hashes, calibration/threshold evidence and byte counts.
The completion manifest must be unchanged and no cell writer active. Paths are
resolved and constrained to that category/seed's attempt chunk directories;
never recursively delete directories or touch final models. Interrupted
deletion can continue only against the same durable inventory/intent.

Active/unvalidated cells and the sole recovery state of interrupted cells are
never compacted. Historical Phase 4C/failed attempt artifacts are not cleaned.
Final models and all failed/interrupted metadata are retained. Deletion is not
trash-recoverable; exact embedding recomputation requires retained inputs and
frozen code. If the frozen budget is insufficient, stop rather than invent a
new cleanup policy. Synthetic tests exercise corruption, missing files,
unsafe paths, active state, stale locks and interrupted compaction.

## Commands and handoff

Pre-execution local QA: **512 passed, 1 skipped** (Windows symlink privilege),
Ruff format/lint and `pip check` passed. Protocol fingerprints reproduced;
CPU-only validation of both models' retained uninterrupted/restarted Phase 4C
artifacts passed. The portable preflight receipt binds those input manifests.
Secret/local-path/large-file and whitespace checks passed. CI must pass before
launch. These are engineering checks, not new model-performance evidence.

Execution fingerprint:
`31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3`.
Measured preflight free space was 161,476,280,320 bytes; the conservative
additional-space budget is 79,020,687,360 bytes. The launcher rechecks the
frozen free-space requirement before execution.

Only after committed freeze and passing QA/CI:

```text
python -m visionguard.visa_matrix_contract
python -m visionguard.visa_matrix --root outputs/phase4d-a-visa-development-matrix --status
python -m visionguard.visa_matrix --root outputs/phase4d-a-visa-development-matrix --run --local-config outputs/phase4d-a-preflight/local-settings.json --execution-fingerprint FROZEN_FINGERPRINT
```

Machine-local paths remain in ignored settings. A hidden, detached Windows
process may run this exact command with stdout/stderr captured to ignored files;
it must not rely on a live Codex turn. Keep the checkout unchanged. Publish
compact portable status/evidence on the draft PR; do not commit checkpoints,
embeddings, maps, request files with local paths or full log directories.

Record wall times per process and native active fit/training time separately.
Native timers may include OS sleep; successful PatchCore fit time excludes
calibration and previous interrupted work. Do not present these as throughput
comparisons. Matrix summaries distinguish pending/active/interrupted/failed/
completed cells. Final reporting includes all negative history and limitations.

No VisA performance, model comparison, independent validation, hybrid benefit
or production claim is allowed. Phase 4D-B remains a separately reviewed and
authorized future task. This PR must not be merged by the agent.
