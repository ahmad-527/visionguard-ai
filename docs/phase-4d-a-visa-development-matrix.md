# Phase 4D-A — completed normal-only VisA development matrix

All **72/72 development cells** are validated: PatchCore **36/36** and
EfficientAD **36/36**. The native supervisor finished on **2026-10-03 at
11:09:06 Berlin**, followed by the operational guard's successful-exit record
at 11:09:10. Both separate, fresh 36-cell CPU semantic/hash validation passes
completed before the full development freeze. Current process inspection at
publication confirmed no scientific worker/supervisor/guard remained running.

**This is development evidence, not test-performance evidence.** Only frozen
fitting normals, fitting-only EfficientAD normalization, and frozen calibration
normals were used. No VisA test performance was evaluated. The final-test lock
remained **CLOSED**; Phase 4D-B was **not started**. There is no model-selection,
hybrid-performance, generalization or production-performance claim here.

Human access history remains **UNKNOWN**; independent reservation remains
**NOT ESTABLISHED**. Dataset integrity, Git chronology and zero exact overlap
do not establish historical independence. A future test evaluation requires
separate human authorization and, by default, must be described as **held-out
VisA evaluation with historical access independence unverified**.

## Frozen identities and chronology

| Identity | SHA-256 / Git SHA |
| --- | --- |
| Pre-data scientific freeze | `c019f5515ee7692b42cda5bcc0595a4d3f00ec1e` |
| Reviewed Phase 4C regular merge | `7bd562d2e5642a3e919805cbcc633acc8a98b941` |
| Frozen execution implementation | `cca5e47ec8c3f68adfa6c90e17a17770b48049d0` |
| `visa-development-matrix-v1` execution fingerprint | `31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3` |
| `patchcore-visa-v1` fingerprint | `3ffcdc37cf3117d319da3e970383c6d0bdb52e2cd42dca87ad3161c5a9bc3191` |
| `efficientad-visa-v1` fingerprint | `78b27feeee044f560287a1ce45000800452344190e7b68003eefb3e2e853f1f9` |
| `visionguard-dual-model-triage-v1` fingerprint | `94442ab3121bccd392e3805b6134710cc6e8c95e8f17b8eccda288f8b1bd672d` |
| Dataset integrity audit | `06e227bb5d2cd26f38010b2c304c62f14f383a81c64f5b2e4c48f1019128f58f` |
| Development membership | `10e7a6c898fb18fbd1b93a115a5a94f3e2ebcecb7a1796d99d8c1e6f3567ce0b` |

The execution contract was committed before the first full cell. All bound
source/configuration hashes and scientific fingerprints reproduced unchanged
at publication. Documentation/evidence commits follow execution; they do not
replace the frozen implementation identity. No history was rewritten.

The Phase 4C merge retains parents `c5bc3f8269a2b507c57a6663360c2846f90922da`
and `86a027b5543ea7db709b1be3091cd8c63b7cf881`; the pre-data freeze remains an
ancestor. This PR must retain chronology: eventual **regular merge commit**
only, never squash/rebase merge. PR #20 remains unmerged for human review.

## Matrix, membership and validation

Execution used PatchCore first, then EfficientAD, in the category order below,
with seeds **42, 123, 2026** within each category. One GPU workload ran at a time.
No category/seed was dropped or selected, no thresholds were rescued, and no
settings changed because of runtime, calibration values or failures.

| Category | Frozen fit normals | Frozen calibration normals | PatchCore cells | EfficientAD cells |
| --- | ---: | ---: | ---: | ---: |
| candle | 810 | 90 | 3/3 | 3/3 |
| capsules | 488 | 54 | 3/3 | 3/3 |
| cashew | 405 | 45 | 3/3 | 3/3 |
| chewinggum | 408 | 45 | 3/3 | 3/3 |
| fryum | 405 | 45 | 3/3 | 3/3 |
| macaroni1 | 810 | 90 | 3/3 | 3/3 |
| macaroni2 | 810 | 90 | 3/3 | 3/3 |
| pcb1 | 814 | 90 | 3/3 | 3/3 |
| pcb2 | 811 | 90 | 3/3 | 3/3 |
| pcb3 | 815 | 90 | 3/3 | 3/3 |
| pcb4 | 814 | 90 | 3/3 | 3/3 |
| pipe_fryum | 405 | 45 | 3/3 | 3/3 |

The reviewed exact memory-safe PatchCore path preserved ordered float32
embeddings and unchanged coreset semantics. Each EfficientAD cell finished
exactly **70,000 optimization steps**, then fit-only normalization and normal
calibration. Completion required valid origin, attempts, checkpoint/model
hashes, finite tensors, calibration membership/order, frozen maximum-order
image and pixel thresholds, and durable manifests. Calibration thresholds are
development artifacts, not accuracy estimates. Strict `score > threshold`
semantics remain unchanged; equality is normal.

The separate model-wide validation passes bind all 36 fresh CPU receipts each:

- PatchCore summary: `f64c2dc9b18a5995196e1f6c12690eb0637aeb1fcc8ba11c0e04bfdb9ad1e9c1`.
- EfficientAD summary: `b1c12497712a3f0a7806145e547a2bab9754f7cd55531ab2d40e46c6e9b4e72a`.

Publication rechecked every frozen receipt and every fresh-pass receipt hash,
exact final-model identities, counts, final steps, and closed-test boundaries.
It did not rerun inference, training, calibration, or archive/test decoding.

## Failures and human-reviewed operational recovery

There were **75 attempts**: 72 completed, **2 failed**, **1 interrupted**.
Final matrix rows show no unresolved failed/interrupted cells; this must not
be mistaken for an absence of historical failures. All original logs, failure
states, snapshots, metadata and receipt hashes remain local and hash-bound.

1. **EfficientAD cashew:42 attempt 1 — FAILED.** Windows WinError 5 denied
   checkpoint-metadata replacement. The last published metadata was step
   45,000 while an independently verified unpublished payload was step 46,000.
   That unpublished state was **not used for resume**. Human review authorized
   a fresh attempt 2 through the unchanged dispatcher, starting at step zero.
2. **Cashew:42 attempt 2 — FAILED.** WinError 5 denied checkpoint-payload
   replacement near step 34,000. Only the published step **33,000** checkpoint
   was later authorized for resume. OneDrive's client had exited, yet the
   failure recurred. File-lock ownership was not proven; OneDrive is not a
   demonstrated root cause, and migration does not prove filters disappeared.
3. **Human-approved runtime copy and cashew attempt 3.** An ordinary local NTFS
   runtime outside OneDrive received byte-for-byte copies; the original
   checkout remained a read-only forensic fallback. Active source/destination
   inventory equality covered **846 files / 16,603,695,807 bytes**, SHA
   `dbb714f307a99e7861e1d3ea84706b2e887088d90b37a6f472df7a50498d00ab`.
   Destination validation checked all **42 prior cells** and the published
   33,000 parent. Attempt 3 restored that exact state, independently validated
   publications at **34,000 and 35,000**, then completed 70,000, normalization
   and calibration. Only reviewed operational matrix state was repaired;
   the native supervisor independently rediscovered completion. No cell was
   manually marked complete. Neither failed attempt was overwritten.
4. **EfficientAD pcb1:42 attempt 1 — INTERRUPTED.** On 2026-10-02 around 01:43
   Berlin, current process inspection confirmed the worker, supervisor and
   old guard had disappeared. Termination cause and actual exit code remain
   **UNKNOWN**. Boot identity was unchanged; inspected Windows events did not
   establish a crash/hardware cause. This absence of events is not proof of
   stability or an intentional termination. The required stop was preserved.
5. **Explicit human-authorized pcb1 continuation.** After forensic snapshots,
   published step **39,000** passed identity, deserialization, finite-state,
   optimizer/scheduler, RNG, fitting/ImageNette-stream and cadence checks.
   Narrow state repair reproduced existing frozen finalizer binding semantics
   and terminalized the original attempt as interrupted with an unknown-exit
   reason, without inventing a return code. Reviewed abandoned locks were
   moved byte-for-byte only after process absence was independently verified.
   New immutable attempt 2 restored 39,000; **40,000 and 41,000** publications
   passed validation. It completed exactly 70,000 plus native normalization
   and frozen calibration. All negative history and both cashew failures
   remained intact; original matrix stop and reviewed recoveries remain bound.

Only ignored launch/import paths and reviewed operational state changed during
these recoveries. Scientific source, settings, threshold estimator, cadence,
model/optimizer/RNG payloads, execution contract and test boundary did not.
No Defender exclusion, automatic hardware retry, or source patch was applied.
OneDrive was not restarted. The unknown process-loss cause remains an unresolved
operational concern despite successful subsequent completion.

## Storage, timing and resource evidence

The frozen retention policy compacted completed, validated PatchCore embedding
chunks in **36 cells**, preserving inventory/ordered-embedding/coreset/bank
hashes and per-cell compaction receipts. Sole resumable or unvalidated state
was not discarded. Large models, checkpoints, maps, telemetry and forensic
snapshots remain ignored/local; compact reports only are committed.

At publication, the matrix-output subtree retained **21,941,903,734 bytes**
(20.435 GiB). This excludes separate dataset/auxiliary assets, migration and
recovery snapshot directories, and the preserved original checkout. Wall time
from the original matrix start to final completion was **337,645.039824 s**
(93.790 h), including pauses/recovery/validation. Sum of recorded model-specific
fit/training timers was **312,231.4606580007 s** (86.731 h); these timers can
include OS sleep and exclude normalization/calibration. They are not a complete
accounting of all failed work or end-to-end compute, and not a speed comparison.

Hardware: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8,192 MiB, driver 561.17;
Windows, Python 3.11.6, Torch 2.9.1+cu126, CUDA 12.6, Anomalib 2.6.0. The
execution contract binds the complete package and pretrained/auxiliary identities.

Across recorded 60-second native/guard samples, maximum observed GPU memory
was **7,882 MiB**, maximum observed temperature **78 C**, minimum observed free
disk **110,722,736,128 bytes**; recorded WHEA events since start remained zero.
These are **sampled observations, not true peaks**; guard streams overlap and
must not be summed as independent observations. Per-cell Torch allocated/
reserved peaks and host-RAM peaks were not recorded by the frozen workers.
Largest recorded PatchCore embedding payload was **5,127,536,640 bytes**
(815 fitting images, float32); exact shapes/bank/ordered hashes are preserved
in the resource evidence. Constrained GPU headroom and unresolved file-lock/
process-loss causes remain risks for future execution.

## Published evidence and reproduction

Under `reports/phase4d-a-visa-development-matrix/`:

- `development-freeze.json`: all 72 final-model/canonical-model/calibration
  hashes, threshold values, normalization, inventories, counts, attempt and
  wall histories, storage and immutable execution contract. SHA
  `d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061`.
- `development-summary.json`: exact counts, historical attempt totals, storage
  and timing, bound to the published freeze.
- `publication-validation.json`: both fresh model-pass receipt inventories and
  the metadata-only final publication checks.
- `recovery-evidence.json`: compact sanitized derivatives binding original
  forensic/migration/recovery receipts by **raw SHA-256**, including exact
  parent/early-publication and pre/post matrix-repair hashes. Root paths are
  replaced by declared placeholders, never original local bytes. It is not a
  byte-identical copy of the raw receipts.
- `resource-evidence.json`: per-cell recorded PatchCore fields and hash-bound
  telemetry sample extrema, with explicit missing-measurement limitations.
- Existing `preflight-validation.json`: reused reviewed Phase 4C acceptance.

Full local development-freeze SHA:
`72d9731049b6cfad4dcad38ce3ed5ee37ed8c2b661b9aa60d0b4223bd934f940`.
Final matrix-manifest SHA:
`b688c695d8bb0f62763ae0bcfc3c9ed6aba4ceb9450c63afb1e6e50bb6ca5419`.

Publication used the **unchanged** `scripts/phase4d_a_publish.py` only after
successful exits, 72-cell validation, both fresh model passes and a valid full
freeze. Its local reproduction command, from a checkout matching the frozen
source/config hashes with the completed ignored artifacts available, is:

```powershell
.\.venv\Scripts\python.exe scripts/phase4d_a_publish.py --root outputs/phase4d-a-visa-development-matrix --repository .
```

This command verifies metadata/hash bindings and publishes evidence; it does
not train or evaluate. Do not invoke the matrix `--run` to reproduce a report.
An evidence/docs commit is not a new scientific execution authorization.

## Final engineering verification and handoff

Final unchanged-source pytest: **512 passed, 1 skipped** (Windows symlink
privilege unavailable). Ruff format (107 files), Ruff lint, pip check,
protocol reproduction, clean imports/development CLI checks passed. Final
provenance, test-lock/root-isolation tests are included in that suite. Published
JSON is finite, compact and path-sanitized; secret/local-path/large-file and
Git whitespace/diff checks passed before commit. CI for Python
**3.11 / 3.12 / 3.13** is checked on PR #20; its current status is authoritative
there rather than inferred from previous successful runs.

No frozen scientific or execution code/configuration was changed. This is an
evidence/documentation handoff only. Review the completed freeze and preserved
negative history before any next-phase authorization. The [Phase 4D-B plan](planning/phase-4d-b-heldout-visa-evaluation.md)
remains **DRAFT — NOT AUTHORIZED FOR EXECUTION**. Unknown human access history,
unestablished independence and unresolved operational causes remain visible.
No VisA test performance was evaluated; Phase 4D-B was not started; test lock
remained CLOSED; PR remains unmerged and commit history was not rewritten.
