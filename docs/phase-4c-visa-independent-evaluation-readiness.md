# Phase 4C — VisA independent-evaluation readiness

Status: **protocols frozen for review; integrity audit and normal-only smoke
passed; confirmatory execution NOT authorized. Operational readiness remains
conditional on PatchCore memory capacity and the historical-access declaration.**

No VisA final-test predictions, AUROC, F1, AU-PRO, triage metrics, example panels,
or qualitative anomaly inspection were produced. Test assets were decoded only
for the explicitly authorized integrity audit. No full 70,000-step training,
36/72-cell matrix, MVTec private access, or Phase 4D execution occurred.

## Review and scientific chronology

- Phase 4B PR #16 was verified merged. Actual base/main anchor:
  `c5bc3f8269a2b507c57a6663360c2846f90922da`.
- The reviewed triage fingerprint was recomputed, not copied from the request.
- PRE-DATA design was committed at
  `c019f5515ee7692b42cda5bcc0595a4d3f00ec1e`, before fetching split rows or
  acquiring/inspecting dataset assets. See [pre-data freeze](phase-4c-predata-freeze.md).
- Verified integrity identities were subsequently bound into the model protocols
  at `6ba4f7e7eace64c5f57ec7f58a0f66057419f15f`, before the smoke fit.
- The successful smoke recorded that exact commit and a clean worktree.
  All scientific settings remain as predeclared. Engineering failures remain
  documented below; they did not trigger scientific tuning.

**MVTec AD 2 `test_public` remains exposed development evidence.** Phase 4A
motivates this hypothesis but validates neither hybrid nor router. Replaying that
split could only be retrospective development analysis, not independent evidence.
VisA independence is also a claim requiring an access-history declaration: no
such human declaration has yet been received in this phase. The audit does not
prove that no contributor previously inspected VisA outcomes. Do not label a
future run confirmatory until a custodian resolves this gate.

## Authoritative acquisition and licensing

Scientific source: [amazon-science/spot-diff, pinned commit](https://github.com/amazon-science/spot-diff/tree/2a692ab575001cbde74d402d897a7286086c6199),
`2a692ab575001cbde74d402d897a7286086c6199`.
The [official README](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/README.md)
identifies the release; [LICENSE-DATASET](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/LICENSE-DATASET)
is CC BY 4.0. The official documentation and license agree. An installed
Anomalib VisA docstring names a different license; it is a secondary-documentation
discrepancy, not the license authority. Its archive checksum agrees with the
observed official download. Attribution and the upstream license still apply;
this PR redistributes no images, masks, archive, weights, or auxiliary assets.

The official S3 release was downloaded once after checking local candidate
locations and available disk space (180,932,112,384 bytes at preflight).
It is retained under ignored `data/visa/`; extraction was not duplicated.

| Identity | Observed value |
|---|---|
| Release | `VisA_20220922.tar` |
| Official source | [Amazon S3 release](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) |
| Archive bytes | 1,929,840,640 |
| Archive SHA-256 | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| Official `split_csv/1cls.csv` SHA-256 | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |
| Audit SHA-256 | `06e227bb5d2cd26f38010b2c304c62f14f383a81c64f5b2e4c48f1019128f58f` |
| Development membership SHA-256 | `10e7a6c898fb18fbd1b93a115a5a94f3e2ebcecb7a1796d99d8c1e6f3567ce0b` |

The [acquisition record](../reports/phase4c-visa-readiness/acquisition.json)
binds URLs, exact upstream Git blobs, SHA-256, and sizes for the split,
`utils/prepare_data.py`, `utils/id2class.py`, README, and dataset license.
Fetched bytes were checked against the pinned upstream Git objects; downloaded
Python was parsed as data where necessary, never executed. The split contained
in the archive was independently checked against the fetched official CSV.

## Integrity audit, not performance analysis

[Audit summary](../reports/phase4c-visa-readiness/audit-summary.json): **passed**.
All 10,821 images and 1,200 anomaly masks decoded; 12 expected categories were
present. All referenced assets existed, original dimensions were valid, anomaly
masks were nonempty and aligned, and mask label values matched the official
category semantics. No missing/orphan assets, malformed rows, or path collisions
were detected. Test normals have no required anomaly-mask file; future authorized
evaluation would represent their ground truth as an all-zero mask.

Exact image identity was checked twice: encoded file SHA-256 and decoded RGB
pixel SHA-256 with mode and original dimensions bound into the digest. Both
methods found **zero** within-train duplicate groups, within-test duplicate
groups, train/test overlap groups, and fit/calibration overlap groups. Duplicate
mask groups were also zero. No near-duplicate diagnostic was run; cryptographic
equality does not exclude visually similar products or all semantic leakage.

The full ignored inventory is bound by SHA-256
`4c16ebb13a1546399a28dc1f41aebeb5dfb062ab0c7c9108cc5bfb541c94ccb2`.
Committed reports contain category integrity counts, not test filenames/labels.

| Category | Fit normals | Calibration normals | Test normal | Test anomaly |
|---|---:|---:|---:|---:|
| candle | 810 | 90 | 100 | 100 |
| capsules | 488 | 54 | 60 | 100 |
| cashew | 405 | 45 | 50 | 100 |
| chewinggum | 408 | 45 | 50 | 100 |
| fryum | 405 | 45 | 50 | 100 |
| macaroni1 | 810 | 90 | 100 | 100 |
| macaroni2 | 810 | 90 | 100 | 100 |
| pcb1 | 814 | 90 | 100 | 100 |
| pcb2 | 811 | 90 | 100 | 100 |
| pcb3 | 815 | 90 | 101 | 100 |
| pcb4 | 814 | 90 | 101 | 100 |
| pipe_fryum | 405 | 45 | 50 | 100 |
| Total | 7,795 | 864 | 962 | 1,200 |

These are inventory facts, not detection/localization results.

## Frozen fit/calibration allocation

For each category's official training normals, sort by the hexadecimal SHA-256
of UTF-8 `visionguard-visa-development-v1\0{category}\0{official_POSIX_image_path}`;
break any hash tie by exact lexical image path. The first `floor(n / 10)` are
calibration; all remaining normals are fit. The minimum calibration size is 19.
Observed counts are 45–90, so no fraction/minimum adjustment was necessary.
The same membership applies to both models and all three seeds. No test normal
was borrowed. No runtime random split, best seed, or per-category rescue exists.

The [membership manifest](../reports/phase4c-visa-readiness/development-membership.json)
contains every permitted normal's portable ID, role, category, dimensions, file
hash and decoded hash. Tests reproduce the allocation from these IDs. Separate
normal-only copies are in ignored `data/visa/development`; the adapter requires
the exact manifest and a matching root seal, verifies image bytes before use,
and rejects test roles and label/mask metadata.

## Model protocol identities and differences

| Protocol | Canonical SHA-256 |
|---|---|
| `patchcore-visa-v1` | `3ffcdc37cf3117d319da3e970383c6d0bdb52e2cd42dca87ad3161c5a9bc3191` |
| `efficientad-visa-v1` | `78b27feeee044f560287a1ce45000800452344190e7b68003eefb3e2e853f1f9` |
| `visionguard-dual-model-triage-v1` (unchanged) | `94442ab3121bccd392e3805b6134710cc6e8c95e8f17b8eccda288f8b1bd672d` |

New YAMLs under `configs/protocols/` inherit the validated parent scientific
configuration, bind the pre-data design, source/audit/membership, allowed seeds,
implementation source bytes, preprocessing, calibration, metrics, artifact
contract, failure policy, and closed test gate. Canonical sorted compact JSON of
the entire YAML document is hashed; comments, whitespace, and key ordering do
not change identity. `visionguard-visa-protocol` verifies the checked-in freeze,
evidence hashes and implementation sources without loading models or data.
Hashes detect drift; they do not grant human approval.

PatchCore retains Anomalib 2.6.0, Wide-ResNet50-2, layer2/layer3, 0.01 coreset,
nine neighbors, pooling/smoothing, ImageNet normalization, bilinear antialiased
256×256 resize and original-coordinate map restoration. The verified pretrained
weight SHA is `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000`.
The smoke also compared every loaded backbone tensor with that frozen file.

EfficientAD retains PDN-S, the reviewed teacher and ImageNette archive identities,
optimizer, learning rate, weight decay, scheduler, 70,000 steps, deterministic
controls, preprocessing and restoration. **The one pre-data scientific adaptation
is normalization source:** fit its unchanged 0.9/0.995 map quantiles on fitting
normals after final training, not the threshold calibration holdout. This keeps
calibration normals out of all model/normalization fitting. It is not a
test-outcome-driven normalization adjustment or cross-model score normalization.
Dataset/split names and metric threshold-source terminology change structurally.

For each model, the image threshold is the maximum calibration-normal image
score; the pixel threshold is the maximum of calibration-normal per-image map
maxima. Anomalous means strictly `score > threshold`. Equality is normal.
Threshold values for the future scientific models do not yet exist; only their
estimator is frozen. The tiny smoke's thresholds MUST NOT be reused as scientific
thresholds. No operational false-positive guarantee is claimed.

## Future metrics and triage

Use all 12 categories × seeds 42, 123, 2026: 36 cells/model, 72 total. Pair only
the same seed and same ordered image IDs. Keep own-model score scales and
thresholds. Both normal → PASS; both anomalous → REJECT; either disagreement →
REVIEW. REVIEW is never forced to binary correctness. Keep both continuous and
frozen binary model maps; no union/intersection, fusion, morphology or routing.

Future model ranking metrics: image AUROC and AU-PRO@0.05 with the inherited
8-connected-region definition. Future threshold metrics: TP/FP/TN/FN, recall,
specificity, precision, image F1, pixel F1 at each model's frozen threshold.
These are prespecified VisionGuard measures, not an official VisA leaderboard
claim. Older inherited `official_*` key names do not change that interpretation.

Use the unchanged, source-hash-bound Phase 4B reducer and
[metric/reporting definitions](phase-4b-hybrid-triage-protocol.md#frozen-future-metrics):
review rate, automatic coverage, selective accuracy, anomaly pass-through,
normal rejection, anomaly review capture, normal review, reject precision,
pass NPV, class-conditioned three-way counts, and separate disagreement
directions. Report every category/seed, pooled counts per seed, unweighted
category macro rates, and mean/sample-SD across seeds. Preserve denominators and
undefined rates; never treat repeated seed observations as independent samples.
No weighted winner score, numeric acceptance target, or new significance test.
Deployment requires a separately specified operational risk/review-workload
preference. No partial matrix may be called complete.

## Engineering smoke and honest failure record

[Machine-readable smoke receipt](../reports/phase4c-visa-readiness/smoke-summary.json)
projects the retained ignored raw report and binds all emitted artifact bytes.
Fixed category candle, seed 42, first four lexical fit normals and first 19
lexical calibration normals, as declared before data. EfficientAD ran **two**
optimization steps, without changing the scientific 70,000-step scheduler.

- Attempt 1 stopped at Git provenance preflight: passing relative repository `.`
  produced a nonmatching Git `safe.directory` exception in the sandbox. No model
  fit happened; the empty attempt directory remains. Using the resolved absolute
  repository argument fixed this invocation issue without changing source or science.
- Attempt 2 passed on the clean freeze commit: PatchCore fit/coreset,
  in-process embedding checkpoint round-trip, EfficientAD intentional stops at
  steps 1 and 2 with reconstruction/restoration, fit-only normalization,
  normal-only calibration, per-model map writer, final checkpoint hashing, and
  same-normal/same-seed triage interface compatibility.
- Measured smoke body wall time: 7.05 seconds, excluding dependency imports,
  asset-hash preflight, and provenance preflight. Peak allocated CUDA:
  PatchCore 189,838,848 bytes; EfficientAD 416,245,248 bytes. Artifacts: 126,371,944
  bytes. These are tiny plumbing observations, not throughput or performance claims.
- The optional 1,000-step probe was not run. Existing 24-cell resource history
  already informs capacity; two steps are not a reliable steady-state estimate.
- No CUDA instability occurred. Only one GPU workload ran. Checkpoint restoration
  was exercised; crash/power-loss durability and full-category memory were not.

## Test seal and infrastructure limits

Ordinary development uses only the normal-only manifest/root. The smoke installs
a Python file-open audit guard against the sibling sealed root. Tests exercise
the guard without real data. This is workflow isolation, **not an OS security
boundary against a malicious process/native extension**; a future evaluation
custodian should provide a separate account or filesystem permissions.

`visionguard-visa-plan` verifies the reviewed protocol and audit, reports the
36-cell plan, and launches neither training nor evaluation. Even matching
`--confirm-independent-test-evaluation`, `--protocol-fingerprint` and
`--dataset-audit-sha` cannot unlock Phase 4C. A later reviewed authorization
implementation is required. There is deliberately no final-test evaluator here.

`ExecutionState` provides atomic manifest publication, exclusive writer locks,
per-attempt origins and terminal immutability, stage history, required identity
fields, artifact/checkpoint hashes, validated skip-completed, and separate failed
versus interrupted status. Source/model/protocol/audit/membership/environment
must agree for resume. Old attempts are not overwritten. A stale lock requires
inspection. The EfficientAD smoke reuses the proven Phase 3B optimizer/scheduler/
RNG checkpoint code without changing the old runner. A production full-matrix
dispatcher and PatchCore cross-process resume integration still need review and
engineering completion before authorization; generic state tests alone do not
certify those paths. No claim of full execution readiness is made.

## All-category capacity: measured versus extrapolated

[Capacity evidence](../reports/phase4c-visa-readiness/capacity-evidence.json) is
reproducible with `scripts/phase4c_resource_evidence.py`. It reads only resource
fields of hash-verified existing MVTec artifacts and VisA integrity dimensions,
never predictions/metrics. It does not benchmark VisA.

| Quantity | Evidence and interpretation |
|---|---|
| PatchCore previous 24 cells | Whole-cell wall times: min 60 s, median 115 s, max 12,869 s; includes nontraining work and interruptions/operational variation, not model speed |
| EfficientAD previous 24 cells | Active 70k-step training: min 2.41 h, median 3.49 h, max 6.80 h; excludes pauses and later evaluation |
| Naive 36-cell EfficientAD sensitivity | 87–245 active GPU hours (median-based 125 h); new data loading and thermals remain unknown |
| Naive 36-cell PatchCore sensitivity | 0.6–129 whole-cell hours; NOT a credible VisA prediction given changed fit counts/memory |
| Combined planning envelope | Roughly 88–374 active/whole-cell hours (4–16 days continuously), plus pauses/evaluation; historical sensitivity, not a confidence interval or guaranteed schedule |
| Maps, both models × 3 seeds | 3,243,420,552 audited test pixels × 18 bytes = 58,381,569,936 bytes (~54.4 GiB) uncompressed float16+uint8 payload; PNG compresses, TIFF/container overhead adds |
| EfficientAD 36 resumable checkpoints | ~2.71 GB from historical checkpoint sizes, before extra immutable attempts/final copies |
| PatchCore 36 memory banks | ~1.47 GB float32 payload from the frozen coreset ratio, excluding backbone and temporary embeddings |

Reserve **at least 100 GiB additional free storage** as an operational planning
budget, not a scientific threshold: maps/checkpoints plus temporary embeddings,
metric scratch, interrupted attempts, and headroom. Inputs/archive and normal
copies are retained separately. A second full map copy can exceed this budget;
review retention before execution. CPU metric time, host peak RAM, new storage
compression and end-to-end runtime remain unknown. Training and CPU evaluation
should run sequentially until measured peak resource safety is established.

**Current 8 GiB GPU blocker:** the largest fitting category has 815 images.
At 1,024 patches × 1,536 float32 features, stored embeddings alone require
5,127,536,640 bytes. Anomalib stacks the live list before clearing it, so list
plus stacked tensor needs at least 10,255,073,280 bytes (~9.55 GiB), excluding
weights, activations and coreset working state. This exceeds available VRAM.
No full fit was attempted. Smaller categories/ratios/resize or fewer EfficientAD
steps are NOT authorized remedies. Human review must choose sufficiently large
hardware or a separately reviewed memory-safe implementation retaining scientific
semantics and validating numerical/resume behavior before test access.

## Reproduction, QA, and stop conditions

Install the lightweight project with `python -m pip install -e ".[analysis,dev]"`.
Imports, protocol checks, synthetic tests and CLI help require no dataset or GPU.
CI builds clean Linux environments on Python 3.11, 3.12 and 3.13. Scientific ML
execution retains the parent Python 3.11/3.12 and pinned ML dependency contract.

Local final checks: 446 tests passed; one existing Windows symlink-privilege test
skipped. Ruff format/lint, `pip check`, editable installation, installed CLI help,
protocol reproduction, and Git whitespace checks passed. Repository-wide secret
pattern and absolute-local-path scans had no matches; no tracked/pending file
exceeded 5 MiB. This pattern scan is not a guarantee against every secret format.
Dataset/model binary assets remain ignored. GitHub CI is the separate clean-
environment check; its observed status is recorded on the PR, not inferred from
local tests.

```text
visionguard-visa-protocol
visionguard-visa-audit --help
visionguard-visa-plan --help
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m pip check
```

Acquisition is an explicit command, never an import side effect. For a fresh
authorized setup, see the CLI help for `visionguard-visa-acquire`; it verifies
source Git blobs/archive bytes, rejects unsafe tar entries and refuses partial
extraction reuse. To reproduce the integrity audit (no model execution):

```text
visionguard-visa-audit --local-root data/visa --acquisition reports/phase4c-visa-readiness/acquisition.json --output reports/phase4c-visa-readiness/audit-summary.json --inventory data/visa/audit-inventory.json
```

Avoid rerunning unchanged expensive audit/download work. Re-audit if input bytes
change or provenance cannot be validated. The documented hashes bind this exact
state, not arbitrary replacement files. Never hand-edit evidence to make it pass.

Stop for source/license identity conflicts, unexpected overlap, bad masks,
inadequate calibration, unreservable independence, required test-driven choices,
hardware failure, altered triage, or any proposed scientific scope change.
Outstanding human gates: historical access declaration; a feasible all-category
PatchCore execution platform/implementation; complete dispatcher/resume review;
reviewed artifact retention and operational risk preferences; explicit Phase 4D
authorization. See the [unauthorized draft plan](planning/phase-4d-visa-confirmatory-execution-plan.md).
