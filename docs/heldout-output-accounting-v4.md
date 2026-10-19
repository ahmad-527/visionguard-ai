# Held-out output accounting v4 — engineering-only review

Issue: #31. This is a separately frozen operational proposal, not a live-run
repair or deployment. The active reviewed v3 implementation and authorization
remain unchanged. Independent human review is required before merge or use.

## Diagnosis and preserved incident

The v3 runner and independent watcher periodically inventory the entire retained
output tree. `inventory()` validates the full ancestor chain again for each
entry; accounting then stats each file again. Cost grows with retained file count
and path depth. Runner `check.last` used the time captured **before** expensive
verification/accounting/publication, so a check taking roughly the interval could
become due immediately again. The watcher already used completion-time scheduling.

Read-only operational investigation observed CPU-active processes and expensive
periodic checks during a prolonged receipt plateau, which subsequently advanced
without intervention. Repeated accounting is a supported bottleneck candidate;
the exact fraction of stage time attributable to scans, CPU/native inference or
output verification was not established. Low GPU utilization is not failure,
and logical I/O counters do not prove disk saturation. No extra live-tree scans
or scientific payload inspection were performed for this proposal. Detailed
operational identifiers and original observations remain in ignored local evidence.

The new telemetry separates provenance and accounting durations. It does not
claim that reducing accounting predicts the active run's completion time.

## Small operational extension

Each process owns an independent metadata ledger. On Windows an overlapped,
recursive directory notification read is armed **before** the full baseline.
A blocking metadata-only reader continuously re-arms it across foreground native
work. Its coalesced backlog is bounded to 8,192 names and 1 MiB estimated name/action
payload; the native buffer is 64 KiB. Python container overhead is additional.
The reader never opens output contents, datasets or models.

Routine checks update only notified paths. A durable write registers its final
absolute size, replacing an earlier observation rather than adding it twice.
Added files, same-size changes, renames, deleted files and moved subtrees are
observed. Directory LastWrite notifications do not cause unrelated stage rescans;
child changes have their own notifications. Newly added/renamed subtrees are
reconciled. Baseline/final/error reconciliation checks ancestors per directory and
each entry's no-follow metadata, not the whole chain separately per file.

Full reconciliation remains mandatory at startup/resume and before final
publication. Non-Windows synthetic/CI hosts retain a linear periodic full-audit
fallback; that is not silently substituted after Windows notification failure.

Overflow, malformed notifications, dead reader/handle, inaccessible paths, unsafe
reparse/cloud entries or root replacement are fatal. No automatic recovery or
inference continuation occurs. After failed accounting, only a bound immutable
`failure.json` can be published, and only after fresh full reconciliation and the
unchanged path, volume, retention and actual-free-space checks. Scientific writes
remain forbidden. If storage prevents that receipt, the original exception gains
a failure-publication note instead of silently bypassing a safeguard. A failed
writer's lock is preserved for independent human process review.

Runner checks are scheduled from **successful completion**. Provenance verification
still runs, hardware checks and owner identities are retained, and the watcher
continues checking its owner and draining updates each owner-check cycle. Actual
free space, fixed volume identity, safe paths, the 128 GiB retained logical-byte
limit and 20 GiB reserve remain checked before writes. All attempts count.

## Scientific and pause/resume invariants

No model loading, preprocessing, calibration, membership, prediction, metric,
triage, aggregation or replay computation is changed. The model/pair scientific
publication formats are unchanged. Operational telemetry gains timing and ledger
metadata; it is not a scientific result or content-authentication certificate.

Synthetic tests compare complete/result/validated publications and PNG/TIFF bytes
against the archived v3 accountant across all 36 pairs/72 model stages. They also
cover controlled pause boundaries, partial-stage preservation, exact resumed
calculations, stale requests, owner/watcher handshake, cleanup failures and refusal
of unclassified/corrupted recovery. Native notification burst/backlog and immutable
failure-evidence cases are included. No actual models or test assets are used.

## Measured synthetic evidence and reproduction

The compact source-bound receipt is `reports/output-accounting-v4/benchmark.json`.
It retains all timing samples, medians, independent expected/observed totals,
Python/platform identifiers and source hashes. Each tree uses 256-byte manufactured
files. Large manufactured workloads and exploratory failures remain ignored locally.

Final Windows/Python 3.11 manufactured-file measurements (median of five samples):

| Files | v3 full scan | v4 idle check | v4 full reconciliation |
| ---: | ---: | ---: | ---: |
| 1,000 | 1.5244 s | 1.9211 ms | 0.1458 s |
| 5,000 | 7.3573 s | 2.0259 ms | 0.2438 s |
| 20,000 | 29.3176 s | 2.1990 ms | 0.4306 s |

Final addition-registration/refresh measurements were 8.10, 8.25 and 9.64 ms.
Expected/observed logical totals matched exactly: 257,024, 1,281,024 and
5,121,024 bytes respectively (including the added 1,024-byte file). At 20,001
entries the shallow Python ledger-size estimate was 6,882,121 bytes; it is not
process RSS and excludes directories, reader/backlog and other process objects.
Receipt SHA-256:
`37e702220f439b80b7494b2773ad0ed3a316e2a9e7bacf7900397cbcf2061107`.

```powershell
python scripts/output_accounting_freeze.py
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m pip check
python scripts/benchmark_output_accounting.py --output outputs/engineering/NEW-unique-name
```

Use an isolated engineering checkout/environment. Do not use these commands on
the active scientific checkout. CI exercises Windows and Linux on Python
3.11, 3.12 and 3.13 with synthetic fixtures only.

## Limitations and negative findings

The first foreground-only observer overflowed in a synthetic pause/resume run;
the full exploratory suite retained eight failures. Historical test fixtures also
required routing to preserved predecessors, and a legacy-budget test adapter
needed the newly required operational fields. Logs and candidate freezes are
retained; none is relabelled as successful final validation. The corrected reader
continuously consumes notifications, but finite buffers may still overflow under
extreme bursts: the response remains STOP, not dropped events or automatic retry.

Accounting authenticates neither file contents nor scientific validity. Metadata
can be transient while another writer publishes; notifications may coalesce, and
registered writes can also produce notifications. Counters are metadata updates,
not unique unauthorized changes. A hostile concurrent writer can exceed a limit
before observation; no instantaneous adversarial filesystem guarantee is claimed.
Existing immutable hashes, provenance and scientific replay remain mandatory.
Hard-linked paths count logical retained bytes as before; physical free-space
checks are separate. Ledger memory is O(file count), and full reconciliation is
still O(file count). Directory/path replacement races are not a new adversarial
filesystem security proof.

Benchmarks are sequential warm-cache metadata measurements on the engineering
volume, with five samples per check. They are not randomized, not USB measurements,
not inference timings, not statistical confidence intervals and not a production
throughput/generalization claim. Provenance, GPU observations, Windows warning
queries and inference/output replay retain their separate costs.

## Freeze and deployment boundary

The v3 fingerprint remains
`0334a17b5b5c8f954e692c4caa47f93eaead0228acb6c45a082a75ca3fbc1c47`.
Exact predecessor source/test bytes and all earlier freeze documents remain
preserved. v4's distinct fingerprint is recorded in
`reports/output-accounting-v4/implementation-freeze-v4.json` and verified against
the unchanged B1/B2/development contracts and 72 published model identities.
This verifies compact published identities, **not real model payloads**, which
this engineering task must not access. Scientific freezes are not regenerated.

v4 implementation fingerprint:
`68e5e1bc3db4170e9dcebc6d7e844e8343e32ee16cce0032c6749e78c894c469`.

No history rewrite, merge, live checkout/environment change, authorization claim,
lock repair, pause/restart or real scientific evaluation is performed. Future
deployment requires independent review and explicit fingerprint-bound human
authorization; existing v3 approval cannot authorize v4. Historical human access
remains UNKNOWN; independent reservation remains NOT ESTABLISHED. The existing
run's qualification and scientific evidence must not be rewritten.
