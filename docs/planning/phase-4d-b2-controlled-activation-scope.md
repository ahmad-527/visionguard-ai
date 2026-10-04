# Controlled activation implementation — not execution authorization

Issue #25. Base: reviewed PR #24 regular merge
`3ab2b3bc9135aa1c1c63aef78fd2d88c1552f1f7`, second parent
`5464f17a0792e1aa8819281599a3f6415f698712`.

All implementation will use separate `heldout_*` modules, tests, workflow and
freeze. B1 and B2 readiness source/fingerprints, models and thresholds remain
unchanged. Real-ID admission and numerically equivalent reducers are tested on
manufactured directory trees only. No real test-root operation is authorized.

Human approvals will not be fabricated. Future execution must require separately
provisioned protected trust metadata, a valid signed approval, reviewed regular
merge identity and an immutable single-use claim/resume lineage. No trust key or
approval will be installed by this engineering task. Missing governance evidence
keeps every real gate CLOSED before root access.

Historical access UNKNOWN; independent reservation NOT ESTABLISHED. No training,
recalibration, tuning, actual held-out evaluation or autonomous merge.
