# VisionGuard v1 delivery status

The canonical development repository is ahmad-527/visionguard-ai. PR36 is
integrated by a normal merge at f13beb897e9223e07866f1f681c457cce6dcfec1, with
parents 6e0bfb0c5cb3496efe760b42871e47a84995c778 and
72f299c186a034f3441df6ef965db3ad2cf400b0. Its tree is the reviewed
8531ffd0961c90534bb94ced085d2af477bc7dbd. The review branch/history are preserved.
The separate published FlawGlyph repository remains an unchanged source snapshot.

The implemented v1 application is a loopback FastAPI service and browser UI in
apps/inspection, a server-managed pinned registry, fixed CPU native adapter,
original-coordinate display, explicit model identity/threshold, bounded
single-request concurrency and truthful failure/quarantine/shutdown semantics.
The core and application distributions intentionally remain separate and both
retain version 0.1.0. No implementation restart or global rename is proposed.

The previous bounded trained workflow completed one restricted extraction, one
restricted output validation and one PatchCore/candle/42 application load with
exactly three generated-image predictions. Canonical tensor identity, strict
restoration, dimensions/finiteness, fixed thresholds, browser display and owned
shutdown passed within that exact private contract. Its authority is permanently
spent. This technical evidence is neither industrial accuracy nor historical
prediction-equivalence evidence; it is not permission for another trained run.

The original release criteria and their current dispositions are in
[v1-release-checklist.md](v1-release-checklist.md). No new feature or broad audit
is required to use the manufactured local demonstration. It deliberately has no
industrial diagnostic meaning. An empty registry returns not ready.

Private post-merge changes prepare a narrow current-Ruff compatibility config,
an ordinary tensor-state ZIP admission gate before Torch deserialization, an
opt-in durable Python-stack fixture-test observer and updated documentation.
They are not published, released, deployed or trained-accepted by this document.
Unchanged model/scientific code remains backed by its original evidence.
Whole-candidate/source-hash acceptance does not transfer to changed files.

Current main's natural push CI ran seven workflows: five succeeded, while the
six ordinary lint jobs failed on Ruff 0.17.1's new WeakSet RUF012 detection in a
historically frozen test. The private config preserves all frozen bytes and
inherits all lint rules, with one file/rule exception for that intentional shared
allocation-owner registry. It changes no runtime dependency or test assertion.
The application/native workflows are pull-request/manual-triggered, so this push
did not rerun them; the exact-byte PR36 evidence is retained rather than fabricated.

The upstream PT2 advisory remains unresolved at the package-version level.
Supported application input is image data, never a client-supplied model/path.
The native factory accepts only an independently approved, pinned, ordinary
tensor-state container and explicitly uses CPU weights_only=True; it has no
export/JIT/download/unsafe-pickle fallback. This is a path-specific boundary,
not package-wide security clearance. See
[v1-operational-security-status.md](v1-operational-security-status.md).

The limited source grant applies to its exact adopted scope. It does not
automatically cover new edits, all extra canonical-repository files, models or
datasets. The private release proposal identifies the additional source scope
and notice/metadata obligations without requesting a blanket ownership guarantee.

The project is not declared fully complete: corrected source publication,
licence scope, final native-security disposition and any release/version action
still require the exact remaining review described in the consolidated result.
Do not repeat a trained test merely to replace a missing administrative decision.

Held-out VisA evaluation with historical access independence unverified.
