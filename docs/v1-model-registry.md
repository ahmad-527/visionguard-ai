# Server-managed registry and native-use gate

The browser chooses a bounded model ID, never a filesystem path, checkpoint,
device, preprocessing or threshold. No default/best model, model-scale fusion,
training, recalibration, download or final-test admission is provided.

An external, operator-owned immutable artifact directory holds the pinned JSON
registry and approved development files. All paths are portable relative paths;
absolute paths, traversal, alternate streams, links and reparse points are
refused. Ancestors are also checked for links. Metadata reads are capped at 1 MiB
per file; checkpoint bytes at 1 GiB. Hashing and deserialization use the same
captured bytes. These checks assume the operator prevents concurrent filesystem
replacement; they are not a cross-process sandbox or proof of provenance.

Registry schema 1 has only `schema_version` and `models` (0–72 entries). An entry
requires `model_id`, `model` (`patchcore`/`efficientad`), `category`, `seed`,
`canonical_state_sha256`, and four `{path, sha256}` records: `artifact`, `science`,
`calibration`, `application_permission`. `science` is JSON of the approved
protocol's **scientific configuration**, not a performance report. `calibration`
contains the existing `image.threshold` and `pixel.threshold`; the service reads
them unchanged and computes no new threshold. All digests are independently
pinned by the operator; the application cannot grant itself permission.

The human-provided permission must bind all of:

- `schema_version: 1`, `scope: application-development-inference`,
  `source_role: development`, `device: cpu`;
- `input_scope: generated-or-development-non-held-out`;
- exact `model_id`, `model`, `category`, `seed`;
- `artifact_sha256`, `canonical_state_sha256`, `science_sha256`,
  `calibration_sha256`;
- a nonempty `human_attestation` and future timezone-aware `expires_at_utc`.

Permission is checked **before any preprocessing/calibration/checkpoint asset**.
The permission and configuration hashes are rechecked before every request,
including reuse of a resident model. Time-limited admission cannot safely cancel
a call already executing; expiry stops new admission, not an active native call.
Operators must establish the attestation's authority independently. JSON alone
does not authenticate its author; no signing or authorization-generation API
exists. Do not commit real attestations or registry/artifact directories.

Native loading is lazy. `/ready.ready` means an eligible registry entry can be
submitted to the request boundary; it does **not** mean native weights loaded.
`native_ready` is true only for an eligible, verified resident native model, with
the worker not quarantined. Checkpoint identity and execution can still fail on
first use; every failure is a non-decision. Models listed before first use are
declared identities, not claims that their weights have been independently read.

The CPU adapter permits Torch 2.9.1, Torchvision 0.24.1 and Anomalib 2.6.0. It
verifies supported architecture/preprocessing, the checkpoint byte digest,
model/category/seed binding, declared and recomputed canonical tensor digests,
strict state loading and equality after loading. It uses the existing frozen
constructor and preprocessing implementation without editing them. Inputs are
RGB, 256×256 bilinear antialiased resize, own model normalization, then restored
to original dimensions; no crop, TTA, tuning or new calibration is introduced.

Only `torch.load(..., weights_only=True, map_location="cpu")` is permitted.
There is no unsafe pickle fallback or custom-global allowlist. Some historical
development payloads include RNG state and may be refused by this safe loader.
If so, a separately reviewed/authorized **evidence-preserving tensor-only export**
would be required; do not modify or relabel the original checkpoint. This export
has not been performed or authorized by the current application work.

Native POST requests additionally require
`X-VisionGuard-Input-Role: generated-or-development-non-held-out`; the browser has
an explicit input attestation. This is an operator assertion, not a detector of
held-out images or an independent historical-access audit.

## Exact outstanding requirement

No current application-use permission bound to a specific development artifact
has been established. Prior matrix execution permission is not reused or
extended. Before real native acceptance, provide the human-approved development
model/category/seed, exact checkpoint + canonical tensor + preprocessing +
existing calibration identities, CPU/input-use scope and expiry, and operator-
pinned registry/permission digests. Identify generated or approved non-held-out
input fixtures. Dataset/weight redistribution rights remain separate. Until then,
native acceptance is **blocked**, not silently replaced by manufactured tests.
