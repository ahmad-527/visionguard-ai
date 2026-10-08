# Proposed native acceptance — not permission

Candidate: **PatchCore / candle / seed 42**, ID `patchcore-candle-42`, CPU only.
This is a deterministic first development cell, not a performance-selected winner.
No checkpoint is loaded by this proposal.

Existing compact development-freeze metadata is carried unchanged through v5
fingerprint `8371cf29bb707bc44781dbfd37f7376dbef8c60e9cc4185567cd1fd595832c9d`:

| Binding | Declared value | Current validation |
| --- | --- | --- |
| Development checkpoint byte SHA-256 | `7c293468dffdc77571621edffd16843c8fe3bdb76c8cb1314b590c7e4dd96b99` | Freeze declaration; bytes not read/rehashed |
| Canonical state SHA-256 | `db1eaeff9786d65ed013981363787ce742dec073f9b5bee9208ce20416e7f575` | Freeze declaration; actual state not revalidated |
| Original calibration SHA-256 | `35ae0e7a0ecc7038808b948906e525229d7c5c4636e8c712adf824f4025c2991` | Freeze declaration; file not opened/rehashed |
| Existing image / pixel threshold | `30.638519287109375` / `29.328676223754883` | Metadata transcription, not recalibration or metrics claim |

The external packet includes protocol/source metadata digests, proposed science
and calibration JSON bytes/hashes, plus three generated 32 x 24 RGB PNG fixtures
(black, checkerboard, gradient) with individual hashes. Proposed JSONs are not
original calibration artifacts or permission files. Preprocessing remains RGB,
exact 256 x 256 bilinear antialiased resize, CHW float32, ImageNet mean/std,
no augmentation; maps restore bilinearly to original coordinates. Existing strict
`score > threshold` is preserved.

Requested approval: one bounded local CPU acceptance session, three individually
submitted generated fixtures, this development model only, existing thresholds,
browser/API map/identity/error checks. No training, calibration, held-out input,
GPU run, checkpoint export, network exposure or evaluation restart. Suggested
validity is one operator-controlled session of at most 24 hours; the human must
choose the actual expiry and establish authority.

Unresolved requirements before execution:

1. Human application-use attestation for exact development checkpoint, canonical
   state, science/calibration bytes, CPU device and generated-input scope.
   Historical evaluation authorization is not repurposed.
2. Operator-verified artifact location/rights and actual byte identities matching
   these declarations. Do not substitute a held-out output checkpoint.
3. Approved external science/calibration metadata compatible with the adapter,
   pinned server registry and permission-file digest. Proposed hashes are not
   permission.
4. Safe `weights_only=True` compatibility. A tensor-only export, if needed,
   requires separate permission/evidence; no unsafe pickle fallback is allowed.
5. Separate ML dependency disposition, including advisory findings in
   `v1-security-and-rights-inventory.md`, before native adoption. This includes
   the public Torch 2.9.1 `weights_only` unpickler advisory; artifact approval
   alone does not close that security review gate.

Manufactured browser/API acceptance and CPU tensor plumbing are engineering
checks, **not native inference acceptance**. Preserve:

> Held-out VisA evaluation with historical access independence unverified.
