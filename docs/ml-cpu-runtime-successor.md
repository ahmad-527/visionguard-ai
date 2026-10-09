# Application CPU runtime successor — review candidate, not activation

Baseline: application PR #34, `310409b6a0e83fa3709edd22976619be48cbb7c8`,
stacked on #33/#32. This profile is separate from historical evaluator freezes
and the root `ml` extra. It must never upgrade a canonical evaluation environment.

Held-out VisA evaluation with historical access independence unverified.

## Dependency choice and security disposition (observed 9 October 2026)

| Component | Candidate | Reason / limitation |
| --- | --- | --- |
| Torch / Torchvision | `2.13.0+cpu` / `0.28.0+cpu` | Official supported pair; CPU-only build, not CUDA or historical equivalence |
| Anomalib / timm | `2.6.0` / `1.0.28` | Preserve existing constructors, architecture and preprocessing implementation |
| lightning / pytorch-lightning | `2.6.6` / `2.6.6` | Current security fix for checkpoint instantiation controls |
| multidict | `6.9.1` | Fix C-extension items-view reference leak; unrelated to historical GPU cleanup |
| pip / setuptools | `26.2.1` / `84.0.0` | Reviewed bootstrap only in new environments |

The complete observed public resolution is constrained in
`requirements/ml-cpu-constraints.txt`. A constraint does not request installation
of a Windows-only dependency on Linux. Platform/Python-specific wheel bytes
are recorded separately in pip installation reports, not assumed identical.
Anomalib's base distribution is installed with explicitly selected Torch wheels;
its video/codec/training extras are not an application requirement.

Upstream verification is independent of an empty scanner result:

| Finding | Exact affected functionality/range | Candidate disposition |
| --- | --- | --- |
| [CVE-2026-24747 / GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) | `weights_only` unpickler/storage metadata, <=2.9.1; patched >=2.10.0 | 2.13.0 exceeds the fix floor. This alone does not clear other Torch findings. |
| [CVE-2025-3000 / GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) | JIT bare list/tuple annotations, current reviewed range <=2.12.1, fix 2.13.0 | 2.13.0 includes [maintainer fix b90c949](https://github.com/pytorch/pytorch/commit/b90c949). Older PYSEC metadata has a narrower range; retain the discrepancy rather than ignore it. Anomalib imports do internally use JIT script. |
| [CVE-2025-3001 / GHSA-qfhq-4f3w-5fph](https://github.com/advisories/GHSA-qfhq-4f3w-5fph) | Malformed LSTM-cell gate dimensions, <2.10.0 | Above fix floor; [maintainer patch](https://github.com/pytorch/pytorch/commit/999d94b5ede5f4ec111ba7dd144129e2c2725b03). No LSTM exposed by this inspection API. |
| [CVE-2026-4538 / GHSA-33x2-ppm4-v46v](https://github.com/advisories/GHSA-33x2-ppm4-v46v) | PT2 / `torch.export.load` unsafe implicit pickle fallback; affected-version metadata unresolved | **UNRESOLVED.** [Proposed upstream fix #176791](https://github.com/pytorch/pytorch/pull/176791) is closed **unmerged**. No patched release established by this investigation. Absence from scanner output is not clearance. |
| [CVE-2026-58659 / GHSA-qqmf-gpg7-g8gw](https://github.com/advisories/GHSA-qqmf-gpg7-g8gw) | Lightning `load_from_checkpoint` `_instantiator` / `_class_path`; current range <2.6.6 | Both packages 2.6.6; [maintainer 2.6.6 release](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.6.6) describes restrictions. [Metadata dispute #21970](https://github.com/Lightning-AI/pytorch-lightning/issues/21970) remains open; current API advisory was updated 24 September. App path does not use Lightning checkpoint loader. |
| [CVE-2026-104874 / GHSA-54p9-h82j-f925](https://github.com/aio-libs/multidict/security/advisories/GHSA-54p9-h82j-f925) | Items-view set union/subtraction reference leak, 6.7.0–6.9.0 | Patched 6.9.1; not a claim that VisionGuard's allocation incident was fixed. |
| Setuptools [2022-40897](https://github.com/advisories/GHSA-r9hx-vwmv-q579), [2024-6345](https://github.com/advisories/GHSA-cx63-2mw6-8hw5), [2025-47273](https://github.com/pypa/setuptools/security/advisories/GHSA-5rjg-fvgr-3xxf) | PackageIndex regex DoS / unsafe downloads / path traversal; fix floors 65.5.1 / 70.0.0 / 78.1.1 | New bootstrap 84.0.0 exceeds recorded floors; canonical bootstrap untouched. |
| [CVE-2026-59890 / GHSA-h35f-9h28-mq5c](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c) | Unicode-normalization MANIFEST exclusion bypass; maintainer affected <=82.0.1 | Reviewed database lists 83.0.0 fix, maintainer patched-version field is empty. 84.0.0 exceeds known range; metadata disagreement explicitly retained. |

Current source JSON snapshots, collection timestamps, content hashes, exact audit
commands and raw exit codes are in the external review packet. PyPI and OSV
`pip-audit 2.10.1` scans returned no findings for the queried installed versions.
A separate query of base `torch==2.13.0` / `torchvision==0.28.0` addresses possible
local `+cpu` version lookup gaps; it does not certify the received CPU binaries.
No blanket ignores, `--fix`, or dependency changes to frozen environments are used.

**Residual security decision:** this candidate permits only the existing
hash-bound tensor-state envelope through `torch.load(weights_only=True,
map_location="cpu")`. No unsafe fallback, added safe globals, PT2 export loading,
JIT archive loading or pretrained download is admitted. Tests verify the actual
restricted call path. Non-reachability of the PT2 path is a scoped assessment,
not remediation of the installed library or approval for untrusted checkpoints.
Trained-model use still requires explicit human security/artifact-use review.

## Reproducible isolated installation

Use a separate checkout at the reviewed successor commit and new CPython
3.11–3.13 virtual environment. Windows uses `.venv/Scripts/python.exe`, Linux
uses `.venv/bin/python`; below `python` means **that new environment only**.

```text
python -m pip install pip==26.2.1 setuptools==84.0.0
python -m pip --isolated install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu torchvision==0.28.0+cpu --report cpu-wheels.json
python -m pip install -c requirements/ml-cpu-constraints.txt -r requirements/ml-cpu-successor.txt -e ".[analysis,dev]" -e "apps/inspection[test]" --report public-dependencies.json
python -m pip check
python scripts/ml_cpu_inventory.py --output dependency-inventory.json --pip-report cpu-wheels.json --pip-report public-dependencies.json
```

Do not install `.[ml]`, mix an extra package index into the CPU install, reuse a
historical environment or replace historical lockfiles. Primary Torch wheels
come from the sole official CPU index (some dependency links redirect to PyPI).
Installation reports record resolved URLs and available archive digests; missing
digests are explicitly empty, not invented. Initial Windows CPython 3.11 hashes:

```text
torch-2.13.0+cpu-cp311-cp311-win_amd64.whl
10717d8b3b67c45a4788bf7ffc0bab1ea1e5ebbedd24466be6100102d141fac1
torchvision-0.28.0+cpu-cp311-cp311-win_amd64.whl
7b6667fd0172463be2a271fb0dbd44b31a7891afd549a66208613ce4cdd79f88
```

No universal byte-identical lock is claimed. The public inventory includes
declared license metadata; it does not grant dataset/model rights or choose a
source license/copyright owner. Existing third-party/model-rights limitations
and the application release checklist remain applicable.

## Compatibility acceptance and limitations

`tests/test_inspection_cpu_successor.py`, opt-in with
`VISIONGUARD_CPU_SUCCESSOR_TESTS=1` and `HF_HUB_OFFLINE=1`, executes actual
Anomalib PatchCore and EfficientAD constructors, safe serialized generated
states, strict restoration, canonical tensor equality, CPU eval/no-grad,
generated PNG inference (direct adapter and HTTP API), finite outputs,
permission revocation after successful native API inference, fixed preprocessing and bilinear
original-coordinate maps. Constant memory-bank and normalization/quantile
buffers are manufactured; no fitting or threshold calibration occurs.
Threshold constants remain unchanged and score equality remains NORMAL under
the strict existing `score > threshold` contract. Opaque pickle globals,
identity/key mismatches, changed/expired manufactured permissions and config
changes are refused. CPU model reachability after close is tested, not a GPU
allocator guarantee. Original constructor exceptions and worker quarantine are
preserved by the application lifecycle regression.

Three evidence classes must remain distinct:

1. Mocked plumbing/permission/browser tests: protocol and failure handling only.
2. Manufactured native CPU execution: actual constructors and tensor-state
   loading, not learned model quality or historical numerical equivalence.
3. Trained-model acceptance: **NOT PERFORMED / NOT AUTHORIZED** here.

The dedicated workflow declares Windows/Ubuntu CPython 3.11/3.12/3.13, runs the
actual native suite plus application/shutdown/freeze acceptance, `pip check`,
and wheel/sdist installed CLI/API/browser-resource smoke tests. The existing
application workflow separately executes manufactured browser-to-API acceptance.
Report exact tested head and completed configurations from hosted evidence,
never infer a matrix pass from this workflow declaration.

Historical v3/v4/v5 freeze verifiers remain unchanged, including v5 fingerprint
`8371cf29bb707bc44781dbfd37f7376dbef8c60e9cc4185567cd1fd595832c9d`.
No new evaluator authorization/freeze fingerprint is created or reused. This
app-only runtime profile is `application-cpu-20261009-v1`; permissions must bind
that profile and all exact `MODEL_VERSIONS`, in addition to existing pinned
checkpoint/science/calibration identities, expiry and input scope.

## Concrete next trained-model acceptance proposal (not approval)

Proposed model: existing development PatchCore / candle / seed 42, CPU, if the
operator confirms its development provenance and use rights. Review this exact
successor commit/dependency inventory and the residual PT2 advisory first.

Previously supplied metadata, **not independently verified in this task**:

| Identity | Proposed value / status |
| --- | --- |
| Checkpoint bytes | `7c293468dffdc77571621edffd16843c8fe3bdb76c8cb1314b590c7e4dd96b99` — UNVERIFIED |
| Canonical tensor state | `db1eaeff9786d65ed013981363787ce742dec073f9b5bee9208ce20416e7f575` — UNVERIFIED |
| Calibration bytes | `35ae0e7a0ecc7038808b948906e525229d7c5c4636e8c712adf824f4025c2991` — UNVERIFIED |
| Scientific/preprocessing config bytes | No independently verified digest supplied for successor admission — UNVERIFIED |
| Existing thresholds | Image 30.638519287109375; pixel 29.328676223754883 — metadata only, no recalibration or independent validation |

The smallest next approval is **one bounded acceptance packet** authorizing
read-only byte/canonical/config/calibration identity verification of explicitly
named development artifacts, followed by CPU loading/inference on generated
images only in this isolated profile. It must establish provenance/model rights,
expiry, exact reviewed commit/profile and permission bindings, and explicitly
disposition the residual PT2 risk. Actual approval/paths remain private and are
not generated by this candidate. A digest mismatch stops acceptance, not repair.

Alternative if that risk is unacceptable: defer trained-model loading until a
maintainer-confirmed PT2 fix is available and retest a new separately versioned
profile. A different tensor-only backend/format would require architecture and
identity-contract review; it is not a quiet downgrade or export of existing
weights. Neither alternative authorizes a held-out rerun, historical equivalence
claim, activation, deployment or release.
