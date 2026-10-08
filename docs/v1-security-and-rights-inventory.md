# Application dependency/security and rights inventory

Snapshot: 9 October 2026 Europe/Berlin (8 October UTC). This is a review
inventory, not a security certification, legal opinion or native-use approval.
The external robustness packet retains exact commands, times, scanner JSON,
distribution/license metadata and notice-file hashes. No trained artifact or
dataset was opened. Historical ML environments were inspected as distribution
metadata only and remain unchanged.

## Tools, freshness and findings

`pip-audit 2.10.1` queried the public PyPI advisory service and OSV for the
application runtime. Engineering and preserved ML metadata were queried through
PyPI. The packet contains query timestamps and returned advisory IDs/fix versions;
these are service snapshots, not proof of a complete database or known freshness
cutoff. See the [tool's scope and limitations](https://github.com/pypa/pip-audit).

| Scope | Observed findings | Disposition |
| --- | --- | --- |
| Earlier clean application/engineering venvs | 22 advisory entries in `pip 23.2.1` and `setuptools 65.5.0`; some entries/aliases duplicate one issue | Retain failed audit. New package acceptance pins `pip 26.2.1` and `setuptools 84.0.0`; separately audit the resulting fresh installation |
| Preserved ML environment | 12 raw advisory entries across `lightning 2.6.5`, `pytorch-lightning 2.6.5`, `multidict 6.7.1`, `setuptools 65.5.0` | No upgrade. Separate successor/environment review before native adoption |
| Local CUDA versions | PyPI cannot audit `torch 2.9.1+cu126` or `torchvision 0.24.1+cu126` by exact version | Public base-version query is separate evidence, not validation of CUDA binaries |
| Public ML base-version query | Four advisory entries for `torch 2.9.1`; none returned for `torchvision 0.24.1` | Retain results; base-version findings also need native-adoption review |
| Unpublished VisionGuard packages | Not present in public advisory databases | Skipped, not cleared; reviewed/tested source is separate evidence |

Preserved ML recommendations: review Lightning `CVE-2026-58659` /
`GHSA-qqmf-gpg7-g8gw` (scanner recommends 2.6.6), multidict
`CVE-2026-104874` / `GHSA-54p9-h82j-f925` (6.9.1), and setuptools advisories
including `CVE-2026-59890` (83.0.0). Full payloads remain external. Version
matches do not establish exploit reachability. The adapter uses
`torch.load(weights_only=True)`, not Lightning `load_from_checkpoint`; this is
not blanket clearance of ML dependencies. No unsafe-load fallback is introduced.

The public Torch query includes `CVE-2026-24747` / `GHSA-63cw-57p8-fm3p`,
a `weights_only` unpickler advisory with a reported fix at 2.10.0, plus
`PYSEC-2026-139` (no fix version returned), `PYSEC-2025-194` (2.13.0) and
`PYSEC-2025-195` (2.10.0). These returned ranges/versions must be independently
assessed in a separate ML successor; exact local CUDA binaries were not audited.
In particular, `weights_only=True` is not proof that the frozen Torch version
is safe for untrusted checkpoints. The
[PyTorch maintainer advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
confirms affected versions through 2.9.1 and the 2.10.0 fix. No native artifact
is loaded in this task.

Only newly created packaging-test environments receive reviewed public
[pip](https://pypi.org/project/pip/) and
[setuptools](https://pypi.org/project/setuptools/) pins. Frozen core/ML
requirements, prior environments, build records and incident evidence are not
rewritten. Recheck advisories before later release/dependency adoption.

## Declared third-party rights

The complete external inventory contains 23 application-runtime, 38
engineering-test and 83 preserved-ML distribution entries, including declared
license expressions/classifiers and available LICENSE/NOTICE hashes. It describes
the observed installation, not every possible resolution of allowed ranges.
Exact final clean-install versions are separately recorded in the packet.

| Runtime package(s), observed version | Declared license metadata |
| --- | --- |
| annotated-doc 0.0.5, annotated-types 0.8.0, anyio 4.15.1 | MIT |
| certifi 2026.7.22 | MPL-2.0 |
| click 8.5.0, httpcore 1.0.9, httpx 0.28.1, idna 3.20 | BSD-3-Clause |
| fastapi 0.143.0, h11 0.16.0, pydantic 2.14.0, pydantic_core 2.50.0 | MIT |
| opentelemetry-api 1.45.1 | Apache-2.0 |
| Pillow 12.3.0 | MIT-CMU |
| PyYAML 6.0.3, typing-inspection 0.4.4 | MIT |
| starlette 1.7.0, uvicorn 0.54.0 | BSD-3-Clause |
| typing_extensions 4.16.0 | PSF-2.0 |
| Earlier pip 23.2.1 / setuptools 65.5.0 bootstrap | MIT metadata/classifier; replaced only in new acceptance venvs |
| visionguard-ai / visionguard-inspection 0.1.0 | Source license unselected; release gate open |

Observed optional ML declarations: Anomalib 2.6.0, timm 1.0.28 and Lightning
2.6.5 declare Apache software licensing; Torch 2.9.1+cu126 declares BSD-3-Clause;
Torchvision 0.24.1+cu126 declares BSD. Declarations and notice hashes are evidence
inputs, not a completed compatibility/redistribution determination. Missing
declarations/notices in the full inventory require review. Retain dependency
notices when assembling an eventual distributable release.

## Model, dataset, browser and source boundaries

- Proposed PatchCore development backbone is
  [timm/wide_resnet50_2.racm_in1k](https://huggingface.co/timm/wide_resnet50_2.racm_in1k),
  revision `30f73aceaaa1911830a9795b83ab1908dba18719` in the preserved protocol.
  Its current model card declares Apache-2.0. This does not independently verify
  local weight provenance, training-data rights or application-use authority.
- [Torchvision model documentation](https://docs.pytorch.org/vision/master/models.html)
  treats weight-use permission separately. EfficientAD teacher rights and any
  redistributed checkpoint need exact-artifact review; Python licenses are not
  model-use approval.
- VisA/MVTec AD 2 remain external under the reviewed dataset strategy and recorded
  upstream terms. No images, masks, annotations, checkpoints or pretrained
  weights are in this patch. Commercial/data redistribution rights are not newly
  determined.
- Browser source is original HTML/CSS/JS; no CDN/fonts/icons/third-party UI assets
  are bundled. Playwright browsers are test-only downloads, not package assets.
- MIT/Apache-2.0 options remain in `v1-source-license-options.md`. No LICENSE,
  copyright ownership or model approval is manufactured.

Limitations: no penetration test, exploit demonstration, transitive binary
supply-chain attestation, commercial-rights clearance or native checkpoint smoke.
Permission JSON requires independently established operator authority, not a
signature service. Loopback does not protect against malicious local processes.
Public deployment is out of scope. Native acceptance/licensing remain human gates.
