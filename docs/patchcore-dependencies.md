# PatchCore dependency, license, and provenance review

## Decision and review date

Reviewed against authoritative upstream sources on 2026-08-28. Phase 2A uses
Anomalib 2.6.0 through a VisionGuard-owned adapter. The release is identified by
PyPI artifacts and Git tag `lib/v2.6.0` at commit
`3759687e76395c4d6d239552d3bf6d72e003da78`. This document contains dependency
and provenance facts, not model results.

## Compatibility decision

- Anomalib 2.6.0 is the current stable PyPI release, published 2026-07-25. It is
  Apache-2.0 and declares Python >=3.10; its classifiers list Python 3.10-3.12.
  VisionGuard therefore supports the ML extra on Python 3.11 and 3.12 in Phase
  2A. The lightweight audit package retains its existing Python >=3.11 contract.
- Anomalib's CUDA 12.6 extra declares PyTorch >=2.6 and torchvision >=0.21.
  VisionGuard pins the upstream-compatible pair PyTorch 2.9.1 and torchvision
  0.24.1. PyTorch publishes matching CPU and CUDA 12.6 wheels for Windows and
  Linux. Anomalib documents PyTorch >=2.6 because earlier versions lack the fix
  for CVE-2025-32434.
- `timm` is pinned to 1.0.28 because Anomalib's PatchCore feature extractor uses
  its public `create_model(..., features_only=True)` interface and Anomalib does
  not itself upper-bound or pin `timm`.
- Anomalib's core dependency set includes Lightning, TorchMetrics, timm, FrEIA,
  OpenCV-headless, scikit-image, scikit-learn, Kornia, pandas, matplotlib, and
  image codecs. Phase 2A does not enable unrelated OpenVINO, VLM, logger, video,
  notebook, or full extras. Exact transitive resolution must be retained from
  each run artifact's machine-generated `resolved_packages` mapping; the
  optional group alone is a direct-dependency pin, not a cross-platform full
  lock.
- CUDA wheels bundle their supported CUDA runtime components. The system CUDA
  toolkit is not used as proof of PyTorch compatibility. Select an official
  PyTorch wheel index explicitly, then verify `torch.cuda.is_available()` and the
  runtime reported by PyTorch.

## Install strategy

Basic auditing remains `python -m pip install -e ".[dev]"` and does not install
the ML stack. Use a separate virtual environment for PatchCore. For CUDA 12.6:

```powershell
python -m pip install torch==2.9.1 torchvision==0.24.1 `
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[ml,dev]"
python -m pip check
```

For a CPU-only engineering environment, replace the index URL with
`https://download.pytorch.org/whl/cpu`. Do not mix CPU and CUDA wheels by copying
environments between machines. The regular CI deliberately installs only
`.[dev]`; it requires neither network downloads, pretrained weights, GPU, nor a
dataset.

## PatchCore API and VisionGuard boundary

Anomalib 2.6.0 exposes `anomalib.models.Patchcore`, accepts a backbone, feature
layers, pretrained flag, coreset sampling ratio, neighbor count, and configured
preprocessor, and emits image scores plus anomaly maps. Its implementation uses
a memory bank and k-center-greedy coreset sampling. VisionGuard owns strict
experiment validation, the official MVTec AD 2 split policy, audit identity,
normal-only thresholds, independent metric sanity checks, environment/seed
capture, and the artifact schema. The adapter disables Anomalib's postprocessor,
evaluator, and visualizer so framework-selected thresholds or metrics cannot
silently become VisionGuard results.

MVTec AD 2 does not match Anomalib's legacy `MVTecAD` dataset contract exactly.
The existing VisionGuard audit contract remains authoritative. A run may expose
only official `train/good` images for memory-bank construction and official
`validation/good` images for normal-only calibration. `test_public` requires an
explicit frozen-configuration state. Private splits are rejected locally and
private labels have no supported access path.

## Backbone and pretrained weights

The Phase 2A backbone is the explicit timm identifier
`wide_resnet50_2.racm_in1k`, with `layer2` and `layer3`. Using the full identifier
avoids the reproducibility risk of the unqualified `wide_resnet50_2` alias
changing its default pretrained configuration.

The authoritative distribution is the Hugging Face model repository
`timm/wide_resnet50_2.racm_in1k`. Its model card identifies Apache-2.0 and the
weights as a WideResNet-50-2 trained on ImageNet-1k with the ResNet Strikes Back
recipe. Its reviewed `main` revision is pinned as
`30f73aceaaa1911830a9795b83ab1908dba18719`; a run fails if the cached revision
does not match. `timm` code is Apache-2.0. However, timm's own license guidance
says the implications of the ImageNet non-commercial research terms for derived
weights are unclear and advises legal review for commercial products.
VisionGuard treats these weights as research-only and does not redistribute
them. The MVTec AD 2 dataset is independently CC BY-NC-SA 4.0 and already limits
this work to non-commercial research/portfolio use.

Weights must be downloaded into the external model cache, never Git. The run
artifact must capture the concrete resolved file's SHA-256 after download. A Hub
repository name, mutable branch, HTTP ETag, or model-card license alone is not a
cryptographic identity. If the cache file cannot be located and hashed, the run
must record that as a failure instead of claiming verified weight provenance.

## Material reproducibility concerns

- Anomalib and timm use mutable dependency ranges upstream; VisionGuard pins
  direct ML packages, but a reviewed lock remains future work before a benchmark.
- GPU nearest-neighbor and coreset operations may not be bitwise deterministic on
  every platform even when deterministic algorithms are requested. Runs record
  seeds and backend settings without claiming perfect determinism.
- The model repository contains both safetensors and pickle-based PyTorch files.
  Prefer safetensors where the installed timm/Hugging Face path supports it.
- Unqualified timm model aliases, cache replacement, or dependency upgrades can
  change resolved weights. The explicit model ID, reviewed repository revision,
  and measured file SHA-256 are mandatory for a traceable real run.
- Local public-test metrics are not the official MVTec AD 2 private evaluation.
  No external documentation or literature metric is a VisionGuard result.

## Authoritative sources

- [Anomalib 2.6.0 on PyPI](https://pypi.org/project/anomalib/2.6.0/)
- [Anomalib 2.6.0 dependency declaration](https://github.com/open-edge-platform/anomalib/blob/lib/v2.6.0/pyproject.toml)
- [Anomalib 2.6.0 PatchCore API](https://github.com/open-edge-platform/anomalib/blob/lib/v2.6.0/src/anomalib/models/image/patchcore/lightning_model.py)
- [Anomalib 2.6.0 timm feature extractor](https://github.com/open-edge-platform/anomalib/blob/lib/v2.6.0/src/anomalib/models/components/feature_extractors/timm.py)
- [Official PyTorch version pairs and wheel indexes](https://pytorch.org/get-started/previous-versions/)
- [timm license and pretrained-weight guidance](https://github.com/huggingface/pytorch-image-models#licenses)
- [Pinned backbone model card](https://huggingface.co/timm/wide_resnet50_2.racm_in1k)
