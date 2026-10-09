"""Separately reviewed application CPU profile, never the frozen evaluator runtime.

Matching this profile is compatibility admission, not security or artifact-use
approval. The unresolved upstream PT2-loader advisory remains a human review gate.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType

from visionguard.inspection_contract import InspectionError

PROFILE_ID = "application-cpu-20261009-v1"
MODEL_VERSIONS = MappingProxyType(
    {
        "torch": "2.13.0+cpu",
        "torchvision": "0.28.0+cpu",
        "anomalib": "2.6.0",
        "timm": "1.0.28",
        "lightning": "2.6.6",
        "pytorch-lightning": "2.6.6",
    }
)


def verify_runtime() -> None:
    """Reject missing/drifted packages and non-CPU builds before deserialization."""
    if sys.version_info[:2] not in {(3, 11), (3, 12), (3, 13)}:
        raise InspectionError("CPU successor supports reviewed Python 3.11-3.13 only")
    for package, expected in MODEL_VERSIONS.items():
        try:
            observed = version(package)
        except PackageNotFoundError as exc:
            raise InspectionError(
                f"CPU successor dependency unavailable: {package}"
            ) from exc
        if observed != expected:
            raise InspectionError(
                f"CPU successor runtime drift: {package} expected {expected}, "
                f"got {observed}"
            )
