"""Machine-generated experiment environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from typing import Any


def _detected(value: Any) -> dict[str, Any]:
    return {"status": "detected", "value": value}


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _package_version(name: str) -> dict[str, Any]:
    try:
        return _detected(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        return _unavailable("package not installed")


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(f"nvidia-smi failed: {type(exc).__name__}")
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append(
                {
                    "name": fields[0],
                    "memory_mib": int(fields[1]),
                    "driver_version": fields[2],
                }
            )
    return _detected(rows) if rows else _unavailable("nvidia-smi returned no GPUs")


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return _unavailable("torch not installed")
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                }
            )
    return _detected(
        {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_devices": devices,
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
    )


def capture_environment(configured_device_policy: str) -> dict[str, Any]:
    """Return JSON-compatible measured, unavailable, and configured values."""

    resolved_packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "schema_version": 1,
        "operating_system": _detected(
            {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            }
        ),
        "python": _detected(
            {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            }
        ),
        "packages": {
            name: _package_version(name)
            for name in ("visionguard-ai", "anomalib", "torch", "torchvision", "timm")
        },
        "resolved_packages": dict(sorted(resolved_packages.items())),
        "nvidia": _nvidia_smi(),
        "torch_backend": _torch_environment(),
        "configured": {"device_policy": configured_device_policy},
    }


def environment_json(configured_device_policy: str) -> str:
    """Serialize environment capture without machine-specific executable paths."""

    return json.dumps(
        capture_environment(configured_device_policy), indent=2, sort_keys=True
    )
