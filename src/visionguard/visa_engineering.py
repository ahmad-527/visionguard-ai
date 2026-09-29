"""Shared pre-test identity and bounded resource measurements (no model imports)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from visionguard.artifacts import capture_git_state
from visionguard.visa_acquire import VisaIntegrityError, sha256_file
from visionguard.visa_guard import AUDIT, FROZEN, MEMBERSHIP, verified_context
from visionguard.visa_protocol import canonical_fingerprint

ENGINEERING_FILES = (
    "src/visionguard/embedding_journal.py",
    "src/visionguard/visa_patchcore.py",
    "src/visionguard/visa_engineering.py",
    "src/visionguard/visa_guard.py",
    "src/visionguard/visa_execution.py",
    "src/visionguard/visa_development.py",
    "src/visionguard/efficientad_benchmark.py",
    "configs/engineering/visa-completion-acceptance-v1.yaml",
)


def environment_identity() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise VisaIntegrityError("CUDA unavailable")
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "anomalib",
                "timm",
                "numpy",
                "scikit-learn",
                "Pillow",
                "scipy",
                "safetensors",
            )
        },
    }


def execution_identity(
    repository: Path, model: str, *, extra_sources=()
) -> tuple[dict, dict]:
    protocol = verified_context(repository, model)
    git = capture_git_state(repository.resolve())
    if git["dirty"]:
        raise VisaIntegrityError(
            "Engineering GPU execution requires a clean Git commit"
        )
    environment = environment_identity()
    for name, expected in protocol["scientific"]["dependencies"].items():
        if name in environment["packages"] and (
            environment["packages"][name].split("+")[0] != str(expected)
        ):
            raise VisaIntegrityError(f"Frozen dependency mismatch: {name}")
    sources = {
        name: sha256_file(repository / name)
        for name in (*ENGINEERING_FILES, *extra_sources)
    }
    identity = {
        "model": model,
        "protocol_fingerprint": FROZEN[model],
        "audit_sha256": AUDIT,
        "membership_sha256": MEMBERSHIP,
        "implementation_commit": git["commit"],
        "source_hashes": sources,
        "source_fingerprint": canonical_fingerprint(sources),
        "environment": environment,
        "environment_fingerprint": canonical_fingerprint(environment),
        "final_test_lock": "closed",
    }
    return protocol, identity


def host_peak_bytes() -> int | None:
    """OS process peak working set/RSS, including imports; no polling estimate."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if read(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.peak)
        return None
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())
