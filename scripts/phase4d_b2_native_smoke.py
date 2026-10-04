"""Bounded frozen-model CUDA smoke on manufactured RGB only, no test loader.

No checkpoint/model/calibration writes. One model at a time, two artificial
frames per model. Hardware/publication failure stops; never automatically retry.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from visionguard.efficientad import canonical_checkpoint_sha256
from visionguard.visa_b2_contract import context
from visionguard.visa_b2_storage import json_once, publish_cell, replay_cell
from visionguard.visa_evaluator import SyntheticSample, require
from visionguard.visa_evaluator_backend import SyntheticFrame, load_development_backend


def memory_observation() -> dict:
    """Measured process peak working set, not all-process RAM attribution."""
    if os.name != "nt":
        import resource

        return {
            "process_peak_working_set_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024
        }

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    require(
        bool(
            psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
        ),
        "Host RAM observation failed",
    )
    return {
        "process_peak_working_set_bytes": counters.PeakWorkingSetSize,
        "process_working_set_bytes": counters.WorkingSetSize,
        "process_private_bytes": counters.PrivateUsage,
    }


def nvidia_observation() -> dict:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, used, total, temperature = [s.strip() for s in output.split(",")]
    return {
        "name": name,
        "device_used_MiB": int(used),
        "device_total_MiB": int(total),
        "temperature_C": int(temperature),
        "scope": "point_observation_not_continuous_device_peak",
    }


def recent_hardware_events(start: str) -> list:
    if os.name != "nt":
        return []
    command = (
        "$events=@(Get-WinEvent -FilterHashtable @{LogName='System';"
        "StartTime=[datetime]'"
        + start
        + "';Level=2,3} -ErrorAction SilentlyContinue | Where-Object "
        "{$_.ProviderName -match "
        "'(^disk$|Ntfs|storahci|stornvme|USB|WHEA|nvlddmkm)'} | "
        "Select-Object TimeCreated,ProviderName,Id); "
        "ConvertTo-Json -InputObject $events -Compress"
    )
    return json.loads(
        subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", command], text=True
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    require(
        torch.cuda.is_available(), "CUDA unavailable; do not invent CPU/GPU equivalence"
    )
    require(
        shutil.disk_usage(args.repository).free >= 148 * 1024**3,
        "Insufficient D: capacity",
    )
    ctx = context(args.repository)
    # Largest bank population from immutable fitting count, before any scores.
    largest = max(
        r["fit_count"]
        for k, r in ctx["published"]["cells"].items()
        if k.startswith("patchcore:")
    )
    category = min(
        k.split(":")[1]
        for k, r in ctx["published"]["cells"].items()
        if k.startswith("patchcore:") and r["fit_count"] == largest
    )
    seed = 42
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    report = {
        "evidence_class": "native_frozen_model_synthetic_engineering_only",
        "category": category,
        "seed": seed,
        "largest_patchcore_fit_count": largest,
        "frames_per_model": 2,
        "test_asset_access": False,
        "real_test_performance_evaluations": 0,
        "device_before": nvidia_observation(),
        "models": {},
    }
    predictions = {}
    samples, frames = [], []
    yy, xx = np.indices((384, 512))
    for i in range(2):
        rgb = np.stack(
            ((xx + i * 17) % 256, (yy + i * 31) % 256, (xx + yy) % 256), axis=2
        ).astype(np.uint8)
        mask = np.zeros((384, 512), dtype=np.uint8)
        if i:
            mask[100:110, 200:210] = 1
        sample_id = f"synthetic:{category}:native-smoke-{i}"
        samples.append(SyntheticSample(sample_id, category, i, mask))
        frames.append(SyntheticFrame(sample_id, category, rgb))
    for model in ("patchcore", "efficientad"):
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        backend = load_development_backend(
            args.repository,
            args.repository / "outputs/phase4d-a-visa-development-matrix",
            model,
            category,
            seed,
            "cuda",
        )
        torch.cuda.synchronize()
        restore_seconds = time.perf_counter() - before
        state_before = canonical_checkpoint_sha256(backend.model.state_dict())
        times, values = [], []
        for frame in frames:
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = backend.predict_synthetic(frame)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            values.append(prediction)
        require(
            canonical_checkpoint_sha256(backend.model.state_dict()) == state_before,
            "Inference changed native model state",
        )
        predictions[model] = values
        shape = list(backend.model.memory_bank.shape) if model == "patchcore" else None
        report["models"][model] = {
            "model_sha256": backend.spec.model_sha256,
            "restore_includes_validation_seconds": restore_seconds,
            "synthetic_inference_seconds": times,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_ram": memory_observation(),
            "device_after": nvidia_observation(),
            "memory_bank_shape": shape,
            "original_map_shapes": [list(p.restored_map.shape) for p in values],
            "native_state_unchanged": True,
            "image_threshold": backend.spec.image_threshold,
            "pixel_threshold": backend.spec.pixel_threshold,
            "float32_map_sha256": [
                hashlib.sha256(p.restored_map.tobytes()).hexdigest() for p in values
            ],
        }
        del backend
        gc.collect()
        torch.cuda.empty_cache()
        require(
            not recent_hardware_events(started),
            "Hardware/storage/GPU warning during smoke; stop",
        )
    specs = tuple(
        ctx["specs"][f"{m}:{category}:{seed}"] for m in ("patchcore", "efficientad")
    )
    before = time.perf_counter()
    origin = {
        "source": "manufactured_in_memory_RGB",
        "B1_fingerprint": ctx["protocol"]["b1_fingerprint"],
        "developer_test_labels_not_benchmark": True,
    }
    directory = args.output / "native-synthetic-cell"
    receipt = publish_cell(
        directory,
        zip(samples, predictions["patchcore"], predictions["efficientad"], strict=True),
        specs,
        origin,
    )
    first = replay_cell(directory, receipt["sha256"], specs, origin)
    second = replay_cell(directory, receipt["sha256"], specs, origin)
    require(first == second, "Native synthetic persisted metric replay differs")
    report.update(
        publication_and_replays_seconds=time.perf_counter() - before,
        publication_sha256=receipt["sha256"],
        deterministic_reducer_replay=True,
        host_ram_final=memory_observation(),
        free_bytes=shutil.disk_usage(args.repository).free,
        hardware_events=recent_hardware_events(started),
        final_test_lock="closed",
    )
    require(not report["hardware_events"], "Hardware instability; stop")
    json_once(args.output / "native-smoke.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
