"""Bounded retention and conservative operational observations; no retries/settings."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import time
from contextvars import ContextVar
from pathlib import Path

from visionguard.heldout_paths import checked, inventory
from visionguard.visa_evaluator import require

ACTIVE_BUDGET = ContextVar("heldout_budget", default=None)


def volume_identity(path: Path):
    if os.name != "nt":
        return path.stat().st_dev
    serial = ctypes.c_ulong()
    require(
        bool(
            ctypes.windll.kernel32.GetVolumeInformationW(
                str(path.anchor), None, 0, ctypes.byref(serial), None, None, None, 0
            )
        ),
        "Output volume unavailable",
    )
    return serial.value


class Budget:
    """All attempts retained in run root count; actual free floor checked per write."""

    def __init__(self, root: Path, limit=128 * 2**30, reserve=20 * 2**30):
        self.root, self.limit, self.reserve = root, limit, reserve
        self.used = (
            sum((root / p).stat().st_size for p in inventory(root))
            if root.exists()
            else 0
        )
        parent = root
        while not parent.exists():
            require(parent.parent != parent, "Disconnected output drive")
            parent = parent.parent
        self.volume = volume_identity(parent)

    def before(self, additional=0):
        checked(self.root, missing=True)
        parent = self.root
        while not parent.exists():
            require(parent.parent != parent, "Disconnected output drive")
            parent = parent.parent
        require(
            volume_identity(parent) == self.volume, "Output volume changed/disconnected"
        )
        require(
            self.used + additional <= self.limit,
            "128 GiB total retained run budget exceeded",
        )
        require(
            shutil.disk_usage(parent).free - additional >= self.reserve,
            "Drive disconnected or safety reserve exhausted",
        )
        require(
            not (self.root / "watcher-stop.json").exists(),
            "Independent watcher requested STOP",
        )

    def refresh(self):
        self.used = sum((self.root / p).stat().st_size for p in inventory(self.root))
        self.before()


def before_write(bytes_count):
    budget = ACTIVE_BUDGET.get()
    if budget:
        budget.before(bytes_count)


def account_write(bytes_count):
    budget = ACTIVE_BUDGET.get()
    if budget:
        budget.used += bytes_count


def observation() -> dict:
    result = {
        "monotonic_seconds": time.monotonic(),
        "pid": os.getpid(),
        "gpu_point": None,
        "cuda_allocator_peak": None,
        "process_peak_working_set_bytes": None,
    }
    if os.name == "nt":
        # PROCESS_MEMORY_COUNTERS layout is pointer-size aware.
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t)
                for n in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        values = Counters()
        values.cb = ctypes.sizeof(values)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        require(
            bool(
                ctypes.windll.psapi.GetProcessMemoryInfo(
                    ctypes.c_void_p(kernel.GetCurrentProcess()),
                    ctypes.byref(values),
                    values.cb,
                )
            ),
            "RAM observation unavailable",
        )
        result["process_peak_working_set_bytes"] = values.PeakWorkingSetSize
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.PIPE,
        )
        rows = [r.strip().split(",") for r in raw.strip().splitlines()]
        require(len(rows) == 1, "Exactly one GPU required")
        uuid, temp, used, total = rows[0]
        temperature = float(temp)
        require(temperature < 85, "GPU temperature safety STOP (85 C)")
        result["gpu_point"] = {
            "uuid": uuid.strip(),
            "temperature_c": temperature,
            "used_mib": float(used),
            "total_mib": float(total),
        }
    except FileNotFoundError:
        pass  # Synthetic CPU hosts need no GPU; real runner requires measurement.
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        result["cuda_allocator_peak"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "allocator_peak_since_explicit_reset_not_device_peak",
        }
    return result


def windows_warnings(start_utc: str) -> list:
    if os.name != "nt":
        return []
    # start travels as stdin, not executable interpolation. Inspection only.
    script = r"""
$ErrorActionPreference='Stop'
$start=[datetime]::Parse([Console]::In.ReadToEnd())
try{
  $all=Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$start}
}catch{
  if($_.FullyQualifiedErrorId -notmatch 'NoMatchingEventsFound'){throw}
  $all=@()
}
$events=@($all | Where-Object {
  ($_.ProviderName -match 'WHEA|disk|Ntfs|stor|USB|nvlddmkm|Kernel-Power') `
    -and $_.Level -le 3
})
ConvertTo-Json -InputObject @($events|Select-Object Id,ProviderName,
    TimeCreated,Level) -Compress
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        input=start_utc,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else []


def monitor(root: Path, budget: Budget, real: bool):
    budget.before()
    row = observation() if real else {"pid": os.getpid(), "synthetic_cpu": True}
    if real:
        require(row["gpu_point"] is not None, "Real GPU measurement unavailable")
    return row
