"""Operational GPU cleanup diagnostics; never weakens allocation-zero checks."""

from __future__ import annotations

import gc

from visionguard.visa_evaluator import require


def release_gpu(real: bool) -> int:
    """Verify zero current-device live allocations, attaching scalar diagnostics.

    Reserved cache bytes are reported separately, not mistaken for live tensor
    allocations. Exceptions retain their original type and propagate unchanged.
    Diagnostics contain no tensors, model state, predictions, or dataset data.
    """
    if not real:
        gc.collect()
        return 0
    import torch

    cuda = torch.cuda
    diagnostic = {
        "counter_scope": "current_cuda_device",
        "phase": "before_gc",
        "allocated_bytes_before_gc": None,
        "allocated_bytes_after_gc": None,
        "allocated_bytes_after_empty_cache": None,
        "reserved_bytes_before_gc": None,
        "reserved_bytes_after_empty_cache": None,
        "device_index": None,
    }
    try:
        diagnostic["allocated_bytes_before_gc"] = int(cuda.memory_allocated())
        if hasattr(cuda, "memory_reserved"):
            diagnostic["reserved_bytes_before_gc"] = int(cuda.memory_reserved())
        if hasattr(cuda, "current_device"):
            diagnostic["device_index"] = int(cuda.current_device())
        gc.collect()
        diagnostic["allocated_bytes_after_gc"] = int(cuda.memory_allocated())
        diagnostic["phase"] = "cuda_synchronize"
        cuda.synchronize()
        diagnostic["phase"] = "cuda_empty_cache"
        cuda.empty_cache()
        diagnostic["phase"] = "allocation_zero_check"
        allocated = int(cuda.memory_allocated())
        diagnostic["allocated_bytes_after_empty_cache"] = allocated
        if hasattr(cuda, "memory_reserved"):
            diagnostic["reserved_bytes_after_empty_cache"] = int(cuda.memory_reserved())
        require(allocated == 0, "Native GPU allocations remain: safe pause unproven")
        return allocated
    except BaseException as error:
        error.gpu_cleanup_diagnostics = diagnostic.copy()
        raise
