"""Best-effort reproducibility configuration with honest capability reporting."""

from __future__ import annotations

import random
from typing import Any

from visionguard.experiment import ReproducibilityConfig


def configure_reproducibility(config: ReproducibilityConfig) -> dict[str, Any]:
    """Apply available seed/backend controls and report what was configured."""

    random.seed(config.seed)
    report: dict[str, Any] = {
        "python": {"status": "configured", "seed": config.seed},
        "numpy": {"status": "unavailable", "reason": "numpy not installed"},
        "torch": {"status": "unavailable", "reason": "torch not installed"},
        "limitations": [
            "Seed and deterministic backend settings do not prove bitwise "
            "reproducibility across devices, package versions, or operations."
        ],
    }
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(config.seed)
        report["numpy"] = {"status": "configured", "seed": config.seed}
    try:
        import torch
    except ImportError:
        return report
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.deterministic_algorithms
    report["torch"] = {
        "status": "configured",
        "seed": config.seed,
        "cuda_seed_all": torch.cuda.is_available(),
        "deterministic_algorithms": config.deterministic_algorithms,
        "cudnn_benchmark": config.cudnn_benchmark,
        "cudnn_deterministic": config.deterministic_algorithms,
    }
    return report
