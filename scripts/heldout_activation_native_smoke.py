"""Four bounded manufactured-RGB native calls: exact old/new interface comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from visionguard.heldout_admission import Frame
from visionguard.heldout_backend import RealFrame, load_backend
from visionguard.heldout_contract import context
from visionguard.heldout_metrics import Sample
from visionguard.heldout_resources import observation
from visionguard.heldout_stages import publish_pair, publish_stage
from visionguard.heldout_storage import json_once
from visionguard.visa_evaluator import require
from visionguard.visa_evaluator_backend import SyntheticFrame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.absolute()
    a.repository = a.repository.absolute()
    import torch

    require(torch.cuda.is_available(), "CUDA unavailable: STOP, no hardware retry")
    ctx = context(a.repository)
    category = "pcb3"
    seed = 42
    require(
        ctx["published"]["cells"]["patchcore:pcb3:42"]["fit_count"]
        == max(
            r["fit_count"]
            for k, r in ctx["published"]["cells"].items()
            if k.startswith("patchcore:")
        ),
        "Predeclared largest fitting category differs",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    yy, xx = np.indices((384, 512))
    rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=2).astype(np.uint8)
    identifier = f"{category}/Data/Images/Normal/artificial-interface-smoke.png"
    sample = Sample(identifier, category, 0, np.zeros((384, 512), np.uint8))
    origin = {
        "evidence_class": "manufactured_native_interface_equivalence",
        "membership": {category: [identifier]},
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }
    records = {}
    stages = []
    specs = []
    for model in ("patchcore", "efficientad"):
        spec = ctx["specs"][f"{model}:{category}:{seed}"]
        specs.append(spec)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        backend = load_backend(
            a.repository,
            a.repository / "outputs/phase4d-a-visa-development-matrix",
            spec,
        )
        restore_seconds = time.perf_counter() - start
        before = backend.state_digest()

        class Comparing:
            def __init__(self, inner, specification, state, restore):
                self.inner, self.spec, self.state, self.restore = (
                    inner,
                    specification,
                    state,
                    restore,
                )

            def predict(self, frame: RealFrame):
                start = time.perf_counter()
                actual = self.inner.predict(frame)
                reference = self.inner.predict_synthetic(
                    SyntheticFrame(
                        f"synthetic:{category}:manufactured-interface",
                        category,
                        frame.rgb,
                    )
                )
                require(
                    actual.score == reference.score
                    and np.array_equal(actual.restored_map, reference.restored_map),
                    "BITWISE native old/new mismatch: human review required",
                )
                records[self.spec.model] = {
                    "model_sha256": self.spec.model_sha256,
                    "state_sha256": self.state,
                    "restore_seconds": self.restore,
                    "two_interface_calls_seconds": time.perf_counter() - start,
                    "score_exact": True,
                    "restored_map_bitwise_equal": True,
                    "map_dtype": str(actual.restored_map.dtype),
                    "shape": list(actual.restored_map.shape),
                    "map_sha256": hashlib.sha256(
                        actual.restored_map.tobytes()
                    ).hexdigest(),
                    "observations": observation(),
                }
                return actual

        path = a.output / model / "attempt-1"
        receipt = publish_stage(
            path,
            [Frame(sample, rgb)],
            spec,
            origin,
            Comparing(backend, spec, before, restore_seconds),
        )
        require(
            backend.state_digest() == before,
            "State changed after native synthetic inference",
        )
        records[model]["state_unchanged"] = True
        stages.append((path, receipt["sha256"], origin))
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    _result, receipt = publish_pair(
        a.output / "pair/attempt-1", stages, tuple(specs), origin
    )
    report = {
        "evidence_class": "manufactured_native_equivalence_not_benchmark",
        "models": records,
        "native_calls": 4,
        "original_dimension_restoration": True,
        "pair_independent_replays": 2,
        "publication_sha256": receipt["sha256"],
        "real_test_asset_access": False,
        "real_test_performance_evaluations": 0,
        "final_test_lock": "CLOSED",
    }
    json_once(a.output / "native-smoke.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
