"""Small generated-input CUDA ownership probe: never loads data/checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import weakref
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier probe evidence; select a new output")
    import numpy as np
    import torch

    from visionguard.heldout_backend import NativeBackend
    from visionguard.heldout_cleanup import release_gpu
    from visionguard.visa_evaluator_backend import SyntheticFrame
    from visionguard.visa_evaluator_synthetic import fixture

    repository = Path(__file__).resolve().parents[1]
    import visionguard

    if not Path(visionguard.__file__).resolve().is_relative_to(repository / "src"):
        raise RuntimeError("Wrong imported checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("Native CUDA unavailable; do not substitute a mock result")

    def guard_diagnostic():
        try:
            release_gpu(True)
        except ValueError as error:
            return error.gpu_cleanup_diagnostics.copy()
        raise AssertionError("A deliberately live allocation must fail the zero guard")

    model_refs = []
    tensor_refs = []

    class GeneratedModel(torch.nn.Module):
        def __init__(self, fail=False):
            super().__init__()
            self.fail = fail
            self.register_buffer(
                "generated_storage", torch.zeros(262144, device="cuda")
            )
            model_refs.append(weakref.ref(self))

        def forward(self, tensor):
            tensor_refs.append(weakref.ref(tensor))
            if self.fail:
                temporary = torch.empty(131072, device=tensor.device)
                tensor_refs.append(weakref.ref(temporary))
                raise RuntimeError("Declared generated-model traceback probe")
            return SimpleNamespace(
                pred_score=tensor.mean(dim=(1, 2, 3)),
                anomaly_map=tensor[:, :1].square(),
            )

    def transform(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    spec = fixture("candle", 42)[3][0]
    frame = SyntheticFrame(
        "synthetic:candle:generated-reference-probe",
        "candle",
        np.zeros((8, 9, 3), dtype=np.uint8),
    )
    release_gpu(True)
    normal = NativeBackend(spec, GeneratedModel(), transform, "cuda")
    initial_digest = normal.state_digest()
    outputs = [normal.predict_synthetic(frame) for _ in range(16)]
    if normal.state_digest() != initial_digest:
        raise AssertionError("Generated native probe mutated model state")
    live_backend = guard_diagnostic()
    del normal
    gc.collect()
    normal_after_release = release_gpu(True)
    if model_refs[0]() is not None or any(ref() is not None for ref in tensor_refs):
        raise AssertionError("Normal generated prediction retained model/GPU input")
    if any(output.restored_map.dtype != np.float32 for output in outputs):
        raise AssertionError("Generated outputs should be host float32 arrays")

    def traceback_probe():
        native = NativeBackend(spec, GeneratedModel(fail=True), transform, "cuda")
        native.predict_synthetic(frame)

    try:
        traceback_probe()
    except RuntimeError as error:
        retained = error
    else:
        raise AssertionError("Declared generated failure did not occur")
    traceback_retention = guard_diagnostic()
    before_detach = model_refs[-1]() is not None
    retained.__traceback__ = None
    retained.__context__ = None
    retained.__cause__ = None
    gc.collect()
    traceback_after_release = release_gpu(True)
    if model_refs[-1]() is not None or any(ref() is not None for ref in tensor_refs):
        raise AssertionError("Detached generated traceback still retained allocations")

    result = {
        "evidence_class": "GENERATED_NON_HELDOUT_ENGINEERING_ONLY",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "imported_checkout": str(repository),
        "torch_version": torch.__version__,
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(),
        "generated_predictions": 16,
        "live_backend_guard_failure": live_backend,
        "allocated_bytes_after_normal_backend_release": normal_after_release,
        "normal_input_tensor_references_released": True,
        "retained_host_arrays_did_not_keep_gpu_inputs_alive": True,
        "traceback_guard_failure": traceback_retention,
        "traceback_kept_generated_model_alive": before_detach,
        "allocated_bytes_after_traceback_detach": traceback_after_release,
        "zero_allocation_guard_preserved": True,
        "original_incident_root_cause_proven": False,
        "dataset_or_checkpoint_access": False,
        "original_run_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, sort_keys=True, indent=2)
        target.write("\n")
    print(json.dumps({"output": str(args.output), "observations": result}, indent=2))


if __name__ == "__main__":
    main()
