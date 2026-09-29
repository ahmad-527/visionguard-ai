"""Reuse unchanged Phase 4C normal-only acceptance; no inference or training."""

import json
from pathlib import Path

from visionguard.visa_acquire import atomic_json, sha256_file
from visionguard.visa_matrix_validation import validate_cell

repository = Path(__file__).resolve().parents[1]
membership = json.loads(
    (
        repository / "reports/phase4c-visa-readiness/development-membership.json"
    ).read_text()
)
results = {}
for model in ("patchcore", "efficientad"):
    for kind in ("uninterrupted", "restarted"):
        root = (
            repository / f"outputs/phase4c-dispatcher-acceptance/{model}-{kind}/{model}"
        )
        manifest = json.loads((root / "manifest.json").read_text())
        result = validate_cell(
            root,
            model,
            "candle",
            42,
            manifest["identity"],
            membership,
            engineering=True,
        )
        results[f"{model}-{kind}"] = {
            "canonical_model_sha256": result["canonical_model_sha256"],
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "status": result["status"],
        }
atomic_json(
    repository / "reports/phase4d-a-visa-development-matrix/preflight-validation.json",
    {
        "reused_phase4c_evidence": results,
        "test_performance_evaluated": False,
        "inference_or_training_performed": False,
    },
)
