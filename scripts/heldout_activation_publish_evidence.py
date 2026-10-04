"""Compact verified engineering evidence; never read a dataset or real approval."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from visionguard.heldout_storage import json_once
from visionguard.visa_acquire import sha256_file
from visionguard.visa_evaluator import require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path.cwd())
    a = p.parse_args()
    repo = a.repository.absolute()
    local = repo / "outputs/phase4d-b2-controlled-activation"
    report = repo / "reports/phase4d-b2-controlled-activation"
    report.mkdir(parents=True, exist_ok=True)

    def load(relative):
        path = local / relative
        return json.loads(path.read_text()), sha256_file(path)

    models, models_sha = load("fresh-development-artifact-verification.json")
    native, native_sha = load("native-interface-2/native-smoke.json")
    acceptance, acceptance_sha = load("durable-acceptance-2/acceptance.json")
    first_failure, first_failure_sha = load(
        "native-interface-1/patchcore/attempt-1/failure.json"
    )
    require(
        models["status"] == "72_development_cells_verified_read_only"
        and len(models["cells"]) == 72,
        "Unverified models",
    )
    require(
        all(
            r["score_exact"]
            and r["restored_map_bitwise_equal"]
            and r["state_unchanged"]
            for r in native["models"].values()
        ),
        "Native equivalence failure",
    )
    require(
        acceptance["bitwise_numeric_B1_equivalence"]
        and acceptance["paired_cells"] == 36
        and acceptance["new_full_replay_calls"] == 0,
        "Artificial matrix incomplete",
    )
    xml = local / "full-pytest.xml"
    suites = ET.parse(xml).getroot()
    counts = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites.iter("testsuite"))
        for k in ("tests", "failures", "errors", "skipped")
    }
    require(counts["failures"] == counts["errors"] == 0, "Full pytest failed")
    # Drop process IDs, GPU UUIDs, monotonic clock origins and local personal paths.
    native_clean = json.loads(json.dumps(native))
    for r in native_clean["models"].values():
        obs = r["observations"]
        obs.pop("pid", None)
        obs.pop("monotonic_seconds", None)
        obs["gpu_point"].pop("uuid", None)
    implementation = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    json_once(
        report / "acceptance-evidence.json",
        {
            "schema_version": 1,
            "implementation_commit": implementation,
            "development": models,
            "development_local_receipt_sha256": models_sha,
            "native_acceptance": native_clean,
            "native_local_receipt_sha256": native_sha,
            "artificial_matrix": acceptance,
            "artificial_local_receipt_sha256": acceptance_sha,
            "negative_candidates": {
                "native_relative_path_guard": {
                    "receipt_sha256": first_failure_sha,
                    "failure": first_failure,
                    "additional_manufactured_native_calls": 2,
                },
                "first_durable_oracle": {
                    "only_per_image_order_differed": True,
                    "all_metrics_and_aggregates_exact": True,
                    "reference_order_corrected_without_tolerance": True,
                    "artifacts_retained": True,
                },
            },
            "full_pytest": counts,
            "pytest_passed": counts["tests"] - counts["skipped"],
            "pytest_raw_xml_sha256": sha256_file(xml),
            "observed_D_free_bytes": shutil.disk_usage(repo).free,
            "real_trust_registry_created": False,
            "real_human_approval_created": False,
            "real_authorization_claim_created": False,
            "real_test_asset_access": False,
            "real_test_performance_evaluations": 0,
            "final_test_lock": "CLOSED",
            "human_access_history": "UNKNOWN",
            "independent_reservation": "NOT ESTABLISHED",
            "original_recoveries_untouched": True,
        },
    )
    print("Compact acceptance evidence published; no real access authorized")


if __name__ == "__main__":
    main()
