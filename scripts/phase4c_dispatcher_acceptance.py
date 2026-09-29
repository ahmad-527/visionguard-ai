"""Declared normal-only dispatcher interruption acceptance; no test evaluation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "development",
        "weight",
        "teacher",
        "imagenette-root",
        "imagenette-archive",
        "root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "-m",
        "visionguard.visa_dispatcher",
        "--repository",
        str(args.repository.resolve()),
        "--development",
        str(args.development.resolve()),
        "--engineering-smoke",
    ]
    reports = {}
    for model, interruptions in (
        (
            "patchcore",
            ["embedding:4", "before_coreset", "after_fit", "after_calibration"],
        ),
        ("efficientad", ["step:1", "after_normalization"]),
    ):
        assets = (
            ["--weight", str(args.weight.resolve())]
            if model == "patchcore"
            else [
                "--teacher",
                str(args.teacher.resolve()),
                "--imagenette-root",
                str(args.imagenette_root.resolve()),
                "--imagenette-archive",
                str(args.imagenette_archive.resolve()),
            ]
        )
        for kind, sequence in (
            ("uninterrupted", [None]),
            ("restarted", [*interruptions, None]),
        ):
            output = args.root / f"{model}-{kind}"
            for index, stop in enumerate(sequence):
                command = [
                    *common,
                    "--model",
                    model,
                    "--output",
                    str(output.resolve()),
                    *assets,
                    "--run-cell" if index == 0 else "--resume-cell",
                    "candle",
                    "42",
                ]
                if stop:
                    command.extend(["--stop-at", stop])
                log = args.root / f"{model}-{kind}-{index}.log"
                with log.open("x", encoding="utf-8") as stream:
                    result = subprocess.run(
                        command, stdout=stream, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode != 0:
                    raise VisaIntegrityError(
                        f"Dispatcher acceptance failed: {model}/{kind}/{index}; STOP"
                    )
                print(
                    json.dumps(
                        {
                            "model": model,
                            "path": kind,
                            "completed_command": index,
                            "stop": stop,
                        }
                    ),
                    flush=True,
                )
            manifest = load(output / model / "manifest.json")
            attempt = manifest["cells"]["candle:42"]["attempts"][-1]
            if attempt["status"] != "development_complete":
                raise VisaIntegrityError(
                    "Dispatcher did not complete declared engineering cell"
                )
            directory = output / model / attempt["directory"]
            reports[f"{model}-{kind}"] = {
                "state": load(directory / "worker-state.json"),
                "calibration": load(directory / "calibration.json"),
                "manifest_sha256": sha256_file(output / model / "manifest.json"),
                "receipt_sha256": sha256_file(directory / "worker-state.json"),
                "attempt_count": len(manifest["cells"]["candle:42"]["attempts"]),
                "directory": directory.relative_to(args.root).as_posix(),
            }
        a, b = reports[f"{model}-uninterrupted"], reports[f"{model}-restarted"]
        if (
            a["state"]["canonical_final_model_sha256"]
            != b["state"]["canonical_final_model_sha256"]
            or a["calibration"] != b["calibration"]
        ):
            atomic_json(
                args.root / "FAILED-equivalence.json",
                {"model": model, "status": "STOP_NO_TOLERANCE"},
            )
            raise VisaIntegrityError("Dispatcher resume changed outputs; STOP")
    atomic_json(
        args.root / "acceptance.json",
        {
            "status": "exact",
            "test_performance_evaluated": False,
            "final_test_lock": "closed",
            "driver_sha256": sha256_file(Path(__file__)),
            "models": {
                model: {
                    "canonical_model_and_calibration_exact": True,
                    "uninterrupted": {
                        k: v
                        for k, v in reports[f"{model}-uninterrupted"].items()
                        if k not in ("state", "calibration")
                    },
                    "restarted": {
                        k: v
                        for k, v in reports[f"{model}-restarted"].items()
                        if k not in ("state", "calibration")
                    },
                }
                for model in ("patchcore", "efficientad")
            },
        },
    )
    print(json.dumps({"dispatcher_acceptance": "exact", "final_test_lock": "closed"}))


if __name__ == "__main__":
    main()
