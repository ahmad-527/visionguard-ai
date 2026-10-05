"""Manufactured metadata workloads only; bounded file count/size, no live root."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from visionguard.heldout_accounting import Ledger
from visionguard.heldout_paths import inventory
from visionguard.visa_acquire import sha256_file
from visionguard.visa_b2_storage import json_once
from visionguard.visa_evaluator import require


def measured(call, repeats=5):
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        values.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(values), "samples_seconds": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.absolute()
    require(not root.exists(), "Synthetic benchmark output must be new")
    root.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    sources = (
        "scripts/benchmark_output_accounting.py",
        "src/visionguard/heldout_accounting.py",
        "src/visionguard/heldout_changes.py",
        "src/visionguard/heldout_paths.py",
    )
    hashes = {p: sha256_file(repository / p) for p in sources}
    rows = []
    for count in (1000, 5000, 20000):
        tree = root / f"manufactured-{count}"
        tree.mkdir()
        for i in range(count):
            directory = tree / f"category-{i % 12}" / f"stage-{i % 36}"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"synthetic-{i}.bin").open("xb") as stream:
                stream.write(b"s" * 256)
        old = measured(
            lambda tree=tree: sum((tree / p).stat().st_size for p in inventory(tree))
        )
        started = time.perf_counter()
        ledger = Ledger(tree)
        baseline = time.perf_counter() - started
        try:
            ledger.refresh()  # Drain synthetic creation/baseline notifications.
            quiet = measured(ledger.refresh)
            full = measured(lambda ledger=ledger: ledger.refresh(full=True))
            changed = tree / "synthetic-addition.bin"
            started = time.perf_counter()
            with changed.open("xb") as stream:
                stream.write(b"a" * 1024)
            ledger.record(changed)
            ledger.refresh()
            write = time.perf_counter() - started
            ledger.refresh(full=True)
            require(ledger.used == count * 256 + 1024, "Synthetic accounting mismatch")
            rows.append(
                {
                    "file_count": count,
                    "payload_bytes_per_file": 256,
                    "legacy_full_scan": old,
                    "new_initial_seconds": baseline,
                    "new_idle_refresh": quiet,
                    "new_full_reconciliation": full,
                    "addition_registration_refresh_seconds": write,
                    "independent_expected_bytes": count * 256 + 1024,
                    "observed_bytes": ledger.used,
                    "ledger_shallow_python_bytes": sys.getsizeof(ledger.files)
                    + sum(
                        sys.getsizeof(k)
                        + sys.getsizeof(v)
                        + sum(sys.getsizeof(item) for item in v)
                        for k, v in ledger.files.items()
                    ),
                    "mode": "windows_notifications"
                    if ledger.observer
                    else "portable_full_reconciliation",
                }
            )
        finally:
            ledger.close()
    require(
        hashes == {p: sha256_file(repository / p) for p in sources},
        "Benchmark source changed during execution: preserve attempt for review",
    )
    json_once(
        root / "benchmark.json",
        {
            "evidence_class": "manufactured_metadata_only_not_inference_performance",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "clock": "perf_counter",
            "repeats": 5,
            "source_sha256": hashes,
            "rows": rows,
            "limitations": [
                "warm-cache sequential comparison, not randomized",
                "small logical files on engineering volume; not live USB",
                "no prediction, map, dataset or model access",
                "idle notification refresh excludes provenance/hardware checks",
                "full reconciliation remains O(number of files)",
            ],
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
