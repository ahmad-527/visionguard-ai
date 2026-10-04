"""Read only Windows trust ancestors; never registry content or dataset roots."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from visionguard.visa_b2_storage import json_once
from visionguard.windows_trust_acl import ANCESTOR_DANGER, collect_chain, validate_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    snapshot = collect_chain(ancestors_only=True)
    validate_chain(snapshot, ancestors_only=True)
    old_rejections = [
        {
            "path": row["path"],
            "sid": r["sid"],
            "rights": r["rights"],
            "propagation": r["propagation"],
            "is_inherited": r["is_inherited"],
        }
        for row in snapshot["paths"]
        for r in row["rules"]
        if r["type"] == "Allow"
        and r["sid"] not in {"S-1-5-18", "S-1-5-32-544"}
        and r["rights"] & ANCESTOR_DANGER
    ]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    json_once(
        args.receipt,
        {
            "observed_utc": datetime.now(UTC).isoformat(),
            "snapshot": snapshot,
            "predecessor_rejections": old_rejections,
            "successor_ancestors": "PASS",
            "production_registry": "NOT VALIDATED",
            "dataset_operations": 0,
            "evaluation_lock": "CLOSED",
        },
    )
    print(
        json.dumps(
            {
                "successor_ancestors": "PASS",
                "old_rejections": len(old_rejections),
                "production_registry": "NOT VALIDATED",
                "evaluation_lock": "CLOSED",
            }
        )
    )


if __name__ == "__main__":
    main()
