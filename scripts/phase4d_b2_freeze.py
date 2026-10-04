"""Create/reproduce B2 readiness source freeze; never authorize evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionguard.visa_b2_contract import FREEZE, build_freeze, verify_freeze
from visionguard.visa_b2_storage import json_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        snapshot = build_freeze(args.repository)
        path = args.repository / FREEZE
        path.parent.mkdir(parents=True, exist_ok=True)
        json_once(path, snapshot)
    else:
        snapshot = verify_freeze(args.repository)
    print(
        json.dumps(
            {"fingerprint": snapshot["fingerprint"], "real_execution_authorized": False}
        )
    )


if __name__ == "__main__":
    main()
