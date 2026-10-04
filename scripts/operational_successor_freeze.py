"""Create once only the distinct v3 operational freeze, or verify it."""

import argparse
import json
from pathlib import Path

from visionguard.operational_successor import FREEZE, build_freeze, verify_freeze
from visionguard.visa_b2_storage import json_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--create-successor", action="store_true")
    args = parser.parse_args()
    if args.create_successor:
        path = args.repository / FREEZE
        path.parent.mkdir(parents=True, exist_ok=True)
        json_once(path, build_freeze(args.repository))
    saved = verify_freeze(args.repository)
    print(
        json.dumps({"fingerprint": saved["fingerprint"], "evaluation_lock": "CLOSED"})
    )


if __name__ == "__main__":
    main()
