"""Verify activation and immutable B1/B2 freezes; never access dataset roots."""

import argparse
import json
from pathlib import Path

from visionguard.heldout_contract import FREEZE, build_freeze, verify_freeze
from visionguard.heldout_storage import json_once


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path.cwd())
    p.add_argument("--create", action="store_true")
    a = p.parse_args()
    if a.create:
        path = a.repository / FREEZE
        path.parent.mkdir(parents=True, exist_ok=True)
        json_once(path, build_freeze(a.repository))
    print(json.dumps(verify_freeze(a.repository), sort_keys=True))


if __name__ == "__main__":
    main()
