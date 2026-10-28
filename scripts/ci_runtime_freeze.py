"""Verify v6 and original-reference history; no activation."""

import json
from pathlib import Path

from visionguard.ci_runtime_successor import verify_freeze

record = verify_freeze(Path.cwd())
print(
    json.dumps(
        {
            "fingerprint": record["fingerprint"],
            "evidence_class": record["document"]["evidence_class"],
            "evaluation_lock": "CLOSED",
            "real_execution_authorized": False,
        }
    )
)
