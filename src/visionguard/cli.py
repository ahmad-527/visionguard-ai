"""Command-line interface for dataset integrity audits."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from visionguard.audit import audit_dataset
from visionguard.config import ConfigurationError, load_dataset_config


def build_parser() -> argparse.ArgumentParser:
    """Construct the audit command argument parser."""

    parser = argparse.ArgumentParser(
        description="Audit a local dataset and write measurements as JSON."
    )
    parser.add_argument("dataset_root", type=Path, help="Local dataset root")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset structure configuration (explicit for path portability)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the generated JSON audit report",
    )
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default="error",
        help="Finding severity that produces a non-zero exit status",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an audit, always writing the report before applying exit policy."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_dataset_config(args.config)
        report = audit_dataset(args.dataset_root, config)
    except (ConfigurationError, NotADirectoryError) as exc:
        logging.error("%s", exc)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    logging.info(
        "Audit %s: %d images, %d masks, %d errors, %d warnings; report: %s",
        summary["status"],
        summary["image_count"],
        summary["mask_count"],
        summary["error_count"],
        summary["warning_count"],
        args.output,
    )
    if args.fail_on == "error" and summary["error_count"]:
        return 1
    if args.fail_on == "warning" and (
        summary["error_count"] or summary["warning_count"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
