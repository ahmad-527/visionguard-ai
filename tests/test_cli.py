from __future__ import annotations

import json
from pathlib import Path

import yaml

from visionguard.cli import main
from visionguard.config import DatasetConfig


def write_config(path: Path, config: DatasetConfig) -> None:
    mask = config.splits["test_public"].mask
    assert mask is not None
    data = {
        "dataset": {
            "name": config.name,
            "version": config.version,
            "categories": list(config.categories),
            "image_extensions": list(config.image_extensions),
            "splits": {
                "train": {"layout": "classified", "conditions": ["good"]},
                "validation": {
                    "layout": "classified",
                    "conditions": ["good"],
                },
                "test_public": {
                    "layout": "classified",
                    "conditions": ["good", "bad"],
                    "mask": {
                        "image_condition": mask.image_condition,
                        "directory": mask.directory,
                        "suffix": mask.suffix,
                    },
                },
                "test_private": {"layout": "flat", "conditions": []},
                "test_private_mixed": {"layout": "flat", "conditions": []},
            },
        }
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_cli_writes_actual_json_report(
    valid_dataset: Path, dataset_config: DatasetConfig, tmp_path: Path
) -> None:
    config_path = tmp_path / "dataset.yaml"
    output_path = tmp_path / "reports/audit.json"
    write_config(config_path, dataset_config)

    result = main(
        [
            str(valid_dataset),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["image_count"] == 6
    assert Path(report["root"]) == valid_dataset.resolve()


def test_cli_returns_failure_after_writing_report(
    valid_dataset: Path, dataset_config: DatasetConfig, tmp_path: Path
) -> None:
    config_path = tmp_path / "dataset.yaml"
    output_path = tmp_path / "audit.json"
    write_config(config_path, dataset_config)
    (valid_dataset / "widget/train/good/000.png").write_bytes(b"broken")

    result = main(
        [
            str(valid_dataset),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 1
    assert json.loads(output_path.read_text())["summary"]["error_count"] == 1
