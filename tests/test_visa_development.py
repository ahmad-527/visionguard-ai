"""Synthetic-only normal adapter and reproducible membership evidence tests."""

import json

import pytest
from PIL import Image

from visionguard.visa_acquire import VisaIntegrityError, atomic_json, sha256_file
from visionguard.visa_audit import inspect_asset
from visionguard.visa_development import DevelopmentDataset, materialize_development


@pytest.fixture
def synthetic_development(tmp_path):
    sealed = tmp_path / "sealed"
    sample = "candle/Data/Images/Normal/synthetic.png"
    image = sealed / sample
    image.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image)
    inventory = tmp_path / "inventory.json"
    atomic_json(
        inventory,
        {
            "records": [
                {
                    "sample_id": sample,
                    "image": sample,
                    "category": "candle",
                    "split": "train",
                    "label": "normal",
                    "role": "fit",
                    "image_identity": inspect_asset(image),
                }
            ]
        },
    )
    audit = tmp_path / "audit.json"
    atomic_json(audit, {"status": "passed", "inventory_sha256": sha256_file(inventory)})
    membership = tmp_path / "membership.json"
    root = tmp_path / "development"
    digest = materialize_development(
        sealed,
        root,
        audit_path=audit,
        inventory_path=inventory,
        membership_path=membership,
    )
    return root, membership, digest, sealed, audit, inventory


def test_reproducible_membership_and_no_test_role(synthetic_development, tmp_path):
    root, membership, digest, sealed, audit, inventory = synthetic_development
    second = tmp_path / "second.json"
    assert (
        materialize_development(
            sealed,
            tmp_path / "second-root",
            audit_path=audit,
            inventory_path=inventory,
            membership_path=second,
        )
        == digest
    )
    assert membership.read_bytes() == second.read_bytes()
    dataset = DevelopmentDataset(root, membership, expected_sha256=digest)
    assert len(dataset.paths("candle", "fit")) == 1
    with pytest.raises(VisaIntegrityError, match="fit/calibration"):
        dataset.paths("candle", "test")
    with pytest.raises(VisaIntegrityError, match="overwrite"):
        materialize_development(
            sealed,
            root,
            audit_path=audit,
            inventory_path=inventory,
            membership_path=membership,
        )


def test_adapter_rejects_modified_image_and_identity(synthetic_development):
    root, membership, digest, *_ = synthetic_development
    with pytest.raises(VisaIntegrityError, match="identity"):
        DevelopmentDataset(root, membership, expected_sha256="0" * 64)
    dataset = DevelopmentDataset(root, membership, expected_sha256=digest)
    image = dataset.paths("candle", "fit")[0]
    image.write_bytes(b"changed synthetic image")
    with pytest.raises(VisaIntegrityError, match="hash"):
        dataset.paths("candle", "fit")


@pytest.mark.parametrize(
    "field,value", [("role", "test"), ("label", "anomaly"), ("mask", "synthetic.png")]
)
def test_no_hidden_label_dependency(synthetic_development, field, value):
    root, membership, _, *_ = synthetic_development
    document = json.loads(membership.read_text())
    document["records"][0][field] = value
    atomic_json(membership, document)
    digest = sha256_file(membership)
    atomic_json(root / "development-seal.json", {"membership_sha256": digest})
    with pytest.raises(VisaIntegrityError, match="metadata"):
        DevelopmentDataset(root, membership, expected_sha256=digest)
