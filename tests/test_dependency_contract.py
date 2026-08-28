from __future__ import annotations

from pathlib import Path

from visionguard.protocol import load_protocol


def pinned_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        requirement = requirement.split(";", 1)[0].strip()
        name, version = requirement.split("==", 1)
        versions[name.lower().replace("_", "-")] = version
    return versions


def test_hardware_manifests_match_frozen_protocol() -> None:
    protocol = load_protocol(Path("configs/protocols/patchcore-mvtecad2-v1.yaml"))[
        "protocol"
    ]["dependencies"]
    for environment in ("torch-cpu.txt", "torch-cu126.txt"):
        versions = pinned_versions(Path("requirements/environments") / environment)
        assert versions["torch"] == str(protocol["torch"])
        assert versions["torchvision"] == str(protocol["torchvision"])


def test_common_ml_lock_matches_frozen_direct_dependencies() -> None:
    protocol = load_protocol(Path("configs/protocols/patchcore-mvtecad2-v1.yaml"))[
        "protocol"
    ]["dependencies"]
    versions = pinned_versions(Path("requirements/locks/ml-common-py311-py312.txt"))

    assert versions["anomalib"] == str(protocol["anomalib"])
    assert versions["timm"] == str(protocol["timm"])
