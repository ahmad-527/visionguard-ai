"""Official VisA acquisition with source identities and guarded extraction.

Downloads data only with an explicit command. No model or image display code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = "2a692ab575001cbde74d402d897a7286086c6199"
ARCHIVE_URL = (
    "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
)
ARCHIVE_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
SOURCE_FILES = (
    "split_csv/1cls.csv",
    "utils/prepare_data.py",
    "LICENSE-DATASET",
    "README.md",
    "utils/id2class.py",
)


class VisaIntegrityError(ValueError):
    """Acquisition or dataset evidence fails a mandatory integrity gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Publish generated metadata only after its complete file is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VisionGuard-audit"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def acquire_sources(root: Path) -> dict:
    """Match actual source bytes to pinned upstream Git blob objects and SHA-256."""
    tree = json.loads(
        _fetch(
            f"https://api.github.com/repos/amazon-science/spot-diff/git/trees/{SOURCE_COMMIT}?recursive=1"
        )
    )
    if tree.get("truncated"):
        raise VisaIntegrityError("Official source tree is incomplete")
    entries = {item["path"]: item for item in tree["tree"]}
    identities = {}
    for relative in SOURCE_FILES:
        entry = entries.get(relative)
        if not entry or entry["type"] != "blob":
            raise VisaIntegrityError(f"Required official source absent: {relative}")
        url = f"https://raw.githubusercontent.com/amazon-science/spot-diff/{SOURCE_COMMIT}/{relative}"
        path = root / relative
        data = path.read_bytes() if path.exists() else _fetch(url)
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if blob != entry["sha"]:
            raise VisaIntegrityError(f"Official source object mismatch: {relative}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        identities[relative] = {
            "url": url,
            "git_blob_sha1": blob,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    return {
        "repository": "amazon-science/spot-diff",
        "commit": SOURCE_COMMIT,
        "git_tree_request_revision": SOURCE_COMMIT,
        "files": identities,
    }


def acquire_archive(path: Path) -> dict:
    """Reuse a verified archive; safely resume partial official-source downloads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = sha256_file(path)
    else:
        partial = path.with_suffix(".tar.partial")
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            ARCHIVE_URL,
            headers={
                "User-Agent": "VisionGuard-audit",
                "Range": f"bytes={offset}-",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise VisaIntegrityError("Download resume range mismatch")
            remaining = int(response.headers["Content-Length"])
            if shutil.disk_usage(path.parent).free < remaining + 30 * 1024**3:
                raise VisaIntegrityError(
                    "Insufficient free disk for download plus extraction"
                )
            hasher = hashlib.sha256()
            if resumed:
                with partial.open("rb") as prior:
                    for chunk in iter(lambda: prior.read(8 * 1024**2), b""):
                        hasher.update(chunk)
            received = 0
            with partial.open("ab" if resumed else "wb") as target:
                for chunk in iter(lambda: response.read(8 * 1024**2), b""):
                    target.write(chunk)
                    hasher.update(chunk)
                    received += len(chunk)
                    if received // (512 * 1024**2) != (received - len(chunk)) // (
                        512 * 1024**2
                    ):
                        print(
                            json.dumps(
                                {
                                    "downloaded_mib": (
                                        received + (offset if resumed else 0)
                                    )
                                    // 1024**2
                                }
                            ),
                            flush=True,
                        )
                target.flush()
                os.fsync(target.fileno())
            if received != remaining:
                raise VisaIntegrityError(
                    "Download size mismatch; partial file retained"
                )
            digest = hasher.hexdigest()
        if digest != ARCHIVE_SHA256:
            raise VisaIntegrityError(
                "Unexpected archive SHA-256; partial retained, STOP"
            )
        partial.replace(path)
    if digest != ARCHIVE_SHA256:
        raise VisaIntegrityError("Existing archive identity unexpected; STOP")
    return {
        "source": ARCHIVE_URL,
        "release": path.name,
        "size_bytes": path.stat().st_size,
        "observed_sha256": digest,
        "secondary_expected_sha256": ARCHIVE_SHA256,
        "status": "verified",
    }


def safe_member_path(name: str) -> PurePosixPath:
    """Tar members must stay portable, relative, and free of special device names."""
    if name.startswith("./"):
        name = name[2:]
    parts = name.rstrip("/").split("/")
    forbidden = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if (
        not name
        or any(ord(character) < 32 for character in name)
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(
            p in ("", ".", "..")
            or p.rstrip(" .") != p
            or p.split(".")[0].upper() in forbidden
            for p in parts
        )
    ):
        raise VisaIntegrityError("Unsafe archive member path")
    return PurePosixPath(*parts)


def extract_verified_archive(archive: Path, target: Path) -> dict:
    """Extract regular files only. Refuse symlinks, traversal, and collisions.

    Caller must first verify archive bytes. Partial extraction is not reused;
    it is retained for explicit inspection rather than overwritten silently.
    """
    receipt = target.parent / "extraction.json"
    if target.exists():
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if (
                prior.get("archive_sha256") == ARCHIVE_SHA256
                and prior.get("status") == "completed"
            ):
                return prior
        raise VisaIntegrityError("Existing extraction has no valid completion receipt")
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        safe = []
        seen = set()
        for member in members:
            if member.isdir() and member.name in (".", "./"):
                continue
            relative = safe_member_path(member.name)
            key = str(relative).casefold()
            if key in seen or not (member.isfile() or member.isdir()):
                raise VisaIntegrityError("Duplicate/special archive member; STOP")
            seen.add(key)
            safe.append((member, relative))
        needed = sum(m.size for m, _ in safe if m.isfile())
        if (
            needed > 40 * 1024**3
            or shutil.disk_usage(target.parent).free < needed + 5 * 1024**3
        ):
            raise VisaIntegrityError("Archive extraction exceeds disk budget")
        target.mkdir()
        for member, relative in safe:
            destination = target.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (
                    tar.extractfile(member) as source,
                    destination.open("xb") as output,
                ):
                    shutil.copyfileobj(source, output, 8 * 1024**2)
    record = {
        "archive_sha256": ARCHIVE_SHA256,
        "status": "completed",
        "file_count": sum(m.isfile() for m, _ in safe),
        "extracted_bytes": needed,
    }
    atomic_json(receipt, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--sources-only", action="store_true")
    args = parser.parse_args(argv)
    source = acquire_sources(args.local_root / "source")
    record = json.loads(args.record.read_text()) if args.record.exists() else {}
    record.update(
        {
            "schema_version": 1,
            "dataset": "VisA",
            "license": "CC-BY-4.0",
            "source": source,
            "audit_schema_version": 1,
        }
    )
    atomic_json(args.record, record)
    if not args.sources_only:
        archive = args.local_root / "VisA_20220922.tar"
        record["archive"] = acquire_archive(archive)
        atomic_json(args.record, record)
        record["extraction"] = extract_verified_archive(
            archive, args.local_root / "sealed"
        )
        atomic_json(args.record, record)
    print(
        json.dumps(
            {
                "status": "completed",
                "sources": len(source["files"]),
                "archive": record.get("archive"),
                "extraction": record.get("extraction"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
