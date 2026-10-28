"""Admit only ordinary tensor-state ZIPs, before any Torch deserialization.

This excludes PT2/export and TorchScript containers; it is not a replacement for
restricted loading, pinned bytes/permission, or upstream security remediation.
No pickle member is read or executed by this metadata-only format check.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath

from visionguard.inspection_contract import InspectionError

_METADATA = {
    "data.pkl",
    "version",
    "byteorder",
    ".format_version",
    ".storage_alignment",
    ".data/serialization_id",
}


def require_state_archive(raw: bytes) -> None:
    """Reject unsupported containers without importing Torch or reading pickle."""
    if not raw.startswith(b"PK\x03\x04"):
        raise InspectionError("Ordinary tensor-state ZIP required; no legacy fallback")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if not 0 < len(entries) <= 8192:
                raise InspectionError("Tensor-state archive member count is invalid")
            names = [item.orig_filename for item in entries]
            if any("\x00" in name for name in names):
                raise InspectionError("NUL in raw tensor-state archive name")
            if len(set(names)) != len(names):
                raise InspectionError("Duplicate tensor-state archive members")
            root = names[0].split("/", 1)[0]
            suffixes = set()
            byte_budget = len(raw)
            declared_total = 0
            for item in entries:
                name = item.orig_filename
                path = PurePosixPath(name)
                if (
                    str(path) != name
                    or path.is_absolute()
                    or len(path.parts) < 2
                    or path.parts[0] != root
                    or set(path.parts) & {".", ".."}
                    or "\\" in name
                    or ":" in name
                    or item.is_dir()
                    or item.flag_bits & 1
                    or item.compress_type != zipfile.ZIP_STORED
                ):
                    raise InspectionError("Unsupported tensor-state archive member")
                if item.compress_size != item.file_size:
                    raise InspectionError("Inconsistent stored-member sizes")
                if not 0 <= item.file_size <= byte_budget:
                    raise InspectionError("Stored member exceeds archive byte budget")
                declared_total += item.file_size
                if declared_total > byte_budget:
                    raise InspectionError("Stored members exceed aggregate byte budget")
                relative = "/".join(path.parts[1:])
                tensor_record = (
                    len(path.parts) == 3
                    and path.parts[1] == "data"
                    and path.parts[2].isascii()
                    and path.parts[2].isdigit()
                )
                if relative not in _METADATA and not tensor_record:
                    raise InspectionError(
                        "PT2/export, TorchScript or other non-state container refused"
                    )
                suffixes.add(relative)
            if not {"data.pkl", "version"} <= suffixes:
                raise InspectionError("Tensor-state ZIP is missing required records")
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        if isinstance(error, InspectionError):
            raise
        raise InspectionError("Invalid tensor-state ZIP") from error
