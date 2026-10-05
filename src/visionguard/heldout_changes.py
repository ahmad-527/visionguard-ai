"""Bounded Windows metadata notifications; no file-content reads.

An outstanding overlapped read covers the interval between checks. Overflow,
malformed events and lost handles are fatal, never interpreted as no changes.
"""

from __future__ import annotations

import ctypes
import struct
import threading
from pathlib import PureWindowsPath

from visionguard.visa_evaluator import require


class WindowsChanges:
    def __init__(self, root):
        from ctypes import wintypes as w

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", w.DWORD),
                ("OffsetHigh", w.DWORD),
                ("hEvent", w.HANDLE),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [
            w.LPCWSTR,
            w.DWORD,
            w.DWORD,
            w.LPVOID,
            w.DWORD,
            w.DWORD,
            w.HANDLE,
        ]
        self.kernel.CreateFileW.restype = w.HANDLE
        self.kernel.CreateEventW.argtypes = [w.LPVOID, w.BOOL, w.BOOL, w.LPCWSTR]
        self.kernel.CreateEventW.restype = w.HANDLE
        self.kernel.ReadDirectoryChangesW.argtypes = [
            w.HANDLE,
            w.LPVOID,
            w.DWORD,
            w.BOOL,
            w.DWORD,
            w.LPVOID,
            ctypes.POINTER(Overlapped),
            w.LPVOID,
        ]
        self.kernel.GetOverlappedResult.argtypes = [
            w.HANDLE,
            ctypes.POINTER(Overlapped),
            ctypes.POINTER(w.DWORD),
            w.BOOL,
        ]
        self.kernel.CancelIoEx.argtypes = [w.HANDLE, ctypes.POINTER(Overlapped)]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.ResetEvent.argtypes = [w.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.kernel.WaitForSingleObject.restype = w.DWORD
        self.handle = self.kernel.CreateFileW(
            str(root), 1, 7, None, 3, 0x42000000, None
        )  # LIST_DIRECTORY; share read/write/delete; overlapped + backup semantics.
        require(
            self.handle not in (None, ctypes.c_void_p(-1).value),
            "Output change observer unavailable",
        )
        self.event = self.kernel.CreateEventW(None, True, False, None)
        if not self.event:
            self.kernel.CloseHandle(self.handle)
            raise OSError(ctypes.get_last_error(), "Change observer event unavailable")
        self.overlapped = Overlapped(hEvent=self.event)
        self.buffer = ctypes.create_string_buffer(65536)
        self.closed = False
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.pending = {}
        self.pending_bytes = 0
        self.error = None
        self.worker = None
        try:
            self._arm()
            self.worker = threading.Thread(
                target=self._pump, name="heldout-metadata-notifications", daemon=True
            )
            self.worker.start()
        except BaseException:
            self.close()
            raise

    def _arm(self):
        self.kernel.ResetEvent(self.event)
        require(
            bool(
                self.kernel.ReadDirectoryChangesW(
                    self.handle,
                    self.buffer,
                    len(self.buffer),
                    True,
                    0x15F,
                    None,
                    ctypes.byref(self.overlapped),
                    None,
                )
            ),
            "Output change observer could not be armed",
        )

    def _read(self):
        count = ctypes.c_ulong()
        if not self.kernel.GetOverlappedResult(
            self.handle, ctypes.byref(self.overlapped), ctypes.byref(count), False
        ):
            require(ctypes.get_last_error() == 996, "Output change observer failed")
            return {}  # ERROR_IO_INCOMPLETE, not a missing/dead observer.
        require(count.value > 0, "Output change notification overflow: human review")
        raw = self.buffer.raw[: count.value]
        self._arm()  # Re-arm before processing; keep coverage during metadata work.
        return decode(raw)

    def _enqueue(self, changes):
        with self.lock:
            for name, actions in changes.items():
                if name not in self.pending:
                    self.pending_bytes += len(name.encode("utf-16-le")) + 64
                self.pending.setdefault(name, set()).update(actions)
            require(
                len(self.pending) <= 8192 and self.pending_bytes <= 2**20,
                "Output notification backlog overflow: human review",
            )

    def _pump(self):
        try:
            while not self.stop.is_set():
                ready = self.kernel.WaitForSingleObject(self.event, 1000)
                if self.stop.is_set():
                    return
                require(ready in (0, 258), "Output change observer wait failed")
                if ready == 0:
                    self._enqueue(self._read())
        except BaseException as error:
            if not self.stop.is_set():
                with self.lock:
                    self.error = error

    def drain(self):
        with self.lock:
            if self.error is not None:
                raise self.error
            require(self.worker.is_alive(), "Output change observer thread stopped")
            result = self.pending
            self.pending = {}
            self.pending_bytes = 0
            return result

    def close(self):
        if not self.closed:
            self.stop.set()
            self.kernel.CancelIoEx(self.handle, ctypes.byref(self.overlapped))
            if self.worker is not None and self.worker.is_alive():
                self.worker.join(timeout=5)
                require(not self.worker.is_alive(), "Output observer shutdown unproven")
            # Cancellation is asynchronous: retain OVERLAPPED and buffer until done.
            count = ctypes.c_ulong()
            self.kernel.GetOverlappedResult(
                self.handle, ctypes.byref(self.overlapped), ctypes.byref(count), True
            )
            self.kernel.CloseHandle(self.handle)
            self.kernel.CloseHandle(self.event)
            self.closed = True

    def __del__(self):
        # Tests/error paths may drop a Budget without explicit close. Never
        # release a Python-owned OVERLAPPED buffer while native I/O is pending.
        if not getattr(self, "closed", True):
            self.close()


def decode(raw: bytes) -> dict[str, set[int]]:
    """Reject traversal/absolute/malformed notification names before stat access."""
    result = {}
    offset = 0
    while True:
        require(offset + 12 <= len(raw), "Truncated output change notification")
        following, action, length = struct.unpack_from("<III", raw, offset)
        require(
            action in range(1, 6) and length > 0 and length % 2 == 0,
            "Invalid output change notification",
        )
        end = offset + 12 + length
        require(end <= len(raw), "Truncated output change name")
        name = raw[offset + 12 : end].decode("utf-16-le")
        path = PureWindowsPath(name)
        require(
            not path.is_absolute()
            and not path.drive
            and not path.root
            and not set(path.parts) & {".", ".."}
            and ":" not in name
            and "\x00" not in name,
            "Unsafe output change name",
        )
        result.setdefault(path.as_posix(), set()).add(action)
        if following == 0:
            require(end == len(raw), "Trailing output change bytes")
            return result
        require(
            following % 4 == 0 and following >= 12 + length,
            "Invalid output change offset",
        )
        offset += following
