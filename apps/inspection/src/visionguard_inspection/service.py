"""Local-first inspection API; all errors are non-decisions, not NORMAL."""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect

from visionguard.inspection_contract import (
    MAX_IMAGE_BYTES,
    QUALIFICATION,
    InferenceBackend,
    InspectionError,
    prepare_image,
    validate_output,
)
from visionguard_inspection.registry import Registry

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resident:
    """A single published backend/identity pair, never independent mutable fields."""

    model_id: str
    backend: InferenceBackend


class Engine:
    """One worker/one resident model. A timeout does not cancel native execution."""

    def __init__(self, registry):
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection")
        self.slot = threading.Lock()
        self.quarantined = False
        self.resident: Resident | None = None
        self.cleanup_pending: Resident | None = None
        self.lifecycle_failure: str | None = None

    @property
    def backend(self):
        resident = self.resident
        return None if resident is None else resident.backend

    @property
    def model_id(self):
        resident = self.resident
        return None if resident is None else resident.model_id

    def retire(self):
        """Unpublish before close and retain failed cleanup without retrying."""
        previous, self.resident = self.resident, None
        if previous is None:
            return
        self.cleanup_pending = previous
        try:
            if hasattr(previous.backend, "close"):
                closer = previous.backend.close
                if inspect.iscoroutinefunction(closer):
                    raise InspectionError("Backend close must be synchronous")
                outcome = closer()
                if inspect.isawaitable(outcome):
                    if inspect.iscoroutine(outcome):
                        # Discard the unexecuted coroutine, not backend cleanup.
                        outcome.close()
                    raise InspectionError(
                        "Backend close did not complete synchronously"
                    )
        except BaseException:
            self.quarantined = True
            self.lifecycle_failure = "backend_close_failed"
            LOGGER.exception("inspection_backend_close_failed")
            raise
        self.cleanup_pending = None

    def shutdown(self):
        """Drain work; prior transition failures cannot become clean shutdowns."""
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.lifecycle_failure is not None:
            raise InspectionError(
                "Prior backend transition failed; resource "
                "cleanup unverified, not retried"
            )
        self.retire()

    def run(self, registration, data):
        image = prepare_image(data)
        registration.validate_use()
        if self.quarantined:
            raise InspectionError("Worker is quarantined; no inference admitted")
        if (
            self.resident is None
            or self.resident.model_id != registration.manifest.model_id
        ):
            self.retire()
            try:
                replacement = registration.factory()
                if not callable(getattr(replacement, "predict", None)):
                    raise InspectionError("Factory did not return an inference backend")
            except BaseException:
                # A failed constructor may have allocated resources before raising.
                # Do not invent rollback/cleanup or reuse the retired model identity.
                self.quarantined = True
                self.lifecycle_failure = "backend_initialization_failed"
                LOGGER.exception("inspection_backend_initialization_failed")
                raise
            self.resident = Resident(registration.manifest.model_id, replacement)
        try:
            return self.infer(self.resident, registration, image)
        except BaseException:
            # This must also hold when the HTTP client/deadline stopped awaiting us.
            self.quarantined = True
            LOGGER.exception("inspection_backend_execution_failed")
            raise

    def infer(self, resident, registration, image):
        output = resident.backend.predict(image)
        decision = validate_output(image, registration.manifest, output)
        # Original-coordinate PNG, one byte/pixel; no lossy map resizing in the UI.
        values = [v for row in output.anomaly_map for v in row]
        low, high = min(values), max(values)
        # Divide first to avoid overflow for finite maps spanning +/-1e308.
        magnitude = max(abs(low), abs(high), 1)
        lower, upper = low / magnitude, high / magnitude
        scale = upper - lower
        normalized = bytes(
            0 if scale == 0 else round((v / magnitude - lower) / scale * 255)
            for v in values
        )
        heatmap = Image.frombytes("L", (image.width, image.height), normalized)
        encoded = io.BytesIO()
        heatmap.save(encoded, format="PNG")
        return {
            "schema_version": 1,
            "status": "succeeded",
            "decision": decision,
            "model": registration.public(),
            "score": output.score,
            "threshold": registration.manifest.threshold,
            "image_sha256": image.image_sha256,
            "width": image.width,
            "height": image.height,
            "heatmap_png_base64": base64.b64encode(encoded.getvalue()).decode(),
            "visualization": {
                "normalization": "per-image min/max display only",
                "minimum": low,
                "maximum": high,
                "constant_map": scale == 0,
            },
            "qualification": QUALIFICATION,
        }


def error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        {
            "status": "failed",
            "decision": None,
            "error": {"code": code, "message": message},
        },
        status_code=status,
    )


def create_app(
    registry: Registry | None = None,
    *,
    inference_timeout: float = 30,
    upload_timeout: float = 15,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> FastAPI:
    registry = Registry() if registry is None else registry
    if not 0 < inference_timeout <= 300 or not 0 < upload_timeout <= 60:
        raise InspectionError("Timeouts must be positive and bounded")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_IMAGE_BYTES:
        raise InspectionError("Upload limit must be 1 to 10 MiB")
    engine = Engine(registry)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            # Drain off the event loop: active responses/deadlines can still advance.
            engine.quarantined = True
            await asyncio.to_thread(engine.shutdown)

    app = FastAPI(
        title="VisionGuard local inspection",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.engine = engine
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"]
    )

    @app.middleware("http")
    async def local_boundary(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            expected_origin = (
                f"{request.url.scheme}://{request.headers.get('host', '')}"
            )
            if request.headers.get("x-visionguard-client") != "inspection-v1" or (
                origin is not None and origin != expected_origin
            ):
                return error(
                    "cross_origin_refused",
                    "Same-origin inspection client required",
                    403,
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'"
        )
        return response

    @app.get("/api/v1/ready")
    async def ready():
        eligible = []
        for entry in registry.entries.values():
            try:
                entry.validate_use()
                eligible.append(entry.manifest.model_id)
            except Exception:
                LOGGER.warning(
                    "inspection_model_ineligible: %s", entry.manifest.model_id
                )
        resident = engine.resident
        available = bool(eligible) and not engine.quarantined
        return JSONResponse(
            {
                "ready": available,
                "busy": engine.slot.locked(),
                "worker_quarantined": engine.quarantined,
                "eligible_model_ids": eligible,
                "resident_model_id": None if resident is None else resident.model_id,
                "cleanup_pending_model_id": None
                if engine.cleanup_pending is None
                else engine.cleanup_pending.model_id,
                "lifecycle_failure": engine.lifecycle_failure,
                "native_ready": resident is not None
                and resident.model_id in eligible
                and registry.select(resident.model_id).mode == "native-development"
                and available,
                "qualification": QUALIFICATION,
            },
            status_code=200 if available else 503,
        )

    @app.get("/api/v1/models")
    async def models():
        return {
            "models": [entry.public() for entry in registry.entries.values()],
            "qualification": QUALIFICATION,
        }

    @app.post("/api/v1/inspect/{model_id}")
    async def inspect(model_id: str, request: Request):
        try:
            registration = registry.select(model_id)
        except InspectionError:
            return error("unknown_model", "Explicit registered model ID required", 404)
        if engine.quarantined:
            return error(
                "worker_quarantined", "Worker state requires operator review", 503
            )
        if (
            registration.mode == "native-development"
            and request.headers.get("x-visionguard-input-role")
            != "generated-or-development-non-held-out"
        ):
            return error(
                "input_role_refused", "Non-held-out input attestation required", 403
            )
        if request.headers.get("content-type", "").split(";", 1)[0] not in {
            "image/png",
            "image/jpeg",
        }:
            return error("unsupported_media", "PNG/JPEG bytes required", 415)
        if not engine.slot.acquire(blocking=False):
            return error("busy", "One inspection is already in progress", 429)
        submitted = False
        try:
            length = request.headers.get("content-length")
            if length is not None and (
                len(length) > 20
                or not length.isascii()
                or not length.isdecimal()
                or int(length) > max_bytes
            ):
                return error("upload_limit", "Encoded request exceeds the limit", 413)

            async def read():
                data = bytearray()
                async for chunk in request.stream():
                    if len(data) + len(chunk) > max_bytes:
                        raise InspectionError("Encoded request exceeds the limit")
                    data.extend(chunk)
                return bytes(data)

            try:
                data = await asyncio.wait_for(read(), upload_timeout)
            except InspectionError:
                return error("upload_limit", "Encoded request exceeds the limit", 413)
            except TimeoutError:
                return error("upload_timeout", "Upload timed out", 408)
            except ClientDisconnect:
                return error(
                    "client_disconnected",
                    "Upload disconnected; no inference admitted",
                    499,
                )
            if engine.quarantined:
                return error(
                    "worker_quarantined",
                    "Worker unavailable; no inference admitted",
                    503,
                )
            future = engine.pool.submit(engine.run, registration, data)
            future.add_done_callback(lambda _: engine.slot.release())
            submitted = True
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(asyncio.wrap_future(future)), inference_timeout
                )
                return JSONResponse(result)
            except TimeoutError as timeout_error:
                if future.done() and isinstance(future.exception(), TimeoutError):
                    # A backend-raised TimeoutError is not the response deadline.
                    LOGGER.error(
                        "inspection_backend_failure",
                        exc_info=(
                            type(timeout_error),
                            timeout_error,
                            timeout_error.__traceback__,
                        ),
                    )
                    engine.quarantined = True
                    return error(
                        "inference_failed",
                        "Inference failed; no decision produced",
                        500,
                    )
                engine.quarantined = True
                return error(
                    "inference_timeout",
                    "Inference timed out; worker quarantined, "
                    "execution may still be active",
                    504,
                )
            except InspectionError:
                if engine.lifecycle_failure is not None:
                    return error(
                        "backend_transition_failed",
                        "Backend transition failed; worker quarantined, "
                        "cleanup unverified",
                        500,
                    )
                return error(
                    "invalid_input_or_output",
                    "Image or model result failed validation",
                    422,
                )
            except Exception:
                LOGGER.exception("inspection_backend_failure", exc_info=True)
                engine.quarantined = True
                return error(
                    "inference_failed", "Inference failed; no decision produced", 500
                )
        except asyncio.CancelledError:
            if submitted:
                engine.quarantined = True
                LOGGER.warning(
                    "inspection_client_cancelled; execution was not cancelled"
                )
            raise
        finally:
            if not submitted:
                engine.slot.release()

    @app.get("/")
    async def index():
        return HTMLResponse(
            files("visionguard_inspection")
            .joinpath("web/index.html")
            .read_text(encoding="utf-8")
        )

    @app.get("/assets/{name}")
    async def asset(name: str):
        content_type = {"app.js": "text/javascript", "style.css": "text/css"}.get(name)
        if content_type is None:
            return Response(status_code=404)
        return Response(
            files("visionguard_inspection").joinpath("web", name).read_bytes(),
            media_type=content_type,
        )

    return app
