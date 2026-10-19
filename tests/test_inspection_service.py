"""Manufactured API tests: no real artifacts, dataset, training or GPU access."""

import base64
import io
import threading
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from PIL import Image

from visionguard.inspection_contract import InspectionError, ModelOutput

pytest.importorskip("visionguard_inspection")
from visionguard_inspection.registry import Registry, manufactured_registry
from visionguard_inspection.service import create_app

HEADERS = {"content-type": "image/png", "x-visionguard-client": "inspection-v1"}


def png(value=0):
    output = io.BytesIO()
    Image.new("L", (7, 3), value).save(output, format="PNG")
    return output.getvalue()


def client(registry=None, **options):
    return TestClient(create_app(registry, **options), base_url="http://127.0.0.1")


@pytest.mark.parametrize("value,decision", [(0, "NORMAL"), (255, "ANOMALOUS")])
def test_success_has_original_map_and_explicit_identity(value, decision):
    with client(manufactured_registry()) as api:
        assert api.get("/api/v1/ready").json()["native_ready"] is False
        result = api.post(
            "/api/v1/inspect/manufactured-demo", content=png(value), headers=HEADERS
        )
        assert result.status_code == 200
        data = result.json()
        assert data["decision"] == decision
        assert data["model"]["mode"] == "manufactured"
        assert data["score"] == value / 255 and data["threshold"] == 0.5
        with Image.open(
            io.BytesIO(base64.b64decode(data["heatmap_png_base64"]))
        ) as image:
            assert image.size == (7, 3)
        assert "historical access independence unverified" in data["qualification"]
        assert api.get("/").status_code == 200
        assert (
            api.get("/assets/app.js")
            .headers["content-type"]
            .startswith("text/javascript")
        )
        assert api.get("/assets/other").status_code == 404


def test_empty_registry_not_ready_and_never_selects_default():
    with client() as api:
        assert api.get("/api/v1/ready").status_code == 503
        assert api.get("/api/v1/models").json()["models"] == []
        assert (
            api.post(
                "/api/v1/inspect/unknown", content=png(), headers=HEADERS
            ).status_code
            == 404
        )


@pytest.mark.parametrize(
    "body,headers,code",
    [
        (b"corrupt", HEADERS, 422),
        (png(), {**HEADERS, "content-type": "application/json"}, 415),
        (png(), {"content-type": "image/png"}, 403),
        (png(), {**HEADERS, "origin": "https://hostile.invalid"}, 403),
        (png(), {**HEADERS, "content-length": "999999999"}, 413),
    ],
)
def test_failure_never_normal(body, headers, code):
    with client(manufactured_registry()) as api:
        response = api.post(
            "/api/v1/inspect/manufactured-demo", content=body, headers=headers
        )
        assert response.status_code == code
        assert response.json()["decision"] is None


def test_actual_stream_limit_not_only_content_length():
    with client(manufactured_registry(), max_bytes=10) as api:
        response = api.post(
            "/api/v1/inspect/manufactured-demo",
            content=iter([b"a" * 8, b"b" * 8]),
            headers=HEADERS,
        )
        assert response.status_code == 413 and response.json()["decision"] is None


def registry_with(backend=None, validate_use=lambda: None):
    entry = next(iter(manufactured_registry().entries.values()))
    return Registry(
        (replace(entry, factory=lambda: backend, validate_use=validate_use),)
    )


def test_backend_exception_is_truthful_and_quarantines():
    class Failed:
        def predict(self, _):
            raise RuntimeError("manufactured failure")

    with client(registry_with(Failed())) as api:
        result = api.post(
            "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
        )
        assert result.status_code == 500 and result.json()["decision"] is None
        assert api.get("/api/v1/ready").json()["worker_quarantined"]
        assert (
            api.post(
                "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
            ).status_code
            == 503
        )


def test_timeout_does_not_release_active_work_or_claim_cancellation():
    release, entered = threading.Event(), threading.Event()

    class Slow:
        def predict(self, _):
            entered.set()
            release.wait(5)
            raise RuntimeError("manufactured late exit")

    with client(registry_with(Slow()), inference_timeout=0.03) as api:
        try:
            result = api.post(
                "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
            )
            assert entered.is_set() and result.status_code == 504
            assert result.json()["decision"] is None
            assert "may still be active" in result.json()["error"]["message"]
            assert api.get("/api/v1/ready").json()["busy"]
            assert (
                api.post(
                    "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
                ).status_code
                == 503
            )
        finally:
            release.set()


def test_busy_refuses_another_request():
    app = create_app(manufactured_registry())
    with TestClient(app, base_url="http://127.0.0.1") as api:
        app.state.engine.slot.acquire()
        try:
            assert (
                api.post(
                    "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
                ).status_code
                == 429
            )
        finally:
            app.state.engine.slot.release()


def test_permission_rechecked_and_ineligible_readiness():
    def refuse():
        raise InspectionError("manufactured expired permission")

    with client(registry_with(validate_use=refuse)) as api:
        assert api.get("/api/v1/ready").status_code == 503
        assert (
            api.post(
                "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
            ).status_code
            == 422
        )


@pytest.mark.parametrize("score,rows", [(float("nan"), None), (0, ((0,),)), (0, None)])
def test_bad_model_output_never_normal(score, rows):
    class Bad:
        def predict(self, image):
            entry = next(iter(manufactured_registry().entries.values()))
            return ModelOutput(
                image.image_sha256,
                entry.manifest.artifact_sha256,
                entry.manifest.preprocessing_sha256,
                score,
                rows,
            )

    with client(registry_with(Bad())) as api:
        response = api.post(
            "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
        )
        assert response.status_code == 422 and response.json()["decision"] is None


@pytest.mark.parametrize(
    "options", [{"max_bytes": 0}, {"inference_timeout": 0}, {"upload_timeout": 99}]
)
def test_configuration_bounds(options):
    with pytest.raises(InspectionError):
        create_app(**options)


@pytest.mark.parametrize("exception", [TimeoutError, MemoryError])
def test_backend_timeout_or_memory_failure_not_response_deadline(exception):
    class Failed:
        def predict(self, _):
            raise exception("manufactured backend fault")

    with client(registry_with(Failed())) as api:
        response = api.post(
            "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
        )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "inference_failed"
        assert response.json()["decision"] is None


def test_equality_threshold_extreme_finite_map_and_hostile_host():
    entry = next(iter(manufactured_registry().entries.values()))

    class Fixture:
        def predict(self, image):
            return ModelOutput(
                image.image_sha256,
                entry.manifest.artifact_sha256,
                entry.manifest.preprocessing_sha256,
                0.5,
                tuple(
                    tuple(-1e308 if x < 3 else 1e308 for x in range(7))
                    for _ in range(3)
                ),
            )

    with client(registry_with(Fixture())) as api:
        response = api.post(
            "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
        )
        assert response.status_code == 200 and response.json()["decision"] == "NORMAL"
        with Image.open(
            io.BytesIO(base64.b64decode(response.json()["heatmap_png_base64"]))
        ) as image:
            assert image.getpixel((0, 0)) == 0 and image.getpixel((6, 0)) == 255
        assert api.get("/", headers={"host": "hostile.invalid"}).status_code == 400


def test_native_input_role_is_explicit_and_not_inferred():
    entry = next(iter(manufactured_registry().entries.values()))
    registry = Registry((replace(entry, mode="native-development"),))
    with client(registry) as api:
        result = api.post(
            "/api/v1/inspect/manufactured-demo", content=png(), headers=HEADERS
        )
        assert result.status_code == 403 and result.json()["decision"] is None
