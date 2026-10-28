"""Bounded generated-input transition, cancellation and draining regressions."""

import asyncio
import io
import threading
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("visionguard_inspection")
import httpx
from fastapi.testclient import TestClient
from PIL import Image
from visionguard_inspection.registry import (
    ManufacturedBackend,
    Registry,
    manufactured_registry,
)
from visionguard_inspection.service import create_app

from visionguard.inspection_contract import InspectionError

HEADERS = {"content-type": "image/png", "x-visionguard-client": "inspection-v1"}


def image():
    stream = io.BytesIO()
    Image.new("RGB", (7, 3), "black").save(stream, format="PNG")
    return stream.getvalue()


def entry(model_id, factory, **changes):
    initial = next(iter(manufactured_registry().entries.values()))
    return replace(
        initial,
        manifest=replace(initial.manifest, model_id=model_id),
        factory=factory,
        **changes,
    )


class Tracked(ManufacturedBackend):
    def __init__(self, manifest, events, close_error=None):
        super().__init__(manifest)
        self.events, self.close_error = events, close_error
        self.closed = False

    def predict(self, data):
        assert not self.closed, "Retired backend was used"
        self.events.append(("predict", self.manifest.model_id))
        return super().predict(data)

    def close(self):
        self.events.append(("close", self.manifest.model_id))
        self.closed = True  # A close may mutate state before failing.
        if self.close_error:
            raise self.close_error


def post(api, model_id):
    return api.post(f"/api/v1/inspect/{model_id}", content=image(), headers=HEADERS)


def test_a_success_b_inspectionerror_then_a_never_uses_stale_identity(caplog):
    events = []
    a = entry("a", lambda: Tracked(a.manifest, events))
    original = InspectionError("manufactured B constructor failure")

    def failure():
        events.append(("factory", "b"))
        raise original

    b = entry("b", failure)
    app = create_app(Registry((a, b)))
    with (
        pytest.raises(InspectionError, match="cleanup unverified, not retried"),
        TestClient(app, base_url="http://127.0.0.1") as api,
    ):
        assert post(api, "a").json()["decision"] == "NORMAL"
        failed = post(api, "b")
        assert failed.status_code == 500 and failed.json()["decision"] is None
        assert failed.json()["error"]["code"] == "backend_transition_failed"
        state = api.get("/api/v1/ready")
        assert state.status_code == 503
        assert state.json()["resident_model_id"] is None
        assert state.json()["lifecycle_failure"] == "backend_initialization_failed"
        assert app.state.engine.backend is None and app.state.engine.model_id is None
        refused = post(api, "a")
        assert refused.status_code == 503 and refused.json()["decision"] is None
        assert events == [("predict", "a"), ("close", "a"), ("factory", "b")]
    assert any(
        record.exc_info and record.exc_info[1] is original for record in caplog.records
    )


def test_close_failure_retains_failed_owner_unpublished_and_is_not_retried():
    events = []
    a = entry(
        "a",
        lambda: Tracked(
            a.manifest, events, InspectionError("close mutated then failed")
        ),
    )
    b = entry("b", lambda: pytest.fail("Replacement must not be constructed"))
    app = create_app(Registry((a, b)))
    with (
        pytest.raises(InspectionError, match="not retried"),
        TestClient(app, base_url="http://127.0.0.1") as api,
    ):
        assert post(api, "a").status_code == 200
        assert post(api, "b").status_code == 500
        state = api.get("/api/v1/ready").json()
        assert state["resident_model_id"] is None
        assert state["cleanup_pending_model_id"] == "a"
        assert state["lifecycle_failure"] == "backend_close_failed"
        assert post(api, "a").status_code == 503
    assert events == [("predict", "a"), ("close", "a")]
    assert app.state.engine.cleanup_pending.model_id == "a"


def test_successful_switch_and_ten_repeated_requests_reuse_only_current_backend():
    events = []
    a = entry("a", lambda: Tracked(a.manifest, events))
    b = entry("b", lambda: Tracked(b.manifest, events))
    app = create_app(Registry((a, b)))
    with TestClient(app, base_url="http://127.0.0.1") as api:
        for _ in range(10):
            assert post(api, "a").json()["model"]["model_id"] == "a"
        assert post(api, "b").json()["model"]["model_id"] == "b"
        assert api.get("/api/v1/ready").json()["resident_model_id"] == "b"
        assert post(api, "a").status_code == 200
    assert [event for event in events if event[0] == "close"] == [
        ("close", "a"),
        ("close", "b"),
        ("close", "a"),
    ]
    assert (
        app.state.engine.resident is None and app.state.engine.cleanup_pending is None
    )


def test_invalid_factory_result_quarantines_before_none_predict():
    app = create_app(Registry((entry("bad", lambda: None),)))
    with (
        pytest.raises(InspectionError, match="cleanup unverified"),
        TestClient(app, base_url="http://127.0.0.1") as api,
    ):
        result = post(api, "bad")
        assert result.status_code == 500 and result.json()["decision"] is None
        assert api.get("/api/v1/ready").status_code == 503
        assert app.state.engine.resident is None


def test_async_close_is_not_reported_as_successful_cleanup():
    events = []

    class AsyncClose(Tracked):
        async def close(self):
            pytest.fail(
                "Unawaited asynchronous cleanup must not be treated as complete"
            )

    a = entry("a", lambda: AsyncClose(a.manifest, events))
    b = entry("b", lambda: pytest.fail("Replacement must not be constructed"))
    app = create_app(Registry((a, b)))
    with (
        pytest.raises(InspectionError, match="not retried"),
        TestClient(app, base_url="http://127.0.0.1") as api,
    ):
        assert post(api, "a").status_code == 200
        assert post(api, "b").status_code == 500
        state = api.get("/api/v1/ready").json()
        assert state["cleanup_pending_model_id"] == "a"
        assert state["lifecycle_failure"] == "backend_close_failed"
        assert state["resident_model_id"] is None


def test_permission_revoked_on_resident_model_refuses_without_predict_or_switch():
    valid, events = True, []

    def permission():
        if not valid:
            raise InspectionError("manufactured permission revoked")

    a = entry("a", lambda: Tracked(a.manifest, events), validate_use=permission)
    app = create_app(Registry((a,)))
    with TestClient(app, base_url="http://127.0.0.1") as api:
        assert post(api, "a").status_code == 200
        valid = False
        assert post(api, "a").status_code == 422
        state = api.get("/api/v1/ready")
        assert state.status_code == 503 and not state.json()["native_ready"]
        assert events == [("predict", "a")]


def test_upload_deadline_and_disconnect_release_slot_without_factory():
    async def scenario(disconnect):
        factories = []
        a = entry("a", lambda: factories.append("called"))
        app = create_app(Registry((a,)), upload_timeout=0.02)
        messages, calls = [], 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b"partial", "more_body": True}
            if disconnect:
                return {"type": "http.disconnect"}
            await asyncio.sleep(0.2)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/inspect/a",
            "query_string": b"",
            "root_path": "",
            "server": ("127.0.0.1", 80),
            "client": ("127.0.0.1", 123),
            "headers": [
                (b"host", b"127.0.0.1"),
                *[(k.encode(), v.encode()) for k, v in HEADERS.items()],
            ],
        }
        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
            assert not app.state.engine.slot.locked()
            assert not factories and not app.state.engine.quarantined
        assert next(m for m in messages if m["type"] == "http.response.start")[
            "status"
        ] == (499 if disconnect else 408)

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


@pytest.mark.parametrize("cancel", [False, True])
def test_real_concurrent_request_and_cancel_keep_active_slot_until_work_exits(cancel):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        initial = next(iter(manufactured_registry().entries.values()))

        class Slow(ManufacturedBackend):
            def predict(self, data):
                entered.set()
                assert release.wait(3), "Bounded fixture release missing"
                return super().predict(data)

        app = create_app(
            Registry((replace(initial, factory=lambda: Slow(initial.manifest)),))
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
            ) as api,
        ):
            task = asyncio.create_task(
                api.post(
                    "/api/v1/inspect/manufactured-demo",
                    content=image(),
                    headers=HEADERS,
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                other = await api.post(
                    "/api/v1/inspect/manufactured-demo",
                    content=image(),
                    headers=HEADERS,
                )
                assert other.status_code == 429 and other.json()["decision"] is None
                if cancel:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert (
                        app.state.engine.slot.locked() and app.state.engine.quarantined
                    )
                    assert (await api.get("/api/v1/ready")).status_code == 503
                release.set()
                if not cancel:
                    assert (await task).status_code == 200
            finally:
                release.set()
                if not task.done():
                    await task
        assert not app.state.engine.slot.locked() and app.state.engine.resident is None

    asyncio.run(scenario())


def test_shutdown_drains_active_work_off_loop_before_close():
    async def scenario():
        entered, release, closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        initial = next(iter(manufactured_registry().entries.values()))

        class Slow(ManufacturedBackend):
            def predict(self, data):
                entered.set()
                assert release.wait(3)
                assert not closed.is_set()
                return super().predict(data)

            def close(self):
                closed.set()

        app = create_app(
            Registry((replace(initial, factory=lambda: Slow(initial.manifest)),))
        )
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as api:
            request = asyncio.create_task(
                api.post(
                    "/api/v1/inspect/manufactured-demo",
                    content=image(),
                    headers=HEADERS,
                )
            )
            assert await asyncio.to_thread(entered.wait, 2)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            try:
                await asyncio.sleep(0.02)
                assert not shutdown.done() and not closed.is_set()
                assert (await api.get("/api/v1/ready")).status_code == 503
                release.set()
                assert (await request).status_code == 200
                await shutdown
            finally:
                release.set()
                await request
                await shutdown
        assert closed.is_set() and app.state.engine.resident is None

    asyncio.run(scenario())
