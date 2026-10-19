"""Real browser -> loopback API acceptance, exclusively manufactured fixtures.

Set VISIONGUARD_BROWSER_TESTS=1; provision a browser explicitly. Tests do not
download browsers or use checkpoint/dataset assets. Default CI uses Chromium;
VISIONGUARD_BROWSER_CHANNEL=msedge uses an existing Windows Edge installation.
"""

import io
import os
import socket
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("playwright")
import uvicorn
from playwright.sync_api import expect, sync_playwright

pytest.importorskip("visionguard_inspection")
from visionguard_inspection.registry import (
    ManufacturedBackend,
    Registry,
    manufactured_registry,
)
from visionguard_inspection.service import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("VISIONGUARD_BROWSER_TESTS") != "1",
    reason="Explicit browser acceptance opt-in required",
)


@contextmanager
def server(registry):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        app = create_app(registry)
        instance = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=instance.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            # Only our ephemeral fixture server. No evaluation processes touched.
            instance.should_exit = True
            thread.join(10)
            assert not thread.is_alive(), "Fixture server did not exit"


def upload(page, value, name="fixture.png"):
    output = io.BytesIO()
    Image.new("RGB", (41, 23), (value, value, value)).save(output, format="PNG")
    page.locator("#image").set_input_files(
        {"name": name, "mimeType": "image/png", "buffer": output.getvalue()}
    )


def test_browser_upload_identity_map_and_failed_request_clear_normal():
    with server(manufactured_registry()) as url, sync_playwright() as playwright:
        channel = os.environ.get("VISIONGUARD_BROWSER_CHANNEL")
        browser = playwright.chromium.launch(headless=True, channel=channel)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(url)
            expect(page.locator("#inspect")).to_be_enabled()
            expect(page.locator("label[for=image]")).to_have_text(
                "PNG or JPEG · up to 10 MiB"
            )
            assert page.locator("#model").input_value() == ""
            page.locator("#model").select_option("manufactured-demo")
            upload(page, 0)
            page.locator("#inspect").click()
            expect(page.locator("#decision")).to_have_text("NORMAL")
            expect(page.locator("#mode")).to_contain_text("NOT NATIVE INFERENCE")
            expect(page.locator("#identity")).to_contain_text("artifact")
            expect(page.locator("#score")).to_contain_text("threshold 0.5")
            assert page.locator("#heatmap").evaluate("c => [c.width, c.height]") == [
                41,
                23,
            ]
            upload(page, 255)
            page.locator("#inspect").click()
            expect(page.locator("#decision")).to_have_text("ANOMALOUS")
            evidence = os.environ.get("VISIONGUARD_BROWSER_EVIDENCE")
            if evidence:
                page.locator("#original").evaluate("image => image.decode()")
                page.screenshot(
                    path=str(Path(evidence) / "browser-success.png"), full_page=True
                )
            # Real API 422, not a route mock: upload corrupt image bytes.
            page.locator("#image").set_input_files(
                {
                    "name": "corrupt.png",
                    "mimeType": "image/png",
                    "buffer": b"not an image",
                }
            )
            expect(page.locator("#result")).to_be_hidden()
            page.locator("#inspect").click()
            expect(page.locator("#status")).to_contain_text(
                "Inspection failed. No decision."
            )
            expect(page.locator("#result")).to_be_hidden()
            expect(page.locator("#decision")).to_have_text("")
            if evidence:
                page.screenshot(
                    path=str(Path(evidence) / "browser-failure.png"), full_page=True
                )
            assert page.evaluate("window.innerWidth") == 1280
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            if evidence:
                page.screenshot(
                    path=str(Path(evidence) / "browser-mobile.png"), full_page=True
                )
        finally:
            browser.close()


def test_browser_empty_registry_is_visibly_not_ready():
    with server(None) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("VISIONGUARD_BROWSER_CHANNEL")
        )
        try:
            page = browser.new_page()
            page.goto(url)
            expect(page.locator("#status")).to_contain_text("Not ready")
            expect(page.locator("#inspect")).to_be_disabled()
            expect(page.locator("#result")).to_be_hidden()
        finally:
            browser.close()


def test_browser_actual_backend_failure_never_keeps_normal():
    class Failed:
        def predict(self, _):
            raise MemoryError("manufactured allocation failure, not real OOM")

    entry = next(iter(manufactured_registry().entries.values()))
    failed_entry = replace(
        entry,
        manifest=replace(entry.manifest, model_id="manufactured-failure"),
        factory=Failed,
    )
    with (
        server(Registry((entry, failed_entry))) as url,
        sync_playwright() as playwright,
    ):
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("VISIONGUARD_BROWSER_CHANNEL")
        )
        try:
            page = browser.new_page()
            page.goto(url)
            expect(page.locator("#inspect")).to_be_enabled()
            page.locator("#model").select_option("manufactured-demo")
            upload(page, 0)
            page.locator("#inspect").click()
            expect(page.locator("#decision")).to_have_text("NORMAL")
            page.locator("#model").select_option("manufactured-failure")
            page.locator("#inspect").click()
            expect(page.locator("#status")).to_contain_text(
                "Inference failed; no decision produced"
            )
            expect(page.locator("#result")).to_be_hidden()
            expect(page.locator("#decision")).to_have_text("")
        finally:
            browser.close()


def test_browser_refuses_auto_oriented_jpeg_without_a_decision():
    metadata = Image.Exif()
    metadata[274] = 6
    encoded = io.BytesIO()
    Image.new("RGB", (41, 23), "red").save(encoded, format="JPEG", exif=metadata)
    with server(manufactured_registry()) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("VISIONGUARD_BROWSER_CHANNEL")
        )
        try:
            page = browser.new_page()
            page.goto(url)
            expect(page.locator("#inspect")).to_be_enabled()
            page.locator("#model").select_option("manufactured-demo")
            upload(page, 0)
            page.locator("#inspect").click()
            expect(page.locator("#decision")).to_have_text("NORMAL")
            page.locator("#image").set_input_files(
                {
                    "name": "oriented.jpg",
                    "mimeType": "image/jpeg",
                    "buffer": encoded.getvalue(),
                }
            )
            page.locator("#inspect").click()
            expect(page.locator("#status")).to_contain_text(
                "Inspection failed. No decision."
            )
            expect(page.locator("#result")).to_be_hidden()
            expect(page.locator("#decision")).to_have_text("")
        finally:
            browser.close()


@pytest.mark.parametrize("late_failure", [False, True])
def test_browser_changed_selection_ignores_late_result_and_updates_readiness(
    late_failure,
):
    entered, release = threading.Event(), threading.Event()
    entry = next(iter(manufactured_registry().entries.values()))

    class Delayed(ManufacturedBackend):
        def predict(self, data):
            entered.set()
            assert release.wait(3), "Bounded fixture release missing"
            if late_failure:
                raise RuntimeError("manufactured late backend failure")
            return super().predict(data)

    delayed = replace(entry, factory=lambda: Delayed(entry.manifest))
    with server(Registry((delayed,))) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("VISIONGUARD_BROWSER_CHANNEL")
        )
        try:
            page = browser.new_page()
            page.goto(url)
            expect(page.locator("#inspect")).to_be_enabled()
            page.locator("#model").select_option("manufactured-demo")
            upload(page, 0)
            page.locator("#inspect").click()
            assert entered.wait(2)
            upload(page, 255, "changed.png")
            expect(page.locator("#status")).to_contain_text("Selection changed")
            release.set()
            # Poll server readiness and UI rather than inventing a completion delay.
            if late_failure:
                expect(page.locator("#inspect")).to_be_disabled()
                page.wait_for_function(
                    "fetch('/api/v1/ready').then(r => r.json())"
                    ".then(s => s.worker_quarantined)"
                )
            else:
                expect(page.locator("#inspect")).to_be_enabled()
            expect(page.locator("#result")).to_be_hidden()
            expect(page.locator("#decision")).to_have_text("")
            expect(page.locator("#status")).to_contain_text("Selection changed")
        finally:
            release.set()
            browser.close()
