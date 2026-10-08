"""Run with python -I outside source, after a clean wheel or sdist installation.

This is manufactured API/package acceptance only, not native ML inference.
"""

import base64
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path

import visionguard_inspection
from fastapi.testclient import TestClient
from PIL import Image
from visionguard_inspection.cli import main
from visionguard_inspection.registry import manufactured_registry
from visionguard_inspection.service import create_app

import visionguard

assert "site-packages" in str(Path(visionguard.__file__)).lower()
assert "site-packages" in str(Path(visionguard_inspection.__file__)).lower()
assert "src/visionguard" not in str(Path(visionguard.__file__)).replace("\\", "/")
try:
    main(["--help"])
except SystemExit as exit_status:
    assert exit_status.code == 0
encoded = io.BytesIO()
Image.new("L", (13, 9), 255).save(encoded, format="PNG")
with TestClient(
    create_app(manufactured_registry()), base_url="http://127.0.0.1"
) as api:
    assert api.get("/").status_code == 200
    for asset in ("app.js", "style.css"):
        assert api.get(f"/assets/{asset}").status_code == 200
    assert api.get("/api/v1/ready").json()["native_ready"] is False
    response = api.post(
        "/api/v1/inspect/manufactured-demo",
        content=encoded.getvalue(),
        headers={"content-type": "image/png", "x-visionguard-client": "inspection-v1"},
    )
    assert response.status_code == 200 and response.json()["decision"] == "ANOMALOUS"
    with Image.open(
        io.BytesIO(base64.b64decode(response.json()["heatmap_png_base64"]))
    ) as image:
        assert image.size == (13, 9)
    failed = api.post(
        "/api/v1/inspect/manufactured-demo",
        content=b"corrupt",
        headers={"content-type": "image/png", "x-visionguard-client": "inspection-v1"},
    )
    assert failed.status_code == 422 and failed.json()["decision"] is None
    oriented = io.BytesIO()
    metadata = Image.Exif()
    metadata[274] = 6
    Image.new("RGB", (13, 9), "red").save(oriented, format="JPEG", exif=metadata)
    refused = api.post(
        "/api/v1/inspect/manufactured-demo",
        content=oriented.getvalue(),
        headers={"content-type": "image/jpeg", "x-visionguard-client": "inspection-v1"},
    )
    assert refused.status_code == 422 and refused.json()["decision"] is None
print(
    json.dumps(
        {
            "status": "passed",
            "scope": "manufactured package/API acceptance",
            "system": platform.platform(),
            "python": platform.python_version(),
            "installed_module": visionguard.__file__,
            "versions": {
                name: version(name)
                for name in (
                    "visionguard-ai",
                    "visionguard-inspection",
                    "fastapi",
                    "starlette",
                    "uvicorn",
                    "Pillow",
                    "httpx",
                )
            },
        }
    )
)
