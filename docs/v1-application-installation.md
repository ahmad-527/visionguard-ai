# Local inspection installation — untagged review candidate

Two distributions are intentional: `visionguard-ai` is the frozen core build
recipe; `visionguard-inspection` packages the API/browser without rewriting that
recipe. Both remain version 0.1.0; **this is not a v1.0 release**. Use the matching
core and application files and SHA-256 manifest from this review packet, not an
unrelated package from an index with the same version number. No weights, dataset,
private application permission or incident bundle is shipped.

## Development setup

In a separate application checkout, not the canonical evaluation checkout:

```text
python -m venv .venv
<venv-python> -m pip install -e ".[analysis,dev]"
<venv-python> -m pip install -e "apps/inspection[test]"
<venv-python> -m pytest tests/test_inspection_contract.py tests/test_inspection_registry.py tests/test_inspection_service.py
```

Windows venv Python is `.venv\Scripts\python.exe`; Linux is `.venv/bin/python`.
The app and browser resources are packaged together; no Node build is required.
Use existing Edge for Windows acceptance by setting
`VISIONGUARD_BROWSER_CHANNEL=msedge`; otherwise explicitly provision Playwright
Chromium in the development environment. Set `VISIONGUARD_BROWSER_TESTS=1` and
run `python -m pytest tests/test_inspection_browser.py`. These tests create and
stop only their own ephemeral loopback fixture servers. They are **manufactured
backend acceptance**, not native model inference or a benchmark.

## Clean installation outside source

Build both packages into an external directory:

```text
python -m build --outdir <external-dist>
python -m build apps/inspection --outdir <external-dist>
```

Create a fresh environment outside the checkout. Install both reviewed wheel
paths together, then run `python -m pip check` and `visionguard-inspect --help`.
Installing both source archives instead exercises the sdist build path. The
reproducible harness `scripts/inspection_package_acceptance.py` takes `VG_DIST`
(containing exactly four artifacts: core/app wheel/sdist) and a new `VG_ENVS`
directory; it refuses to overwrite environments. Each install then runs
`python -I <absolute-path>/scripts/inspection_install_smoke.py` from outside
source. Isolated imports must come from site-packages and UI resources must work.
Index access for runtime/build dependencies is necessary unless an approved
offline dependency mirror has been prepared. Offline installation is unverified.

## Local service

```text
visionguard-inspect --manufactured-demo --port 8765
```

Open `http://127.0.0.1:8765/`. Select the manufactured model explicitly and upload
generated PNG/JPEG fixtures only. The intensity fixture is not a trained model;
its NORMAL/ANOMALOUS labels have **no industrial diagnostic meaning**.

Without the demo flag or a pinned native registry, `/api/v1/ready` returns 503.
Only `127.0.0.1` and `localhost` are CLI binding options; one worker is enforced.
Do not wrap this factory in a public/multi-worker server or reverse proxy without
a separate security/resource/deployment review. Same-origin restrictions are
browser protections, not local-user authentication.

Native invocation, once exact development-artifact use has been authorized:

```text
visionguard-inspect --registry <absolute-registry-json> --registry-sha256 <operator-pinned-digest>
```

Read `v1-model-registry.md` first. Do not load historical artifacts to complete a
demo; do not install ML into the canonical checkout. Guarded repository CLIs
still require an explicit reviewed source checkout: freeze/config records are
not wheel resources. A release/upgrade must match source and artifact hashes,
not just 0.1.0 package metadata. Rollback uses a separate environment with prior
reviewed packages; it must not edit original evaluation or incident evidence.

The current packet records actual Windows Python 3.11.6 and Ubuntu WSL Python
3.12.3 acceptance. The declared future CI matrix is Windows/Ubuntu Python
3.11/3.12/3.13, but **declaration is not hosted execution evidence**. See the
external validation JSON for exact versions, commands, skips and results.

> Held-out VisA evaluation with historical access independence unverified.
