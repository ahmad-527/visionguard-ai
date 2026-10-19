# Local inspection installation — untagged review candidate

Current integration: PR36 normal merge f13beb897e9223e07866f1f681c457cce6dcfec1.
Use the separately reviewed private correction worktree only after its own
adoption. The canonical development repository is visionguard-ai; FlawGlyph
remains a separate unchanged source snapshot. See v1-release-checklist.md for
the observed current CI failure and prepared private fix.

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
After a failed backend transition, no model can be reused and shutdown reports
cleanup unverified. Do not interpret an HTTP 500/504 as a successful rollback or
permission to restart any completed evaluation. Failed cleanup is not retried.
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

The separate CPU successor has its own exact requirements/constraints and
runtime-bound permission contract. Follow `ml-cpu-runtime-successor.md` for that
candidate, **not** the historical root `ml` extra. Matching its package profile
does not resolve the PT2 advisory or authorize any trained artifact loading.

The exact PR36 candidate completed nine workflows/36 jobs on Windows/Ubuntu
Python 3.11-3.13 before integration. The seven naturally triggered main workflows
are separately recorded; ordinary lint failed on new Ruff 0.17.1 behaviour.
The private compatibility fix is not yet public CI evidence. Windows Python
3.11 has the completed, exact-scope one-shot trained PatchCore acceptance;
the wider CPU platform/model matrix is manufactured evidence, not additional
trained acceptance. Public/multi-worker service and GPU application support
are outside this v1 operating scope.

For durable fixture-test reporting without periodic native timed dumps, run:

    python scripts/inspection_test_observer.py --output <new-external-observer-dir> tests/test_inspection_state_archive.py tests/test_inspection_observer.py

Use an output directory that does not exist; the observer never replaces old
evidence. It captures bounded Python code locations, not frame locals, and keeps
fatal reporting. This command creates no trained authority.

Native candidates now require an ordinary tensor-state ZIP; PT2/export,
TorchScript, legacy pickle, compressed, duplicate or escaping records are
refused before deserialization. This is stricter format admission, not a claim
that upstream Torch export is patched. Follow v1-operational-security-status.md.
Do not use the permanently consumed acceptance permission or registry as a
startup example. No live trained startup is authorised by this documentation.

> Held-out VisA evaluation with historical access independence unverified.
