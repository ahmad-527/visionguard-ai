> 28 October 2026 successor: use application-cpu-20261028-v2 and the explicit
> packaging/inspection-v2 recipe. The dated earlier 0.1.0/v1 observations below
> remain historical evidence; they are not current installation instructions or
> new CI/acceptance results. See [the successor](ci-runtime-successor-v6.md).

# Local inspection installation — untagged review candidate

Historical integration: PR36 normal merge f13beb897e9223e07866f1f681c457cce6dcfec1
on 19 October 2026. The 26 October correction is authorised for its own follow-up
PR/checks and conditional normal merge. Use exact commit/artifact bindings from
that review. The canonical development repository is visionguard-ai; FlawGlyph
remains a separate unchanged source snapshot. See v1-release-checklist.md for
dated CI observations; they are not assertions about a later main.

Two distributions are intentional: `visionguard-ai` is the frozen core build
recipe; `visionguard-inspection` packages the API/browser without rewriting that
recipe. Both remain version 0.1.0; **this is not a v1.0 release**. Use the matching
core and application files and SHA-256 manifest from this review packet, not an
unrelated package from an index with the same version number. No weights, dataset,
private application permission or incident bundle is shipped.

## Current successor installation

Create a fresh Python 3.11-3.13 project venv without system-site inheritance.
Install pip 26.2.1, setuptools 84.2.1, hatchling 1.32.4 and build 1.6.1 there.
Then use that venv's Python:

    python scripts/install_successor.py --profile application --source <new-external-source-directory>
    visionguard-inspect --manufactured-demo --port 8765

For native fixtures, install the exact official CPU Torch/Torchvision wheels
first, then choose --profile native. This does not permit trained artifact use.
To build the matched pair with updated notices:

    python scripts/successor_build.py --source <new-external-build-source> --dist <new-external-dist>

The historical commands below document the earlier recipe and must not be used
to force Pillow 13 into a <13 dependency declaration. The 0.2.0 metadata reflects
the successor explicitly. No release/tag or package-index publication is implied.

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
offline dependency mirror has been prepared. The 19 October private proposal
verified offline wheel/sdist installations against its exact approved dependency copies;
that result does not describe arbitrary fresh index-free environments.

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
Python 3.11-3.13 before its 19 October integration. Its seven main-push workflows
are separately recorded: six ordinary lint jobs failed on Ruff 0.17.1 behaviour.
As recorded on 19 October, the compatibility fix had private checks only.
The 26 October follow-up requires its own CI; these historical counts must not
be used as the later branch/main result. Windows Python
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
refused before deserialization. Raw names containing NUL are rejected before
normalization or Torch import. Stored compressed/uncompressed sizes must match;
each member and their aggregate must fit the captured archive byte length.
This is stricter format admission, not a claim
that upstream Torch export is patched. Follow v1-operational-security-status.md.
Do not use the permanently consumed acceptance permission or registry as a
startup example. No live trained startup is authorised by this documentation.

> Held-out VisA evaluation with historical access independence unverified.
