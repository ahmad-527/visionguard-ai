# Cleanup diagnostics v5 — adoption candidate only

This separately versioned successor contains allocation-byte diagnostics and truthful
failure handling. It does **not** establish or fix the historical allocation leak.
No held-out data, models, scientific outputs or original incident records are used
as engineering fixtures. Nothing is activated, deployed, pushed or merged.

## Freeze-chain correction

`tests/test_heldout_pause.py` is restored byte-for-byte to base commit
`d5aae83010e32524bb4787afdcc9ff32aeceb6ee`. Its additional unsafe-owner-exit
watcher coverage lives in `tests/test_heldout_cleanup.py`; the real synthetic CPU
watcher must exit successfully and publish a digest-bound stopped receipt with
`safe_owner_exit: false`, without manufacturing a safe exit.

The v3 freeze is verified unchanged. The preserved v4 freeze verifies its four
changed predecessor paths against exact base bytes in
`reports/cleanup-diagnostics-v5/predecessor`. These archives are verification
evidence, not executable substitutes. The v5 builder calls the v4 verifier, which
calls the v3 verifier, and retains identical contract, 72 model identities and
environment. The v5 fingerprint binds current source, CI, new tests, protocols,
archives and historical freeze documents. No historical freeze is overwritten.

The current engineering contract verifies v5, not an old authorization identity.
The existing v4 fingerprint cannot authorize this successor. Creating the candidate
is not execution authorization, a signature, a merge or deployment approval.

## Preserved behavior

The current-device zero-live-allocation guard remains strict. Reserved cache is
reported separately. Original cleanup exceptions propagate, including when an
unsafe watcher finish or immutable failure receipt publication also fails. Unsafe
exit retains the writer lock and does not publish a safe-exit acknowledgement.
Future failure receipts explicitly identify phase, UTC time, owner/invocation,
completion publication and watcher/lock outcome.

## Synthetic verification

The dedicated output-accounting CI workflow verifies v3, v4 and v5 and runs the
new cleanup tests alongside freeze-chain, accounting, execution and pause tests
on its existing Windows/Linux and Python matrix. Local checks use manufactured
fixtures only; they do not prove every CI platform or historical allocation owner.

```powershell
python scripts/operational_successor_freeze.py
python scripts/output_accounting_freeze.py
python scripts/cleanup_successor_freeze.py
python -m pytest tests/test_heldout_accounting.py tests/test_accounting_successor.py tests/test_operational_successor.py tests/test_cleanup_successor.py tests/test_heldout_execution.py tests/test_heldout_pause.py tests/test_heldout_cleanup.py
python -m ruff check .
python -m ruff format --check .
```

The established create-once workflow is
`python scripts/cleanup_successor_freeze.py --create-successor` in the isolated
engineering checkout. It refuses to overwrite an existing candidate. Final hashes,
exact commands, results and failed attempts are delivered in a new external review
packet. Original STOP, lock, scientific evidence and prior review packets remain
untouched. Lock reconciliation, recovery and deployment require separate authority.
