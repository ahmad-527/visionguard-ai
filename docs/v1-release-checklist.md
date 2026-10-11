# v1.0 release checklist — NOT release approval

| Gate | Current disposition | Evidence / next requirement |
| --- | --- | --- |
| Engineering adoption | Reviewed v5 remains a draft candidate; no activation | Parent PR #33 depends on accounting PR #32; no historical source/freeze modification |
| HTTP/registry/UI slice | Implemented | `apps/inspection`, `inspection_contract.py`; external test/patch/hash packet |
| Manufactured browser/API acceptance | Implemented tests and local verification | Real Edge -> ephemeral loopback FastAPI, generated PNGs and corrupt-input failure; no trained model |
| Core/app wheel and sdist | Harness prepared; final results in external packet | Clean Windows 3.11 / Ubuntu WSL 3.12 installs outside source; site-packages/resource/CLI/API checks |
| Native development inference | **Blocked** | Exact artifact-use permission/identities and authorized non-held-out fixture scope; see `v1-model-registry.md` |
| Native tensor plumbing | Separate manufactured test evidence | CPU tensors, fake constructor; not trained model/real artifact acceptance |
| Hosted application matrix | **Final-commit verification required** | Windows/Ubuntu Python 3.11/3.12/3.13; dated external review packet records actual run/head/job results, not a declaration |
| Results publication | **Pending review** | Prepared reconciliation; no scientific numbers/digests independently revalidated here |
| Source license | **Human decision required** | MIT/Apache-2.0 options; no license silently selected |
| Security/dependency/model license review | **Release qualification required** | Dated advisory/package/license inventory and limitations; local-only service is not authenticated/public deployment, and model-use/redistribution rights remain separate |
| Distribution identity/version/support matrix | **Pending release review** | Untagged 0.1.0 core/app pair pinned by hashes; decide release version/support only after gates |
| End-to-end native acceptance | **Missing** | Approved development state -> generated/non-held-out upload -> CPU -> original map -> browser; no tuning/rerun |
| Release/activation/deployment/tag | **Not authorized** | Separate human authorization after review; do not treat green tests as permission |

Final local test outcomes, failures/skips, source hashes and package digests are
in the external application review packet. No numerical result in this checklist
represents scientific evaluation. Frozen original STOP/lock, outputs and incident
evidence remain untouched. This app does not reconcile an original lock, change
monitoring policy or claim that the historical allocation leak was fixed.

> Held-out VisA evaluation with historical access independence unverified.
