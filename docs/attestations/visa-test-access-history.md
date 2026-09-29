# VisA test-access history

Human access-history declaration: **UNKNOWN**

Independent-reservation status: **NOT ESTABLISHED**

No declaration is inferred from software, archive hashes, overlap checks, clean
Git state, or absence of recorded test metrics. No agent may complete this form
on a contributor's behalf without explicit authorized input.

## Contributor/user declaration — received

Before pre-data freeze commit `c019f5515ee7692b42cda5bcc0595a4d3f00ec1e`, did any
contributor use VisA test images, anomaly masks, test labels, model predictions,
performance metrics, or qualitative failure examples to make scientific choices?

The user explicitly declared **UNKNOWN** in the authorized Phase 4C final
human-attestation update on 2026-09-29. To the best of the user's current
knowledge, they cannot establish with sufficient confidence whether, before
the pre-data freeze commit above, VisA test images, anomaly masks, test labels,
model predictions, performance metrics, or qualitative test examples may
previously have been inspected or used by any contributor whose work influenced
VisionGuard.

This response applies to all six evidence types. UNKNOWN is not NO and does not
establish that exposure did occur. Contributor-by-contributor details, exposure
dates and affected decisions have not been established. No additional declaration
about later non-integrity exposure was supplied in this response.

## Repository/tool evidence — distinct from declaration

The recorded pre-data commit precedes acquisition/audit in this workflow.
The authorized integrity audit decoded test assets for inventory, shape, mask
semantics and cryptographic overlap checks without performance evaluation or
qualitative display. The audit passed and is bound to the committed report.
Development engineering uses fitting/calibration normals only. These are
verifiable workflow facts, not proof of every contributor's prior access.

| Evidence class | Current status |
|---|---|
| Human access-history declaration | UNKNOWN |
| Repository chronology | VERIFIED for the recorded workflow |
| Dataset integrity | VERIFIED by the bound integrity audit |
| Independent reservation of VisA test evidence | NOT ESTABLISHED |

Git history, absence of committed results, archive hashes, integrity audit,
zero train/test overlap and lack of known performance metrics do not establish
historical independence. All engineering readiness evidence remains valid;
it answers different questions from independent reservation.

## Independence decision

**NOT ESTABLISHED.** VisA must not currently be described as an independent
confirmatory benchmark, untouched independent evidence, or proof of
generalization from an independently reserved test set.

Unless this uncertainty is credibly resolved before test execution, a future
authorized evaluation must be described as **held-out VisA evaluation with
historical access independence unverified**. The limitation must remain visible
in every future report; the phrase "external robustness evaluation" alone must
not conceal it. See the two conditional paths in the
[Phase 4D draft](../planning/phase-4d-visa-confirmatory-execution-plan.md).

This is governance/provenance metadata, not a scientific hyperparameter. Frozen
protocols, fingerprints, membership, thresholds, seeds, model settings and the
dataset audit are unchanged. Historical engineering receipts retain their
then-current pending-attestation status; this document records the later human
response without rewriting those receipts.

Final-test lock: **CLOSED**. Phase 4D: **NOT AUTHORIZED**. Neither the declaration
nor PR readiness authorizes either future evaluation path.
