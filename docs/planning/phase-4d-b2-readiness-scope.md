# Phase 4D-B2 readiness scope — engineering only

Issue: https://github.com/ahmad-527/visionguard-ai/issues/23

Start from reviewed B1 regular merge
`d335b91d10489b4c0811530bd4d69418f68fb2ff`. Keep B1 implementation/protocol/gate
unchanged. Add separate B2 readiness integration, synthetic admission and
known-answer full-matrix replay, immutable output/restart receipts, negative
authorization tests, and bounded native synthetic GPU smoke on frozen models.

No VisA final-test enumeration, stat, hashing, decoding, inference or performance
evaluation is authorized. No training, recalibration, threshold/model changes,
best-seed selection or localization fusion. The B1 gate remains permanently
closed. All real B2 entry points remain closed in this readiness stage; human
approval of the final implementation and an independently recorded authorization
are required in a subsequent execution stage. A matching hash is not permission.

Frozen inputs:

- B1 fingerprint: `0578ca3bc7765657fe568e7db670088490639ef5f011315ab043391c8a8fcb49`.
- Published 72-model freeze: `d75712cad7fcfa9309c12882d869db65627e7c325723c48459ef8e1da90be061`.
- Development contract: `31a5b53d91bf7f609b5dfa3a97b87c4c6617a767d9096fa845f774a5f9bfefc3`.
- Historical access UNKNOWN; independent reservation NOT ESTABLISHED.
- Future interpretation: Held-out VisA evaluation with historical access
  independence unverified.
- Expected scope from committed audit only: 2,162 images, 12,972 model-image calls,
  12 categories × seeds 42/123/2026 × 2 models, 36 same-seed pairs.
- Future D: output capacity: 128 GiB run budget + 20 GiB reserve = 148 GiB minimum.

Preserve all 72 models, thresholds, previous failures/recoveries, C: originals
and Git chronology. Open early draft PR; do not merge. Final readiness freeze,
tests, measured resources, limits and CI will be recorded before requesting real
execution permission. Completing readiness does not start held-out evaluation.
