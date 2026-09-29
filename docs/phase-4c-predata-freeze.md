# Phase 4C pre-data scientific design

This distinct commit precedes VisA split-row inspection, acquisition, extraction,
and integrity auditing. It freezes choices, not dataset observations or model
performance. Phase 4B PR #16 was verified merged at
`c5bc3f8269a2b507c57a6663360c2846f90922da`; its fingerprint recomputed to
`94442ab3121bccd392e3805b6134710cc6e8c95e8f17b8eccda288f8b1bd672d`.

The [machine-readable design](../configs/protocols/visa-predata-design-v1.yaml)
fixes all 12 categories and seeds 42/123/2026. Per category, order official
training-normal identities by SHA-256 of UTF-8
`visionguard-visa-development-v1\0{category}\0{image_path}`, breaking any digest
tie by image path. The first `floor(n/10)` are calibration; the remainder fit.
Require at least 19 calibration images and no cross-role exact content overlap.
Membership is identical for both models and all seeds. No test normals may be
borrowed and no fraction may be revised after outcomes.

Transfer the reviewed model scientific configurations, including EfficientAD's
70,000 steps. VisA lacks an official validation split. For EfficientAD, fit the
unchanged internal normalization quantiles on fitting normals after training;
reserve the calibration normals exclusively for threshold estimation. This
role change is declared before seeing any VisA data to separate threshold
calibration from model/normalization fitting. It does not promise operational
false-positive control. All other changes must be structural dataset bindings.

Acquire official source metadata and archive, audit identities/decode/masks and
cryptographic duplicates, and stop on scientific-integrity errors. Later final
protocols may bind observed archive/split/membership hashes only after the audit
passes; this design must not be silently rewritten to accommodate a failure.
The reviewed triage rule stays unchanged. No final-test performance evaluation,
full training matrix, or Phase 4D execution is authorized.
