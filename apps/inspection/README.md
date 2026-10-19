# VisionGuard local inspection application — untagged candidate

Separate packaging leaves the historically frozen core build recipe unchanged.
Install the matching reviewed `visionguard-ai` core distribution first, then this
distribution. Do not substitute an unrelated 0.1.0 core wheel merely because its
version matches. Verify both distribution hashes from the review packet.

`visionguard-inspect --manufactured-demo` starts a loopback-only generated-intensity
fixture, **not native machine-learning inference**. Without an explicitly pinned,
authorized registry or the demo flag, the service reports not ready.

See `docs/v1-application-installation.md`, `docs/v1-application-limitations.md` and
`docs/v1-release-checklist.md` in the reviewed source checkout for setup, approval
requirements and outstanding release gates. The application does not train,
calibrate, select a winning model or access evaluation datasets on the server.

> Held-out VisA evaluation with historical access independence unverified.

No v1.0 release, activation, deployment or source-license decision is implied.
