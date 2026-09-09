# Release readiness v0.1

Status: code and private artifact gates ready; external publication decisions open

The v0.1 implementation is prepared for private distribution through retained CI
artifacts. The supported artifact set is:

- platform-specific CPython wheels for Apple Silicon macOS and Windows x86-64;
- the retained Rust source archive for the exact commit;
- benchmark and replay artifacts for the supported platforms.

The source archive gate checks that the retained archive contains the locked
dependency metadata, Rust and Python sources, schemas, fixtures, CI workflow,
validation scripts, benchmark entry points, documentation, and QMC direction
number data required to rebuild and review the private artifacts.

Before any public publication, the following non-code decisions must be resolved
outside the repository:

- private artifact repository location and access policy;
- internal licensing terms for private consumers;
- whether a public open-source license will ever be selected.

Until those decisions are recorded, the repository remains private-distribution
only. The absence of a selected license means publication of the repository does
not grant permission to use, modify, or redistribute the code.
