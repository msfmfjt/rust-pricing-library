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

The technical release gate for a candidate commit is the same command set used
by CI and documented in the repository README:

- Local Volatility reference fixture validation;
- Path Dependence and Early Exercise reference fixture validation;
- JSON Schema validation;
- local Markdown link validation;
- retained source archive validation;
- Rust formatting, Clippy, unit/integration tests, and statistical acceptance;
- dependency-direction validation;
- Rust API documentation generation;
- Python extension build, wheel smoke test, Python smoke suite, benchmark run,
  and benchmark artifact validation.

Passing these checks establishes that the commit is ready for private artifact
retention on the supported CI platforms. It does not resolve publication,
licensing, or access-control decisions.

Path Dependence and Early Exercise have completed their P0-P8 and E0-E8 gates.
Their supported Apple Silicon macOS and Windows x86-64 replay fixtures are
frozen, and the latest pull-request CI for this status passes the complete
Rust, Python wheel, statistical acceptance, replay, and benchmark matrix.

Before any public publication, the following non-code decisions must be resolved
outside the repository:

- private artifact repository location and access policy;
- internal licensing terms for private consumers;
- whether a public open-source license will ever be selected.

Until those decisions are recorded, the repository remains private-distribution
only. The absence of a selected license means publication of the repository does
not grant permission to use, modify, or redistribute the code.
