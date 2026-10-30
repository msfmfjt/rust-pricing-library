# Release readiness v0.1

Status: historical v0.1 baseline; current development candidate gates incomplete

The original v0.1 baseline completed the private artifact gates described below.
Readiness applies to the exact validated commit and does not transfer to later
development candidates. The 2026-10-29 [native crash review](validation/native-crash-triage-2026-10-29.md)
records two failed and six capacity-cancelled wheel selections at PR #190's head.
That development candidate is not ready: the native failures remain unresolved
and incomplete selections still require successful execution. A successful
rerun alone does not establish a repair for an intermittent crash.

The supported private artifact set is:

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

Path Dependence and Early Exercise completed their P0-P8 and E0-E8 baseline
gates. Their supported Apple Silicon macOS and Windows x86-64 replay fixtures
are frozen. Each subsequent candidate must pass the complete Rust, Python wheel,
statistical acceptance, replay and benchmark matrix against those fixtures.

Before any public publication, the following non-code decisions must be resolved
outside the repository:

- private artifact repository location and access policy;
- internal licensing terms for private consumers;
- whether a public open-source license will ever be selected.

Until those decisions are recorded, the repository remains private-distribution
only. The absence of a selected license means publication of the repository does
not grant permission to use, modify, or redistribute the code.
