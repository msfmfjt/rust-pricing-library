# European Black–Scholes replay fixtures

These fixtures freeze complete Pseudo-MC and randomized Sobol QMC full-risk
executions on the two MVP targets. Each case contains the normalized request and
result, Request and Plan fingerprints, exact binary64 bit patterns for estimator
and CRN-validation diagnostics, RNG table checksums, and the effective execution
policies.

`scripts/check_replay_fixture.py` performs an exact UTF-8 byte comparison. The
supported-wheel CI jobs regenerate the evidence with the pinned toolchain and
compare it with the matching platform file:

- `european_bs-macos-aarch64.json` for Apple Silicon macOS;
- `european_bs-windows-x86_64.json` for Windows x86-64 MSVC.

The primary estimates currently match across the two platforms, while a small
number of CRN difference diagnostics differ in low bits. They are intentionally
stored separately because the reproducibility contract guarantees bitwise
identity only on the same platform. Fixture changes require a reviewed numerical
explanation; regenerating a file is not, by itself, acceptance evidence.
