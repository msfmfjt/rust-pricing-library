# Replay fixtures

These fixtures freeze complete replay evidence on supported targets. Each case
contains the normalized request and result, Request and Plan fingerprints, exact
binary64 bit patterns for estimator and CRN-validation diagnostics where
applicable, RNG table checksums, and the effective execution policies.

`scripts/check_replay_fixture.py` performs an exact UTF-8 byte comparison. The
supported-wheel CI jobs regenerate available evidence with the pinned toolchain
and compare it with the matching platform file:

- `european_bs-macos-aarch64.json` for Apple Silicon macOS;
- `european_bs-windows-x86_64.json` for Windows x86-64 MSVC.
- `local_volatility-macos-aarch64.json` for Apple Silicon macOS Price-only
  Local Volatility Pseudo-MC and randomized Sobol QMC.

The primary estimates currently match across the two platforms, while a small
number of CRN difference diagnostics differ in low bits. They are intentionally
stored separately because the reproducibility contract guarantees bitwise
identity only on the same platform. Fixture changes require a reviewed numerical
explanation; regenerating a file is not, by itself, acceptance evidence.
