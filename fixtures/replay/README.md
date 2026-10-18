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
- `local_volatility-macos-aarch64.json` for Apple Silicon macOS Local
  Volatility Price-only and Delta/Gamma/Vega/VegaKT Pseudo-MC and randomized
  Sobol QMC.
- `local_volatility-windows-x86_64.json` for Windows x86-64 MSVC Local
  Volatility Price-only and Delta/Gamma/Vega/VegaKT Pseudo-MC and randomized
  Sobol QMC.
- `path_dependence-macos-aarch64.json` for exact and smoothed Digital, smoothed
  discrete and continuous Barrier, partially fixed Arithmetic Asian, and Fixed
  Lookback calculations on Apple Silicon macOS.
- `path_dependence-windows-x86_64.json` for the same Path Dependence cases on
  Windows x86-64 MSVC.
- `early_exercise-macos-aarch64.json` for American Put Price-only and
  fixed-policy Delta/Gamma/Vega under Pseudo-MC and randomized Sobol QMC. Its
  complete result records the fitted policy, training and valuation domains,
  stopping indices, and regression diagnostics.
- `early_exercise-windows-x86_64.json` for the same Early Exercise cases and
  complete replay state on Windows x86-64 MSVC.

The primary estimates currently match across the two platforms, while a small
number of CRN difference diagnostics differ in low bits. They are intentionally
stored separately because the reproducibility contract guarantees bitwise
identity only on the same platform. Fixture changes require a reviewed numerical
explanation; regenerating a file is not, by itself, acceptance evidence.

## Local-node allocation correction, 2026-10-03

The historical macOS Local Volatility fixture before terminal replication reflects the previously validated
integrated-adjoint and exclusion-audit corrections. The
[earlier macOS fixture](historical/local_volatility-macos-aarch64-before-node-allocation-audit.json)
is retained byte-for-byte. The [46-field change audit](local-vol-node-allocation-change-audit-v1.json)
confirms that every changed field is inside the VegaKT report of the two
full-risk cases. Prices, Delta, Gamma, scalar local Vega, requests, fingerprints
and execution/RNG diagnostics are unchanged. Independent initial-row price
bumps pass 48 checks; see the [numerical explanation](../../design/validation/vega-kt-backtest-allocation-audit-v0.1.md).

This freezes replay of `local_vol_node_allocation_diagnostic_v1`, **not**
acceptance of the unfinished paper VegaKT decomposition. Its numerical
acceptance thresholds and the companion's diagnostic-only guard are unchanged.
The Windows Local Volatility fixture still represents the earlier algorithm;
it must be regenerated and reviewed on Windows before that platform's current
replay can pass. A macOS capture is not substituted for Windows evidence.

Run `python scripts/audit_local_vol_replay.py <native-candidate.json> --output
benchmark-results/local-vol-candidate-audit.json` to inspect a candidate. This
read-only audit allows only the 46 field paths reviewed above, checks the
diagnostic labels and scalar/bucket/residual accounting, and rejects changes
to prices, other Greeks, requests and fingerprints. Its
`requires_native_numerical_review` status does not approve a replacement.
The exact replay gate remains unchanged. Native capture provenance and an
independent allocation probe remain required; see the
[cross-platform record](../../design/validation/library-cross-platform-2026-10-04.md).


## Terminal replication extension, 2026-10-04

The current macOS capture uses `terminal_payoff_replication_iv_v1` for its two
European risk cases. The immediately preceding
[capture](historical/local_volatility-macos-aarch64-before-terminal-replication.json)
is retained byte-for-byte. The [218-field audit](local-vol-terminal-replication-change-audit-v3.json)
records 212 VegaKT fields, two method-dependent plan fingerprints and four
pseudo-MC Delta/Gamma SE changes of one ULP from the existing scalar statistics
reduction. Price-only cases, all point estimates, requests and RNG inputs are
unchanged. Both pseudo-MC and RQMC market-IV risks match independent quote price
differences within 6.804e-10, including deterministic continuous-dividend carry.
Sample moments for exact replication are unavailable, not fabricated as zero.
See [P0 smile acceptance](../../design/validation/vega-kt-smile-acceptance-v0.3.md).

The exact byte-comparison gate remains unchanged. The old
`audit_local_vol_replay.py` remains an audit of the historical 46-field
node-allocation correction and deliberately rejects this new algorithm; its
tests use the retained historical captures. It must not be used to approve the
terminal-replication transition. Windows retains its earlier capture pending
regeneration and review on that platform; no cross-platform acceptance is claimed.
