# Product Graph division AAD review — 2026-10-27

## Finding

This follows #187 (`f0b25f60beb88555de0b6c082d2b1d0c9b41c063`), with main
still at `45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. No merge is authorized;
the separate #172 mutation-testing branch is outside this work.

P1: the shared payoff reverse pass can silently return a zero denominator
sensitivity for a finite quotient with a finite nonzero derivative. Its
`-seed * numerator / (denominator * denominator)` expression can overflow
the square, or underflow an intermediate product, before the final division.

The installed parent implementation reproduces this through public Python
`GraphProduct` / `PricingTask` APIs with `C / (Spot * C)`, Spot 100 and an
observation/payment on the valuation date:

| C | Price | Delta before correction | Analytical Delta |
| --- | --- | --- | --- |
| 1 | 0.01 | -0.0001 | -0.0001 |
| 1e200 | 0.01 | 0 | -0.0001 |
| 1e-200 | evaluation error | nonfinite intermediate | -0.0001 |

Both Python regressions fail on that wheel; the first Rust public-graph
regression also fails against the unmodified #187 source. The baseline wheel
is the retained #185 installation: its native binary matches the installed
RECORD SHA-256 `3e32847fe49c09fdffef509a4d356ba698091eecc5977ad6aa2a6730bfad6ad0`.
The production Rust source, Cargo inputs and toolchain are unchanged between
#185 and #187.

## Correction and replay impact

The fix belongs in the shared payoff reverse kernel. Graph source objects,
Task selection, Python wrappers and each financial model continue to use that
kernel without separate validation or derivative implementations.

When the numerator product and denominator square are normal binary64 values,
the original primitive operations and order remain unchanged. Otherwise,
finite nonzero operands are decomposed exactly into signed fractions and
binary exponents. The fraction products stay in range; the exponent is applied
at the end, with a single final rounding if the result is subnormal. Exact zero
seeds/numerators retain a zero denominator contribution. No clamp or numerical
tolerance is introduced, and an actually overflowing adjoint still fails.

This is a correction to the existing quotient derivative, not a new payoff or
risk convention. No opcode, tape layout, source schema, public signature or
fingerprint encoding changes. Ordinary outputs and their identities must remain
byte-identical. Extreme inputs that previously returned an incorrect zero or
failed because of these intermediates now return the corrected derivative;
reproducing that historical behavior requires the historical implementation.
Other unrepresentable graph intermediates, including a genuinely overflowing
numerator adjoint, remain errors. This is not a claim of uniform relative
accuracy for every binary64 expression or a change to any calibration method.

## Regression coverage

- Rust public graph tests cover positive/negative scales from `2^-1000` to
  `2^1000`, both terminal and pre-dividend observation adjoints, overflowing
  and underflowing seeded products, subnormal input/output values, ties to zero,
  zero seeds and true overflow/domain failures. Analytical powers of two use
  exact equality. Ordinary nonbinary operands retain the old derivative bits.
- Ten additional references use `Decimal.from_float` inputs and 2,500-digit
  evaluation of `-w*x/(y*y)`. Their maximum permitted difference is two ULPs,
  including subnormal results; these nonzero references cannot pass as zero.
- Installed-wheel tests exercise current/future observations, MC/RQMC,
  workers 1/3, complete task replay, and basket price-to-AAD reselection with
  both Deltas and cross Gamma. Binary rescaling requires exact equality of
  prices, risk blocks and sampling errors. Basket Deltas are also checked
  against the pathwise homogeneity identities `P/S1` and `-P/S2`.
- `scripts/capture_task_risk_replay.py` captures 70 complete existing cases:
  nine source families under six execution variants, and 16 continuous-barrier
  covariance/reporting cases. Run it with the parent and candidate installed
  in separate environments and compare the output bytes.

Local validation uses the pinned Rust 1.98.1 toolchain:

- Workspace/all-feature tests excluding the binding crate completed with exit
  zero. The six new Rust regressions pass; the focused Graph/Task/reuse/report
  selection passes all 35 tests with all features and without default features.
  Native Python-binding Rust tests pass all seven cases. Five explicit ignored
  statistical/path-dependent/early-exercise acceptance tests also pass.
- A fresh CPython 3.12 release wheel passes package, metadata, dependency SBOM,
  ABI, installation and all 341 runtime callable checks. The two new Python
  regressions and 66 related installed-wheel tests pass. An initial related-test
  command named a nonexistent `test_multi_asset_hw` module and failed; the
  corrected command executes the actual `test_hull_white` module successfully.
- All 70 complete before/after captures are byte-identical, SHA-256
  `d1c94f2fe16e96f4a13c9385c20c492e701f4a2d47a5fda450dc665e9e8be816`.
  The [replay evidence](graph-division-replay-2026-10-27.json) lists every case
  and retains the capture-script and candidate-wheel digests.
- Formatting, all-target check, all-target/all-feature Clippy with warnings
  denied, Rust docs and dependency direction pass. All 18 reference/schema/link
  scripts and six checker suites pass; unavailable physical macOS/LLDB checks
  are platform skips, not evidence of local native-debugger operation.

The full local benchmark runner stopped because it could not observe process
peak RSS for `benchmark_local_vol`; the numerical executable itself exited
successfully. The benchmark gate is **not passed locally**, and its measurement
requirement is unchanged. The European replay generator succeeds, but its
frozen-fixture checker rejects `linux-x86_64`: the retained goldens support
macOS ARM64 and Windows x86_64 only. No fixture or platform marker is changed
to force a comparison. All four frozen replay checks and the complete
benchmark/memory checks remain hosted gates. Full installed-wheel numerical shards, full
default/minimal suites and the remaining heavy ignored panels also require CI.

## Independent CI blocker

#187's Linux, macOS and Windows Rust jobs pass. The Windows job
`121972530970` log confirms execution of all three failure-propagation tests
and successful report-migration tests in default, minimal and all-feature runs.
This resolves the false-success issue described in the
[Windows review](windows-ci-failure-review-2026-10-27.md).

#186's macOS wheel shards `121915875827` (coupled provider) and `121915875594`
(custom provider) genuinely stop on invalid frees. Their LLDB artifacts
`12711366433` and `12734717299` were downloaded and digest-verified. Both show
`_dispatch_cache_cleanup` / `_pthread_tsd_cleanup` on a system workqueue thread,
while the main thread waits for Rayon and pricing workers remain in path-chain
evaluation. This matches the detection site in the earlier
[native crash record](macos-path-ledger-crash-2026-10-26.json), but does not
identify the origin of corruption. A loaded Accelerate/BLAS image alone is not
evidence of causation. No allocator, threading policy, test selection, retry
policy, sample count or tolerance is changed for these failures.

The independent extended risk/price panels on #186 passed on all three
platforms (11 risk and 27 price tests each). That does not resolve the macOS
wheel crash or replace current-head CI. Complete wheel validation and the
native-crash investigation remain merge blockers.
