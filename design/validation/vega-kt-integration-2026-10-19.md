# VegaKT integration with current main, 2026-10-19

The integration PR is [#147](https://github.com/msfmfjt/rust-pricing-library/pull/147).
It combines the complete VegaKT/backtesting workspace with main
`965a5f654218bfe7d6172c87a3600bafb5fa8f3f`, including Bass local volatility,
Andersen–Broadie and the newer rough-volatility APIs.

## Preserved historical acceptance

The original 614-file workspace change was captured without modifying its
checkout, index, installed native library or local evidence. Commit
`2a699bfa2b28cc77dbda75f87057c66194c9a679` preserves that source snapshot.
The original eight-day manifest and accepted-runtime checks passed before
integration.

The [eight-day record](vega-kt-quote-control-day-008-v0.1.md) covers 8 of 252
synthetic trading dates, 240 product-days and 8,192 comparisons. It is historical
evidence for its recorded source and binary, not certification of this merged
source, an annual completion claim, unrestricted-smile acceptance or live-market
hedging evidence. No numerical thresholds or historical hashes were rewritten.
Large raw outputs under `benchmark-results/` remain in the original local
checkout and are not part of the source archive.

## Integration corrections

- The terminal VegaKT estimators now allow a continuous Barrier whose history
  has already resolved survival. The compiled payoff is then the terminal
  vanilla or the fixed rebate. Live monitoring remains outside this shortcut.
  A regression compares both Call/Put sides, knock-in/out and both histories,
  pseudo-MC/RQMC, and zero/nonzero carry. Zero carry exercises the pathwise
  estimator; nonzero carry exercises exact terminal replication.
- The wheel contract includes the existing, opt-in `retain_samples=False`
  argument. Mutation tests reject its removal or enabling it by default.
- Clean-wheel validation installs the companion's validation and constraints
  dependencies, including the schema validator used by the native API tests.
- Provenance unit tests use a temporary test-only acceptance manifest and also
  check that the real historical manifest rejects changed sources. Production
  provenance checks remain strict; this does not promote the merged binary.
- Documentation labels local raw evidence as such instead of linking to files
  absent from a clean source archive.
- Linux CI exposed build-dependent last-bit differences in both `np.interp`
  and BLAS. Explicit fused hat interpolation and sparse, ascending-column
  fused accumulation reproduce every bit of both
  accepted matrices, including a correctly rounded Python 3.12 fallback. The
  matrix hashes and rejection of changed grids remain unchanged.
- The cached-failure unit test supplies a native-module double, so the companion
  suite needs no native installation. The smile tangent test checks quadratic
  convergence above the price-roundoff floor using 2e-5/1e-5 bumps and retains
  its original 1e-6 fine-bump absolute bound; numerical tolerances are unchanged.
- Native wheel jobs retain two identical Local Volatility replay captures with
  the source commit and toolchain before the extended Python suite. The later
  benchmark regeneration and exact platform replay gates remain required.
- Rustdoc treats the Gaussian-variance matrix equation as code, preventing its
  matrix indices from being parsed as a broken documentation link.
- Replay-audit mutation tests write explicit UTF-8/LF bytes on Windows, so they
  exercise numerical and structural rejection instead of failing early on CRLF
  formatting. The production replay format and byte-comparison gate are unchanged.

## Validation

The merged Rust suite passed 1,011 tests (84 explicitly ignored). The separate
Python-binding Rust suite passed 3 tests, and the statistical, path-dependence
and early-exercise acceptance suites passed 5 tests. Workspace/all-target
compilation, all-feature Clippy with warnings denied, formatting and strict
Rustdoc passed. Six wheel-contract tests and 16 independent-reference, schema
and documentation guards also passed.

[Backtesting CI](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39178908181)
passed on Linux, macOS and Windows: 295 companion tests, 4 raw-basis tests and
10 runtime-provenance tests on each platform. Eighteen original/integrated basis
comparisons (4/6 intervals, three date scales and three policies) preserved the
complete matrix bytes and construction audits, including the frozen hashes.

The macOS release wheel built and all four benchmark replay files matched their
tracked platform fixtures exactly. Two further native Local Volatility captures
matched each other and the tracked golden. An independent discounted Black
calculation with forward `S*Dq/Dr`, terminal reporting-IV interpolation and
central quote bumps agreed with both full-risk cases within 3.9e-10 (unchanged
2e-6 absolute bound). Mutations of the price or bucket allocation were rejected
by the review checks. The complete clean-wheel Python suite and final platform
CI are tracked in the PR before merge.

Native Windows replay from the same integrated source was captured twice in
[CI run 39179699272](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39179699272).
The captures match exactly and both pass the unchanged replay gate after the
reviewed golden update. The [Windows audit](../../fixtures/replay/local-vol-terminal-replication-windows-change-audit-v1.json)
preserves native source/toolchain provenance and accounts for all 212 changes:
206 VegaKT fields, two plan fingerprints and four one-ULP standard-error changes.
Prices, non-VegaKT Greek point estimates, requests and execution inputs are unchanged.
All 12 independent quote checks pass within 3.855e-10; negative price and bucket
mutations are rejected. The prior Windows fixture is preserved byte-for-byte.

An exploratory discovery run of all `scripts/test_*.py` is not a portable
acceptance suite: some tests require the unshipped historical replay/shard files
and their original accepted source/binary, and unrelated Heston reference tests
also require `mpmath`. That run is not reported as passing. The strict historical
guards must reject this merged source until new numerical evidence is produced.
