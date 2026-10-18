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
- Linux CI exposed BLAS-dependent last-bit differences in the coupled basis.
  Sparse, ascending-column fused accumulation now reproduces every bit of both
  accepted matrices, including a correctly rounded Python 3.12 fallback. The
  matrix hashes and rejection of changed grids remain unchanged.
- The cached-failure unit test supplies a native-module double, so the companion
  suite needs no native installation. The smile tangent test checks quadratic
  convergence above the price-roundoff floor using 2e-5/1e-5 bumps and retains
  its original 1e-6 fine-bump absolute bound; numerical tolerances are unchanged.

## Validation

Completed before the integration correction: workspace/all-target compilation,
all-feature Clippy with warnings denied, formatting, 294 companion tests,
6 wheel-contract tests, 10 runtime-provenance tests, 4 raw-basis tests,
16 independent-reference/schema/documentation guards and source-archive checks.
The resolved-Barrier regression passed after the correction. Full merged Rust,
release-wheel and platform CI results are tracked in the PR before merge.

An exploratory discovery run of all `scripts/test_*.py` is not a portable
acceptance suite: some tests require the unshipped historical replay/shard files
and their original accepted source/binary, and unrelated Heston reference tests
also require `mpmath`. That run is not reported as passing. The strict historical
guards must reject this merged source until new numerical evidence is produced.
