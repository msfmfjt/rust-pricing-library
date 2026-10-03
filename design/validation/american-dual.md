# Andersen–Broadie validation

Implementation contract: [ADR 0027](../adr/0027-american-dual.md).
User API and reproduction commands: [primal-dual guide](../../docs/library/american-dual.md).

## Coverage

The private exact-tree tests enumerate every path for optimal, always-exercise,
and expiry-only policies. They verify the lower/upper ordering and exact
equality for the optimal rule. An independently enumerated two-date example
with one inner sample checks the upward noise bias in both exercise regions.

The public [adapter tests](../../crates/pricing/tests/american_dual.rs) cover:

- single-expiry equality with the existing pricing API and zero dual gap;
- deterministic Call/Put values, valuation-date exercise and cash/proportional
  dividend collisions against an explicit deterministic carry/jump formula;
- unchanged LSM lower values and policy fingerprints with discrete dividends;
- exact repeated runs and invariance across one and two workers;
- agreement of the existing constant-volatility Black-76 and BS adapters;
- zero inner counts, RQMC, Greeks and random-coordinate overflow rejection;
- independent CRR lattice comparisons for a put, yield-paying call, and
  cash/proportional-dividend put (using a residual-martingale lattice),
  using three outer seeds and 16/256 continuation inner paths.

The lattice uses exercise dates aligned to calendar days, with two and four
steps per day. Its refinement difference must be under 0.03. Statistical
acceptance uses five standard errors plus the 0.03 lattice allowance, separately
on each side. This is a regression gate, not a claim that every realized 95%
interval covers the reference. Inner-count refinement is checked across seeds;
individual Monte Carlo estimates need not be monotone in inner count.

The [Python boundary tests](../../tests/python/test_american_dual.py) check
LSM equivalence with dividends under BS/Black-76 and both antithetic settings,
expiry-only and deterministic time-zero exercise limits, repeat/worker/request
round-trip replay, all exposed metadata and read-only properties. They also
check zero counts, unsigned integer boundaries, unsupported models/engines/risks
and invalid execution settings. Scalar versus joint statistical reductions are
compared with floating-point tolerance where their operation order differs.

## Execution evidence

Local validation on 2026-10-03, macOS Apple Silicon, Rust 1.98.1:

- Three exact kernel/namespace tests and five public conformance tests pass.
- Workspace tests (all features, excluding the Python binding crate): 654
  passed, zero failed, 51 explicitly ignored. The binding crate's three Rust
  tests pass separately. The pricing crate also passes 643 tests without
  default features (51 ignored); the all-target workspace check passes.
- The release acceptance test passes all 18 combinations (three products ×
  three outer seeds × two inner counts), including independent lattice
  refinement. The existing early-exercise statistical acceptance tests also pass.
- Workspace Clippy with all targets/features and denied warnings, Rust API
  documentation, reference fixtures, schemas, dependency direction, source
  archive contents and local Markdown links pass.
- The CPython 3.13.1 macOS arm64 release wheel passes the full archive,
  metadata, type-stub and runtime API checks in a clean environment. All 167
  Python tests pass, including eight new dual-boundary tests; all examples
  invoked by the wheel smoke suite pass. The Python and Rust dual examples
  produce identical complete outputs, including both fingerprints.
- Existing MC uncertainty and path-dependence statistical acceptance pass.
  The full Rust/installed-wheel benchmark suite completes, report validation
  passes, and all four local platform replay fixtures match (European BS,
  local volatility, path dependence, early exercise). These are compatibility
  checks; no dual-method performance improvement is claimed.
- All 27 extended-model price acceptance tests and all 11 extended-risk
  acceptance tests pass. The price run retains the five earlier completed
  cases and finishes the remaining 22 using four independent processes, each
  with one test thread. Seeds, sampling sizes, refinements and acceptance
  budgets are unchanged. Rust source checksums match before and after the run.
  The price tests cover 74 panels and 222 quote points; the risk tests cover
  738 derivative sweeps with no rejected comparison. The largest normalized
  price-budget usage is 0.803089; the largest required-bump derivative-error
  budget usage is 0.037159. See the
  [retained case-by-case results](american-dual-release-gates.json) for source
  identity, log checksums, numerical summaries and reproduction commands.
  These are regression checks of existing models; they do not extend dual
  pricing to those models or add dual Greeks.

Average duality gap across outer seeds 71, 97 and 113:

| Product | Four-step/day tree | N2=16, N3=9 | N2=256, N3=129 |
| --- | ---: | ---: | ---: |
| Put, q=0 | 5.95602312 | 0.66509645 | 0.28323050 |
| Call, q=0.10 | 5.77566733 | 0.87168115 | 0.39607653 |
| Put, cash 2 and proportion 0.03 | 8.15420960 | 0.57645043 | 0.16805936 |

All cases use S=K=100, r=0.05, sigma=0.20, the four dates in the example,
8,192 antithetic training units and 2,048 antithetic outer units. Cash and
proportional dividends occur together on the second exercise date. Sampling
error can put a realized upper estimate below the lattice price; e.g. seed 97
in the last case gives 8.13531033 with SE 0.07389999. The upper-bound claim is
about its expectation, not every realized estimate.

The runnable example uses 4,096 outer units, N2=256 and N3=128 and returns:

| Quantity | Value |
| --- | ---: |
| Lower estimate | 5.71965283 |
| Lower SE | 0.05841363 |
| Upper estimate | 6.00306245 |
| Upper SE, including covariance | 0.05785276 |
| Gap | 0.28340962 |
| Gap SE | 0.00471393 |
| Asymptotic 95% bracket | [5.60516422, 6.11645178] |

Cross-platform CI and continuous-exercise convergence are separate from this
local finite-grid validation. There is no versioned dual-result wire adapter.
