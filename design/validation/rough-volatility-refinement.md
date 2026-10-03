# Rough-volatility finite-grid and lift-kernel refinement

Prepared 2026-10-03 from PR #118 head
`ea16853afbf7282aa53e09e4f2a4f8523543657a`, source tree
`1012eed6ec3cde2b72506af7ee9ec9f7e409eb35`.
The downloaded source tree and Rust 1.98.1 toolchain artifact were hash-verified.
This follow-up changes tests, reference fixtures, CI and documentation only.
Production pricing, model definitions, RNG layouts, fingerprints, APIs, schemas
and Cargo.lock are unchanged. Main and the existing PR branches are not modified.

[Model contracts](../../docs/models/rough-volatility-families.md) and
[earlier two-step references](rough-volatility-families.md) remain applicable.

## Coupled stochastic time-grid panel

The test `six_families_coupled_time_grid_price_refinement` uses all six families,
H=0.1/0.3, seeds 91/1973, spot and call strike 100, maturity one year,
zero rates and no dividends. Grids 16/32/64 are each compared with 128 steps:
72 comparison rows, including 24 final-level acceptance rows. The nonzero
volatility parameters follow the earlier two-step panel, with H varied:

| Family | Fixed parameters |
| --- | --- |
| Rough Heston | v0=.04, kappa=.7, theta=.055, nu=.18, rho=-.65 |
| Lifted Heston | Same Heston parameters; 20 factors, ratio=2.5 |
| Quadratic rough Heston | z0=.15, lambda=1.1, eta=.5, a=.8, b=.25, c=.02 |
| Mixed rough Bergomi | weights=.35/.65, eta=.45/1.05, rho=-.65, xi0=.04 exp(.08t) |
| Rough SABR | beta=1, eta=.75, rho=-.65, xi0=.04 exp(.06t) |
| RFSV | stationary initial state, kappa=1.3, log-vol diffusion=.2, mean=log(.2) |

Every sample averages a path and its full antithetic path. A separately written
two-pass variance computes the SE of the paired coarse-minus-fine payoff.
It does not add separate coarse and fine variances and discard their covariance.
Samples are stored in index order; four test-only workers do not alter RNG
coordinates or aggregation order. The production pricing scheduler is unchanged.

The final 64/128 requirements, fixed before the first numerical run, are:

- `abs(mean difference) + 4 * paired SE <= 0.10`;
- `0 < paired SE <= 0.01`.

These budgets are **price currency units**, not implied-volatility basis points.
Earlier 16/128 and 32/128 levels are diagnostics, not acceptance conditions.
Per-row statistical checks are not simultaneous confidence guarantees.
The paired/unpaired SE ratio is reported but has no optimized cutoff.

### Joint Gaussian construction

Brownian increments on a coarse cell are sums of its fine increments. For the
four power-kernel models the coarse newest-cell integral is sampled conditionally
on all fine Brownian increments and fine newest-cell residuals in that cell,
with a fresh independent Gaussian for the remaining conditional variance.
The conditional coefficients are independently retained in
[refinement.json](../../fixtures/rough-volatility/refinement.json).
Endpoint-weighted QUADPACK and 128-node Gauss-Jacobi integration agree on all
nontrivial cross-cell covariance integrals. Covariance identities and identity
marginals of every output shock vector are tested algebraically, not just via
empirical sample correlations.

This is a **pairwise** coupling of each coarse model to the fine model. Reusing
extra coordinates between different coarse levels is not claimed to construct
the full joint Brownian-integral law across every level simultaneously. The
reported paired SEs use only the valid pairwise laws.

The lift uses summed Brownian increments, with its weights and rates held fixed.
RFSV selects shared latent values from the fine Gaussian vector and applies a
separately coded triangular whitening transform for the coarse grid. It shares
the already independently validated fOU covariance function, not the production
factorization code. Identity marginals and coincident latent nodes are checked
for both stationary and fixed-initial-value cases.

### Initial precision failure and revised sample count

The first full run used 32,768 antithetic sampling units for every case. All
H=0.3 final-level rows passed. For H=0.1, ten final-level rows failed the SE cap;
four also failed the combined budget. The largest final-level paired SE was
0.027833200 and the largest bound was 0.163481745. The RFSV H=0.1 rows passed.
This initial failed run is retained in the delivery evidence.

The second protocol increases **all H=0.1 cases** to 524,288 units, retaining
all original units as a prefix. H=0.3 remains at 32,768 units. It preserves both
seeds, all models and parameters, grids, payoff, coupling and both budgets.
This is a disclosed precision-driven sample increase, not a tolerance increase
or a claim that the original run passed.

### Executed results

Executed on Linux x86_64, Rust/Cargo 1.98.1 and Python 3.13.5. The final
release panel passed all 72 comparison rows and all 24 final-level acceptance
conditions. The five fast tests passed separately in debug, release and
no-default-features modes. Workspace/all-target/all-feature Clippy with warnings
denied passed. `cargo test --locked --workspace` passed 654 tests with zero
failures and 48 ignored tests; this does not claim execution of all pre-existing
ignored acceptance panels. The four Python guard tests, independent reference scripts,
schemas, dependency-direction, Markdown links and a source-archive check passed.
The existing two-step reference/estimator release target was rerun with ignored
tests explicitly included: both tests passed.

| Family | Maximum `abs(gap)+4*paired SE`, 64/128 | Maximum paired SE, 64/128 |
| --- | ---: | ---: |
| rough_heston | 0.063913227 | 0.006913246 |
| lifted_heston | 0.025159679 | 0.004831467 |
| quadratic_rough_heston | 0.057641166 | 0.004003421 |
| mixed_rough_bergomi | 0.022370230 | 0.004393668 |
| rough_sabr | 0.019660842 | 0.004056514 |
| rfsv | 0.011548213 | 0.002104737 |

The overall maxima are 0.063913227 and
0.006913246, against unchanged budgets 0.10 and 0.01.
The largest absolute final-level mean difference is
0.041627482. The largest paired/unpaired SE ratio at the
final level is 0.527875; no efficiency
cutoff was selected. All 36 H=0.3 rows reproduce the initial run's printed
numerical results exactly despite the change from 32 to four test workers.
The delivery retains initial and final logs, parsed rows and their parser.

Earlier grids are deliberately not advertised as accepted: for example,
H=0.1 quadratic rough Heston at 16/128 (seed 1973) has a mean price difference
0.164547762 and `abs(gap)+4*SE=0.189149187`. The 64/128 comparison passing does
not turn that coarser grid into an accurate one or bound the remaining bias.

## Lift-kernel controls, distinct from time-grid tests

The retained fixture compares the implementation's generated weights/rates
against an independent formula using Python `math.gamma`. It checks H=0.1/0.3,
factor counts 8/16/32/64/128/256, and lags 1/128, 1/32, 1/8, 1/2 and 1 year.

An initial trial used the illustrative sequence `r_n=1+10*n^(-.9)` from
Abi Jaber, section 3.1. It did not meet the pre-set 2.5% maximum relative
sampled-lag kernel-error budget at 256 factors: the maxima were 3.8901% for
H=.1 and 20.2119% for H=.3. The initial failure is retained and is not reported
as successful convergence at that finite factor count.

The revised **test protocol**, not a production model change, uses
`r_n=exp(3/sqrt(n))`. Here `r_n -> 1` and `n*log(r_n) -> infinity`, shrinking
log-bin widths while expanding the integration interval in both directions.
With the unchanged 2.5% budget, the maximum sampled-lag errors at 256 factors
are 0.089567% for H=.1 and 0.931454% for H=.3, and decrease across every tested
factor count. The production factory still takes an explicit factor count and
ratio; no optimized default or automatic tolerance guarantee has been added.

These sampled-lag kernel checks are not an L2(0,T) error bound, do not include
the singular point t=0, and do not establish option-price accuracy. In particular,
a small adjacent-factor price change or a small sampled-lag kernel error would
not bound remaining continuous-time price bias.

## Fast controls and reproduction

Five fast Rust tests check 36 identity-marginal panels, six cross-cell covariance
cases, stationary/conditioned RFSV shared latent nodes, 12 lift kernels, and
12 public MC price/SE reconstructions at 64/128 steps. The latter uses 64
antithetic units and validates the public adapter's multistep aggregation, not
an independent continuous-time model price.

Four Python guard tests check independent fixture reconstruction, seven corrupted
or truncated fixtures, seven removed CI gates, and required source-archive
membership. Their first run caught a missing registration of the new test script;
the registration was added rather than weakening the guard.

```shell
python scripts/check_rough_volatility_refinement.py
python -m unittest discover -s scripts -p 'test_rough_volatility_refinement.py'
cargo test --locked -p pricing --test rough_volatility_refinement
cargo test --locked --no-default-features -p pricing --test rough_volatility_refinement
cargo test --locked --release -p pricing --test rough_volatility_refinement -- --include-ignored --nocapture
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
python scripts/check_markdown_links.py
```

The dedicated refinement job runs the ignored acceptance test explicitly on
Linux/macOS/Windows and retains `rough-volatility-refinement.log`. Source-archive
checks require the new files, explicit numerical command and retained artifact.
A workflow definition or a previous PR's success is not evidence that the new
head's remote checks have passed.

## Limits and remaining work

The finest grid is a comparison level, **not an exact reference**. These finite
comparisons neither prove a convergence order nor bound the remaining time,
truncation, kernel/lift or model bias. General nonuniform grids, continuous-time
or Fourier references, parameter stress, martingale diagnostics, broader
path-dependent payoffs, rough SABR beta below one, conditional-RFSV price
refinement, factor-price refinement and all new model combinations remain
separate work. Production admission remains open; calibration, Greeks/AAD,
VIX/SSR and Fourier/Riccati pricing are not added here.

## References

- Abi Jaber, *Lifting the Heston model*, sections 3.1–3.2 and Appendix A:
  <https://arxiv.org/abs/1810.04868>.
- Bennedsen, Lunde and Pakkanen, *Hybrid scheme for Brownian semistationary
  processes*: <https://arxiv.org/abs/1507.03004>.
