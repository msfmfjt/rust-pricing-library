# Extended-model price accuracy

This panel adds independent price acceptance to the experimental
[Bergomi LSV](../roadmaps/lsv-roadmap-v0.1.md),
[Hull–White](../roadmaps/hull-white-roadmap-v0.1.md),
[rough Bergomi](../../docs/models/rough-bergomi.md) and
[local-correlation](../../docs/models/local-correlation.md) implementations.
The original base revision is `04219c8d58602de77e25c44ab672d96c40c41763`.
It supplements existing limiting-case, pathwise derivative and replay tests.
It also corrects a deterministic-carry omission exposed by the new panel in
the shared 1F/2F/rough LSV payoff adapter. Public APIs, wire formats, calibrated
path algorithms and experimental model status are unchanged. Nonzero-carry
prices and payoff adjoints now include `F_cont(0,t)/S0`; all three LSV plan
fingerprint domains advance to version 2 so old and corrected plans cannot alias.

## Carry regression found by the panel

On the base revision, the 2-year 1F skew case failed with maximum IV error
378.59 bp despite ensemble SE below 2.3 bp. The 1-year 2F refinement base also
failed (261.77 bp). The LSV process evolves a martingale starting at spot, but
the shared LV payoff adapter expects continuous carry to have been applied to
its input states. Passing the unscaled martingale omitted carry from prices.

The correction applies the continuous-forward/spot ratio before dividend
boundary validation and observation reconstruction, and applies the same chain
rule to both post-event and pre-event payoff adjoints. Fast tests check the
zero-vol-of-vol Black limit for 1F, 2F and rough models with no cash, midpoint
cash and expiry cash. They also check recalibrated variance adjoints against
common-random-number price bumps, including the price-only/recorded-path match.
Existing zero-carry path algorithms, calibration, seeds and statistical
budgets are preserved. This fixes the intended deterministic-market contract;
it does not introduce a new calibration convention or close S3/H3 acceptance.

## Run and evidence

```sh
cargo test --locked --release -p pricing --test extended_model_acceptance -- --ignored --nocapture --test-threads=1
```

The seven oracle/gate/carry-regression tests also run in the normal test suite. CI explicitly
runs the five expensive acceptance tests on Ubuntu, macOS and Windows. Each job
retains `extended-model-accuracy.log` for 14 days, including on failure.
`EXTENDED_ACCURACY` lines contain JSON with every requested strike, target IV,
raw price, conditional pricing SE, calibration/pricing seed, numerical
resolution, error budget and failure. `EXTENDED_REFINEMENT` records changes
under independent refinement samples. `EXTENDED_LOCAL_CORRELATION` records
global projection/fallback counts; assertions additionally require supported
terminal interpolation nodes at the tested strikes.

The executable specification is
[extended_model_acceptance.rs](../../crates/pricing/tests/extended_model_acceptance.rs),
with [single-asset](../../crates/pricing/tests/cases/extended_single_asset.rs) and
[multi-asset](../../crates/pricing/tests/cases/extended_multi_asset.rs) panels.
Every price passes through the public pricing-plan API and its production random
dimension ordering. No test-only path simulator or alternate Sobol mapping is
used to obtain a passing price.

## Independent targets and cases

All panels use spot 100, continuously compounded discount/dividend rates 3%/1%,
ACT/365F and valuation date 2026-01-01. They price unsmoothed OTM puts/calls at
log-forward strikes -0.12, 0 and 0.12. The oracle directly evaluates Black's
formula and inverts each observed price by bracketed bisection. Invalid prices,
nonfinite metrics, missing quotes and insufficient vega fail; no quote is dropped
or clipped into the IV domain.

The LSV marginal targets are synthetic market-IV surfaces. Flat IV is 20%; skew
IV is `sqrt((0.04 + 0.002*T)*(1 - 0.2*x + 0.05*x*x))`. Quote maturities are
2026-07-02, 2027-01-01 and 2028-01-01; log-strike nodes span -1.2 to 1.2.
The public quote-backed variance/density compiler creates the calibration
targets and rejects floor/cap repairs. Expected option prices come directly
from the prescribed marginal IV, not a second simulation of the same model.
A fast coordinate check bounds the difference between the independent
polynomial IV and the natural-cubic market interpolation at the tested strikes
by 0.1 IV bp; this difference remains inside the total error budget.

| Panel | Cases |
| --- | --- |
| Deterministic LSV | 1F long skew; 2F short flat and long skew; rough short flat and long skew |
| Stochastic-rate LSV | 1F, 2F and rough with 2-year skew targets and nonzero rate/volatility correlations |
| Gaussian BS/HW | 2-year options, mean reversion 0 and 0.2, equity/rate correlation -0.4, 0 and 0.4; piecewise rate volatility |
| Joint local correlation | BS assets, 2F-LSV/BS assets, rough-LSV/HW/BS assets; both basket and first-constituent options |
| Refinement | 1-year skew 2F and rough LSV; particle count, time steps and bandwidth changed one at a time |

Single-asset 1F parameters are `(k,nu,rho)=(2,0.7,-0.5)`; 2F parameters are
`k=(3,0.3), nu=0.5, theta=0.35, rho_S=(-0.55,-0.2), rho_12=0.25`;
rough parameters are `(H,eta,rho)=(0.12,0.6,-0.5)`. HW mean reversion is 0.2,
rate volatilities are 0.012/0.015 with a knot at 40% of maturity, equity/rate
correlation is 0.25 and volatility/rate correlations are -0.1 (and 0.02 for 2F).

Long deterministic 2F/rough cases include paid cash 2 plus a 3% proportional
dividend at the midpoint. Stochastic-rate cases place that event at expiry.
The oracle applies the exact affine strike/forward transformation. HW targets
refer to the continuous normalized equity coordinate; this does not certify
conversion from physical-equity market IVs with intermediate stochastic cash
offsets. Those are different contracts.

For BS/HW the independent oracle integrates covariance kernels by Simpson
quadrature, split at rate-volatility knots:

```text
V = sigma_S^2*T + integral [sigma_r(u)^2*B(a,T-u)^2
                          + 2*rho*sigma_S*sigma_r(u)*B(a,T-u)] du
B(a,s) = (1-exp(-a*s))/a; B(0,s) = s
```

The expected Black IV is `sqrt(V/T)`. A separate polynomial identity checks the
zero-mean-reversion quadrature. The reference uses no production HW covariance
or transition routine. The public exact Gaussian engine uses eight time steps
and inserts the rate-volatility knot.

The multi-asset panel has two equal spots, equal carry and basket weights 1/2.
Constituent IV is 28%, basket IV 23.5%, and correlation endpoints are -0.3/0.95.
Thus the physical basket is exactly its normalized calibration coordinate
times the deterministic forward, making its independent Black target valid.
Joint 2F/rough parameters and HW correlations are fixed in the case source.
Pricing the first constituent through the same multivariate calibration checks
that fitting the basket also preserves its marginal. Remote/early fallback
nodes are reported, not silently treated as supported calibration observations.

## Error gates

One IV basis point means 0.0001 absolute volatility (20% to 20.01%). Every panel
must satisfy all of the following fixed budgets:

| Metric, in IV bp | LSV / local correlation | Gaussian BS/HW |
| --- | ---: | ---: |
| Maximum absolute ensemble IV error | 15 | 2 |
| RMSE across three strikes | 8 | 1 |
| Worst individual seed IV error | 30 | 4 |
| Ensemble SE | 4 | 1 |
| Conditional pricing SE of ensemble mean | 3 | 1 |

The base uses 16,384 calibration particles, 128 steps, kernel bandwidth 0.05,
4,096 RQMC points per scramble, eight scrambles, antithetics and Brownian bridge.
BS/HW uses 16,384 points per scramble. Execution uses two workers and block
size 256. Calibration seeds are 1709, 2903 and 4001; each pricing seed is its
calibration seed XOR `0xd1b54a32d192ed03`.

Each outer run independently calibrates and prices. Between-run SE is the
sample standard deviation of the three prices divided by sqrt(3). Conditional
SE of their mean is `sqrt(sum(pricing_SE^2))/3`. The reported ensemble SE is the
larger of these two, converted by target Black vega: adding them would count
pricing noise twice. This three-seed panel is a regression check, not evidence
of nominal confidence-interval coverage. Fixed price-error budgets are never
enlarged by the observed SE; excess noise fails its own gate.

Refinement doubles particles to 32,768, doubles time steps to 256, or narrows
bandwidth to 0.035. Each axis uses a different seed offset (10,000/20,000/30,000).
All absolute gates still apply. Each strike's IV change must be at most
`5 bp + 3*sqrt(SE_base^2 + SE_refined^2)`. Independent samples justify that
comparison; monotonic sampled-error reduction is not assumed. This establishes
consistency under these refinements, not a convergence rate or an extrapolated
zero-step/zero-bandwidth result.

## Linux measurement

The initial panel ran on Linux with pinned Rust 1.98.1 in release mode. The
carry-sensitive deterministic and refinement panels were rerun after the fix;
the Gaussian/HW/multi-asset paths do not call the corrected deterministic LSV
payoff adapter. All 28 panels and 84 quote points passed their fixed budgets,
using 252 price evaluations: three independent seed pairs per panel, shared
across strikes. Maximums below are over
each group; RMSE is the largest three-strike panel RMSE, not a pooled average.

| Group | Panels | Max IV error, bp | Max panel RMSE, bp | Worst seed, bp | Max ensemble SE, bp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deterministic LSV | 5 | 5.0423 | 3.4384 | 7.2464 | 2.4345 |
| Stochastic-rate LSV | 3 | 3.1695 | 2.1670 | 5.9626 | 2.2734 |
| Gaussian BS/HW | 6 | 0.0365 | 0.0232 | 0.0703 | 0.0526 |
| Local correlation / constituent | 6 | 6.1861 | 5.5323 | 11.0449 | 3.5263 |
| 2F / rough refinement | 8 | 4.0323 | 2.8215 | 7.5604 | 2.6793 |

The largest IV change under refinement was 5.916 bp; the largest ratio of
observed change to its allowed bound was 0.455. The original failing 2-year
1F case improved from 378.59 bp to 3.20 bp with the carry correction. No
sampling, error-budget or production RNG-layout change was needed to pass it.

Local compilation used `CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=1` and
`RUSTFLAGS='-C lto=off -C codegen-units=1 -C llvm-args=-threads=1'` to avoid
empty-object/linker failures in the execution environment. These settings are
not committed as build defaults. CI runs the repository's standard pinned
toolchain and release configuration on all three OSes and retains the actual
commit/toolchain alongside the numerical log. Linux measurements alone do not
assert native cross-platform replay acceptance.

## Remaining scope

These are representative price gates. They do not certify extreme smiles,
tail strikes, very long maturities, large vol-of-vol, pure uncalibrated Bergomi
price surfaces, path-dependent products, Greeks, physical-IV conversion with
stochastic cash offsets, all local-correlation mixtures, or nominal confidence
coverage. Refinement is currently for deterministic 2F/rough LSV; the HW and
multi-asset cases have multiple seeds and independent price targets but need
their own broader refinement studies before a production acceptance claim.
