# Lifted Heston: fixed-grid factor-price validation

[Model reference](../../docs/models/rough-volatility-families.md) ·
[Earlier time-grid checks](rough-volatility-refinement.md)

## Status and scope

This validation-only increment isolates **finite-factor quadrature error at a
fixed time grid**. It adds no production pricing formula, model default, random
layout, public API, JSON tag, calibration or Greek. The six-model extension
remains experimental. The base is PR #120, commit `95437523ec235f69dcfd282a1b8511ebf2b19706`.

Unlike the earlier positive-lag exponential-kernel check, this experiment
compares option prices. Its reference is the **infinite-factor limit of the
existing semi-implicit, full-truncation discretization**, not the exact
continuous-time rough-Heston price and not the existing hybrid rough-Heston
scheme at the same time grid. Neither 256 factors nor 64 time steps is advertised
as a generally sufficient production configuration.

## Algebra of the independent reference

The model/factory uses the fractional Laplace measure of [1], with
`alpha=H+1/2` and `0<H<1/2`:

```text
mu(dx) = x^(-alpha) dx / [Gamma(alpha) Gamma(1-alpha)]
K(t) = integral exp(-x*t) mu(dx).
```

The following discrete reference is derived here by eliminating the factors of
the implemented recurrence. Write `v_j=max(raw_v_j,0)` and

```text
A_j = kappa(theta-v_j) dt_j + nu sqrt(v_j) dW_j
U_k(j+1) = (U_k(j)+A_j)/(1+x_k dt_j),  U_k(0)=0.
```

On an arbitrary finite increasing grid, substitution gives

```text
raw_v_i = v0 + sum_{j=0}^{i-1} G(i,j) A_j
G(i,j) = sum_k w_k product_{ell=j}^{i-1} (1+x_k dt_ell)^(-1).
```

The reference evaluates this scalar convolution without evolving any factors.
It retains the same positive-part operation in both the common increment and
the asset diffusion. This identity is valid pathwise, including negative raw
variance nodes; replacing the raw variance by zero *before* storing it is not
part of the reference.

On a **uniform** grid of spacing `dt`, replace the finite measure by `mu`:

```text
g_l = integral (1+x*dt)^(-l) mu(dx),  l>=1
    = dt^(alpha-1) Gamma(l+alpha-1) / [Gamma(alpha) Gamma(l)].
```

For the second equality use `y=x*dt/(1+x*dt)` and the beta integral on `(0,1)`.
The integrand has endpoint powers `-alpha` and `l+alpha-2`, both greater than
`-1`. The Rust reference uses the independent recurrence

```text
g_1 = dt^(alpha-1)
g_(l+1) = g_l (l+alpha-1)/l.
```

It calls neither the production kernel factory nor a production Gamma routine.
The Brownian boundary `H=1/2` is defined separately as the zero-rate unit-mass
factor and has `g_l=1`; the singular density formula is not evaluated at that
boundary. Its recurrence agrees with the same limit.

**Do not substitute `K(l*dt)` or a hybrid-cell weight for `g_l`.** These are
different finite-time discretizations. Increasing the factor count at fixed
`dt` does not by itself remove time-discretization or full-truncation bias.

## Independent kernels and path controls

`fixtures/rough-volatility/lifted-factor-prices.json` retains the complete fixed
protocol, 20 infinite-factor kernel values and 24 finite-factor kernels sampled
at five lags each. `scripts/check_lifted_heston_factors.py` checks these without
importing the Rust extension. It evaluates the beta integral with
endpoint-weighted QUADPACK and cross-checks a separate log-beta/log-gamma
expression. The finite weights/rates use direct geometric-bin endpoint formulas,
not the production `exp_m1` implementation.

The four fast Rust tests verify:

- The retained protocol, infinite-kernel recurrence and finite rational kernels.
- Scalar-convolution reconstruction of **384 full production paths** on the
  nonuniform grid `0, .003, .009, .05, .2, .65, 1`, for H=.1/.3 and 8/64/256
  factors. Every forward, variance, raw state and negative-node count is checked;
  a guard requires at least one negative raw variance node.
- The H=.5 boundary and splitting the unit zero-rate factor into three positive
  weights with the same total mass, sharing one variance Brownian motion.
- **24 public price/SE reconstructions**, varying 8/64 time steps, 8/256 factors,
  three strikes and antithetic on/off. Payoffs and sample errors come from the
  scalar-convolution reference, not the production path or payoff evaluator.

The latter controls share the production random-number coordinates and model
parameters. They do not independently validate the RNG distribution. Finite
factory values are separately checked against the Python kernel fixture.

## Price experiment fixed before execution

All inputs and acceptance budgets were retained before the first numerical
price run:

| Input | Specification |
| --- | --- |
| H | 0.1, 0.3 |
| Initial forward / strikes | 100 / 80, 100, 120 |
| Horizon | 0.25, 1 year |
| Uniform time steps | 64, fixed throughout each factor comparison |
| v0 / kappa / theta / nu / rho | 0.04 / 0.7 / 0.055 / 0.18 / -0.65 |
| Factor counts | 8, 16, 32, 64, 128, 256 |
| Geometric ratio | `exp(3/sqrt(factor_count))` |
| Seeds | 91, 1973 |
| Independent units | 8,192 antithetic pairs per H/horizon/seed |
| Asset / independent-variance normals | Same 128 coordinates for every factor count and the reference |
| Price acceptance level | 256 factors versus the infinite-measure discrete reference |
| Price budget | `abs(mean gap)+4*paired_SE <= 0.025` |
| Paired precision budget | `0 < paired_SE <= 0.002` |

The grid, parameters, common Brownian increments and payoff are held fixed.
The ratio shrinks while the truncated Laplace interval expands with factor
count; this is a test sequence, not a new factory default. The earlier time-grid
panel's fixed 20-factor/ratio-2.5 configuration is a different configuration.

One reference path and the six **public production** finite-factor paths are
computed for each sign of a normal vector. Each sign-pair is averaged before
estimating means or variances. All strikes use the same terminal values.
The paired SE is the two-pass sample standard deviation of the per-unit price
differences divided by `sqrt(8192)`; it is not the sum of standalone price SEs.

There are **144 price rows**. Only the **24 rows at 256 factors** are acceptance
conditions. Smaller counts are diagnostic. A small paired SE says nothing by
itself about precision of the absolute reference or model price; both standalone
SEs are printed alongside each paired comparison.

## Fixed-grid forward and truncation diagnostics

For every scenario and factor count, including the reference, a separate row
reports the sampled terminal forward and its antithetic-unit SE, with fixed
conditions `abs(mean-100)<=5*SE` and `0<SE<=0.15`. All **56 moment rows** are
checked. These are finite-grid sampling diagnostics, not continuous-time
martingale certificates or simultaneous confidence guarantees.

The asset step uses the left-node nonnegative variance. Its conditional
lognormal increment has mean one when conditioned on the available history.
The covariance of asset and variance increments is preserved, but future-node
variance must not be used in that asset increment. Raw-negative-node frequencies
are also printed; they are not filtered out or suppressed by an acceptance rule.
The denominator is `2*8192*64`, including all updated variance nodes on both signs.

## Executed numerical results

Linux x86_64, pinned Rust/Cargo 1.98.1. The release test explicitly includes the
ignored numerical panel. **The first price run passed, with the original
8,192 units, seeds, parameters, ratio sequence and budgets unchanged.**

| H | Largest absolute gap at 256 | Largest paired SE at 256 | Largest abs(gap)+4SE at 256 |
| --- | ---: | ---: | ---: |
| 0.1 | 0.000422468243 | 0.000092271422 | 0.000784668436 |
| 0.3 | 0.002534291923 | 0.000269000442 | 0.003574624797 |

The maxima in each column need not occur in the same scenario. Values are in
price currency units, **not implied-volatility basis points**.

For diagnostic context, the maximum `abs(gap)+4SE` across the 24 scenarios at each
factor count was:

| Factors | Maximum comparison value |
| --- | ---: |
| 8 | 0.217231115316 |
| 16 | 0.137339273851 |
| 32 | 0.078729531879 |
| 64 | 0.037749641297 |
| 128 | 0.014028992497 |
| 256 | 0.003574624797 |

This observed ordering is not a general monotonic-price or convergence-rate
theorem. There is no post-hoc acceptance threshold at the smaller counts.
The maximum sampled forward deviation was 1.272934 SE; the maximum forward SE
was 0.082764859255. The largest raw-negative-node fraction was 0.042064666748.
**Truncation is materially exercised, not proven harmless.**

An initial Python fixture-generation expression overflowed when directly raising
`1+x*dt` to a large lag. The reference generator was corrected to
`exp(-lag*log1p(x*dt))`. The kernel formula and numerical budgets were not changed;
no numerical price panel had run at that point. This was a generator arithmetic
issue, not an initial price-acceptance failure.

## Reproduction and CI

```shell
python scripts/check_lifted_heston_factors.py
python -m unittest discover -s scripts -p 'test_lifted_heston_factors.py'
cargo test --locked -p pricing --test lifted_heston_factor_prices
cargo test --locked --no-default-features -p pricing --test lifted_heston_factor_prices
cargo test --locked --release -p pricing --test lifted_heston_factor_prices -- --include-ignored --nocapture
```

The dedicated `lifted-heston-factors.yml` workflow runs the independent Python
references and debug/minimal/release Rust controls on Linux, macOS and Windows,
and retains `lifted-heston-factor-prices.log`. Its release command explicitly
runs the ignored price panel. Missing artifacts fail the evidence-upload step.

Four Python guard tests check the retained reference and protocol, reject ten
corruptions (including NaN, infinity, missing/duplicate scenarios and weakened
inputs), require all nine CI/evidence snippets through the actual source-archive
checker, and require all six new validation inputs in the source archive.

Defining these gates is not evidence of remote success. PR/CI status is recorded
separately from the executed local numerical results.

## Remaining boundaries

There is no continuous-time pricing oracle, Fourier/Riccati implementation,
full-truncation bias bound, Greek, calibration, VIX/SSR, stochastic-rate/dividend
extension or production-admission claim in this change. The nonuniform-grid
checks reconstruct finite-factor paths only; the infinite-factor price panel is
uniform-grid and MC only. RQMC factor refinement, path-dependent prices, different
parameter regimes, time/factor joint limits and wider H/horizon ranges remain
separate validation work. Local results do not substitute for remote CI/review.

## Reference

[1] Eduardo Abi Jaber, *Lifting the Heston model*, Quantitative Finance 19(12),
1995-2013 (2019), [arXiv:1810.04868](https://arxiv.org/abs/1810.04868).
The fixed-grid rational-kernel identity above is derived explicitly in this
protocol; it is not asserted to be a quoted formula from the paper.
