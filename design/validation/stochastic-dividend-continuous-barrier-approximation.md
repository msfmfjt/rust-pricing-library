# Continuous stochastic-dividend Barrier price approximation

The Rust and Python `StochasticDividendContinuousBarrierPlan` provides an
explicit, price-only rough residual-LSV approximation for a continuously
monitored Up/Down, Call/Put, knock-in/out Barrier. Compile with
`compile_rough_bergomi_lsv`, a Local Volatility target in funded residual-equity
coordinates and a price-only request. The existing `StochasticDividendPlan`
continues to reject live continuous monitoring.

The result uses `StochasticDividendPrice` and scheme
`buehler-rough-residual-lsv-continuous-physical-log-bridge-approx-v1`.
The scheme participates in the plan fingerprint. The plan exposes calibration
inputs and price evaluation, but no Greek methods. Generic risk flags, payoff
smoothing and smoothing-width ladders reject before calibration.

## Finite-grid definition

The simulated stock is reconstructed as `S = a*f + b*Y + c`, with post-cash
coefficients at each grid node. Let sigma be the causal residual-equity
volatility actually used for that step, nu the dividend-factor volatility, and
rho the correlation between their Brownian drivers. The two physical log-Spot
loadings at the left node are

- `e = a*f*sigma/S`;
- `d = b*Y*nu/S`.

Freeze `q = e² + d² + 2*rho*e*d` over the interval. Production evaluates the
equivalent PSD form `(e+rho*d)² + (1-rho)*(1+rho)*d²` to preserve nonnegativity
at singular correlation boundaries. This includes the stochastic reserve;
residual-equity variance alone is not the physical stock variance.

For two safe stock endpoints L and R and barrier B, conditional survival in the
local lognormal proxy is

`1 - exp(-2*log(B/L)*log(B/R)/(q*dt))`.

The shared stable bridge primitive evaluates this expression. Interval
survivals accumulate in log space, and the complementary knock-in weight uses
`-expm1(log_survival)`. Zero variance uses endpoint survival. Exact equality
is a hit. No bridge uniform or extra random coordinate is consumed.

Each interval runs from the left **post-cash** stock to the right **pre-cash**
stock. Both sides of the right cash jump are then tested, including a jump at
the monitoring end. The terminal intrinsic uses post-cash stock at expiry.
Later stock evolution does not change a monitoring window that has ended.

This construction is a local bridge approximation. The nonlinear two-factor
stock and rough Volterra history do not have this exact conditional crossing
law. In particular, the exact newest Volterra cell is correlated with the
within-step equity path; freezing sigma does not remove that approximation.
There is no claimed general weak convergence rate for this implementation.
[Gobet's Euler killed-diffusion study](https://doi.org/10.1016/S0304-4149(99)00109-X)
is background for bridge monitoring of diffusion approximations, not an
accuracy theorem for this rough, nonlinear split.

## Contract and uncertainty

- Continuous monitoring begins at valuation and ends at the final declared
  monitoring date. That endpoint is retained in the execution grid.
- Required past `historical_hit` summarizes the entire past interval. A past
  hit is absorbing. If monitoring ended strictly before valuation, history
  selects vanilla or fixed cash without observing current Spot.
- Monitoring ending on valuation observes current Spot when history is unhit.
- Rebate is fixed cash on the inactive branch, independent of notional.
  All payments use the contractual payment-date discount.
- The existing positive-stock, correlation, funded residual and calibration
  validations apply, including to resolved contracts.
- MC errors use independent antithetic units; RQMC errors use scramble means.
  They exclude finite-particle calibration uncertainty, grid bias, bridge
  approximation error and model uncertainty.

Calibrated leverage and the stochastic split retain their existing definitions.
The optional volatility trace records the same causal sigma during evolution;
it neither replays the quadratic rough history nor reads next-node variance.
Legacy pricing uses the same evolution and sampling order as before.

## Validation

The [Rust controls](../../crates/pricing/src/engine/risk/stochastic_dividends/continuous_barrier/tests.rs)
cover both stochastic loadings and correlation signs, singular variance limits,
causal volatility traces, KI+KO parity, pre/post-cash hits, monitoring end,
absorbing histories, initial equality, large-notional fixed rebates, API
rejections and independent MC/RQMC error aggregation. The flat, no-cash,
eta-zero GBM limit agrees with independent one-dimensional Gaussian terminal
quadrature for Up/Down Calls/Puts with notional 2 and rebate 7.

The [NumPy reference](../../tests/python/rough_dividend_continuous_reference.py)
independently evolves the f/Y split, reconstructs physical stock and applies the
bridge probability. It uses PCG64, a different Gaussian basis and retained
calibration inputs. It never calls Rust paths, payoffs, risks or RNGs.
The [fixture](../../fixtures/stochastic-dividends/rough-continuous-barrier-reference.json)
contains four H=0.1/0.3 Up-out Call and Down-out Put cases on eight steps,
with 32 batches of 8,192 antithetic pairs. The
[Python binding tests](../../tests/python/test_rough_dividend_continuous.py)
check production RQMC against these means with 16 scrambles of 8,192 points.
The gate is absolute price difference plus four combined standard errors below
0.10, with reference SE below 0.025 and production SE below 0.015.

A separate paired 16/32/64/128-step panel holds the eight-step leverage surface,
cash schedule and monitoring window fixed. Coarse/fine Brownian increments and
newest Volterra-cell integrals have the exact joint Gaussian covariance. Each
pair uses 16 independent batches of 2,048 antithetic pairs.

| Case | 64→128 price change | Paired SE | Absolute change + 4 SE |
|---|---:|---:|---:|
| H=0.1 Up-out Call | -0.0003174 | 0.0208755 | 0.0838193 |
| H=0.1 Down-out Put | -0.0035620 | 0.0117320 | 0.0504899 |
| H=0.3 Up-out Call | 0.0123758 | 0.0094831 | 0.0503083 |
| H=0.3 Down-out Put | -0.0080114 | 0.0034786 | 0.0219259 |

The last paired-change gate is 0.15. Endpoint-only prices are retained
separately and bound zero-rebate knockout bridge prices from above on each
sampled path. These checks measure finite-algorithm agreement and observed
grid changes. They do not bound distance to the true continuous-time price,
prove monotone convergence, or cover recalibration/particle error across grids.

The three-OS Barrier job runs the Rust controls. Linux regenerates all retained
NumPy batches and coupled refinements. The source archive and wheel contract
include the new API, tests and [example](../../examples/python/rough_dividend_continuous_barrier.py).

## Remaining work

Continuous Barrier sensitivities require differentiating the bridge estimator
and its discontinuous endpoint/jump branches; discrete graph adjoints are not
valid substitutes. Independent fine-path studies, calibration-aware refinement
and wider H/volatility/correlation/near-barrier panels remain necessary before
claiming broad continuous-time accuracy. This price API is an explicit
approximation, not an extension of the discrete hard-Delta guarantee.
