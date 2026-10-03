# Continuous stochastic-dividend Barrier price approximation

The Rust and Python `StochasticDividendContinuousBarrierPlan` provides an
explicit rough residual-LSV approximation for a continuously
monitored Up/Down, Call/Put, knock-in/out Barrier. Compile with
`compile_rough_bergomi_lsv`, a Local Volatility target in funded residual-equity
coordinates and a price-only request. The existing `StochasticDividendPlan`
continues to reject live continuous monitoring.

The result uses `StochasticDividendPrice` and scheme
`buehler-rough-residual-lsv-continuous-physical-log-bridge-approx-v1`.
The scheme participates in the plan fingerprint. The plan exposes calibration
inputs, price evaluation, finite-bump Spot Delta/Gamma and recalibrated parallel
residual Local-volatility risk. Generic risk flags, payoff
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

## Finite-bump Spot risk

`evaluate_spot_bump_risk(SpotBump)` in Rust and
`evaluate_spot_bump_risk(spot_absolute_bump=... | spot_relative_bump=...)` in
Python evaluate central price differences at half/base/double bump sizes.
Exactly one strictly positive, finite bump is required. The returned immutable
`StochasticDividendContinuousBarrierSpotRisk` exposes:

- `price`, the same base price and sampling error as `evaluate()`;
- `spot_bumps`, three absolute bumps in half/base/double order;
- `delta_estimates` and `delta_standard_errors`, paired at each bump;
- `delta` and `standard_error`, selecting the base bump;
- `bump_differences`, Delta(h/2)-Delta(h) and Delta(h)-Delta(2h), and their paired
  `bump_difference_standard_errors`;
- `payoff_evaluations`, seven times the base evaluated-path count;
- `risk_fingerprint`, including the base plan, method, requested absolute/relative
  convention and actual bump ladder.

The method tag is
`buehler-rough-residual-lsv-continuous-bridge-crn-spot-bump-v1`.
Each scenario re-anchors funded residual equity and the leverage surface's
initial residual level. Calibrated leverage **values** and log-moneyness nodes
stay fixed, as implied by the existing scale-invariant residual-LSV calibration.
Normalized f/Y states and causal step volatilities can therefore be shared.
Physical stock, physical log variance, survival weights, pre/post-cash hit
branches and terminal intrinsic are all recomputed for each scenario. Past hit
history remains fixed. Cash means, curves, model inputs, target and grid stay
fixed. There is no reverse tape or discrete-payoff adjoint in this method.

All six Spot scenarios must be representable and leave positive funded residual
equity, even for resolved contracts. Invalid ladders reject before sampling;
there is no clamp or one-sided fallback. Current Spot/barrier equality is allowed:
the result describes a finite price change across that boundary, not a guarantee
that a derivative exists there. No zero-bump limit or extrapolation is taken.

Sampling units pair both Spot scenarios and antithetic paths; RQMC then uses
independent scramble means. Adjacent bump gaps also use paired observations,
not independent-error propagation. Their errors exclude calibration uncertainty,
time-grid/bridge bias and finite-bump bias. Gaps are diagnostics, not bounds on
exact Delta error or a certificate that a selected bump is sufficiently small.
The uncertainty scope is `sampling_only_fixed_calibration_grid_bridge_and_bump`.

## Finite-bump Gamma

`evaluate_gamma_bump_risk(SpotBump)` in Rust and
`evaluate_gamma_bump_risk(spot_absolute_bump=... | spot_relative_bump=...)` in
Python use the same six shifted Spots and base price. For every shared path,
Gamma(h) is `(P(S+h)-2*P(S)+P(S-h))/h²`, for h/2, h and 2h. Production subtracts
the center price before adding the two changes and divides by h twice. This
preserves exact fixed-rebate cancellation and avoids forming h² explicitly.
There are seven payoff evaluations per state path, including the base price;
there are no additional random coordinates, calibration runs or reverse passes.

The immutable `StochasticDividendContinuousBarrierGammaRisk` result exposes
`price`, `spot`, `spot_bumps`, `gamma_estimates`, `gamma_standard_errors`,
`bump_differences` and `bump_difference_standard_errors`. The gaps are
Gamma(h/2)-Gamma(h) and Gamma(h)-Gamma(2h). `gamma` and `standard_error` select
the base bump. `delta` and `delta_standard_error` report the paired central
**price** difference at that same base bump, matching `evaluate_spot_bump_risk`.
They are not a pathwise/AAD Delta. Gamma and all its gaps are formed on paired
samples before MC reduction or RQMC scramble-mean aggregation.

The method is `buehler-rough-residual-lsv-continuous-bridge-crn-price-gamma-v1`.
Its separate `risk_fingerprint` includes the method, base plan, bump convention
and ladder. The existing price and Spot-Delta method/fingerprints stay unchanged.
Its `uncertainty_scope` is `sampling_only_fixed_calibration_grid_bridge_and_bump`.

The same funding, frozen-history and re-anchoring rules as Spot-bump Delta apply.
Initial equality and endpoint/cash-jump branch changes are included in the
finite price difference; an exact second derivative need not exist. Smaller
bumps may amplify sampling noise and floating-point cancellation. Gamma gaps
are diagnostics rather than a bound on derivative error. No zero-bump limit,
Richardson extrapolation, bump-size recommendation or precision guarantee is
implied. In particular, the MC/RQMC SE excludes finite-bump, time-grid, bridge
and calibration error. Generic Gamma request flags still reject.

## Recalibrated parallel Local-volatility risk

`evaluate_parallel_local_volatility_risk(local_volatility_bump)` in Rust and
`evaluate_parallel_local_volatility_risk(local_volatility_bump=...)` in Python
shift the square root of every **original** residual-equity target variance node:
`v -> (sqrt(v) + shift)²`. The three absolute-volatility bump sizes are h/2, h and
2h. For example h=0.01 is one volatility point; a 0.20 node becomes 0.19/0.21 at
the base bump. This coordinate is parallel residual Local volatility, **not** a
parallel shift of quoted market implied volatility or a VegaKT projection.

Each of the six targets is shifted before variance interpolation onto the
unchanged execution grid. Shifting sqrt(interpolated variance) would define a
different perturbation for a non-flat grid. Every original target node must
remain positive, representably shifted and within the original variance
floor/cap. The entire ladder is validated before calibration. There is no clamp,
one-sided fallback or implicit boundary repair, including for resolved payoffs.

Each scenario reruns the finite-particle rough-LSV calibration with the original
model, seed, particle count, bandwidth, minimum effective sample policy and
trace setting. It then re-evolves f/Y and causal volatility with common valuation
normals, recomputing bridge variance, survival, cash-jump branches and terminal
payoff. Spot, cash means, dates, curves, model parameters, correlations and past
hit state remain fixed. Unlike a Spot bump, the normalized state path cannot be
reused across these changed volatility surfaces. Reverse traces are not required.

`StochasticDividendContinuousBarrierLocalVolatilityRisk` returns the base `price`,
`local_volatility_bumps`, `vega_estimates`, `vega_standard_errors`, adjacent
`bump_differences` and `bump_difference_standard_errors`. `vega` and
`standard_error` select the base bump and are **per unit absolute volatility**.
`vega_per_vol_point` and `standard_error_per_vol_point` multiply those by 0.01;
they convert the reporting unit, rather than compute a one-way bump P&L.

`recalibration_count` is six. `scenario_evaluated_paths` and `payoff_evaluations`
are seven times `price.evaluated_paths`, including the base valuation path and
excluding calibration particle paths. The method is
`buehler-rough-residual-lsv-continuous-bridge-parallel-local-vol-recalibrated-crn-v1`.
The risk fingerprint includes the base plan, coordinate/method and bump ladder.
MC errors pair scenario/antithetic observations; RQMC errors use independent
scramble means. Bump gaps also use paired observations.

The uncertainty scope is `pricing_only_fixed_calibration_seed_grid_bridge_and_bump`.
It conditions on the common **calibration seed**, while the calibrated leverage
values change with the target. It excludes calibration sampling uncertainty,
finite-bump bias, grid/bridge error and model uncertainty. Calibration fallback
branch changes remain part of the finite algorithm. Gaps are diagnostics, not
bounds on exact Vega error. Generic Vega/VegaKT request flags still reject.

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

Spot-risk controls compare all eight contract styles against separately
recompiled/recalibrated shifted prices. MC and RQMC errors are independently
reconstructed with and without antithetic paths. They also cover worker replay,
absorbing fixed cash at notional 1e18, initial equality, ended-unhit history,
absolute/relative fingerprints and unfunded/unrepresentable bump rejection.
Both Delta and Gamma controls cover these scenarios, including paired errors
for each second price difference and adjacent Gamma gap. Python checks
immutable/detached results, concurrent calls and argument errors. An additional
no-cash, flat-variance, eta-zero control compares all eight continuous contract
variants against independent one-dimensional Gaussian quadrature. For Gamma
at absolute bumps 1/2/4, it uses eight RQMC scrambles of 8,192 points, an error
gate of five sampling SEs plus 0.0003, and SE below 0.004.

The same independent NumPy implementation supplies a separate Spot-bump panel
for all four reference cases, using 32 batches of 8,192 antithetic pairs, seed
20261005, and absolute bumps 0.5/1/2. It independently reconstructs and evolves
all six shifted markets. Production comparisons use 16 RQMC scrambles of 8,192
points. For each Delta, absolute difference plus four combined SEs must be below
0.03; for each adjacent gap the gate is 0.015. Both reference and production
Delta SEs must be below 0.004. These validate finite-bump risk of the finite-grid
bridge algorithm, not a continuous-time derivative.

An independent NumPy Gamma panel uses 32 batches of 8,192 antithetic pairs,
seed 20261006, and absolute bumps 1/2/4 for the same four stochastic-dividend
cases. Production uses 16 RQMC scrambles of 32,768 points. Every Gamma and
adjacent Gamma gap must have absolute difference plus four combined SEs below
0.02; reference and production Gamma SEs must be below 0.003. These are
finite-algorithm checks, not accuracy claims about zero-bump Gamma.

Parallel Local-volatility controls compare all eight contract styles against
separately compiled/recalibrated original-target bumps, including exact leverage
surface equality on the non-flat target. They reconstruct MC/RQMC paired errors,
check worker replay and absorbing fixed cash, and reject nonfinite, unrepresentable,
negative-volatility and floor/cap-crossing ladders. Four no-cash eta-zero GBM
contracts match independent terminal Gaussian quadrature at volatility bumps
0.005/0.01/0.02, with eight scrambles of 8,192 points, a 5 SE + 0.03 gate per unit
volatility and SE below 1.0.

A separate [Local-volatility fixture](../../fixtures/stochastic-dividends/rough-continuous-local-vol-reference.json)
retains all six recalibrated surfaces for each of four H=0.1/0.3 Up-out Call and
Down-out Put cases. Inputs come from separate full request compilation.
The NumPy reference independently re-evolves and values each scenario with
32 batches of 8,192 antithetic pairs, seed 20261007. It validates **valuation
conditional on those calibration inputs**, not an independent particle calibration
implementation. Python also recompiles each target to check the retained inputs.
Production comparisons use 16 scrambles of 32,768 points. Per unit volatility,
each Vega and adjacent gap must have absolute difference plus four combined SEs
below 2.0; reference and production Vega SEs must be below 0.35. The former gate
is 0.02 when expressed per vol point. None of these tests certifies market-IV
Vega or a zero-bump derivative.

The three-OS Barrier job runs the Rust controls. Linux regenerates all retained
NumPy batches and coupled refinements. The source archive and wheel contract
include the new API, tests and [example](../../examples/python/rough_dividend_continuous_barrier.py).

## Remaining work

Zero-bump continuous Barrier sensitivities require treating the bridge
estimator and its discontinuous endpoint/jump branches; discrete graph adjoints
are not valid substitutes. Quoted market-IV Vega/VegaKT, bucketed target risk
and other model/market risk remain unsupported by this continuous wrapper. Independent fine-path studies, calibration-aware refinement
and wider H/volatility/correlation/near-barrier panels remain necessary before
claiming broad continuous-time accuracy. This API is an explicit
approximation, not an extension of the discrete hard-Delta guarantee.
