# Multi-marginal Bass local volatility

The experimental Rust `pricing::bass_lv` and Python `BassLvModel` APIs implement
the Brownian Bass-LV construction in Sections 1-3 and mapping sensitivities in
Sections 5.2-5.4 of Conze and Henry-Labordere,
*Bass Construction with Multi-Marginals: Lightspeed Computation in a New Local
Volatility Model*, May 25, 2021, [SSRN 3853085](https://ssrn.com/abstract=3853085).
The supplied 16-page PDF is the implementation reference.

## Usage

```python
import rust_pricing as rp

marginals = [rp.BassMarginal.lognormal(t, 100.0, 0.20)
             for t in [0.5, 1.0, 2.0]]
model = rp.BassLvModel.calibrate(100.0, marginals)
plan = model.compile_simulation([0.25, 0.75, 1.25, 1.75, 2.0])
price = plan.price_asian(100.0, paths=100_000, seed=42)
print(price.price, price.standard_error)
print(model.local_volatility(0.75, 100.0))
```

Run the [complete example](../../examples/python/bass_lv.py) after
`python -m maturin develop --locked` in the repository virtual environment.
Rust callers use `BassMarginal`, `BassLvConfig`, `BassLvModel::calibrate`, and
`compile_simulation`. `BassSimulationPlan::price` accepts a Rust closure over
requested observations. Python custom payoffs can use `sample_paths`.

For the shared product/market/risk API, use
`Model.bass_local_volatility(...)` in `PricingRequest`, then
`PricingPlan.compile(request).evaluate()`. Rust uses
`ModelSpec::BassLocalVolatility(BassLvSpec::new(...))`. The
[common-request example](../../examples/python/bass_lv_request.py) includes
an Asian with a known fixing, carry, cash/proportional dividends, delayed
payment, Delta/Gamma, parallel Vega and quote-node VegaKT.

## Inputs and financial coordinate

`BassMarginal(expiry, spots, probabilities)` defines a continuous CDF, linear
between knots and constant outside its finite support. At least three
nonnegative strictly increasing spots and strictly increasing probabilities
from exactly 0 to exactly 1 are required. It represents piecewise constant
densities, without atoms. Moments and calls are integrated analytically.

Expiries strictly increase. All means equal the initial spot within
`1e-9 * spot`. The convex-order validator examines call-price differences at
the union of both strike grids and every interior stationary point, detecting
arbitrage between knots as well. Variance increments no greater than
`1e-10 * spot^2` are rejected; identical/degenerate successive marginals are
not covered.

`BassMarginal.lognormal` creates a **tabulated approximation**: truncate the
normal coordinate at `+/- tail_std`, normalize its CDF, and scale price knots
to give the specified mean exactly. Defaults are 1601 knots and seven normal
standard deviations. This is a convenience and validation fixture, not a raw
option-quote fitter. Supply arbitrage-consistent market CDFs for smiles.

The standalone coordinate is a driftless positive martingale. Its price methods' discount
factor discounts the payoff only; it does not add carry or change forwards.
Standalone callers must transform both marginals and payoffs for carry and
dividends. The common-request integration performs physical-spot reconstruction
using the existing `Market` affine dividend coordinates, as described below.
Neither entry point automatically converts physical-spot IV quotes into
residual-equity IV quotes.

## Common request and physical-spot payoffs

The model specification takes maturity year fractions (ACT/365F from the
request's valuation date), log-forward-moneyness nodes, time-major absolute
IVs, a dense projection grid, Bass/projection configurations and `iv_bump`.
It calibrates a normalized residual-equity martingale `X`, with `X(0)=1` and
`E[X(t)]=1`. Existing discount and dividend curves give the canonical forward
`F_f(t)=S0*D_q(t)/D_r(t)`. In the library's affine-coordinate convention:

```text
S(t) = A(t)*S0 + B(t)*F_f(t)*X(t)
k    = log(K_f/F_f(t))
K_f  = (K - A(t)*S0)/B(t)
```

Input IVs refer to this residual coordinate. With fixed-cash dividends, raw
Black IVs quoted on physical spot strikes generally need conversion before
model construction. Reconstruction includes future cash reserves, proportional
dividends and both sides of a dividend jump where the payoff observes them.
Spot bumps keep cash amounts fixed and recompute the affine coordinate while
holding the normalized IV surface fixed. Only `sticky_log_moneyness` is
supported. Curves are held fixed for these sensitivities.

Shared European, weighted/seasoned Asian, fixed-strike lookback, digital and
discretely monitored barrier payoffs are supported. Known fixings and payment
dates follow the existing product contracts. Digital/barrier risks retain the
shared smoothing requirement; Compact-C2 smoothing and its width ladder use
the same surrogate for price and risks. Continuous barrier monitoring and
American early exercise are explicitly rejected. Observations must lie inside
the model horizon; a later payment date only changes discounting.

Both pseudo Monte Carlo and randomized Sobol sampling support antithetics and
Brownian bridge ordering. The bridge reorders Brownian increments, not barrier
crossings. An antithetic pair is one sampling unit; for RQMC, an entire scramble
is one unit. At least two independent units are required. Prices, risk errors
and covariances use these units, with deterministic reduction and same-platform
replay across worker counts. Requested work is capped at 100 million sample
matrix values per pseudo-MC run or per RQMC scramble.

Delta and Gamma use paired central spot differences, with the requested Gamma
step or the shared validation step when only Delta is requested. Scalar Vega
uses an all-quotes parallel IV bump. VegaKT uses complete quote-by-quote
recalibration, also with common random numbers. Requesting either IV risk
currently compiles all `2Q+2` scenarios, even for scalar Vega alone. Reported
method is `central_bump`, with VegaKT policy `recalibrated-central-crn-v1`.
These common-request risks do not invoke terminal-map AAD.

VegaKT reporting dates and log-moneyness axes must match the market quote axes.
All quotes are retained: `relative_density_threshold` is not applied to these
direct quote derivatives. There is no additional allocation/projection step;
the result's projection fields report the separately bumped parallel Vega,
the bucket sum, and their residual. Price/bucket covariance and optional full
bucket covariance are in raw sampling-unit units, following the shared result
contract. Finite-step and grid effects can leave a nonzero residual.

`PricingPlan` exposes `bass_calibration_diagnostics`,
`bass_projection_diagnostics` and `bass_vega_scenario_diagnostics` in Rust and
Python. Inspect these alongside Monte Carlo errors: sampling uncertainty does
not include grid, calibration, tail-truncation or finite-difference bias.
Common results include a `bass_finite_grid` warning. Detailed Bass calibration
diagnostics stay on the plan; they are not serialized in result JSON.

The current v3 request schema adds the `bass_local_volatility` model tag and
explicit numerical parameters. Requests/results round-trip through existing
JSON APIs and carry request/plan fingerprints. v1/v2 documents containing the
new tag are rejected; those schemas and existing golden bytes are unchanged.
Older v3 readers that predate this additive model tag cannot read Bass requests.

## eSSVI and implied-surface adapter

Python `BassMarginal.from_essvi(expiry, spot, slices, log_moneyness_nodes)`
returns a `BassMarginalProjection` with `.marginal` and `.diagnostics`.
It accepts existing `EssviSlice` objects and optional `terminal_theta_slope`.
Rust `BassMarginal::from_surface` accepts any `ImpliedVarianceSurface` and a
`BassSurfaceProjectionConfig`. The surface must be quoted in the same
martingale coordinate as Bass-LV, with `k = log(K / spot)`.

The adapter differentiates the undiscounted forward call:

```text
F(K) = 1 + dC/dK = Phi(-d2) + phi(d2) * w_k / (2 sqrt(w))
```

The skew term is essential. The existing surface evaluator also validates
positive total variance and density. Evaluate the upper CDF through its small
survival tail to avoid floating-point reversals near probability one.

The supplied log-strike grid must have 33..100001 strictly increasing finite
nodes spanning zero. Each omitted tail must be at most
`tail_probability_tolerance` (default `1e-7`). Normalize the retained CDF to
unit mass, discard repeated tail CDF values, and scale all price nodes to
restore the mean. Reject a relative scaling change greater than
`relative_mean_tolerance` (default `1e-4`). A decreasing CDF is not repaired.

Diagnostics report both tail probabilities, retained mass/node count,
unscaled mean, mean scale, and maximum call-price error at all supplied strike
nodes. Small tail probability alone does not control lost first moments;
inspect the mean correction and call-price errors too. Each projected
marginal is still a finite-grid approximation. Calibration separately checks
their convex order. Widen/refine the input grid if projection changes cause
that check to fail.

See the [eSSVI and mapping-risk example](../../examples/python/bass_lv_risk.py).

## Equations and numerical choices

For the first expiry, `g(w) = Q_1(Phi(w / sqrt(T_1)))`. Subsequent intervals
solve equations (1)-(3), with `dt = T_(i+1) - T_i`:

```text
F_next = CDF_i(K_dt * Q_(i+1)(K_dt * F))
g      = Q_(i+1)(K_dt * F)
f(t,w) = (K_(T_(i+1)-t) * g)(w)
```

`K` is the Gaussian heat kernel and `*` is convolution. The initial CDF is
Gaussian with variance `dt * Var(mu_i) / (Var(mu_(i+1))-Var(mu_i))`. The grid
spans `+/- grid_width * sqrt(initial_latent_variance + dt)`. We retain the
translation gauge selected by initialization, without median recentering.

CDFs and maps are linear interpolants on an odd uniform grid. Instead of
Gauss-Hermite quadrature, the heat operator uses exact Gaussian integrals of
linear basis functions with constant endpoint extensions. A Toeplitz stencil
is reused in iterations. Gaussian tails beyond ten standard deviations are
negligible at double precision. Exact integration refers to the represented
interpolants: finite support, grid interpolation and convergence tolerance
remain approximations of the continuous model.

The unshifted norm `||A F - F||_infinity` must meet `cdf_tolerance`.
Iteration exhaustion reports the interval, count and residual. Brownian CDF
boundary mass must meet `tail_tolerance`. Initial-map interpolation slightly
changes its mean: solve `f(0,W_0) = spot`, using Section 5's fixed-spot
convention. The uncorrected discrepancy is exposed as `initial_spot_error`.

## Simulation and pricing

Observations must be finite, nonnegative, strictly increasing, and within the
calibrated horizon. Plans insert zero and every intervening market expiry.
Each step advances `W` by `sqrt(dt) * Z`. At a market boundary, retain the
previous interval's terminal spot and invert the next interval's initial map,
as in equations (5)-(8). There is no Euler time stepping.

Date maps are precomputed. Pricing reuses scratch buffers without per-path
allocation. Sampling uses existing addressable Philox valuation coordinates.
Same seed, model and observation grid reproduce paths on the same platform;
changing the augmented grid changes random-coordinate assignments.
`time_nodes` and `normal_count` describe externally supplied normal arrays.
Correlated external arrays require a common augmented grid across assets.

Returned rows contain only requested observations. An observation at zero is
the initial spot and is an Asian fixing if requested. Inserted market dates
are not additional fixings. European pricing uses the final requested
observation; Asian pricing averages all requested observations equally. Both
accept calls/puts and a positive deterministic discount factor. Standard error
is the IID sample-mean error, excluding calibration/discretization errors.

`mapping(t,w)` directly evaluates the heat-convolved terminal interpolant.
`local_volatility(t,s)` solves for the latent coordinate and returns `f_w/f`.
Interior market dates use the next interval; the final expiry has no forward
local volatility. Simulation interpolates precomputed date maps, so its
off-grid values differ from direct mapping evaluation by interpolation error.

Leaving the Brownian grid or an unrepresentable boundary inverse raises an
error instead of clipping. Increase both width and point count to expand the
domain while retaining resolution.

## Diagnostics and accuracy

Intervals report dates, bounds/spacing, iteration count, `cdf_residual`,
`boundary_tail_probability`, and `marginal_cdf_error`. The last propagates
distributions through preceding resets using equation (12), comparing against
all supplied CDF knots and the 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%
quantiles. It is an estimate at these probes, not a certified global bound.

`cdf_tolerance` is not a total repricing tolerance. Sparse inputs with jumps
in density give kinks in the quantile map and can have first-order grid error.
Refine `grid_points` and input marginals separately, inspect marginal errors,
and compare price changes with sampling uncertainty. See the
[validation record](../../design/validation/bass-local-volatility.md).

## Scope and compatibility

This experimental API preserves existing model behavior and extends the current
v3 request schema with a new model tag. It supports single-asset calibration, local
volatility, paths, European/Asian and Rust custom path payoffs, implied-surface
projection, terminal-map sensitivities, recalibrated market-IV VegaKT and the
shared request/payoff/risk integration described above.

A vanilla-hedge portfolio solver and Section 6 Bass2 future-skew/stochastic-volatility
extensions are not implemented. Continuous
barrier crossings and early exercise are not inferred from coarse paths.
The PDF does not supply machine-readable SX5E market quotes; those market
prices and the paper's claimed speed ratio are not reproduced here.

## Terminal-map sensitivities (Section 5)

`BassMappingBump(interval, left, center, right)` defines a compact triangular
hat in the interval's latent Brownian coordinate. Intervals are zero-based;
`left < center < right` must lie within the calibrated grid. Hats are sampled
on that grid and must resolve to a nonzero function. Up to 256 hats can be
compiled with `model.compile_mapping_risk(observation_times, bumps)`.

The amplitude adds `b * h(w)` to the interval's terminal price map. Results are
**price per unit terminal-map amplitude**, not implied-volatility Vega and not
derivatives through eSSVI projection or fixed-point recalibration. The hat
definitions, Brownian grids, spot, times, RNG coordinates and other maps are
held fixed. The same finite heat operator and date interpolation as price
calculation are differentiated, including inverse maps at expiry boundaries.
Initial inversion contributes `dW0/db = -df(0,W0)/db / f_w(0,W0)`, keeping the
initial spot fixed.

`price_european` and `price_asian` on the risk plan return a `BassMappingRisk`
containing `.estimate`, `.sensitivities`, and their IID `.standard_errors`.
The price and its error exactly match `.simulation` with the same arguments.
The reverse sweep accumulates observation partials, carries the Brownian
adjoint backward, and differentiates every intervening reset. Vanilla payoff
ties use derivative zero; interpolation knots use the right segment. Generic
Rust `price` requires a callback filling observation derivatives. Python and
Rust `path_sensitivities` accept explicit normals and payoff partials for
external path functionals. Discontinuous payoffs need an appropriate estimator;
ordinary pathwise derivatives do not infer their jump contributions.

`bumped_simulation(amplitudes)` holds the same spot and grids, adds all mapping
perturbations, and returns a price plan for common-random-number checks. It
rejects nonfinite, negative or decreasing terminal maps. It does not recalibrate
to the original marginals and therefore returns only a simulation plan.

`vanilla_call(interval, strike, discount_factor=1)` computes a deterministic
call and its gradients. It transports the CDF and differentiated CDF through
the expiry maps as in equations (12)-(15), including the initial Gaussian
shift derivative. It integrates calls exactly against the resulting
piecewise-linear CDF and terminal map, including endpoint mass. This is a
semi-analytic **finite-grid** calculation, not an exact continuous-marginal
price. `bumped_vanilla_call` supplies an independent finite-difference check
of that deterministic price. Hats on later intervals have exactly zero
influence on earlier calls. Grid refinement remains necessary when comparing
this deterministic transport to Monte Carlo or market calls.

## Recalibrated market-IV VegaKT

`BassMarketIvModel::calibrate` accepts the library's `MarketIvSurface`, a dense
log-strike projection grid, and the Bass/projection configurations. Python
`BassMarketIvModel.calibrate` takes the surface's maturity axis, log-moneyness
axis and flattened IV quotes directly. Quotes use **absolute volatility**
(`0.20` means 20%) in **time-major** order: all strikes at the first maturity,
then all strikes at the second maturity. Both axes need at least two nodes.
The coordinate is `log(K/spot)` with a constant martingale forward; carry and
affine dividends must already have been transformed into this coordinate.

The existing `natural-cubic-w-linear-time-v1` surface interpolates total
variance in log strike and time. Every supplied maturity becomes a Bass
marginal. Projection nodes must lie within the quote strike range, resolve
the smile and cover both tails sufficiently. No strike extrapolation or
arbitrage repair is supplied. `.model` exposes the calibrated Bass model;
`.projection_diagnostics` reports the source-to-marginal errors.

`market.compile_vega_kt(observation_times, bump_size=1e-4)` builds a reusable
`BassVegaKtRiskPlan`. For each quote it re-evaluates the complete pipeline at
`sigma_j + h` and `sigma_j - h`: IV interpolation, CDF extraction, truncation,
mean normalization, marginal calibration, automatic Brownian-grid bounds,
initial inversion and observation maps. Spot, quote axes, projection nodes,
numerical settings, dates and deterministic discount factor stay fixed. It
also builds two all-quotes-at-once scenarios. Cost and storage grow with the
number of quotes: there are `2Q + 2` scenario calibrations. Compile once to
reuse those scenarios for different strikes/payoffs/seeds.

This estimator is named **`recalibrated-central-crn-v1`**. It is central finite
difference with common random numbers, not AAD through the fixed-point solver.
The independent Section 5 mapping AAD above remains available. Results identify
the method, bump size, quote axes and base IVs. European/Asian helpers and a
generic deterministic Rust path-payoff callback are supported.

For each path and quote, `D_j = discount * (payoff_up - payoff_down)/(2h)`.
`.sensitivities` returns the sample mean in currency per unit absolute IV;
`.standard_errors` is the IID error of these **paired differences**, including
their common-random-number covariance. `.vega_per_vol_point` and
`.standard_errors_per_vol_point` multiply these by `0.01`, consistent with the
library's VegaKT scale. The base `.estimate` matches price-only evaluation on
the same stream. `.parallel_sensitivity` and `.parallel_standard_error` come
from the separate simultaneous bump. `.bucket_sum` is the sum of node risks;
its error uses per-path sums and includes cross-bucket covariance. Finite bump
and discretization effects can make this sum differ from the parallel result.

`.diagnostics` contains every scenario's projection/calibration diagnostics,
shift, initial spot error, and flattened quote index (`None` for parallel).
Any failed scenario aborts the compilation with its index and signed shift;
there is no partial vector, bump reduction or one-sided fallback. Nonpositive
or unrepresentably small bumps and nonpositive down-bumped IVs are rejected.
`market.bumped(shifts)` performs independent full recalibration for manual
checks or other IV scenarios.

Sampling errors exclude finite-difference bias, calibration error and grid
error. Check a bump ladder and refine both grids; tightening the fixed-point
tolerance below a grid's residual floor can fail to converge. The validation
fixtures use `cdf_tolerance=1e-7`, IV bumps `2e-4/1e-4/5e-5`, and Brownian grids
401/801. No automatic choice of a universally accurate bump is claimed.
See the [runnable VegaKT example](../../examples/python/bass_lv_vega_kt.py).
