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
residual Local-volatility risk, reporting projection with joint covariance, and
parallel/selected retained-quote IV risk. Generic risk flags, payoff
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

## Recalibrated bucketed Local-volatility risk

`evaluate_bucketed_local_volatility_risk(local_volatility_bump, node_indices)`
in Rust and the same keyword-only arguments in Python shift selected original
residual Local-volatility nodes **one at a time**. An index is zero-based and
row-major: `time_index * log_moneyness_nodes.len() + x_index`. The nonempty list
must contain unique, in-range indices. Output rows retain this selection order;
there is no sorting or implicit choice of all nodes. For a 3 by 3 original grid,
`node_indices=[4, 0]` selects the middle node followed by the first node. Use
`list(range(9))` explicitly for all nine nodes. Time-zero nodes are included
when selected; this API does not drop or project them into reporting-IV buckets.

Each selected node receives the same h/2, h and 2h absolute-volatility ladder:
`v_i -> (sqrt(v_i) +/- h)^2`, with all other original values unchanged. The full
set of shifted targets validates before calibration, including positivity,
representability and the original variance floor/cap for shifted values.
Unselected values at a bound need no perturbation. Targets are then interpolated
in variance and recalibrated exactly as in the parallel API. Model, calibration
seed/configuration, valuation shocks, Spot, curves, cash, dates and history stay
fixed. No reverse trace or discrete-payoff adjoint is used.

The immutable `StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk`
returns `price`, `node_indices`, original `time_nodes` and `log_moneyness_nodes`,
and `local_volatility_bumps`. `vega_estimates` and `vega_standard_errors` have
one row per selected node and three half/base/double columns. The corresponding
`bump_differences` and `bump_difference_standard_errors` have two columns:
Vega(h/2)-Vega(h) and Vega(h)-Vega(2h). All estimates are per unit absolute
residual Local volatility; multiply values and errors by 0.01 for a vol point.
These are finite node-price differences, not market-IV Vega or VegaKT.

`sum_vega_estimates`, `sum_vega_standard_errors`, `sum_bump_differences` and
`sum_bump_difference_standard_errors` report the selected-node sum, formed on
paired observations before reduction. Sum errors include cross-node covariance;
adding marginal variances would give a different uncertainty estimate. At finite
bumps, the sum of individually shifted-node risks need not equal a simultaneous
parallel shift, even when all nodes are selected. No forced reconciliation,
zero-bump extrapolation or derivative-error bound is applied.

For N selected nodes, `recalibration_count` is 6N. Both `scenario_evaluated_paths`
and `payoff_evaluations` are `(6N+1)*price.evaluated_paths`, including the base
valuation but excluding calibration particle paths. Memory retains these scenario
surfaces, and execution cost grows with the number of selected nodes. MC uses
paired independent sampling units and RQMC uses independent scramble means.
The uncertainty scope is `pricing_only_fixed_calibration_seed_grid_bridge_and_bump`;
calibration uncertainty, model error and grid/bridge/bump bias are excluded.
The method is
`buehler-rough-residual-lsv-continuous-bridge-bucketed-local-vol-recalibrated-crn-v1`.
The fingerprint covers the base plan, method, ordered node selection and ladder.

## Reporting-IV projection of finite node risks

`evaluate_reporting_iv_projection(local_volatility_bump, relative_density_threshold)`
in Rust, with the same keyword-only arguments in Python, connects the finite
node risks to the existing residual-LSV **reporting map**. The model must retain
an explicit `reporting_iv_basis`; the request remains price-only. There is no
implicit basis or density threshold. The threshold must lie in (0,1], and the
basis must cover every positive original target maturity. Invalid maps reject
before scenario calibration or valuation. Different reporting time and
log-moneyness grids are supported.

This API deliberately has a separate result and policy:
`StochasticDividendContinuousBarrierReportingIvRisk` and
`local_vega_density_reporting_iv_projection_v1`. It is **not** a quoted-market-IV
bump, a derivative through surface/Dupire calibration, or the complete
Gamma-transition/equation-(3)-(11) VegaKT operator. It reproduces the
`local_vega_density_from_node_adjoints` followed by
`project_local_vega_nodes_to_reporting_iv` convention used by the existing
residual-LSV reporting route. No first-order truncation rate for this continuous
Barrier projection is claimed.

For each original target node, all six finite Local-volatility scenarios are
recalibrated and valued as above. Let `A[t,i,h]` denote that paired node-price
difference. The deterministic reporting map is:

1. Convert positive-time rows to density weights `A[t,i,h]/m_i`, using original
   log-moneyness hat areas: half the adjacent interval at an edge, half the span
   between adjacent neighbors in the interior.
2. Compute call-density rows from the retained reporting total-variance surface,
   at the **original target** times and log-moneyness nodes, with normalized
   residual forward 1. Density ratios and the excluded-mass diagnostic are
   invariant to this forward normalization. There is no physical-Spot smile
   remapping. Use the contiguous qualifying domain containing the nearest
   log-moneyness-to-zero node; the lower index breaks ties.
3. Project active density weights through bilinear reporting-IV basis weights.
   Maturities must be covered; spatial tails go to the nearest reporting edge.
   Time-zero rows are not projected.
4. Report the original node sum as `pre_projection_estimates`, the bucket sum as
   `projected_sum_estimates`, and their difference as `residual_estimates`.

The reporting basis, implied volatilities and active domains stay fixed across
all bumps. In particular, the residual includes the change from **node risk to
density weights**, as well as time-zero and excluded-node contributions. It need
not vanish even when every positive-time node is active, and it is not an error
bound or solely omitted-tail risk. An arbitrary reporting basis need not have
produced the original target; this API neither checks that relationship nor
rebuilds the target from the basis. The exact map above defines the output.

Bucket rows follow `[reporting maturity][reporting log-moneyness]`; the three
columns are half/base/double **Local-volatility** bumps. `bucket_estimates` and
`bucket_standard_errors` expose each result. `bump_differences` and
`bump_difference_standard_errors` have the two adjacent ladder gaps. Multiply
values and errors by 0.01 for the existing volatility-point reporting convention;
this scaling does not turn them into an actual quoted-IV bump P&L.

Each original node observation is transformed before MC reduction or RQMC
scramble aggregation. Errors for buckets, their sum, the pre-projection sum and
the residual therefore include cross-node covariance. Marginal node standard
errors cannot be projected independently.

Optional joint estimator covariance is available through Rust
`evaluate_reporting_iv_projection_with_covariance(h, threshold)` or Python
`evaluate_reporting_iv_projection(..., full_covariance=True)`. The default returns
`estimator_covariance=None`; the opt-in returns a symmetric `(5B+10)` square matrix
for B reporting buckets. `covariance_labels` identifies its exact row/column order:
price; then each bucket's three half/base/double risks and two adjacent gaps;
then the pre-projection, projected-sum and residual triples. Labels use the
corresponding result field and zero-based indices, e.g. `bucket_estimates[1][1]`.
The label list is also available without requesting the matrix.

The matrix is **covariance of the estimated means**, not raw sample covariance:
`C[i,j] = sum_u ((X[u,i]-mean[i])*(X[u,j]-mean[j])) / (U*(U-1))`.
MC observations are individual paths, or paired antithetic averages when enabled;
RQMC observations are whole scramble means, never individual Sobol points.
Diagonal entries equal squared reported standard errors up to floating-point
rounding. For fixed weights w on these coordinates, the combined estimate has
sampling variance `w^T C w`. Price/risk, cross-bucket, cross-bump and aggregate
covariances are included. To use volatility-point coordinates, multiply risk
rows/columns by 0.01 each; leave the price coordinate unscaled. Gaps and sums make
this matrix singular by construction, so consumers must not assume it is invertible.

MC accumulates centered cross-moments per fixed reduction block and merges them
in a deterministic tree, without retaining path observations. RQMC uses a centered
two-pass product of its retained scramble means. The opt-in adds no calibration
or payoff evaluations and leaves all existing estimates/errors unchanged. It adds
quadratic storage/work in report width: MC retains a triangular matrix per logical
reduction block; RQMC computes the matrix only after within-scramble reductions.
Worker-count and concurrent-call replay are covered. The matrix carries the same
conditional uncertainty scope as the existing errors; it does not include
calibration uncertainty or grid/bridge/bump/map bias.
`positive_target_time_nodes`, `target_log_moneyness_nodes`, active-domain start/end
indices, excluded probability masses and the threshold identify the density
filter. Excluded masses use the shared finite-grid hat quadrature, not exact
tail integrals. The reporting grid and retained IV values are returned separately.

All original nodes, including time zero, are evaluated. For N nodes the method
performs 6N recalibrations and `(6N+1)*price.evaluated_paths` scenario paths/payoffs,
excluding calibration particles. No previously returned risk object is reused:
the shared observations are needed for valid transformed errors. The base price
and existing risk APIs retain their numerical ordering and fingerprints.

The method tag is
`buehler-rough-residual-lsv-continuous-bridge-reporting-iv-projection-crn-v1`.
Its fingerprint includes the base plan (including the retained basis), map
policy, Local-volatility ladder and density threshold. Requesting the covariance
matrix additionally hashes `joint_estimator_covariance_v1`; default fingerprints
remain unchanged. The uncertainty scope is
`pricing_only_fixed_calibration_seed_grid_bridge_bump_and_projection`.
Calibration sampling uncertainty, model error, and grid/bridge/bump/reporting-map
bias are excluded. Generic Vega/VegaKT request flags still reject.

## Recalibrated parallel quote-IV risk

The explicit quote source supports finite parallel **residual-forward IV** risk.
Rust constructs `MarketIvSurface`, materializes its strict `local_variance_grid`,
then binds it once to a compiled continuous plan with `with_market_iv_surface`.
Python constructs `MarketIvSurface(maturity_nodes, log_moneyness_nodes, implied_volatilities)`,
uses `source.local_volatility_model(...)` in the request, and passes
`market_iv_surface=source` to `compile_rough_bergomi_lsv`. The plan exposes
`supports_market_iv_risk`. The original price factory without a source remains
unchanged and the quote-risk method rejects. A reporting-IV basis is neither
required nor treated as a quote source.

The source must rebuild every original target variance **exactly**, with the
same target axes and floor/cap and without any repair. A mismatched source or
re-binding an already bound Rust plan rejects. Binding changes the plan fingerprint
but leaves the base price, sampling error and calibrated surface unchanged.
The fingerprint includes `continuous-residual-market-iv-original-grid-dupire-v1`,
the interpolation label, quote-axis lengths/values and all input IVs. Requests
still serialize the explicit grid; the caller must retain and re-supply the
separate quote source when compiling a restored request.

This reuses the existing `natural-cubic-w-linear-time-v1` contract: form
`w_ij=T_i*sigma_ij^2`, interpolate w by a natural cubic spline in log moneyness,
and linearly in time. Use the right slope at interior quote times; at the final
quote time and outside the quote-time domain use constant-IV tails. No strike
extrapolation is permitted. Quote axes and original target axes may differ,
including quote maturities beyond expiry. The entire original target log grid
must be covered. Time-zero target variance uses the first positive original
sample time. The constructor checks positive finite IV, calendar slope and
Durrleman density at quote samples; strict target construction also checks the
original target samples and rejects floor/cap repair. These are sampled checks,
not a globally arbitrage-free fit. There is no SSVI/eSSVI optimizer or raw
physical-stock quote conversion in this API.

`evaluate_parallel_market_iv_risk(implied_volatility_bump=h)` evaluates:

1. Shift every retained IV input by each of `-h/2,+h/2,-h,+h,-2h,+2h`, at fixed
   quote coordinates, Spot, curves, cash dividends and model parameters.
2. Reconstruct each quote surface and recompute Dupire variance `w_T/g` on the
   **original** target axes. Validate all six surfaces/targets before scenario
   calibration, rejecting nonpositive or unrepresentable shifts, sampled
   calendar/density violations and any floor/cap repair.
3. Interpolate each original variance grid onto the unchanged execution grid,
   then fully recalibrate leverage with the same particle seed/configuration.
4. Evolve six separate f/Y/volatility paths with common valuation normals and
   recompute continuous bridge, endpoint and cash-jump payoffs. The base price
   and three central price differences plus two adjacent gaps share observations.

The result is `StochasticDividendContinuousBarrierMarketIvRisk`, with three
Vega estimates/SEs in half/base/double order, two paired gap estimates/SEs,
quote coordinates/values, base price, method and fingerprint. Values are currency
per unit absolute IV; `vega_per_vol_point` and its SE multiply the base-bump
quantity by 0.01. MC uses independent paths or antithetic pairs; RQMC uses
scramble means. The work count is six recalibrations and seven scenario paths
per base path, excluding calibration particles. On a nonflat smile the result
differs from shifting sqrt(Local variance); the reporting projection is also a
separate convention. Selected quote bumps use the bucket API below; neither
finite-bump API supplies the full VegaKT operator.

The method is
`buehler-rough-residual-lsv-continuous-bridge-parallel-market-iv-recalibrated-crn-v1`.
Its risk fingerprint combines the source-bound plan and bump ladder.
Uncertainty scope is
`pricing_only_fixed_calibration_seed_grid_bridge_and_quote_interpolation`.
Calibration noise and interpolation/grid/bridge/bump bias remain excluded.
For stochastic cash dividends, these inputs describe the funded residual
forward f. Unconverted physical-Spot Black IV is not an f smile; this API does
not supply or differentiate that model-dependent conversion. See the
[runnable quote-IV example](../../examples/python/rough_dividend_market_iv.py).

Controls independently derive natural-cubic/Dupire values for a nonflat smile,
recompile each shifted target, and compare all eight Barrier contract styles.
Paired errors are rebuilt from individual MC units and RQMC scramble means,
with and without antithetics. Flat-IV, eta-zero, no-cash limits are checked
against one-dimensional Gaussian barrier quadrature at all three bump sizes
for four knockout styles. Python uses a separate NumPy natural-spline solve
and Dupire formula, plus source/target mismatch, coverage, calendar and repair
rejection, fixed-cash history, worker/concurrent replay and detached arrays.

## Recalibrated selected quote-IV buckets

Rust `evaluate_bucketed_market_iv_risk(h, &[4, 0])` and Python
`evaluate_bucketed_market_iv_risk(implied_volatility_bump=h, quote_indices=[4, 0])`
move each selected retained quote separately. `quote_indices` must be nonempty,
unique and in range; each is `maturity_index * quote_log_moneyness_nodes.len() +
strike_index`. Results retain the requested order. Indices refer to the retained
quote grid, which may differ from both original and refined variance grids.
A quote maturity beyond expiry can still contribute through time interpolation
and the Dupire time derivative; it is not silently dropped.

For each quote, the method rebuilds the source at six shifts `±h/2, ±h, ±2h`,
recomputes Dupire on the original target axes, refines variance onto the same
execution grid and fully recalibrates leverage with the fixed calibration seed.
Every shifted quote surface and original/refined target validates before any
scenario calibration begins. Nonpositive or unrepresentable quote changes,
sampled arbitrage and floor/cap repairs reject the entire request. The source
interpolation, strict provenance and coordinate contract above remain in force.

`StochasticDividendContinuousBarrierBucketedMarketIvRisk` returns `price`,
`quote_indices`, `quote_maturity_nodes`, `quote_log_moneyness_nodes`, original
`implied_volatilities`, `interpolation` and `implied_volatility_bumps`. Rows of
`vega_estimates` and `vega_standard_errors` correspond to the selected quotes;
columns are the half/base/double ladder. `bump_differences` and
`bump_difference_standard_errors` contain the paired half-minus-base and
base-minus-double gaps. `sum_vega_estimates`, `sum_vega_standard_errors`,
`sum_bump_differences` and `sum_bump_difference_standard_errors` aggregate only
selected quotes. Estimates are currency per unit absolute residual-forward IV;
multiply both estimates and errors by 0.01 for per-vol-point reporting.

Sums and gaps are computed on shared observations before MC/RQMC reduction,
so sum errors include cross-quote covariance. MC units are paths or antithetic
pairs; RQMC units are scramble means. Do not add marginal standard errors.
The finite sum of individually bumped quotes need not equal the simultaneous
parallel bump, even when all quotes are selected. Small-bump convergence is a
diagnostic, not a guaranteed zero-bump derivative at nonsmooth branches.
For N selected quotes there are 6N full recalibrations and
`(6N+1)*price.evaluated_paths` scenario paths/payoffs, excluding calibration
particles. No full covariance matrix or reporting-basis projection is produced.

The method tag is
`buehler-rough-residual-lsv-continuous-bridge-bucketed-market-iv-recalibrated-crn-v1`.
Its fingerprint hashes the source-bound plan, ordered quote indices and bump
ladder. The uncertainty scope is
`pricing_only_fixed_calibration_seed_grid_bridge_and_quote_interpolation`.
Errors condition on the calibration and exclude calibration noise, interpolation,
grid, bridge and finite-bump bias. The API does not provide physical-Spot quote
conversion, a constrained smile refit or the full VegaKT operator.

Controls compare all eight Barrier styles with separately compiled scenarios
using an independent closed natural-cubic/Dupire calculation. Independent path
or scramble observations reconstruct each bucket, sum and gap error with and
without antithetics. They cover beyond-expiry quote influence, selection order,
single-bucket identities, worker replay, invalid selections/bumps, strict caps,
missing provenance, fixed-cash histories and small-bump sum/parallel convergence.
Python checks four stochastic reference cases with a separate NumPy cubic solve,
plus detached/frozen arrays, concurrent replay and argument errors. Calibration
and pricing remain the shared Rust implementation in these scenario comparisons;
these tests do not establish independent calibration or continuous-time accuracy.

## Validation

Joint-covariance controls reconstruct every matrix entry independently from
separately recompiled scenario prices and the hand-specified reporting map,
including base-price cross-covariances. MC/RQMC with and without antithetics are
covered; mixed price/risk weights reproduce directly sampled portfolio variance.
An offset control checks centered accumulation at 1e12, deterministic block/worker
replay and evaluation-error propagation. Python checks matrix coordinates,
diagonals, symmetry, positive semidefiniteness within rounding, reconstruction of
sum/gap/residual errors, detached nested arrays, concurrent replay and zero
covariance for resolved fixed-cash history.

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

Bucketed controls compare all eight contract styles against separately compiled
single-node target shifts at time zero, an interior time and the terminal row,
including exact recalibrated leverage equality. MC/RQMC controls independently
reconstruct node, sum and gap errors with and without antithetic paths. Selection
order, worker replay, fixed cash, original-versus-refined grid metadata,
nonfinite/unrepresentable bumps, selected floor/cap boundaries and unselected
boundary nodes are covered. Python additionally checks nested detached arrays,
immutable results, concurrent evaluation and argument rejection.

The [bucketed fixture](../../fixtures/stochastic-dividends/rough-continuous-bucketed-local-vol-reference.json)
retains separately compiled inputs for nodes 4 and 0 in H=0.1 Up-out Call and
Down-out Put cases. These inputs were produced with the parent d82bb42 wheel,
before the bucket API existed. NumPy uses 32 batches of 8,192 antithetic pairs
with seed 20261008, shared across nodes. Four node panels and two paired sum
panels compare against production with 16 scrambles of 32,768 points. For each
node/sum Vega and adjacent gap, absolute difference plus four combined SEs must
be below 2.0 per unit volatility (0.02 per vol point); node/sum Vega SEs must be
below 0.35 in both implementations. This validates valuation conditional on the
retained recalibration inputs, not an independent particle calibration or a
market-IV/zero-bump sensitivity.

Reporting-projection controls use a separately specified density-hat/bilinear
matrix on nonmatching time and spatial grids, with both full and restricted
active domains. Separate full-request scenario prices reconstruct all bucket,
sum, pre-projection, residual and gap errors for MC/RQMC, with and without
antithetics. Controls cover worker replay, frozen fixed cash, missing/insufficient
basis coverage, threshold/bump validation and fingerprints.

The [reporting fixture](../../fixtures/stochastic-dividends/rough-continuous-reporting-iv-reference.json)
retains all nine nodes' separately compiled 0598117 scenario inputs for the
H=0.1 Up-out Call case. NumPy independently re-evolves and values all 54 scenarios,
using 32 batches of 8,192 antithetic pairs, seed 20261009. An independent
flat-IV density and hat/bilinear map supplies two projection panels, at thresholds
1e-8 and 0.9. These include edge assignment on a narrower reporting grid and
interpolation onto different reporting maturities. Production uses 16 scrambles
of 32,768 points. Every bucket, gap, pre-projection, projected sum and residual
must have absolute difference plus four combined SEs below 4.0 in raw reporting
units, with reference/production SE below 0.7. Calibration inputs are checked by
separate recompilation. These checks establish agreement of the finite algorithm
and stated reporting convention; they do not validate market-IV hedge risk or
the full VegaKT operator.

The three-OS Barrier job runs the Rust controls. Linux regenerates all retained
NumPy batches and coupled refinements. The source archive and wheel contract
include the new API, tests and [example](../../examples/python/rough_dividend_continuous_barrier.py).

## Remaining work

Zero-bump continuous Barrier sensitivities require treating the bridge
estimator and its discontinuous endpoint/jump branches; discrete graph adjoints
are not valid substitutes. The full VegaKT operator, physical-Spot quote conversion/refitting and other model/market risks remain
unsupported by this continuous wrapper. Parallel and selected retained-quote
rebuilding are supported under the interpolation contract above; the reporting
projection is a separate convention. Independent fine-path studies, calibration-aware refinement
and wider H/volatility/correlation/near-barrier panels remain necessary before
claiming broad continuous-time accuracy. This API is an explicit
approximation, not an extension of the discrete hard-Delta guarantee.
