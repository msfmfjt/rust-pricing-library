# Production hard Barrier Spot risk by survival conditioning

`StochasticDividendPricingPlan::evaluate_lsv_hard_barrier_spot_risk()` returns
hard-payoff price, physical-Spot Delta and their sampling errors for rough
residual-equity LSV. It is a separate opt-in method on a price-only plan,
available in Rust and on Python `StochasticDividendPlan`.
The existing pathwise `evaluate_lsv_spot_risk()` retains its smoothing
requirement for discontinuous payoffs.

## Contract and estimator

Supported contracts are discrete up/down knock-in/out calls and puts with optional fixed cash rebates and
all monitoring dates strictly after valuation. Payment may follow expiry;
cash means beyond expiry remain funded. Historical
or initial observations, smoothing requests, non-rough models, and degenerate
conditional equity laws are rejected. Continuous monitoring remains rejected
by the shared compiler. Other Greeks and automatic request-level dispatch are
not added by this method.

The conditioning requires `abs(rho_DV)<1` and

```text
q = (rho_SV-rho_SD*rho_DV)/sqrt(1-rho_DV^2),
1-rho_SD^2-q^2 > 0.
```

Each step reserves four independent normal coordinates, in dividend,
orthogonal-volatility, newest Volterra-cell residual and free-equity order.
The rough history uses the production hybrid weights and finite-grid variance
centering. The leverage lookup uses the original left-constant time row and
the existing log-equity interpolation and slope. All original grid knots,
contractual observations and cash-event ordering are retained.

Given the first two normal coordinates, next equity is lognormal. The
symmetric dividend update makes next Y affine in next f. At each monitored
node, requiring pre-cash stock below an Up barrier gives an upper equity
cutoff; requiring post-cash stock above a Down barrier gives a lower cutoff.
Sample the truncated normal and multiply by its
survival probability. Differentiate both the probability weight and the
inverse-normal transport, carrying f/Y Spot tangents into subsequent leverage
lookups. The final equity innovation is integrated analytically, including
the exercise and monitoring boundaries. The terminal free-equity coordinate
remains reserved but unused.

The knock-out estimator is the survival-weighted vanilla payoff. Knock-in is
a coupled vanilla payoff minus that knock-out. Terminal intrinsic value uses
post-cash stock; monitoring checks the pre/post-cash maximum for Up and
minimum for Down. The result includes contractual
notional and the discount factor to the payment date. A fixed cash rebate
pays on the inactive branch, independently of notional. Its probability
and Spot tangent are combined with the payoff inside each sampling unit;
see the [rebate derivation and controls](stochastic-dividend-hard-barrier-rebates.md).

The [independent Python reference](stochastic-dividend-survival-barrier-reference.md)
implements this law without calling the Rust transition, payoff, derivative,
normal generator or reduction code. Its retained calibrations are inputs.

## Spot convention, sampling and numerical policy

Physical Spot changes the funded residual-equity anchor. Particle calibration
and leverage interpolation are homogeneous in that anchor, so calibrated
leverage values remain unchanged. The unconditional normalized f/Y law is
Spot-independent. Conditioned paths move with Spot through their cutoffs;
their state and weight tangents must therefore be included. The reconstruction
coefficient derivative is the deterministic carry growth, not one.

The new method shares the existing deterministic MC/RQMC reduction. MC errors
use independent antithetic-pair means when enabled; RQMC errors use independent
scramble means. Brownian bridge operates on the independent normal coordinates
before conditional transport. Knock-in's two legs are combined within each
sampling unit, preserving their covariance, including the rebate. `evaluated_paths` counts signed
input samples, not the two coupled legs separately.

The method label is
`buehler-rough-residual-lsv-hard-barrier-survival-spot-v2`. The returned price
fingerprint hashes the compiled plan fingerprint with this method label,
distinguishing the conditional estimator and coordinate order from ordinary
hard-indicator valuation. Repeated calls do not mutate the plan or change
ordinary price/Spot-risk behavior. The result's uncertainty scope remains
`pricing_conditional_on_calibration`.

Normal transport uses complementary tails near probability one. It does not
clamp quantiles or floor positive survival probabilities. Impossible survival
(a nonpositive upper equity cutoff) contributes exactly zero surviving vanilla
payoff and tangent; a knock-out rebate remains payable. A nonpositive Down cutoff imposes no restriction. An unrepresentable quantile, nonpositive/nonfinite evolution,
underflowed survival weight or invalid conditional scale returns an error.
The existing normal CDF/quantile approximations are used; this is a floating-
point estimator, not certified tail arithmetic. No fallback to smoothed or
finite-bump risk is performed.

## Python API

Compile a price-only request with
`StochasticDividendPlan.compile_rough_bergomi_lsv(...)`, then call the dedicated
method. The [runnable example](../../examples/python/rough_dividend_hard_barrier.py)
builds a Down-and-Out Put with delayed payment and a fixed cash rebate:

```python
risk = plan.evaluate_lsv_hard_barrier_spot_risk()
print(risk.price.value, risk.price.standard_error)
print(risk.delta, risk.delta_standard_error)
```

The method releases the GIL while Rust evaluates and returns the existing
immutable `StochasticDividendLsvSpotRisk`/`StochasticDividendPrice` types.
`retain_reverse_trace=False` is sufficient. The returned price is the
conditional hard estimator, with its own fingerprint; it need not equal the
ordinary hard-indicator `plan.evaluate()` estimate sample for sample.
`risk.uncertainty_scope` is `pricing_sampling_only_scale_invariant_calibration`,
and `risk.price.uncertainty_scope` is `pricing_conditional_on_calibration`.
Both exclude calibration uncertainty and time-grid bias.

Unsupported compiled plans raise `PricingError`. Generic Delta request flags
continue to use existing validation: an unsmoothed Barrier with `delta=True`
is rejected by request validation, and stochastic-dividend factories do not
accept automatic Delta dispatch even when smoothing is present. The dedicated
method accepts neither a smoothing argument nor a fallback estimator.

[Python binding controls](../../tests/python/test_rough_dividend_hard_barrier.py)
cover two retained independent references with the original numerical budgets,
all sixteen rebate contract/Hurst combinations against recompiled Spot bumps,
MC/RQMC counts, immutable metadata, numerical worker replay and concurrent
calls on a shared plan. Execution policy remains part of the fingerprint;
recompiling with another worker count preserves numerical results, not the
fingerprint. Scope tests cover smoothing, initial monitoring, vanilla products,
non-rough LSV and singular conditional equity laws; ordinary APIs retain their
behavior. The wheel smoke contract checks the method name and signature,
discovers the new tests and runs the example. Source archives require both.

## Validation

Nine fast unit tests cover:

- Per-sample analytic Delta against two full-recalibration Spot bumps, at
  H=0.1/0.3/0.5, with and without terminal monitoring, for all eight
  direction/side/style combinations, with and without a rebate and notional=2
  (2e-6 absolute tolerance).
- MC means and standard errors against explicit sampling-unit reductions,
  both with and without antithetics, plus identical results with 1/2 workers.
- Rejection of initial observations, Asian
  contracts and singular conditioning correlations.
- Knock-in/out parity and notional scaling on common inputs with delayed
  payment, including both price and Delta for Up/Down and Call/Put.
- Complementary-tail transport when the ordinary CDF rounds to one, and
  errors for unrepresentable quantiles and nonfinite normals.
- Nonpositive cutoffs: impossible Up survival versus unrestricted Down survival.
- Fixed cash rebate parity and independence from notional, terminal/nonterminal
  monitoring, empty exercise intervals, and rare-hit complementary tails.

A public integration control checks replay, counts, uncertainty scope,
fingerprint separation, smoothing rejection and unchanged ordinary valuation.
The original Up Call numerical panel checks seven independent references at seeds
193/877: one exact zero-eta/kappa limit, H=0.1/0.3 two-step quadrature, and four
4/8-step survival cases with 2/4 observations. Each uses 16 scrambles of
16,384 points with antithetics and Brownian bridge (524,288 signed samples).

For each quantity, require

```text
combined_SE = sqrt(production_scramble_SE^2 + independent_reference_SE^2),
abs(estimate-reference) + 4*combined_SE < 0.03 for price, < 0.015 for Delta;
production_scramble_SE < 0.003.
```

The deterministic-reference cases have no reference sampling SE; their
quadrature agreement tolerances are covered by the separate reference tests.
The Monte Carlo reference error is included explicitly. The first numerical
run passed all 28 rows without changing counts, seeds or budgets. Observed
maximum comparison margins were 0.023180748 for price and 0.002491302 for
Delta; maximum production SEs were 0.001950279 and 0.000430114 respectively.

```sh
cargo test --locked -p pricing --lib lsv_hard_barrier
cargo test --locked -p pricing --no-default-features --lib lsv_hard_barrier
cargo test --locked --release -p pricing --test stochastic_dividend_barrier_reference -- --include-ignored --nocapture
```

The full Barrier integration suite includes the earlier smoothed comparisons
as regression controls. Its six numerical panels emit 212 JSON rows: 72
earlier reference comparisons, 28 Up Call production comparisons and 48
[direction/side extension comparisons](stochastic-dividend-hard-barrier-styles.md),
plus 64 [rebate comparisons](stochastic-dividend-hard-barrier-rebates.md). The
three-OS Barrier CI job retains these in the existing reference log and runs
the nine fast controls in `stochastic-dividend-hard-barrier-risk.log`.

This validates the stated discrete laws and Spot convention conditional on
finite-particle calibration. It does not certify continuous-time accuracy,
calibration uncertainty, all parameter regimes or other barrier styles.
The separate [fixed-surface refinement panel](stochastic-dividend-hard-barrier-refinement.md)
measures independent hard price/Delta changes against a finite 128-step law;
it is not a universal error bound for this API.
