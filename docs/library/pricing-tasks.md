# Complete tasks, capability queries and valuation reports

`PricingTask` saves a validated source together with the operation to run.
`PricingSource` has single-asset, multi-asset and stochastic-dividend variants.
The immutable task compiles through existing native plans and returns both a
common `ValuationReport` and the original typed `native_result`.

The [Python example](../../examples/python/pricing_task.py) runs the
[complete task fixtures](../../fixtures/task-v1). The design and compatibility
impact are recorded in [ADR 0030](../../design/adr/0030-pricing-task-capability-report.md).

The [backtest provider](vega-kt-backtesting.md#common-taskreport-evidence) uses
common tasks for ordinary valuation and exposes replayable `PricingEvidence`.
Its ledger retains narrower model/risk acceptance rules; dedicated Gamma-chain
and paired-sample reports remain authoritative for those operations.

## Query before executing

Source `capability(operation)` performs pure inspection: it does not calibrate,
generate paths or allocate an AAD workspace. Its status and reason are typed
Rust enums and stable Python strings:

| Status | Meaning |
| --- | --- |
| `unsupported` | The requested operation is unavailable or a known source prerequisite fails; `reason` and `detail` explain why. |
| `conditional` | Static checks passed; native compilation and risk preflight remain necessary. |
| `supported` | The compiled task passed its native operation preflight. Numerical evaluation can still fail. |

To see what one source offers before choosing an operation, list them:

```python
task = rp.PricingTask.from_source(source)
print(task.source_family)                 # for example "local_stochastic_volatility"
for capability in task.available_operations():
    print(capability.operation, capability.status, capability.reason)
```

In Rust, `PricingSource::family()` and `PricingSource::available_operations()`
return the same information. The list covers `evaluate`, first-order `aad` and
`local_variance_risk`; basket cross Gamma depends on its bump and is queried
with `capability`.

Task construction, capability queries and compilation share the operation
selection rules. Native evaluation and compiled preflight share the same guards
for differentiability, reverse traces and Gamma bumps. Covariance, resource,
realized calibration and numerical acceptance remain the existing compiler's
responsibility. A source query never claims these have passed.

Compiled basket Gamma preflight also checks that the requested bump produces
distinct, finite Spot values using the evaluator's arithmetic. A positive bump
inside `(0, 1)` can still be too small to move Spot in floating point. Such a
selection returns `unsupported` with `native_preflight_required`; compiling or
reselecting that operation fails before pricing paths are generated. A failed
selection leaves the original compiled task usable.

Known rejection reasons include `dedicated_risk_api`, `discontinuous_payoff`,
`reverse_trace_required` and `invalid_gamma_bump`. A source selected for price
can inspect another operation without compiling it. A compiled task uses
`capability()` for its saved operation and `capability_for(operation)` for another
operation against the same realized plan.

## Reuse compilation and calibration

Use `CompiledPricingTask.with_operation(...)` to select another operation while
sharing the compiled native plan, calibrated surfaces and retained reverse traces:

```python
task = rp.PricingTask.from_source(source, operation="evaluate")
price_plan = task.compile()
price = price_plan.evaluate()
if price_plan.capability_for("local_variance_risk").status == "supported":
    risk_plan = price_plan.with_operation("local_variance_risk")
    risk = risk_plan.evaluate()
    assert risk_plan.plan_fingerprint == price_plan.plan_fingerprint
    saved_risk_task = risk_plan.task.to_json()
```

In Rust the same methods accept `PricingOperation`; `task()` returns the selected
replayable task. `PricingTask::with_operation` only selects a source operation,
and its subsequent `compile()` builds a fresh plan. Use the method on an already
compiled task when reuse is intended.

Selection returns a new immutable handle. It shares source and plan ownership,
so dropping the original handle is safe. Different operations and Gamma bumps
have their own task fingerprints and reports while retaining the same native
plan identity. The selected `task` can be serialized and compiled independently
for replay; no compiled state is added to the wire format.

Source prerequisites and native preflight run on every selection. Missing reverse
traces, discontinuous payoffs or inadmissible Gamma bumps remain errors; selection
does not enable traces or change inputs. In Python, failures of these source/native
checks use the existing `compile` error stage and selected operation; invalid
operation labels or argument combinations retain validation errors. A failed selection
leaves the original plan usable. Change source inputs, seeds, grids or execution
policy by creating and compiling a new source task.

This is explicit ownership within a process, with no global cache, eviction policy
or cross-source fingerprint lookup. Cloning a Rust compiled task shares its native
plan. Evaluation still owns separate path/risk workspaces and runs the selected
native algorithm, including any required calibration pullbacks or bump work.
Selection itself performs no compilation/calibration; it still validates the
operation and computes its task fingerprint. Concurrent operations on shared
handles preserve deterministic reduction and existing sampling-error semantics.
The sharing boundary is recorded in [ADR 0034](../../design/adr/0034-compiled-task-reuse.md).

## Operations and supported compositions

| Source | `evaluate` | `aad` | `local_variance_risk` |
| --- | --- | --- | --- |
| Standard single-asset BS / Black-76 / LV / Bass-LV | Existing configured price/risk/LSM request | Use configured standard risk with `evaluate` | Use configured native risk request |
| 1F/2F/rough Bergomi SV | Price | Native AAD | Unavailable |
| Deterministic Bergomi or rough-family LSV | Price | Use dedicated APIs | Recalibrated target-grid risk |
| Single-asset Hull–White BS / Bergomi SV / LSV | Price | Native hybrid AAD, including retained target risk | Use hybrid AAD |
| Direct rough-volatility families | Price | Use dedicated APIs | Unavailable |
| Multi-asset BS/LV, mixed Bergomi LSV, shared HW, joint Local Correlation | Price | Native full AAD; optional full cross Gamma | Included in native AAD |
| Stochastic dividends with BS or 1F/2F/rough Bergomi SV, or BS/HW | Price | Native first-order AAD | Unavailable |
| Deterministic-rate stochastic-dividend Bergomi LSV | Price | Use target-grid operation | Recalibrated residual-equity target-grid risk |

The same support, generated from `SourceFamily::exposes`, which capability
queries and task construction also read. `source.family()` returns the row and
`source.available_operations()` returns the capability of each operation for
one source. "Exposed" means the common task offers the operation for that
family; the payoff, reverse-trace, compilation and native preflight conditions
still apply. "Not exposed" is the `dedicated_risk_api` capability reason.

<!-- operation-matrix:begin (generated; see crates/pricing/tests/operation_matrix.rs) -->
| Source family | `evaluate` | `aad` | `aad` with `gamma_relative_bump` | `local_variance_risk` |
| --- | --- | --- | --- | --- |
| `standard_single_asset` | exposed | not exposed | not exposed | not exposed |
| `stochastic_volatility` | exposed | exposed | not exposed | not exposed |
| `local_stochastic_volatility` | exposed | not exposed | not exposed | exposed |
| `rough_family_lsv` | exposed | not exposed | not exposed | exposed |
| `single_asset_hull_white` | exposed | exposed | not exposed | not exposed |
| `direct_rough_volatility` | exposed | not exposed | not exposed | not exposed |
| `multi_asset` | exposed | exposed | exposed | not exposed |
| `stochastic_dividend` | exposed | exposed | not exposed | not exposed |
| `stochastic_dividend_lsv` | exposed | not exposed | not exposed | exposed |
<!-- operation-matrix:end -->

The matrix describes the common task operations. Specialized parameter,
correlation, Hurst, market-IV, Gamma and dual-bound APIs retain their own
contracts when not exposed here. Continuous-barrier monitoring wrappers are
also separate. No new model/product combination or numerical acceptance rule
is implied by a source format. Hard live indicators reject pathwise risk;
explicit graph smoothing or the native supported smoothing contract is needed.

`aad.gamma_relative_bump` is optional and belongs only to multi-asset tasks.
It must be finite and in `(0,1)` and pass the native representability checks.
Without it, basket AAD returns first-order risk. Standard single-asset Greeks
remain part of the source's existing `RiskRequest` and `evaluate` operation.

## Measures

The `measures` operation asks for quantities by their financial meaning, with
the estimation method stated on each one
([ADR 0043](../../design/adr/0043-measure-oriented-analysis-facade.md)).
It is the path new risk is added on; `evaluate`, `aad` and
`local_variance_risk` keep their meaning and their saved documents.

```python
M = rp.MeasureRequest
task = rp.PricingTask.from_source(
    source, operation="measures",
    measures=[M.spot_delta(), M.spot_gamma(relative_bump=0.01),
              M.parallel_market_iv_vega()],
)
report = task.compile().evaluate().report
delta = report.risk("spot_delta")          # the raw-unit row
rows = report.to_records()                 # price and every coordinate, flat
```

```rust
let measures = MeasureSet::new(vec![
    MeasureRequest::native(Measure::SpotDelta),
    MeasureRequest::spot_gamma(SpotBump::relative(0.01)?),
])?;
let task = PricingTask::new(source, PricingOperation::Measures(measures))?;
```

| Measure | Meaning |
| --- | --- |
| `price` | Always part of the result, listed or not. |
| `spot_delta` | First derivative to Spot under the source's smile dynamics. |
| `spot_gamma` | Second derivative to Spot. |
| `parallel_market_iv_vega` | Parallel shift of the market implied-volatility surface; for a constant-volatility model, its single volatility. |
| `vega_kt` | Bucketed quote-node Vega on the grid given with the request. |
| `model_parameters` | Derivatives to the native model parameters listed with the request, one row per parameter. Not a Vega. |
| `correlations` | Derivatives to the Brownian correlations listed with the request, one row per correlation. |
| `dividend_cash` | Derivatives to the mean of each cash dividend, one row on an ex-dividend-time axis. |
| `cross_gamma` | The full Spot Gamma matrix of a basket, one row per ordered pair of underlyings. |
| `discount_curve` | Derivatives to the log discount factor of each discount-curve pillar, with their DV01 projection: two rows on a discount-time axis. |

| Method | Meaning |
| --- | --- |
| `native` | The backend's own estimator, with nothing chosen by the caller. |
| `native_spot_bump` | The backend's own second-order estimator: a central difference of its native Deltas on common random numbers. The caller states the bump. |
| `native_calibrated` | The backend's own estimator on a calibrated source. The caller states the calibration treatment: `fixed` or `recalibrated`. |
| `native_spot_bump_calibrated` | The backend's own second-order Spot estimator on a calibrated source. The caller states the bump and the calibration treatment. |
| `bump_and_revalue` | A central difference of complete revaluations: the source is rebuilt with the input moved up and down, and each side is priced on the task's own random coordinates. The caller states the bump size and the calibration treatment. |

A method is never substituted. Standard single-asset Gamma is a bump of native
Deltas, so `spot_gamma` with `native` is `unsupported` with reason
`no_native_estimator` and a detail that names `native_spot_bump`.

Rough-family LSV (`rough_family_lsv`) has native Spot Delta and Gamma under two
conventions, so the caller states which one with the calibration treatment:

- `fixed`: the frozen-leverage Delta and Gamma. The calibrated leverage and its
  reference funded-forward anchor are held, and the initial residual equity
  moves with Spot.
- `recalibrated`: the sticky-moneyness Delta and Gamma. The relative
  local-variance target is held in log-moneyness and recalibrated; by scale
  equivariance the leverage values do not change and the anchor follows Spot.
  This is not sticky-strike market-IV risk.
- `not_applicable` is `bump_calibration_mismatch`. A digital or other
  discontinuous payoff without explicit smoothing is `discontinuous_payoff`.

Each row is the dedicated API's estimate and error, bit for bit
(`evaluate_frozen_leverage_delta`, `evaluate_sticky_moneyness_delta`, and the
matching `*_gamma_bump` with the absolute bump the request resolves to). The
Gamma row is the full-bump second difference; the half-bump Gamma and the
paired difference between the two stay in `native_result`. The rough-family
plan fixes no risk selection when it is compiled, so `with_operation` can move
between price-only and Spot selections on one compiled task. Bergomi LSV
(`local_stochastic_volatility`) has no native Spot estimator and offers
`bump_and_revalue` only.

An LSV target is a local-variance grid, not market implied volatilities. It
has a `parallel_market_iv_vega` only when it retains a reporting IV basis
whose strict Dupire construction (natural-cubic total variance, linear in
time, no floor or cap repair) on the target's own axes, floor and cap
reproduces the grid bit for bit. Then a move of the basis IVs is a move of
the target, and the calibration follows:

- On rough-family LSV, `parallel_market_iv_vega` with `native_calibrated` and
  `recalibrated` is the dedicated market-IV risk's parallel Vega
  (`market_iv_risk_plan(surface).evaluate()`), bit for bit, per one absolute
  IV unit added to every basis node. It differentiates through the particle
  recalibration, so it needs `retain_reverse_trace=true`
  (`reverse_trace_required` otherwise). The node risks stay in
  `native_result.market_iv`. Its error exists only under RQMC.
- On both LSV families, `bump_and_revalue` moves every basis IV by an
  absolute size, rebuilds the target from the moved surface and recalibrates
  each side.
- `fixed` is `bump_calibration_mismatch` for both methods: a market-IV move
  reaches LSV prices only through the calibration, so a held calibration
  would report zero. A relative bump is `model_combination`, since a surface
  has no single level to scale. A target without a basis, or with one that
  does not reproduce it, is `model_combination`.

Rough-family LSV also has native model-parameter risk. One `model_parameters`
request lists the parameters by role, and the report has one row per
parameter in that order, each with its typed `model_parameter`:

```python
M.model_parameters(["vol_of_vol", "correlation", "hurst"],
                   calibration="recalibrated")
report.model_parameter_risk("hurst")
```

| Model | Parameters | Native API |
| --- | --- | --- |
| Rough Heston | `initial_variance`, `mean_reversion`, `long_run_variance`, `vol_of_vol`, `correlation`, `hurst` | `evaluate_heston_parameter_risk` |
| Lifted Heston | the same without `hurst` (a finite lift has no Hurst) | `evaluate_heston_parameter_risk` |
| Quadratic rough Heston | `initial_state`, `mean_reversion`, `vol_of_vol`, `quadratic`, `shift`, `variance_floor`, `hurst` | `evaluate_quadratic_heston_parameter_risk` |
| Mixed rough Bergomi | `component_vol_of_vol.<i>`, `correlation`, `hurst` | `evaluate_mixed_bergomi_parameter_risk` (`_with_hurst` when `hurst` is listed) |

- Each row is the dedicated API's total derivative and error, bit for bit,
  per one absolute unit of the parameter. The target is held fixed and the
  particle calibration is differentiated, so the method is
  `native_calibrated` with `recalibrated`. `fixed` is
  `bump_calibration_mismatch`; the fixed-leverage part
  (`direct_adjoints`) stays in `native_result.model_parameters.risk`, without
  an error of its own.
- The dedicated API is run with Hurst exactly when `hurst` is listed. Its
  result, with the identifier of each coordinate, is
  `native_result.model_parameters`.
- It needs `retain_reverse_trace=true` (`reverse_trace_required`) and a
  differentiable payoff or explicit smoothing (`discontinuous_payoff`).
- A parameter the model does not have is `model_combination`, with the
  model's list in the detail. Rough SABR and RFSV have no native parameter
  risk (`no_native_estimator`).
- These are not Vegas. Volatility is compared across models only through
  `parallel_market_iv_vega` and `vega_kt`.

A direct rough-volatility source (`direct_rough_volatility`, a rough model
without a local-volatility target) is not calibrated, so its native methods
state no calibration treatment and its rows are `not_applicable`:

```python
measures=[M.spot_delta(), M.spot_gamma(relative_bump=0.01),
          M.model_parameters(["vol_of_vol", "hurst"])]
```

- `spot_delta` with `native` is `evaluate_delta`: pathwise through payoff,
  path and dividend escrow, with every model parameter, the curves and the
  dividend amounts held.
- `spot_gamma` with `native_spot_bump` is `evaluate_gamma_bump` with the
  absolute bump the request resolves to (a relative bump scales the physical
  Spot). The row is the full-bump second difference; the half-bump Gamma
  and the paired difference stay in `native_result`.
- `model_parameters` with `native` is the fixed-model parameter risk, for
  the same four models and parameter names as the table above. The kernel
  (or Hurst), Spot, curves, dividends and Gaussian draws are held. Rough
  Heston gets its five scalar parameters from
  `evaluate_heston_parameter_risk` and Hurst from `evaluate_hurst_risk`, and
  only the run a listed parameter needs is made. Quadratic rough Heston and
  mixed rough Bergomi take Hurst in the same run, exactly when `hurst` is
  listed.
- Every row is the dedicated API's estimate and error, bit for bit, and has
  an error under both samplers. All six models have Spot Delta and Gamma;
  rough SABR and RFSV have no parameter risk (`no_native_estimator`). A
  calibrated method (`native_calibrated`) is `no_native_estimator`, and a
  discontinuous payoff without smoothing is `discontinuous_payoff`.
- `native_result` is a `RoughVolatilityMeasureRisk` holding each dedicated
  result unchanged; `model_parameters` has typed `heston`, `hurst`,
  `quadratic_heston` and `mixed_bergomi` parts, present when they were run.

Stochastic-dividend sources have native Spot Delta and Gamma:

- Without a calibration step (`stochastic_dividend`: Black-Scholes, Bergomi
  or rough Bergomi residual equity, deterministic or Hull-White rates),
  `spot_delta` with `native` is the first coordinate of `evaluate_aad`, and
  `spot_gamma` with `native_spot_bump` is `evaluate_gamma`: a central
  difference of AAD Deltas on common random numbers. The Q-mean cash
  dividends, the dividend model, correlations, curves, dates and grid are
  held, and the rows are `not_applicable`.
- On a residual-equity LSV (`stochastic_dividend_lsv`) they are
  `evaluate_lsv_spot_risk` and `evaluate_lsv_gamma`. A Spot move re-anchors
  the calibrated surface to the moved funded residual equity, and its
  squared-leverage values do not change: by scale invariance this is the
  recalibrated Delta. So the methods are `native_calibrated` and
  `native_spot_bump_calibrated` with `recalibrated`. There is no
  frozen-leverage estimator, so `fixed` is `no_native_estimator`;
  `bump_and_revalue` with `fixed` remains. `not_applicable` is
  `bump_calibration_mismatch`.
- Each row is the dedicated API's estimate and error, bit for bit; the
  Gamma row is the base-bump estimate, with the half- and double-bump
  ladder in `native_result`. A discontinuous payoff without smoothing is
  `discontinuous_payoff`. The native result is a
  `StochasticDividendMeasureRisk`.

A stochastic-dividend source without a calibration step also has native
correlation risk. One `correlations` request lists the correlations by the
roles of their drivers, and the report has one row per correlation in that
order, each with its typed `correlation`:

```python
M.correlations(["equity_dividend", "spot_volatility.0"])
report.correlation_risk("equity_dividend")
```

| Source | Correlations |
| --- | --- |
| Black-Scholes residual equity | `equity_dividend` |
| Black-Scholes with Hull-White rates | `equity_dividend`, `equity_rate`, `dividend_rate` |
| One-factor or rough Bergomi | `equity_dividend`, `spot_volatility.0`, `dividend_volatility.0` |
| Two-factor Bergomi | `equity_dividend`, `spot_volatility.0`, `spot_volatility.1`, `dividend_volatility.0`, `dividend_volatility.1`, `volatility_factors` |

- Each row is the raw partial of `evaluate_correlation_aad`, bit for bit:
  one symmetric pair of the driver correlation matrix is varied and every
  other entry held, with no projection to a valid matrix and no
  recalibration. The method is `native` and the row is `not_applicable`.
- The joint reverse also differentiates Spot, volatility, cash means,
  curves and the volatility model's parameters; those coordinates stay in
  `native_result.correlations.risk`.
- A correlation the source does not have is `model_combination`, with the
  source's list in the detail. A discontinuous payoff without smoothing is
  `discontinuous_payoff`; a correlation at the edge of the valid domain
  fails at evaluation.
- Residual-equity LSV correlation risk is a finite difference with stated
  bumps, partly recalibrated, and has no measure adapter yet.

The same sources have native model-parameter risk through the
`model_parameters` measure with no calibration treatment. The labels are the
same vocabulary as on rough-volatility sources, with the dividend and rate
roles added:

```python
M.model_parameters(["dividend_volatility", "rate_volatility.0"])
report.model_parameter_risk("dividend_volatility")
```

| Source | Parameters | Native API |
| --- | --- | --- |
| Black-Scholes residual equity | `initial_volatility`, `dividend_mean_reversion`, `equity_linkage`, `dividend_volatility` | `evaluate_aad` |
| One-factor Bergomi | the four above, `factor_mean_reversion.0`, `vol_of_vol` | `evaluate_bergomi_aad` |
| Two-factor Bergomi | the four above, `factor_mean_reversion.0`, `factor_mean_reversion.1`, `vol_of_vol`, `mixing_weight` | `evaluate_bergomi_aad` |
| Rough Bergomi | the four above, `hurst`, `vol_of_vol` | `evaluate_rough_aad` |
| Black-Scholes with Hull-White rates | the four above, `rate_mean_reversion`, `rate_volatility.<i>` per knot | `evaluate_hull_white_aad` |

- The run follows the list. When only the four Buehler and residual
  volatility parameters are listed it is `evaluate_aad`, the same run as a
  listed native Delta. When any model parameter is listed it is the model's
  extended reverse from the table. Each row is that run's derivative and
  error, bit for bit.
- The model is fixed: every other parameter, Spot, correlations, cash
  means, curves, dates, grid and Gaussian draws are held. The method is
  `native` and the row is `not_applicable`. `initial_volatility` is the
  derivative to the initial residual volatility, not a market-IV Vega.
- The whole run, with every coordinate it produced, is in
  `native_result.model_parameters.risk`.
- A parameter the source does not have is `model_combination`, with the
  source's list in the detail.
- Residual-equity LSV parameter risks (`evaluate_lsv_dividend_model_risk`,
  the Bergomi and rough Bergomi LSV parameter risks) are finite differences
  with stated bumps and have no measure adapter yet.

Cash-dividend risk is one `dividend_cash` row per source, with one value per
cash dividend of the market's schedule on an `ex_time` axis:

```python
M.dividend_cash()                             # without a calibration step
M.dividend_cash(calibration="recalibrated")   # residual-equity LSV
report.risk("dividend_cash")
```

- Without a calibration step (Black-Scholes, Bergomi and rough Bergomi
  residual equity, deterministic or Hull-White rates) the method is `native`
  and the values are the `cash_mean` coordinates of `evaluate_aad`, bit for
  bit. They are pathwise through the Buehler escrow split, the funded reserve
  and the payment discount. When Delta, or only the four basic model
  parameters, are listed too, all of them share that one run.
- On a residual-equity LSV the method is `native_calibrated` with
  `recalibrated`, from `evaluate_lsv_market_risk`. A cash mean only
  re-anchors the funded residual equity, and the calibration is invariant to
  that scale, so the squared leverage is unchanged, as for its Spot Delta.
  `fixed` is `no_native_estimator`.
- Each value is per one currency unit of that dividend's risk-neutral mean,
  with Spot, the dividend and volatility models, correlations, curves, dates
  and grid held. The native run is in `native_result.dividend_cash`.
- A market without cash dividends is `model_combination`; a discontinuous
  payoff without smoothing is `discontinuous_payoff`.

A basket has native Spot measures from its full AAD (`evaluate_aad`, the
same run as the `aad` operation):

```python
M.spot_delta()                                       # one row per asset
M.cross_gamma(relative_bump=0.01)                    # one row per ordered pair
M.spot_delta(calibration="recalibrated")             # LSV or Local Correlation basket
M.cross_gamma(relative_bump=0.01, calibration="recalibrated")
```

- `spot_delta` has one row per underlying, carrying that underlying's id.
  `cross_gamma` has one row per ordered pair `[i, j]`: the unsymmetrized
  central difference of the native Delta of `i` under a relative Spot bump
  of `j`, on common random numbers. Rows, errors and the native result
  (`MultiAssetPrice`) are those of `evaluate_aad`, bit for bit.
- A basket without a calibration step states no treatment (`native`,
  `native_spot_bump`). An LSV or Local Correlation basket states
  `recalibrated`: its marginals are calibrated in forward-normalized
  coordinates and the reverse runs through the joint calibration, so the
  native derivative is the sticky-moneyness, recalibrated one. A missing
  treatment on a calibrated basket, or one stated on an uncalibrated basket,
  is `bump_calibration_mismatch`. `fixed` is `fixed_calibration_unavailable`.
- The Gamma bump is relative and lies in (0,1). An absolute bump is
  `model_combination`. A missing reverse trace is `reverse_trace_required`,
  and a discontinuous payoff is `discontinuous_payoff`.
- `spot_gamma` on a basket remains `bump_and_revalue` (one row per asset);
  `cross_gamma` is its native, matrix-valued counterpart.

Discount-curve risk is two rows on a `discount_time` axis, one value per
pillar of the discount curve (including its fixed `t = 0` pillar):
`discount_curve.log_df.<method>` per unit log discount factor, and
`discount_curve.dv01.<method>`, the signed price change for +1bp in each
continuously compounded zero-rate pillar (`-1e-4 t` times the first row).

```python
M.discount_curve()                              # without a calibration step
M.discount_curve(calibration="recalibrated")    # residual-equity LSV, calibrated basket
[r for r in report.risks if r.measure == "discount_curve"]
```

- Stochastic dividends without a calibration step (deterministic or
  Hull-White rates): `native`, the `discount_log_df` coordinates of
  `evaluate_aad`, bit for bit. Delta, `dividend_cash` and the four basic
  model parameters share that run.
- A residual-equity LSV: `native_calibrated` with `recalibrated`, from
  `evaluate_lsv_market_risk`, which re-anchors the funded residual equity as
  for its Spot Delta. `dividend_cash` shares that run. `fixed` is
  `no_native_estimator`.
- A basket with shared Hull-White rates: the `hull_white_curve_risk` of
  `evaluate_aad`, bit for bit, in the Delta's run, under the basket's
  treatment rules for `spot_delta`. A deterministic-rate basket has no
  native curve derivative and refuses it with `no_native_estimator`.
- The repo-spread curve is held. Its pillars are in the native result
  (`native_result.discount_curve` on dividends, `hull_white_curve_risk` on
  baskets) but have no measure.

Which measures a source family produces, and by which method:

<!-- measure-matrix:begin (generated; see crates/pricing/tests/operation_matrix.rs) -->
| Source family | `price` | `spot_delta` | `spot_gamma` | `parallel_market_iv_vega` | `vega_kt` | `model_parameters` | `correlations` | `dividend_cash` | `cross_gamma` | `discount_curve` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `standard_single_asset` | `native` | `native`, `bump_and_revalue` | `native_spot_bump`, `bump_and_revalue` | `native`, `bump_and_revalue` | `native` | not exposed | not exposed | not exposed | not exposed | not exposed |
| `stochastic_volatility` | `native` | `bump_and_revalue` | `bump_and_revalue` | not exposed | not exposed | not exposed | not exposed | not exposed | not exposed | not exposed |
| `local_stochastic_volatility` | `native` | `bump_and_revalue` | `bump_and_revalue` | `bump_and_revalue` | not exposed | not exposed | not exposed | not exposed | not exposed | not exposed |
| `rough_family_lsv` | `native` | `native_calibrated`, `bump_and_revalue` | `native_spot_bump_calibrated`, `bump_and_revalue` | `native_calibrated`, `bump_and_revalue` | not exposed | `native_calibrated` | not exposed | not exposed | not exposed | not exposed |
| `single_asset_hull_white` | `native` | `bump_and_revalue` | `bump_and_revalue` | `bump_and_revalue` | not exposed | not exposed | not exposed | not exposed | not exposed | not exposed |
| `direct_rough_volatility` | `native` | `native`, `bump_and_revalue` | `native_spot_bump`, `bump_and_revalue` | not exposed | not exposed | `native` | not exposed | not exposed | not exposed | not exposed |
| `multi_asset` | `native` | `native`, `native_calibrated`, `bump_and_revalue` | `bump_and_revalue` | not exposed | not exposed | not exposed | not exposed | not exposed | `native_spot_bump`, `native_spot_bump_calibrated` | `native`, `native_calibrated` |
| `stochastic_dividend` | `native` | `native`, `bump_and_revalue` | `native_spot_bump`, `bump_and_revalue` | not exposed | not exposed | `native` | `native` | `native` | not exposed | `native` |
| `stochastic_dividend_lsv` | `native` | `native_calibrated`, `bump_and_revalue` | `native_spot_bump_calibrated`, `bump_and_revalue` | not exposed | not exposed | not exposed | not exposed | `native_calibrated` | not exposed | `native_calibrated` |
<!-- measure-matrix:end -->

`task.available_measures()` returns the row for one source, and
`task.measure_capabilities(measures)` answers each request with a status,
reason and detail. On a compiled task the same call answers against the
realized plan. In Rust these are `PricingSource::available_measures()`,
`PricingSource::measure_capabilities()` and
`CompiledPricingTask::measure_capabilities()`. `not exposed` cells answer
`dedicated_risk_api`: the quantity may exist behind `aad`,
`local_variance_risk` or a dedicated plan, but has no measure adapter yet.

### Bump and revalue

`bump_and_revalue` works for every source that can price, so every model gets
Spot Delta and Gamma through the same request. It is never chosen for the
caller, and its rows are labelled `bump_and_revalue`.

```python
delta = M.bump_and_revalue("spot_delta", relative=0.01,
                           calibration="recalibrated")
task.bump_cost([delta])        # {"calibrations": 2, "evaluations": 2}
```

- **What moves.** `spot_delta` and `spot_gamma` move Spot, one asset at a time
  for a basket (one row per asset). `parallel_market_iv_vega` moves the single
  volatility of a Black-Scholes or Black-76 equity, or every IV of an LSV
  target's reproducing reporting basis (absolute size, `recalibrated` only;
  see above); any other source is refused with `model_combination`. Delta and Gamma of the same size share
  their two sides.
- **Calibration is stated.** A calibrated source (LSV, Bass-LV, Local
  Correlation) needs `recalibrated` or `fixed`; any other source needs
  `not_applicable`. A wrong pairing is `bump_calibration_mismatch`.
  - `recalibrated` calibrates each side again with the same seeds. For LSV
    the target is in log-forward moneyness, so the leverage follows the
    forward (sticky moneyness).
  - `fixed` reuses the task's own realized calibration on both sides and runs
    no calibration. For LSV the unbumped leverage is kept as a function of the
    absolute level of the state it was calibrated on (sticky strike) and each
    side's paths start at its own bumped Spot. That state is the forward for
    deterministic rates, the escrowed forward for Hull-White rates and the
    funded residual equity for stochastic dividends. The two conventions agree
    exactly when the leverage does not depend on the level, and differ
    otherwise; which one a desk uses is a modelling choice the report row
    states.
  - `fixed` is available for single-asset LSV: Bergomi one- and two-factor,
    rough Bergomi and the rough families with deterministic rates, Bergomi
    and rough Bergomi LSV with Hull-White rates, and residual-equity LSV with
    stochastic dividends. The other calibrated sources refuse it with
    `fixed_calibration_unavailable`, because reusing their calibration does
    not express sticky strike. Bass-LV paths are generated from the
    calibrated marginals alone, so a reused calibration would ignore the
    bumped Spot and report zero Delta. Baskets are not offered it: their
    deterministic-rate LSV marginals are calibrated in forward-normalized
    coordinates (`f/F` starts at one), where a reused calibration is already
    the sticky-moneyness answer, and sticky strike would need a
    time-dependent shift of each surface and of the Local Correlation
    function. Use `recalibrated` for them.
- **Cost before running.** `bump_cost(measures)` on a task, or the capability
  detail, states the calibrations and evaluations the bumps add. Each scenario
  is two compilations and two evaluations, and a Gamma request adds one
  recorded evaluation of the task's own plan for its paired error. On a compiled
  task, `bump_cost()` reports what was actually run; the counts agree.
- **Paired standard error.** Each side is priced once while the shared
  executor records every sampling unit's value: a pseudo-random path, or one
  RQMC scramble's point average. Every engine is covered: the standard engine
  (including Bass-LV, American exercise and continuous barriers), stochastic
  and rough volatility, Hull-White hybrids, LSV, baskets and stochastic
  dividends. The estimate is the mean of the paired per-unit differences and
  its standard error is that mean's, so common random numbers cancel in the
  error as they do in the value. It excludes bump bias. With `recalibrated`
  the estimate is conditional on the calibration, as the report's
  uncertainty scope says. Should an engine's reductions not show its units,
  the row falls back to the plain difference of side prices with
  `standard_errors` `None`, never zero.
- **Exactness.** The estimate equals the difference of the rebuilt sources'
  prices up to summation order (the per-unit mean and the engine's
  fixed-block reduction add in different orders).
- Bumped sides are compiled with the task. `with_operation` reuses them for
  the same bump requests and refuses a different bump.

Rules that follow from asking for risk in one place:

- A standard single-asset source keeps its nested `RiskRequest` for
  `evaluate`. Under `measures` that request must select no Greek; otherwise the
  task is refused with `ambiguous_risk_selection`. Its numerical settings
  (smile dynamics, payoff smoothing, checkpoint and tile sizes) still apply.
- A measure list is the same calculation as a request that had set the
  corresponding flags: the native result, the plan fingerprint and the report
  numbers are bit-identical under pseudo-random and randomized quasi-Monte
  Carlo sampling.
- A standard plan fixes its Greeks at compilation. `with_operation` on a
  compiled task therefore accepts only the selection it was compiled with;
  another selection needs a new `compile()`. Price-only measures and
  `evaluate` share a plan.
- The list is ordered and holds each measure once. Order is part of the task
  fingerprint and does not change any number.

Report rows from a `measures` task carry a typed `measure` and the requested
`method_detail` next to the existing display `name`. `report.risk(measure)`
returns the row in raw units; the market-scaled row of the same measure follows
it in `risks`. Bucketed VegaKT stays under `vega_kt` with its covariance
layout. `to_records()` flattens the price and every risk coordinate into plain
rows of `measure`, `name`, `underlying_ids`, `coordinate`, `value`,
`standard_error`, `unit`, `method`, `calibration` and `uncertainty_scope`
(Python adds `task_fingerprint`), so a data frame can be built without the
library depending on one.

## Complete saved inputs

A single-asset source embeds the existing [pricing specification](pricing-specification.md).
`MultiAssetSpecification` retains the generic payoff graph, ordered markets,
model per asset, raw dated correlation matrices and all six tolerances, marginal
LSV factors and particle settings, optional full driver correlations, shared
Hull–White inputs and paired targets, joint Local Correlation endpoints,
calibration settings, retained market-IV provenance, engine and execution policy.
Raw matrices are preserved even when native canonicalization tolerates a small
asymmetry; this also preserves native diagnostics and plan identities.

`StochasticDividendSpecification` combines the full equity specification with
the Buehler reserve model, dividend/volatility correlations, optional rate
correlation and dividend split-grid maximum step. Fixed cash dividend events
remain in the nested market. Direct SV/HW step settings must match the dividend
step. LSV retains its target grid separately. Unused or conflicting inputs are
rejected. No compiled leverage grids, particle states or reverse workspaces are
serialized.

In Rust, construct `PricingTask::new(source, operation)`, using
`PricingSource::from(specification)` or the corresponding multi-asset/dividend
source. `MultiAssetModelSpec::new(models, correlation)` supplies default absent
extensions; set its typed optional configurations before creating the immutable
`MultiAssetSpecification`. Serializers and parsers are
`pricing_task_to_json`, `pricing_task_to_pretty_json`,
`parse_pricing_task_json` and `fingerprint_pricing_task`.

## Python

```python
import rust_pricing as rp

# Construct from a single-asset source, or load any complete task from JSON.
source = rp.PricingSpecification.from_json(specification_json)
task = rp.PricingTask.from_specification(source, operation="evaluate")
status = task.capability("aad")
print(status.status, status.reason, status.detail)

saved_task = task.to_json()
plan = rp.PricingTask.from_json(saved_task).compile()
assert plan.capability().status == "supported"
result = plan.evaluate()
report = result.report
print(report.value, report.standard_error, report.uncertainty_scope)
for risk in report.risks:
    print(risk.name, risk.unit, risk.method, risk.calibration, risk.standard_errors)
saved_report = report.to_json()
restored_report = rp.ValuationReport.from_json(saved_report)
```

Use `task.with_operation("aad", gamma_relative_bump=0.01)` for a basket,
or `task.with_operation("local_variance_risk")` for a supported LSV source.
Use `PricingTask.from_source(source)` with typed single-asset, multi-asset or
stochastic-dividend inputs; see [typed Python construction](typed-python-sources.md).
Python objects are immutable, collection getters return owned snapshots, and
compilation/evaluation release the GIL. The [stub](../../rust_pricing.pyi) freezes
public signatures; JSON errors identify the task or report document kind.

## Risk and uncertainty meaning

`ValuationReport` includes task and plan fingerprints, library version, platform,
currency, price and standard error, estimator, independent sampling units,
evaluated paths, calibration seeds, uncertainty scope, typed risk blocks and
[retained calibration diagnostics](calibration-reports.md).
Each block has a name, ordered underlying IDs, Cartesian coordinate axes,
row-major values, optional standard errors, a typed unit, native method,
calibration treatment and fixed-input/coordinate convention.

Scalar blocks have no axes and one value. Vector/grid shapes and standard-error
lengths are checked on import. Cross Gamma is the native unsymmetrized
`dDelta_i/dSpot_j` matrix, represented by ordered underlying pairs. Density
adjoints without exposed native grids use an explicit `native_parameter_index`
axis. The saved task and native result provide the remaining model context.

Raw volatility derivatives are per unit volatility; market-scaled Vega is per
one volatility point. Local-variance derivatives are per unit **variance**.
Discount-curve adjoints are per log discount factor; DV01 is the signed change
for a +1bp continuously compounded zero-rate pillar. Spot-scaled Delta/Gamma
retain native 1% conventions. These quantities are never silently interchanged.

Calibration treatment describes differentiation, independently of uncertainty:
`recalibrated` includes the native finite-particle pullback with fixed random
inputs; standard LSM risk records `fixed_exercise_policy`. Price/risk sampling
errors for LSV or joint calibration remain **conditional on the realized
calibration**. They exclude calibration noise and discretization/kernel bias.
LSM sampling errors are conditional on its learned exercise policy.

An absent standard-error array means **unavailable**, not zero. For example,
several native MC calibration adjoints have no retained covariance; RQMC may
provide errors across independent scramble estimates. The report copies native
estimates and available errors without recomputing sampling reductions.
The existing typed `vega_kt` payload retains its covariance and diagnostics.

Native Hull–White and stochastic-dividend AAD results also carry ordered
`RiskParameterKind` values in Rust (`parameter_kinds`) and expose
`parameter_kinds` / `parameter_units` snapshots in Python. These align with
`parameter_labels` and `derivatives`. Units come from the native differentiation
coordinates, never from the spelling of the display labels. The report rejects
inconsistent metadata lengths. Explicit model-parameter sensitivities retain
their native unit convention; vol-of-vol is not market-IV Vega.

Failures expose [structured Python error metadata](python-errors.md), including
the selected task operation and a capability reason for source-preflight rejection.

The report JSON is a normalized reporting contract. It is not a serialization
of every native diagnostic or a compiled plan. Selected retained calibration
diagnostics are copied from the compiled plan into the common report. Additional
diagnostics remain on native plans/results and are recovered by executing the saved task.
For a single-asset task, that native value is the existing `SpecificationResult`
(and its own `native_result` in Python); other task variants return their native
price/risk object directly.

## JSON versions and replay identity

The independent document kinds are `pricing_task` v1 to v3 and
`valuation_report` v1 to v3. A writer emits the oldest version that can express
the document, so nothing written before a version existed changes: a task is v3
only when its operation is `measures`, and a report is v3 only when a row
carries a typed measure. The current
[task schema](../../schemas/task-v3/pricing_task.schema.json) and
[report schema](../../schemas/report-v3/valuation_report.schema.json) accept
every version the writer emits (task v1 to v3, report v2 and v3) and are exported
by `pricing_task_json_schema()` / `valuation_report_json_schema()` in Python
and `current_pricing_task_schema()` / `current_valuation_report_schema()` in Rust.
Schemas describe the current writer. The report reader also accepts v1 and
normalizes it to v2, retaining unavailable calibration diagnostics as an absent
field. Existing v1-only report consumers must upgrade for newly written reports.
Nested single-asset source readers also
accept specification v1 and normalize it to v2.

Optional object fields are omitted; ordered optional configurations use
`{"type":"absent"}` or `{"type":"present","value":...}`. JSON `null`, duplicate
keys, unknown fields and nonfinite numbers are rejected by the existing bounded
parser. Rust additionally validates shapes, values and model consistency.

Rust callers needing the input version on a report failure can use
`parse_valuation_report_json_with_context(input, limits)`. Its `WireParseError`
exposes `schema_version() -> Option<u32>` and the original `wire_error()`;
`into_wire_error()` recovers that error by value. The version is absent when
bounded JSON validation or envelope decoding fails. Existing
`parse_valuation_report_json` keeps its `Result<ValuationReport, WireError>`
signature, messages and error variants. Both entry points run the same parser
and preserve the same v1-to-v2 migration.

The task fingerprint uses the domain of the emitted version (`pricing/task/v1\0`
to `pricing/task/v3\0`) and covers the complete
source **and operation**, including Gamma bump, the ordered measure list, calibration seeds, worker count
and reduction-block size. JSON whitespace/key order do not affect identity.
The report links task and native plan identities. Common plan fingerprints use
`blake3-256:<hex>`; basket native results keep their existing bare-hex spelling,
with identical hash bytes. A price task and risk task can share a native plan
fingerprint while having different task fingerprints.

Legacy request/result v1–v3, specification v1/v2, graph v1, their fingerprint
domains and numerical outputs are unchanged. A legacy reader rejects the new
document kind. Exact replay still follows existing version/platform/toolchain
and execution-policy guarantees; the format does not claim cross-platform bit
identity.

## Extension and validation

Add a typed source variant or operation, strict versioned wire conversion,
explicit compiler dispatch, capability rule and report adapter together.
Keep model dispatch outside path loops. Reuse native preflight for both direct
and common evaluation; do not duplicate a permissive support table. Add native
conformance/replay, fingerprint sensitivity and rejection tests for each
composition. A changed persisted numerical meaning requires an explicit schema
and migration decision.

Verification covers [Rust conformance](../../crates/pricing/tests/pricing_task.rs),
[Python contracts](../../tests/python/test_pricing_task.py),
[offline schemas](../../scripts/check_task_json_schema.py), existing numerical
regression tests, wheel/stub validation and source-archive inclusion. Small
fixtures exercise contracts and deterministic replay; they are not accuracy
acceptance panels. Existing statistical and extended-model accuracy gates remain
in CI without relaxed tolerances or exclusions.

Contract-generated lazy selection, exact comparisons, logarithms and exponentials
use extended graph v2 / specification v3 / task v2 documents. Legacy sources
retain their existing emitted versions and fingerprint domains. Current schema
accessors include version guards for both forms; see [Contract MC integration](contracts.md#native-mc-compilation).
