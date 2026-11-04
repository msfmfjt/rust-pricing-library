# VegaKT hedging backtesting

Status: P0 European terminal market-IV risk passes the
[general-smile acceptance v0.3](../../design/validation/vega-kt-smile-acceptance-v0.3.md).
The gate passes 1,077 checks (770 smile and 307 fixed-market checks). The six
legacy comparisons retain their original failed observations and are explicitly
superseded by comparisons with identical market shocks. Explicit four/six-step scopes also cover dated Asian and fixed Lookback
Call/Put products with accepted variance extensions. Arbitrary markets and
historical hedge effectiveness remain outside the accepted scope. The
[joint-risk and P&L extension](../../design/validation/vega-kt-portfolio-market-v0.1.md)
records the additional declared-market checks separately.

Use `RustPricingProvider(numerical_scope="p0_terminal_replication")` or CLI
`--validated-p0` for this scope. Every live contract must use an accepted native
terminal method, have its exact expiry in the reporting basis, and pass a market
price calibration check of `6 * price_SE + 0.002` per unit. Outputs record
`numerically_validated_p0`. An inconsistent input surface stops replay.
The adapter continues to reject construction without an explicit mode.
`diagnostic_only=True` / `--diagnostic-only` retains `unvalidated` status.
A completed ledger alone is not numerical acceptance.

An additional explicit scope, `RustPricingProvider(numerical_scope="multi_step_gamma")`
or CLI `--validated-multi-step`, connects the conditional Gamma chain to the
European ledger. It accepts four/six zero-carry intervals, five quotes at each
two-step reporting maturity, a checked triangular calibration, and terminal
Call/Put contracts. Precision failures stop replay before risk or orders are
released. `provider.gamma_audit(snapshot, contract)` retains the calibration and
same-contract covariance audit. The result status is
`numerically_validated_multi_step_european`; see the
[scope and acceptance record](../../design/validation/local-vol-multi-step-provider-v0.1.md).
Asian/Lookback observation and settlement support remains outside this ledger scope.
The separate `multi_step_path_gamma` scope (`--validated-path`) accepts dated
Asian/Lookback contracts, immutable fixing histories and observation-based
settlement. [Fixing-day Lookback valuation](../../design/validation/local-vol-lookback-fixing-day-v0.1.md)
freezes the known observation independently of spot and retains the existing
[history and settlement contract](../../design/validation/local-vol-path-ledger-v0.1.md).


General smiles use `terminal_payoff_replication_iv_v1`: the payoff's signed
terminal call weights are multiplied by continuum vanilla Vega before the
transpose IV interpolation, as in paper equation (9). This market risk is exact;
sample variance, price covariance and bucket SE are not applicable and remain
absent, rather than manufactured as zero estimates. Eligible flat zero-carry
requests retain `flat_variance_terminal_measure_pathwise_v1` and their actual
sample statistics. Other payoffs retain diagnostic local-node allocation.
See the [earlier allocation audit](../../design/validation/vega-kt-backtest-allocation-audit-v0.1.md)
for its limitations. Each instrument retains its actual `Risk.native_policy`.

Fixed market shocks are enabled explicitly through `market_shock_log_moneyness`.
Both held and lot-rounded proposed positions retain residual-risk audits.
The configured shock definition is saved and checked on resume and aggregation.
A grid-refinement comparison must preserve both baseline IV interpolation and
market shock directions; see the [v0.2 gate](../../design/validation/vega-kt-market-shocks-v0.2.md)
and [daily integration](../../design/validation/vega-kt-market-shock-replay-v0.1.md).

The companion package in [backtesting](../../backtesting/README.md) uses
`rust_pricing` for price, Delta and Local Volatility VegaKT. Python manages the
explicit daily schedule, positions, funding, orders, fills and cash ledger.
Pricing wire formats are unchanged. A native projection bug has been corrected;
rebuild the native library to use that correction.

## Run the example

After building/installing the native extension using the
[development instructions](../../README.md#development), install the companion:

```shell
python -m pip install ./backtesting
python examples/python/vega_kt_backtest.py --paths 2048 --validated-p0
```

Without installing the companion, set `PYTHONPATH=backtesting` when running from
the repository root. The [example](../../examples/python/vega_kt_backtest.py)
constructs an 11-session synthetic eSSVI history, one short target and three
hedge candidates spanning two expiries. Option quotes are model prices plus
synthetic spreads; the example is an integration check, not a historical study
or evidence that one hedge strategy is better. Earlier synthetic P&L outputs
also used the defective native buckets and must not be used to rank strategies.

The output directory is `benchmark-results/vega-kt-backtest`. Each of
`unhedged`, `delta`, `scalar_vega` and `vega_kt` has:

- `input.json`: all input snapshots, canonical pricing requests, contracts,
  entry positions and effective configuration;
- `result.json`: manifest, ledgers, orders, fills, risks, solver decisions,
  summary and resumable state;
- eleven CSV tables for inspection, including `settlements.csv` for fixing/payment
  events, `pnl_components.csv` for reconciled daily and cumulative P&L, and
  `cost_basis.csv` for realized/unrealized P&L, `greek_pnl.csv` for spot-risk explanations,
  and `iv_alignment.csv` for fixed-strike IV alignment diagnostics;
- `checkpoint.json`: state hash, input-prefix hash and configuration identity.

Replay one result without rebuilding synthetic inputs:

```shell
python -m rust_pricing_backtest benchmark-results/vega-kt-backtest/vega_kt/input.json --output benchmark-results/vega-kt-replay --validated-p0
```

`run_backtest(..., stop_after=5)` stops after the fifth snapshot and returns
`partial`. Pass its `checkpoint` to `run_backtest` with the original complete
input schedule to resume. CLI `--checkpoint` accepts the saved file. Processed
history, code, provider binary or configuration changes invalidate the checkpoint.
Unprocessed market values may change; the session schedule remains fixed.

## Provider configuration lifetime

`RustPricingProvider` binds its acceptance policy, execution settings, native
module and identity once, when constructed. Its configuration is held in an
immutable record. Existing constructor keywords and attribute reads remain
available. Assignment or deletion of `numerical_scope`, `worker_threads`,
`reduction_block_size`, `variance_extension`, `variance_basis_policy`,
`sampling_policy`, `diagnostic_only`, `cache_max_entries`, `identity` or `rp`
raises `AttributeError`. Calling `__init__` again on a live provider is also
rejected. The native-module reference is fixed for that provider's lifetime.

To change a policy or execution setting, construct a new provider:

```python
diagnostic = RustPricingProvider(diagnostic_only=True, worker_threads=1)
accepted = RustPricingProvider(numerical_scope="p0_terminal_replication",
                               worker_threads=4)
```

The second provider starts with an empty cache and applies its own acceptance
checks. Diagnostic results cannot acquire accepted status through a setting
assignment. Similarly, changing execution settings cannot reuse evidence made
with the previous execution settings under the same provider identity.

`clear_cache()` releases retained work while preserving all settings and the
provider identity. Its concurrency and snapshot-switch behavior remain as
described below. Numerical/execution changes require a new provider identity and
cannot resume an incompatible checkpoint. Cache capacity alone remains outside
the numerical identity, so separate providers with different capacities can
resume compatible checkpoints. The implementation change itself invalidates
older source hashes under the existing rules.

This changes no ledger/checkpoint schema or numerical acceptance policy.
See [ADR 0038](../../design/adr/0038-immutable-provider-configuration.md).

## Native wheel compatibility

Constructing `RustPricingProvider` now checks the installed native API before
hashing its binary, creating caches or doing numerical work. A matching
`rust-pricing==0.1.0` version alone does not establish compatibility: builds from
different revisions can expose different task/report APIs.

The check covers native classes, result members and the call signatures used by
the selected path, including replayable task serialization:

| Provider path | Required native capabilities |
| --- | --- |
| Diagnostic or P0 | Common task/report valuation, discount-curve evaluation and VegaKT allocation fields |
| Multi-step or path Gamma | Common task/report valuation and the dedicated multi-step Gamma audit |
| Extension or named-basis Gamma | The above plus ordinary plan evaluation and retained samples for joint covariance |
| Quote-control | Dedicated ordinary plan evaluation and paired Gamma samples; unused common task/report APIs are not required |

Missing classes/members or incompatible arguments raise
`BacktestError(code="unsupported_native_version")`, with the detected native
version and API paths that failed. For example, an older wheel lacking
`PricingTask` is rejected at construction, instead of raising `AttributeError`
after a backtest has started. A missing/unloadable native dependency raises
`BacktestError(code="native_import_failure")` and preserves the original import
error as its cause. Install or rebuild the native wheel from the same source
revision as the companion package, using the correct Python ABI. An unavailable
native binary file also fails construction because provenance cannot be bound.

The companion package remains importable without the optional native wheel.
Compatible extra APIs and optional arguments are allowed. These structural
checks do not run a sample trade, certify numerical accuracy or replace the
wheel's full compatibility tests. Request-specific report shapes, calibration,
Gamma recovery, covariance and precision gates still run during evaluation.
No numerical method, tolerance, accepted scope or serialized schema changes.
See [ADR 0039](../../design/adr/0039-native-provider-compatibility.md).

## Common task/report evidence

The ordinary native valuation in `RustPricingProvider` runs through
`PricingSpecification.from_request` and `PricingTask`, then converts the
`ValuationReport` into the ledger's `Risk`. The conversion checks currency,
underlying identity, scalar shape, raw units, calibration semantics and
pricing-sampling uncertainty. Existing request, product, Local Volatility,
fixed-grid Delta, reporting-coordinate and numerical acceptance checks still
apply. This bridge does not extend the ledger to every model supported by the
[common task API](pricing-tasks.md).

```python
risk = provider.evaluate(snapshot, contract)
evidence = provider.pricing_evidence(snapshot, contract)
if evidence is not None:
    task_json = evidence.task.to_json()
    report_json = evidence.report.to_json()
    replay = evidence.task.compile().evaluate().report
    assert replay.to_json() == report_json
```

`PricingEvidence` is an immutable record containing the actual task, its full
common report, `payment_date`, `payment_scaling_factor` and `scope`. The accessor
uses the same acceptance path and cache entry as `evaluate`; a failed valuation
does not release evidence. Cache hits do not compile another task. The evidence
contains source and output, with no retained compiled plan. Native warnings and
Gamma bump diagnostics remain available to the ledger conversion.

| Provider path | Evidence scope | Risk coverage |
| --- | --- | --- |
| Diagnostic or accepted P0 | `requested_risks` | Ordinary price/Greeks and native VegaKT, including its available covariance and projection metadata |
| Accepted multi-step Gamma | `ordinary_price_and_greeks` | Ordinary price/Greeks; accepted market-IV Gamma-chain buckets and same-product covariance remain in `gamma_audit` |
| Quote-control sampling policy | `None` | Dedicated paired-sample report in `gamma_audit`; no substitute common task is evaluated |

Common reports are the unadjusted native outputs. For deterministic delayed
payment, multiply monetary values, raw Greeks and their standard errors by
`payment_scaling_factor` to obtain the corresponding ledger values; multiply
covariances by its square. Coordinates, implied volatilities and forward levels
are not scaled. The task itself remains independently replayable at its original
payoff date. Per-contract reports never establish cross-instrument covariance;
the existing `joint_market_risk` path and its gates remain authoritative.

Evidence is opt-in in-memory output, not a new ledger or checkpoint field.
Existing serialized schemas and `RiskProvider` implementations are unchanged.
See [ADR 0037](../../design/adr/0037-backtest-task-report-bridge.md).

## Provider cache lifetime

`RustPricingProvider` retains one active snapshot, identified by its complete
content hash rather than its ID or date alone. Starting a valuation on another
snapshot replaces the cached risks, task/report evidence, Gamma audits, shared-surface guard and
quote-control native samples. Returning to an earlier snapshot recomputes its
risks and audits using the usual numerical gates. Within the active snapshot,
market/reporting-surface consistency is checked before reuse and again before a
new result is published, including when contracts are evaluated concurrently.

The optional constructor keyword `cache_max_entries=128` limits each of two LRU
caches: risk/task-evidence/Gamma-audit entries, and native quote-control audit/sample
entries. It must be a positive integer. A risk, its evidence and audit are evicted together;
entry eviction does not discard the active snapshot's shared-surface guard.
This bounds retained entry counts, not bytes or temporary numerical workspaces.
Large grids and sample matrices still require memory proportional to their size.
Small capacities can cause repeated native evaluations, including the virtual
quote control. Eviction does not truncate samples, change their pairing or bypass
acceptance gates. Cached rejected native audits still reject; evicted ones are
recomputed and checked again.

Call `provider.clear_cache()` to release retained state when a run finishes or
before reusing the provider for another batch. Existing returned risks and
evidence and detached audits remain valid. In-flight calls keep their own snapshot state until
completion and cannot refill a replacement cache. Independent histories should
use separate providers to avoid displacing one another's cached work.

Cache contents are never serialized, and cache capacity is not part of the
numerical provider identity. Fresh providers and different capacities can resume
the same checkpoint under the existing implementation/input compatibility rules.
The implementation change itself still invalidates older implementation hashes.
See [ADR 0036](../../design/adr/0036-backtest-provider-cache.md).

## Failure reports and recovery

Expected market, risk-quality, solver and native pricing errors stop replay.
By default, `run_backtest` raises `BacktestRunError`, a `BacktestError` subclass,
whose `result` contains the failed report. To receive that report directly:

```python
result = run_backtest(config, history, contracts, portfolio, provider,
                      on_error="return")
result.write("results")
if result.status == "failed":
    print(result.failure)
```

`on_error` controls how the failure is delivered; both settings stop at the same
snapshot. `compare_strategies(..., on_error="return")` retains each strategy's
status so one failure does not suppress the other reports. A failed/partial
strategy must not be ranked against a completed strategy as if coverage matched.

Default result schema version 10 retains `failure` (null on success). It records the error
code/message, attempted snapshot/time, last completed time, remaining cash and
positions, unpaid claims and pending order. The reader also accepts version 1–9
results. New requests use version 6 for the explicit constraint mode and gross cap.
Request versions 1–5 retain execution-only cash checks. Version 1–4
requests have no external flows. Version 1–3
requests retain disabled numeraire reporting; versions 1/2 also retain their
original payment-on-expiry rule.
Checkpoint format is version 8, including applied capital IDs, cost basis, Greek explanations, IV alignment and budget audits.
With explicit market shock nodes, request/result/checkpoint versions are 7/11/9;
these add the configured basis, policy and mandatory daily risk audits. The
result reader accepts versions 1–11. Omitted nodes preserve the default formats.
Earlier checkpoints cannot be resumed; the
implementation and provider hashes also change when code or the native binary changes.

Each simulated snapshot commits only after carry, external capital, settlement, fills, valuation, P&L reconciliation
and the next decision all succeed. A failed snapshot rolls back these changes,
including changes to a previously pending order. Saved tables and the checkpoint
therefore end **before the failed snapshot**, not partway through that day.
Target entry premiums/fees belong to initialization, before the first snapshot;
even a first-snapshot failure retains them and resumes without entering twice.
Rollback applies to the replay ledger; providers must remain deterministic for
the same input, as their external state is not rolled back.

Unprocessed dates have no ledger rows. With zero completed snapshots, P&L and
valuation metrics are null, not zero. Summaries include status, coverage and
last completed time. Terminal hedging error is null for failed and partial runs.
The CLI writes the normal JSON/CSV/checkpoint files on an expected replay failure
and exits **1**. It exits 0 on completed replay; numerical acceptance is separate.

After correcting the failed and subsequent market snapshots, use `--checkpoint`
or the Python checkpoint argument to retry the failed date. Processed inputs,
the calendar, contracts, configuration and provider identity must still match.
Calendar/configuration fixes require a new run. Invalid initial inputs, invalid
checkpoints, provider construction errors and unexpected programming exceptions
raise without a replay report; `on_error="return"` does not suppress those errors.
The [failure-recovery validation](../../design/validation/vega-kt-backtest-failure-recovery-v0.1.md)
records the tested boundaries and remaining limitations.

## Input and execution contract

Use `Contract`, `Position`, `Quote`, `RiskBasis`, `SettlementFixing`, `PaymentDiscount`, `Snapshot` and
`BacktestConfig` from `rust_pricing_backtest`. The entry point is:

```python
provider = RustPricingProvider(diagnostic_only=True)
result = run_backtest(config, history, contracts, portfolio, provider)
result.write("results")
```

Contracts are European, cash-settled, single-asset and single-currency.
`payment_date=None` means payment on expiry; an explicit date may be on or after
expiry. Price and Greeks in each native request are per unit
(`notional=1`); the runner applies signed quantities and contract multipliers
exactly once. `entry_price` is supplied explicitly and need not equal the initial
model mark. Its difference and any entry fee are included in day-one P&L.

Snapshots have one timezone-aware timestamp per valuation date, separate
observation/availability timestamps on quotes, surface availability and source.
The caller supplies the session calendar. A live position needs an observation
on its expiry date; the runner rejects settlement using a later day's spot.

Each contract selects its settlement convention:

| Contract fields | Behavior |
| --- | --- |
| `settlement_rule="snapshot_spot"`, `settlement_fixing_id=None` | Use expiry-snapshot spot mid. This compatibility default is recorded as an assumption. |
| `settlement_rule="explicit_fixing"`, `settlement_fixing_id="close"` | Require `snapshot.settlement_fixings["close"]` on expiry, with no spot fallback. |

`SettlementFixing` contains `fixing_date`, `value`, timezone-aware `observed_at`
and `available_at`, plus nonempty `source` and `revision`. `value` is the
underlying's settlement price, not the option's payoff; it must be finite and
nonnegative. The runner applies call/put payoff, signed quantity and multiplier.
Several contracts can reference the same fixing ID. Spot quotes still price
stock trades and stock marks, so an official fixing need not equal spot mid.
The synthetic example uses an explicit `synthetic_close` fixing; it is not a
real market data record.

Every supplied fixing must already be available at the snapshot time, even if
unused, before the snapshot reaches a risk provider. On expiry, the requested
fixing date must equal the contract expiry; its observation must also fall on
that date **in the snapshot timestamp's timezone**. Missing, future or wrong-date
fixings fail the snapshot and use the same rollback/recovery path as other
market errors. A pending order that expires without a fill creates no payoff
and needs no fixing for that unheld contract.

Settlement cashflow rows retain the rule, underlying price, unit payoff,
quantity and multiplier. Explicit-fixing rows also retain the complete fixing,
its ID and hash; snapshot-spot rows retain the source and spot quote. The manifest
lists `snapshot_spot_settlement_contracts`. Provenance is supplied by the caller;
these fields do not establish that a series was historically available or that
its source is an exchange. Changing a processed fixing invalidates the checkpoint.
A later revision does not rewrite an already paid cashflow. Explicit correction
cashflows, overnight fixing publication and averaging/auction-specific payoff
models remain outside this slice. The input pricing
model must be appropriate for the contractual point fixing; a settlement record
does not change the native pricing model or add intraday maturity support.

Version 2–5 requests require both settlement contract fields and a
`settlement_fixings` object in every snapshot (empty when none). Version 1 inputs
must omit those fields and are interpreted with the legacy snapshot-spot rule.
See the [settlement validation record](../../design/validation/vega-kt-backtest-settlement-v0.1.md).

## Payment dates and unpaid claims

On expiry the runner fixes the signed payoff and removes the option position.
If payment is later, it retains a separate claim in `receivables` until the
payment date. Each target/hedge claim remains identifiable even when their net
value is zero. The payment date must be present in the supplied schedule; the
runner never pays against a later snapshot. A zero payoff leaves no unpaid claim.

At each intervening valuation supply `snapshot.payment_discounts[payment_date]`,
a `PaymentDiscount(valuation_date, factor, observed_at, available_at, source,
revision)`. Date keys are ISO dates in JSON. Factors must be finite and positive;
values above one are permitted. The valuation date must equal the snapshot date,
availability must precede the snapshot, and observation age uses
`max_quote_age_seconds`. A same-day factor, if supplied, must equal one. No
interpolation, funding-rate substitution or stale-factor fallback is applied
to these explicit payment-date factors.

The daily equity includes `receivable_value = sum(amount * factor)`. Gross
claims and their discount provenance are saved in `ledger.receivables`; they
are not counted again as option value. Funding applies only to actual cash.
On payment, cash changes by the fixed amount and the claim is removed. All
fixing/payment events appear in `settlements.csv`; only actual payments appear
as settlement cashflows. Payment cash limits and missing-factor failures use
the same atomic snapshot rollback. A completed replay with unpaid claims has
`fully_settled=false` and no terminal hedging error.

For a delayed contract, the native LV adapter prices the existing expiry-paid
vanilla and multiplies price, Delta, bucket risks, projection diagnostics and
standard errors by `D(payment)/D(expiry)` from that request's deterministic
valuation curve. The new public `DiscountCurve.discount(time)` method evaluates
the existing Rust log-linear curve (including its flat-forward extrapolation);
time is ACT/365F. The supplied payment factor must agree with the native curve.
`Risk.payment_date`, `payment_scaling_factor` and provenance record this
adjustment. Other providers must explicitly report the contractual payment date
for delayed contracts; returning an unlabelled expiry-paid price is rejected.
This adjustment holds the valuation curve fixed for equity/volatility risks;
it does not add rate hedging, stochastic-rate models or default risk.

Version 3–5 requests require `payment_date` on contracts (null means expiry) and
`payment_discounts` on snapshots (an empty object when unused). Version 1/2
requests must omit these fields and retain payment on expiry. New result schema
10 and checkpoint schema 8 retain unpaid state, capital IDs, P&L attribution, cost basis, Greek explanations, IV alignment and budget audits across resume.

Run a two-calendar-day lag example, extending the synthetic weekday schedule
through all payment dates:

```shell
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --output benchmark-results/vega-kt-delayed-payment-example
```

The example rejects lags that put a payment on a weekend; it does not invent a
business-day adjustment. See the [payment validation record](../../design/validation/vega-kt-backtest-payments-v0.1.md).

## Orders, carry and execution

Orders use the current snapshot and are filled at the **next supplied snapshot**.
Historical EOD data therefore means next-session EOD execution. Buys use ask,
sells bid, with configured slippage and per-unit fees. These are execution-model
assumptions. Quotes appearing only after a decision are not eligible at that
decision. Missing or stale quotes for held/filling positions cause a typed error.
The slice implements `fail_run`, not a silent fallback to old Greeks.

Carry accrues first using prior-snapshot rates and ACT/365F elapsed days:
cash interest compounds continuously, stock borrow uses prior spot and simple
accrual. A replay fixture may supply same-day cash dividend entitlement/payment
per held share. **The native pricing adapter only accepts zero-dividend inputs**;
physical cash-dividend quote conversion and delayed dividend payment are not implemented.

After carry, external capital movements book, expiring target and hedge options fix,
due payments settle, previous orders fill,
and positions are marked. The next order is then decided. Default end policy
liquidates remaining hedges and stock using last-snapshot bid/ask under a
precommitted terminal rule, exempt from ordinary per-rebalance trade limits.
It does not invent a sale price for an unexpired target. `mark_to_market` retains
all positions. Terminal hedging error is populated only when the portfolio is
fully settled; a completed replay may still end with open positions.

Cash constraints apply to initial premiums, net carry/capital/settlement and proposed fills.
An unaffordable rebalance basket is rejected atomically, with its reason in the
order log. Terminal liquidation that breaches the cash floor raises an error.
The default `constraint_mode="execution_cash"` preserves the original box/L1
optimizer and execution-time cash check. Enable the joint solver below to include
cash and gross notional when choosing option quantities.

## Decision-time cash and gross constraints

Install `python -m pip install "./backtesting[constraints]"`, then configure:

```python
config = BacktestConfig(
    constraint_mode="decision_cash_gross",
    gross_notional_limit=2000.0,
    cash_floor=-1000000.0,
)
```

These are illustrative currency limits. The gross definition is fixed by
`decision_quote_cash_gross_delta_neutral_qp_v1`:

```text
gross = current_spot_mid * (
    sum_live_targets abs(quantity) * multiplier
  + sum_live_hedges abs(quantity) * multiplier
  + abs(stock)
)
```

Quantities net within each contract/book; target and hedge books do not offset
one another. Expired options and fixed unpaid claims are excluded. This is an
underlying-equivalent gross exposure cap, not option premium value, a risk-weighted
exposure or a collateral/margin model. An explicit finite nonnegative cap is
required; supplying a cap with the legacy mode is rejected.

For VegaKT/scalar Vega the objective remains
`||.01*(v+A*h)||² + ridge*||h||² + turnover_penalty*sum(cost_j*abs(h_j-h_prev_j))`.
The scalar strategy sums buckets before optimization. Stock is eliminated as
`n = -target_delta - option_deltas @ h`. Position/trade/short boxes, the stock
bounds, cash floor and gross cap therefore constrain the same continuous solve.
Stock trade costs enter cash; the turnover objective retains its option-only
penalty convention. Quotes, Greeks and available cash are from this decision
snapshot; future funding, capital, settlements and prices are not anticipated.

Decision cash uses the exact current bid/ask execution formula including slippage
and per-contract/per-share fees. Absolute holdings and trades use linear
epigraphs. A HiGHS feasibility LP precedes the SciPy SLSQP quadratic solve.
An independent nonnegative-multiplier KKT check gates acceptance: primal,
stationarity and complementarity residuals must each be at most `solver_tolerance`.
Rows are divided by `max(1, max(abs(row)))`; the objective is divided by the
recorded `objective_scale`. Residuals are in these numerical scales, not currency.
The solver's status/message/success are also retained, but a success flag alone
does not pass the gate. `projected_gradient` is null for this solver.

Continuous quantities are checked again using actual absolute values. Options
are then lot-rounded; stock is recomputed, clipped to its configured bounds and
lot-rounded. Cash/gross are checked again, and the resulting Delta residual is
recorded. No integer-optimality or rounding repair is attempted. An infeasible
continuous problem, failed KKT check or unaffordable rounded basket fails the
snapshot atomically; it never relaxes limits. An alternative integer basket may
exist even when this rounding policy fails. Delta-only uses the exact continuous
stock target with the same feasibility checks; unhedged only checks its fixed book.

At the next snapshot the whole actual basket is checked again. Changed prices
can produce `rejected_cash_limit` or `rejected_gross_limit`, leaving holdings and
cash unchanged. Terminal liquidation violations fail the run. Held gross is
checked after settlement and fills on **every** snapshot, including dates without
rebalance. Corrective pending fills can restore compliance; a remaining breach
caused by a market move fails and rolls back that snapshot. There is no automatic
margin liquidation or waiting period. Existing pre-trade carry/settlement cash
checks still apply, so an already breached cash floor cannot be repaired by a sale.

Decision/continuous checks allow only 16 binary64 ULPs at the maximum of one and
the absolute compared amounts; actual execution retains the strict cash-floor
comparison. Solver tolerance does not expand financial budgets. Diagnostics retain
decision inputs, option bounds/previous quantities/costs, continuous solution,
rounded estimates/Delta residual and actual proposed execution budgets. SciPy and
NumPy versions participate in checkpoint and ensemble identities. The default
mode does not import or require SciPy. Candidate IDs are sorted; a canonical
tie-break across solver versions for nonunique zero-ridge solutions remains open.

```shell
python examples/python/vega_kt_backtest.py --paths 256 --diagnostic-only --gross-notional-limit 2000 --output benchmark-results/vega-kt-budget-example
```

See the [budget validation record](../../design/validation/vega-kt-backtest-budget-v0.1.md).

## Risk basis and native constraints

The adapter consumes a canonical `PricingRequest` JSON string for each live
instrument in `Snapshot.pricing_requests`. It validates date, underlying,
currency, strike, side, expiry, unit notional, spot, model and requested risk.
All instruments in one snapshot must share the same market and reporting IV
surface. Existing `PricingRequest.from_json` still validates the pricing wire
contract. Inputs must have point-in-time calibration provenance; this package
does not fit raw historical option quotes.

The requested method is `lv_paper_decomposition` in canonical f-IV coordinates.
For P0 terminal Europeans the exact static vanilla hedge is a closed-form case
of the [normative LV decomposition](../models/local-vol-vegakt-calculation-specifications.md).
The accepted mode requires zero dividends and an exact reporting expiry, and
allows independent simulation/reporting grids and additional later quote dates.
The local grid still ends at the instrument expiry. Native price and local Vega
continue to come from MC; IV risk is not the gradient of an arbitrary supplied
local-variance grid. The market calibration check prevents conflating the two.

Diagnostic mode retains the historical adapter grid restrictions: every positive
local time is a reporting maturity, with identical spatial axes. It does not
promote general AD/Gamma recovery to accepted status. Delta uses the existing
frozen-local-variance-grid Spot bump, publicly labelled `sticky_log_moneyness`.
No calibration-gradient risk is silently substituted, and no full Delta/IV P&L
attribution is claimed by this numerical scope.

VegaKT is raw currency per absolute volatility. The solver minimizes
`||0.01(v + A h)||² + ridge ||h||² + turnover_penalty sum(cost |h-h_prev|)`.
`A` includes contract multipliers, and `h` is the resulting total position.
Diagnostics record singular values, numerical rank, condition number (null for
non-full-column-rank matrices), objective and projected-gradient convergence.
Per-contract price/bucket SE, projection residual and edge thresholds are
configurable. Cross-instrument uncertainty is not inferred from independent
bucket SEs. Projection residual is never folded into tradable risk buckets.

Lot rounding snaps within eight binary64 ulps of an integer and otherwise rounds
toward zero, constrained to admissible lots. Held contracts remain in the risk
matrix if excluded from new trading; their positions are fixed. Scalar Vega
uses the nearest-strike eligible option and explicitly closes prior liquid
scalar hedges when rolling. The four strategies share the data, timing, funding
and cost assumptions. Identical-contract hedges are excluded by default and can
be enabled for cancellation tests.

## Reconciled P&L components

`result.pnl_components` and `pnl_components.csv` contain one row per committed
snapshot. All nine components are signed contributions in the portfolio's
currency. A positive number increases equity; fees and execution costs normally
have negative signs. These are exact cashflow/mark-change amounts, subject to
floating-point rounding, rather than Delta/Gamma/Vega approximations.

Let `dV` denote today's value minus the previous committed value, `R` the signed
present value of unpaid claims, and trade quantities be positive for purchases.
The initial marks and claims are zero; initial equity is `initial_cash`.

| Component | Calculation |
| --- | --- |
| `target` | `dV_target + dR_target + target_payments + entry_premium_cash` |
| `option_hedge` | `dV_hedge + dR_hedge + hedge_payments - sum(trade_quantity * multiplier * mid)` |
| `stock` | `dV_stock - sum(stock_trade_quantity * mid)` |
| `dividend` | Cash dividend receipts/payments, including short-stock obligations |
| `funding` | Actual cash interest, positive or negative |
| `stock_borrow` | Negative borrow-charge cashflow |
| `entry_fees` | Negative sum of target entry fees, on the first committed day only |
| `trading_fees` | Negative sum of today's option/stock fill fees |
| `spread_slippage` | `-sum(trade_quantity * multiplier * (execution_price - mid))` across option/stock fills |

`entry_premium_cash` is the signed premium from the original positions, excluding
fees. Target marks follow `target_mark`; hedge and stock marks use mid quotes.
First-day target P&L therefore includes any entry-price/mark difference.
No actual cash is moved by attribution: execution costs are already in the
cash ledger and are separated here only for explanation. Rejected baskets have
no fills and contribute no trading fees or spread/slippage. Terminal liquidation
uses the same formulas. External capital changes cash but does not enter any
of the nine P&L components; see the capital movement convention below.

Unpaid-claim valuation changes remain in their target/hedge component, including
discount accretion and curve changes. They do not enter cash `funding`. A claim
marked at 9.9 before a payment of 10 contributes 0.1 on payment: the payment and
removal of the claim offset. The `details` object retains option-value changes,
claim changes, settlement cash, initial premium and mid-price trade principal;
these fields explain the components and must not be added again to the total.

`total` is the sum of the nine components. It is compared independently to
`ledger_pnl`, the equity change less net external capital. `reconciliation_error = total - ledger_pnl`
is retained even when nonzero; it is never assigned to a component. The
`cumulative` object contains running totals by component. `cumulative_total`
is compared with ledger equity minus initial cash and cumulative net external
capital in the same way. `details.external_net_cashflow` separately audits
booked capital against the input amount; its difference is recorded as
`external_cashflow_reconciliation_error`, with the same daily tolerance.
Capital mismatch raises `external_cashflow_reconciliation` and rolls back the day.
`summary.pnl_components` holds the final committed cumulative components, and
`summary.pnl_reconciliation` records coverage and observed differences.

The manifest freezes `pnl_attribution.method = cashflow_mark_changes_v2` and
accounting tolerances. Daily tolerance is `1e-10 + 1e-12 * scale`, where scale
is the maximum of 1, beginning/end gross accounting balances, absolute daily
cashflows, absolute mid-price trade principal, and absolute entry premium plus
entry fees. Gross balances sum the absolute cash, target, hedge and stock
values plus absolute individual claim values. Cumulative tolerance sums daily
tolerances. These handle floating-point accounting only; they are not pricing
or strategy acceptance thresholds. Each output row records its tolerance and
both differences. A mismatch raises `pnl_reconciliation` and rolls back the
whole snapshot, preserving earlier attribution for exact retry.

With no completed snapshots, component summaries and reconciliation status are
null; no observation is invented. Legacy result versions 1–3 load with an empty
attribution table when absent; the reader does not invent historical components.
Greek explanations, cost-basis and discounted metrics are separate reports
described below. See the
[P&L attribution validation record](../../design/validation/vega-kt-backtest-pnl-attribution-v0.1.md).

## Moving-average realized and unrealized P&L

`result.cost_basis` and `cost_basis.csv` contain one row per committed snapshot.
The manifest fixes `cost_basis_policy.method = moving_average_execution_price_v1`.
Target entries use `Position.entry_price` in portfolio input order; option and
stock fills use actual execution prices, including spread/slippage. Fees remain
separate cash expenses. Target and hedge books are separate even if the same
contract is held in both. A positive quantity is long; signed cost can be negative.

Same-direction additions use the quantity-weighted average price. A partial
close realizes `signed_closed_quantity * multiplier * (execution_price - average_price)`
and retains the average on the remainder. A reversal closes the old position
and opens the excess at the new execution price. Opposite initial entry lots
follow the same rules: changing their input order can change the split between
realized and unrealized P&L, although their total remains unchanged.

Expiry realizes `signed_fixed_payoff - remaining_signed_option_cost` and removes
the option basis. The unpaid claim starts at signed face cost. Its unrealized
P&L is signed present value minus signed face amount, so discount accretion and
curve changes remain visible until payment. Payment removes the claim basis;
the payoff is not realized a second time. Zero payoffs close the option cost
without leaving a delayed claim. This is a research accounting convention, not
a tax-lot or statutory accounting implementation. Realization at fixing does
not mean the payment has arrived.

For one long option entered at 9.5, a fixed payoff of 10 and no other cashflows:

| Event | Realized P&L | Unrealized P&L | Total P&L |
| --- | --- | --- | --- |
| Fixing; payment discount factor 0.98 | 0.5 | −0.2 | 0.3 |
| Unpaid; discount factor 0.99 | 0.5 | −0.1 | 0.4 |
| Payment | 0.5 | 0 | 0.5 |

Each row's `groups` contains `target`, `option_hedge` and `stock`, with cumulative
realized/unrealized P&L, daily realized changes and changes in unrealized P&L,
open/claim cost, market value, spread/slippage allocation and reconciliation errors.
`positions` snapshots signed quantity, multiplier, average price and signed cost
by book/contract; `claims` retains each unpaid claim's face cost and book.
`cash_realized_components` contains cumulative dividend, funding, stock-borrow,
entry-fee and trading-fee contributions. These cash components plus asset realized
P&L produce the top-level `realized_pnl`; asset unrealized P&L sums to
`unrealized_pnl`. `summary.cost_basis` retains the final committed totals and groups.

Spread/slippage is already included in execution cost basis. Therefore each
asset group's total reconciles to its existing mid-price `pnl_components` total
**plus** that group's signed spread/slippage contribution. Do not subtract
spread/slippage again from cost-basis P&L or add this report to `pnl_components`.
External capital is excluded. Funding and fees remain realized contributions,
including when capital movements change funding or feasible trades.

The engine independently checks inventory quantities, signed costs, claim IDs and
face amounts, per-group totals/changes, spread allocation, and the overall
daily/cumulative capital-adjusted nominal ledger P&L. It uses the existing daily
and cumulative accounting tolerances and retains the differences without assigning
them to an asset. Position/claim mismatches or `cost_basis_reconciliation` errors
roll back the entire snapshot, including fills and cost-basis changes. Rejected
baskets do not change basis. Checkpoints retain basis and cumulative realization,
so retries do not duplicate realized gains or losses.

With no committed observations, the table is empty and `summary.cost_basis` is
null. Legacy result versions 1–6 load an absent table as empty; no historical
cost basis is synthesized. Cost basis was introduced in result version 7 and
checkpoint version 5; current result/checkpoint versions are 10/8. The current request is version 6.
These realized/unrealized amounts are
in nominal portfolio currency. Dividing them by the current numeraire is not
the flow-timed discounted attribution described below.

See the [cost-basis validation record](../../design/validation/vega-kt-backtest-cost-basis-v0.1.md).

## Lagged Delta and optional Gamma explanations

`result.greek_pnl` and `greek_pnl.csv` explain the exact asset P&L already recorded
in `pnl_components`. Each interval uses the **previous committed snapshot's**
post-fill holdings and per-contract Greeks, with `dS = current_spot_mid - previous_spot_mid`.
Today's pending fills and terminal liquidation do not receive retroactive exposure
to that interval. The stock term uses the previous stock quantity. The opening
RiskBasis supplies the Delta convention; the native adapter uses frozen local
variance grids. Input Greeks are per unit, so holdings and contract multipliers
are applied once.

For each of `target`, `option_hedge` and `stock`:

```text
delta_pnl = opening_portfolio_delta * dS
gamma_pnl = 0.5 * opening_portfolio_gamma * dS * dS  (when available)
available_greek_pnl = sum of calculated Delta/Gamma terms
unexplained_pnl = actual_asset_pnl - available_greek_pnl
```

The unexplained amount intentionally retains time decay, volatility/curve changes,
dividends' effects on option value, cross/higher-order terms, price/risk noise,
market-versus-model mark differences and payoff-fixing effects. Unpaid-claim
discount changes also remain unexplained. Exact cash dividend, funding, borrow,
entry-fee, trading-fee and spread/slippage components are retained separately in
`non_asset_components`. External capital is excluded by the exact ledger.

`Risk.gamma` is optional, in currency per spot unit squared for one contract unit.
Custom providers must use the same spot convention as their Delta. Negative
finite estimates are retained. `gamma_method` and `gamma_spot_bump` preserve
optional provenance. The native adapter copies requested public `gamma_raw` and
diagnostic method/bump; it does not modify the supplied pricing request to enable
Gamma. A delayed-payment Gamma uses the same deterministic discount ratio as
price and Delta; the spot bump itself is not currency-scaled.

If any nonzero opening option position in a group lacks Gamma, that group's
`opening_gamma` and `gamma_pnl` are null and `gamma_missing_contracts` identifies
the missing inputs. Its entire Gamma contribution stays unexplained; partial
Gamma is not labelled complete. Other complete groups can still contribute Gamma
to the top-level `available_greek_pnl`. Stock and groups with no opening options
have known zero Gamma. A missing Gamma remains null even for a zero spot change.
The initial observation has no interval: Greek values are null and initial
entry-to-mark differences remain unexplained, with entry fees kept separate.

`vega_pnl` remains null with `vega_status = not_computed_risk_and_delta_iv_convention_unvalidated`.
The separate IV audit below matches absolute maturities and fixed strikes; it
does not assume that the native LV decomposition is a market-IV quote gradient.
An accepted IV-shock interpretation consistent with the frozen-grid Delta, plus
validated VegaKT, is still required before adding a Vega term to PNL-04.

Rows retain the interval dates, calendar-day count, spot shock, opening inventory
and risk hashes, risk warnings, asset groups, cash/cost components, cumulative
totals and arithmetic reconciliation differences. `summary.greek_pnl` reports
coverage, intervals with complete Gamma and cumulative available/unexplained/cash
amounts. The manifest fixes `opening_holdings_delta_optional_gamma_v1`.
The available terms plus residual plus exact non-asset components reconcile to
nominal daily/cumulative P&L using the existing tolerances. This identity is not
a test that Greeks are accurate or that the residual is small. It does not
replace the independent, residual-free accounting check in `pnl_components`.
Nonfinite calculations or failed arithmetic checks roll back the snapshot.
With no completed observations, cumulative explanation metrics are null.

Current result version 10 and checkpoint version 8 preserve the report and
opening-risk history. Result versions 1–9 remain readable; an absent explanation
table loads empty without inventing historical Greeks. Optional Gamma is already
supported inside native requests.

To include native Gamma in the synthetic example:

```shell
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --numeraire-rate 0.01 --capital-flow-demo --gamma-relative-bump 0.0001 --output benchmark-results/vega-kt-greek-pnl-example
```

In the current native LV implementation, Delta and Gamma share the spot bump.
For this zero-dividend adapter, relative bump `0.0001` matches the existing
Delta default. A different Gamma bump also changes Delta estimates and can
change hedge quantities and P&L. It is an explicit numerical input recorded in
the requests, not a reporting-only switch or an accepted accuracy threshold.
This spot Gamma does not implement the missing paper VegaKT gamma-recovery pipeline.

See the [Greek explanation validation record](../../design/validation/vega-kt-backtest-greek-pnl-v0.1.md).

## Fixed-strike IV alignment audit

`result.iv_alignment` and `iv_alignment.csv` audit geometry and missing sensitivity
before adopting IV changes for P&L. `Risk.reporting_iv` optionally supplies
row-major canonical IV at the same nodes as its buckets; `reporting_forwards`
supplies one positive forward per absolute maturity. Both must be present or
both absent, with positive finite values and matching shape. Supplied surfaces
from evaluated instruments must agree exactly within a snapshot. The native
adapter reads IV from public VegaKT coordinates and computes `F(T)=spot/D(T)`
with the public valuation discount curve; this adapter requires zero dividends.
IV and forward levels are not multiplied by the delayed-payment scaling factor.

For each opening node `(T,k_previous)`, the audit reconstructs the canonical
strike and moves to the current moneyness coordinate:

```text
K = F_previous(T) * exp(k_previous)
k_current = k_previous + log(F_previous(T)) - log(F_current(T))
dIV = current_IV(T,k_current) - previous_IV(T,k_previous)
```

Absolute maturity `T` must be present exactly in the current grid and must still
be in the future. At that maturity, IV is interpolated linearly between current
log-moneyness nodes; exact nodes use weight one. The report retains source indices
and weights. It performs no maturity interpolation, extrapolation, edge clamping
or nearest-date matching. Expired nodes, missing absolute dates, missing surfaces
and strikes outside the domain have null `delta_iv` with explicit reasons.
Only `canonical_f_iv` is supported; other coordinates produce an unsupported status.

For each target/hedge group, `opening_raw_buckets` sums signed opening quantities
times multipliers times raw instrument buckets. `covered_bucket_dot` is their
dot product with matched IV changes only. `absolute_instrument_buckets` sums
absolute instrument contributions **before** offsetting different contracts;
covered/uncovered absolute risk and its coverage fraction therefore do not hide
missing sensitivity behind net cancellation. `complete_bucket_dot` is null if
any nonzero absolute instrument risk is uncovered. A zero-risk group has a null
coverage fraction and `no_opening_risk` status; its known zero dot is not evidence
that market data was available. Node coverage and risk coverage are distinct.

All dot products remain **diagnostic**. `pnl_adopted` is always false and
`application = diagnostic_only_not_added_to_greek_pnl`. The LV paper decomposition
is not defined as an independent market-quote gradient, and fixed-strike IV
changes have not been made consistent with the native frozen-local-grid Delta.
Blocks also identify unvalidated native VegaKT and incomplete risk coverage.
Even a complete geometric match does not enable a Vega P&L term, reduce the
Greek residual or establish numerical acceptance.

The audit uses surfaces already returned for held or evaluated candidate
instruments. It does not trigger additional native pricing solely to populate
the report. After options disappear, a strategy can therefore have no current
surface; the output records this rather than recycling a stale one. Strategies
can have different surface availability despite sharing input market history.

Rows retain opening/current basis IDs, surface hashes, contributing contract IDs,
node coordinates/status/weights, group dot products and missing-risk amounts.
`summary.iv_alignment` reports observation/interval counts and node-match coverage.
The manifest fixes `absolute_maturity_fixed_strike_iv_audit_v1`. Inconsistent
supplied surfaces or nonfinite arithmetic fail and roll back the whole snapshot.
Missing optional surfaces or incomplete geometric coverage are audit statuses,
not failures of the existing price/cash ledger. Future changes cannot alter
earlier audit rows. The audit table was introduced in result version 9 and
checkpoint version 7; current versions are 10/8. Older results load an absent
audit as empty. Surface metadata is a provider output.

Run the synthetic audit using the existing Delta bump:

```shell
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --numeraire-rate 0.01 --capital-flow-demo --gamma-relative-bump 0.0001 --output benchmark-results/vega-kt-iv-alignment-example
```

See the [IV alignment validation record](../../design/validation/vega-kt-backtest-iv-alignment-v0.1.md).

## Common numeraire and discounted hedging error

Select a common money-market account with `BacktestConfig(numeraire_id="cash_account")`.
Each snapshot then requires `numeraire=NumeraireObservation(...)` with the same
`account_id`, the portfolio's `currency_id`, a session `valuation_date`, a
strictly positive finite `value`, `observed_at`, `available_at`, `source` and
`revision`. The observation must already be available and no older than
`max_quote_age_seconds`. All supplied observations are checked before provider
access, even when reporting is disabled. Missing required observations or
mismatched dates/accounts/currencies stop and roll back that snapshot.

The caller supplies values of one consistently based, accrued money-market
account `B_t`. Values may fall under negative rates. The runner preserves their
scale: `B_0` need not equal one, and the path must not be rebased between sessions.
It records the supplied provenance; that metadata alone does not verify the
economic construction of an external account series. Borrowing/lending rates
used by a strategy and payment-date discount factors remain separate inputs.
The runner does not infer a numeraire from either. With `numeraire_id=None`
(default), all discounted outputs are null.

Let `E_initial` be `initial_cash` before target entry at the first session,
`B_initial` the account value at that same session and `F_i` signed external
capital (deposit positive). The ledger records:

```text
discounted_equity_t = E_t / B_t
discounted_daily_pnl_t = E_t / B_t - E_previous / B_previous - sum_today(F_i / B_i)
discounted_cumulative_pnl_t = E_t / B_t - E_initial / B_initial - sum_to_date(F_i / B_i)
```

For the first daily row, the previous state is the initial cash and first
account value. Entry-price differences and fees enter that day's P&L once.
The units are portfolio value per supplied account unit. Multiplying every
account value by a constant divides all discounted values by that constant;
compare strategies only with the same account path and scaling.
`compare_strategies` shares the supplied path; manifests retain its full
`numeraire_policy.input_hash`, account and currency IDs, method and normalization.
Adding this reporting does not change orders, fills, cash or nominal P&L.

`summary.discounted_pnl` contains the last committed cumulative P&L, starting
and final account values, observation count and reconciliation evidence.
Its `terminal_hedging_error` is populated only for a completed, fully settled
run. Partial, failed or still-open runs keep it null. With no committed rows,
the entire discounted summary is null. The existing nominal terminal error
continues to be reported separately. No settlement or trade cash is subtracted
again from discounted equity. Each external flow is converted at its own booking
snapshot, not at the final account value. `discounted_external_net_cashflow`
and `discounted_cumulative_external_cashflow` retain these adjustments in the
ledger; `summary.discounted_pnl.external_net_cashflow` contains their cumulative sum.

`pnl_components.csv`/JSON add a nested `discounted` explanation:

```text
pnl_at_current_numeraire = nominal_component_total / B_t
numeraire_carry = E_previous / B_t - E_previous / B_previous
discounted_total = pnl_at_current_numeraire + numeraire_carry
```

The carry term is necessary even when nominal P&L is zero. For example, equity
100 with `B` changing from 2 to 4 has discounted P&L `25 - 50 = -25`.
It is a change of measurement, not an extra cash charge. This two-part sum and
its cumulative sum independently reconcile to the discounted ledger. Daily
tolerance is nominal tolerance divided by `B_t`, plus `1e-10 + 1e-12 * scale`,
where scale is the maximum of 1, absolute current discounted equity, and absolute
previous equity converted at the previous/current account values. Cumulative
tolerance sums daily tolerances. Differences remain visible; nonfinite values
or mismatches stop the snapshot with `nonfinite_discounted_pnl` or
`discounted_pnl_reconciliation`.

Request versions 4–6 require `config.numeraire_id` and each snapshot's `numeraire`
(null when unused); request versions 1–3 must omit both. Old results remain
readable without inventing missing discounted fields. Checkpoint format is
8; changed source/input hashes also invalidate checkpoints from older implementations.
Changing an already processed account observation invalidates resume; repairing
an unprocessed observation is permitted under the normal recovery rules.

The synthetic example can create an explicit constant-rate account using
calendar-day ACT/365F continuous accrual, starting at one:

```shell
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --numeraire-rate 0.01 --output benchmark-results/vega-kt-numeraire-example
```

See the [numeraire validation record](../../design/validation/vega-kt-backtest-numeraire-v0.1.md).

## External investor capital

Supply `snapshot.external_cashflows` as a sequence of `ExternalCashflow` records
(default empty). Each record has a run-wide unique `flow_id`, timezone-aware
`effective_at` and `available_at`, a nonzero finite signed `amount`, portfolio
`currency_id`, `source` and `revision`. Positive amounts are investor deposits;
negative amounts are withdrawals. Trading principal, settlement, dividends,
funding and fees retain their existing classifications and are not external capital.

`effective_at` must equal the snapshot timestamp exactly and `available_at`
must not be later. This daily engine does not interpolate intraday capital,
accrue partial-day interest or silently move a skipped cashflow to the next day.
The caller must supply a complete capital history. Empty input explicitly means
no capital movement at that snapshot; completeness is not inferred externally.

The order at each snapshot is: validate inputs; accrue the preceding interval's
funding/stock-borrow/dividend; book external capital; fix options and pay due
claims; check the cash floor; execute pending trades; value and reconcile.
Incoming capital starts earning/paying interest in the following interval.
It can fund same-snapshot settlements and pending hedge fills. First-snapshot
capital books after target-entry initialization and cannot rescue an initial
premium cash-floor failure; use sufficient `initial_cash` for entry.

Capital records at the same timestamp form a simultaneous net batch. Every
record is retained in deterministic `flow_id` order, but no artificial
intermediate cash limit is applied between these simultaneous movements.
The existing cash-floor check applies after net carry, capital and settlements,
before trades. A withdrawal exceeding that budget fails the snapshot; it is
not automatically funded by selling assets. Funding and later cash constraints
can change when capital is changed, so strategy P&L need not remain identical.

```text
daily_pnl_t = E_t - E_previous - external_net_cashflow_t
cumulative_pnl_t = E_t - initial_cash - cumulative_external_net_cashflow_t
```

All nominal P&L statistics and the terminal hedging error use these adjusted
amounts. This is absolute currency P&L accounting, not a time-weighted or
money-weighted return calculation. The discounted formulas above additionally
convert each flow at its own snapshot's common account value. The existing
nine P&L contributions and numeraire carry still reconcile without adding
capital as a profit component.

`cashflows.csv`/JSON record each event as `kind="external_capital"`, with its
ID, full input, content hash and, when enabled, the booking numeraire value/hash
and `discounted_amount`. The ledger retains daily/cumulative nominal and
discounted external adjustments. `summary.external_capital` reports committed
event count, net amount and positive gross deposit/withdrawal totals. Failed
snapshots contribute no capital to these reports. Capital uses the existing
cashflow table; the current output has eleven CSV tables including cost basis, Greek explanations and IV alignment.

IDs already processed, or repeated in a snapshot, raise
`duplicate_external_cashflow`, even if amount/source/revision changes.
The checkpoint preserves the applied-ID map. A later failure rolls back both
capital and IDs with all other state, allowing an exact retry. Processed capital
inputs cannot be edited on resume; unprocessed inputs can be repaired under
the normal prefix rule. Automatic replacement of previously booked events is
not implemented. `external_cashflow_policy` records timing, batching and a
full input hash, shared by all four strategies in `compare_strategies`.

Request versions 5/6 require `external_cashflows` in every snapshot, with `[]`
when empty. Versions 1–4 must omit it and preserve zero-capital behavior.
Capital accounting was introduced in result version 6; current results use
version 10. Old results remain readable with their original fields. Current
checkpoint version 8 preserves applied-ID, cost-basis, explanation, IV audit and budget state.

The example's `--capital-flow-demo` books +100 on the third session and −40 on
the final session, identically across all four strategies:

```shell
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --numeraire-rate 0.01 --capital-flow-demo --output benchmark-results/vega-kt-capital-example
```

See the [capital-flow validation record](../../design/validation/vega-kt-backtest-capital-v0.1.md).

## Paired terminal-error scenario statistics

`rust_pricing_backtest.ensemble` aggregates the **same book and observation
schedule** across outer market scenarios. It reads completed replay artifacts;
daily P&L observations are never treated as independent terminal samples.
Nominal and discounted errors are separate configurations. Discounted mode
requires an available shared numeraire and the same initial normalization.
Current per-run result/request/checkpoint versions are 10/6/8. A cohort must
share the constraint mode, limits and optimizer backend versions.

```python
from rust_pricing_backtest.ensemble import (
    EnsembleConfig, ScenarioResults, aggregate_scenarios,
)

# results_by_strategy comes from compare_strategies on one supplied history.
scenario = ScenarioResults(
    scenario_id="market-001", replication_id="replication-001",
    source="recorded-market-generator-v1", results=results_by_strategy,
)
report = aggregate_scenarios([scenario], EnsembleConfig())
report.write("benchmark-results/scenario-comparison")
```

The default `sampling_design="dependent_or_unknown"` gives descriptive estimates
without confidence intervals. Use `independent_simulation` only for a declared
independent outer simulation, with an `outer_randomness` mapping containing its
`algorithm`, `domain` and nonnegative integer `seed`. Replication IDs and seeds
must be distinct, and the generator source/algorithm/domain must agree.
Independence is a caller declaration, not inferred from timestamps or established
by these checks. Repeated market histories are rejected in either mode. Repeated
replication IDs in dependent mode are counted and never enable IID inference.

Every supplied result must agree on the portfolio, entry costs, instrument
master, complete observation schedule, strategy settings except the strategy
name, execution/funding/settlement policies, provider/source hashes, and recorded
runtime. Native pricing controls also freeze inner seeds, path counts, requested
risks and grid axes across scenarios. Custom providers must encode their numerical
settings in `provider.identity`. Within a scenario the full input-history hashes
must match across strategies, including quotes, funding, capital and numeraire.
Across scenarios market paths and subsequent numeraire values can vary; capital
inputs and initial numeraire normalization stay fixed. Different-horizon trades
or rolling historical books need a separate aggregation design.

Only the intersection of all four completed, fully settled runs with the selected
terminal metric contributes to any strategy statistic or improvement ratio.
Missing, partial, failed, unsettled and metric-unavailable runs retain explicit
reasons, coverage, last processed timestamp and failure code. They remove the
entire scenario from the common sample; no missing error becomes zero. Run counts
and independently eligible terminal counts remain visible for each strategy.
Complete-case estimates can be biased when failures depend on the market path.
The report identifies this selection explicitly rather than claiming a population
estimate that includes failed paths.

Statistics use signed terminal error for mean, sample standard deviation
(`ddof=1`) and RMSE, and **loss = negative terminal error** for quantiles/ES.
The quantile is the empirical inverse CDF. ES integrates that empirical quantile
over the upper tail, weighting a fractional boundary observation; it does not
average every tied loss above the quantile. Negative loss estimates are preserved.
The report also gives nominal execution costs and absolute traded values averaged
over the paired sample, and `1 − Var(strategy error)/Var(Delta error)` plus the
paired mean-error difference. A baseline variance at or below the configured
floor returns null, with a reason.

Defaults, all stored in `EnsembleConfig`, are research settings:

| Setting | Default |
| --- | --- |
| Loss levels | 95%, 99% |
| Confidence level | 95% |
| Bootstrap | 2,000 paired-scenario resamples; PCG64 seed 20261003 |
| Minimum sample count for intervals | 20 |
| Minimum effective tail observations | 20 |
| Minimum Delta variance | 1e-12 in squared error units |

The same resampled scenario indexes serve every strategy. Percentile interval
endpoints use linear quantiles. The seed and sorted scenario IDs make aggregation
reproducible in the recorded runtime. Intervals describe outer-market variation
**conditional on fixed pricing randomness**; they do not measure inner pricing
Monte Carlo uncertainty. If any resample has undefined baseline variance, the
variance-improvement interval is null and the invalid resample count is reported;
those draws are not silently dropped. Fewer than 20 effective upper-tail
observations means `reference_only` for the point estimate and no tail interval.
Thus the defaults need at least 400/2,000 paired samples for 95%/99% tail intervals;
that threshold alone does not establish accuracy. Dependent-history/block-bootstrap
inference is not implemented.

The independent outer-market integration example keeps initial marks and the
portfolio common. It applies illustrative, uncalibrated spot/volatility shocks
using calendar-day increments and separates outer-market SeedSequence entropy
`[0x56454741, market_seed]`, fixed inner pricing seed 7, and bootstrap seed. The
full planned scenario request is written before running the first path.

```shell
python examples/python/vega_kt_scenarios.py --scenarios 24 --paths 256 --diagnostic-only --numeraire-rate 0.01 --output benchmark-results/vega-kt-scenario-example
python -m rust_pricing_backtest.ensemble benchmark-results/vega-kt-scenario-example/ensemble-input.json --output benchmark-results/vega-kt-scenario-example/reaggregated
```

`ensemble-input.json` is a version-1 `hedging_backtest_ensemble_request` with
`config` and `scenarios`. Each scenario has the API metadata above and a `results`
mapping from strategy names to result JSON paths, resolved relative to the request.
Omit a strategy key to record a known missing run; a supplied unreadable path is
an input error, not silently missing. An empty `results` mapping retains a wholly
missing planned scenario; at least one result in the entire request must identify
the cohort. `ensemble.json` is a version-2
`hedging_backtest_ensemble`, with comparison settings/hashes, sampling assumptions,
per-scenario inclusion, summary statistics and retained risk-validation metadata.
`scenario_runs.csv` has all requested scenario/strategy pairs; nulls are blank.
CLI exit codes are 0 for a complete cohort, 1 for an incomplete cohort saved with
exclusions, and 2 for rejected inputs. Code 0 does **not** mean numerical acceptance.
Aggregates preserve the input validation scope; diagnostic and accepted-P0 runs
are not silently merged into one accepted cohort.

See the [scenario-statistics validation record](../../design/validation/vega-kt-backtest-ensemble-v0.1.md).

## Offline JSON Schema validation

The companion ships Draft 2020-12 schemas for the current exported wire formats:

| Document | Version | Schema file |
| --- | --- | --- |
| Replay request | 6 | `request-v6.schema.json` |
| Replay result, including all eleven tables | 10 | `result-v10.schema.json` |
| Checkpoint, including retained state/tables | 8 | `checkpoint-v8.schema.json` |
| Ensemble request | 1 | `ensemble-request-v1.schema.json` |
| Ensemble result | 2 | `ensemble-v2.schema.json` |
| Request with fixed market shocks | 7 | `request-v7.schema.json` |
| Result with fixed market shocks | 11 | `result-v11.schema.json` |
| Checkpoint with fixed market shocks | 9 | `checkpoint-v9.schema.json` |
| Ensemble with fixed market shocks | 3 | `ensemble-v3.schema.json` |

The new selectors use `definitions-v2.schema.json`; the unchanged ensemble
request uses v1. Historical request-v5/result-v9/checkpoint-v7/ensemble-v1 schemas
and the frozen v1 definitions remain bundled. Fixed-market-shock selectors use
the new `definitions-v3.schema.json`, for sixteen resources total. Files are under
`backtesting/rust_pricing_backtest/schemas/` and included in the companion wheel.
No native pricing extension is needed for structural validation. Install the
optional extra and validate existing JSON files:

```shell
python -m pip install "./backtesting[validation]"
python -m rust_pricing_backtest.validation benchmark-results/vega-kt-backtest/vega_kt/input.json benchmark-results/vega-kt-backtest/vega_kt/result.json benchmark-results/vega-kt-backtest/vega_kt/checkpoint.json
python -m rust_pricing_backtest.validation --export-schemas benchmark-results/backtest-schemas
```

The CLI identifies each document by its kind/version. Results keep that header
in `manifest`; checkpoints retain their existing version/state/hash envelope,
without adding a new `document_kind` field. It emits one JSON record per file,
exits 0 if every supplied document passes, and exits 1 if any cannot be validated.
The record contains the schema ID and bundle content hash, the original
risk-validation metadata when present, `semantic_checks="not_run"` and
`financial_acceptance="not_evaluated"`. Export alone needs no validation extra.
The validator resolves only bundled resources and never downloads remote schemas
or follows result-file paths listed inside an ensemble request.

Python callers should pass decoded JSON, not dataclasses or NumPy scalars:

```python
from pathlib import Path
from rust_pricing_backtest.validation import loads_document, validate_document

document = loads_document(Path("result.json").read_text(encoding="utf-8"))
check = validate_document(document)
```

`loads_document` rejects duplicate keys, NaN/Infinity and numbers that overflow to
nonfinite floating-point values. `validate_document` checks required fields,
closed record shapes, number/boolean types, ranges, nulls and tagged status
conditions. It checks calendar dates and explicit-offset ISO timestamps (the
export profile uses `T` and RFC 3339-style `Z`/`±HH:MM` offsets). It requires failed
results to carry failure details, incomplete/unsettled results to have null
terminal errors, unmatched IV nodes to have null shocks, and the current IV audit
to remain diagnostic. Native pricing-request strings are separately decoded to
check JSON syntax; their pricing-model schema and financial conventions are still
validated by the native adapter. `contentMediaType` in JSON Schema alone is an
annotation and does not perform that additional decoding.

Provider validation/policy metadata, outer-generator metadata and native pricing
control metadata remain documented extension points. Core input and report
records reject unexpected fields. Request schemas preserve constructor defaults
accepted by the current reader. Formal validation is opt-in and does not change
replay or legacy-read behavior. The dispatcher accepts the current versions
and the previous v5/v9/v7/v1 wire contracts. Earlier formats require their
existing readers. Re-run original inputs to create current artifacts; do not
merely relabel a version number. New ensemble reports can describe a uniformly
historical v9 cohort, but cannot mix it with a v10 or v11 cohort. Fixed-market
ensembles also require identical configured shocks and policy across strategies.

Structural validity does **not** check arithmetic reconciliation, content-hash
equality, unique IDs across rows, chronological/availability order, array shapes
against the risk basis, comparison populations, or statistical independence.
Those checks remain in the existing readers, replay engine and aggregation API.
Schema validity does not upgrade native VegaKT's `unvalidated` status.

External validators should load the sixteen resources locally by `$id`, starting with
the desired selector schema. Enable date/date-time format checks: Draft 2020-12
validators can otherwise treat formats as annotations. The provided CLI enables
them explicitly, with no optional timezone-format dependency or schema download.

The explicit schema specification is maintained in
[generate_backtest_schemas.py](../../scripts/generate_backtest_schemas.py), rather
than inferred from a successful example. Generate with that script and run
`python scripts/generate_backtest_schemas.py --check` to detect stale resources.
Five frozen wire examples live in `fixtures/backtesting/wire/`; their recorded
source hashes are historical provenance, not claims about the current checkout.
The backtesting CI installs the validation and constraints extras and tests both format generation
and malformed documents. Without that extra, schema-dependent tests are skipped;
the local validation record below ran all of them.

See the [schema validation record](../../design/validation/vega-kt-backtest-schemas-v0.1.md).

## Path-dependent Gamma research

The native plan exposes `evaluate_vega_kt_gamma_recovery_diagnostic(gamma_strikes,
relative_regularization=...)` for positive-time Asian/Lookback Gamma recovery in
spatially flat, zero-carry pseudo-MC. This uses the compiled path payoff's AD and
the full conditional operator, with signed masses, sampling uncertainty and
regularization/coverage audits. See the [validation record](../../design/validation/local-vol-path-gamma-recovery-v0.1.md).
It does not extend the backtest adapter's European contract scope or its accepted
P0 status; initial Gamma, nonflat recovery, barriers and full IV-hedge validation
remain separate work.

## Verification and remaining scope

```shell
python -m unittest discover -s backtesting/tests -v
python -m unittest discover -s tests/python -p 'test_vega_kt*.py' -v
python -m rust_pricing_backtest.smile_numerics --policy fixtures/backtesting/smile-policy-v0.3.json --output benchmark-results/vega-kt-smile-v0.3
```

The first suite checks hand-computed cash/settlement/funding, fees, multiplier
and sign conventions, matrix solutions, constraints, scalar rolls, held-but-
excluded contracts, future data, missing expiry observations, serialization
and checkpoint replay. The second compares native adapter outputs directly to
public pricing results and completes four-strategy JSON replay. It is also
discovered by the existing installed-wheel smoke suite. A separate workflow
tests the companion package on Linux, macOS and Windows; local verification is
not a claim that remote CI has run.

The last command runs the combined P0 terminal-smile gate: 770 smile checks
plus the 307 fixed-market checks. It tests equation-(9) quote differences,
SSVI/eSSVI surface rebuilds with paired CRN uncertainty, reporting/time/seed
refinement, exact hedge solutions and omitted-direction risk. It exits **1**
on any failed current check or unresolved legacy failure. The six original
failed comparisons retain their observations and explicit passing replacements;
unknown failures cannot be superseded. The manually dispatched workflow retains
the full report on failure. See the [acceptance record](../../design/validation/vega-kt-smile-acceptance-v0.3.md).
General path-dependent Gamma recovery remains separate work.

The separately versioned comparison fixes 9-, 17- and 33-node market shock
families, refines only their numerical reporting grids, and audits independent
fine-grid residual risk:

```shell
python -m rust_pricing_backtest.market_shock_numerics --policy fixtures/backtesting/market-shock-policy-v0.2.json --output benchmark-results/vega-kt-market-shocks
```

Its exit code describes that restricted v0.2 gate. The original gate still exits
1, and its failed checks are embedded in the new report. For explicit research
optimization, `MarketShockMap` and `solve_market_hedge` are available from
`rust_pricing_backtest.market_shocks`; every solution includes the full reporting
residual and an orthogonal-complement audit. Marginal standard errors alone
cannot be used to transform covariance.

To apply the fixed shocks in daily replay:

```python
config = BacktestConfig(
    strategy="vega_kt",
    market_shock_log_moneyness=(-0.4, 0.0, 0.4),
)
```

Every reporting grid must include the configured knots and the same endpoints.
The nodes stay fixed in log-forward-moneyness for the run; their physical strikes
move with the forward. Maturities match each current snapshot exactly. This is
not a cross-date or maturity-grid alignment rule. Candidate filtering and both
optimization backends use transformed risk for `vega_kt`; other strategies keep
their original objectives. `risks[*].market_shocks` audits target and held risk
after fills. `decisions[*].market_shocks` preserves optimizer inputs, continuous
and rounded proposed risk, including the orthogonal complement. No transformed
standard errors are inferred from marginal bucket errors. The default `None`
keeps the existing objective. The native diagnostic guard remains in force.

To isolate the missing initial-time local-grid sensitivity with paired repricing:

```shell
python -m rust_pricing_backtest.allocation_probe --policy fixtures/backtesting/allocation-probe-policy-v0.1.json --output benchmark-results/vega-kt-allocation-probe
```

This diagnostic perturbs only the explicitly supplied initial local-volatility
row. It is not a market-IV bump. It applies only to
`local_vol_node_allocation_diagnostic_v1`. Other methods, including the new
terminal measure, receive `not_applicable` (exit 2), never a passing result or
a missing-initial-risk diagnosis. For the applicable method, passing explains
its residual; it does not make the paper VegaKT gate pass. Historical version-1
probe reports remain unchanged; the applicability-aware report is version 2.

The diagnostic workload command measures 252 sessions, 10 targets, 20 hedge
candidates and a 5 × 9 basis, with a verified candidate/call count:

```shell
python -m rust_pricing_backtest.benchmark --policy fixtures/backtesting/performance-policy-v0.1.json --output benchmark-results/vega-kt-performance-full --diagnostic-only
```

It uses only 256 independent sampling units and five local time intervals. Its
timing is a capacity observation, not an accuracy-qualified performance claim.
The report separates input creation, replay and file writing and records the
native binary hash and process peak RSS where supported.

The [requirements](../../design/vega-kt-hedging-backtest-requirements-v0.1.md)
remain broader than this first implementation. Outstanding work includes:
actual market/data selection; raw quote calibration; general path-dependent
Gamma acceptance beyond the P0 terminal scope; full accepted H0–H3 evidence;
252-session performance budgets; collateral/margin and integer-quantity optimization;
fixing-correction cashflows and delayed dividends; validated IV/VegaKT shock interpretation with the Delta convention and full PNL-04 coverage;
intraday capital timing; dependent-history inference and heterogeneous trade cohorts;
legacy-schema migrations; hold-position mode;
non-vanilla lifecycle adapters and historical validation H4.

Input snapshots and pricing requests are retained alongside their hashes. The
manifest records the companion source hash, native binary hash, effective
configuration and execution environment. A user-supplied code revision labels
the checkout but does not replace these content hashes.


The native `evaluate_vega_kt_initial_gamma_diagnostic(boundaries)` also supplies a
conditional Gamma measure for the first positive time in the flat-space,
zero-carry scope. It preserves known history, includes both tails and compares
exact first-interval integration with native AD using paired uncertainty.
This diagnostic does not enable path-dependent instruments through the accepted
P0 provider; see [validation](../../design/validation/local-vol-initial-gamma-measure-v0.1.md).


`evaluate_vega_kt_gamma_recovery_audit(strikes, policy=...)` combines a
pilot-selected positive-time recovery with the conditional initial measure.
The original pricing sample range is held out from regularization selection;
validation failures remain explicit. The accepted P0 provider still does not
promote path-dependent contracts on the basis of this diagnostic. See the
[selection policy and limits](../../design/validation/local-vol-gamma-selection-v0.1.md).

`evaluate_vega_kt_atom_chain_diagnostic(strikes, policy=...)` now connects those
Gamma measures to IV buckets and hedges, retaining joint sampling uncertainty
and optional full covariance across maturities. It withholds the estimate if
recovery validation fails. This is a separate, flat calibrated-model diagnostic;
it does not enable path products in the accepted P0 provider. See the
[atom-chain validation](../../design/validation/local-vol-atom-chain-v0.1.md).

For flat maturity-IV experiments, `rust_pricing_backtest.flat_iv` supplies
`recalibrate_flat_market_iv(payload, maturity_shifts)` and
`maturity_risk_statistics(atom_chain_diagnostic)`. Recalibration rebuilds forward
variances from shifted cumulative market variances and rejects invalid calendar
variance instead of clipping it. Risk projection requires full joint covariance.
The optional native `preserve_local_vega_total=True` recovery policy removes the
lost-integral bias in these maturity shocks; it does not promote path-dependent
contracts through the accepted P0 provider. See
[conservative recovery validation](../../design/validation/local-vol-conservative-gamma-v0.1.md).

`rust_pricing_backtest.discrete_smile.TwoStepSmileCalibrator(payload)` fits
terminal vanilla IV quotes to actual two-interval log-Euler prices, keeping the
first Gaussian marginal fixed. `calibrate(terminal_iv_shifts)` returns a copied
request and a price/residual/rank audit; `terminal_iv_tangent()` returns the
minimum-norm local-variance response to terminal quote IV. Calibration uses no
exotic payoff or recovered Gamma. Its scope requires zero carry, unit discount,
two matching simulation intervals, and initially calibrated quotes. It fits the
listed strikes, with a declared minimum-norm local-variance extension. Refined
quadrature must confirm the price tolerance; infeasible quotes and failed solves
are rejected without clipping. This companion diagnostic does not enable a new
accepted provider or native nonflat Gamma recovery. See the
[calibration checks and separate Gamma quote failures](../../design/validation/local-vol-discrete-smile-calibration-v0.1.md).

The native recovery policy accepts optional
`local_vega_conservation_directions[row][direction][source_node]` to preserve
specified signed local-Vega functionals. For a flat two-step model, freeze the
calibrator's terminal-IV tangent U before sampling and pass `(U.T/(2*sigma1))`
as the sole recovery row. Combine with `preserve_local_vega_total=True` to impose
all equalities simultaneously. The diagnostic returns same-sample
`local_vega_directions`, `reconstructed_directions`, and `direction_differences`
moments. These support calibrated quote-risk checks with actual marginal SE;
they are distinct from the atom-chain reporting-IV buckets. Pilot/held-out
spatial and Gamma-noise gates still apply. See the
[retained failures and refinement results](../../design/validation/local-vol-gamma-quote-conservation-v0.1.md).


### 非平坦な第2区間への拡張

`PricingPlan.evaluate_vega_kt_two_step_smile_gamma_audit` は、2区間モデルの
終端quote再較正方向を年率局所分散の座標で評価する追加診断である。
`TwoStepSmileCalibrator` の接線をそのまま方向行列へ転置して渡せる。
通常のバックテストproviderは変更せず、初期Gamma・一般多区間の全面IV受入も未完了。
[検証記録](../../design/validation/local-vol-nonflat-gamma-v0.1.md)に範囲・再実行方法を残す。


非平坦な将来に対する初期成分は、別API
`evaluate_vega_kt_initial_smile_gamma_audit` で有限領域Gammaと初期年率分散感応度を検証できる。
対象は2区間・一定第1分散で、第2分散は固定する。市場IV bucketの全面受入や
通常バックテストproviderの昇格には直結させない。
[検証と残件](../../design/validation/local-vol-initial-smile-gamma-v0.1.md)。

### 2区間の共同市場IV診断

`TwoStepSmileCalibrator.calibrate_joint(first_iv_shift, terminal_iv_shifts)` は、
第1満期のflat IVを平行移動してから第2分散rowを再較正する。
終端shiftがゼロなら終端スマイルを保つ。`joint_iv_tangent()` は第1一定分散と
第2分散節点の全接線を返す。既存の固定第1分布 `calibrate` / `terminal_iv_tangent` は維持する。

`rust_pricing_backtest.joint_smile.evaluate_joint_market_gamma(payload, bins, policy=...)`
は較正検証、接線構築、同じpayloadのcompile、初期・第2Gammaの共同診断を一度に行う。
第1parallel IVと各終端quote IVの共分散を含み、初期近似・第2回復・共同照合のいずれかが
不合格なら `estimate=None`。市場IVの一般的な全bucketやバックテストproviderの受入は未確立。
[再較正した価格差分との比較](../../design/validation/local-vol-joint-smile-gamma-v0.1.md)。

### 3区間の終端スマイル

`rust_pricing_backtest.three_step_smile.ThreeStepTerminalSmileCalibrator(payload)` は
最初の2遷移とquoteを固定し、実際の非Gaussian第2分布に対する第3分散rowを較正する。
`calibrate(terminal_iv_shifts)` は新requestと監査情報、`terminal_iv_tangent()` は年率分散接線Uを返す。
同じpayloadをcompileし、`evaluate_vega_kt_three_step_terminal_gamma_audit` にU.Tを渡す。
価格精度・rank・分散境界・source積分精度に失敗した場合は明示的に拒否する。
この終端単独APIは初期・中間Gammaを含まない。中間との結合は次節の別診断で扱う。
[数値受入記録](../../design/validation/local-vol-three-step-gamma-v0.1.md)。

### 3区間の中間・終端スマイルの結合

同じcalibratorの `calibrate_joint(middle_iv_shifts, terminal_iv_shifts)` は、第1分散rowを固定し、
中間を較正してから変更後の分布で終端を再較正する。`joint_iv_tangent()` は第2・第3rowの
年率分散接線を返し、列の順序は中間quote、終端quote。中間だけを動かす場合も終端分散を補償する。

`rust_pricing_backtest.middle_smile.evaluate_middle_terminal_market_gamma` は、
履歴条件付き中間Gammaと終端Gammaの同一パス方向統計を合成する。
どちらかの回復・照合が失敗すれば結合推定値を返さない。
SEは終端MC統計、中間の決定論的数値許容差は別出力とし、全方向間共分散は返さない。
第1満期IV・任意区間数は未対応。通常providerへの昇格も行わない。
[数値受入記録](../../design/validation/local-vol-middle-smile-gamma-v0.1.md)。

### 第1平行IVを含む3満期のGamma

`ThreeStepTerminalSmileCalibrator.calibrate_all_maturities(first_iv_shift, middle_iv_shifts, terminal_iv_shifts)` は、
最初の平行IV、中間quote、終端quoteの順で再較正する。後続満期のshiftを0にすると、その市場quoteを保持する。
戻り値は全満期の価格検査に合格したrequestと監査記録。途中の未較正requestは返さない。
`all_maturity_iv_tangent()` は第1平行年率分散＋中間節点＋終端節点を行、
第1平行IV＋中間quote＋終端quoteを列とする接線を返す。

```python
from rust_pricing_backtest.three_step_full_smile import evaluate_three_step_market_gamma

audit = evaluate_three_step_market_gamma(
    request_payload, [80., 90., 100., 110., 120.],
    policy={
        "candidate_regularizations": [.01, .003, .001, .0003, .0001],
        "pilot_sampling_units": 262144,
        "reconstruction_relative_limit": .03,
        "gamma_relative_standard_error_limit": .25,
        "mc_margin_multiplier": 2.,
    },
)
# 各満期5 quotesなら、第1平行IV＋中間5点＋終端5点の11方向。
# 三つのGamma検査がすべて合格した場合だけaudit["estimate"]を返す。
```

通常providerに置き換わる診断ではなく、3区間・平坦第1row・後続非平坦rowの限定された入口。
第1満期の各スマイル点は独立に較正できないため、`full_market_iv_bucket_acceptance` は
`not_established` を維持する。[検証記録](../../design/validation/local-vol-initial-three-gamma-v0.1.md)。


### 可変区間数と非平坦な第1市場スマイル

`rust_pricing_backtest.multi_step_smile.MultiStepTerminalSmileCalibrator` は2..16区間に対応する。
報告満期はsimulation knotsの部分集合で、最後の満期を含める。
第1報告満期が複数遷移後なら非平坦スマイルを扱える。全基準quotesのモデル価格一致を検査する。

```python
from rust_pricing_backtest.multi_step_smile import MultiStepTerminalSmileCalibrator

calibrator = MultiStepTerminalSmileCalibrator(request_payload)
tangent, tangent_audit = calibrator.terminal_iv_tangent()
# 行は最後の年率局所分散節点、列は終端IV quote。
recalibrated, calibration_audit = calibrator.calibrate([0., 0., 0.0001, 0., 0.])
# 上記は終端5 quotesの例。先行分散rowと先行市場quotesは固定。
# tangent.T をnativeの multi_step_terminal_gamma_audit の保存方向へ渡す。
```

較正失敗は例外とし、無理な分散clippingは行わない。IVは小数表示の絶対値。
この入口は終端Gamma用の診断であり、先行Gammaや全満期スマイルの同時変更には未対応。
[4・6区間、Asian/lookbackを含む数値受入](../../design/validation/local-vol-multi-step-gamma-v0.1.md)。


### 各報告満期のスマイルを連鎖させる

`MultiStepSmileCalibrator` は、各報告満期直前の年率分散rowを選択し、報告されない区間を固定する。
第1報告満期は2回以上の遷移後に置く。後続満期のshiftが0なら、そのquotesを保持する。

```python
from rust_pricing_backtest.multi_step_full_smile import (
    MultiStepSmileCalibrator, evaluate_multi_step_market_gamma,
)

c = MultiStepSmileCalibrator(request_payload)
# shiftsは[報告満期][quote]。IV単位は小数表示の絶対値。
recalibrated, calibration_audit = c.calibrate_all_maturities(shifts)
tangent, tangent_audit = c.all_maturity_iv_tangent()
audit = evaluate_multi_step_market_gamma(
    request_payload, [80., 90., 100., 110., 120.],
    subdivisions=512, relative_tolerance=1e-6,
)
# estimateは数値精度・MC精度・native照合がすべて通った場合のみ。
```

4区間2満期で10方向、6区間3満期で15方向を検証した。
これは指定した較正延長規則に対する診断。全モデルや通常providerの全面受入は主張しない。
[検証記録](../../design/validation/local-vol-path-gamma-chain-v0.1.md)。


### 条件付きGammaの市場方向間共分散

`evaluate_multi_step_market_gamma` の成功時には `market_sample_covariance` と
`market_mean_covariance` が返る。`coordinate_order` 順のD×D行列で、後者は前者を
独立標本数で割った値。係数wで市場方向を合成する場合、その平均の分散は
`w.T @ market_mean_covariance @ w`。方向間の相殺を保持する。
失敗時は両方 `None`。raw共分散は `gamma_audit` 内に診断として残る。
[数値受入と適用範囲](../../design/validation/local-vol-path-covariance-v0.1.md)。


### 条件付きADによる経路Gammaの照合

`evaluate_multi_step_market_gamma` のnative診断は、最終乱数を解析積分したADを使って
中間区間のGammaを照合する。成功時の市場リスク・共分散の座標と単位は同じ。
精度不足では推定値と市場共分散を抑止し、raw診断を保持する。
[受入記録](../../design/validation/local-vol-conditional-ad-v0.1.md)。

## Explicit variance-extension diagnostics

[All-transition calibration and Gamma](../../design/validation/local-vol-variance-extension-v0.1.md) adds simultaneous all-row and linear-time extensions and a custom-basis interface. The extension metric is explicit and retained in diagnostics. These APIs do not change the calibration rule of existing backtest providers.

The opt-in [extension provider](../../design/validation/local-vol-extension-provider-v0.1.md) connects the two preset rules to dated Call-side European, Asian and Lookback backtests. Select `RustPricingProvider(numerical_scope="multi_step_extension_gamma", variance_extension="all_rows")` (or `"linear_time"`), or use `--validated-extension all_rows`. This scope requires the extension explicitly and rejects Put-side and custom-basis requests. Each rule has a distinct checkpoint identity.

The separate [Put-enabled scope](../../design/validation/local-vol-extension-put-v0.1.md)
accepts Call/Put European, arithmetic Asian and fixed Lookback contracts with the two presets.
Select `numerical_scope="multi_step_extension_put_gamma"` and `variance_extension="all_rows"`
or `"linear_time"`, or use `--validated-put-extension linear_time`.
For Lookback Puts, the effective fixed strike is included in the native integration boundaries;
the existing subdivision, error and sampling requirements remain in force.
Acceptance includes independent Put price differences and mixed Put-target/Call-hedge settlement and resume.
Custom bases and cross-instrument covariance remain outside this scope.

The diagnostic [custom-basis API](../../design/validation/local-vol-custom-basis-v0.1.md)
builds a tensor matrix from time and moneyness knots and records positive parameter scales.
Pass the returned matrix to `evaluate_extended_market_gamma(extension="custom", variance_basis=basis)`
and retain its construction audit alongside the Gamma audit. Two declared configurations
are checked with independent Call/Put price differences.

The [named-basis provider](../../design/validation/local-vol-custom-basis-provider-v0.1.md)
connects these two constructions to Call/Put European, Asian and Lookback ledgers.
Select `numerical_scope="multi_step_custom_basis_gamma"` and
`variance_basis_policy="row_hats"` or `"weighted_time_hats"`, without `variance_extension`.
The CLI uses `--validated-custom-basis row_hats`. Construction settings and basis hashes are retained
in audits; changing the selected basis invalidates saved checkpoints. Arbitrary matrices remain diagnostic-only.

Use the [raw-basis numerical checker](../../design/validation/local-vol-raw-basis-v0.1.md)
to test a supplied cell-by-parameter matrix on the accepted market fixtures.
It performs fresh quote recalibration, independent Call/Put price differences and native Gamma/covariance checks.
Passing a matrix does not promote it to the ordinary ledger provider.

The [coupled-basis ledger policy](../../design/validation/local-vol-coupled-basis-provider-v0.1.md)
adds `variance_basis_policy="coupled_row_hats"` / `--validated-custom-basis coupled_row_hats`.
It uses only the two fixed four/six-interval matrices verified by the raw-matrix checker,
and rejects a different generated matrix. The parameter metric, mixing rule and hashes
are retained in risk audits; checkpoints bind the selected basis and current implementation.

The [current per-matrix checker](../../design/validation/local-vol-current-raw-basis-v0.1.md)
restores raw-matrix diagnostics after the named-policy integration. It verifies the current
accepted runtime and delegates to the unchanged numerical implementation. Preflight results
are explicitly not numerical acceptance; continuation checks both report integrity and exact
input/runtime compatibility. Ledger code and checkpoint identities are unchanged by this addition.

## 商品間の推定誤差とVegaKT損益説明

[通常バックテストへの接続](../../design/validation/vega-kt-joint-replay-v0.1.md)では、
`BacktestConfig(max_joint_bucket_standard_error=...)` を指定し、保有・発注予定の合同精度検査と
保存済み期首リスクによる日次IV損益説明を有効にできる。新形式での保存・再開と通常CLIに対応する。

[市場拡張・合同共分散・損益説明の実装記録](../../design/validation/vega-kt-portfolio-market-v0.1.md)を参照。
`portfolio_market_risk.evaluate_portfolio` で共通経路の異商品共分散を計算し、
`explain_backtest` で保存済み台帳に対応する期首リスクから追加の損益説明を作成する。
固定strikeのIV変化にDeltaを合わせ、期限切れ・欠落・外挿が必要な区間は未説明とする。
既存の台帳と旧受入記録は保持する。
