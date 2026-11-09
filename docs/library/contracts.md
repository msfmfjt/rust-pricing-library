# Contract sources and lifecycle

[ADR 0041](../../design/adr/0041-contract-lifecycle-mc.md) records the migration
and Monte Carlo integration boundary. `pricing::contract::Contract` is the
canonical source. Derived residuals and Product Graphs must not be edited as a
second contractual definition.

`NumericExpr`, `BooleanExpr` and `DateExpr` are separate Rust types. Numeric
expressions support arithmetic, min/max/abs, conditional selection and discrete
aggregations. Predicates include exact comparisons, logical operations and date
comparisons. Explicit fixing maps can evaluate expressions without simulation;
unresolved observations and nonfinite arithmetic fail. Conditional expressions
and Boolean operators short circuit. No exercise optimization is implied.

`Observation` records a stable ID, underlying, spot/pre-dividend/rate kind,
Gregorian date and fixing source. Optional clock, timezone and calendar metadata
are preserved. Simulation support is checked separately; saving metadata does
not promise a model can produce it. Dates are literal Gregorian dates; calendar
adjustment is the caller's responsibility.

A `Cashflow` carries stable ID, currency, amount, receive/pay direction,
determination, payment and expected settlement dates. Amounts may be signed;
direction multiplies their sign. `Term` provides empty, bundle, conditional,
reverse, scale and discrete barrier composition. Barriers use inclusive up/down
boundaries. Contract v2 adds holder exercise rights and continuously monitored
vanilla options. Cash-flow IDs must be unique across the complete source, including
mutually exclusive branches. Unknown references, duplicate observation IDs,
nonfinite constants, dates before dependencies and excessive depth/size fail
at construction. The owned tree has no reference cycles.

`analyze()` returns required observations (the fixing schedule), potential and
unconditionally determined cash flows, underlyings, currencies, barrier dates
and horizon. Potential amounts retain their expressions, with enclosing Scale
and Reverse applied. Analysis validates all source branches before extracting
only live deterministic dependencies; known arithmetic and date conditions are
folded for determined payments. Currencies are never
implicitly aggregated. A constant amount under a stochastic condition is not a
determined obligation.

`to_json()` / `from_json(bytes, JsonLimits)` preserve meaning and stable IDs.
[Contract schema v1](../../schemas/contract-v1/contract.schema.json) is separate
from frozen pricing request/result and graph formats. Decoding uses the existing
bounded duplicate-rejecting parser and then Rust semantic validation. Unknown
fields and versions fail. Observation declaration order is canonicalized by ID;
term and cash-flow order remain contractual. Fingerprints identify source data,
not algebraic equivalence. Ordinary terms keep v1 encoding and fingerprints. Contracts containing a v2
option term use [Contract schema v2](../../schemas/contract-v2/contract.schema.json).
Lifecycle checkpoint versions follow the original contract.

See [source conformance](../../crates/pricing/tests/contract_source.rs) for typed
construction, semantic identities, strict JSON rejection and dependency checks.

## Lifecycle replay

`advance(&contract, events, as_of)` returns a residual contract, all determined
cash flows (with settlement status), outstanding payments, remaining observations,
applied event IDs and barrier states. The same operation is available as
`Contract::advance` and `LifecycleResult::advance`. The latter combines retained
history with new events and rejects moving the valuation date backwards.

Fixing events carry the complete observation identity and publication date.
Duplicate IDs with identical content are idempotent; reused IDs or inconsistent
fixings fail. Future events remain in history but are not applied. On the same
date, fixings precede exercise and barrier reports, then determination confirmations, termination and settlement;
stable event IDs break remaining ties. The date-only interface treats same-day
fixings without an event as current observations, consistent with native MC;
strictly earlier required observations must have a fixing. Intraday cutoffs
require a separately supported simulation mapping.

Determination confirmations check the signed amount/currency derived from the
contract. Settlements require an already determined obligation and cannot precede
the contractual settlement date. An overdue payment remains outstanding until an
explicit settlement event; it is never silently deleted. Termination cancels
cash flows determined strictly after its date and preserves existing obligations.
These are explicit v1 termination semantics, not a closeout valuation model.

A checkpoint saves only the original contract, unique event history and as-of
date. `LifecycleResult::from_json` replays them through normal validation rather
than trusting an independently editable residual or balance. Lifecycle barriers
share inclusive comparison semantics with the source expressions; observing a
hit removes remaining monitoring dependencies. See the
[lifecycle tests](../../crates/pricing/tests/contract_lifecycle.rs) for replay,
settlement, early redemption and original/residual future cash-flow equivalence.

## Native MC compilation

`contract.compile_mc(as_of, ordered_underlyings)` lowers the validated source or
lifecycle residual into currency groups. Supply the model's ordered underlying
universe, including an anchor for fixed payments. Each group exposes immutable
`product()` and `payoff()` derivatives and cash-flow IDs in output order.
`CompiledContract::pricing_task(template, operation)` replaces the product in an
existing `PricingSpecification`; `multi_asset_task` does the same for a basket
source. Templates retain market, model, numeraire, seed, simulation grid policy,
execution and risk settings. Multi-currency contracts require explicitly separate
native tasks from `groups()`; the convenience adapter rejects multiple currencies
rather than dropping another currency. `stochastic_dividend_task` preserves the
existing dividend dynamics and driver correlations.
An empty contract lowers to no groups; a native task uses a zero cash flow at
valuation. Overdue unpaid payments require explicit settlement/closeout handling.

The native compiler supplies observation dates and time-grid mapping. Spot and
pre-dividend spot reuse their existing simulation states. Intraday observations,
rate fixings and conflicting fixing-source identities mapping to one native
observation fail. The native capability/preflight checks still govern model,
volatility, rates, correlations and risk. In particular, single-asset stochastic
rates with distinct payment dates remain unsupported; the multi-asset path uses
its existing per-payment pathwise discount factors. No deterministic-rate
substitution is made.

Exact subexpressions and observations are interned during contract lowering;
constants and known branches are eliminated before graph compilation. The
existing tape adds exact comparison, lazy selection, logarithm and exponential
operations. An immutable control-flow schedule reads only selected branches,
including nested early termination. Shared intermediate results are memoized
per path, and literal values are prepared once per workspace and tape identity. Geometric averages require positive observations. Reuse
`payoff.workspace()` with `evaluate_reusing` to retain path arrays and call-stack
capacity. Native calls on new lazy tapes also reuse scratch buffers per worker,
with at most 1 MiB retained and an owned fallback for reentrant callbacks.
Existing nonbranching graphs retain their forward instruction order
and allocation count. No contract AST is evaluated in a Monte Carlo path loop.

Deterministic multi-payment discount ratios, curve pullbacks, covariance-aware
aggregation, MC/RQMC statistics and supported AAD use the native implementation.
Live hard comparisons reject pathwise risk; they are not assigned an invented
pathwise derivative. Explicit MC parameter-bump repricing can use the same seed
and numerical configuration for finite-difference Greeks. Bump bias and sampling
covariance must be assessed separately; no PDE engine or estimator substitution
is introduced.

Graph sources containing the new operations write `product_graph` v2, nested
specifications write v3, and tasks write v2. Legacy graphs/specifications/tasks
continue writing v1/v2/v1 with unchanged source fingerprint domains. New operations
under old envelope versions are rejected. Current schema accessors accept both
legacy emitted and extended versions; frozen historical schema files are unchanged.
The [extended schema checks](../../scripts/check_contract_mc_json_schema.py)
exercise new sources and reject version downgrades. Native and contract regression
coverage is in [contract MC tests](../../crates/pricing/tests/contract_mc.rs).

## Instrument and Python builders

`InstrumentBuilder::new(id, currency, underlying, fixing_source)` generates the
same source types for `forward`, `european`, `digital`, `asian`, `swap`,
`discrete_barrier` and `autocall`. No builder owns a simulation algorithm.
`Contract::bundle`, `reversed` and `scaled` compose validated sources. Cash-flow
and observation IDs are stable names derived from the builder ID; use different
builder IDs for separate trades. Analysis returns fixing schedules in date/time,
underlying and ID order, independently of canonical serialization order.

* Digital call/put boundaries are inclusive and payout is a cash amount.
* Asian aggregation supports arithmetic/geometric average and running min/max.
  Geometric averages require positive observed values.
* Swap periods specify fixing/payment dates and positive accrual fractions;
  fixed and floating legs remain separate signed cash flows. Future rate
  observations currently fail native MC compilation because the graph does not
  expose a simulated rate fixing. Published fixings can be reduced and the
  resulting fixed payments can be priced with existing curves.
* Discrete barrier rebates are absolute cash amounts paid at the specified final
  payment date. Both knock-in/out and up/down boundaries use inclusive hits.
* Autocall levels are absolute spot levels; coupon rates multiply notional.
  A qualifying coupon is paid on a call date. A call pays notional and cancels
  later coupons/principal. At maturity, principal is notional above the
  protection level; below it, principal is `notional * final_spot / initial_spot`.
  No memory coupon is implied. Version 1 supports at most 20 observation periods
  within the source depth and graph-output bounds.

```rust
use pricing::contract::InstrumentBuilder;
use pricing::core::{CurrencyId, UnderlyingId};
use pricing::product::OptionSide;
let expiry = "2027-09-04".parse()?;
let contract = InstrumentBuilder::new("call", CurrencyId::new(2),
    UnderlyingId::new(1), "exchange.close")?
    .european(OptionSide::Call, 100.0, 1.0, expiry, expiry)?;
let compiled = contract.compile_mc("2026-09-04".parse()?, vec![UnderlyingId::new(1)])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Python exposes immutable `ContractBuilder`, `Contract` and `ContractLifecycle`.
`Contract.from_json` constructs arbitrary typed source trees beyond the builders;
`Contract.json_schema()` describes their strict structure. A complete example is
[examples/python/contracts.py](../../examples/python/contracts.py).

```python
builder = rp.ContractBuilder("call", 2, 1, "exchange.close")
contract = builder.european("call", 100., 1., "2027-09-04", "2027-09-04")
# source is an existing complete PricingSpecification, MultiAssetSpecification,
# or StochasticDividendSpecification.
task = contract.pricing_task(source, operation="evaluate")
report = task.compile().evaluate().report
schedule = json.loads(contract.analysis_json())["required_observations"]
event = contract.fixing_event_json("fix", schedule[0]["id"], 120.)
rolled = contract.advance("[" + event + "]", "2027-09-04")
checkpoint = rolled.to_json()
```

`pricing_task(..., operation="aad")` uses the existing source capability checks.
Hard digitals, autocalls and barriers reject unsupported pathwise estimators;
there is no silent smoothing. Source errors expose `code`, `stage` and `field`
on `ValueError`. Native task errors retain their existing typed metadata.
`to_graphs(as_of, underlying_ids)` returns one graph per currency and does not
sum or convert currencies. JSON analysis/payment methods return detached data.
Heavy native lowering, replay and task compilation/evaluation release the GIL.

Run focused conformance with `cargo test -p pricing --test contract_source
--test contract_lifecycle --test contract_mc --test contract_builders` and the
installed-wheel suite `python -m unittest discover -s tests/python -p test_contracts.py`.
Run `cargo run --release -p pricing --example benchmark_contract` for compilation,
payoff and complete MC timings. Defaults are 1,000,000 payoff paths, 262,144
MC sampling units, 12 observation dates, two workers, one warmup and seven
measured repetitions with alternating execution order. Override these with
`CONTRACT_BENCH_PATHS`, `CONTRACT_BENCH_SAMPLES`, `CONTRACT_BENCH_PERIODS`,
`CONTRACT_BENCH_WORKERS` and `CONTRACT_BENCH_REPETITIONS`; periods must divide 12.
Raw paired samples and immutable evaluation-program bytes are reported.
The benchmark measures reserved workspace bytes and capacity growth directly.
Native exercise and continuous-barrier plans request compact forward evaluation.
Other graphs can opt in with `SourceGraph::compile_compact`. Linear programs in
this mode reuse registers after their last read. Reverse mode
has access to the original SSA values. In branching graphs, only shared calls
need memo flags. LSM releases training path matrices before allocating valuation
paths. Run `cargo run --release -p pricing --example benchmark_contract_advanced`
for seven paired native/Contract timings, values, errors and a path-storage check.

The [Phase 0–4 verification record](../../design/validation/contract-dsl/README.md)
links the five draft PRs, local checks, raw benchmark/mutation evidence, hosted
CI snapshot, measured performance envelope and explicit capability limits.

## Exercise and continuous monitoring

A Bermudan uses its contractual exercise dates. An American has a legal window
and requires a separate approximation grid for valuation. Each date on the grid
must be in the window, in strictly increasing order, with expiry last. Supply an
LSM configuration on the pricing template or with the pricing call. The training
and valuation engines must use independent sampling as required by the native
solver. Existing fixed-policy risk diagnostics apply.

```python
from pathlib import Path
import rust_pricing as rp

request = rp.PricingRequest.from_json(
    Path("fixtures/v3/pricing_request_american.golden.json").read_text())
source = rp.PricingSpecification.from_request(request, worker_threads=2)
builder = rp.ContractBuilder("put", 2, 1, "official.close")
option = builder.american("put", 100., 1., "2026-09-04", "2027-09-04")
task = option.pricing_task(source, american_grid=["2027-03-04", "2027-09-04"])
report = task.compile().evaluate().report
```

In Rust, pass `OptionFeature::Bermudan { dates }`, `OptionFeature::American { start }`
or `OptionFeature::ContinuousBarrier(terms)` to `InstrumentBuilder::vanilla_option`.
Build the native task with `Contract::pricing_task(&spec, operation, &settings)`.
`AdvancedMcSettings` carries the explicit grid, optional LSM override and bridge opt-in.
The [paired benchmark results](../../design/validation/contract-dsl/phase5-benchmark.json)
include seven measured runs per task and a 10,000-path register-reuse check.

Analysis lists each potential Bermudan payment date. These alternatives share a cashflow ID because exercising exhausts the right. American analysis records the legal window and its terminal payment.

On actual exercise, obtain the published fixing identity with
`exercise_observation_json(option_id, date)`. Supply a fixing and an
`exercise` event with that `exercise_id` to `advance`. A notice can arrive on a
legal American date outside the pricing grid. At expiry the lifecycle uses the
published terminal fixing. Prior payments remain outstanding until settled.

For a continuous barrier, construct `builder.continuous_barrier(...)` and pass
`brownian_bridge=True` to `pricing_task`. Crossing or touching the level in the
inclusive window counts as a hit. Rebates are absolute cash paid on the stated
payment date. A cumulative `barrier_status` event carries `barrier_id` and `hit`.
No-hit history must extend through valuation or the end of monitoring, whichever
comes first. Continuous historical monitoring cannot be inferred from safe
closing prices. See [ADR 0042](../../design/adr/0042-contract-exercise-and-monitoring.md)
for model assumptions and the supported native task shapes.
