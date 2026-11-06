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
boundaries. Exercise and continuous barriers are explicitly rejected in this
version. Cash-flow IDs must be unique across the complete source, including
mutually exclusive branches. Unknown references, duplicate observation IDs,
nonfinite constants, dates before dependencies and excessive depth/size fail
at construction. The owned tree has no reference cycles.

`analyze()` returns required observations (the fixing schedule), potential and
unconditionally determined cash flows, underlyings, currencies, barrier dates
and horizon. Potential amounts retain their expressions. Currencies are never
implicitly aggregated. A constant amount under a stochastic condition is not a
determined obligation.

`to_json()` / `from_json(bytes, JsonLimits)` preserve meaning and stable IDs.
[Contract schema v1](../../schemas/contract-v1/contract.schema.json) is separate
from frozen pricing request/result and graph formats. Decoding uses the existing
bounded duplicate-rejecting parser and then Rust semantic validation. Unknown
fields and versions fail. Observation declaration order is canonicalized by ID;
term and cash-flow order remain contractual. Fingerprints identify source data,
not algebraic equivalence. There is no migration from an earlier Contract
schema; future versions must retain v1 readers or publish an explicit migration.

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
date, fixings precede determination confirmations, termination and settlement;
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
