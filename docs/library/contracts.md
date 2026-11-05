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
