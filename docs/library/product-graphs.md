# Generic product graphs

`GraphProduct` connects the existing payoff graph to Rust `ProductSpec`, Python
builders, JSON source replay and the multi-asset facade. Products made from the
existing graph operations no longer need their own wrapper or wire variant.
Existing vanilla, Asian, barrier, basket and autocall builders remain available.
Rust callers that exhaustively match `ProductSpec` must handle its new `Graph`
variant; existing builder calls retain their signatures.

The [runnable example](../../examples/python/product_graph.py) constructs a
coupon and a capped call with different observation and payment dates, saves
the complete specification, and evaluates price, Delta and Vega. The source
contract is recorded in [ADR 0029](../../design/adr/0029-product-graph-source.md).

## Construction and ownership

In Rust, construct `pricing::product::GraphProduct::new(currency, underlyings,
source_graph, payment_dates)`, then use `ProductSpec::Graph(product)`. The graph
contains typed `SourceOpcode` nodes. Each output has exactly one payment date;
repeated output nodes are distinct cash flows. All amounts are in the declared
currency. Underlying order is retained for multi-asset market/model alignment.

In Python the builder returns node IDs. `finish` validates and copies the source
into an immutable `GraphProduct`; further builder mutations cannot change it.

```python
import rust_pricing as rp

b = rp.SourceGraphBuilder()
spot = b.spot(1, "2027-09-04")
call = b.maximum(b.subtract(spot, b.literal(100.)), b.literal(0.))
product = b.finish([call], ["2027-09-06"], currency_id=2, underlying_ids=[1])
saved = product.to_json()
restored = rp.GraphProduct.from_json(saved)
single_asset_product = rp.Product.from_graph(restored)
multi_asset_product = rp.MultiAssetProduct.from_graph(restored)
```

Arithmetic, hard/smoothed min, max and indicator, negation, spot observations,
pre-dividend observations and barrier hit-state composition are available.
`spot(..., pre_dividend=True)` uses the existing before-jump convention.
`minimum`, `maximum` and `indicator` accept `smoothing_half_width`; this is part
of the contractual graph. Request-level smoothing does not rewrite a graph.
The complete signatures are in [rust_pricing.pyi](../../rust_pricing.pyi).

Known historical fixings must be literal nodes. A live observation before the
valuation date or a payment before valuation is rejected by the pricing facade.
An observation on valuation date remains spot-sensitive unless replaced by a
literal. For each output, validation checks its own observation dependencies
against its payment date; another output may observe a later date.

Validation also rejects cycles, duplicate node IDs, missing references,
undeclared underlyings, invalid smoothing widths and resource-limit breaches.
The existing default graph limits apply, including at most 64 outputs.
Dead source nodes are retained for identity and still undergo structural
validation; they do not create live observations or risk requirements.

## Pricing boundaries

| Entry point | Graph behavior |
| --- | --- |
| `PricingRequest` / `PricingSpecification` | Exactly one declared underlying; existing market/model and engine checks apply |
| Deterministic-rate standard, SV, LSV and rough plans | Multiple cash flows are compiled into one discounted aggregate |
| Single-asset Hull–White with nonzero rate volatility | Outputs must share a payment date; multiple payment dates are rejected |
| `MultiAssetPricingPlan` / Python `MultiAssetPlan` and existing extensions | Ordered multi-asset graph and per-output payment dates; existing stochastic-rate cash-flow treatment applies |

For deterministic rates, the compiler weights output `i` by
`P(0, payment_i) / P(0, last_payment)` and sums the weighted outputs. The existing
executor applies the final payment discount. Means, variance and pathwise risks
therefore use the aggregate cash flow on each sampled path, retaining covariance
between cash flows. No new numerical kernel or dispatch in path loops is added.

Curve AAD also differentiates the compile-time discount ratios. The compiler
retains each cash-flow amount's tape slot and the ratio's log-discount-factor
weights; evaluation reuses the existing payoff forward values to add this
market contribution before sampling statistics are reduced. This includes
shared and constant-folded outputs, payments at valuation, interpolation and
extrapolation, and preserves covariance in curve-risk standard errors. These
weights are compiled state and are never saved in the product source.

A live hard indicator rejects pathwise risk requests. Use an explicit smoothed
indicator when that approximation is intended; hard min/max retain their
existing almost-everywhere derivative convention. Existing model-specific risk
restrictions still apply. Graphs do not implicitly provide continuous barrier
monitoring or early exercise: those require the corresponding existing product
and plan contracts. Invalid path arithmetic is an evaluation error.

Division AAD retains the usual `-seed * numerator / denominator^2` arithmetic
when both intermediate products are normal binary64 values. Outside that range,
it separates binary exponents before multiplying, so a finite sensitivity is
not lost merely because the denominator's square overflows or underflows.
Zero denominators and genuinely nonfinite adjoints still fail. This does not
make every intermediate in an arbitrary graph representable; see the
[division numerical review](../../design/validation/graph-division-review-2026-10-27.md).

Rust `MultiAssetProduct::from_product_graph` and `to_product_graph`, and Python
`MultiAssetProduct.from_graph` and `to_graph`, bridge the existing multi-asset
builders. The product graph document stores the product only; multi-asset
markets, correlations and model/calibration inputs remain separate plan inputs.

## JSON and compatibility

The independent `product_graph` v1 document contains currency, ordered
underlyings, typed nodes and `{output_node, payment_date}` cash flows.
[Its schema](../../schemas/product-graph-v1/product_graph.schema.json) is also
available through `current_product_graph_schema()` in Rust and
`product_graph_json_schema()` in Python. Readers reuse the bounded JSON parser
and reject unknown and duplicate fields. Rust additionally validates graph
semantics that JSON Schema cannot express.

`pricing_specification` v2 embeds this source as
`product: {"type": "graph", "graph": ...}`. Its reader still accepts v1 built-in
products; all writers normalize to v2. Source fingerprints now use the v2 domain
version, including for an imported v1 specification. The historical v1 schema
and fixtures stay frozen. Native built-in plan fingerprints are unchanged.

Legacy `pricing_request` / `pricing_result` v1–v3 contracts are unchanged.
Legacy request readers and writers reject graph products. To save a graph
request, use `PricingSpecification.from_request(...).to_json()`; native result
JSON remains available. A graph request uses a separate internal fingerprint
domain, so its identity cannot collide with a legacy request. Save the source
specification with the result to retain the complete replay inputs.

Graph source fingerprints ignore JSON formatting and node declaration order
(nodes serialize by ID), but retain node IDs, dead nodes, underlying order,
output order, payment dates and smoothing parameters. They identify source
data, not mathematical equivalence of arbitrary expressions.

Single-asset compiled plans with multiple payment dates include the
`graph-cashflow-discount-pullback-v1` fingerprint tag for the curve-AAD behavior.
This does not change graph source identities, built-in plan identities or
single-payment-date graph plan identities.

## Extending products

Compose new payoffs with existing operations and expose an optional convenience
builder; Rust, Python and JSON already share the generic source boundary. A new
primitive still needs an explicit opcode, compiler/evaluator/adjoint behavior,
typed wire mapping and a schema compatibility decision. A new exercise or
observation law needs a separately reviewed engine contract. Keep compiled
tapes, market-dependent discount weights and execution caches out of saved
source data.

Conformance covers [Rust](../../crates/pricing/tests/product_graph.rs),
[Python](../../tests/python/test_product_graph.py), the wheel API contract and
[offline schemas](../../scripts/check_product_graph_json_schema.py).
