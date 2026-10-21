# ADR 0029: Expose validated product graphs across public boundaries

Date: 2026-10-21. Status: implemented; generic product source API.

## Context

The engine and Rust multi-asset facade already accept source payoff graphs,
but the single-asset product union, Python builders and JSON only expose named
products. Architecture review item 2 requires a reusable product source so a
new composition of supported operations needs no wrapper or serialization tag.

## Decision

Add immutable `GraphProduct`: currency, ordered underlying IDs, `SourceGraph`
and one payment date per output. Validate the existing DAG/resource rules and
each cash flow's observation dependencies before constructing the product.
Validate valuation-dependent constraints when a pricing request/plan is built.
Historical fixings are literal source nodes. Keep existing named builders and
typed source/compiled separation under ADR 0011.

Add `ProductSpec::Graph`, Python `SourceGraphBuilder` and immutable `GraphProduct`,
plus conversion to/from the existing `MultiAssetProduct`. Python delegates to
Rust validation; it adds no evaluation callbacks. Standard model dispatch stays
outside numerical loops. Single-asset requests reject multi-asset sources.

Lower deterministic-rate cash flows to a single discount-weighted aggregate at
compile time. This retains covariance between cash flows in price/risk sampling
statistics and reuses existing numerical kernels. Stochastic-rate single-asset
adapters reject different payment dates unless rate volatility is identically
zero; the existing multi-asset facade retains its per-cash-flow implementation.
Explicit smooth opcodes control smoothing. A live hard indicator rejects
pathwise Greeks. Do not infer early-exercise or continuous-barrier semantics.

## Wire and reproducibility impact

- Introduce strict `product_graph` v1 source JSON and fingerprint domain.
  Canonicalize node declaration order, preserve IDs and contractual ordering.
- Introduce `pricing_specification` v2 with the graph variant. Read v1 built-in
  sources, write v2 and compute source fingerprints at version 2. Historical v1
  schemas/fixtures remain frozen; callers must expect a normalized source's
  fingerprint to change when upgrading from the earlier implementation.
- Preserve legacy request/result v1–v3 schemas, fixtures and built-in request/
  native-plan identities. Reject graphs at legacy request JSON boundaries.
  Use a separate internal graph-request identity for native plan/result replay.
- Save no compiled state. Multi-asset product replay does not claim to save the
  separate markets, correlations or model/calibration settings.
- Add no runtime dependency and retain the three-crate dependency direction.
- Adding the `Graph` variant requires downstream exhaustive Rust matches on
  `ProductSpec` to add an arm. Existing named-builder signatures are unchanged.

## Verification and extension boundary

Compare graph vanilla price, Delta and Vega with native builders under MC/RQMC;
test multi-payment discounting and sampling statistics, stochastic-rate guards,
multi-asset conversion, immutable Python construction, strict rejection and
source/result replay. Keep schemas, fixtures, documentation, wheel/stub contracts
and source archives in existing CI gates. New primitives or observation laws
require their own numerical review; graph composition alone does not broaden
existing model or risk support.

See the [usage and extension guide](../../docs/library/product-graphs.md).
