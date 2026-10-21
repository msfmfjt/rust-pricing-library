# Complete pricing source specifications

`PricingSpecification` is an additive source API for saving the actual equity
model, rate model, calibration inputs and numerical settings with a pricing
request. Rust and Python compile this immutable source through the same existing
pricing plans. Neither compiled workspaces nor particle states are serialized.

The [runnable Python example](../../examples/python/pricing_specification.py)
loads four [source fixtures](../../fixtures/specification-v1), compiles them and
reports their estimates. The design decision is recorded in
[ADR 0028](../../design/adr/0028-pricing-specification.md).

## Supported compositions

| Equity source | Rates | Common evaluation | Time grid |
| --- | --- | --- | --- |
| Existing `ModelSpec` (BS, Black-76, LV, Bass-LV) | Deterministic | Existing standard price/risk and supported LSM behavior | Existing model grid |
| BS | Hull–White | Price only | Required `maximum_step` |
| 1F/2F Bergomi or rough Bergomi SV | Deterministic or Hull–White | Price only | Required `maximum_step` |
| 1F/2F Bergomi or rough Bergomi LSV | Deterministic or Hull–White | Price only | Local-variance target grid and existing refinement |
| Rough Heston, lifted Heston, quadratic rough Heston, mixed rough Bergomi, rough SABR, RFSV | Deterministic | Price only, direct or LSV | Direct: required `maximum_step`; LSV: target grid |

The selected backend's product, correlation, resource and numerical restrictions
still apply. For example, rough SABR LSV requires beta one, and a correlation
matrix must be positive semidefinite. These remain compiler checks. This API
does not broaden any model's numerical acceptance or production support status.
Existing specialized risk, stochastic-dividend, multi-asset and dual-bound APIs
remain separate entry points.

The constructor rejects an unsupported composition, missing or unused step,
inconsistent Hull–White LSV target, extension Greeks, extension LSM, and unused
AAD checkpoint/tile/width-ladder settings. Payoff smoothing is retained and
evaluated by the existing backend. Unsupported input never falls back to BS or
silently drops risk requests.

## Rust

```rust
use pricing::{PricingSpecification, parse_request_json, JsonLimits};
use pricing::mc::ExecutionPolicy;

fn example(bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let request = parse_request_json(bytes, JsonLimits::DEFAULT)?;
    let source = PricingSpecification::from_request(
        request, ExecutionPolicy::new(2, Some(256))?,
    );
    let saved = pricing::specification_to_json(&source)?;
    let restored = pricing::parse_specification_json(saved.as_bytes(), JsonLimits::DEFAULT)?;
    let result = restored.compile()?.evaluate()?;
    println!("{} +/- {}", result.value(), result.standard_error());
    Ok(())
}
```

For an extension, construct `PricingModelSpec::new(equity, rates)` with
`EquityModelSpec`, `RateModelSpec` and, for Bergomi models,
`VolatilityFactorSpec`. `source.with_model(model, maximum_step)` preserves the
product, market, engine and risk while validating the new composition. Or build
from these typed inputs directly using `PricingSpecification::new` and
`PricingNumerics::new(policy)`. Public composition enums and configurable
structs are non-exhaustive so future variants can be added at this boundary.

Hull–White `vol_rate_correlations` follows the volatility driver's declared
order: no entries for BS, one for 1F/rough Bergomi, two for 2F Bergomi. The
equity/volatility correlation stays with the volatility factor. All rates and
correlations are saved even when a loading is zero.

LSV stores the local-variance target, any reporting-IV basis, particle count,
calibration seed, bandwidth, minimum effective sample count and trace policy.
Hull–White LSV also requires a matching forward-density target. If that target
retains a market-IV surface, the source stores it and the reader verifies that
it reproduces both target arrays. Direct rough models store their actual initial
variance or forward-variance curve; they expose no placeholder BS volatility.

## Python

```python
import rust_pricing as rp

source = rp.PricingSpecification.from_request(
    request, worker_threads=2, reduction_block_size=256,
)
saved = source.to_pretty_json()
restored = rp.PricingSpecification.from_json(saved)
compiled = restored.compile()
result = compiled.evaluate()
print(result.value, result.standard_error, result.specification_fingerprint)
native = result.native_result
```

Extension compositions are currently constructed from typed JSON through
`from_json`; the [fixtures](../../fixtures/specification-v1) show the input
shape. Python objects are immutable, compilation/evaluation release the GIL,
and validation issues identify `pricing_specification` and its schema version.
The exact public signatures are in [rust_pricing.pyi](../../rust_pricing.pyi).

`native_result` preserves the existing `PricingResult`, `HullWhitePrice` or
`LsvPrice`, including native diagnostics and risk conventions. In Rust,
`SpecificationOutput` retains the corresponding native values; its `Hybrid`
variant also carries the existing pure-SV `HullWhitePrice`. LSV standard error
is conditional on the realized calibration and excludes calibration noise.
No universal result-wire format is introduced in this step.

## JSON identity and compatibility

The document kind is `pricing_specification`. The current writer uses version
`2`, adding [generic product graphs](product-graphs.md); the reader also accepts
v1 built-in products and normalizes them to v2. The v1 schema and fixtures remain
frozen. The [current Draft 2020-12 schema](../../schemas/specification-v2/pricing_specification.schema.json)
is also exposed by `current_specification_schema()` and
`specification_json_schema()`. Existing request/result v1/v2/v3 readers,
writers, fixtures and fingerprint domains are unchanged. The old request
reader deliberately rejects the new document kind.

The source fingerprint hashes a canonical, domain-separated representation of
the complete source, including explicit worker count and reduction-block size. The source fingerprint
version is now 2, so importing a v1 source changes its source fingerprint;
financial inputs and native plan fingerprints are preserved.
It is distinct from the unchanged native plan fingerprint. JSON whitespace and
key order have no effect. Changes to model parameters, calibration inputs,
market data, seeds or numerical settings change the source identity.

Wrapping a legacy request preserves its migration provenance in memory. Saving
it as a new source document normalizes that history: a reloaded document has
its own source identity, not the original v1/v2 bytes. Financial inputs and
calculation behavior are retained. Reproducible replay remains subject to the
existing library-version, platform and numerical-policy guarantees; a source
fingerprint is not a promise of bit identity across arbitrary implementations.

Readers reuse the bounded JSON parser, reject duplicate and unknown fields,
and validate constructors and composition rules. JSON Schema checks structure
and expressible composition rules; Rust performs array-shape, calibration,
market and numerical consistency checks as well.

## Extending the boundary

Add a validated domain variant, a strict source DTO and explicit compiler
dispatch to an existing or separately reviewed backend. Keep dispatch outside
path/time-step loops. Add source round trips, native-result/plan equivalence,
fingerprint sensitivity and rejection cases; update the schema generator,
Python contract and support table together. Changing persisted numerical
meaning requires an explicit schema/migration decision. Do not serialize
execution caches or add string-keyed model parameter bags.

Validation is in [Rust conformance tests](../../crates/pricing/tests/pricing_specification.rs),
[Python tests](../../tests/python/test_pricing_specification.py), the wheel
contract gate, and [offline schema checks](../../scripts/check_specification_json_schema.py).
