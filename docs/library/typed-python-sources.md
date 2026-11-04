# Typed Python pricing sources

Build complete sources with immutable Python settings, then select an operation
with `PricingTask.from_source`. The binding calls the same Rust source constructors,
capability checks and compilers as the JSON reader. It does not serialize inputs
to JSON internally or implement a second model dispatcher.

The [runnable example](../../examples/python/typed_pricing_sources.py) constructs
single-asset, stochastic-dividend and basket sources without JSON inputs.
The [type stub](../../rust_pricing.pyi) lists every signature and accepted type.

## Single-asset composition

Use existing `Product`, `Market`, `Model`, `Engine` and `RiskRequest` objects:

```python
model = rp.PricingModelSpec(
    rp.EquityModelSpec.stochastic_volatility(
        0.2, rp.VolatilityFactorSpec.bergomi(0.7, 0.2, -0.3)
    )
)
source = rp.PricingSpecification(
    valuation_date, product, market, model, engine, rp.RiskRequest(),
    rp.PricingNumerics(worker_threads=1, reduction_block_size=32, maximum_step=0.25),
)
task = rp.PricingTask.from_source(source, operation="aad")
report = task.compile().evaluate().report
```

| Configuration | Composition |
| --- | --- |
| `VolatilityFactorSpec` | `bergomi`, `bergomi_two_factor`, or `rough_bergomi(existing_rough_model)` |
| `EquityModelSpec` | `standard(Model)`, `stochastic_volatility`, `local_stochastic_volatility`, `rough_volatility`, or `rough_local_stochastic_volatility` |
| `RateModelSpec` | `deterministic()` or `hull_white(HullWhiteModel, ...)` |
| `PricingModelSpec` | Equity composition with optional rates; omission means deterministic rates |
| `LsvParticleConfig` | Particle count, calibration seed, bandwidth, minimum effective samples and reverse-trace policy |
| `PricingNumerics` | Execution policy, optional maximum step, and optional existing `LsmConfig` |

For LSV, pass an existing local-volatility `Model`, a factor and
`LsvParticleConfig`. Hybrid LSV also requires a matching `HullWhiteLsvTarget`
through `forward_density`; keep its market-IV provenance when market-IV risk
is needed. Existing `RoughVolatilityModel` constructors supply the rough-family
SV/LSV variants. `Product.from_graph(graph)` accepts a single-asset graph.

`maximum_step` is required for direct SV/rough SV and standard Hull–White
compositions. LSV uses its calibration grid. Unused maximum steps, incompatible
forward-density targets, extension risks and LSM combinations are rejected by
the existing Rust rules. Standard risks remain in `RiskRequest`; extension risks
are selected as task operations. A constructor does not add supported combinations.

## Multi-asset composition

`MultiAssetModelSpec(models, correlation, ...)` reuses `Model` and
`CorrelationSchedule`. Its optional inputs are:

- `lsv`: an ordered sequence of existing `MultiAssetLsvConfig`,
  `MultiAssetLsv2FactorConfig`, `MultiAssetRoughLsvConfig`, or `None` per asset.
- `driver_correlations`: the complete matrices in native driver order.
- `hull_white`: `MultiAssetHullWhiteConfig(rate_model, rate_correlations,
  lsv_targets=...)`. Omitted targets become one absent target per asset;
  supplied targets retain their order and market-IV provenance.
- `local_correlation`: an existing `LocalCorrelationConfig`, including its
  second driver matrices, calibration policy and optional hybrid density target.

Pass the model configuration with a `GraphProduct`, ordered `Market` inputs,
`Engine`, valuation date, positive maximum step and execution policy to
`MultiAssetSpecification`. Product, market and correlation underlying order must
agree. Assets share the product currency and discount curve. `None` means an
absent optional configuration in Python; the existing JSON writer continues to
use explicit presence tags in ordered arrays.

## Stochastic dividends

`BuehlerDividendModel` holds the reserve dynamics: mean reversion, equity linkage,
dividend volatility and equity/dividend correlation. Expected cash events remain
in the existing `Market` dividend schedule.

`StochasticDividendSpecification(equity, dividends, maximum_step=..., ...)`
combines it with a full price-only `PricingSpecification`. Supply one
`volatility_correlations` entry per Bergomi factor; BS uses an empty sequence.
BS/Hull–White requires an explicit `rate_correlation`. The source validates the
same model/grid combinations as the JSON path.

## Ownership, validation and replay

All new configuration/source classes are frozen and own cloned Rust inputs.
Nested Python lists can be reused or changed after construction without changing
the configuration or task. Configurations describe source inputs, not compiled
surfaces or mutable particle workspaces. Their exact accepted inputs can be
inspected through the resulting task's canonical `to_json()`.

Primitive validated types reject invalid values during construction. Configuration
records defer combination checks until their complete source is constructed.
Native grid-dependent covariance, resource and calibration checks still occur
at compilation; a conditional capability is not numerical acceptance.
New native validation failures carry `PricingError` metadata with
`stage="validation"`; existing reused builders retain their existing exception
classes. Typed-argument mismatches use `TypeError`; reused LSV configuration
extraction retains its `ValidationError` contract. See
[structured errors](python-errors.md) for codes, fields and cause chains.

`PricingTask.from_source` accepts any of the three source classes. Existing
`PricingSpecification.from_request`, `PricingSpecification.from_json`,
`PricingTask.from_specification` and `PricingTask.from_json` retain their
signatures and behavior. A typed source and an equivalent parsed source have
identical canonical JSON, source/task/plan fingerprints and numerical reports
under the existing replay guarantees. There are no schema or fingerprint-domain
changes. Serialization and [capabilities/reports](pricing-tasks.md) stay shared.

[Contract tests](../../tests/python/test_typed_pricing_sources.py) compare against
pre-existing golden JSON, including raw tolerated correlation asymmetry, density
provenance, joint calibration, full risks and standard errors. These small
contract fixtures supplement the existing numerical accuracy gates.
