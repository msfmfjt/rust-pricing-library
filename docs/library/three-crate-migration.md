# Three-crate migration

Baseline: `0488c415c03f93bcbf7a5bfff43c3a948938e2cf`, fetched from remote `main`
before implementation. Branch: `refactor/three-crate-workspace`.

The workspace changes from ten crates to exactly three:
`pricing-python → pricing → pricing-numerics`. No model, product, Greek,
calibration method, numerical algorithm, schema version, or algorithm ABI is added.
The baseline includes Bergomi LSV, Hull–White, affine cash dividends and rough
Bergomi; those already-merged implementations are retained. There were no open
PRs when the baseline was fixed.

## Public paths and ownership

| Former crate / entry | Definition or execution owner | Supported public entry |
| --- | --- | --- |
| `pricing-core` | `pricing/src/core` | `pricing::core` |
| `pricing-market` | `pricing/src/market` | `pricing::market` |
| `pricing-product` | `product` definitions; `engine/payoff` compiler/tape/reverse | `pricing::product` |
| `pricing-models` | `models` value types; `engine/processes` simulation | `pricing::models` |
| `pricing-mc` | `engine/mc`, `sampling`, `processes`, `calibration`, `payoff` | `pricing::mc`, including `hull_white` and `lsv` |
| `pricing-aad` | `engine/aad` | AAD policy/workspace types re-exported from `pricing::mc` |
| `pricing-risk` | `risk` public types; `engine/risk` computation | `pricing::risk` |
| Root request/result/error | `api` | Existing `pricing::*` exports |
| `SimulationPlan`, `PricingPlan` | `engine/plan` and private inherent implementations | Existing root types, `compile`, `evaluate`, pricing functions |
| `pricing::hull_white`, `pricing::lsv` | `engine/risk` plan orchestration | Same public modules, types and methods |
| `pricing::analytical` | `engine/analytic` | Same public module and functions |
| JSON | `wire` | Same root parsing, writing, migration and fingerprint API |
| `pricing-numerics` | Independent numerical crate | Same numerical functions |
| `pricing-python` | Same Python adapters | `import rust_pricing` and all existing public names |

The old `pricing` facade is the compatibility boundary. Re-exports refer to the
same moved types; no duplicate financial types or wrapper crates are introduced.
The compile-time [facade inventory](../../crates/pricing/tests/facade_compatibility.rs)
pins 377 baseline public names across 14 module paths. Rust type diagnostic names
and rustdoc source URLs can change with their defining module; these are not wire
identifiers. Existing signatures, runtime semantics, JSON and Python contracts
remain the target of the acceptance and compatibility checks.

Direct dependencies on the seven removed crates **must migrate**; this is not a
fully nonbreaking workspace change. For example:

```toml
# Before
[dependencies]
pricing-market = { path = "../rust-pricing-library/crates/pricing-market" }
pricing-product = { path = "../rust-pricing-library/crates/pricing-product" }

```

After replacing the preceding dependency entries:

```toml
[dependencies]
pricing = { path = "../rust-pricing-library/crates/pricing" }
```

```rust
// Before
use pricing_market::LogLinearDiscountCurve;
use pricing_product::EuropeanVanillaSpec;

// After
use pricing::market::LogLinearDiscountCurve;
use pricing::product::EuropeanVanillaSpec;
```

Old AAD callers can use
`pricing::mc::{AadConfigError, AadTilePolicy, AlignedF64Buffer, CheckpointPolicy, SoaWorkspace}`.
The old dependency-role marker helpers are retained for source compatibility;
`pricing_numerics::foundation_role()` returns its previous constant without
importing financial core types.

## Features, tests and data

The baseline `pricing` manifest declared no features. Its unconditional
`pricing-risk` dependency enabled `pricing-mc/aad`, including the AAD import,
executor method, AAD test and `aad_enabled()` helper. All four are now
unconditional. Default, all-feature and no-default-feature financial builds retain
that behavior. This is not the introduction of an optional `pricing/aad` feature.
The old standalone MC-without-AAD package configuration disappears with that
package; direct users migrate to the existing facade functionality.

Python retains `default = []`, `extension-module = ["pyo3/extension-module"]`,
`rust_pricing`, PyO3 0.29.2 and the same maturin build command. As in baseline CI,
all-feature Rust tests exclude `pricing-python`; its unit tests run separately
without extension-module linking. Default workspace tests are also checked.

All old unit tests move with their implementation. Standalone integration targets
are renamed to avoid collisions:

| Old package and test target | New `cargo test -p pricing --test ...` target |
| --- | --- |
| `pricing-market`: `essvi_reference` | `market_essvi_reference` |
| `pricing-market`: `market_iv` | `market_market_iv` |
| `pricing-market`: `standard_ssvi_reference` | `market_standard_ssvi_reference` |
| `pricing-models`: `hull_white`, `rough_bergomi` | `models_hull_white`, `models_rough_bergomi` |
| `pricing-mc`: `hull_white`, `lsv`, `rough_bergomi` | `mc_hull_white`, `mc_lsv`, `mc_rough_bergomi` |

The baseline has eight Rust examples (four benchmarks, four replays), all already
under `pricing`, and no separate Cargo `benches` targets. They remain detectable
as examples. Existing fixture/schema bytes and Python sources, examples and type
stubs are unchanged. The Joe–Kuo binary moves from `crates/pricing-mc/data` to
`crates/pricing/data`, retaining its digest and third-party notice. The source
archive and wheel SBOM checkers now require the three-crate graph and moved asset.

## Deliberate layout choices

The former 10,708-line `monte_carlo.rs` is a four-line compatibility entry.
Compiled state, compile, sampling, constant-volatility path/AAD calculations,
local-volatility forward/reverse, payoff/barrier handling, LSM path matrices and
risk reporting have separate implementation files. The engine remains private.
Risk invokes execution and collects results; lower modules do not assemble risk
reports. Calibration and its reverse are adjacent for both Bergomi and hybrid LSV.

Small stateless mathematical methods stay with the existing model value types;
no model representation, QR abstraction or RNG implementation is redesigned.
Existing public compiled-plan/tape types remain public through their original
entry points. They are distinct from request/specification types and are not
serialized as executable state. The short high-level `PricingPlan` orchestration
stays in `engine/risk/pricing.rs` to keep width-ladder valuation adjacent to risk.
Shared `SimulationPlan` state is in `engine/plan`.

The dependency checker retains its existing manifest/path/version checks, adds
the exact three-crate graph and scans selected source boundaries. It is a small
text guard supplementing Rust visibility, not a complete dependency parser.
Historical reference records retain their original crate names; current
architecture and conformance references use the new modules.
