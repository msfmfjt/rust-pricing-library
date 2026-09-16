# Library guide

[Documentation](../README.md) · [Model reference](../models/README.md)

The library exposes Rust APIs through the `pricing` crate and typed Python APIs
through `rust_pricing`. The workspace dependency chain is
`pricing-python → pricing → pricing-numerics`.

## Getting started

The [repository README](../../README.md#development) has the pinned toolchain,
build and verification commands. For local Python development, install the
repository's Rust toolchain, maturin and NumPy, activate a Python virtual
environment, then build the extension from the repository root:

```shell
python -m maturin develop --locked
python examples/python/european_bs.py
```

Start with the [European pricing example](../../examples/python/european_bs.py).
It builds market data, a product, a model, an engine configuration and a risk
request, then compiles a reusable plan and inspects estimates and diagnostics.
The [Python type stub](../../rust_pricing.pyi) lists the typed entry points.

Generate the Rust API reference from the repository root with:

```shell
cargo doc --locked --workspace --all-features --no-deps
```

Open `target/doc/pricing/index.html` for the financial API or
`target/doc/pricing_numerics/index.html` for independent numerical utilities.
Model-specific adapters and their limitations are indexed in the
[model reference](../models/README.md).

## Products, risk and diagnostics

| Topic | Reference | Runnable Python example |
| --- | --- | --- |
| European vanilla, MC/RQMC and Greeks | [European diagnostics](../models/european-bs-diagnostics-v0.1.md) | [European Black–Scholes](../../examples/python/european_bs.py) |
| Digital, Barrier, Asian and Lookback | [Numerical contracts](path-dependence-numerical-contracts-v0.1.md), [diagnostics](path-dependence-diagnostics-v0.1.md) | [Path dependence](../../examples/python/path_dependence.py) |
| American/Bermudan exercise and fixed-policy LSM risk | [Numerical contracts](early-exercise-numerical-contracts-v0.1.md), [diagnostics](early-exercise-diagnostics-v0.1.md) | [American LSM](../../examples/python/american_lsm.py) |
| Basket, Worst-of and Autocallable | [Multi-asset contracts](../models/multi-asset-v0.1.md) | [Multi-asset pricing](../../examples/python/multi_asset.py) |
| Local variance risk and VegaKT | [Local Volatility contracts](../models/local-vol-vegakt-numerical-contracts-v0.1.md), [diagnostics](../models/local-vol-vegakt-diagnostics-v0.1.md) | [Local Volatility/VegaKT](../../examples/python/local_vol_vegakt.py) |

Greeks, smoothing policies and supported products depend on the selected plan.
Use the corresponding numerical contract before combining product and model
features. Experimental model adapters have separate entry points and limitations.

## Compatibility and operations

- [Wire-schema compatibility](wire-schema-compatibility.md): accepted request and
  result versions, writer output, migrations and covariance layouts.
- [Three-crate migration](three-crate-migration.md): Rust dependency changes,
  supported public paths and Python compatibility.
- [Benchmarking](benchmarking-v0.1.md): native Rust and installed-wheel Python
  timing commands, host metadata and interpretation.

The [design index](../../design/README.md) contains development decisions,
acceptance reports and release planning. Required contributor checks are in
[CONTRIBUTING.md](../../CONTRIBUTING.md).
