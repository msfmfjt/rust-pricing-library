# Model reference

Published sources for the models and numerical methods are collected in the
[model and numerical-method bibliography](../references.md). Each detailed
page repeats the entries relevant to its own scope.

[Documentation](../README.md) · [Library guide](../library/README.md)

These documents describe the implemented model conventions, calibration,
simulation, risk outputs and limitations. The accepted BS/Local Volatility
baselines and the experimental model adapters have different support boundaries;
consult the individual contracts for the selected product and plan.

## Constant and local volatility

| Model | Reference | Runnable Python example |
| --- | --- | --- |
| Black–Scholes and Black-76 | [European diagnostics](european-bs-diagnostics.md), [constant-volatility product contracts](../library/path-dependence-numerical-contracts.md) | [European Black–Scholes](../../examples/python/european_bs.py), [path dependence](../../examples/python/path_dependence.py) |
| Local Volatility, SSVI/eSSVI, Dupire and VegaKT | [Numerical contracts](local-vol-vegakt-numerical-contracts.md), [diagnostics](local-vol-vegakt-diagnostics.md) | [Local Volatility/VegaKT](../../examples/python/local_vol_vegakt.py) |

For analytical reference functions, see the
[Rust analytical API](../../crates/pricing/src/analytical.rs) and its
[implementation](../../crates/pricing/src/engine/analytic/closed_form.rs).

## Stochastic and rough volatility

| Model | Reference | Runnable Python example |
| --- | --- | --- |
| One-factor Bergomi LSV | [Coordinates, particle calibration and risk](lsv-numerical-contracts.md) | [Bergomi LSV](../../examples/python/bergomi_lsv.py) |
| Two-factor Bergomi LSV | [Factor, correlation and calibration contracts](bergomi-two-factor-lsv.md) | [Two-factor Bergomi](../../examples/python/multi_asset_bergomi_two_factor.py) |
| Rough Bergomi and rough-LSV | [Volterra scheme, calibration and risk](rough-bergomi.md) | [Rough Bergomi](../../examples/python/rough_bergomi.py) |

## Hull–White hybrids

Start with the [equity/Hull–White numerical contracts](hull-white-numerical-contracts.md)
and the [Python example](../../examples/python/hull_white_lsv.py).
The current default paid-cash convention is documented in
[affine dividends](hull-white-affine-dividends.md); the older
[escrowed cash-dividend contract](hull-white-cash-dividends.md) applies when
explicitly selected.

- [First-order AAD](hull-white-aad.md): Spot, volatility, curve and
  recalibrated paired-target sensitivities.
- [Market-IV VegaKT](hull-white-vegakt.md): quote coordinates, interpolation,
  risk units and limitations.
- [Bergomi and common Hull–White](bergomi-hull-white.md): joint one-/two-factor
  volatility and stochastic-rate transitions for single- and multi-asset plans.

## Multi-asset models

| Configuration | Reference | Runnable Python example |
| --- | --- | --- |
| BS and Local Volatility | [Products, correlations and risk](multi-asset.md) | [Multi-asset pricing](../../examples/python/multi_asset.py) |
| One-/two-factor Bergomi LSV | [Marginal calibration and joint drivers](multi-asset-lsv.md) | [Multi-asset LSV](../../examples/python/multi_asset_lsv.py) |
| Bergomi LSV with a shared Hull–White rate | [Common-rate contracts](bergomi-hull-white.md) | [Bergomi/Hull–White](../../examples/python/bergomi_hull_white.py) |
| Rough-LSV with a shared Hull–White rate | [Rough joint-process contracts](multi-asset-rough-bergomi.md) | [Multi-asset rough-LSV](../../examples/python/multi_asset_rough_bergomi.py) |

## Local Correlation

[Particle Local Correlation](local-correlation.md) calibrates a
state-dependent mixture of two PSD correlation schedules to a normalized basket
variance target. It supports deterministic-rate BS/LV and Bergomi LSV assets,
and shared Hull–White configurations with BS, Bergomi LSV or rough-LSV.
See the [BS/LV example](../../examples/python/local_correlation.py) and the
[LSV/Hull–White example](../../examples/python/local_correlation_lsv_hw.py)
for calibration, pricing and joint basket/constituent volatility risk.
