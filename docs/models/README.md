# Model reference

[Documentation](../README.md) · [Library guide](../library/README.md)

These documents describe the implemented model conventions, calibration,
simulation, risk outputs and limitations. The accepted BS/Local Volatility
baselines and the experimental model adapters have different support boundaries;
consult the individual contracts for the selected product and plan.

## Constant and local volatility

| Model | Reference | Runnable Python example |
| --- | --- | --- |
| Black–Scholes and Black-76 | [European diagnostics](european-bs-diagnostics-v0.1.md), [constant-volatility product contracts](../library/path-dependence-numerical-contracts-v0.1.md) | [European Black–Scholes](../../examples/python/european_bs.py), [path dependence](../../examples/python/path_dependence.py) |
| Local Volatility, SSVI/eSSVI, Dupire and VegaKT | [Numerical contracts](local-vol-vegakt-numerical-contracts-v0.1.md), [diagnostics](local-vol-vegakt-diagnostics-v0.1.md) | [Local Volatility/VegaKT](../../examples/python/local_vol_vegakt.py) |

For analytical reference functions, see the
[Rust analytical API](../../crates/pricing/src/analytical.rs) and its
[implementation](../../crates/pricing/src/engine/analytic/closed_form.rs).

## Stochastic and rough volatility

| Model | Reference | Runnable Python example |
| --- | --- | --- |
| One-factor Bergomi LSV | [Coordinates, particle calibration and risk](lsv-numerical-contracts-v0.1.md) | [Bergomi LSV](../../examples/python/bergomi_lsv.py) |
| Two-factor Bergomi LSV | [Factor, correlation and calibration contracts](bergomi-two-factor-lsv-v0.1.md) | [Two-factor Bergomi](../../examples/python/multi_asset_bergomi_two_factor.py) |
| Rough Bergomi and rough-LSV | [Volterra scheme, calibration and risk](rough-bergomi-v0.1.md) | [Rough Bergomi](../../examples/python/rough_bergomi.py) |

## Hull–White hybrids

Start with the [equity/Hull–White numerical contracts](hull-white-numerical-contracts-v0.1.md)
and the [Python example](../../examples/python/hull_white_lsv.py).
The current default paid-cash convention is documented in
[affine dividends](hull-white-affine-dividends-v0.1.md); the older
[escrowed cash-dividend contract](hull-white-cash-dividends-v0.1.md) applies when
explicitly selected.

- [First-order AAD](hull-white-aad-v0.1.md): Spot, volatility, curve and
  recalibrated paired-target sensitivities.
- [Market-IV VegaKT](hull-white-vegakt-v0.1.md): quote coordinates, interpolation,
  risk units and limitations.
- [Bergomi and common Hull–White](bergomi-hull-white-v0.1.md): joint one-/two-factor
  volatility and stochastic-rate transitions for single- and multi-asset plans.

## Multi-asset models

| Configuration | Reference | Runnable Python example |
| --- | --- | --- |
| BS and Local Volatility | [Products, correlations and risk](multi-asset-v0.1.md) | [Multi-asset pricing](../../examples/python/multi_asset.py) |
| One-/two-factor Bergomi LSV | [Marginal calibration and joint drivers](multi-asset-lsv-v0.1.md) | [Multi-asset LSV](../../examples/python/multi_asset_lsv.py) |
| Bergomi LSV with a shared Hull–White rate | [Common-rate contracts](bergomi-hull-white-v0.1.md) | [Bergomi/Hull–White](../../examples/python/bergomi_hull_white.py) |
| Rough-LSV with a shared Hull–White rate | [Rough joint-process contracts](multi-asset-rough-bergomi-v0.1.md) | [Multi-asset rough-LSV](../../examples/python/multi_asset_rough_bergomi.py) |

## Local Correlation

[Particle Local Correlation](local-correlation-v0.1.md) calibrates a
state-dependent mixture of two PSD correlation schedules to a normalized basket
variance target. The initial adapter supports deterministic-rate BS/LV assets;
see the [Python example](../../examples/python/local_correlation.py) for
calibration, pricing and joint basket/constituent volatility risk.

Requirements, ADRs, implementation roadmaps and historical acceptance evidence
are indexed separately under [design](../../design/README.md).
