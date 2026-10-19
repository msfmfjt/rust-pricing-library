# Model and calculation reference

This section is organized by reusable model components and calculation
methods. Pages describing combinations of components are kept separately as
composition guides.

[Documentation](../README.md) · [Library guide](../library/README.md) ·
[Product reference](../products/README.md) · [Bibliography](../references.md)

## Model components

The component index is the starting point when selecting an asset, rate,
surface, correlation or cash-flow component.

[Open the model component index](components/README.md)

| Component family | Main topics |
| --- | --- |
| Equity dynamics | Black–Scholes/Black-76, Local Volatility, LSV, Bergomi, rough Bergomi and [six additional experimental rough families](rough-volatility-families.md) |
| Multi-marginal local volatility | [Bass-LV calibration and simulation](bass-local-volatility.md) |
| Rate dynamics | Hull–White short-rate model and integrated-rate simulation |
| Volatility surfaces | SVI/SSVI/eSSVI, Dupire local variance and market-IV coordinates |
| Multi-asset and correlation | Basket state, PSD correlations and Particle Local Correlation |
| Cash flows | [Escrowed dividends and calibration coordinates](hull-white-cash-dividends.md), [stochastic cash dividends](stochastic-dividends.md) |

## Calculation methods

The calculation-method index collects simulation, calibration, differentiation
and path-treatment methods independently of the model that uses them.

[Open the calculation-method index](methods/README.md)

| Method family | Main topics |
| --- | --- |
| Monte Carlo and RQMC | Philox streams, antithetic sampling, Sobol' sequences and uncertainty |
| Fourier pricing | [Rough/Lifted Heston forward transforms, Riccati integration and European inversion](rough-heston-fourier.md) |
| Fourier forward risk | [Fixed-model Forward Delta/Gamma and diagnostic limits](rough-heston-fourier-greeks.md) |
| Calibration | Particle calibration, leverage fitting and surface interpolation |
| Early exercise | Least-squares Monte Carlo, regression QR and fixed-policy valuation |
| Differentiation and risk | AAD, finite-difference validation and VegaKT |
| Path treatment | Smoothing, barrier bridges and discrete-event handling |

## Composition guides

These pages describe supported combinations of otherwise independent
components. They are useful for implementation limits and runnable examples,
but are not the primary organization of the reference.

[Open the composition-guide index](compositions/README.md)

| Composition | Guide |
| --- | --- |
| Rough Bergomi with stochastic cash dividends | [Price-only hybrid composition](stochastic-dividends.md#rough-bergomi-price-composition) |
| Bergomi with Hull–White | [Common-rate contracts](bergomi-hull-white.md) |
| Multi-asset Bergomi LSV | [Marginal calibration and joint drivers](multi-asset-lsv.md) |
| Multi-asset rough Bergomi LSV with Hull–White | [Rough joint-process contracts](multi-asset-rough-bergomi.md) |
| Two-factor Bergomi LSV | [Factor and calibration contracts](bergomi-two-factor-lsv.md) |

Product-specific path dependence and early-exercise contracts remain in the
[library reference](../library/README.md), because they are calculation and
payoff contracts rather than model components.

- [Stochastic cash dividends with Hull–White](stochastic-dividends-hull-white.md):
  correlated discounted-cash forecasts, constant residual volatility and explicit basic AAD.

## Experimental Heston price calibration

The [multi-expiry price-calibration API](heston-calibration.md) fits selected
Rough/Lifted Heston parameters using analytic Fourier Jacobians. Fit status is
separate from solver termination; this is not an IV or general six-family fitter.

- [Heston IV grid validation and staged recalibration](heston-iv-refinement.md)

- [Heston IV disjoint holdout validation](heston-iv-holdout.md)

[Six-family MC Spot Delta and particle LSV AAD](rough-family-aad-lsv.md) documents the explicit risk coordinates and experimental limits.

[Six-family LSV physical Spot Delta](rough-family-lsv-spot-delta.md) distinguishes fixed leverage from a sticky-relative-local-variance target.

[Additional rough-family market-IV quote risk](rough-family-market-iv.md) includes the discrete Dupire and particle-calibration transpose.

[Explicit rough-family finite-bump MC Gamma](rough-family-gamma.md)

[Pure-SV Heston MC parameter adjoints](heston-mc-parameter-risk.md)

[Pure-SV MC Hurst sensitivity](heston-mc-hurst-risk.md) is available separately.

- [Heston LSV recalibrated parameter risk](heston-lsv-parameter-risk.md)

- [Mixed rough Bergomi eta/rho MC and LSV risk](mixed-bergomi-parameter-risk.md)
- [Quadratic rough Heston scalar/Hurst MC and recalibrated LSV risk](quadratic-heston-parameter-risk.md).

- [Mixed rough Bergomi weight and Forward Variance Curve risk](mixed-bergomi-shape-risk.md).
