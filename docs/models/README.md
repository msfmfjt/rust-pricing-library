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
| Equity dynamics | Black–Scholes/Black-76, Local Volatility, LSV, Bergomi and rough Bergomi |
| Rate dynamics | Hull–White short-rate model and integrated-rate simulation |
| Volatility surfaces | SVI/SSVI/eSSVI, Dupire local variance and market-IV coordinates |
| Multi-asset and correlation | Basket state, PSD correlations and Particle Local Correlation |
| Cash flows | [Escrowed dividends and calibration coordinates](hull-white-cash-dividends.md) |

## Calculation methods

The calculation-method index collects simulation, calibration, differentiation
and path-treatment methods independently of the model that uses them.

[Open the calculation-method index](methods/README.md)

| Method family | Main topics |
| --- | --- |
| Monte Carlo and RQMC | Philox streams, antithetic sampling, Sobol' sequences and uncertainty |
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
| Bergomi with Hull–White | [Common-rate contracts](bergomi-hull-white.md) |
| Multi-asset Bergomi LSV | [Marginal calibration and joint drivers](multi-asset-lsv.md) |
| Multi-asset rough Bergomi LSV with Hull–White | [Rough joint-process contracts](multi-asset-rough-bergomi.md) |
| Two-factor Bergomi LSV | [Factor and calibration contracts](bergomi-two-factor-lsv.md) |

Product-specific path dependence and early-exercise contracts remain in the
[library reference](../library/README.md), because they are calculation and
payoff contracts rather than model components.
