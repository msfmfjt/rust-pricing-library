# Calculation methods

These pages organize the numerical methods independently of the model
component that consumes them. The detailed calculation specifications remain
authoritative for the exact implementation policy and tolerances.

## Monte Carlo and quasi-Monte Carlo

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Monte Carlo, antithetic sampling and uncertainty | [Benchmark baseline](../../library/benchmarking.md) | Pricing and validation |
| Randomized Sobol' QMC | [European diagnostics](../european-bs-diagnostics.md), [early-exercise calculation specifications](../../library/early-exercise-calculation-specifications.md) | Pricing and LSM training/valuation |
| Philox counter-based streams | [Benchmark baseline](../../library/benchmarking.md) | Reproducible parallel simulation |
| Volterra hybrid scheme | [Rough Bergomi contracts](../rough-bergomi.md) | Rough-volatility paths |

## Calibration and interpolation

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Particle calibration | [LSV calculation specifications](../lsv-calculation-specifications.md), [Local Correlation](../local-correlation.md) | Leverage and local-correlation targets |
| Surface interpolation and local variance | [Local Volatility calculation specifications](../local-vol-vegakt-calculation-specifications.md) | SSVI/eSSVI and Dupire |

## Early exercise and linear algebra

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Least-squares Monte Carlo | [Early Exercise calculation specifications](../../library/early-exercise-calculation-specifications.md) | American and Bermudan exercise |
| Column-pivoted Householder QR | [Early Exercise calculation specifications](../../library/early-exercise-calculation-specifications.md) | Continuation regression |

## Differentiation and risk

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Adjoint algorithmic differentiation | [Hull–White AAD calculation specifications](../hull-white-aad.md) | Path, curve and calibration sensitivities |
| VegaKT / quote-node risk | [Hull–White VegaKT calculation specifications](../hull-white-vegakt.md), [Local Volatility/VegaKT calculation specifications](../local-vol-vegakt-calculation-specifications.md) | Market-IV risk coordinates |

## Path treatment

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Smoothing and payoff graphs | [Path Dependence calculation specifications](../../library/path-dependence-calculation-specifications.md) | Differentiable digital and barrier payoffs |
| Brownian-bridge barrier treatment | [Path Dependence diagnostics](../../library/path-dependence-diagnostics.md) | Continuous barrier monitoring |
