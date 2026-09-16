# Calculation methods

These pages organize the numerical methods independently of the model
component that consumes them. The detailed contracts remain authoritative for
the exact implementation policy and tolerances.

## Monte Carlo and quasi-Monte Carlo

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Monte Carlo, antithetic sampling and uncertainty | [Benchmark baseline](../../library/benchmarking.md) | Pricing and validation |
| Randomized Sobol' QMC | [European diagnostics](../european-bs-diagnostics.md), [early-exercise contracts](../../library/early-exercise-numerical-contracts.md) | Pricing and LSM training/valuation |
| Philox counter-based streams | [Benchmark baseline](../../library/benchmarking.md) | Reproducible parallel simulation |
| Volterra hybrid scheme | [Rough Bergomi contracts](../rough-bergomi.md) | Rough-volatility paths |

## Calibration and interpolation

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Particle calibration | [LSV numerical contracts](../lsv-numerical-contracts.md), [Local Correlation](../local-correlation.md) | Leverage and local-correlation targets |
| Surface interpolation and local variance | [Local Volatility numerical contracts](../local-vol-vegakt-numerical-contracts.md) | SSVI/eSSVI and Dupire |

## Early exercise and linear algebra

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Least-squares Monte Carlo | [Early Exercise numerical contracts](../../library/early-exercise-numerical-contracts.md) | American and Bermudan exercise |
| Column-pivoted Householder QR | [Early Exercise numerical contracts](../../library/early-exercise-numerical-contracts.md) | Continuation regression |

## Differentiation and risk

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Adjoint algorithmic differentiation | [Hull–White AAD contracts](../hull-white-aad.md) | Path, curve and calibration sensitivities |
| VegaKT / quote-node risk | [Hull–White VegaKT contracts](../hull-white-vegakt.md), [Local Volatility/VegaKT contracts](../local-vol-vegakt-numerical-contracts.md) | Market-IV risk coordinates |

## Path treatment

| Method | Detailed reference | Main use |
| --- | --- | --- |
| Smoothing and payoff graphs | [Path Dependence numerical contracts](../../library/path-dependence-numerical-contracts.md) | Differentiable digital and barrier payoffs |
| Brownian-bridge barrier treatment | [Path Dependence diagnostics](../../library/path-dependence-diagnostics.md) | Continuous barrier monitoring |
