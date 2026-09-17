# Product reference

This section lists the financial products supported by the library. Product
contracts are independent of the model components used to value them; see the
[model reference](../models/README.md) for models, calculation methods and
supported combinations.

[Documentation](../README.md) · [Library guide](../library/README.md) ·
[Model reference](../models/README.md)

The [product and model compatibility matrix](compatibility.md) summarizes
which pricing entry points accept each product family.

## Supported products

| Product | Contract reference | Example |
| --- | --- | --- |
| European vanilla | [European diagnostics](../models/european-bs-diagnostics.md) | [European Black–Scholes](../../examples/python/european_bs.py) |
| Digital | [Path-dependence calculation specifications](../library/path-dependence-calculation-specifications.md) | [Path dependence](../../examples/python/path_dependence.py) |
| Barrier | [Path-dependence calculation specifications](../library/path-dependence-calculation-specifications.md) | [Path dependence](../../examples/python/path_dependence.py) |
| Arithmetic Asian | [Path-dependence calculation specifications](../library/path-dependence-calculation-specifications.md) | [Path dependence](../../examples/python/path_dependence.py) |
| Fixed-strike Lookback | [Path-dependence calculation specifications](../library/path-dependence-calculation-specifications.md) | [Path dependence](../../examples/python/path_dependence.py) |
| American vanilla | [Early-exercise calculation specifications](../library/early-exercise-calculation-specifications.md) | [American LSM](../../examples/python/american_lsm.py) |
| Bermudan vanilla | [Early-exercise calculation specifications](../library/early-exercise-calculation-specifications.md) | [American LSM](../../examples/python/american_lsm.py) |
| Basket | [Multi-asset contracts](../models/multi-asset.md) | [Multi-asset pricing](../../examples/python/multi_asset.py) |
| Worst-of | [Multi-asset contracts](../models/multi-asset.md) | [Multi-asset pricing](../../examples/python/multi_asset.py) |
| Autocallable | [Multi-asset contracts](../models/multi-asset.md) | [Multi-asset pricing](../../examples/python/multi_asset.py) |

## Product features

- European vanilla, Digital and Barrier products support call/put or payout
  conventions defined by their individual contracts.
- Barrier products support up/down and in/out styles, with discrete or
  continuous monitoring. Continuous monitoring uses the documented bridge
  treatment.
- Arithmetic Asian products support explicitly scheduled observations,
  including partially fixed observations where allowed by the contract.
- Fixed-strike Lookbacks use discrete observations and a historical extremum
  when one is supplied.
- American and Bermudan vanilla products use scheduled exercise and the
  documented least-squares Monte Carlo policy. Bermudan vanilla is represented
  by the same `AmericanVanillaSpec` with multiple exercise dates.
- Basket, Worst-of and Autocallable products are multi-asset graph products.
  Autocallables support dated observations, payment lags and memory or
  non-memory termination policies.

The available risk outputs and model combinations depend on the selected plan.
Product contracts define the payoff and event conventions; model-specific
limitations remain in the [model reference](../models/README.md).
