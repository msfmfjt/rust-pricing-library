# Product and model compatibility

This page describes the documented pricing entry points for each product
family. A check mark means that the product is accepted by the corresponding
pricing path; it does not imply that every risk output or every dividend
convention is available.

The detailed product and model contracts remain authoritative for inputs,
limitations and risk semantics.

| Product family | BS / Black-76 / Local Volatility | Single-equity Bergomi LSV and rough-LSV | Single-equity Hull–White adapters | Multi-asset BS / Local Volatility | Multi-asset LSV / rough-LSV / Hull–White |
| --- | --- | --- | --- | --- | --- |
| European vanilla | ✓ | ✓ | ✓ | — | — |
| Digital | ✓ | ✓ | ✓ | — | — |
| Discrete Barrier | ✓ | ✓ | ✓ | — | — |
| Continuous Barrier | ✓ | — | — | — | — |
| Arithmetic Asian | ✓ | ✓ | ✓ | — | — |
| Fixed-strike Lookback | ✓ | ✓ | ✓ | — | — |
| American / Bermudan vanilla | ✓ | — | — | — | — |
| Basket | — | — | — | ✓ | ✓ |
| Worst-of | — | — | — | ✓ | ✓ |
| Autocallable | — | — | — | ✓ | ✓ |

## Boundary notes

- The first column covers the standard single-asset `PricingRequest` path with
  `BlackScholes`, `Black76` or `LocalVolatility` models.
- Bergomi LSV and rough-LSV adapters support the existing payoff graph for
  price evaluation, but reject American/Bermudan exercise and continuous
  Barrier monitoring. One- and two-factor Bergomi LSV use
  `BergomiLsvPricingPlan`; deterministic-rate rough-LSV uses
  `RoughBergomiLsvPricingPlan` without Hull–White inputs. Discontinuous Digital and discrete Barrier risk requires
  the documented smoothing contract.
- Hull–White adapters use a common one-factor rate process. They support the
  documented BS, one-/two-factor Bergomi LSV and rough-LSV entry points, subject
  to their matching target, rate, dividend and correlation contracts. American/
  Bermudan exercise and continuous Barrier monitoring are outside these
  adapters.
- Multi-asset products are `Basket`, `WorstOf` and `Autocallable`. The
  multi-asset APIs support BS/Local Volatility assets, particle-calibrated
  Bergomi LSV assets, and the documented common-rate extensions. All assets
  share one currency; common Hull–White paths use one shared one-factor rate
  process.
- Bermudan vanilla is represented by `AmericanVanillaSpec` with multiple
  exercise dates; it is not a separate `ProductSpec` variant.

## Risk boundaries

Risk output is a second compatibility dimension:

| Model path | Main documented risk outputs and restrictions |
| --- | --- |
| Black–Scholes / Black-76 | Price, Delta, Gamma and Vega; market-IV VegaKT is not available for constant-volatility models |
| Local Volatility | Price, pathwise risks, local-variance risk and VegaKT when the reporting-IV basis is supplied |
| Bergomi LSV | Price and recalibrated local-variance risk through the explicit LSV risk API; product smoothing is required for discontinuous payoff risk |
| Hull–White LSV / rough-LSV | Delta, cross Gamma, BS Vega, target/quote risk, curve risk and dividend-curve risk according to the selected entry point and retained reverse trace |
| Multi-asset LSV / Hull–White | Per-asset Delta, cross Gamma and model-specific target, quote or curve risk; model-parameter and correlation Greeks remain outside the documented boundary |

For exact risk availability, follow the linked [product reference](README.md),
[model reference](../models/README.md) and [calculation methods](../models/methods/README.md).
