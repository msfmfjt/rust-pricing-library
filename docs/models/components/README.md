# Model components

Each entry below describes one reusable component. Combination-specific
behavior is linked separately from the [composition guides](../compositions/README.md).
Published sources are collected in the [bibliography](../../references.md).

## Equity dynamics

| Component | Detailed reference | Example |
| --- | --- | --- |
| Black–Scholes and Black-76 | [European diagnostics](../european-bs-diagnostics.md) | [European Black–Scholes](../../../examples/python/european_bs.py) |
| Local Volatility and Dupire | [Local Volatility calculation specifications](../local-vol-vegakt-calculation-specifications.md) | [Local Volatility](../../../examples/python/local_vol_vegakt.py) |
| Local-stochastic volatility | [LSV calculation specifications](../lsv-calculation-specifications.md) | [Bergomi LSV](../../../examples/python/bergomi_lsv.py) |
| Pure stochastic volatility | [1F/2F/rough Bergomi](../pure-stochastic-volatility.md) | [Pure Bergomi](../../../examples/python/pure_bergomi.py) |
| Bergomi volatility factors | [Two-factor factor contracts](../bergomi-two-factor-lsv.md) | [Two-factor Bergomi](../../../examples/python/multi_asset_bergomi_two_factor.py) |
| Rough Bergomi volatility | [Rough Bergomi contracts](../rough-bergomi.md) | [Rough Bergomi](../../../examples/python/rough_bergomi.py) |

## Rates and carry

| Component | Detailed reference | Example |
| --- | --- | --- |
| Hull–White short rate | [Hull–White calculation specifications](../hull-white-calculation-specifications.md) | [Hull–White LSV](../../../examples/python/hull_white_lsv.py) |
| Cash-dividend coordinates | [Escrowed reserves and calibration coordinates](../hull-white-cash-dividends.md) | [Hull–White LSV](../../../examples/python/hull_white_lsv.py) |

## Surfaces and calibration targets

| Component | Detailed reference | Related method |
| --- | --- | --- |
| SVI, SSVI and eSSVI | [Local Volatility calculation specifications](../local-vol-vegakt-calculation-specifications.md) | [Surface bibliography](../../references.md#core-pricing-models-and-volatility-surfaces) |
| Local variance and Dupire density | [Local Volatility calculation specifications](../local-vol-vegakt-calculation-specifications.md) | [Local Volatility diagnostics](../local-vol-vegakt-diagnostics.md) |
| LSV leverage target | [LSV calculation specifications](../lsv-calculation-specifications.md) | [Calibration methods](../methods/README.md#calibration-and-interpolation) |

## Multi-asset and correlation

| Component | Detailed reference | Example |
| --- | --- | --- |
| Multi-asset state and products | [Multi-asset pricing](../multi-asset.md) | [Multi-asset pricing](../../../examples/python/multi_asset.py) |
| Particle Local Correlation | [Local Correlation](../local-correlation.md) | [BS/LV Local Correlation](../../../examples/python/local_correlation.py) |
