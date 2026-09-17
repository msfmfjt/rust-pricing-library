# Composition guides

Composition guides explain how independent model components are connected.
They are intentionally secondary to the [model component index](../components/README.md)
and [calculation-method index](../methods/README.md).

| Composition | Guide | Example |
| --- | --- | --- |
| Two-factor Bergomi LSV | [Factor and calibration contracts](../bergomi-two-factor-lsv.md) | [Two-factor Bergomi](../../../examples/python/multi_asset_bergomi_two_factor.py) |
| Bergomi with Hull–White | [Common-rate contracts](../bergomi-hull-white.md) | [Bergomi/Hull–White](../../../examples/python/bergomi_hull_white.py) |
| Multi-asset Bergomi LSV | [Marginal calibration and joint drivers](../multi-asset-lsv.md) | [Multi-asset LSV](../../../examples/python/multi_asset_lsv.py) |
| Multi-asset rough Bergomi LSV with Hull–White | [Rough joint-process contracts](../multi-asset-rough-bergomi.md) | [Multi-asset rough-LSV](../../../examples/python/multi_asset_rough_bergomi.py) |
| Hull–White with paid cash dividends | [Affine-dividend contracts](../hull-white-affine-dividends.md) | [Hull–White LSV](../../../examples/python/hull_white_lsv.py) |

These pages document supported composition boundaries, shared drivers,
calibration dependencies and combined risk outputs; they do not redefine the
underlying component contracts.
