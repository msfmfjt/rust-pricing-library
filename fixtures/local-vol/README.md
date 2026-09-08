# Local Volatility and VegaKT equation fixtures

`reference-cases-v0.1.json` is the Gate L0 equation-level reference set for
policy `local_vol_vegakt_v1`. It is intentionally independent of any production
Rust implementation.

The SSVI and eSSVI values were evaluated with 70-digit Python `Decimal`
arithmetic. Transition-cell values use SciPy's binary64 normal CDF only when the
fixture was authored; the committed checker uses the Python standard-library
`erfc` formulation and a documented binary64 tolerance. Algebraic cases use
decimal arithmetic or exact rational identities.

Sources and equation provenance are recorded in every case group. The VegaKT
paper's published figures are not used as numerical truth because their full
market surface, grids, and raw values are unavailable. Updates require review of
the inputs, policy version, provenance, and all expected values.

Run:

```shell
python scripts/check_local_vol_reference_fixture.py
```
