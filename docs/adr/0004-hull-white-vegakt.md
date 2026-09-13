# Quote-node VegaKT for equity/Hull–White LSV

Date: 2026-09-13. Status: experimental implementation decision.

The user requested VegaKT after the hybrid AAD extension. Returning local
variance and density adjoints is insufficient to identify market-IV risk:
both must reach the same explicit IV source.

Add a quote-node surface with natural-cubic total variance in log moneyness
and linear total variance in time. Keep its exact transpose and original
quotes on targets constructed with `from_market_iv`. Append quote derivatives
and a parallel derivative to the existing AAD result after the particle VJP.
The stable JSON schema and default pricing/target constructors stay unchanged.

This defines independent IV-node bumps under a documented interpolation. It
does not infer quote sensitivities from arbitrary target arrays or differentiate
an unknown SSVI/eSSVI fitting procedure. Cash-model quotes use the existing
escrow F coordinate; physical quote conversion is still the caller's boundary.

Natural-cubic interpolation is not globally arbitrage constrained. Reject
invalid quote/target samples, strike extrapolation and any floor/cap repair.
Document time-knot derivatives, constant-IV time tails and time-zero copying.
Retain the interpolation source in plan fingerprints, and compute parallel
RQMC uncertainty from scramble sums so that bucket covariance is included.

The first-order AAD's branch conventions and conditional uncertainty remain.
See the [VegaKT contracts](../hull-white-vegakt-v0.1.md) for units, limitations,
API and full-recalibration validation. This extends the VegaKT boundary of
[ADR 0003](0003-hull-white-aad.md); broader H5 acceptance remains open.
