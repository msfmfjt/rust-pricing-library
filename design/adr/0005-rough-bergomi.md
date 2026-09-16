# Rough Bergomi and rough-LSV in the equity/Hull–White engine

Date: 2026-09-13. Status: experimental implementation decision.

The user requested rough volatility after the LSV, stochastic-rate, dividend,
AAD and VegaKT extensions. Add a Riemann–Liouville rough Bergomi factor to the
existing hybrid engine so the same payoff, discount, dividend and particle
reverse paths can be used.

Use a nonuniform kappa=1 Volterra hybrid scheme: sample the newest singular
kernel integral jointly with the equity and HW increments, and use optimal
cell-average weights on older intervals. Include a fifth Gaussian block for
the near-cell residual. Center the lognormal variance factor using the actual
discrete variance. This preserves its unit mean on the chosen grid; it is an
explicit finite-grid choice rather than a claim of exact continuous paths.

Expose pure rough Bergomi with flat initial forward variance, and rough-LSV
with a separately calibrated leverage surface. Keep eta as the coefficient of
log variance. The older Bergomi coefficient nu multiplies log volatility.
The Brownian boundary is H=1/2, with eta=2*nu when comparing LSV limits.

HW may have zero volatility for deterministic rates. Escrowed cash dividends
and paired variance/density target construction retain their existing model
coordinates. At fixed H/eta, the rough driver is independent of the active
Spot, initial-curve and IV risk inputs. Reuse the discrete particle VJP and
quote transpose; do not advertise model-parameter adjoints.

The direct convolution costs O(time_steps^2) per path and supports nonuniform
event grids without a new dependency. FFT/lift acceleration, a nonflat initial
forward-variance curve, rough Heston, H/eta fitting and sensitivities, and broad
calibration/risk refinement are future work. Stable JSON model/risk selection
and the previous acceptance gates remain as documented.

See the [numerical and API contracts](../../docs/models/rough-bergomi-v0.1.md), including
the independent covariance/conditional-price references and full-recalibration
VegaKT tests. This extends [ADR 0004](0004-hull-white-vegakt.md).
