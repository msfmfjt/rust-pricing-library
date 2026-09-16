# Hull–White paid-cash affine dividends

Date: 2026-09-14. Experimental.
This is the current default dividend contract. The older
[escrowed contract](hull-white-cash-dividends.md) still applies only when
explicitly selected; its statements that default cash is rejected are superseded.

## Model and API

Use the existing `compile_bs`, `compile_lsv`, `compile_rough_bergomi` or
`compile_rough_lsv`. In Python omit `cash_dividend_model` or pass None. Explicit
`"escrowed"` retains the former model. In Rust the corresponding
`*_with_cash_dividends` methods retain that explicit escrowed behavior.

```python
plan = rp.HullWhiteEquityPlan.compile_bs(
    request, rates,
    equity_rate_correlation=0.25,
    maximum_step=0.05,
    worker_threads=4,
    cash_dividend_model=None,
)
price = plan.evaluate()
risk = plan.evaluate_aad()
```

An active fixed-cash plan reports `affine-paid-cash-realized-carry-v1` and
`risky_spot=S0`. No initial future-dividend reserve is subtracted. Cash after the
pricing horizon has no price, grid, calibration or risk effect in the default
model. Metadata is None if no fixed cash is paid within that horizon.

Let `G0(t)=Q0(t)/P0(t)` and let U be the existing normalized HW equity state:

```text
S_i = b_i * G0(t_i) * U_i + c_i
c_before_i = c_(i-1) * exp(
    logG0(t_i)-logG0(t_(i-1))
    + integrated_HW_shift(t_(i-1),t_i) + I_i-I_(i-1)
)
b_i = (1-beta_i)*b_(i-1)
c_i = (1-beta_i)*c_before_i-D_i
U_after_i = U_before_i
```

For a non-event node, beta=D=0. The time-zero U is pre-event S0 and c=0.
A time-zero payout is applied before observations. Expiry-date dividends are
also applied before terminal evaluation. Barrier observations retain both sides
of an event; touching remains a hit. Active ex-dates must be on the path grid.
No extra independent random driver is introduced for the jump.

In deterministic-rate BS/LV, physical Spot is `S=A*S0+B*f`. Propagate A at r-q
between events; B stays piecewise constant because f already carries. For one
fixed cash D paid at t_i, the physical forward is

```text
S0*G0(T) - D*G0(T)/G0(t_i),   t_i <= T.
```

Each later proportional payout scales earlier cash offsets as well as equity.
`EquityForward.forward()` remains the canonical f forward, while
`evaluate().spot_contract_forward` is the physical forward. Standalone
`AffineDividendTransform::new` without market curves remains the zero-carry
algebraic event transform; transforms attached to `EquityForward` include carry.

## Volatility, calibration and risk coordinates

BS volatility applies to the continuous U state (equivalently the deterministic
continuous f state), not the physical S or the legacy residual equity. With
stochastic volatility, leverage is fitted in U. Do not feed physical-spot IVs or
old escrow-coordinate IVs to the new target without converting/refitting them.
The library does not currently implement that market-quote conversion.

The default HW LSV and rough-LSV pass no reserve to the discounted-particle
calibration. Rate corrections still exist, but bond-reserve loading, its cross
moment and the escrowed quadratic are absent. Existing paired local variance
and T-forward density requirements still apply. Quote-backed VegaKT transposes
both target arrays into the supplied continuous-coordinate IV surface.

`evaluate_aad()` reverses every post/pre Spot observation, then the paid-cash
offset recurrence, its deterministic carry ratios and original curve pillars.
The centered HW state paths are fixed under initial-curve refits. Finally reverse
the continuous equity path and, once per pricing mean/RQMC scramble, the finite
particle calibration. Initial curves also affect the payment discount prefactor.
Spot bumps keep each D and beta fixed; no derivative of a frozen D/S0 is used.

First-order risk remains conditional on fixed grids, model parameters, RNG and
calibration branch choices. LSV AAD needs `retain_reverse_trace=True`.
RQMC errors use independent scrambles and retain cross-bucket covariance for
parallel Vega. Price-only and AAD use the same physical observations.

## Validation and boundaries

Focused regression tests cover cash/proportional collisions at zero and expiry,
carried forwards, the HW sigma_r=0 limit, pathwise discounted gains, all initial
curve pillars versus common-random-number differences, payment lag, all quoted
VegaKT buckets for LSV and rough-LSV, direct rough Delta/Vega, and worker replay.
Future-only cash must not create a reserve through the AAD compatibility context.
The existing CI suite continues to test the explicit escrowed model separately.

Any nonfinite or nonpositive physical observation is an error. Fixed cash plus
an unbounded equity diffusion cannot guarantee positive Spot; the implementation
does not floor paths, silently discard them, or substitute escrowed pricing.
This is not a physical-spot smile calibration, nor a new global positivity model.
American exercise, continuous Barrier monitoring and smoothing-width ladders
remain unsupported in the experimental HW adapters. No Gamma, dividend-amount
sensitivity, HW calibration/parameter risk or quote-conversion chain rule is added.
