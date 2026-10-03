# Heston Black-IV calibration and SSVI targets

Experimental positive-forward European calibration for the Rough Heston and
fixed-kernel Lifted Heston Fourier models. This is an additive adapter over
[price calibration](heston-calibration.md); no existing pricing or risk formula
is replaced. It is not a new SSVI model or a calibration of SSVI parameters.

## Objective and units

`HestonIvCalibrationQuote` holds positive maturity (years), Forward, Strike,
discount, target Black implied volatility, option side, and `iv_scale`.
Volatilities are absolute annualized units: 20% is `0.20`; one IV basis point is
`0.0001`. Quote scales are fixed positive divisors, not variance or percent units.

For quote i the residual and analytic Jacobian are

```text
r_i(p) = (BlackIV(P_i(p); F_i,K_i,D_i,T_i) - target_IV_i) / iv_scale_i
J_ij   = (dP_i/dp_j) / BlackVega(F_i,K_i,D_i,T_i,model_IV_i) / iv_scale_i
objective = 0.5 * sum_i r_i(p)^2
```

Model Vega is re-evaluated at every trial. This is **not** the approximation
`(P-P_target)/Vega_target`. The existing cached Riccati scalar/Hurst tangents
provide dP/dp without bumps. At fixed parity inputs call and put have equal
parameter derivatives and IVs, subject to floating-point precision. The solver's
variable-coordinate scale is distinct from quote `iv_scale`; returned Jacobians
are in natural parameter coordinates and do not include that variable scale.

`fit_achieved` means every final scaled IV residual meets `residual_tolerance`.
With `iv_scale=1`, `residual_tolerance=0.0005` is a five-IV-bp criterion.
Stationarity at a box boundary, exhausted evaluation budgets and small steps
are not automatically fits. Invalid IV trial points are counted and rejected;
invalid initial points are errors. No quote is silently dropped.

## Numerical IV domain

Black inversion normalizes by `D*max(F,K)` and inverts the OTM time value using
bounded bisection of total standard deviation, with at most 96 iterations.
The experimental domain requires total standard deviation in `(0,8]`, normalized
OTM time value strictly above `32*f64::EPSILON`, and normalized standard-deviation
Vega at least `1e-10`. Prices at/beyond intrinsic or the upper bound, unrepresentable
normalizations, nonfinite results and ill-conditioned Vega are errors. No price
clamping, artificial volatility return or Vega floor is applied. These guards
exclude some otherwise mathematically valid deep-wing/short-expiry cases.

The bracket stopping target is `1e-12` annualized IV; finite precision and the
normal-CDF approximation still apply. An independent price-back-substitution
condition is also checked. These settings control inversion arithmetic, not the
Fourier model's total numerical error. Target IVs must round-trip through Black
price within `1e-8*max(1,IV)` on construction.

`model_vegas` are price per one absolute annualized IV unit.
`price_quadrature_differences` and `price_tail_indicators` remain **price** diagnostics.
They are not IV error estimates, omitted-tail bounds or sampling errors; Gamma,
Riccati, model, calibration and MC biases are not bounded by them.

## SSVI input

Rust `HestonIvCalibrationQuote::from_ssvi` samples the existing validated
`StandardSsvi` at `(T, log(K/F))` and sets `target_IV=sqrt(w/T)`. It chooses the
OTM side, call at ATM. It does not infer Forward or discount from Spot or dividends.
The caller must supply coordinates in the same measure and dividend convention as
the positive-forward Heston model. No price comparison with cash-dividend Spot
options is implied by using an SSVI volatility with a different convention.

Python `HestonCalibrationSsviSurface.power_law(...)` and `.heston_like(...)` wrap
that same immutable market surface and `ThetaPchip`. They retain the existing
surface admissibility checks and short/long extrapolation; no independent copy of
the SSVI implementation is introduced. `.quote(T,F,K,D,iv_scale=1)` returns an IV
quote. Standard power-law here is
`phi(theta)=eta/(theta**gamma*(1+theta)**(1-gamma))`, not `eta/sqrt(theta)` alone.
SSVI rho is a surface-shape parameter, not Heston's price/variance correlation.
An arbitrary SSVI surface need not be exactly attainable by either Heston family.

## API

The new Rust/Python types are `HestonIvCalibrationQuote`,
`HestonIvCalibrationProblem`, `HestonIvCalibrationEvaluation`, and
`HestonIvCalibrationResult`. Reuse `HestonCalibrationVariable` and solver options.
Rough Heston permits six selected directions; finite Lifted Heston permits the
five scalar directions with fixed weights/rates. Positive initial variance is
required, as for parent scalar-risk calibration. At most 4,096 quotes and 64
maturities are accepted; inherited computation-budget checks remain active.

```python
surface = rp.HestonCalibrationSsviSurface.power_law(
    [0.25, 0.75, 1.5], [0.01, 0.03, 0.06], 0.04,
    rho=-0.5, eta=0.35, gamma=0.5,
)
quotes = [surface.quote(t, 100.0, strike, 0.97)
          for t in [0.25, 0.75, 1.5] for strike in [85.0, 100.0, 115.0]]
problem = rp.HestonIvCalibrationProblem.compile(initial_model, quotes, variables)
result = problem.calibrate(residual_tolerance=5e-4)
print(result.fit_achieved, result.termination)
print(result.evaluation.model_implied_volatilities)
```

Complete runnable [example](../../examples/python/heston_iv_calibration.py).
Rust `quote.to_price_quote()` is provided for deliberate conversion to the parent
price objective, with `price_scale=1`; it does not create IV residuals.

## References and limitations

Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*,
[arXiv:1204.0646](https://arxiv.org/abs/1204.0646), specifies the SSVI total variance
surface. The Black inverse derivative above follows by differentiating the Black
pricing identity; see also the independently implemented Black and Vega formulas
in [QuantLib](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/blackformula.cpp).
No QuantLib implementation is copied or introduced as a dependency.

The [validation protocol](../../design/validation/heston-iv-calibration.md)
distinguishes independent input/Markov references, synthetic rough recovery,
finite-grid repricing and SSVI model mismatch. No global/unique optimum, parameter
identifiability, static-arbitrage validation of arbitrary IV quote grids, automatic
Fourier error control or real-market calibration is claimed. Recalibrated market
risk, AAD/VegaKT, lift-kernel fitting and new LSV/rate/dividend compositions remain
outside this API. A Fourier IV fit does not remove the existing MC scheme's bias.
