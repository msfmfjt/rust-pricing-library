# Pure stochastic volatility

One-factor Bergomi, two-factor Bergomi and rough Bergomi support pure SV
without a Local Volatility target or particle calibration. All three use
the same payoff, escrowed-dividend, MC/RQMC and first-order AAD integration
as their Hull–White counterparts. This is an experimental single-asset API.

## Dynamics and initial variance

For the Markovian models, let

$$
dX_i=-k_iX_i\,dt+dW_i,\quad X_i(0)=0,\qquad
Y_t=\sum_i w_iX_i(t),\qquad
v_t=\sigma_0^2\exp\bigl(2\nu Y_t-2\nu^2 V(t)\bigr),
$$

where $V(t)=\operatorname{Var}_Q[Y_t]$. One factor uses $w_1=1$; two factors
use the [existing normalized mixing weights](bergomi-two-factor-lsv.md).
With $B(k,t)=(1-e^{-kt})/k$, extended continuously by $B(0,t)=t$,

$$
V(t)=\sum_{i,j}w_iw_j\rho_{ij}B(k_i+k_j,t).
$$

The implementation computes this variance as a sum of squared joint Gaussian
loadings, avoiding cancellation near singular negative factor correlations.
It centers the simulated factor by $Y_t-\nu V(t)$. Consequently
$\mathbb E_Q[v_t]=\sigma_0^2$ at every node. Merely fixing an LSV leverage
surface to a constant would omit this normalization: the historical LSV
multiplier is uncentered because its deterministic scaling is absorbed by
particle calibration.

`vol_of_vol` is **nu, the coefficient of log volatility**, in both Markovian
models. [Rough Bergomi](rough-bergomi.md) retains **eta, the coefficient of
log variance**, and its finite-grid Volterra centering. At H=1/2, eta=2*nu,
rough agrees with zero-mean-reversion one-factor pure Bergomi on the same
equity time grid and shared random draws, up to rounding.

All pure APIs currently use flat initial forward variance
`xi0(t)=sigma0^2`, supplied by a `Model.black_scholes(sigma0)` request. This is
an initial **Q-measure variance expectation**, not a Black IV curve or a
T-forward variance expectation under stochastic rates. Factor parameters are
specified directly; there is no IV/SSR parameter fitting or nonflat xi0 input.

The variance applies to residual risky equity after funding all fixed cash
dividends, including supplied payments beyond option expiry. Physical-spot
observations are reconstructed using the existing
[escrowed reserve](hull-white-cash-dividends.md). Proportional/mixed payouts,
event order, deterministic carry and payment lags keep the shared contracts.

## Rust and Python APIs

| Rates | Rust plan | Python plan |
| --- | --- | --- |
| Deterministic market curves | `pricing::stochastic_volatility::StochasticVolatilityPricingPlan` | `StochasticVolatilityPlan` |
| Hull–White | `pricing::hull_white::HullWhiteEquityPricingPlan` | `HullWhiteEquityPlan` |

Both plans offer `compile_bergomi`, `compile_bergomi_two_factor` and
`compile_rough_bergomi`. Pure SV compilers require a price-only Black–Scholes
base request; the sigma is an input to SV, not an instruction to use BS dynamics.
Use `evaluate()` for price and `evaluate_aad()` for the separate risk interface.

```python
plan = rp.StochasticVolatilityPlan.compile_bergomi_two_factor(
    request,  # Model.black_scholes(0.2), RiskRequest()
    mean_reversions=[0.7, 2.1], vol_of_vol=0.6, mixing_weight=0.35,
    spot_correlations=[-0.5, -0.3], factor_correlation=0.25,
    maximum_step=1/64, worker_threads=2,
)
price = plan.evaluate()
risk = plan.evaluate_aad()
```

The deterministic facade uses an exact zero-volatility HW rate component;
callers supply no rate model or rate correlations. It deliberately retains
the hybrid random layout: four Gaussian blocks per step for 1F and five
for 2F/rough, including zero-loading rate directions. This gives identical
fingerprints and results to the explicitly equivalent zero-rate-volatility
HW plan. It does not claim the efficiency of a reduced-dimension SV engine.

HW 1F accepts a `HybridCorrelation` in Rust; Python takes equity/vol,
equity/rate and vol/rate correlations. HW 2F accepts the factor model's two
spot correlations, factor correlation, an equity/rate correlation and two
vol/rate correlations. The complete Brownian matrix must be PSD, even at
zero rate volatility. Factor order is preserved.

See the [runnable example](../../examples/python/pure_bergomi.py).
The existing `BergomiLsvPlan`, `Bergomi2FactorLsvPlan`, `RoughBergomiLsvPlan`
and hybrid `compile_lsv*` methods continue to request calibrated LSV explicitly.

## Risk, diagnostics and supported products

AAD returns Spot Delta, `initial_volatility` sensitivity dPrice/dsigma0,
and the initial discount/carry curve adjoints. The right sigma derivative
remains finite at sigma0=0. Model parameters, correlations, dividend quotes,
dates, grids and random draws are fixed. No retained calibration trace is needed.
Pure SV has no quote-node VegaKT, parameter Greeks or Gamma.

`calibration_method` and `calibration_seed` are `None`; uncertainty is
`pricing_only`, conditional on model inputs and the finite time grid. It
excludes discretization error. One-/two-factor pure schemes are
`bergomi-one-factor-pure-hw-normalized-log-euler-v1` and
`bergomi-two-factor-pure-hw-normalized-log-euler-v1`; rough retains its scheme.
Every factor parameter and correlation contributes to the fingerprint.

European, Asian, Lookback, Digital and discretely monitored Barrier products
reuse the hybrid payoff integration. Digital/Barrier AAD requires explicit
smoothing. American/Bermudan exercise, continuous Barrier monitoring and
smoothing-width ladders are rejected. This addition does not register pure
SV constituents in `MultiAssetPlan`. The stable JSON schema continues to carry
the base request; SV selection and its parameters live in the explicit plan API.

## References

- Bergomi, *Smile Dynamics II*, [SSRN 1493302](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), for lognormal forward-variance dynamics.
- The [two-factor specification](bergomi-two-factor-lsv.md) defines the library's weights, correlations and exact joint OU innovations.
- The [rough specification](rough-bergomi.md) gives the Volterra scheme and its primary references.
- [ADR 0012](../../design/adr/0012-pure-stochastic-volatility.md) records compatibility and validation impact.
