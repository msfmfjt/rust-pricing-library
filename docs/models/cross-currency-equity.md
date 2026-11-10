# Foreign equity options with quanto and FX-linked settlement

`CrossCurrencyEquityOption` prices a European call or put in its settlement
currency. Supply an existing European product in that currency, the equity
market in the stock's currency, and the settlement-currency discount curve.
Rust exposes the model in `pricing::cross_currency`. Python exposes the same
class in `rust_pricing`. [Runnable USD/JPY example](../../examples/python/cross_currency_equity.py)

Let $S_T$ be the stock fixing in its own currency and $X_T$ the FX fixing in
settlement currency per unit of stock currency. A USD stock paid in JPY uses
JPY per USD. Both fix at expiry $T$. The payment date $P$ defaults to expiry
and may be later. The call payoffs below are per share. For a put, reverse
the subtraction inside the positive part.

| Python settlement | Call payoff | Strike currency |
| --- | --- | --- |
| `quanto` | $X_{fixed}(S_T-K_f)^+$ | Stock currency |
| `floating_fx` | $X_T(S_T-K_f)^+$ | Stock currency |
| `composite` | $(X_TS_T-K_d)^+$ | Settlement currency |

## Inputs and pricing

Use deterministic ACT/365F curves for settlement discounting $D_d$, foreign
discounting $D_f$ and continuous dividend carry $D_q$, under one consistent
collateral convention. The input $X_0$ is cash FX for value on the valuation
date. A market quote with a spot settlement lag must first be rolled to that
date. Equity and FX follow joint GBM with constant annualized volatilities
$\sigma_S,\sigma_X$ and correlation $\rho$ in the stated FX direction.

Write $F_S=S_0D_q(T)/D_f(T)$ and $F_X=X_0D_f(T)/D_d(T)$. The closed-form
calculation takes a Black expectation with the following forward, volatility
and payout multiplier, then applies notional and $D_d(P)$.

| Settlement | Black forward | Volatility | Multiplier |
| --- | --- | --- | --- |
| Quanto | $F_S\exp(-\rho\sigma_S\sigma_X T)$ | $\sigma_S$ | $X_{fixed}$ |
| Floating FX | $F_S$ | $\sigma_S$ | $F_X$ |
| Composite | $F_S F_X$ | $\sqrt{\sigma_S^2+\sigma_X^2+2\rho\sigma_S\sigma_X}$ | 1 |

With expiry-day payment, the floating-FX price equals the foreign vanilla PV
times cash FX. Positive stock/FX correlation lowers the quanto call price
and raises the composite call price. [Derivation, HKUST lecture notes, pp. 66–70](https://www.math.hkust.edu.hk/~maykwok/courses/ma571/06_07/Topic4.pdf)

`analytical_price()` returns the settlement-currency PV. `price_mc(engine)`
evaluates the contractual payoff on two correlated terminal draws under the
settlement-currency measure using the existing pseudo-MC or randomized Sobol
engine. The returned estimate includes its standard error. On the fixing date
the current spots supply the fixings and the result is exact. Valuation after
expiry requires historical fixings and is outside this entry point.

This API covers price with continuous dividend carry. A supplied discrete
dividend schedule raises a validation error. The regression suite checks
the published quanto benchmark, put–call parity with delayed payment, limiting
cases and both sampling engines. [Tests](../../tests/python/test_cross_currency_equity.py) · [QuantLib quanto benchmark](https://github.com/lballabio/QuantLib/blob/master/test-suite/quantooption.cpp)
