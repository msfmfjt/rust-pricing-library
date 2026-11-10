# Foreign equity options with quanto and FX-linked settlement

`CrossCurrencyEquityOption` prices a single-stock option or payoff graph in
its settlement currency. Supply an existing product in that currency, the equity
market in the stock's currency, and the settlement-currency discount curve.
Rust exposes the model in `pricing::cross_currency`. Python exposes the same
class in `rust_pricing`. [Runnable USD/JPY example](../../examples/python/cross_currency_equity.py)

Let $S_T$ be the stock fixing in its own currency and $X_T$ the FX fixing in
settlement currency per unit of stock currency. A USD stock paid in JPY uses
JPY per USD. For European vanilla, both fix at expiry $T$. The payment date
$P$ defaults to expiry and may be later. The call payoffs below are per share. For a put, reverse
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

`analytical_price()` returns the European vanilla PV. Its `price_mc(engine)`
evaluates the contractual payoff on two correlated terminal draws under the
settlement-currency measure using the existing pseudo-MC or randomized Sobol
engine. The returned estimate includes its standard error. On the fixing date
the current spots supply the fixings and the result is exact. Valuation after
expiry requires historical fixings and is outside this entry point.

This API covers price with continuous dividend carry. A supplied discrete
dividend schedule raises a validation error. The regression suite checks
the published quanto benchmark, put–call parity with delayed payment, limiting
cases and both sampling engines. [Tests](../../tests/python/test_cross_currency_equity.py) · [QuantLib quanto benchmark](https://github.com/lballabio/QuantLib/blob/master/test-suite/quantooption.cpp)

## Exotic payoffs

Pass an arithmetic Asian, fixed-strike lookback, cash or asset digital,
discrete or continuous barrier, scheduled-exercise vanilla, or single-equity
graph to the same Python constructor and call `price_mc(engine)`. Rust accepts
these products through `CrossCurrencyEquityOption::from_product`. Each exotic
carries its payment dates in its product definition.

| Settlement | Path and cash-flow convention |
| --- | --- |
| `quanto` | Observe the stock in foreign units. Multiply every cash flow, including any rebate, by the fixed FX rate. |
| `floating_fx` | Observe the stock in foreign units. Asian, lookback, digital and barrier payoffs fix FX at expiry. A payoff graph fixes FX on each output's payment date. Early exercise fixes and pays at exercise. |
| `composite` | Every observation is the contemporaneous product of FX and stock price. Supply strikes, barrier levels, cash rebates and historical stock fixings in settlement-currency units. |

For example, a composite arithmetic Asian averages $X_i S_i$ on its observation
dates. A floating-FX Asian averages $S_i$, applies the foreign strike and
converts the result at expiry. An FX fixing before the final observation, or a
barrier on FX alone, requires an explicit equity/FX payoff model outside this API.

With deterministic rates and constant volatilities, each convention reduces to
a one-factor equity path. The quanto uses the domestic-measure foreign-stock
drift. Its forward on **each** observation date includes the covariance adjustment.
The composite follows the domestic equity law with the combined volatility
above. A floating-FX cash flow fixed at $T$ and paid at $P$ uses the foreign
equity law and multiplies the native foreign PV by $F_X(T)D_d(P)/D_f(P)$.
When FX fixes on payment, the multiplier is $X_0$, including for each graph
output and at an exercise stopping time. The reported standard error and
confidence interval use the same currency conversion. [Kwok and Wong, path-dependent foreign equity options](https://www.math.hkust.edu.hk/~maykwok/piblications/wongcurrency.pdf)

Continuous barriers use a conditional Brownian bridge split at changes of
log carry. Historical barrier hits refer to the monitoring asset specified
by the settlement convention. For early exercise, use `price_mc(engine, lsm=config)`
with an `LsmConfig` specifying independent training paths for the exercise
policy. Single-stock autocalls use `Product.from_graph(graph)`, with
the graph obtained from `contract.to_graphs(valuation_date, [stock_id])`.
Arithmetic and geometric averages, knock states and early-redemption conditions
in these graphs use the original contractual operations. [Exotic pricing tests](../../tests/python/test_cross_currency_exotics.py)
