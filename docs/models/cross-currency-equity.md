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

Supply initial ACT/365F curves for settlement discounting $D_d$, foreign
discounting $D_f$ and continuous dividend carry $D_q$, under one consistent
collateral convention. The input $X_0$ is cash FX for value on the valuation
date. A market quote with a spot settlement lag must first be rolled to that
date. With the scalar `equity_volatility` input, equity and FX follow joint GBM under deterministic rates with constant annualized volatilities
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

The analytical formula uses continuous dividend carry and deterministic rates.
For simulated cash dividends, select the Buehler model described below. The regression suite checks
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
| `floating_fx` | Observe the stock in foreign units. Asian, lookback, digital and barrier payoffs fix FX at expiry. A payoff graph defaults to FX on each payment date and accepts an earlier fixing for each payment. Early exercise fixes and pays at exercise. |
| `composite` | Every observation is the contemporaneous product of FX and stock price. Supply strikes, barrier levels, cash rebates and historical stock fixings in settlement-currency units. |

For example, a composite arithmetic Asian averages $X_i S_i$ on its observation
dates. A floating-FX Asian averages $S_i$, applies the foreign strike and
converts the result at expiry. An FX fixing before the last equity observation
needed for the same cash flow, or a barrier on FX alone, requires a joint equity/FX payoff model.

With deterministic rates and constant volatilities, each convention reduces to
a one-factor equity path. The quanto uses the domestic-measure foreign-stock
drift. Its forward on **each** observation date includes the covariance adjustment.
The composite follows the domestic equity law with the combined volatility
above. A floating-FX cash flow fixed at $T$ and paid at $P$ uses the foreign
equity law and multiplies the native foreign PV by $F_X(T)D_d(P)/D_f(P)$.
When FX fixes on payment, the multiplier is $X_0$, including at an exercise
stopping time. The reported standard error and
confidence interval use the same currency conversion. [Kwok and Wong, path-dependent foreign equity options](https://www.math.hkust.edu.hk/~maykwok/piblications/wongcurrency.pdf)

Continuous barriers use a conditional Brownian bridge split at changes of
log carry. Historical barrier hits refer to the monitoring asset specified
by the settlement convention. For early exercise, use `price_mc(engine, lsm=config)`
with an `LsmConfig` specifying independent training paths for the exercise
policy. Single-stock autocalls use `Product.from_graph(graph)`, with
the graph obtained from `contract.to_graphs(valuation_date, [stock_id])`.
Arithmetic and geometric averages, knock states and early-redemption conditions
in these graphs use the original contractual operations. [Exotic pricing tests](../../tests/python/test_cross_currency_exotics.py)

### Multi-payment autocalls

The quarterly USD/JPY example pays a 2% coupon when the observation meets
the coupon barrier. Missed coupons lapse. A call pays the principal and any coupon due that period.
If the note reaches maturity below 70% of its initial reference, redemption
is proportional to final performance. For `quanto` and `floating_fx`, the
contract states the principal and trigger levels in USD. For `composite`, it
states the principal in JPY and tests the triggers against the JPY value of
the stock on each observation date. [Runnable autocall example](../../examples/python/cross_currency_autocall.py)

For a single-stock memory autocall, build `MultiAssetProduct.autocallable` with
`memory=True` and pass `Product.from_graph(product.to_graph())` to the pricing
model. A recovered coupon uses the FX fixing for the date when it is actually
paid. The existing `on_autocall` and `on_maturity` settings govern outstanding
coupons at redemption.

A floating-FX coupon or redemption uses the FX rate on its payment date by
default. To use an earlier fixing, pass pairs of (payment date, FX fixing date)
to `with_payment_fx_fixings`. For example, a coupon and principal due July 14
can both convert at the July 7 observation.

```python
note = note.with_payment_fx_fixings([("2026-07-14", "2026-07-07")])
estimate = note.price_mc(engine)
```

Each pair applies to all cash flows on that payment date, including amounts
that depend on earlier coupon or call events. Unlisted payments use payment-day FX.
The fixing date must be on or after valuation and the last equity observation
that determines the payment, and on or before payment. Each call supplies a
replacement schedule. For a foreign amount $A$ determined by fixing date $F$
and paid on $P$, under deterministic rates the domestic PV is $X_0 D_f(F) D_d(P)/D_d(F)\,\mathbb{E}_f[A]$.
The simulation aggregates all payments on the same paths before estimating the
standard error. [Autocall validation cases](../../tests/python/test_cross_currency_autocall.py)

### Selecting an equity model

Pass a `Model` or `EquityModelSpec` through the `equity_model` keyword in
place of `equity_volatility`. For example, the following selects a stochastic
foreign-equity volatility for a quarterly note.

```python
factor = rp.VolatilityFactorSpec.bergomi(1.2, 0.3, -0.4)
equity = rp.EquityModelSpec.stochastic_volatility(0.20, factor)
note = rp.CrossCurrencyEquityOption(
    valuation, rp.Product.from_graph(graph), market, jpy_curve, settlement="composite",
    equity_model=equity, fx_spot=150., fx_volatility=.10, correlation=.35, maximum_step=.02,
)
```

The model path covers Black–Scholes, Black-76, local volatility, Bass-LV,
one- and two-factor Bergomi, rough Bergomi, rough and lifted Heston, quadratic
rough Heston, mixed rough Bergomi, rough SABR and RFSV. The Bergomi and six
rough-family models also accept their particle-calibrated LSV specifications.
Rough SABR LSV uses the native lognormal (`beta=1`) calibration. Vanilla,
Asian, lookback, digital, discretely monitored barriers and single-stock
cash-flow graphs use the selected equity model. The original constant-volatility
path also handles continuous barriers and scheduled early exercise.

Equity parameters and leverage targets refer to the foreign market. FX is
lognormal, with one input correlation to the equity Brownian increment. Its
remaining innovation is independent of the equity model's orthogonal factors.
For quanto and composite payoffs the native foreign-equity increments acquire
the domestic-measure drift shift. Composite uses the joint FX path at each
fixing. Floating-FX uses the foreign measure with the contractual fixing and
payment dates. MC standard errors measure sampling uncertainty. Time-step and
particle-calibration errors depend on the chosen numerical inputs.
[Model comparisons and autocall tests](../../tests/python/test_cross_currency_models.py)

### Domestic and foreign Hull–White rates

Call `option.with_hull_white_rates(domestic, foreign, correlations=matrix)`
before `price_mc`. Both models use their own input discount curve, mean
reversion and piecewise-constant rate volatilities. The indices of the
four-by-four Brownian correlation matrix are equity, FX quoted in domestic
units per foreign unit, domestic rate and foreign rate. The equity/FX entry
must match the option. Omitting the matrix makes both rate drivers independent
of each other, the stock and FX. A zero-volatility rate model fixes that
currency's rates. [USD/JPY autocall with two short rates](../../examples/python/cross_currency_autocall.py)

Simulation uses the domestic pricing measure. Each step jointly samples the
equity and FX Brownian increments, both OU endpoints and both rate integrals.
The foreign rate carries the drift adjustment from its covariance with FX.
Each coupon is discounted on its own payment date using the simulated domestic
money-market account. The stock, FX and discount factor all use the same
realization of the rate integrals, including between fixing and payment.
Local-volatility and leverage lookups include realized foreign rate carry.
A Bergomi LSV specification with a foreign forward-density target uses the
native Hull–White particle calibration. A specification with a local-variance
target alone uses its foreign deterministic-rate leverage surface.
[Bond, delayed FX and option price benchmarks](../../tests/python/test_cross_currency_rates.py)

The four-factor input specifies rate correlations with equity and FX. Other
equity-model innovations are independent of the rate innovations conditional
on the simulated stock increments. The equity model retains its own
equity/volatility correlations. The asset dynamics use their native time
discretization. The rate endpoint and integral covariances use exact Gaussian
transitions on each piecewise-constant rate-volatility interval.


### Stochastic foreign cash dividends

Supply a foreign market with dated fixed-cash means and pass a
`BuehlerDividendModel` through the option’s `dividend_model` keyword.
Cash amounts are in the stock currency and specify expectations under its
money-market measure. The model funds every supplied future cash dividend,
including those after the final equity observation. A graph can observe both
sides of a dividend event using `builder.spot(..., pre_dividend=True)` and
`builder.spot(...)`. Composite settlement converts both at that observation’s FX.

With deterministic foreign rates, use Black, one-factor or two-factor Bergomi,
rough Bergomi, or any of their three residual-equity LSV calibrations. The
domestic rate can be deterministic or Hull–White. With stochastic foreign
rates the Buehler/Hull–White combination uses constant residual-equity
volatility and the native discounted conditional dividend reserve. Dividend
correlations with FX, rates and volatility factors are the product of their
respective correlations with the residual-equity driver. Pricing requires
strictly future fixed-cash events. [Dividend-claim and native-price comparisons](../../tests/python/test_cross_currency_dividends.py)

### Several stocks in one foreign currency

Use `CrossCurrencyEquityBasket` for a basket, worst-of, memory autocall or other
dated payoff graph on stocks that share one foreign discount curve. Supply the
markets and equity models in the graph’s declared underlying order. Each stock
has its own initial price, dividend carry, volatility model and last observation
date. The graph sets each payment amount and governs coupon recovery, early
redemption and downside participation. [Two-stock USD/JPY quarterly note](../../examples/python/cross_currency_worst_of_autocall.py)

The correlation matrix orders the stocks first, then FX. When passing
`rates=(domestic_hw, foreign_hw)`, append the domestic and foreign short-rate
drivers in that order. For two stocks this gives a 5 × 5 matrix. Correlations
are constant and may be singular. The simulation uses one realization of FX
and of both rates across all stocks and payments. Volatility and dividend
innovations left over after conditioning each equity model on its stock driver
are independent between stocks. A `dividend_models` list can select the
Buehler model for individual names under the same combinations as single-stock
pricing.

For quanto and floating FX, supply foreign-currency stock references and cash
amounts. For composite, supply domestic-currency references and cash amounts.
Each composite observation uses the same contemporaneous FX rate for all
stocks. In floating FX, payment-day fixing is the default. Use
`with_payment_fx_fixings([(payment, fixing), ...])` to set observation-day FX.
All cash flows sharing a payment date share the specified fixing. The result
reports the sampling error of the complete note in the settlement currency.
[Joint moments, foreign-currency equivalence and model tests](../../tests/python/test_cross_currency_basket.py)


### Stocks in different foreign currencies

`CrossCurrencyEquityBasket.from_currencies` prices a graph of US dollar and
euro shares in JPY. Supply one market per stock in graph order. Each market’s
currency ID selects its discount curve and FX quote. A settlement-currency
share uses the settlement short rate and observes FX at one. Stocks in the same
currency share that curve and FX path. Each stock keeps its own dividend
carry and model. [Runnable USD/EUR worst-of autocall in JPY](../../examples/python/cross_currency_usd_eur_autocall.py)

| Settlement | Performance for coupon and call triggers | Principal and coupon amounts |
| --- | --- | --- |
| `quanto` | Each share price divided by its reference in its own currency | JPY by default. For a foreign amount, supply its `payout_currency_id` and a contractual `fixed_fx_rate`. |
| `floating_fx` | The same local-currency performances | Specify USD or EUR as the `payout_currency_id`. Each amount converts once at that currency’s contractual FX fixing. |
| `composite` | Each share price times its own contemporaneous JPY exchange rate, divided by its JPY reference | JPY. Use JPY amounts and references throughout the graph. |

With `from_currencies`, the graph currency identifies settlement and
`payout_currency_id` denominates all coupons and principal.
The worst performer can change across observations. For floating FX the
cash denomination continues to follow the term sheet whichever name is worst.
Use `with_payment_fx_fixings` for fixing-to-payment lags.

A cash leg may use a currency that has no share in the basket. Supply its
quote and its discount curve, for example `currency_curves={GBP: gbp_curve}`
for a GBP coupon on JPY and USD shares. An explicit curve for a share currency
must agree with the stock market. No FX quote is needed for a JPY share in a
JPY settlement. A wholly domestic basket accepts `fx_quotes=[]`.

Each `fx_quotes` entry gives the currency ID, cash FX in settlement units per
foreign unit, and annual FX volatility. For US and European stocks in that
order and quotes ordered USD then EUR, the correlation order is US stock,
European stock, JPY per USD, JPY per EUR. Supply the four-by-four matrix for
deterministic rates. To simulate stochastic rates, pass
`rates=[jpy_hw, usd_hw, eur_hw]` and append the JPY, USD and EUR rate
drivers to make a seven-by-seven matrix. The two yen FX quotes determine
the implied USD/EUR cross on each path.

Simulation uses the JPY pricing measure. Each stock and foreign short rate
receives the drift adjustment for its own yen exchange rate. All FX paths
use the same JPY short-rate integral and their own foreign-rate integral.
The pricing tests check discounted minima against a joint lognormal formula,
as well as foreign bonds and early redemption with coupon memory.
[Mixed-currency benchmarks](../../tests/python/test_cross_currency_mixed_basket.py)

### Separate coupon and principal currencies

Build both legs from the same worst-of terms. For a JPY 150,000 principal and a
USD 20 quarterly coupon, set `notional=150_000` in the terms and
`coupon_amount=20` on each observation. The coupon currency applies to ordinary
coupons, recovered memory coupons and `call_coupon_amount`. The principal
currency applies to the early redemption and to the protected or reduced
principal at maturity.

```python
coupon, principal = rp.MultiAssetProduct.autocallable_legs(
    **terms, coupon_currency_id=USD, principal_currency_id=JPY)
note = rp.CrossCurrencyEquityBasket.from_legs(
    valuation_date, [coupon, principal], markets, jpy_curve,
    settlement_currency_id=JPY, performance="local", equity_models=models,
    fx_quotes=quotes, correlations=correlations, rates=[jpy_hw, usd_hw, eur_hw])
note = note.with_payment_fx_fixings(list(zip(coupon_payment_dates, coupon_observation_dates)))
estimate = note.price_mc(engine)
```

Each foreign leg uses its own JPY exchange rate. A recovered coupon converts
in full at the fixing for the payment that releases it. By default each foreign
payment fixes on its payment date. To fix earlier, supply one entry per distinct
foreign payment date. Foreign payments on the same date share the fixing date.
JPY principal can depend on a later equity observation even when it shares a
payment date with an earlier-fixed USD coupon.

Use `performance="local"` for local stock references or `performance="composite"`
for JPY stock references. Under either convention, both legs follow the same
equity, FX and rate paths. The engine discounts their sum in JPY and estimates
the standard error for the whole note. [Runnable split-currency note](../../examples/python/cross_currency_usd_eur_autocall.py)
