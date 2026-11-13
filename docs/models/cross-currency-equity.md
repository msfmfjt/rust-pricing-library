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
cash-flow graphs, continuously monitored barriers and scheduled exercise use the
selected equity model. On joint paths a continuous barrier uses the left-end
instantaneous diffusion for each interval, including the covariances of stock,
FX and the reserve for future cash dividends. Diffusion arrives at the pre-cash
spot and the post-cash jump can also hit the barrier. Non-lognormal rough SABR
uses an arithmetic bridge in price units. The bridge with variable coefficients
or stochastic rates is a time-discretized approximation. Check convergence
in `maximum_step` separately from the reported sampling error.
[Joint monitoring tests](../../tests/python/test_cross_currency_joint_barrier.py)

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

Scheduled-exercise options use independent policy-training and valuation
paths. For domestic and foreign stochastic rates, for example, pass
`rp.LsmConfig(training_engine, state_variables=["spot", "domestic_rate_factor",
"foreign_rate_factor"], max_degree=2)`. The initial configuration uses only
the monitored spot. The optional `fx_spot` is quoted in settlement-currency
units. At each decision the regression target includes the realized discount
to the future stopping time. A floating option compares exercise in local
currency and converts at its actual stopping time. Sampling error on a price is
conditional on the fitted policy. Finite training samples, the selected basis
and the model time step can also affect that value.
[Exercise benchmarks](../../tests/python/test_cross_currency_joint_exercise.py)


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
rates the residual equity can have constant volatility, one-factor Bergomi or
two-factor Bergomi. Use `EquityModelSpec.stochastic_volatility` to attach
either Bergomi model to the residual share. The two-factor constructor takes
two mean reversions, volatility of volatility, the mixing weight, two
stock/volatility correlations and the correlation between the volatility drivers.
The dividend driver shares the specified stock exposure. Its remaining Brownian
innovation is independent of both volatility drivers' residual innovations.
Supply future cash means under the stock currency's money-market measure,
including dividends after the last contractual observation.
[Dividend-claim benchmarks](../../tests/python/test_cross_currency_dividends.py) ·
[Dividend decomposition, Buehler](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2639318)

#### Conditional reserve and mid-life state

For the one-factor model, the residual equity factor `f` has instantaneous
volatility `sigma0 * exp(eta * (X - eta * Var(X_age)))`. Here `X` is the
uncentered Ornstein–Uhlenbeck factor with unit diffusion under the foreign
measure. In the two-factor model, replace `X` in the volatility formula by
the normalized weighted sum of the two OU coordinates and use the variance
of that sum, including the cross covariance. Each coordinate has its own
mean reversion. [Bergomi, Smile dynamics III](https://www.risk.net/media/download/920801/download)

Each outstanding cash dividend is valued under its foreign payment-date
forward measure. The conditional projected dividend factor has the form
`A(t,X1,X2) * f + B(t) * Y + C(t)`, where `Y` is the stochastic cash factor.
The implementation retains the analytical constant-volatility coefficients and
computes the state-dependent correction to `A` by backward Gaussian quadrature.
It includes the stock/rate covariance and the correlation between stock and
volatility. A zero volatility of volatility gives the constant-volatility reserve.
For unequal reversions and two active weights, the backward calculation uses
a 65 × 65 factor grid, three Gaussian nodes per dimension and cubic interpolation
in the interior. At the edges and when looking up the reserve on a simulated
path it uses bilinear interpolation. Equal reversions or an inactive weight
give an exact reduction to one OU coordinate. The one-dimensional calculation
uses 257 grid points and seven Gaussian nodes. Both grids include rate knots
and time intervals of at most one week. Each axis extends nine OU standard
deviations to the cash date beyond the absolute initial coordinate. Quadrature
uses constant boundary values for its tails, and a realized path outside the
grid raises an error. Price standard errors measure sampling uncertainty.
Time stepping and reserve interpolation also affect accuracy.
[Independent forward-simulation benchmark](../../crates/pricing/src/engine/processes/stochastic_dividends/hull_white/bergomi_reserve.rs)

At issue the normalized residual equity and cash factors start at one and
the volatility factor starts at zero. On a later date, rebuild the basket from
the remaining contractual outputs and current market curves, spots and cash
forecasts. Supply the elapsed model time in years and the estimated current
factors for each stock. The US stock in the April example uses these values.

```python
live = live.with_dividend_state(us_stock_id, model_age=age, volatility_factor=.16,
                                equity_factor=.97, dividend_factor=1.06)
```

For the two-factor US stock, pass both uncentered coordinates in the same order
as the two mean reversions. For example, its April state is
`volatility_factor=(.16, -.09)`. Each stock has an independent choice of model
and its own observed state.

The model resumes mean reversion and the variance centering at that age.
Each remaining cash quote is the **current expected ex-date amount per share**.
The engine divides it by the conditional mean of its dividend factor and
funds the remaining residual shares from the current physical spot. Omitting
the state initializes a newly calibrated model at the valuation date.
On bumps, historical FX, past observations and the supplied factors stay fixed.
`equity_volatility` shifts the residual volatility scale, `equity_vol_of_vol`
shifts the Bergomi volatility of volatility, and `dividend_volatility`
shifts the Buehler dividend diffusion. For a seasoned position,
`bergomi_factor_1` and `bergomi_factor_2` bump each current OU coordinate
in model order, holding the other observed inputs fixed. These risks use the underlying ID. Spot,
currency curves, FX and joint scenarios use the existing basket risk interface.
Run `python examples/python/cross_currency_stochastic_dividend_autocall.py --two-factor`
for issue and April PVs, factor sensitivities and a ten-basis-point increase
in USD short-rate volatility.
[USD-coupon, JPY-principal autocall example](../../examples/python/cross_currency_stochastic_dividend_autocall.py) ·
[JPY funding, replay and risk tests](../../tests/python/test_cross_currency_bergomi_cash_rates.py)

#### Calibrating the current states from option premiums

On a seasoned two-factor basket, pass two European premiums per selected stock
in short, then long maturity order. Each tuple contains expiry, call/put, strike,
the listed option's market forward, and premium per share in the stock currency.
Expiry is also the payment date. The following April 10 inputs are illustrative.

```python
quotes = {
    10: [("2026-07-10", "call", 99., 98.20, 3.34), ("2027-04-10", "put", 96., 95.46, 7.34)],
    20: [("2026-07-10", "call", 76., 75.89, 3.55), ("2027-04-10", "put", 72., 71.91, 7.09)],
}
report = live.bergomi_quote_risk(quotes, engine, bump=.005, replications=8)
fast, slow = report.states[10]
usd_short, usd_long = report.vegas[10]
print(usd_short.value, usd_short.standard_error)  # JPY per +1 volatility point
```

The current OU coordinates supplied via `with_dividend_state` are the initial
guess. The fit keeps model age, normalized residual-share and dividend states,
cash forecasts and all model parameters fixed. The two pricing instruments use
the stock's local currency measure, its Hull–White curve and its equity/rate
correlation. Distinct expiries and two separately identifiable factors are
required. The fit checks the scaled option-price Jacobian before and after
calibration. It reports a failure if the quotes cannot be fitted.

The quote convention uses Black implied volatility, ACT/365F, the supplied
forward and the stock currency's discount factor. For each tenor the engine
replaces its premium by the Black prices at IV plus and minus the bump, refits
both coordinates to each pair, and revalues the original note. The other
premium and the other stock's fitted state stay fixed. Historical FX, the
coupon record and the JPY principal come from the original contract. Each
vega is `0.01 * (PV_up - PV_down) / (2 * bump)` in settlement currency.

One independent replication covers the base fit, all bumped fits and paired
note revaluations. Common random numbers within each fit and each pair reduce
noise. Separate seeds drive calibration, out-of-sample local premium checks,
and note valuation. The report averages the complete replications and reports
their sample standard deviation divided by the square root of their count.
This measures simulation uncertainty conditional on the input premiums and
model. Quote spreads, model error and finite-grid or finite-bump bias require
separate analysis. `present_value` is the mean of the independently calibrated
note PVs. `price_residuals` gives out-of-sample model premium minus input premium
in each stock currency. `implied_volatilities`, `price_residuals` and `vegas`
follow the short/long ordering. [Runnable calibration and risk example](../../examples/python/cross_currency_autocall_quote_risk.py) ·
[Replication and fixed-payment checks](../../tests/python/test_cross_currency_quote_risk.py)

#### Hedging the four quote vegas

Use the same short/long quote pairs to size trades in the listed options.
Quantities are signed trades from a zero hedge position. Positive buys the
option, negative sells it. By default the position is one long note and
one option unit covers one share. Set `note_quantity` for a signed book size.
Set `contract_sizes` by stock in short/long order for actual contract
multipliers. With `whole_contracts=True`, each trade is rounded to the
nearest integer contract, with halfway cases rounded away from zero.

```python
hedge = live.bergomi_quote_hedge(quotes, engine, contract_sizes={10: (100.,100.), 20: (100.,100.)},
                                whole_contracts=True, bump=.005, replications=8)
print(hedge.theoretical_quantities[20], hedge.quantities[20])  # signed EUR contracts
for book in (hedge.theoretical_hedged, hedge.hedged):
    eur_long = book.vegas[20][1]
    print(eur_long.value, eur_long.standard_error)
```

For a single quote bump, the other three market premiums are fixed. The
settlement-currency vega per long option contract is the central difference
in its local Black premiums, multiplied by current spot FX, shares per contract
and `0.01 / (2 * bump)`. The ideal signed trade is the negative of the note
position's quote vega divided by that contract vega. `unit_vegas`,
`theoretical_quantities`, `quantity_standard_errors` and `quantities` have the
same stock and maturity ordering as the inputs. The sizing standard error
applies to the unrounded quantity.

The engine freezes the trades and repeats the short/long fits on independent
random numbers. Each bumped valuation prices the note and all held European
options on the same settlement-currency paths. The hedge options exercise on
local physical shares and convert at their expiry FX. The original note keeps
its own observation and FX-fixing terms. `unhedged.vegas` is the sizing estimate,
`theoretical_hedged.vegas` gives residuals for frozen fractional quantities, and
`hedged.vegas` gives residuals for the selected trades (integer contracts when
requested). Both checks use the same holdout calibrations. For a JPY book all
three sets of sensitivities are JPY per volatility point. Each residual's standard error includes refitting and
portfolio sampling uncertainty conditional on the fixed trade quantities.
The hedged PV is the gross value of the held instruments before the acquisition
cost of the options. [Reproducible USD/EUR hedge](../../examples/python/cross_currency_autocall_quote_hedge.py)

#### Uncertainty from choosing the option quantities

Run `python examples/python/cross_currency_autocall_hedge_study.py` to repeat the
entire calculation for 100 long notes, using 100 shares per option contract.
The example first reproduces the previous seed as a reference. Its summary
uses 16 new, independent cycles. Each cycle uses eight sizing replications
and eight holdout replications. Each replication refits the states and uses a
separate valuation stream. The fractional and integer books share the holdout
fits and random streams. Each short/long position is rounded separately in
each cycle.

For cycle `b`, let `r_b` be the mean residual vega for one maturity bucket
and `s_b` its holdout standard error conditional on that cycle's chosen trades.
The study reports three scales of uncertainty.

- `fixed_se = sqrt(mean(s_b²))` estimates the typical standard error of a
  single eight-replication holdout with preset quantities.
- `full_se = sample_sd(r_b)` estimates the standard error for one complete
  sizing and holdout cycle, including variation in the selected quantities.
- `mean_se = full_se / sqrt(cycles)` is the standard error of the mean across
  independent cycles. Use this scale to assess the reproducibility of the
  mean sign. The count of negative cycles also measures sign consistency.

For each bucket, `rounding` summarizes the paired difference
`integer residual − fractional residual` within each complete cycle.
`mean_absolute` is the average size of that difference before positive and
negative rounding effects cancel across cycles.
`contracts` gives the frequency of each integer trade across sizing cycles.
All output vegas are yen per volatility point for 100 notes. Fractional
residuals and their errors scale linearly to one note by dividing by 100.
Integer positions depend on book size and are rounded after sizing the book.
The uncertainty estimates condition on the illustrative market data and
model settings. [Study script](../../examples/python/cross_currency_autocall_hedge_study.py)

The April 10 example produced the following results with 16 new cycles. Residuals
and standard errors in this table are JPY per volatility point for **100 notes**.
The paired difference is integer minus fractional, with the SE of its 16-cycle mean.

| Option | Fractional mean residual | Fixed-quantity SE per cycle | Full-cycle SE | Integer mean residual | Integer fixed-quantity SE per cycle | Integer full-cycle SE | Paired mean difference ± SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| USD 3m call | -410.2 | 751.9 | 1,047.8 | -297.7 | 752.1 | 1,356.4 | +112.5 ± 203.2 |
| USD 12m put | -56.7 | 1,128.4 | 1,946.6 | +484.3 | 1,128.6 | 2,373.4 | +540.9 ± 388.4 |
| EUR 3m call | -476.1 | 1,134.0 | 1,661.3 | -601.2 | 1,133.6 | 2,123.9 | -125.1 ± 217.5 |
| EUR 12m put | +656.0 | 1,220.0 | 1,838.6 | +142.0 | 1,219.7 | 2,440.5 | -513.9 ± 295.4 |

The EUR 12m put fractional residual was negative in 4 of 16 new cycles.
Per note its mean was +6.56 JPY, with SE 4.60 JPY and an approximate 95%
Student-t interval of [-3.24, 16.36] JPY (15 degrees of freedom). The
reference cycle reproduced the earlier −33.87 JPY per note. Its 100-note
integer trades were 2, 5, 8 and 9 contracts, in table order. In that reference
cycle the four rounding differences were -1,209, +1,756, +1,027 and -2,067 JPY per point.
[Interval for a mean](https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

To repeat the second independent batch, run the example with
`--first-cycle 17 --cycles 16`. To reproduce the pooled study from scratch,
run it with `--cycles 32`. Cycle zero is the original reference and is
excluded from both estimates. Each positive cycle index selects a distinct
block of 48 master seeds, 24 for sizing and 24 for holdout valuation.

Across cycles 1–32 the following standard errors apply to a single run on
100 notes, in JPY per volatility point.

| Option | Fractional fixed-quantity SE | Fractional full-cycle SE | Integer fixed-quantity SE | Integer full-cycle SE |
|---|---:|---:|---:|---:|
| USD 3m call | 733 | 904 | 733 | 1,339 |
| USD 12m put | 1,100 | 1,685 | 1,100 | 1,780 |
| EUR 3m call | 1,098 | 1,462 | 1,098 | 1,891 |
| EUR 12m put | 1,372 | 1,881 | 1,372 | 2,064 |

The next table shows the mean residual and SE of the 32-cycle mean for the
same 100 notes. In each cycle the fractional and integer books are valued
on the same holdout paths. The last column averages the absolute paired
difference across cycles.

| Option | Fractional mean ± SE | Integer mean ± SE | Integer minus fractional ± paired SE | Mean absolute difference |
|---|---:|---:|---:|---:|
| USD 3m call | −194 ± 160 | −40 ± 237 | +155 ± 142 | 716 |
| USD 12m put | +43 ± 298 | +787 ± 315 | +745 ± 219 | 1,203 |
| EUR 3m call | −430 ± 258 | −586 ± 334 | −156 ± 145 | 737 |
| EUR 12m put | +349 ± 333 | +282 ± 365 | −67 ± 254 | 1,233 |

In cycles 17–32 the EUR 12m put fractional residual was negative 10 times.
Its per-note mean was +0.42 ± 4.83 JPY per volatility point, with a 95%
Student-t interval of [−9.87, +10.72]. Across all 32 independent cycles
it was negative 14 times, with a per-note mean of +3.49 ± 3.33 and a 95%
interval of [−3.29, +10.27]. A negative residual occurred repeatedly, but
the data do not establish a negative mean. With integer contracts, the
100-note EUR 12m put residual was negative in 12 of 32 cycles.

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
foreign payment date. This common schedule applies to all floating legs.
JPY principal can depend on a later equity observation even when it shares a
payment date with an earlier-fixed USD coupon.

Use `performance="local"` for local stock references or `performance="composite"`
for JPY stock references. Under either convention, both legs follow the same
equity, FX and rate paths. The engine discounts their sum in JPY and estimates
the standard error for the whole note. [Runnable split-currency note](../../examples/python/cross_currency_usd_eur_autocall.py)

`with_leg_fx` replaces the terms of a single foreign leg by index in the
list passed to `from_legs`. For example, fix a USD coupon and EUR redemption
independently even when both pay in JPY on the same date:

```python
note = note.with_leg_fx(0, schedule=[(payment, coupon_fixing)])
note = note.with_leg_fx(1, fixed_fx_rate=160.0)  # contractual JPY per EUR
```

A leg-specific fixing can precede equity determination. The joint simulation
values both dates on the same path. For a fixing before valuation, supply
`known_fixings=[(fixing_date, published_fx)]` with the corresponding schedule.
A supplied valuation-day fixing also takes precedence over current spot.
Unlisted payments use payment-day FX. Calling `with_leg_fx(index)` restores
that leg's payment-day convention.

### Valuation after issue

For a previously issued autocall, call `remaining_cashflows(as_of, ...)` on
each original leg before passing the residual graphs to `from_legs`. Pass
published `fixings=[(underlying_id, observation_date, value), ...]` in the
observation units of the note. For composite performance use the historical
share price times the historical settlement-currency FX at that observation.
Pre-dividend observations have a separate `pre_dividend_fixings` input.

Pass `settled_outputs` as zero-based indices of confirmed payments in that
graph. The residual carries missed coupons forward, removes cash flows
cancelled by an observed call, and retains determined payments still due.
A nonzero overdue amount needs explicit settlement or closeout. A fully
redeemed and settled note has zero residual value. If FX for an unpaid amount
was already published, use `with_leg_fx` on the residual with its fixing date
and published rate. The runnable example rolls the USD-coupon, JPY-principal
note to April 10. Its first USD 20 coupon fixed at 147 JPY/USD, so it still
pays JPY 2,940 on April 14. It reports the remaining JPY PV, both equity
and FX deltas, and sensitivities to all three rate curves with sampling errors.
[Worked note and risk example](../../examples/python/cross_currency_usd_eur_autocall.py) · [Seasoned note tests](../../tests/python/test_cross_currency_seasoned.py)

### Market sensitivities and scenario P&L

`basket.sensitivity("fx_spot", USD, 0.1, engine)` returns the central price
difference divided by twice the absolute quote bump. The result and its
standard error are in settlement units per one unit change in the quote.
For `equity_spot`, `equity_volatility`, and `dividend_yield` supply a stock
ID. For `fx_spot`, `fx_volatility`, and `discount_rate` supply a currency
ID. Multiply rate sensitivity by 0.0001 for a one-basis-point exposure, or
volatility sensitivity by 0.01 for one volatility point. Both sides of a
central bump must be valid. Scalar equity volatility applies to Black and
initial Bergomi volatility inputs.

`with_market_shift(factor, id, shift)` returns a shifted basket. Fixed FX,
historical fixings, stock reference levels and already-determined amounts
retain their contractual values. A parallel zero-rate shift is applied to
all markets in its currency and to the discounting or FX-forward curve.
For a one-sided bump, or a scenario built with new surfaces, rate parameters
or correlations, use `base.scenario_pnl(scenario, engine)`. This reports
scenario PV minus base PV. Both valuations must share the as-of, settlement
currency, grid and random layout. The standard error is estimated from paired
changes across paths or randomized scrambles.
[Risk benchmarks](../../tests/python/test_cross_currency_risk.py)

Use `order=2` for a centered second difference (equity or FX gamma for a
spot bump). `CrossCurrencyEquityOption` also supports `with_market_shift`,
`sensitivity` and `scenario_pnl`. Vanilla uses its closed form where available.
Other single-stock products repeat their native pricing calculation on common
seeds. The `replications` argument defaults to eight full runs per scenario.
For Bermudans, pass `lsm`; exercise policies are retrained for each bump and
each independent replication. Errors include policy-training variability.
A fixed historical FX is never bumped in a remaining cash-flow graph.

For an issued vanilla, digital, Asian, discrete barrier or lookback,
`product.to_graph(original_as_of, currency_id=amount_currency)` exports its
dated payoff. A vanilla can also specify `payment_date`. Freeze history on
that graph with `remaining_cashflows`, then value the outstanding leg with
`from_legs`. A floating single-stock payoff originally fixing at expiry
should retain that fixing in the leg schedule, including after expiry.
A continuously monitored barrier accepts a confirmed `historical_hit` when
its original monitoring schedule includes dates before valuation. After a hit,
or once the monitoring window has ended with a confirmed no-hit, use
`to_graph(as_of, currency_id=...)` and freeze the final stock fixing to value
a payment outstanding after expiry.

For a scheduled-exercise right call
`product.after_exercise_history(as_of, lapsed_dates=[...], exercised_on=...)`.
Explicitly lapse each past decision preceding exercise or valuation. With
outstanding rights the result retains the remaining schedule. A confirmed
exercise returns the payoff at the actual exercise date. Export it in its
contractual amount currency and replay its equity fixing, FX fixing and
settlement just as for a vanilla. A fully lapsed right is worth zero. [Dated-product replay](../../tests/python/test_cross_currency_seasoned.py)
