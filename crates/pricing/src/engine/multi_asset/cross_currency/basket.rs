//! Multi-stock payoff graphs valued jointly in the settlement currency.
use super::basket_leg::{BasketLeg, CashConversion};
use super::basket_rates::BasketRateStep;
use super::exotic::payment_fx_schedule;
use super::model_path::{StockPath, joint_grid};
use super::rates::CrossCurrencyHullWhite;
use super::{EquityFxSettlement, dividend_path, sampling};
use crate::core::{CurrencyId, Date, DayCountConvention, NonNegativeF64, PositiveF64};
use crate::market::{
    CorrelationFactor, CorrelationToleranceConfig, DiscountCurve, EquityMarket,
    LogLinearDiscountCurve,
};
use crate::mc::{EngineConfig, ExecutionPolicy, LocalVolTimeGrid};
use crate::models::{BuehlerDividendModel, HullWhite1Factor, ModelSpec};
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{GraphLimitPolicy, GraphProduct};
use crate::{EquityModelSpec, Estimate};
use std::{collections::BTreeMap, ops::Range, sync::Arc};

#[path = "basket_risk.rs"]
pub(super) mod risk;
pub use risk::BasketRiskFactor;
#[path = "quote_risk.rs"]
pub mod quote_risk;
type PathPayoff<'a> = dyn Fn(&[f64]) -> Result<f64, Error> + Sync + 'a;

/// One foreign equity in the graph's declared underlying order.
#[derive(Clone, Debug)]
pub struct ForeignEquity {
    pub market: EquityMarket,
    pub model: EquityModelSpec,
    pub dividend_model: Option<BuehlerDividendModel>,
}

/// Cash FX and annualized lognormal volatility, quoted in settlement units
/// per unit of the named currency. Quote order defines the FX driver order.
#[derive(Clone, Copy, Debug)]
pub struct BasketFxQuote {
    currency: CurrencyId,
    spot: PositiveF64,
    volatility: NonNegativeF64,
}
impl BasketFxQuote {
    pub fn new(currency: CurrencyId, spot: f64, volatility: f64) -> Result<Self, Error> {
        Ok(Self {
            currency,
            spot: PositiveF64::new(spot, "fx_spot")?,
            volatility: NonNegativeF64::new(volatility, "fx_volatility")?,
        })
    }
}

/// Ordered FX quotes with optional explicit foreign discount curves. Equity
/// markets supply their own currency curves; cash-only currencies supply one here.
#[derive(Clone, Debug)]
pub struct BasketCurrencyMarkets {
    pub fx_quotes: Vec<BasketFxQuote>,
    pub discount_curves: BTreeMap<CurrencyId, Arc<LogLinearDiscountCurve>>,
}

impl From<Vec<BasketFxQuote>> for BasketCurrencyMarkets {
    fn from(fx_quotes: Vec<BasketFxQuote>) -> Self {
        Self {
            fx_quotes,
            discount_curves: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct ForeignCurrency {
    quote: BasketFxQuote,
    discount: LogLinearDiscountCurve,
}

#[derive(Clone, Debug)]
pub struct CrossCurrencyEquityBasket {
    valuation: Date,
    legs: Vec<BasketLeg>,
    stocks: Vec<ForeignEquity>,
    dividend_states: BTreeMap<crate::core::UnderlyingId, dividend_path::BergomiAsOf>,
    domestic_curve: Arc<LogLinearDiscountCurve>,
    currencies: Vec<ForeignCurrency>,
    stock_rate: Vec<usize>,
    maximum_step: f64,
    // Brownian driver order is identical to the public correlation matrix.
    drivers: CorrelationFactor,
    settlement_currency: CurrencyId,
    rates: Option<Vec<HullWhite1Factor>>,
}

pub(super) const TOLERANCE: CorrelationToleranceConfig = CorrelationToleranceConfig {
    symmetry_abs_tol: 1e-12,
    diagonal_abs_tol: 1e-12,
    psd_abs_tol: 1e-12,
    psd_rel_tol: 0.,
    zero_pivot_abs_tol: 1e-12,
    zero_pivot_rel_tol: 0.,
};

impl CrossCurrencyEquityBasket {
    /// Correlation order is the declared equities, quoted FX and, when supplied,
    /// the domestic and foreign short rates. FX is domestic per foreign unit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        valuation: Date,
        product: GraphProduct,
        stocks: Vec<ForeignEquity>,
        domestic_curve: Arc<LogLinearDiscountCurve>,
        settlement: EquityFxSettlement,
        fx: f64,
        fx_volatility: f64,
        correlations: Vec<Vec<f64>>,
        rate_models: Option<(HullWhite1Factor, HullWhite1Factor)>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        let currency = stocks
            .first()
            .ok_or(Error::Invalid(
                "one foreign equity is required per graph underlying",
            ))?
            .market
            .currency();
        let payout = if matches!(settlement, EquityFxSettlement::Composite) {
            product.currency()
        } else {
            currency
        };
        Self::from_currencies(
            valuation,
            product,
            stocks,
            domestic_curve,
            settlement,
            payout,
            vec![BasketFxQuote::new(currency, fx, fx_volatility)?],
            correlations,
            rate_models.map(|(d, f)| vec![d, f]),
            maximum_step,
        )
    }

    /// One quote per distinct foreign currency. The driver order is stocks,
    /// listed FX quotes, then (if supplied) settlement rate and rates in quote order.
    /// The payout currency denominates all coupons and principal before conversion.
    #[allow(clippy::too_many_arguments)]
    pub fn from_currencies(
        valuation: Date,
        product: GraphProduct,
        stocks: Vec<ForeignEquity>,
        domestic_curve: Arc<LogLinearDiscountCurve>,
        settlement: EquityFxSettlement,
        payout_currency: CurrencyId,
        currency_markets: impl Into<BasketCurrencyMarkets>,
        correlations: Vec<Vec<f64>>,
        rate_models: Option<Vec<HullWhite1Factor>>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        let markets = currency_markets.into();
        let foreign_payout = markets
            .fx_quotes
            .iter()
            .position(|q| q.currency == payout_currency);
        let domestic_payout = payout_currency == product.currency();
        let conversion = match settlement {
            EquityFxSettlement::Composite if domestic_payout => CashConversion::Fixed(1.),
            EquityFxSettlement::Quanto { fixed_fx_rate }
                if foreign_payout.is_some() || (domestic_payout && fixed_fx_rate.get() == 1.) =>
            {
                CashConversion::Fixed(fixed_fx_rate.get())
            }
            EquityFxSettlement::FloatingFx => CashConversion::Floating(foreign_payout.ok_or(
                Error::Invalid("floating FX requires a quoted payout currency"),
            )?),
            _ => {
                return Err(Error::Invalid(
                    "invalid cash-flow currency or fixed rate for selected settlement",
                ));
            }
        };
        let domestic = product.currency();
        Self::build(
            valuation,
            vec![BasketLeg::new(
                product,
                conversion,
                matches!(settlement, EquityFxSettlement::Composite),
            )],
            stocks,
            domestic,
            domestic_curve,
            markets,
            correlations,
            rate_models,
            maximum_step,
        )
    }

    /// Compile a dated contract or its lifecycle residual, retaining every
    /// currency group. Market order defines the underlying and driver order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_contract(
        valuation: Date,
        contract: &crate::contract::Contract,
        stocks: Vec<ForeignEquity>,
        settlement_currency: CurrencyId,
        domestic_curve: Arc<LogLinearDiscountCurve>,
        composite: bool,
        currency_markets: impl Into<BasketCurrencyMarkets>,
        correlations: Vec<Vec<f64>>,
        rate_models: Option<Vec<HullWhite1Factor>>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        let ids: Vec<_> = stocks
            .iter()
            .map(|s| s.market.forward().underlying())
            .collect();
        let legs = contract
            .compile_mc(valuation, ids.clone())
            .and_then(|compiled| compiled.into_settlement_legs(settlement_currency, ids))
            .map_err(Error::numerical)?;
        Self::from_legs(
            valuation,
            legs,
            stocks,
            settlement_currency,
            domestic_curve,
            composite,
            currency_markets,
            correlations,
            rate_models,
            maximum_step,
        )
    }

    /// Graph currencies denominate their respective cash flows. Foreign amounts
    /// convert at their contractual fixing and all legs share the same paths.
    #[allow(clippy::too_many_arguments)]
    pub fn from_legs(
        valuation: Date,
        products: Vec<GraphProduct>,
        stocks: Vec<ForeignEquity>,
        settlement_currency: CurrencyId,
        domestic_curve: Arc<LogLinearDiscountCurve>,
        composite: bool,
        currency_markets: impl Into<BasketCurrencyMarkets>,
        correlations: Vec<Vec<f64>>,
        rate_models: Option<Vec<HullWhite1Factor>>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        let markets = currency_markets.into();
        let legs = products
            .into_iter()
            .map(|product| {
                let conversion = if product.currency() == settlement_currency {
                    CashConversion::Fixed(1.)
                } else {
                    let k = markets
                        .fx_quotes
                        .iter()
                        .position(|q| q.currency == product.currency())
                        .ok_or(Error::Invalid(
                            "missing FX quote for cash-flow leg currency",
                        ))?;
                    CashConversion::Floating(k)
                };
                Ok(BasketLeg::new(product, conversion, composite))
            })
            .collect::<Result<_, Error>>()?;
        Self::build(
            valuation,
            legs,
            stocks,
            settlement_currency,
            domestic_curve,
            markets,
            correlations,
            rate_models,
            maximum_step,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        valuation: Date,
        legs: Vec<BasketLeg>,
        stocks: Vec<ForeignEquity>,
        settlement_currency: CurrencyId,
        domestic_curve: Arc<LogLinearDiscountCurve>,
        markets: BasketCurrencyMarkets,
        correlations: Vec<Vec<f64>>,
        rate_models: Option<Vec<HullWhite1Factor>>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        PositiveF64::new(maximum_step, "maximum_step")?;
        let n = stocks.len();
        let BasketCurrencyMarkets {
            fx_quotes,
            discount_curves,
        } = markets;
        let m = fx_quotes.len();
        let underlyings = legs
            .first()
            .ok_or(Error::Invalid("at least one cash-flow leg is required"))?
            .graph
            .underlyings();
        if legs
            .iter()
            .any(|leg| leg.graph.underlyings() != underlyings)
        {
            return Err(Error::Invalid(
                "cash-flow legs must declare the same ordered underlyings",
            ));
        }
        if n == 0 || n != underlyings.len() {
            return Err(Error::Invalid(
                "one foreign equity is required per graph underlying",
            ));
        }
        let indices: BTreeMap<_, _> = fx_quotes
            .iter()
            .enumerate()
            .map(|(i, q)| (q.currency, i))
            .collect();
        if indices.len() != m || indices.contains_key(&settlement_currency) {
            return Err(Error::Invalid(
                "FX quotes must name distinct foreign currencies",
            ));
        }
        if rate_models.as_ref().is_some_and(|r| r.len() != 1 + m) {
            return Err(Error::Invalid(
                "supply the settlement rate followed by one rate per FX quote",
            ));
        }
        let mut curves = vec![None; m];
        for (ccy, curve) in &discount_curves {
            let k = indices.get(ccy).ok_or(Error::Invalid(
                "currency curve requires a corresponding FX quote",
            ))?;
            curves[*k] = Some(curve.as_ref());
        }
        let mut stock_rate = Vec::with_capacity(n);
        for (stock, &id) in stocks.iter().zip(underlyings) {
            let market = stock.market.forward();
            if market.underlying() != id {
                return Err(Error::Invalid("markets must follow graph underlying order"));
            }
            let (rate, df) = if stock.market.currency() == settlement_currency {
                (0, domestic_curve.as_ref())
            } else {
                let &k = indices
                    .get(&stock.market.currency())
                    .ok_or(Error::Invalid("missing FX quote for stock currency"))?;
                (k + 1, *curves[k].get_or_insert(market.discount_curve()))
            };
            stock_rate.push(rate);
            if df.times() != market.discount_curve().times()
                || df.discount_factors() != market.discount_curve().discount_factors()
            {
                return Err(Error::Invalid(
                    "stocks in one currency must share its discount curve",
                ));
            }
            if stock.dividend_model.is_some() {
                dividend_path::validate(&stock.model, market)?;
                if rate_models
                    .as_ref()
                    .is_some_and(|r| !r[rate].is_deterministic())
                {
                    dividend_path::validate_stochastic_rate(&stock.model)?;
                }
            } else if market
                .discrete_dividends()
                .is_some_and(|s| !s.events().is_empty())
            {
                return Err(Error::Invalid(
                    "foreign cash dividends require a dividend model for that stock",
                ));
            }
        }
        let currencies = fx_quotes
            .into_iter()
            .zip(curves)
            .map(|(quote, curve)| {
                let discount = curve
                    .ok_or(Error::Invalid(
                        "each FX quote needs a discount curve from an equity market or currency_curves",
                    ))?
                    .clone();
                Ok(ForeignCurrency { quote, discount })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let market_width = n + m;
        let width = market_width + rate_models.as_ref().map_or(0, Vec::len);
        if correlations.len() != width {
            return Err(Error::Invalid(
                "correlation order is equities, FX quotes, then the settlement and foreign rates when supplied",
            ));
        }
        let factor = CorrelationFactor::compile(correlations, TOLERANCE)?;
        let rates = rate_models.filter(|r| r.iter().any(|m| !m.is_deterministic()));
        let drivers = if rates.is_none() && width > market_width {
            CorrelationFactor::compile(
                factor
                    .canonical()
                    .chunks_exact(width)
                    .take(market_width)
                    .map(|row| row[..market_width].to_vec())
                    .collect(),
                TOLERANCE,
            )?
        } else {
            factor
        };
        if legs
            .iter()
            .any(|leg| leg.graph.payment_dates().iter().any(|&p| p < valuation))
        {
            return Err(Error::Invalid(
                "historical payments require realized cash flows",
            ));
        }
        Ok(Self {
            valuation,
            legs,
            stocks,
            dividend_states: BTreeMap::new(),
            domestic_curve,
            currencies,
            stock_rate,
            maximum_step,
            drivers,
            settlement_currency,
            rates,
        })
    }

    fn equity_rates(&self, stock: usize) -> Result<Option<CrossCurrencyHullWhite>, Error> {
        let Some(rates) = &self.rates else {
            return Ok(None);
        };
        let rate = self.stock_rate[stock];
        if rates[0].is_deterministic() && rates[rate].is_deterministic() {
            return Ok(None);
        }
        let n = self.stocks.len();
        let r = n + self.currencies.len();
        let width = self.drivers.dimension();
        let matrix = if let Some(k) = rate.checked_sub(1) {
            let order = [stock, n + k, r, r + rate];
            order.map(|i| order.map(|j| self.drivers.canonical()[i * width + j]))
        } else {
            // The domestic stock uses the settlement rate, with no FX exposure.
            let rho = self.drivers.canonical()[stock * width + r];
            [
                [1., 0., rho, rho],
                [0., 1., 0., 0.],
                [rho, 0., 1., 1.],
                [rho, 0., 1., 1.],
            ]
        };
        CrossCurrencyHullWhite::new(rates[0].clone(), rates[rate].clone(), matrix).map(Some)
    }

    /// Each pair is (payment, FX fixing). Unlisted payments fix on payment.
    pub fn with_payment_fx_fixings(mut self, schedule: &[(Date, Date)]) -> Result<Self, Error> {
        let floating: Vec<_> = self
            .legs
            .iter_mut()
            .filter(|leg| leg.is_floating())
            .collect();
        if floating.is_empty() {
            return Err(Error::Invalid(
                "payment FX fixings require a floating FX cash-flow leg",
            ));
        }
        let fixings = payment_fx_schedule(
            floating.iter().flat_map(|leg| leg.graph.payment_dates()),
            self.valuation,
            schedule,
        )?;
        for leg in floating {
            let schedule: Vec<_> = fixings
                .iter()
                .filter(|(pay, _)| leg.graph.payment_dates().contains(pay))
                .map(|(&p, &f)| (p, f))
                .collect();
            if schedule.iter().any(|(p, f)| p != f) {
                let g = &leg.graph;
                let dates = g
                    .payment_dates()
                    .iter()
                    .map(|p| *fixings.get(p).unwrap_or(p))
                    .collect();
                GraphProduct::new(
                    g.currency(),
                    g.underlyings().to_vec(),
                    g.source_graph().clone(),
                    dates,
                )
                .map_err(|_| {
                    Error::Invalid(
                        "FX cannot fix before the last equity observation determining its payment",
                    )
                })?;
            }
            leg.set_terms(self.valuation, None, &schedule, &[])?;
        }
        Ok(self)
    }

    /// Condition the residual share, cash dividends and Bergomi OU factor on
    /// their as-of states. Future cash quotes are as-of foreign-measure means.
    pub fn with_dividend_state(
        self,
        underlying: crate::core::UnderlyingId,
        model_age: f64,
        volatility_factor: f64,
        equity_factor: f64,
        dividend_factor: f64,
    ) -> Result<Self, Error> {
        self.with_dividend_factors(
            underlying,
            model_age,
            &[volatility_factor],
            equity_factor,
            dividend_factor,
        )
    }

    /// Supply one OU coordinate for each factor, in model order.
    pub fn with_dividend_factors(
        mut self,
        underlying: crate::core::UnderlyingId,
        model_age: f64,
        volatility_factors: &[f64],
        equity_factor: f64,
        dividend_factor: f64,
    ) -> Result<Self, Error> {
        let stock = self
            .stocks
            .iter()
            .find(|s| s.market.forward().underlying() == underlying)
            .ok_or(Error::Invalid("unknown stock in dividend state"))?;
        let factor = dividend_path::validate_stochastic_rate(&stock.model)?;
        if stock.dividend_model.is_none() || factor.is_none() {
            return Err(Error::Invalid(
                "a dividend state requires Buehler dividends and Bergomi equity volatility",
            ));
        }
        if !factor.is_some_and(|f| f.dimension() == volatility_factors.len()) {
            return Err(Error::Invalid(
                "supply one volatility coordinate per Bergomi factor, in model order",
            ));
        }
        if volatility_factors
            .iter()
            .any(|&v| !v.is_finite() || model_age == 0. && v != 0.)
        {
            return Err(Error::Invalid(
                "Bergomi factor must be finite and starts at zero when model age is zero",
            ));
        }
        let snapshot = dividend_path::BergomiAsOf {
            age: NonNegativeF64::new(model_age, "model_age")?.get(),
            x: [
                volatility_factors[0],
                volatility_factors.get(1).copied().unwrap_or(0.),
            ],
            factors: crate::models::BuehlerDividendState::new(equity_factor, dividend_factor)
                .map_err(Error::numerical)?,
        };
        self.dividend_states.insert(underlying, snapshot);
        Ok(self)
    }

    /// Replace one foreign leg's conversion independently. Each fixing can
    /// precede the amount determination; already published rates are explicit.
    pub fn with_leg_fx(
        mut self,
        leg_index: usize,
        fixed_fx_rate: Option<f64>,
        schedule: &[(Date, Date)],
        known_fixings: &[(Date, f64)],
    ) -> Result<Self, Error> {
        self.legs
            .get_mut(leg_index)
            .ok_or(Error::Invalid("unknown cash-flow leg index"))?
            .set_terms(self.valuation, fixed_fx_rate, schedule, known_fixings)?;
        Ok(self)
    }

    pub fn price_mc(
        &self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        self.with_path_payoff(|times, layout, value| {
            sampling::estimate(times, layout, engine, execution, value)
        })
    }

    fn with_path_payoff<T>(
        &self,
        evaluate: impl FnOnce(&[f64], sampling::Layout, &PathPayoff<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let payoffs = self
            .legs
            .iter()
            .map(|leg| {
                let payments: Vec<_> = leg
                    .graph
                    .payment_dates()
                    .iter()
                    .map(|&pay| {
                        let (fixing, conversion) = leg.conversion(pay);
                        (pay, fixing, conversion)
                    })
                    .collect();
                Ok((
                    leg.graph
                        .source_graph()
                        .compile(GraphLimitPolicy::DEFAULT)?,
                    payments,
                    leg.composite,
                    leg.quantity,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let asset_index: BTreeMap<_, _> = self.legs[0]
            .graph
            .underlyings()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect();
        let mut ends = vec![self.valuation; self.stocks.len()];
        let mut dates: BTreeMap<Date, usize> = payoffs
            .iter()
            .flat_map(|(_, flows, ..)| flows.iter().map(|&(_, f, _)| (f, 0)))
            .collect();
        dates.insert(self.valuation, 0);
        let last_payment = self
            .legs
            .iter()
            .map(|leg| leg.graph.last_payment_date())
            .max()
            .expect("nonempty legs");
        for leg in &self.legs {
            let graph = &leg.graph;
            dates.extend(graph.payment_dates().iter().map(|&p| (p, 0)));
            for &(id, d) in graph.observations() {
                if d < self.valuation {
                    return Err(Error::Invalid(
                        "historical stock observations require known fixings",
                    ));
                }
                let i = asset_index[&id];
                ends[i] = ends[i].max(d);
                dates.insert(d, 0);
            }
        }
        if last_payment == self.valuation {
            let mut total = 0.;
            for (payoff, payments, composite, quantity) in &payoffs {
                let spot = |id, _| {
                    asset_index.get(&id).map(|&i| {
                        self.stocks[i].market.forward().spot().get()
                            * if *composite {
                                self.stock_rate[i]
                                    .checked_sub(1)
                                    .map_or(1., |k| self.currencies[k].quote.spot.get())
                            } else {
                                1.
                            }
                    })
                };
                let amounts = payoff.evaluate_with_pre_dividend_spots(spot, spot)?;
                total += quantity
                    * amounts
                        .iter()
                        .zip(payments)
                        .map(|(a, &(_, _, c))| {
                            a * c.conversion(|k| self.currencies[k].quote.spot.get())
                        })
                        .sum::<f64>();
            }
            return evaluate(&[0.], sampling::Layout::default(), &|_| Ok(total));
        }
        let time = |d| DayCountConvention::Act365F.year_fraction(self.valuation, d);
        let mut events: Vec<_> = dates.keys().map(|&d| time(d)).collect();
        if let Some(rates) = &self.rates {
            events.extend(
                rates
                    .iter()
                    .flat_map(|r| r.volatility_times().iter())
                    .copied()
                    .filter(|&t| t <= time(last_payment)),
            );
        }
        let grid = joint_grid(
            self.stocks
                .iter()
                .zip(&ends)
                .map(|(s, &d)| (&s.model, s.market.forward(), time(d))),
            events,
            self.maximum_step,
        )?;
        let times = grid.nodes();
        for (&d, index) in &mut dates {
            *index = times
                .binary_search_by(|t| t.total_cmp(&time(d)))
                .map_err(|_| Error::Invalid("cash flow date missing from grid"))?;
        }
        let n = times.len() - 1;
        let vols: Vec<_> = self
            .currencies
            .iter()
            .map(|c| c.quote.volatility.get())
            .collect();
        let rate_steps = self
            .rates
            .as_ref()
            .map(|r| BasketRateStep::compile(r, &self.drivers, &vols, times))
            .transpose()?;
        let mut residual_count = 0;
        let mut stocks = Vec::new();
        for (i, (stock, end)) in self.stocks.iter().zip(ends).enumerate() {
            let path = stock.compile(
                &times[..=dates[&end]],
                self.equity_rates(i)?.as_ref(),
                residual_count,
                self.dividend_states
                    .get(&stock.market.forward().underlying())
                    .copied(),
            )?;
            residual_count = path.residuals.end;
            stocks.push(path);
        }
        let domestic = times
            .iter()
            .map(|&t| self.domestic_curve.discount(t))
            .collect::<Result<Vec<_>, _>>()?;
        let fx_forwards = self
            .currencies
            .iter()
            .map(|c| {
                times
                    .iter()
                    .zip(&domestic)
                    .map(|(&t, d)| Ok(c.quote.spot.get() * c.discount.discount(t)? / d))
                    .collect::<Result<Vec<_>, Error>>()
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let payoffs: Vec<_> = payoffs
            .into_iter()
            .map(|(payoff, payments, composite, quantity)| {
                let indices: Vec<_> = payments
                    .iter()
                    .map(|&(p, f, c)| (dates[&p], dates[&f], c))
                    .collect();
                (payoff, indices, composite, quantity)
            })
            .collect();
        let roots: Vec<_> = times.windows(2).map(|w| (w[1] - w[0]).sqrt()).collect();
        let width = self.drivers.dimension();
        let m = self.currencies.len();
        let fx_index = stocks.len();
        let rate_index = fx_index + m;
        let quanto_drift: Vec<_> = self
            .stock_rate
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                rate.checked_sub(1).map_or(0., |k| {
                    self.drivers.canonical()[i * width + fx_index + k]
                        * self.currencies[k].quote.volatility.get()
                })
            })
            .collect();
        let bridge_count = self.rates.as_ref().map_or(0, Vec::len);
        let blocks = width + bridge_count;
        let layout = sampling::Layout {
            equity_steps: 0,
            equity_dimension: residual_count,
            brownian_blocks: 0,
            extra_blocks: blocks,
        };
        evaluate(times, layout, &|normals| {
            let shared = &normals[residual_count..];
            let mut innovations: Vec<Vec<f64>> = stocks
                .iter()
                .map(|s| Vec::with_capacity(s.steps + s.residuals.len()))
                .collect();
            let mut fx: Vec<Vec<f64>> = self
                .currencies
                .iter()
                .map(|c| vec![c.quote.spot.get()])
                .collect();
            let mut discounts = vec![1.];
            let mut carry = vec![vec![0.]; m + 1];
            let mut factors = vec![vec![0.]; m + 1];
            // Rate slot zero is the settlement currency. FX slot k uses rate k+1.
            let mut rate_state = vec![0.; m + 1];
            let mut integrated = vec![0.; m + 1];
            let mut fx_brownian = vec![0.; m];
            let mut brownian = vec![0.; width];
            let mut bridge = vec![0.; bridge_count];
            for (j, &root) in roots.iter().enumerate() {
                let z = |k: usize| shared[k * n + j];
                for (i, value) in brownian.iter_mut().enumerate() {
                    *value = self.drivers.lower()[i * width..i * width + i + 1]
                        .iter()
                        .enumerate()
                        .map(|(l, c)| c * z(l))
                        .sum();
                }
                for (i, stock) in stocks.iter().enumerate() {
                    if j < stock.steps {
                        innovations[i].push(brownian[i] - quanto_drift[i] * root);
                    }
                }
                if let Some(steps) = &rate_steps {
                    for (k, v) in bridge.iter_mut().enumerate() {
                        *v = z(width + k);
                    }
                    steps[j].advance(
                        &mut rate_state,
                        &mut integrated,
                        &brownian[rate_index..],
                        &bridge,
                    );
                }
                let dd = domestic[j + 1] * (-integrated[0]).exp();
                if !dd.is_finite() || dd <= 0. {
                    return Err(Error::Invalid(
                        "nonpositive or nonfinite simulated discount",
                    ));
                }
                discounts.push(dd);
                for k in 0..=m {
                    carry[k].push(integrated[k]);
                    factors[k].push(rate_state[k]);
                }
                for k in 0..m {
                    fx_brownian[k] += root * brownian[fx_index + k];
                    let sigma = vols[k];
                    let x = fx_forwards[k][j + 1]
                        * (integrated[0] - integrated[k + 1] + sigma * fx_brownian[k]
                            - 0.5 * sigma * sigma * times[j + 1])
                            .exp();
                    if !x.is_finite() || x <= 0. {
                        return Err(Error::Invalid("nonpositive or nonfinite simulated FX"));
                    }
                    fx[k].push(x);
                }
            }
            let mut spots = Vec::with_capacity(stocks.len());
            for ((stock, mut shocks), &k) in stocks.iter().zip(innovations).zip(&self.stock_rate) {
                shocks.extend_from_slice(&normals[stock.residuals.clone()]);
                spots.push(match &stock.path {
                    Some(path) => path.spots(
                        stock.spot,
                        &shocks,
                        &stock.cash_carry,
                        self.rates.as_ref().map(|_| &carry[k][..=stock.steps]),
                        &factors[k][..=stock.steps],
                    )?,
                    None => vec![(stock.spot, None)],
                });
            }
            let observed = |id, d, pre, composite| {
                let j = *dates.get(&d)?;
                let i = *asset_index.get(&id)?;
                let (post, before) = spots[i][j];
                Some(
                    (if pre { before.unwrap_or(post) } else { post })
                        * if composite {
                            self.stock_rate[i].checked_sub(1).map_or(1., |k| fx[k][j])
                        } else {
                            1.
                        },
                )
            };
            let mut total = 0.;
            for (payoff, cashflows, composite, quantity) in &payoffs {
                let flows = payoff.evaluate_with_pre_dividend_spots(
                    |u, d| observed(u, d, false, *composite),
                    |u, d| observed(u, d, true, *composite),
                )?;
                total += quantity
                    * flows
                        .iter()
                        .zip(cashflows)
                        .map(|(a, &(p, f, c))| a * discounts[p] * c.conversion(|k| fx[k][f]))
                        .sum::<f64>();
            }
            Ok(total)
        })
    }
}

struct CompiledEquity {
    path: Option<StockPath>,
    spot: f64,
    cash_carry: Vec<f64>,
    steps: usize,
    residuals: Range<usize>,
}

impl ForeignEquity {
    fn compile(
        &self,
        nodes: &[f64],
        rates: Option<&CrossCurrencyHullWhite>,
        residual_begin: usize,
        snapshot: Option<dividend_path::BergomiAsOf>,
    ) -> Result<CompiledEquity, Error> {
        let market = self.market.forward();
        let steps = nodes.len() - 1;
        let path = if steps > 0 {
            let prefix =
                LocalVolTimeGrid::compile(nodes.to_vec(), f64::MAX).map_err(Error::numerical)?;
            Some(StockPath::compile(
                &self.model,
                &prefix,
                market,
                rates,
                self.dividend_model,
                snapshot,
            )?)
        } else {
            None
        };
        let count = path.as_ref().map_or(0, |p| p.layout(steps).0 - steps);
        let cash_carry = nodes
            .iter()
            .map(|&t| {
                Ok(market.dividend_curve().discount(t)? / market.discount_curve().discount(t)?)
            })
            .collect::<Result<_, Error>>()?;
        Ok(CompiledEquity {
            path,
            spot: market.spot().get(),
            cash_carry,
            steps,
            residuals: residual_begin..residual_begin + count,
        })
    }
}
