//! Paired scenario revaluation in settlement units on common random numbers.
use super::*;
use crate::core::{NonNegativeF64, UnderlyingId};
use crate::models::{Black76Spec, BlackScholesSpec};

#[derive(Clone, Copy, Debug)]
pub enum BasketRiskFactor {
    EquitySpot(UnderlyingId),
    EquityVolatility(UnderlyingId),
    DividendYield(UnderlyingId),
    FxSpot(CurrencyId),
    FxVolatility(CurrencyId),
    DiscountRate(CurrencyId),
}

pub(in crate::engine::multi_asset::cross_currency) fn shifted_curve(
    curve: &LogLinearDiscountCurve,
    shift: f64,
) -> Result<Arc<LogLinearDiscountCurve>, Error> {
    let factors = curve
        .times()
        .iter()
        .zip(curve.discount_factors())
        .map(|(&t, &d)| d * (-shift * t).exp())
        .collect();
    Ok(Arc::new(LogLinearDiscountCurve::new(
        curve.id(),
        curve.times().to_vec(),
        factors,
    )?))
}

impl ForeignEquity {
    pub(in crate::engine::multi_asset::cross_currency) fn shift_equity(
        &mut self,
        factor: BasketRiskFactor,
        shift: f64,
    ) -> Result<(), Error> {
        let market = self.market.forward();
        match factor {
            BasketRiskFactor::EquitySpot(_) => {
                let f = market.with_spot(PositiveF64::new(
                    market.spot().get() + shift,
                    "equity_spot",
                )?)?;
                self.market = EquityMarket::new(self.market.currency(), f);
            }
            BasketRiskFactor::DividendYield(_) => {
                let f = market.with_curves(
                    Arc::new(market.discount_curve().clone()),
                    shifted_curve(market.dividend_curve(), shift)?,
                )?;
                self.market = EquityMarket::new(self.market.currency(), f);
            }
            BasketRiskFactor::EquityVolatility(_) => match &mut self.model {
                EquityModelSpec::Standard(ModelSpec::BlackScholes(spec)) => {
                    *spec = BlackScholesSpec::new(spec.volatility().get() + shift)?
                }
                EquityModelSpec::Standard(ModelSpec::Black76(spec)) => {
                    *spec = Black76Spec::new(spec.volatility().get() + shift)?
                }
                EquityModelSpec::StochasticVolatility {
                    initial_volatility, ..
                } => {
                    *initial_volatility =
                        NonNegativeF64::new(*initial_volatility + shift, "initial_volatility")?
                            .get()
                }
                _ => {
                    return Err(Error::Invalid(
                        "this model uses a volatility surface or curve; revalue the bumped input with scenario_pnl",
                    ));
                }
            },
            _ => return Err(Error::Invalid("expected an equity risk factor")),
        }
        Ok(())
    }
}

impl CrossCurrencyEquityBasket {
    /// Add an absolute quote shift, keeping contractual amounts and observed
    /// historical fixings fixed. Rates are continuously compounded zero rates.
    pub fn with_market_shift(
        mut self,
        factor: BasketRiskFactor,
        shift: f64,
    ) -> Result<Self, Error> {
        if !shift.is_finite() {
            return Err(Error::Invalid("market shift must be finite"));
        }
        match factor {
            BasketRiskFactor::FxSpot(ccy) | BasketRiskFactor::FxVolatility(ccy) => {
                let quote = &mut self
                    .currencies
                    .iter_mut()
                    .find(|c| c.quote.currency == ccy)
                    .ok_or(Error::Invalid("unknown foreign FX risk currency"))?
                    .quote;
                match factor {
                    BasketRiskFactor::FxSpot(_) => {
                        quote.spot = PositiveF64::new(quote.spot.get() + shift, "fx_spot")?
                    }
                    _ => {
                        quote.volatility =
                            NonNegativeF64::new(quote.volatility.get() + shift, "fx_volatility")?
                    }
                }
            }
            BasketRiskFactor::EquitySpot(id)
            | BasketRiskFactor::EquityVolatility(id)
            | BasketRiskFactor::DividendYield(id) => {
                let stock = self
                    .stocks
                    .iter_mut()
                    .find(|s| s.market.forward().underlying() == id)
                    .ok_or(Error::Invalid("unknown equity risk underlying"))?;
                stock.shift_equity(factor, shift)?;
            }
            BasketRiskFactor::DiscountRate(ccy) => {
                let bumped = if ccy == self.settlement_currency {
                    self.domestic_curve = shifted_curve(&self.domestic_curve, shift)?;
                    self.domestic_curve.clone()
                } else {
                    let currency = self
                        .currencies
                        .iter_mut()
                        .find(|c| c.quote.currency == ccy)
                        .ok_or(Error::Invalid("unknown discount-curve risk currency"))?;
                    let curve = shifted_curve(&currency.discount, shift)?;
                    currency.discount = (*curve).clone();
                    curve
                };
                for s in self
                    .stocks
                    .iter_mut()
                    .filter(|s| s.market.currency() == ccy)
                {
                    let f = s.market.forward();
                    s.market = EquityMarket::new(
                        ccy,
                        f.with_curves(bumped.clone(), Arc::new(f.dividend_curve().clone()))?,
                    );
                }
            }
        }
        Ok(self)
    }

    /// Settlement-currency PV change (scenario minus base), with a paired standard error.
    pub fn scenario_pnl(
        &self,
        scenario: &Self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        self.paired_change(scenario, None, 1., engine, execution)
    }

    /// Central finite difference per unit of the selected quote. Both bumped
    /// levels must be admissible. One-sided changes can use scenario_pnl.
    pub fn sensitivity(
        &self,
        factor: BasketRiskFactor,
        bump: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        PositiveF64::new(bump, "bump")?;
        let up = self.clone().with_market_shift(factor, bump)?;
        let down = self.clone().with_market_shift(factor, -bump)?;
        down.paired_change(&up, None, 0.5 / bump, engine, execution)
    }

    /// Central second difference per squared quote unit on the same paths.
    pub fn curvature(
        &self,
        factor: BasketRiskFactor,
        bump: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        PositiveF64::new(bump, "bump")?;
        let up = self.clone().with_market_shift(factor, bump)?;
        let down = self.clone().with_market_shift(factor, -bump)?;
        down.paired_change(&up, Some(self), 1. / bump.powi(2), engine, execution)
    }

    fn paired_change(
        &self,
        other: &Self,
        center: Option<&Self>,
        scale: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        if self.valuation != other.valuation
            || self.settlement_currency != other.settlement_currency
        {
            return Err(Error::Invalid(
                "paired revaluation requires the same valuation date and settlement currency",
            ));
        }
        self.with_path_payoff(|times, layout, base| {
            other.with_path_payoff(|other_times, other_layout, scenario| {
                if times != other_times || layout != other_layout {
                    return Err(Error::Invalid(
                        "paired revaluation requires the same simulation grid and driver layout",
                    ));
                }
                if let Some(center) = center {
                    return center.with_path_payoff(|t, l, mid| {
                        if t != times || l != layout {
                            return Err(Error::Invalid(
                                "curvature requires a common simulation grid and driver layout",
                            ));
                        }
                        sampling::estimate(times, layout, engine, execution, |z| {
                            let m = mid(z)?;
                            Ok(((scenario(z)? - m) - (m - base(z)?)) * scale)
                        })
                    });
                }
                sampling::estimate(times, layout, engine, execution, |z| {
                    Ok((scenario(z)? - base(z)?) * scale)
                })
            })
        })
    }
}
