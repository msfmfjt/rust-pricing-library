//! Native single-stock sensitivities, including bridge barriers and exercise policies.
use super::basket::risk::shifted_curve;
use super::*;
use crate::engine::risk::report::estimate_from_statistics;
use crate::mc::{DeterministicStatistics, PseudoMcConfig, RqmcConfig};

fn replicate(engine: EngineConfig, i: u32) -> Result<EngineConfig, Error> {
    let offset = u64::from(i).wrapping_mul(0x9e3779b97f4a7c15);
    Ok(match engine {
        EngineConfig::PseudoMonteCarlo(c) => EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                c.master_seed().wrapping_add(offset),
                c.independent_sampling_units().get(),
                c.variance_reduction(),
            )
            .map_err(Error::numerical)?,
        ),
        EngineConfig::RandomizedQuasiMonteCarlo(c) => EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                c.points_per_scramble().get(),
                c.scramble_count().get(),
                c.master_scramble_seed().wrapping_add(offset),
                c.variance_reduction(),
            )
            .map_err(Error::numerical)?,
        ),
    })
}

impl CrossCurrencyEquityOption {
    pub fn with_market_shift(&self, factor: BasketRiskFactor, shift: f64) -> Result<Self, Error> {
        if !shift.is_finite() {
            return Err(Error::Invalid("market shift must be finite"));
        }
        let mut stock = ForeignEquity {
            market: self.foreign_market.clone(),
            model: self
                .equity_model
                .clone()
                .unwrap_or(EquityModelSpec::Standard(ModelSpec::BlackScholes(
                    self.dynamics.equity,
                ))),
            dividend_model: self.dividends,
        };
        let mut spot = self.fx_spot.get();
        let mut vol = self.dynamics.fx.volatility().get();
        let mut domestic = self.domestic_curve.clone();
        let foreign = stock.market.currency();
        match factor {
            BasketRiskFactor::EquitySpot(id)
            | BasketRiskFactor::DividendYield(id)
            | BasketRiskFactor::EquityVolatility(id)
                if id == self.product.underlying() =>
            {
                stock.shift_equity(factor, shift)?
            }
            BasketRiskFactor::FxSpot(ccy) if ccy == foreign => spot += shift,
            BasketRiskFactor::FxVolatility(ccy) if ccy == foreign => vol += shift,
            BasketRiskFactor::DiscountRate(ccy) if ccy == self.product.currency() => {
                domestic = shifted_curve(&domestic, shift)?
            }
            BasketRiskFactor::DiscountRate(ccy) if ccy == foreign => {
                let f = stock.market.forward();
                stock.market = EquityMarket::new(
                    foreign,
                    f.with_curves(
                        shifted_curve(f.discount_curve(), shift)?,
                        Arc::new(f.dividend_curve().clone()),
                    )?,
                );
            }
            _ => return Err(Error::Invalid("unknown equity or currency risk identifier")),
        }
        let mut shifted = Self::from_model_with_dividends(
            self.valuation_date,
            self.product.clone(),
            &stock.market,
            domestic,
            spot,
            stock.model,
            vol,
            self.dynamics.correlation,
            self.settlement,
            Some(self.payment_date),
            self.maximum_step,
            stock.dividend_model,
        )?;
        if let Some(r) = &self.rates {
            shifted = shifted.with_hull_white_rates(
                r.domestic.clone(),
                r.foreign.clone(),
                Some(r.correlations),
            )?;
        }
        if !self.payment_fx_fixings.is_empty() {
            let dates: Vec<_> = self
                .payment_fx_fixings
                .iter()
                .map(|(&p, &f)| (p, f))
                .collect();
            shifted = shifted.with_payment_fx_fixings(&dates)?;
        }
        Ok(shifted)
    }

    pub fn scenario_pnl(
        &self,
        other: &Self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<&LsmConfig>,
        replications: u32,
    ) -> Result<Estimate, Error> {
        self.paired_change(other, None, 1., engine, execution, lsm, replications)
    }

    pub fn sensitivity(
        &self,
        factor: BasketRiskFactor,
        bump: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<&LsmConfig>,
        replications: u32,
    ) -> Result<Estimate, Error> {
        PositiveF64::new(bump, "bump")?;
        self.with_market_shift(factor, -bump)?.paired_change(
            &self.with_market_shift(factor, bump)?,
            None,
            0.5 / bump,
            engine,
            execution,
            lsm,
            replications,
        )
    }

    pub fn curvature(
        &self,
        factor: BasketRiskFactor,
        bump: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<&LsmConfig>,
        replications: u32,
    ) -> Result<Estimate, Error> {
        PositiveF64::new(bump, "bump")?;
        self.with_market_shift(factor, -bump)?.paired_change(
            &self.with_market_shift(factor, bump)?,
            Some(self),
            1. / bump.powi(2),
            engine,
            execution,
            lsm,
            replications,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn paired_change(
        &self,
        other: &Self,
        center: Option<&Self>,
        scale: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<&LsmConfig>,
        replications: u32,
    ) -> Result<Estimate, Error> {
        if self.valuation_date != other.valuation_date
            || self.product.currency() != other.product.currency()
        {
            return Err(Error::Invalid(
                "paired revaluation requires the same valuation date and settlement currency",
            ));
        }
        let closed_form = |o: &Self| {
            matches!(o.product, ProductSpec::EuropeanVanilla(_))
                && o.equity_model.is_none()
                && o.rates.is_none()
                && o.dividends.is_none()
        };
        if lsm.is_none() && closed_form(self) && closed_form(other) {
            let lo = self.analytical_price()?;
            let hi = other.analytical_price()?;
            let difference = match center {
                Some(c) => {
                    let m = c.analytical_price()?;
                    (hi - m) - (m - lo)
                }
                None => hi - lo,
            };
            let value = difference * scale;
            return Estimate::new(value, 0., value, value, EstimatorKind::Analytical, 1)
                .map_err(Error::numerical);
        }
        if replications < 2 {
            return Err(Error::Invalid(
                "risk estimation needs at least two independent replications",
            ));
        }
        let mut values = Vec::new();
        for i in 0..replications {
            let engine = replicate(engine, i)?;
            let training = lsm
                .map(|c| {
                    LsmConfig::new(
                        replicate(c.training_engine(), i)?,
                        c.state_variables().to_vec(),
                        c.basis().clone(),
                        c.itm_abs_tolerance(),
                        c.cpqr_config(),
                        c.max_matrix_elements(),
                    )
                    .map_err(Error::numerical)
                })
                .transpose()?;
            let base = self.price_mc_with_lsm(engine, execution, training.clone())?;
            let mid = center
                .map(|c| c.price_mc_with_lsm(engine, execution, training.clone()))
                .transpose()?;
            let scenario = other.price_mc_with_lsm(engine, execution, training)?;
            let (lo, hi) = (base.value().get(), scenario.value().get());
            values.push(mid.map_or(hi - lo, |v| (hi - v.value().get()) - (v.value().get() - lo)));
        }
        let kind = match engine {
            EngineConfig::PseudoMonteCarlo(_) => EstimatorKind::PseudoMonteCarlo,
            EngineConfig::RandomizedQuasiMonteCarlo(_) => EstimatorKind::RandomizedQuasiMonteCarlo,
        };
        estimate_from_statistics(
            DeterministicStatistics::from_ordered_values_two_pass(&values),
            u64::from(replications),
            scale,
            kind,
        )
        .map_err(Error::numerical)
    }
}
