use super::{CrossCurrencyEquityOption, EquityFxSettlement};
use crate::core::{Date, DayCountConvention, PositiveF64};
use crate::market::{
    DiscountCurve, EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext,
};
use crate::mc::{EngineConfig, ExecutionPolicy, LsmConfig};
use crate::models::{BlackScholesSpec, ModelSpec};
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{
    GraphProduct, GraphProductError, ProductSpec, SourceGraphBuilder, SourceOpcode,
};
use crate::risk::{RiskRequest, SmileDynamics};
use crate::{Estimate, PricingRequest, SimulationPlan};
use std::collections::BTreeMap;
use std::sync::Arc;

impl CrossCurrencyEquityOption {
    /// Set (payment date, FX fixing date) pairs for a floating-FX graph.
    /// Unspecified payments fix on payment. Each fixing must be on or after
    /// the last equity observation determining the associated payments.
    pub fn with_payment_fx_fixings(mut self, schedule: &[(Date, Date)]) -> Result<Self, Error> {
        let (EquityFxSettlement::FloatingFx, ProductSpec::Graph(graph)) =
            (self.settlement, &self.product)
        else {
            return Err(Error::Invalid(
                "payment FX fixings require a floating_fx payoff graph",
            ));
        };
        let mut fixings = graph
            .payment_dates()
            .iter()
            .map(|&payment| (payment, None))
            .collect::<BTreeMap<_, _>>();
        for &(payment, fixing) in schedule {
            let entry = fixings.get_mut(&payment).ok_or(Error::Invalid(
                "FX schedule contains an unknown payment date",
            ))?;
            if fixing < self.valuation_date || fixing > payment {
                return Err(Error::Invalid(
                    "require valuation date <= FX fixing date <= payment date",
                ));
            }
            if entry.replace(fixing).is_some() {
                return Err(Error::Invalid(
                    "duplicate payment date in FX fixing schedule",
                ));
            }
        }
        self.payment_fx_fixings = schedule.iter().copied().collect();
        if schedule.iter().all(|(payment, fixing)| payment == fixing) {
            self.foreign_fixing_product = None;
            return Ok(self);
        }

        // At FX fixing F the foreign amount is known. Its domestic value for
        // payment P is X(F) * amount * Dd(P)/Dd(F). Use F as the equivalent
        // foreign payment date so the foreign engine discounts it with Df(F).
        let source = graph.source_graph();
        let mut builder = SourceGraphBuilder::from_graph(source);
        let mut outputs = Vec::with_capacity(source.outputs().len());
        let mut dates = Vec::with_capacity(source.outputs().len());
        let time = |date| DayCountConvention::Act365F.year_fraction(self.valuation_date, date);
        for (&output, &payment) in source.outputs().iter().zip(graph.payment_dates()) {
            let fixing = fixings[&payment].unwrap_or(payment);
            let value = if fixing == payment {
                output
            } else {
                let log_ratio = self.domestic_curve.log_discount(time(payment))?
                    - self.domestic_curve.log_discount(time(fixing))?;
                let ratio = builder
                    .literal(PositiveF64::new(log_ratio.exp(), "payment_discount_ratio")?.get())?;
                builder.push(SourceOpcode::Multiply {
                    left: output,
                    right: ratio,
                })?
            };
            outputs.push(value);
            dates.push(fixing);
        }
        // GraphProduct validates that each output is determined by its own date.
        let adjusted = GraphProduct::new(
            graph.currency(),
            graph.underlyings().to_vec(),
            builder.finish(outputs),
            dates,
        )
        .map_err(|error| match error {
            // The original graph validated with the same cash flows and stock.
            // Only bringing their dates forward can invalidate an observation.
            GraphProductError::Invalid(_) => Error::Invalid(
                "FX cannot fix before the last equity observation that determines its payment",
            ),
            other => Error::numerical(other),
        })?;
        self.foreign_fixing_product = Some(ProductSpec::Graph(adjusted));
        Ok(self)
    }

    pub(super) fn price_exotic(
        &self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<LsmConfig>,
    ) -> Result<Estimate, Error> {
        let equity = self.foreign_market.forward();
        let (forward, model, multiplier) = match self.settlement {
            EquityFxSettlement::Quanto { fixed_fx_rate } => {
                let forward = EquityForward::new(
                    equity.underlying(),
                    equity.spot(),
                    self.domestic_curve.clone(),
                    Arc::new(self.quanto_carry_curve()?),
                );
                (forward, self.dynamics.equity, fixed_fx_rate.get())
            }
            EquityFxSettlement::FloatingFx => {
                // Changing numeraire weights the entire equity path up to FX fixing.
                // Graph cashflows are valued at their FX dates. Exercise pays immediately.
                let factor = match &self.product {
                    ProductSpec::Graph(_) | ProductSpec::AmericanVanilla(_) => self.fx_spot.get(),
                    _ if self.payment_date == self.product.expiry() => self.fx_spot.get(),
                    _ => {
                        let time = DayCountConvention::Act365F
                            .year_fraction(self.valuation_date, self.payment_date);
                        self.fx_forward.get()
                            * (self.domestic_curve.log_discount(time)?
                                - equity.discount_curve().log_discount(time)?)
                            .exp()
                    }
                };
                (equity.clone(), self.dynamics.equity, factor)
            }
            EquityFxSettlement::Composite => {
                // X*S is a domestic equity paying the original continuous dividend yield.
                let spot =
                    PositiveF64::new(self.fx_spot.get() * equity.spot().get(), "composite_spot")?;
                let forward = EquityForward::new(
                    equity.underlying(),
                    spot,
                    self.domestic_curve.clone(),
                    Arc::new(equity.dividend_curve().clone()),
                );
                (
                    forward,
                    BlackScholesSpec::new(self.dynamics.composite_volatility())?,
                    1.0,
                )
            }
        };
        let factor = PositiveF64::new(multiplier, "fx_pv_multiplier")?.get();
        let market = MarketContext::Equity(EquityMarket::new(self.product.currency(), forward));
        let request = PricingRequest::new_with_lsm(
            self.valuation_date,
            self.foreign_fixing_product
                .as_ref()
                .unwrap_or(&self.product)
                .clone(),
            market,
            ModelSpec::BlackScholes(model),
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            lsm,
        )
        .map_err(Error::numerical)?;
        let estimate = SimulationPlan::compile_with_carry_breakpoints(&request, execution)
            .and_then(|plan| plan.execute())
            .map_err(Error::numerical)?
            .pricing_result
            .value;
        if factor == 1.0 {
            return Ok(estimate);
        }
        let ci = estimate.confidence_interval();
        Estimate::new(
            estimate.value().get() * factor,
            estimate.standard_error().get() * factor,
            ci.lower().get() * factor,
            ci.upper().get() * factor,
            estimate.estimator(),
            estimate.effective_sampling_units().get(),
        )
        .map_err(Error::numerical)
    }

    // Dq*Dd/Df times the quanto drift adjustment. Taking the union of the
    // three log-linear curves' pillars preserves their carry between fixings.
    fn quanto_carry_curve(&self) -> Result<LogLinearDiscountCurve, Error> {
        let equity = self.foreign_market.forward();
        let foreign = equity.discount_curve();
        let dividend = equity.dividend_curve();
        let covariance = self.dynamics.correlation
            * self.dynamics.equity.volatility().get()
            * self.dynamics.fx.volatility().get();
        let mut times: Vec<_> = [foreign, dividend, self.domestic_curve.as_ref()]
            .into_iter()
            .flat_map(|curve| curve.times().iter().copied())
            .filter(|&time| time < self.time)
            .collect();
        times.extend([0.0, if self.time > 0.0 { self.time } else { 1.0 }]);
        times.sort_by(f64::total_cmp);
        times.dedup();
        let factors = times
            .iter()
            .map(|&time| {
                Ok(
                    (dividend.log_discount(time)? + self.domestic_curve.log_discount(time)?
                        - foreign.log_discount(time)?
                        - covariance * time)
                        .exp(),
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(LogLinearDiscountCurve::new(dividend.id(), times, factors)?)
    }
}
