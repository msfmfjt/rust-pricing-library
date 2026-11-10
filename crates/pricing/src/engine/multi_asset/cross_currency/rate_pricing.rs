use super::joint_path::{JointPath, JointPlan};
use super::{CrossCurrencyEquityOption, CrossCurrencyHullWhite, EquityFxSettlement, sampling};
use crate::core::{Date, DayCountConvention};
use crate::mc::{EngineConfig, ExecutionPolicy, LsmConfig};
use crate::models::ModelSpec;
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{BarrierMonitoring, GraphLimitPolicy, ProductSpec};
use crate::{EquityModelSpec, Estimate};
use std::collections::BTreeMap;

impl CrossCurrencyEquityOption {
    pub(super) fn price_with_rates(
        &self,
        rates: &CrossCurrencyHullWhite,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<LsmConfig>,
    ) -> Result<Estimate, Error> {
        let product = &self.product;
        if matches!(product, ProductSpec::AmericanVanilla(_))
            || lsm.is_some()
            || matches!(product,ProductSpec::Barrier(b) if b.monitoring()==BarrierMonitoring::Continuous)
        {
            return Err(Error::Invalid(
                "joint Hull-White equity/FX requires dated payoff observations",
            ));
        }
        let base = EquityModelSpec::Standard(ModelSpec::BlackScholes(self.dynamics.equity));
        let model = self.equity_model.as_ref().unwrap_or(&base);
        let time = |d| DayCountConvention::Act365F.year_fraction(self.valuation_date, d);
        let payoff = product
            .source_graph(self.valuation_date)?
            .compile(GraphLimitPolicy::DEFAULT)?;
        let spot_dates = payoff
            .terminal_observations()
            .into_iter()
            .chain(payoff.pre_dividend_observations());
        let mut dates: BTreeMap<Date, usize> = spot_dates.map(|(_, d)| (d, 0)).collect();
        let last_equity = dates
            .keys()
            .next_back()
            .copied()
            .unwrap_or(self.valuation_date);
        let payments = match product {
            ProductSpec::Graph(g) => g.payment_dates(),
            _ => std::slice::from_ref(&self.payment_date),
        };
        let fx_dates: Vec<_> = payments
            .iter()
            .map(|&p| match product {
                ProductSpec::Graph(_) => self.payment_fx_fixings.get(&p).copied().unwrap_or(p),
                _ => product.expiry(),
            })
            .collect();
        for &d in payments.iter().chain(&fx_dates) {
            dates.insert(d, 0);
        }
        if dates.keys().any(|&d| d < self.valuation_date) {
            return Err(Error::Invalid(
                "historical equity/FX observation or payment requires fixing data",
            ));
        }
        let end = time(*dates.keys().next_back().unwrap());
        if end == 0. {
            return self.price_model(model, self.maximum_step, engine, execution, None);
        }
        let plan = JointPlan::new(self, model, rates, dates, last_equity, Vec::new())?;
        let dates = &plan.dates;
        let cashflows: Vec<_> = payments
            .iter()
            .zip(fx_dates)
            .map(|(p, f)| (dates[p], dates[&f]))
            .collect();
        sampling::estimate(&plan.times, plan.layout, engine, execution, |normals| {
            let JointPath {
                spots,
                fx,
                discounts,
                ..
            } = plan.simulate(normals, false)?;
            let observed = |d, pre| {
                dates.get(&d).map(|&i| {
                    let (post, before) = spots[i];
                    let s = if pre { before.unwrap_or(post) } else { post };
                    if matches!(self.settlement, EquityFxSettlement::Composite) {
                        s * fx[i]
                    } else {
                        s
                    }
                })
            };
            let flows = payoff.evaluate_with_pre_dividend_spots(
                |_, d| observed(d, false),
                |_, d| observed(d, true),
            )?;
            Ok(flows
                .iter()
                .zip(&cashflows)
                .map(|(a, &(p, f))| {
                    let conversion = match self.settlement {
                        EquityFxSettlement::Quanto { fixed_fx_rate } => fixed_fx_rate.get(),
                        EquityFxSettlement::FloatingFx => fx[f],
                        EquityFxSettlement::Composite => 1.,
                    };
                    a * discounts[p] * conversion
                })
                .sum())
        })
    }
}
