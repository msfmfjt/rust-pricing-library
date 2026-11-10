use super::model_path::{EquityPath, model_times};
use super::{CrossCurrencyEquityOption, EquityFxSettlement, sampling};
use crate::core::{Date, DayCountConvention};
use crate::market::DiscountCurve;
use crate::mc::{EngineConfig, ExecutionPolicy, LocalVolTimeGrid, LsmConfig};
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{BarrierMonitoring, GraphLimitPolicy, ProductSpec};
use crate::{EquityModelSpec, Estimate, EstimatorKind};
use std::collections::BTreeMap;

impl CrossCurrencyEquityOption {
    pub(super) fn price_model(
        &self,
        model: &EquityModelSpec,
        maximum_step: f64,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        lsm: Option<LsmConfig>,
    ) -> Result<Estimate, Error> {
        let product = self
            .foreign_fixing_product
            .as_ref()
            .unwrap_or(&self.product);
        if matches!(product, ProductSpec::AmericanVanilla(_)) || lsm.is_some() {
            return Err(Error::Invalid(
                "this equity/FX path requires a payoff with contractual exercise decisions",
            ));
        }
        if matches!(product,ProductSpec::Barrier(b) if b.monitoring()==BarrierMonitoring::Continuous)
        {
            return Err(Error::Invalid(
                "the selected equity/FX path supports discrete barrier observations",
            ));
        }
        let payoff = product
            .source_graph(self.valuation_date)?
            .compile(GraphLimitPolicy::DEFAULT)?;
        let time = |date| DayCountConvention::Act365F.year_fraction(self.valuation_date, date);
        let foreign = self.foreign_market.forward();
        let payments = match product {
            ProductSpec::Graph(g) => g.payment_dates(),
            _ => std::slice::from_ref(&self.payment_date),
        };
        let weights = payments
            .iter()
            .map(|&payment| {
                if payment < self.valuation_date {
                    return Err(Error::Invalid("payment precedes valuation date"));
                }
                Ok(match self.settlement {
                    EquityFxSettlement::Quanto { fixed_fx_rate } => {
                        fixed_fx_rate.get() * self.domestic_curve.discount(time(payment))?
                    }
                    EquityFxSettlement::Composite => self.domestic_curve.discount(time(payment))?,
                    EquityFxSettlement::FloatingFx if matches!(product, ProductSpec::Graph(_)) => {
                        self.fx_spot.get() * foreign.discount_curve().discount(time(payment))?
                    }
                    EquityFxSettlement::FloatingFx => {
                        self.fx_forward.get() * self.domestic_curve.discount(time(payment))?
                    }
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut observations: BTreeMap<Date, usize> = payoff
            .terminal_observations()
            .into_iter()
            .chain(payoff.pre_dividend_observations())
            .map(|(_, date)| (date, 0))
            .collect();
        if observations.keys().any(|&d| d < self.valuation_date) {
            return Err(Error::Invalid("unresolved historical equity observation"));
        }
        let equity_spot = foreign.spot().get();
        let total = |flows: Vec<f64>| -> Result<f64, Error> {
            let result = flows.iter().zip(&weights).map(|(a, w)| a * w).sum::<f64>();
            if !result.is_finite() {
                return Err(Error::Invalid("nonfinite equity/FX present value"));
            }
            Ok(result)
        };
        let horizon = observations.keys().next_back().map_or(0., |&d| time(d));
        if horizon == 0. {
            let spot = if matches!(self.settlement, EquityFxSettlement::Composite) {
                equity_spot * self.fx_spot.get()
            } else {
                equity_spot
            };
            let v = total(
                payoff.evaluate_with_pre_dividend_spots(|_, _| Some(spot), |_, _| Some(spot))?,
            )?;
            return Estimate::new(v, 0., v, v, EstimatorKind::Analytical, 1)
                .map_err(Error::numerical);
        }
        let mut events: Vec<_> = observations.keys().map(|&d| time(d)).collect();
        events.extend(
            model_times(model)
                .iter()
                .copied()
                .filter(|&t| t > 0. && t <= horizon),
        );
        let grid = LocalVolTimeGrid::compile(events, maximum_step).map_err(Error::numerical)?;
        let times = grid.nodes();
        for (&date, index) in &mut observations {
            *index = times
                .binary_search_by(|t| t.total_cmp(&time(date)))
                .map_err(|_| Error::Invalid("equity fixing absent from simulation grid"))?;
        }
        let equity = EquityPath::compile(model, &grid, equity_spot)?;
        let steps = times.len() - 1;
        let (dimension, brownian_blocks) = equity.layout(steps);
        let carry = times
            .iter()
            .map(|&t| foreign.forward(t).map(|f| f / equity_spot))
            .collect::<Result<Vec<_>, _>>()?;
        let fx_forwards = times
            .iter()
            .map(|&t| {
                Ok(self.fx_spot.get()
                    * (foreign.discount_curve().log_discount(t)?
                        - self.domestic_curve.log_discount(t)?)
                    .exp())
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let root_dt: Vec<_> = times.windows(2).map(|w| (w[1] - w[0]).sqrt()).collect();
        let sigma = self.dynamics.fx.volatility().get();
        let rho = self.dynamics.correlation;
        let independent = ((1. - rho) * (1. + rho)).sqrt();
        let composite = matches!(self.settlement, EquityFxSettlement::Composite);
        let domestic_measure = !matches!(self.settlement, EquityFxSettlement::FloatingFx);
        let extra_fx = composite && sigma != 0. && independent != 0.;
        sampling::estimate(
            times,
            sampling::Layout {
                equity_steps: steps,
                equity_dimension: dimension,
                brownian_blocks,
                extra_blocks: usize::from(extra_fx),
            },
            engine,
            execution,
            |normals| {
                // Under the domestic measure each foreign Brownian driver has drift
                // minus its covariance with log FX. All equity kernels put the stock
                // driver first, followed by its independent orthogonal innovations.
                let mut shocks = normals[..dimension].to_vec();
                if domestic_measure {
                    for (z, dt) in shocks.iter_mut().zip(&root_dt) {
                        *z -= rho * sigma * dt;
                    }
                }
                let mut spots = equity.states(equity_spot, &shocks, None)?;
                let mut brownian = 0.;
                for (i, s) in spots.iter_mut().enumerate() {
                    *s *= carry[i];
                    if composite {
                        if i > 0 {
                            let j = i - 1;
                            let residual = if extra_fx {
                                independent * normals[dimension + j]
                            } else {
                                0.
                            };
                            brownian += root_dt[j] * (rho * normals[j] + residual);
                        }
                        *s *= fx_forwards[i]
                            * (sigma * brownian - 0.5 * sigma * sigma * times[i]).exp();
                    }
                }
                let observation = |_, d| observations.get(&d).map(|&i| spots[i]);
                total(payoff.evaluate_with_pre_dividend_spots(observation, observation)?)
            },
        )
    }
}
