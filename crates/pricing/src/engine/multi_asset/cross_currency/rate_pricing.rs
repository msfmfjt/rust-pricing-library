use super::model_path::{EquityPath, model_times};
use super::rates::RateState;
use super::{CrossCurrencyEquityOption, CrossCurrencyHullWhite, EquityFxSettlement, sampling};
use crate::core::{Date, DayCountConvention};
use crate::market::DiscountCurve;
use crate::mc::{EngineConfig, ExecutionPolicy, LocalVolTimeGrid, LsmConfig};
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
        let equity_end = time(last_equity);
        let mut events: Vec<_> = dates.keys().map(|&d| time(d)).collect();
        events.extend(
            model_times(model)
                .iter()
                .copied()
                .filter(|&t| t > 0. && t <= equity_end),
        );
        events.extend(
            rates
                .domestic
                .volatility_times()
                .iter()
                .chain(rates.foreign.volatility_times())
                .copied()
                .filter(|&t| t <= end),
        );
        let grid =
            LocalVolTimeGrid::compile(events, self.maximum_step).map_err(Error::numerical)?;
        let times = grid.nodes();
        for (&d, index) in &mut dates {
            *index = times
                .binary_search_by(|t| t.total_cmp(&time(d)))
                .map_err(|_| Error::Invalid("missing rate or fixing date"))?;
        }
        let equity_steps = dates.get(&last_equity).copied().unwrap_or(0);
        let foreign = self.foreign_market.forward();
        let spot = foreign.spot().get();
        let path = if equity_steps == 0 {
            None
        } else {
            let prefix = LocalVolTimeGrid::compile(times[..=equity_steps].to_vec(), f64::MAX)
                .map_err(Error::numerical)?;
            Some(EquityPath::compile_for_rates(
                model, &prefix, foreign, rates,
            )?)
        };
        let (dim, blocks) = path.as_ref().map_or((0, 0), |p| p.layout(equity_steps));
        let sigma = self.dynamics.fx.volatility().get();
        let rho = self.dynamics.correlation;
        let rate_steps = rates.steps(times, sigma)?;
        let n = rate_steps.len();
        let mut dd = Vec::new();
        let mut fx_forward = Vec::new();
        let mut stock_carry = Vec::new();
        for &t in times {
            dd.push(self.domestic_curve.discount(t)?);
            let df = foreign.discount_curve().discount(t)?;
            fx_forward.push(self.fx_spot.get() * df / dd.last().unwrap());
            stock_carry.push(foreign.dividend_curve().discount(t)? / df);
        }
        let cashflows: Vec<_> = payments
            .iter()
            .zip(fx_dates)
            .map(|(p, f)| (dates[p], dates[&f]))
            .collect();
        // Five extra normal blocks condition FX and both rates on the equity
        // Brownian. After the last equity observation a sixth supplies its tail.
        let tail = equity_steps < n;
        let layout = sampling::Layout {
            equity_steps,
            equity_dimension: dim,
            brownian_blocks: blocks,
            extra_blocks: 5 + usize::from(tail),
        };
        sampling::estimate(times, layout, engine, execution, |normals| {
            let extra = &normals[dim..];
            let mut shocks = normals[..dim].to_vec();
            for (z, interval) in shocks.iter_mut().zip(times[..=equity_steps].windows(2)) {
                *z -= rho * sigma * (interval[1] - interval[0]).sqrt();
            }
            let mut rate_carry = Vec::with_capacity(equity_steps + 1);
            let mut state = RateState::default();
            let mut fx = Vec::with_capacity(n + 1);
            let mut discounts = Vec::with_capacity(n + 1);
            for i in 0..=n {
                if i > 0 {
                    let j = i - 1;
                    let stock_normal = if j < equity_steps {
                        normals[j]
                    } else {
                        extra[5 * n + j]
                    };
                    rate_steps[j].advance(
                        &mut state,
                        [
                            stock_normal,
                            extra[j],
                            extra[n + j],
                            extra[2 * n + j],
                            extra[3 * n + j],
                            extra[4 * n + j],
                        ],
                    );
                }
                let [domestic, foreign] = state.integrated;
                let x = fx_forward[i]
                    * (domestic - foreign - 0.5 * sigma * sigma * times[i]
                        + sigma * state.fx_brownian)
                        .exp();
                let discount = dd[i] * (-domestic).exp();
                if !x.is_finite() || x <= 0. || !discount.is_finite() || discount <= 0. {
                    return Err(Error::Invalid(
                        "nonfinite or nonpositive simulated FX or discount",
                    ));
                }
                if i <= equity_steps {
                    rate_carry.push(foreign);
                }
                fx.push(x);
                discounts.push(discount);
            }
            let mut spots = match &path {
                Some(p) => p.states(spot, &shocks, Some(&rate_carry))?,
                None => vec![spot],
            };
            for (i, s) in spots.iter_mut().enumerate() {
                *s *= stock_carry[i] * rate_carry[i].exp();
                if matches!(self.settlement, EquityFxSettlement::Composite) {
                    *s *= fx[i];
                }
            }
            let observation = |_, d| dates.get(&d).map(|&i| spots[i]);
            let flows = payoff.evaluate_with_pre_dividend_spots(observation, observation)?;
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
