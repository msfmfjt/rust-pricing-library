//! Conditional interval-survival weighting for the actual monitored stock or stock times FX.
use super::exercise::intrinsic;
use super::joint_path::{JointPath, JointPlan, PathUse};
use super::{CrossCurrencyEquityOption, EquityFxSettlement, sampling};
use crate::core::DayCountConvention;
use crate::engine::processes::hull_white::covariance_loading;
use crate::mc::{
    BarrierBridgeDirection, BarrierBridgeInterval, BarrierBridgeIntervalInput, EngineConfig,
    ExecutionPolicy,
};
use crate::models::RoughVolatilityModel;
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{BarrierDirection, BarrierSpec, BarrierStyle};
use crate::{EquityModelSpec, Estimate, EstimatorKind};

struct Monitor<'a> {
    barrier: &'a BarrierSpec,
    composite: bool,
    arithmetic: bool,
    fx_vol: f64,
    // Stock, FX, and foreign rate Brownian drivers. The dividend residual is orthogonal to all three.
    loading: [[f64; 3]; 3],
    end: Option<usize>,
}

impl Monitor<'_> {
    fn touched(&self, level: f64) -> bool {
        match self.barrier.direction() {
            BarrierDirection::Up => level >= self.barrier.barrier().get(),
            BarrierDirection::Down => level <= self.barrier.barrier().get(),
        }
    }

    fn log_survival(&self, path: &JointPath, times: &[f64]) -> Result<f64, Error> {
        if self.barrier.historical_hit() == Some(true) {
            return Ok(f64::NEG_INFINITY);
        }
        let Some(end) = self.end else {
            return Ok(0.);
        };
        let x = |i: usize| if self.composite { path.fx[i] } else { 1. };
        if self.touched(path.spots[0].0 * x(0)) {
            return Ok(f64::NEG_INFINITY);
        }
        let level = self.barrier.barrier().get();
        let mut log_survival = 0.;
        for i in 0..end {
            let left = path.spots[i].0 * x(i);
            let (post, pre) = path.spots[i + 1];
            let right = pre.unwrap_or(post) * x(i + 1);
            if self.touched(right) || self.touched(post * x(i + 1)) {
                return Ok(f64::NEG_INFINITY);
            }
            let [stock, dividend, rate] = path.loadings[i];
            let weights = [stock * x(i), left * self.fx_vol, rate * x(i)];
            let variance = (0..3)
                .map(|j| {
                    (0..3)
                        .map(|k| self.loading[k][j] * weights[k])
                        .sum::<f64>()
                        .powi(2)
                })
                .sum::<f64>()
                + (dividend * x(i)).powi(2);
            let dt = times[i + 1] - times[i];
            if self.arithmetic {
                if !variance.is_finite() {
                    return Err(Error::Invalid("nonfinite physical barrier variance"));
                }
                if variance > 0. {
                    let log_hit = -2. * (left - level) * (right - level) / (variance * dt);
                    log_survival += (-log_hit.exp_m1()).ln();
                }
            } else {
                let variance = variance / left.powi(2);
                let direction = match self.barrier.direction() {
                    BarrierDirection::Up => BarrierBridgeDirection::Up,
                    BarrierDirection::Down => BarrierBridgeDirection::Down,
                };
                log_survival += BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
                    direction,
                    left_state: left,
                    right_state: right,
                    left_barrier: level,
                    right_barrier: level,
                    left_local_variance: variance,
                    right_local_variance: variance,
                    dt,
                })
                .map_err(Error::numerical)?
                .log_survival();
            }
        }
        Ok(log_survival)
    }
}

fn cash(barrier: &BarrierSpec, spot: f64, log_survival: f64) -> f64 {
    let survival = log_survival.exp();
    let hit = -log_survival.exp_m1();
    let (active, inactive) = match barrier.style() {
        BarrierStyle::KnockOut => (survival, hit),
        BarrierStyle::KnockIn => (hit, survival),
    };
    active * barrier.notional().get() * intrinsic(barrier.side(), spot, barrier.strike().get())
        + inactive * barrier.rebate().map_or(0., |v| v.get())
}

impl CrossCurrencyEquityOption {
    pub(super) fn price_joint_barrier(
        &self,
        barrier: &BarrierSpec,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        let historic = barrier.monitoring_dates()[0] < self.valuation_date;
        if historic != barrier.historical_hit().is_some() {
            return Err(Error::Invalid(
                "past monitoring requires a published barrier hit state",
            ));
        }
        let end = *barrier.monitoring_dates().last().unwrap();
        let composite = matches!(self.settlement, EquityFxSettlement::Composite);
        let arithmetic = matches!(&self.equity_model,Some(EquityModelSpec::RoughVolatility(RoughVolatilityModel::RoughSabr(m))) if m.beta!=1.);
        let foreign = self.foreign_market.forward();
        let time = |d| DayCountConvention::Act365F.year_fraction(self.valuation_date, d);
        let rho = self.dynamics.correlation;
        let (sr, xr) = self
            .rates
            .as_ref()
            .map_or((0., 0.), |r| (r.correlations[0][3], r.correlations[1][3]));
        let mut monitor = Monitor {
            barrier,
            composite,
            arithmetic,
            fx_vol: if composite {
                self.dynamics.fx.volatility().get()
            } else {
                0.
            },
            loading: covariance_loading([[1., rho, sr], [rho, 1., xr], [sr, xr, 1.]])
                .map_err(Error::numerical)?,
            end: None,
        };
        let payoff = |s, fx, df, log_survival| {
            let conversion = match self.settlement {
                EquityFxSettlement::Quanto { fixed_fx_rate } => fixed_fx_rate.get(),
                EquityFxSettlement::FloatingFx => fx,
                EquityFxSettlement::Composite => 1.,
            };
            df * conversion * cash(barrier, if composite { s * fx } else { s }, log_survival)
        };
        if self.payment_date == self.valuation_date {
            let s = foreign.spot().get();
            let fx = self.fx_spot.get();
            let hit = barrier.historical_hit() == Some(true)
                || (end == self.valuation_date
                    && monitor.touched(if composite { s * fx } else { s }));
            let v = payoff(s, fx, 1., if hit { f64::NEG_INFINITY } else { 0. });
            return Estimate::new(v, 0., v, v, EstimatorKind::Analytical, 1)
                .map_err(Error::numerical);
        }
        let events = [
            foreign.discount_curve(),
            foreign.dividend_curve(),
            self.domestic_curve.as_ref(),
        ]
        .into_iter()
        .flat_map(|c| c.times().iter().copied())
        .filter(|&t| t > 0. && t <= time(end))
        .collect();
        let dates = barrier
            .monitoring_dates()
            .iter()
            .copied()
            .filter(|&d| d >= self.valuation_date)
            .chain([barrier.expiry(), self.payment_date]);
        let plan = JointPlan::for_option(self, dates, barrier.expiry(), events)?;
        monitor.end = plan.dates.get(&end).copied();
        let expiry = plan.dates[&barrier.expiry()];
        let pay = plan.dates[&self.payment_date];
        sampling::estimate(&plan.times, plan.layout, engine, execution, |normals| {
            let mode =
                if monitor.end.is_some_and(|i| i > 0) && barrier.historical_hit() != Some(true) {
                    PathUse::ContinuousBarrier
                } else {
                    PathUse::Cashflows
                };
            let path = plan.simulate(normals, mode)?;
            Ok(payoff(
                path.spots[expiry].0,
                path.fx[expiry],
                path.discounts[pay],
                monitor.log_survival(&path, &plan.times)?,
            ))
        })
    }
}
