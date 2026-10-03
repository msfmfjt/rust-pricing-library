//! Continuous monitoring by a left-frozen physical log-Spot bridge.
//! This is a time-discretization approximation, not the conditional crossing
//! law of the nonlinear f/Y split or of the rough Volterra process.
use super::*;
use crate::mc::{BarrierBridgeDirection, BarrierBridgeInterval, BarrierBridgeIntervalInput};
use crate::models::BuehlerDividendState;
use crate::product::{
    BarrierDirection, BarrierMonitoring, BarrierSpec, BarrierStyle, OptionSide, ProductSpec,
};

mod bucketed_local_volatility;
pub use bucketed_local_volatility::StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk;
mod local_volatility;
mod sampling;
mod spot_bump;
pub use local_volatility::StochasticDividendContinuousBarrierLocalVolatilityRisk;
pub use spot_bump::{
    StochasticDividendContinuousBarrierGammaRisk, StochasticDividendContinuousBarrierSpotRisk,
};

const SCHEME: &str = "buehler-rough-residual-lsv-continuous-physical-log-bridge-approx-v1";

/// Opt-in price approximation for a continuously monitored rough residual-LSV
/// Barrier with stochastic cash dividends and deterministic rates.
///
/// The interval variance is the instantaneous physical log-Spot variance frozen
/// at the left post-cash node. Cash jumps are checked separately. Spot risk is
/// an explicit finite-bump Delta/Gamma estimate; discrete graph adjoints are never used.
/// Sampling errors exclude calibration uncertainty and all discretization bias.
#[derive(Clone, Debug)]
pub struct StochasticDividendContinuousBarrierPlan {
    inner: StochasticDividendPricingPlan,
    barrier: BarrierSpec,
    monitoring_end: Option<usize>,
}

impl StochasticDividendContinuousBarrierPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn compile_rough_bergomi_lsv(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        factor: RoughBergomi,
        dividend_volatility_correlation: f64,
        particles: LsvParticleConfig,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let ProductSpec::Barrier(barrier) = request.product() else {
            return Err(invalid("continuous_bridge_requires_barrier").into());
        };
        if barrier.monitoring() != BarrierMonitoring::Continuous {
            return Err(invalid("continuous_bridge_requires_continuous_monitoring").into());
        }
        let risk = request.risk();
        if risk.delta()
            || risk.gamma().is_some()
            || risk.vega()
            || risk.vega_kt().is_some()
            || risk.payoff_smoothing().is_some()
            || risk.payoff_smoothing_width_ladder().is_some()
        {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "continuous stochastic-dividend bridge is price-only and unsmoothed",
            });
        }
        let mut inner = StochasticDividendPricingPlan::compile_rough_bergomi_lsv_impl(
            request,
            model,
            factor,
            dividend_volatility_correlation,
            particles,
            maximum_step,
            policy,
            true,
        )?;
        let end = DayCountConvention::Act365F.year_fraction(
            request.valuation_date(),
            *barrier
                .monitoring_dates()
                .last()
                .expect("nonempty monitoring"),
        );
        let monitoring_end = if end < 0.0 || barrier.historical_hit() == Some(true) {
            None
        } else {
            Some(
                inner
                    .time_nodes()
                    .binary_search_by(|t| t.total_cmp(&end))
                    .map_err(|_| invalid("continuous_bridge_monitoring_end"))?,
            )
        };
        // The private inner plan is never returned and has no public risk route.
        inner.risk_supported = false;
        let mut hash = blake3::Hasher::new();
        hash.update(inner.fingerprint.as_bytes());
        hash.update(SCHEME.as_bytes());
        inner.fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        Ok(Self {
            inner,
            barrier: barrier.clone(),
            monitoring_end,
        })
    }

    pub fn evaluate(&self) -> Result<StochasticDividendPrice, MonteCarloError> {
        let mut price = self
            .inner
            .evaluate_with_path_payoff(|shocks| self.path_payoff(shocks))?;
        price.scheme = SCHEME;
        Ok(price)
    }

    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.inner.plan_fingerprint()
    }
    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        self.inner.time_nodes()
    }
    #[must_use]
    pub const fn random_factor_count(&self) -> usize {
        self.inner.random_factor_count()
    }
    #[must_use]
    pub const fn risky_spot(&self) -> f64 {
        self.inner.risky_spot()
    }
    #[must_use]
    pub const fn scheme(&self) -> &'static str {
        SCHEME
    }
    #[must_use]
    pub fn lsv_time_nodes(&self) -> &[f64] {
        self.inner.lsv_time_nodes().expect("compiled LSV")
    }
    #[must_use]
    pub fn lsv_log_moneyness_nodes(&self) -> &[f64] {
        self.inner.lsv_log_moneyness_nodes().expect("compiled LSV")
    }
    #[must_use]
    pub fn lsv_squared_leverage(&self) -> &[f64] {
        self.inner.lsv_squared_leverage().expect("compiled LSV")
    }
    #[must_use]
    pub fn lsv_initial_residual_equity(&self) -> f64 {
        self.inner
            .lsv_initial_residual_equity()
            .expect("compiled LSV")
    }

    fn touched(&self, spot: f64) -> bool {
        match self.barrier.direction() {
            BarrierDirection::Up => spot >= self.barrier.barrier().get(),
            BarrierDirection::Down => spot <= self.barrier.barrier().get(),
        }
    }

    fn log_survival(
        &self,
        path: &StochasticDividendPathPlan,
        states: &[BuehlerDividendState],
        spots: &[(f64, Option<f64>)],
        volatilities: &[f64],
    ) -> Result<f64, MonteCarloError> {
        if self.barrier.historical_hit() == Some(true) {
            return Ok(f64::NEG_INFINITY);
        }
        let Some(end) = self.monitoring_end else {
            return Ok(0.0);
        };
        if self.touched(spots[0].0) {
            return Ok(f64::NEG_INFINITY);
        }
        let direction = match self.barrier.direction() {
            BarrierDirection::Up => BarrierBridgeDirection::Up,
            BarrierDirection::Down => BarrierBridgeDirection::Down,
        };
        let model = path.model();
        let mut log_survival = 0.0;
        for i in 0..end {
            let right = spots[i + 1].1.unwrap_or(spots[i + 1].0);
            // Diffusion arrives at pre-cash Spot. Both sides of the jump belong
            // to continuous monitoring, including a jump at its final endpoint.
            if self.touched(right) || self.touched(spots[i + 1].0) {
                return Ok(f64::NEG_INFINITY);
            }
            let [a, b, _] = path.nodes()[i].coefficients();
            let equity = a * states[i].equity() / spots[i].0 * volatilities[i];
            let dividend = b * states[i].dividend() / spots[i].0 * model.dividend_volatility();
            let variance =
                physical_log_variance(equity, dividend, model.equity_dividend_correlation());
            let interval = BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
                direction,
                left_state: spots[i].0,
                right_state: right,
                left_barrier: self.barrier.barrier().get(),
                right_barrier: self.barrier.barrier().get(),
                left_local_variance: variance,
                right_local_variance: variance,
                dt: path.times()[i + 1] - path.times()[i],
            })?;
            log_survival += interval.log_survival();
        }
        Ok(log_survival)
    }

    fn path_payoff(&self, shocks: &[f64]) -> Result<f64, MonteCarloError> {
        let (states, volatilities) = self
            .inner
            .path
            .evolve_rough_path_with_volatilities(shocks)?;
        self.payoff_from_trace(&self.inner.path, &states, &volatilities)
    }

    fn payoff_from_trace(
        &self,
        path: &StochasticDividendPathPlan,
        states: &[BuehlerDividendState],
        volatilities: &[f64],
    ) -> Result<f64, MonteCarloError> {
        let spots = path
            .nodes()
            .iter()
            .zip(states)
            .map(|(node, state)| node.spots(*state))
            .collect::<Result<Vec<_>, _>>()?;
        let log_survival = self.log_survival(path, states, &spots, volatilities)?;
        let survival = log_survival.exp();
        // Keep small knock-in probabilities when exp(log_survival) rounds to 1.
        let hit = -log_survival.exp_m1();
        let (active, inactive) = match self.barrier.style() {
            BarrierStyle::KnockOut => (survival, hit),
            BarrierStyle::KnockIn => (hit, survival),
        };
        let rebate = self.barrier.rebate().map_or(0.0, |v| v.get());
        let value = if active == 0.0 {
            rebate
        } else {
            let terminal = spots.last().expect("nonempty path").0;
            let signed = match self.barrier.side() {
                OptionSide::Call => terminal - self.barrier.strike().get(),
                OptionSide::Put => self.barrier.strike().get() - terminal,
            };
            let vanilla = self.barrier.notional().get() * signed.max(0.0);
            active * vanilla + inactive * rebate
        };
        Ok(self.inner.base.discount() * value)
    }
}

// PSD form avoids a slightly negative result at a singular correlation boundary.
fn physical_log_variance(equity: f64, dividend: f64, rho: f64) -> f64 {
    (equity + rho * dividend).powi(2) + (1.0 - rho) * (1.0 + rho) * dividend.powi(2)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod spot_bump_tests;

#[cfg(test)]
mod local_volatility_tests;

#[cfg(test)]
mod bucketed_local_volatility_tests;
