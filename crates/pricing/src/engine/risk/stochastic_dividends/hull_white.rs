//! BS/Buehler/Hull--White adapter. Explicit basic AAD holds rate-model
//! parameters and correlations fixed; deterministic-rate plans are unchanged.
mod aad;
use super::StochasticDividendPrice;
use crate::core::DayCountConvention;
use crate::engine::processes::stochastic_dividends::hull_white::{
    STOCHASTIC_DIVIDEND_HW_SCHEME, StochasticDividendHullWhitePathPlan,
};
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan,
    VarianceReduction, inverse_standard_normal,
};
use crate::models::hull_white::b;
use crate::models::stochastic_dividends::{invalid, positive};
use crate::models::{BuehlerDividendModel, HullWhite1Factor, ModelSpec, StochasticDividendError};
use crate::{Fingerprint, MonteCarloError, PricingRequest, SimulationPlan};

#[derive(Clone, Debug)]
pub struct StochasticDividendHullWhitePricingPlan {
    base: SimulationPlan,
    path: StochasticDividendHullWhitePathPlan,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    fingerprint: Fingerprint,
    payment_constant: f64,
    payment_duration: f64,
    market: crate::market::EquityForward,
    rates: HullWhite1Factor,
    equity_rate_correlation: f64,
    dividend_rate_correlation: f64,
    payment_time: f64,
    risk_supported: bool,
}
impl StochasticDividendHullWhitePricingPlan {
    /// Request BS volatility applies to the discounted residual-equity factor.
    /// Cash schedule amounts remain Q means, not collateral-forward quotes.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_bs(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        rates: HullWhite1Factor,
        equity_rate_correlation: f64,
        dividend_rate_correlation: f64,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let risk = request.risk();
        if risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some() {
            return Err(StochasticDividendError::Unsupported {
                feature: "HW stochastic-dividend Greeks; submit a price-only request",
            }
            .into());
        }
        let ModelSpec::BlackScholes(bs) = request.model() else {
            return Err(StochasticDividendError::Unsupported {
                feature: "stochastic residual volatility with stochastic-dividend HW",
            }
            .into());
        };
        let expiry = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), request.product().expiry());
        let payment = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), request.product().payment_date());
        if expiry <= 0.0 || payment < expiry {
            return Err(invalid("positive_horizon_and_payment").into());
        }
        let base = SimulationPlan::compile_hybrid_base(request, policy)?;
        let market = request.market().equity().forward();
        let mut times = base.hybrid_observation_times().to_vec();
        times.push(expiry);
        times.extend(
            rates
                .volatility_times()
                .iter()
                .copied()
                .filter(|&t| t <= expiry),
        );
        if let Some(schedule) = market.discrete_dividends() {
            times.extend(
                schedule
                    .events()
                    .iter()
                    .map(|e| e.ex_time())
                    .filter(|&t| t <= expiry),
            );
        }
        let grid = LocalVolTimeGrid::compile(times, maximum_step)?;
        let path = StochasticDividendHullWhitePathPlan::compile_bs(
            market,
            model,
            bs.volatility().get(),
            &rates,
            equity_rate_correlation,
            dividend_rate_correlation,
            &grid,
        )?;
        if let EngineConfig::RandomizedQuasiMonteCarlo(config) = request.engine() {
            RqmcPlan::compile(config, path.random_dimension())?;
        }
        let mut hash = blake3::Hasher::new();
        hash.update(STOCHASTIC_DIVIDEND_HW_SCHEME.as_bytes());
        hash.update(b"Q-mean-cash-adaptive-simpson-1e-12-1e-11-depth20-rank-major-v1");
        hash.update(base.plan_fingerprint().as_bytes());
        for x in [
            model.mean_reversion(),
            model.equity_linkage(),
            model.dividend_volatility(),
            model.equity_dividend_correlation(),
            rates.mean_reversion(),
            equity_rate_correlation,
            dividend_rate_correlation,
            maximum_step,
            payment,
        ] {
            hash.update(&x.to_bits().to_le_bytes());
        }
        hash.update(&(rates.volatility_times().len() as u64).to_le_bytes());
        for &x in rates
            .volatility_times()
            .iter()
            .chain(rates.volatilities())
            .chain(path.times())
        {
            hash.update(&x.to_bits().to_le_bytes());
        }
        let payment_constant =
            rates.relative_discount(expiry, 0.0)? * rates.relative_bond(expiry, payment, 0.0)?;
        positive(payment_constant, "conditional_payment_discount")?;
        let payment_duration = b(rates.mean_reversion(), payment - expiry);
        Ok(Self {
            base,
            path,
            engine: request.engine(),
            policy,
            fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            payment_constant,
            payment_duration,
            market: market.clone(),
            rates,
            equity_rate_correlation,
            dividend_rate_correlation,
            payment_time: payment,
            risk_supported: request.product().supports_pathwise_risk()
                || request.risk().payoff_smoothing().is_some(),
        })
    }
    pub fn evaluate(&self) -> Result<StochasticDividendPrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.policy)?;
        let dimension = self.path.random_dimension();
        let (statistics, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics(count, |p| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    self.sample(z, bridge.as_ref(), config.variance_reduction().antithetic())
                })?;
                (stats, count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut prices = Vec::new();
                for scramble in 0..config.scramble_count().get() {
                    let stats = executor.try_map_reduce_statistics(count, |p| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, p, d)
                                    .map_err(|_| invalid("rqmc_uniform"))?;
                                inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        self.sample(z, bridge.as_ref(), config.variance_reduction().antithetic())
                    })?;
                    prices.push(stats.sum().total() / count as f64);
                }
                let scrambles = u64::from(config.scramble_count().get());
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    scrambles,
                    u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                )
            }
        };
        let variance = statistics
            .moments()
            .sample_variance()
            .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
        let value = statistics.sum().total() / units as f64;
        let standard_error = (variance / units as f64).sqrt();
        if !value.is_finite() || !standard_error.is_finite() {
            return Err(invalid("price_estimator").into());
        }
        Ok(StochasticDividendPrice {
            value,
            standard_error,
            independent_sampling_units: units,
            evaluated_paths: paths,
            plan_fingerprint: self.fingerprint,
            scheme: self.scheme(),
        })
    }

    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    pub fn time_nodes(&self) -> &[f64] {
        self.path.times()
    }
    pub const fn random_factor_count(&self) -> usize {
        4
    }
    pub const fn risky_spot(&self) -> f64 {
        self.path.risky_spot()
    }
    pub const fn scheme(&self) -> &'static str {
        STOCHASTIC_DIVIDEND_HW_SCHEME
    }
    pub fn cash_times(&self) -> &[f64] {
        self.path.cash_times()
    }
    pub fn initial_dividend_claim_values(&self) -> &[f64] {
        self.path.initial_dividend_claim_values()
    }
    pub fn initial_dividend_forwards(&self) -> &[f64] {
        self.path.initial_dividend_forwards()
    }
    fn bridge(&self, vr: VarianceReduction) -> Result<Option<BrownianBridgePlan>, MonteCarloError> {
        if !vr.brownian_bridge() {
            return Ok(None);
        }
        BrownianBridgePlan::compile(self.path.times().to_vec(), 1)
            .map(Some)
            .map_err(|e| MonteCarloError::LocalVol(e.into()))
    }
    fn sample(
        &self,
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
    ) -> Result<f64, MonteCarloError> {
        if let Some(bridge) = bridge {
            // Leading Sobol coordinates are all terminal normals. Apply
            // the bridge to independent factors before the model correlation.
            let count = self.random_factor_count();
            for factor in 0..count {
                let input = z
                    .iter()
                    .skip(factor)
                    .step_by(count)
                    .copied()
                    .collect::<Vec<_>>();
                let output = bridge
                    .apply_one_factor(&input)
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
                for (step, value) in output.into_iter().enumerate() {
                    z[count * step + factor] = value;
                }
            }
        }
        let mut value = 0.0;
        for &sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|v| sign * v).collect::<Vec<_>>();
            let states = self.path.evolve_path(&shocks)?;
            let spots = states
                .iter()
                .enumerate()
                .map(|(index, state)| self.path.spots(index, *state))
                .collect::<Result<Vec<_>, _>>()?;
            // The shared graph already includes P0(payment). Multiply only
            // by D(0,expiry)*P(expiry,payment)/P0(payment), conditioning on the
            // last state rather than simulating unnecessary post-expiry noise.
            let terminal = states.last().ok_or(invalid("terminal_state"))?;
            let relative = self.payment_constant
                * (-terminal.integrated_rate_factor()
                    - self.payment_duration * terminal.rate_factor())
                .exp();
            positive(relative, "relative_payment_discount")?;
            value += relative * self.base.hybrid_spot_payoff(self.path.times(), &spots)?;
        }
        Ok(value / if antithetic { 2.0 } else { 1.0 })
    }
}
