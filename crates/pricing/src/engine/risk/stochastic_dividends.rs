//! Deterministic-rate Buehler cash-dividend pricing. No calibration or reverse
//! capability is implied by this separate, price-only entry point.

use crate::core::DayCountConvention;
use crate::engine::processes::stochastic_dividends::StochasticDividendPathPlan;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan,
    VarianceReduction, inverse_standard_normal,
};
use crate::models::stochastic_dividends::invalid;
use crate::models::{
    Bergomi1Factor, Bergomi2Factor, BuehlerDividendModel, ModelSpec, STOCHASTIC_DIVIDEND_SCHEME,
    StochasticDividendError,
};
use crate::{Fingerprint, MonteCarloError, PricingRequest, SimulationPlan};

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendPrice {
    pub value: f64,
    pub standard_error: f64,
    pub independent_sampling_units: u64,
    pub evaluated_paths: u128,
    pub plan_fingerprint: Fingerprint,
    pub scheme: &'static str,
}

/// Constant volatility of the normalized residual-equity martingale, with a
/// stochastic cash reserve. It is not Black-Scholes volatility of physical S.
#[derive(Clone, Debug)]
pub struct StochasticDividendPricingPlan {
    base: SimulationPlan,
    path: StochasticDividendPathPlan,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    fingerprint: Fingerprint,
}

impl StochasticDividendPricingPlan {
    pub fn compile_bs(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let risk = request.risk();
        if risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some() {
            return Err(StochasticDividendError::Unsupported {
                feature: "Greeks; submit a price-only request",
            }
            .into());
        }
        let ModelSpec::BlackScholes(bs) = request.model() else {
            return Err(StochasticDividendError::Unsupported {
                feature: "nonconstant residual-equity volatility",
            }
            .into());
        };
        let expiry = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), request.product().expiry());
        if expiry <= 0.0 {
            return Err(invalid("positive_horizon").into());
        }
        // This also rejects American exercise and continuously monitored barriers.
        let base = SimulationPlan::compile_hybrid_base(request, policy)?;
        let market = request.market().equity().forward();
        let mut times = base.hybrid_observation_times().to_vec();
        times.push(expiry);
        if let Some(schedule) = market.discrete_dividends() {
            times.extend(
                schedule
                    .events()
                    .iter()
                    .map(|e| e.ex_time())
                    .filter(|t| *t <= expiry),
            );
        }
        let grid = LocalVolTimeGrid::compile(times, maximum_step)?;
        let path =
            StochasticDividendPathPlan::compile(market, model, bs.volatility().get(), &grid)?;
        if let EngineConfig::RandomizedQuasiMonteCarlo(config) = request.engine() {
            RqmcPlan::compile(config, path.random_dimension())?;
        }
        let mut hash = blake3::Hasher::new();
        hash.update(STOCHASTIC_DIVIDEND_SCHEME.as_bytes());
        hash.update(b"rank-major-before-correlation-price-only");
        hash.update(base.plan_fingerprint().as_bytes());
        for x in [
            model.mean_reversion(),
            model.equity_linkage(),
            model.dividend_volatility(),
            model.equity_dividend_correlation(),
            maximum_step,
        ] {
            hash.update(&x.to_bits().to_le_bytes());
        }
        for &time in path.times() {
            hash.update(&time.to_bits().to_le_bytes());
        }
        let fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        Ok(Self {
            base,
            path,
            engine: request.engine(),
            policy,
            fingerprint,
        })
    }

    /// Request BS volatility is sigma0, not physical stock implied volatility.
    pub fn compile_bergomi(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        factor: Bergomi1Factor,
        dividend_volatility_correlation: f64,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let mut plan = Self::compile_bs(request, model, maximum_step, policy)?;
        plan.path = plan
            .path
            .with_bergomi(factor, dividend_volatility_correlation)?;
        plan.finish_bergomi()?;
        Ok(plan)
    }
    pub fn compile_bergomi_two_factor(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        factor: Bergomi2Factor,
        dividend_volatility_correlations: [f64; 2],
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let mut plan = Self::compile_bs(request, model, maximum_step, policy)?;
        plan.path = plan
            .path
            .with_bergomi_two_factor(factor, dividend_volatility_correlations)?;
        plan.finish_bergomi()?;
        Ok(plan)
    }
    fn finish_bergomi(&mut self) -> Result<(), MonteCarloError> {
        if let EngineConfig::RandomizedQuasiMonteCarlo(config) = self.engine {
            RqmcPlan::compile(config, self.path.random_dimension())?;
        }
        let mut hash = blake3::Hasher::new();
        hash.update(self.fingerprint.as_bytes());
        self.path.hash_volatility(&mut hash);
        self.fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        Ok(())
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

    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        self.path.times()
    }
    #[must_use]
    pub const fn random_factor_count(&self) -> usize {
        self.path.random_factor_count()
    }
    #[must_use]
    pub const fn risky_spot(&self) -> f64 {
        self.path.risky_spot()
    }
    #[must_use]
    pub const fn scheme(&self) -> &'static str {
        self.path.scheme()
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
            let spots = self
                .path
                .nodes()
                .iter()
                .zip(&states)
                .map(|(node, state)| node.spots(*state))
                .collect::<Result<Vec<_>, _>>()?;
            // The shared payoff already includes the deterministic collateral
            // discount to contractual payment. Do not discount a second time.
            value += self.base.hybrid_spot_payoff(self.path.times(), &spots)?;
        }
        Ok(value / if antithetic { 2.0 } else { 1.0 })
    }
}
