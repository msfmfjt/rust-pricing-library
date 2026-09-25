//! Deterministic-rate Buehler cash-dividend pricing. No calibration or reverse
//! capability is implied by compilation. First-order risk is requested explicitly.

mod aad;
mod gamma;
pub(crate) mod hull_white;
mod lsv;
mod lsv_parameters;
pub use aad::StochasticDividendAadRisk;
pub use gamma::StochasticDividendGammaRisk;
pub use lsv::{StochasticDividendLocalVarianceRisk, StochasticDividendLsvSpotRisk};
pub use lsv_parameters::StochasticDividendLsvBergomiRisk;

use crate::core::DayCountConvention;
use crate::engine::processes::stochastic_dividends::StochasticDividendPathPlan;
use crate::market::LocalVarianceGrid;
use crate::mc::lsv::{CalibratedBergomiLsv, LsvParticleConfig, calibrate_bergomi_lsv_parallel};
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan,
    VarianceReduction, inverse_standard_normal,
};
use crate::models::stochastic_dividends::invalid;
use crate::models::{
    Bergomi1Factor, Bergomi2Factor, BuehlerDividendModel, ModelSpec, RoughBergomi,
    STOCHASTIC_DIVIDEND_SCHEME, StochasticDividendError,
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

impl StochasticDividendPrice {
    /// Monte Carlo uncertainty only. LSV prices condition on the finite
    /// particle calibration; calibration sampling/model uncertainty is excluded.
    #[must_use]
    pub fn uncertainty_scope(&self) -> &'static str {
        if self.scheme.contains("residual-lsv") {
            "pricing_conditional_on_calibration"
        } else {
            "pricing_only"
        }
    }
}

#[derive(Clone, Debug)]
enum StochasticDividendLsvCalibration {
    One {
        calibration: CalibratedBergomiLsv<Bergomi1Factor>,
        original_target: LocalVarianceGrid,
        dividend_volatility_correlation: f64,
    },
    Two {
        calibration: CalibratedBergomiLsv<Bergomi2Factor>,
        original_target: LocalVarianceGrid,
    },
}

/// Constant-volatility or pure Bergomi residual equity with a stochastic cash
/// reserve. The request volatility is the initial residual-equity volatility,
/// not physical-stock implied volatility. `evaluate_aad` requests first-order
/// risk explicitly; constructors continue to accept price-only requests.
/// Rough plans also support basic AAD, H/eta AAD and finite-bump Spot Gamma;
/// rough correlation AAD remains unsupported.
#[derive(Clone, Debug)]
pub struct StochasticDividendPricingPlan {
    base: SimulationPlan,
    path: StochasticDividendPathPlan,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    fingerprint: Fingerprint,
    market: crate::market::EquityForward,
    payment_time: f64,
    risk_supported: bool,
    lsv: Option<StochasticDividendLsvCalibration>,
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
            market: market.clone(),
            payment_time: DayCountConvention::Act365F
                .year_fraction(request.valuation_date(), request.product().payment_date()),
            risk_supported: request.product().supports_pathwise_risk()
                || request.risk().payoff_smoothing().is_some(),
            lsv: None,
        })
    }

    /// Build the stochastic-dividend execution grid and a residual-equity
    /// Local-variance calibration target. The target is interpreted in the funded
    /// residual coordinate F_res, not physical stock S=a*f+b*Y+c.
    fn compile_lsv_base(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<(Self, LocalVarianceGrid, LocalVarianceGrid, LocalVolTimeGrid), MonteCarloError>
    {
        let risk = request.risk();
        if risk.delta() || risk.gamma().is_some() || risk.vega() {
            return Err(StochasticDividendError::Unsupported {
                feature: "Delta, Gamma and scalar Vega for residual-equity LSV; use the dedicated recalibration-aware risk methods",
            }
            .into());
        }
        let ModelSpec::LocalVolatility(target) = request.model() else {
            return Err(StochasticDividendError::Unsupported {
                feature: "residual-equity LSV requires a LocalVolatility calibration target",
            }
            .into());
        };
        let expiry = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), request.product().expiry());
        if expiry <= 0.0 {
            return Err(invalid("positive_horizon").into());
        }
        let base = SimulationPlan::compile_hybrid_base(request, policy)?;
        let market = request.market().equity().forward();

        // Start from the established LSV grid so every contractual and target
        // knot is retained, then optionally refine further for the Buehler split.
        let lsv_grid = base.lsv_time_grid()?;
        // The shared LocalVol runtime keeps all deterministic dividend events,
        // including cash after option expiry. Buehler funding also keeps those
        // cash means, but the stochastic path itself must stop at expiry.
        let mut required_times = lsv_grid
            .nodes()
            .iter()
            .copied()
            .filter(|t| *t <= expiry)
            .collect::<Vec<_>>();
        required_times.extend(
            target
                .local_variance_grid()
                .time_nodes()
                .iter()
                .copied()
                .filter(|t| *t <= expiry),
        );
        if let Some(schedule) = market.discrete_dividends() {
            required_times.extend(
                schedule
                    .events()
                    .iter()
                    .map(|e| e.ex_time())
                    .filter(|t| *t <= expiry),
            );
        }
        required_times.push(expiry);
        let grid = LocalVolTimeGrid::compile(required_times, maximum_step)?;

        let original = target.local_variance_grid().clone();
        let x = original.log_moneyness_nodes();
        let mut values = Vec::with_capacity(grid.nodes().len() * x.len());
        for &t in grid.nodes() {
            for &k in x {
                values.push(original.interpolate(t, k)?.value);
            }
        }
        let refined = LocalVarianceGrid::new(
            grid.nodes().to_vec(),
            x.to_vec(),
            values,
            original.floor(),
            original.cap(),
        )?;

        // sigma0 is unused after the calibrated leverage is attached. Zero keeps
        // this temporary path explicit and avoids assigning physical-stock IV
        // meaning to the request's LocalVolatility model.
        let path = StochasticDividendPathPlan::compile(market, model, 0.0, &grid)?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"buehler-residual-lsv-base-v1");
        hash.update(base.plan_fingerprint().as_bytes());
        for v in [
            model.mean_reversion(),
            model.equity_linkage(),
            model.dividend_volatility(),
            model.equity_dividend_correlation(),
            maximum_step,
        ] {
            hash.update(&v.to_bits().to_le_bytes());
        }
        for &time in path.times() {
            hash.update(&time.to_bits().to_le_bytes());
        }
        let fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        Ok((
            Self {
                base,
                path,
                engine: request.engine(),
                policy,
                fingerprint,
                market: market.clone(),
                payment_time: DayCountConvention::Act365F
                    .year_fraction(request.valuation_date(), request.product().payment_date()),
                risk_supported: request.product().supports_pathwise_risk()
                    || request.risk().payoff_smoothing().is_some(),
                lsv: None,
            },
            original,
            refined,
            grid,
        ))
    }

    /// Particle-calibrated 1F Bergomi LSV for funded residual equity, coupled
    /// to Buehler stochastic cash dividends. The LocalVolatility request is a
    /// target for F_res, not for reconstructed physical stock.
    pub fn compile_bergomi_lsv(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        factor: Bergomi1Factor,
        dividend_volatility_correlation: f64,
        particles: LsvParticleConfig,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let (mut plan, original_target, target, grid) =
            Self::compile_lsv_base(request, model, maximum_step, policy)?;
        let calibration = calibrate_bergomi_lsv_parallel(
            &target,
            factor,
            plan.path.risky_spot(),
            particles.clone(),
            &DeterministicExecutor::new(policy)?,
        )?;
        let leverage = calibration.surface().clone();
        plan.path =
            plan.path
                .with_bergomi_lsv(factor, dividend_volatility_correlation, leverage)?;
        plan.lsv = Some(StochasticDividendLsvCalibration::One {
            calibration,
            original_target,
            dividend_volatility_correlation,
        });
        plan.finish_lsv(&particles)?;
        debug_assert_eq!(plan.path.times(), grid.nodes());
        Ok(plan)
    }

    /// Two-factor counterpart of compile_bergomi_lsv.
    pub fn compile_bergomi_two_factor_lsv(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        factor: Bergomi2Factor,
        dividend_volatility_correlations: [f64; 2],
        particles: LsvParticleConfig,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let (mut plan, original_target, target, grid) =
            Self::compile_lsv_base(request, model, maximum_step, policy)?;
        let calibration = calibrate_bergomi_lsv_parallel(
            &target,
            factor,
            plan.path.risky_spot(),
            particles.clone(),
            &DeterministicExecutor::new(policy)?,
        )?;
        let leverage = calibration.surface().clone();
        plan.path = plan.path.with_bergomi_two_factor_lsv(
            factor,
            dividend_volatility_correlations,
            leverage,
        )?;
        plan.lsv = Some(StochasticDividendLsvCalibration::Two {
            calibration,
            original_target,
        });
        plan.finish_lsv(&particles)?;
        debug_assert_eq!(plan.path.times(), grid.nodes());
        Ok(plan)
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
    /// Rough vol-of-vol is eta in log variance, not Bergomi's log-volatility nu.
    /// The request BS volatility supplies sigma0; no market-IV calibration occurs.
    pub fn compile_rough_bergomi(
        request: &PricingRequest,
        model: BuehlerDividendModel,
        factor: RoughBergomi,
        dividend_volatility_correlation: f64,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let mut plan = Self::compile_bs(request, model, maximum_step, policy)?;
        plan.path = plan
            .path
            .with_rough_bergomi(factor, dividend_volatility_correlation)?;
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

    fn finish_lsv(&mut self, particles: &LsvParticleConfig) -> Result<(), MonteCarloError> {
        self.finish_bergomi()?;
        let mut hash = blake3::Hasher::new();
        hash.update(self.fingerprint.as_bytes());
        hash.update(b"particle-calibration-v1");
        hash.update(&(particles.particle_count() as u64).to_le_bytes());
        hash.update(&particles.seed().to_le_bytes());
        hash.update(&particles.log_bandwidth().to_bits().to_le_bytes());
        hash.update(
            &particles
                .minimum_effective_samples()
                .to_bits()
                .to_le_bytes(),
        );
        hash.update(&[u8::from(particles.retain_reverse_trace())]);
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
    pub fn scheme(&self) -> &'static str {
        self.path.scheme()
    }

    /// Calibration surface in the funded residual-equity coordinate, when this
    /// is a stochastic-dividend LSV plan.
    #[must_use]
    pub fn lsv_time_nodes(&self) -> Option<&[f64]> {
        self.path.lsv_surface().map(|s| s.times())
    }
    #[must_use]
    pub fn lsv_log_moneyness_nodes(&self) -> Option<&[f64]> {
        self.path.lsv_surface().map(|s| s.log_nodes())
    }
    #[must_use]
    pub fn lsv_squared_leverage(&self) -> Option<&[f64]> {
        self.path.lsv_surface().map(|s| s.squared_leverage())
    }
    #[must_use]
    pub fn lsv_initial_residual_equity(&self) -> Option<f64> {
        self.path.lsv_surface().map(|s| s.initial_f())
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
