//! Explicit two-stage LSV API: calibrate an LV target once, then evaluate existing
//! compiled products with independent MC/RQMC paths. This experimental boundary
//! keeps calibrated Local-variance risk distinct from market-IV VegaKT.

use crate::core::PathIndex;
use crate::engine::processes::rough_lsv::ROUGH_RANDOM_BLOCKS;
use crate::market::LocalVarianceGrid;
use crate::mc::lsv::{
    BERGOMI_LSV_SCHEME, BERGOMI_TWO_FACTOR_LSV_SCHEME, BergomiLsvPath, BergomiLsvPlan,
    CalibratedBergomiLsv, CalibratedRoughBergomiLsv, LSV_CALIBRATION_REVERSE, LsvError,
    LsvLeverageSurface, LsvParticleConfig, ROUGH_BERGOMI_LSV_SCHEME, RoughBergomiLsvPath,
    RoughBergomiLsvPlan, calibrate_bergomi_lsv_parallel, calibrate_rough_bergomi_lsv_parallel,
};
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolTimeGrid, RandomDomain, RqmcPlan, VarianceReduction,
    inverse_standard_normal,
};
use crate::models::{Bergomi1Factor, BergomiDynamics, ModelSpec, RoughBergomi};
use crate::{Fingerprint, MonteCarloError, PricingRequest, SimulationPlan};

#[derive(Clone, Debug, PartialEq)]
pub struct LsvPrice {
    pub value: f64,
    /// Pricing uncertainty conditional on the one realized calibration surface.
    /// This excludes calibration noise and discretization / smoothing bias.
    pub standard_error: f64,
    pub independent_sampling_units: u64,
    pub evaluated_paths: u128,
    pub calibration_seed: u64,
    pub plan_fingerprint: Fingerprint,
    pub scheme: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LsvLocalVarianceRisk {
    pub price: LsvPrice,
    pub time_nodes: Box<[f64]>,
    pub log_moneyness_nodes: Box<[f64]>,
    /// dPrice / d(relative Dupire variance node), including recalibration.
    pub node_adjoints: Box<[f64]>,
    /// Available for independent RQMC scrambles. Conditional on calibration.
    pub standard_errors: Option<Box<[f64]>>,
    pub method: &'static str,
}

type RiskOutput = Option<(Vec<f64>, Option<Vec<f64>>)>;

/// A calibrated LSV model the shared pricing core can evaluate.
trait CalibratedModel: Clone + std::fmt::Debug + Send + Sync {
    type Plan: PathModel;
    const SCHEME: &'static str;
    /// Independent Gaussian blocks per time step in the pricing layout.
    const RANDOM_BLOCKS: usize;
    fn surface(&self) -> &LsvLeverageSurface;
    fn config(&self) -> &LsvParticleConfig;
    fn target(&self) -> &LocalVarianceGrid;
    fn reverse_leverage(&self, adjoints: &[f64]) -> Result<Vec<f64>, LsvError>;
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<Self::Plan, LsvError>;
}

/// The path plan of a calibrated LSV model.
trait PathModel: Clone + std::fmt::Debug + Send + Sync {
    type Path;
    fn times(&self) -> &[f64];
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError>;
    fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError>;
    fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<Self::Path, LsvError>;
    fn path_states(path: &Self::Path) -> &[f64];
    fn leverage_adjoints(path: &Self::Path, seeds: &[f64]) -> Result<Box<[f64]>, LsvError>;
}

impl<F: BergomiDynamics> CalibratedModel for CalibratedBergomiLsv<F> {
    type Plan = BergomiLsvPlan<F>;
    const SCHEME: &'static str = if F::FACTOR_COUNT == 1 {
        BERGOMI_LSV_SCHEME
    } else {
        BERGOMI_TWO_FACTOR_LSV_SCHEME
    };
    const RANDOM_BLOCKS: usize = 1 + F::FACTOR_COUNT;
    fn surface(&self) -> &LsvLeverageSurface {
        self.surface()
    }
    fn config(&self) -> &LsvParticleConfig {
        self.config()
    }
    fn target(&self) -> &LocalVarianceGrid {
        self.target()
    }
    fn reverse_leverage(&self, adjoints: &[f64]) -> Result<Vec<f64>, LsvError> {
        self.reverse_leverage(adjoints)
    }
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<BergomiLsvPlan<F>, LsvError> {
        self.pricing_plan(grid)
    }
}

impl<F: BergomiDynamics> PathModel for BergomiLsvPlan<F> {
    type Path = BergomiLsvPath<F>;
    fn times(&self) -> &[f64] {
        self.times()
    }
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        self.pseudo_shocks(seed, path, domain)
    }
    fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        self.evolve_states(initial_f, shocks, states)
    }
    fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<BergomiLsvPath<F>, LsvError> {
        self.evolve_path(initial_f, shocks)
    }
    fn path_states(path: &BergomiLsvPath<F>) -> &[f64] {
        path.states()
    }
    fn leverage_adjoints(path: &BergomiLsvPath<F>, seeds: &[f64]) -> Result<Box<[f64]>, LsvError> {
        Ok(path.reverse(seeds)?.squared_leverage)
    }
}

impl CalibratedModel for CalibratedRoughBergomiLsv {
    type Plan = RoughBergomiLsvPlan;
    const SCHEME: &'static str = ROUGH_BERGOMI_LSV_SCHEME;
    const RANDOM_BLOCKS: usize = ROUGH_RANDOM_BLOCKS;
    fn surface(&self) -> &LsvLeverageSurface {
        self.surface()
    }
    fn config(&self) -> &LsvParticleConfig {
        self.config()
    }
    fn target(&self) -> &LocalVarianceGrid {
        self.target()
    }
    fn reverse_leverage(&self, adjoints: &[f64]) -> Result<Vec<f64>, LsvError> {
        self.reverse_leverage(adjoints)
    }
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<RoughBergomiLsvPlan, LsvError> {
        self.pricing_plan(grid)
    }
}

impl PathModel for RoughBergomiLsvPlan {
    type Path = RoughBergomiLsvPath;
    fn times(&self) -> &[f64] {
        self.times()
    }
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        self.pseudo_shocks(seed, path, domain)
    }
    fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        self.evolve_states(initial_f, shocks, states)
    }
    fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<RoughBergomiLsvPath, LsvError> {
        self.evolve_path(initial_f, shocks)
    }
    fn path_states(path: &RoughBergomiLsvPath) -> &[f64] {
        path.states()
    }
    fn leverage_adjoints(
        path: &RoughBergomiLsvPath,
        seeds: &[f64],
    ) -> Result<Box<[f64]>, LsvError> {
        Ok(path.reverse(seeds)?.squared_leverage)
    }
}

/// Everything after the model choice: target refinement, calibration, the
/// independent MC/RQMC pricing loop and the calibrated local-variance reverse.
#[derive(Clone, Debug)]
struct LsvPricingCore<C: CalibratedModel> {
    base: SimulationPlan,
    calibration: C,
    path_plan: C::Plan,
    original_target: LocalVarianceGrid,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    risk_supported: bool,
    fingerprint: Fingerprint,
}

#[derive(Clone, Debug)]
pub struct BergomiLsvPricingPlan<F: BergomiDynamics = Bergomi1Factor> {
    core: LsvPricingCore<CalibratedBergomiLsv<F>>,
}

/// Deterministic-rate rough Bergomi LSV. The same two-stage contract as
/// `BergomiLsvPricingPlan`, with the rough driver's three Gaussian blocks.
#[derive(Clone, Debug)]
pub struct RoughBergomiLsvPricingPlan {
    core: LsvPricingCore<CalibratedRoughBergomiLsv>,
}

impl<F: BergomiDynamics> BergomiLsvPricingPlan<F> {
    /// `target_request.model` is the Dupire LocalVolatility target. The request
    /// must be Price-only: use `evaluate_local_variance_risk` explicitly for the
    /// new derivative contract. All product, smoothing and dividend semantics
    /// are compiled by the existing public request path.
    pub fn compile(
        target_request: &PricingRequest,
        factor: F,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let tag: &[u8] = if F::FACTOR_COUNT == 1 {
            b"pricing/bergomi-lsv-plan/v2\0"
        } else {
            b"pricing/bergomi-two-factor-lsv-plan/v2\0"
        };
        LsvPricingCore::compile(
            target_request,
            particles,
            policy,
            tag,
            &factor.parameters(),
            |target, initial_f, particles, executor| {
                calibrate_bergomi_lsv_parallel(target, factor, initial_f, particles, executor)
            },
        )
        .map(|core| Self { core })
    }
    #[must_use]
    pub fn calibration(&self) -> &CalibratedBergomiLsv<F> {
        &self.core.calibration
    }
    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.core.fingerprint
    }
    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.core.policy
    }
    pub fn evaluate(&self) -> Result<LsvPrice, MonteCarloError> {
        self.core.evaluate()
    }
    pub fn evaluate_local_variance_risk(&self) -> Result<LsvLocalVarianceRisk, MonteCarloError> {
        self.core.evaluate_local_variance_risk()
    }
}

impl RoughBergomiLsvPricingPlan {
    /// `target_request.model` is the Dupire LocalVolatility target, as for
    /// `BergomiLsvPricingPlan`. Rates, carry and dividends follow the request's
    /// deterministic market; the rough driver starts at the valuation time.
    pub fn compile(
        target_request: &PricingRequest,
        model: RoughBergomi,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        LsvPricingCore::compile(
            target_request,
            particles,
            policy,
            b"pricing/rough-bergomi-lsv-plan/v2\0",
            &[model.hurst(), model.vol_of_vol(), model.correlation()],
            |target, initial_f, particles, executor| {
                calibrate_rough_bergomi_lsv_parallel(target, model, initial_f, particles, executor)
            },
        )
        .map(|core| Self { core })
    }
    #[must_use]
    pub fn calibration(&self) -> &CalibratedRoughBergomiLsv {
        &self.core.calibration
    }
    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.core.fingerprint
    }
    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.core.policy
    }
    pub fn evaluate(&self) -> Result<LsvPrice, MonteCarloError> {
        self.core.evaluate()
    }
    pub fn evaluate_local_variance_risk(&self) -> Result<LsvLocalVarianceRisk, MonteCarloError> {
        self.core.evaluate_local_variance_risk()
    }
}

impl<C: CalibratedModel> LsvPricingCore<C> {
    fn compile(
        target_request: &PricingRequest,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
        tag: &[u8],
        parameters: &[f64],
        calibrate: impl FnOnce(
            &LocalVarianceGrid,
            f64,
            LsvParticleConfig,
            &DeterministicExecutor,
        ) -> Result<C, LsvError>,
    ) -> Result<Self, MonteCarloError> {
        let ModelSpec::LocalVolatility(target) = target_request.model() else {
            return Err(MonteCarloError::UnsupportedModel {
                model: "Bergomi LSV requires a LocalVolatility calibration target",
            });
        };
        let risk = target_request.risk();
        if risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some() {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Bergomi LSV: use price_only and evaluate_local_variance_risk",
            });
        }
        let base = SimulationPlan::compile_hybrid_base(target_request, policy)?;
        let grid = base.lsv_time_grid()?;
        let original_target = target.local_variance_grid().clone();
        // Calibration and pricing share all contractual and dividend nodes.
        let x = original_target.log_moneyness_nodes();
        let mut values = Vec::with_capacity(grid.nodes().len() * x.len());
        for &t in grid.nodes() {
            for &k in x {
                values.push(original_target.interpolate(t, k)?.value);
            }
        }
        let refined = LocalVarianceGrid::new(
            grid.nodes().to_vec(),
            x.to_vec(),
            values,
            original_target.floor(),
            original_target.cap(),
        )?;
        let initial_f = target_request.market().equity().forward().spot().get();
        // The calibration is bit-identical for any worker count; the plan's own
        // execution policy only sets how many threads share each time step.
        let calibration = calibrate(
            &refined,
            initial_f,
            particles,
            &DeterministicExecutor::new(policy)?,
        )?;
        let path_plan = calibration.pricing_plan(grid)?;
        let mut hash = blake3::Hasher::new();
        hash.update(tag);
        hash.update(base.plan_fingerprint().as_bytes());
        for v in parameters.iter().copied().chain([
            calibration.config().log_bandwidth(),
            calibration.config().minimum_effective_samples(),
        ]) {
            hash.update(&v.to_bits().to_be_bytes());
        }
        hash.update(&(calibration.config().particle_count() as u64).to_be_bytes());
        hash.update(&calibration.config().seed().to_be_bytes());
        hash.update(&[u8::from(calibration.config().retain_reverse_trace())]);
        for v in calibration.surface().squared_leverage() {
            hash.update(&v.to_bits().to_be_bytes());
        }
        let fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        let risk_supported =
            target_request.product().supports_pathwise_risk() || risk.payoff_smoothing().is_some();
        Ok(Self {
            base,
            calibration,
            path_plan,
            original_target,
            engine: target_request.engine(),
            policy,
            risk_supported,
            fingerprint,
        })
    }

    fn evaluate(&self) -> Result<LsvPrice, MonteCarloError> {
        Ok(self.run(false)?.0)
    }

    fn evaluate_local_variance_risk(&self) -> Result<LsvLocalVarianceRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV discontinuous payoff requires explicit smoothing",
            });
        }
        if !self.calibration.config().retain_reverse_trace() {
            return Err(LsvError::ReverseTraceNotRetained.into());
        }
        let (price, risk) = self.run(true)?;
        let (values, standard_errors) = risk.expect("risk was requested");
        Ok(LsvLocalVarianceRisk {
            price,
            time_nodes: self.original_target.time_nodes().into(),
            log_moneyness_nodes: self.original_target.log_moneyness_nodes().into(),
            node_adjoints: values.into_boxed_slice(),
            standard_errors: standard_errors.map(Vec::into_boxed_slice),
            method: LSV_CALIBRATION_REVERSE,
        })
    }

    fn target_reverse(&self, leverage: &[f64]) -> Result<Vec<f64>, MonteCarloError> {
        let refined = self.calibration.reverse_leverage(leverage)?;
        let m = self.original_target.log_moneyness_nodes().len();
        let mut original = vec![0.0; self.original_target.values().len()];
        for (r, &t) in self.calibration.target().time_nodes().iter().enumerate() {
            for (j, &x) in self
                .original_target
                .log_moneyness_nodes()
                .iter()
                .enumerate()
            {
                self.original_target
                    .interpolate(t, x)?
                    .transpose_accumulate(refined[r * m + j], &mut original, m);
            }
        }
        Ok(original)
    }

    fn bridge(&self, vr: VarianceReduction) -> Result<Option<BrownianBridgePlan>, MonteCarloError> {
        if !vr.brownian_bridge() {
            return Ok(None);
        }
        BrownianBridgePlan::compile(self.path_plan.times().to_vec(), 1)
            .map(Some)
            .map_err(|e| MonteCarloError::LocalVol(e.into()))
    }

    fn apply_bridge(
        &self,
        shocks: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
    ) -> Result<Vec<f64>, MonteCarloError> {
        if let Some(bridge) = bridge {
            let n = self.path_plan.times().len() - 1;
            let mut out = Vec::with_capacity(shocks.len());
            for block in shocks.chunks_exact(n) {
                out.extend(
                    bridge
                        .apply_one_factor(block)
                        .map_err(|e| MonteCarloError::LocalVol(e.into()))?,
                );
            }
            Ok(out)
        } else {
            Ok(shocks)
        }
    }

    fn sample(
        &self,
        shocks: &[f64],
        path: u64,
        risk: bool,
        antithetic: bool,
        output: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        for sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = shocks.iter().map(|z| z * sign).collect::<Vec<_>>();
            let initial_f = self.calibration.surface().initial_f();
            let weight = if antithetic { 0.5 } else { 1.0 };
            if !risk {
                // Price only: the same states without the reverse-mode records.
                let mut states = Vec::new();
                self.path_plan
                    .evolve_states(initial_f, &shocks, &mut states)?;
                let (price, _) = self.base.lsv_payoff(&states, PathIndex::new(path), false)?;
                output[0] += weight * price;
                continue;
            }
            let recorded = self.path_plan.evolve_path(initial_f, &shocks)?;
            let (price, seeds) = self.base.lsv_payoff(
                C::Plan::path_states(&recorded),
                PathIndex::new(path),
                true,
            )?;
            output[0] += weight * price;
            if let Some(seeds) = seeds {
                let adjoints = C::Plan::leverage_adjoints(&recorded, &seeds)?;
                for (v, &a) in output[1..].iter_mut().zip(adjoints.iter()) {
                    *v += weight * a;
                }
            }
        }
        Ok(())
    }

    fn run(&self, risk: bool) -> Result<(LsvPrice, RiskOutput), MonteCarloError> {
        let executor = DeterministicExecutor::new(self.policy)?;
        let width = 1 + if risk {
            self.calibration.surface().squared_leverage().len()
        } else {
            0
        };
        let (price_stats, units, paths, risk_output) = match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => {
                let n = engine.independent_sampling_units().get();
                let bridge = self.bridge(engine.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(n, width, |i, out| {
                    let z = self.path_plan.pseudo_shocks(
                        engine.master_seed(),
                        i,
                        RandomDomain::Valuation,
                    )?;
                    let z = self.apply_bridge(z, bridge.as_ref())?;
                    self.sample(&z, i, risk, engine.variance_reduction().antithetic(), out)
                })?;
                let gradient = if risk {
                    let leverage = stats[1..]
                        .iter()
                        .map(|s| s.sum().total() / n as f64)
                        .collect::<Vec<_>>();
                    Some((self.target_reverse(&leverage)?, None))
                } else {
                    None
                };
                (stats[0], n, engine.evaluated_paths(), gradient)
            }
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => {
                let dimension = u32::try_from(
                    C::RANDOM_BLOCKS * (self.path_plan.times().len() - 1),
                )
                .map_err(|_| LsvError::InvalidInput {
                    field: "random_dimension",
                    index: 0,
                })?;
                let qmc = RqmcPlan::compile(engine, dimension)?;
                let bridge = self.bridge(engine.variance_reduction())?;
                let n = engine.points_per_scramble().get();
                let mut prices = Vec::with_capacity(engine.scramble_count().get() as usize);
                let mut risks = Vec::new();
                for scramble in 0..engine.scramble_count().get() {
                    let stats = executor.try_map_reduce_statistics_vector(n, width, |i, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc.uniform(scramble, i, d).map_err(|_| {
                                    LsvError::InvalidInput {
                                        field: "rqmc_coordinate",
                                        index: d as usize,
                                    }
                                })?;
                                inverse_standard_normal(u).map_err(|_| LsvError::InvalidInput {
                                    field: "rqmc_normal",
                                    index: d as usize,
                                })
                            })
                            .collect::<Result<Vec<_>, LsvError>>()?;
                        let z = self.apply_bridge(z, bridge.as_ref())?;
                        self.sample(
                            &z,
                            u64::from(scramble) * n + i,
                            risk,
                            engine.variance_reduction().antithetic(),
                            out,
                        )
                    })?;
                    prices.push(stats[0].sum().total() / n as f64);
                    if risk {
                        risks.push(
                            self.target_reverse(
                                &stats[1..]
                                    .iter()
                                    .map(|s| s.sum().total() / n as f64)
                                    .collect::<Vec<_>>(),
                            )?,
                        );
                    }
                }
                let count = u64::from(engine.scramble_count().get());
                let gradient = if risk {
                    let mut means = Vec::new();
                    let mut errors = Vec::new();
                    for j in 0..self.original_target.values().len() {
                        let values = risks.iter().map(|r| r[j]).collect::<Vec<_>>();
                        let stat = DeterministicStatistics::from_ordered_values_two_pass(&values);
                        means.push(stat.sum().total() / count as f64);
                        errors.push(standard_error(stat, count)?);
                    }
                    Some((means, Some(errors)))
                } else {
                    None
                };
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    count,
                    n as u128
                        * count as u128
                        * if engine.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                    gradient,
                )
            }
        };
        let price = LsvPrice {
            value: price_stats.sum().total() / units as f64,
            standard_error: standard_error(price_stats, units)?,
            independent_sampling_units: units,
            evaluated_paths: paths,
            calibration_seed: self.calibration.config().seed(),
            plan_fingerprint: self.fingerprint,
            scheme: C::SCHEME,
        };
        Ok((price, risk_output))
    }
}

fn standard_error(stat: DeterministicStatistics, n: u64) -> Result<f64, MonteCarloError> {
    let variance = stat
        .moments()
        .sample_variance()
        .ok_or(MonteCarloError::InsufficientSamplingUnits { count: n })?;
    let error = (variance / n as f64).sqrt();
    if !error.is_finite() {
        return Err(LsvError::InvalidInput {
            field: "estimator_standard_error",
            index: 0,
        }
        .into());
    }
    Ok(error)
}
