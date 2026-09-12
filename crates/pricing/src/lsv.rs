//! Explicit two-stage LSV API: calibrate an LV target once, then evaluate existing
//! compiled products with independent MC/RQMC paths. This experimental boundary
//! keeps calibrated Local-variance risk distinct from market-IV VegaKT.

use crate::{Fingerprint, MonteCarloError, PricingRequest, SimulationPlan};
use pricing_core::PathIndex;
use pricing_market::LocalVarianceGrid;
use pricing_mc::lsv::{
    BERGOMI_LSV_SCHEME, BergomiLsvPlan, CalibratedBergomiLsv, LSV_CALIBRATION_REVERSE, LsvError,
    LsvParticleConfig, calibrate_bergomi_lsv,
};
use pricing_mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, RandomDomain, RqmcPlan, VarianceReduction, inverse_standard_normal,
};
use pricing_models::{Bergomi1Factor, ModelSpec};

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

#[derive(Clone, Debug)]
pub struct BergomiLsvPricingPlan {
    base: SimulationPlan,
    calibration: CalibratedBergomiLsv,
    path_plan: BergomiLsvPlan,
    original_target: LocalVarianceGrid,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    risk_supported: bool,
    fingerprint: Fingerprint,
}

impl BergomiLsvPricingPlan {
    /// `target_request.model` is the Dupire LocalVolatility target. The request
    /// must be Price-only: use `evaluate_local_variance_risk` explicitly for the
    /// new derivative contract. All product, smoothing and dividend semantics
    /// are compiled by the existing public request path.
    pub fn compile(
        target_request: &PricingRequest,
        factor: Bergomi1Factor,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
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
        let base = SimulationPlan::compile(target_request, policy)?;
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
        let calibration = calibrate_bergomi_lsv(&refined, factor, initial_f, particles)?;
        let path_plan = calibration.pricing_plan(grid)?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"pricing/bergomi-lsv-plan/v1\0");
        hash.update(base.plan_fingerprint().as_bytes());
        for v in [
            factor.mean_reversion(),
            factor.vol_of_vol(),
            factor.correlation(),
            calibration.config().log_bandwidth(),
            calibration.config().minimum_effective_samples(),
        ] {
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

    #[must_use]
    pub fn calibration(&self) -> &CalibratedBergomiLsv {
        &self.calibration
    }
    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }

    pub fn evaluate(&self) -> Result<LsvPrice, MonteCarloError> {
        Ok(self.run(false)?.0)
    }

    pub fn evaluate_local_variance_risk(&self) -> Result<LsvLocalVarianceRisk, MonteCarloError> {
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
            let n = shocks.len() / 2;
            let mut out = bridge
                .apply_one_factor(&shocks[..n])
                .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
            out.extend(
                bridge
                    .apply_one_factor(&shocks[n..])
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?,
            );
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
            let states = self
                .path_plan
                .evolve_path(self.calibration.surface().initial_f(), &shocks)?;
            let (price, seeds) =
                self.base
                    .lsv_payoff(states.states(), PathIndex::new(path), risk)?;
            let weight = if antithetic { 0.5 } else { 1.0 };
            output[0] += weight * price;
            if let Some(seeds) = seeds {
                let adj = states.reverse(&seeds)?;
                for (v, &a) in output[1..].iter_mut().zip(&adj.squared_leverage) {
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
                let dimension =
                    u32::try_from(2 * (self.path_plan.times().len() - 1)).map_err(|_| {
                        LsvError::InvalidInput {
                            field: "random_dimension",
                            index: 0,
                        }
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
            scheme: BERGOMI_LSV_SCHEME,
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
