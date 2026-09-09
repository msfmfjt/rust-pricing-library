use pricing_aad::{AadTilePolicy, CheckpointPolicy, SoaWorkspace};
use pricing_core::{Date, DayCountConvention, PathIndex, PositiveF64, SchemaVersion, UnderlyingId};
use pricing_market::{
    AffineDividendTransform, CurveRegion, DiscountCurve, EquityForward, LocalVarianceGrid,
};
use pricing_mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolDividendCheckpointSchedule, LocalVolLogEulerPlan, LocalVolTimeGrid,
    Philox4x32, PseudoMcConfig, RandomCoordinate, RandomDomain, RqmcConfig, RqmcPlan,
    RqmcPlanError, inverse_standard_normal,
};
use pricing_models::ModelSpec;
use pricing_product::{CompiledPayoff, GraphFingerprint, GraphLimitPolicy, ProductSpec};
use pricing_risk::{GammaConfig, SmileDynamics, SpotBump};

use crate::{
    Diagnostics, Estimate, EstimatorKind, Fingerprint, MonteCarloError, PricingRequest,
    PricingResult, PricingWarning, ReplayMetadata, ResultBuildError, RiskEstimate, RiskReport,
    RiskUnit, fingerprint_request,
};

const NORMAL_95: f64 = 1.959_963_984_540_054;
const PRICE: usize = 0;
const DELTA: usize = 1;
const VEGA: usize = 2;
const GAMMA: usize = 3;
const BUMP_DELTA: usize = 4;
const DELTA_DIFFERENCE: usize = 5;
const BUMP_VEGA: usize = 6;
const VEGA_DIFFERENCE: usize = 7;
const BUMP_GAMMA: usize = 8;
const GAMMA_DIFFERENCE: usize = 9;
const PATHWISE_COMPONENTS: usize = 10;
const AAD_WORKSPACE_SLOTS: usize = 5;
const DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP: f64 = 1.0e-4;
const DEFAULT_VALIDATION_VOLATILITY_BUMP: f64 = 1.0e-4;
const PLAN_FINGERPRINT_VERSION: u32 = 1;

/// Immutable one-expiry plan for the European Black-Scholes MC/RQMC slice.
#[derive(Clone, Debug)]
pub struct SimulationPlan {
    valuation_date: Date,
    expiry: Date,
    underlying: UnderlyingId,
    time: f64,
    forward: f64,
    spot: f64,
    discount: f64,
    volatility: f64,
    total_variance: f64,
    standard_deviation: f64,
    payoff: CompiledPayoff,
    engine: EngineConfig,
    execution_policy: ExecutionPolicy,
    aad_tile_policy: AadTilePolicy,
    checkpoint_policy: CheckpointPolicy,
    request_delta: bool,
    request_gamma: Option<GammaConfig>,
    request_vega: bool,
    smile_dynamics: SmileDynamics,
    validation_spot_bump: f64,
    validation_volatility_bump: f64,
    request_fingerprint: [u8; 32],
    plan_fingerprint: Fingerprint,
    discount_region: CurveRegion,
    dividend_region: CurveRegion,
    market_forward: EquityForward,
    local_volatility: Option<LocalVolRuntime>,
}

#[derive(Clone, Debug)]
struct LocalVolRuntime {
    grid: LocalVarianceGrid,
    plan: LocalVolLogEulerPlan,
    dividends: Option<AffineDividendTransform>,
    dividend_schedule: Option<LocalVolDividendCheckpointSchedule>,
    terminal_affine_a: f64,
    terminal_affine_b: f64,
}

impl LocalVolRuntime {
    fn maximum_total_variance(&self) -> f64 {
        self.plan
            .time_grid()
            .nodes()
            .windows(2)
            .zip(
                self.grid
                    .values()
                    .chunks(self.grid.log_moneyness_nodes().len()),
            )
            .map(|(times, row)| {
                let dt = times[1] - times[0];
                let row_max = row.iter().copied().fold(0.0_f64, f64::max);
                dt * row_max
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
struct LocalVolBumpRuntimes {
    down_spot: f64,
    down: LocalVolRuntime,
    up_spot: f64,
    up: LocalVolRuntime,
    spot_bump: f64,
}

fn compile_local_vol_runtime(
    grid: LocalVarianceGrid,
    market_forward: &EquityForward,
    expiry_time: f64,
    terminal_affine_a: f64,
    terminal_affine_b: f64,
) -> Result<LocalVolRuntime, MonteCarloError> {
    let first_time = grid.time_nodes()[0];
    let last_time = grid.time_nodes()[grid.time_nodes().len() - 1];
    if first_time.to_bits() != 0.0_f64.to_bits() || last_time.to_bits() != expiry_time.to_bits() {
        return Err(MonteCarloError::InvalidLocalVolatilityTimeGrid {
            expiry_bits: expiry_time.to_bits(),
            first_bits: first_time.to_bits(),
            last_bits: last_time.to_bits(),
        });
    }
    let maximum_step = grid
        .time_nodes()
        .windows(2)
        .map(|times| times[1] - times[0])
        .fold(0.0_f64, f64::max);
    let time_grid = if let Some(dividends) = market_forward.discrete_dividends() {
        LocalVolTimeGrid::compile_with_dividends(
            grid.time_nodes().to_vec(),
            dividends,
            maximum_step,
        )?
    } else {
        LocalVolTimeGrid::compile(grid.time_nodes().to_vec(), maximum_step)?
    };
    if time_grid.nodes() != grid.time_nodes() {
        return Err(MonteCarloError::InvalidLocalVolatilityTimeGrid {
            expiry_bits: expiry_time.to_bits(),
            first_bits: first_time.to_bits(),
            last_bits: last_time.to_bits(),
        });
    }
    let mut forwards = Vec::with_capacity(time_grid.nodes().len());
    for time in time_grid.nodes().iter().copied() {
        forwards.push(market_forward.evaluate(time)?.forward);
    }
    let plan = LocalVolLogEulerPlan::new(time_grid, forwards)?;
    let dividends = market_forward.discrete_dividends().cloned();
    let dividend_schedule = dividends
        .as_ref()
        .map(|dividends| LocalVolDividendCheckpointSchedule::compile(plan.time_grid(), dividends))
        .transpose()?;
    Ok(LocalVolRuntime {
        grid,
        plan,
        dividends,
        dividend_schedule,
        terminal_affine_a,
        terminal_affine_b,
    })
}

impl SimulationPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let engine = request.engine();
        let ProductSpec::EuropeanVanilla(product) = request.product();
        let time =
            DayCountConvention::Act365F.year_fraction(request.valuation_date(), product.expiry());
        let market_forward = request.market().equity().forward();
        let forward_evaluation = market_forward.evaluate(time)?;
        let discount = market_forward.discount_curve().discount(time)?;
        let spot = market_forward.spot().get();
        let (volatility, total_variance, local_volatility) = match request.model() {
            ModelSpec::BlackScholes(model) => {
                let volatility = model.volatility().get();
                (volatility, volatility * volatility * time, None)
            }
            ModelSpec::LocalVolatility(model) => {
                if request.risk().vega() || request.risk().vega_kt().is_some() {
                    return Err(MonteCarloError::UnsupportedRiskForModel {
                        model: request.model().name(),
                    });
                }
                let runtime = compile_local_vol_runtime(
                    model.local_variance_grid().clone(),
                    market_forward,
                    time,
                    forward_evaluation.affine_coordinate.a(),
                    forward_evaluation.affine_coordinate.b(),
                )?;
                (0.0, runtime.maximum_total_variance(), Some(runtime))
            }
        };
        if !total_variance.is_finite() {
            return Err(MonteCarloError::NonFiniteTotalVariance {
                bits: total_variance.to_bits(),
            });
        }
        if total_variance > 0.0 {
            let independent_units = match engine {
                EngineConfig::PseudoMonteCarlo(config) => config.independent_sampling_units().get(),
                EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                    u64::from(config.scramble_count().get())
                }
            };
            if independent_units < 2 {
                return Err(MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                });
            }
        }
        let payoff = product.source_graph()?.compile(GraphLimitPolicy::DEFAULT)?;
        let request_fingerprint = *fingerprint_request(request)?.as_bytes();
        let aad_tile_policy = AadTilePolicy::resolve(
            execution_policy.reduction_block_size().get(),
            request.risk().aad_tile_capacity(),
        )?;
        let checkpoint_policy = CheckpointPolicy::resolve(request.risk().checkpoint_interval());
        let plan_fingerprint = build_plan_fingerprint(
            request_fingerprint,
            payoff.tape_fingerprint(),
            execution_policy,
            aad_tile_policy,
            checkpoint_policy,
        );
        if let Some(gamma) = request.risk().gamma() {
            let bump = resolve_spot_bump(gamma, spot);
            if !bump.is_finite() || bump >= spot {
                return Err(MonteCarloError::InvalidGammaBump {
                    spot_bits: spot.to_bits(),
                    bump_bits: bump.to_bits(),
                });
            }
        }
        let validation_spot_bump = request
            .risk()
            .gamma()
            .map_or(spot * DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP, |gamma| {
                resolve_spot_bump(gamma, spot)
            });
        let validation_volatility_bump = if volatility == 0.0 {
            0.0
        } else {
            DEFAULT_VALIDATION_VOLATILITY_BUMP.min(volatility * 0.5)
        };
        Ok(Self {
            valuation_date: request.valuation_date(),
            expiry: product.expiry(),
            underlying: product.underlying(),
            time,
            forward: forward_evaluation.forward,
            spot,
            discount,
            volatility,
            total_variance,
            standard_deviation: total_variance.sqrt(),
            payoff,
            engine,
            execution_policy,
            aad_tile_policy,
            checkpoint_policy,
            request_delta: request.risk().delta(),
            request_gamma: request.risk().gamma(),
            request_vega: request.risk().vega(),
            smile_dynamics: request.risk().smile_dynamics(),
            validation_spot_bump,
            validation_volatility_bump,
            request_fingerprint,
            plan_fingerprint,
            discount_region: forward_evaluation.discount_region,
            dividend_region: forward_evaluation.dividend_region,
            market_forward: market_forward.clone(),
            local_volatility,
        })
    }

    #[must_use]
    pub const fn valuation_date(&self) -> Date {
        self.valuation_date
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    #[must_use]
    pub const fn forward(&self) -> f64 {
        self.forward
    }

    #[must_use]
    pub const fn discount(&self) -> f64 {
        self.discount
    }

    #[must_use]
    pub const fn total_variance(&self) -> f64 {
        self.total_variance
    }

    #[must_use]
    pub const fn payoff_fingerprint(&self) -> GraphFingerprint {
        self.payoff.tape_fingerprint()
    }

    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }

    #[must_use]
    pub const fn request_fingerprint(&self) -> Fingerprint {
        Fingerprint::from_bytes(self.request_fingerprint)
    }

    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.plan_fingerprint
    }

    pub fn execute(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => self.execute_pseudo(engine),
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => self.execute_rqmc(engine),
        }
    }

    fn execute_pseudo(&self, engine: PseudoMcConfig) -> Result<MonteCarloPrice, MonteCarloError> {
        if let Some(local_volatility) = &self.local_volatility {
            return self.execute_local_vol_pseudo(engine, local_volatility);
        }
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let generator = Philox4x32::from_seed(engine.master_seed());
        let antithetic = engine.variance_reduction().antithetic();
        let statistics = if self.risk_enabled() {
            executor.try_map_reduce_statistics_array_with_aad_workspace(
                engine.independent_sampling_units().get(),
                self.aad_tile_policy.resolved_capacity(),
                AAD_WORKSPACE_SLOTS,
                |sampling_unit, lane, workspace| {
                    let normal = self.normal(&generator, sampling_unit);
                    let primary = self.pathwise_values(normal, lane, workspace)?;
                    if antithetic {
                        let mate = self.pathwise_values(-normal, lane, workspace)?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                    }
                },
            )?
        } else {
            let price = executor.try_map_reduce_statistics(
                engine.independent_sampling_units().get(),
                |sampling_unit| {
                    let normal = self.normal(&generator, sampling_unit);
                    let primary = self.discounted_payoff(normal)?;
                    if antithetic {
                        let mate = self.discounted_payoff(-normal)?;
                        Ok::<f64, pricing_product::GraphError>((primary + mate) * 0.5)
                    } else {
                        Ok::<f64, pricing_product::GraphError>(primary)
                    }
                },
            )?;
            let mut values = [DeterministicStatistics::default(); PATHWISE_COMPONENTS];
            values[PRICE] = price;
            values
        };

        let independent_units = engine.independent_sampling_units().get();
        let price = statistics[PRICE].sum().total() / independent_units as f64;
        let sampling_variance = if self.total_variance == 0.0 {
            0.0
        } else {
            statistics[PRICE].moments().sample_variance().ok_or(
                MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                },
            )?
        };
        let estimator_variance = sampling_variance / independent_units as f64;
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        debug_assert_eq!(estimate.value().get().to_bits(), price.to_bits());
        let risks = self.build_risk_report(
            &statistics,
            independent_units,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let risk_diagnostics = self.build_risk_diagnostics(
            &statistics,
            independent_units,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let warnings = extrapolation_warnings(self.discount_region, self.dividend_region);
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(warnings),
            replay: ReplayMetadata::new(
                SchemaVersion::CURRENT,
                self.request_fingerprint,
                crate::version(),
                format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            ),
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths: engine.evaluated_paths(),
            diagnostics: MonteCarloDiagnostics {
                master_seed: engine.master_seed(),
                estimator: EstimatorKind::PseudoMonteCarlo,
                scramble_count: None,
                direction_checksum: None,
                scramble_checksum: None,
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
            },
        })
    }

    fn execute_rqmc(&self, engine: RqmcConfig) -> Result<MonteCarloPrice, MonteCarloError> {
        if let Some(local_volatility) = &self.local_volatility {
            return self.execute_local_vol_rqmc(engine, local_volatility);
        }
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let qmc = RqmcPlan::compile(engine, 1)?;
        let antithetic = engine.variance_reduction().antithetic();
        let points = engine.points_per_scramble().get();
        let mut replicate_values: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        for scramble in 0..engine.scramble_count().get() {
            let within = if self.risk_enabled() {
                executor.try_map_reduce_statistics_array_with_aad_workspace(
                    points,
                    self.aad_tile_policy.resolved_capacity(),
                    AAD_WORKSPACE_SLOTS,
                    |point, lane, workspace| {
                        let normal = self.rqmc_normal(&qmc, scramble, point)?;
                        let primary = self.pathwise_values(normal, lane, workspace)?;
                        if antithetic {
                            let mate = self.pathwise_values(-normal, lane, workspace)?;
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                                |component| (primary[component] + mate[component]) * 0.5,
                            ))
                        } else {
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                        }
                    },
                )?
            } else {
                let price = executor.try_map_reduce_statistics(points, |point| {
                    let normal = self.rqmc_normal(&qmc, scramble, point)?;
                    let primary = self.discounted_payoff(normal)?;
                    if antithetic {
                        let mate = self.discounted_payoff(-normal)?;
                        Ok::<f64, MonteCarloError>((primary + mate) * 0.5)
                    } else {
                        Ok::<f64, MonteCarloError>(primary)
                    }
                })?;
                let mut values = [DeterministicStatistics::default(); PATHWISE_COMPONENTS];
                values[PRICE] = price;
                values
            };
            replicate_values.push(std::array::from_fn(|component| {
                within[component].sum().total() / points as f64
            }));
        }

        let statistics: [DeterministicStatistics; PATHWISE_COMPONENTS] =
            std::array::from_fn(|component| {
                let values = replicate_values
                    .iter()
                    .map(|replicate| replicate[component])
                    .collect::<Vec<_>>();
                DeterministicStatistics::from_ordered_values_two_pass(&values)
            });
        let independent_units = u64::from(engine.scramble_count().get());
        let sampling_variance = if self.total_variance == 0.0 {
            0.0
        } else {
            statistics[PRICE].moments().sample_variance().ok_or(
                MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                },
            )?
        };
        let estimator_variance = sampling_variance / independent_units as f64;
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let risks = self.build_risk_report(
            &statistics,
            independent_units,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let risk_diagnostics = self.build_risk_diagnostics(
            &statistics,
            independent_units,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: ReplayMetadata::new(
                SchemaVersion::CURRENT,
                self.request_fingerprint,
                crate::version(),
                format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            ),
        };
        let multiplier = if antithetic { 2_u128 } else { 1_u128 };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths: u128::from(points)
                * u128::from(engine.scramble_count().get())
                * multiplier,
            diagnostics: MonteCarloDiagnostics {
                master_seed: engine.master_scramble_seed(),
                estimator: EstimatorKind::RandomizedQuasiMonteCarlo,
                scramble_count: Some(engine.scramble_count().get()),
                direction_checksum: Some(qmc.direction_checksum()),
                scramble_checksum: Some(qmc.scramble_checksum()),
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
            },
        })
    }

    fn risk_enabled(&self) -> bool {
        self.request_delta || self.request_gamma.is_some() || self.request_vega
    }

    fn execute_local_vol_pseudo(
        &self,
        engine: PseudoMcConfig,
        local_volatility: &LocalVolRuntime,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let antithetic = engine.variance_reduction().antithetic();
        let brownian_bridge = engine.variance_reduction().brownian_bridge();
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        let statistics = executor.try_map_reduce_statistics_array_tiled(
            engine.independent_sampling_units().get(),
            self.aad_tile_policy.resolved_capacity(),
            |sampling_unit| {
                let shocks = local_volatility.plan.path_shocks(
                    engine.master_seed(),
                    sampling_unit,
                    RandomDomain::Valuation,
                    brownian_bridge,
                )?;
                let primary = self.local_vol_pathwise_values(
                    local_volatility,
                    bump_runtimes.as_ref(),
                    &shocks,
                    PathIndex::new(sampling_unit),
                )?;
                if antithetic {
                    let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                    let mate = self.local_vol_pathwise_values(
                        local_volatility,
                        bump_runtimes.as_ref(),
                        &mate_shocks,
                        PathIndex::new(sampling_unit),
                    )?;
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                        |component| (primary[component] + mate[component]) * 0.5,
                    ))
                } else {
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                }
            },
        )?;
        let independent_units = engine.independent_sampling_units().get();
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let sampling_variance = statistics[PRICE].moments().sample_variance().ok_or(
            MonteCarloError::InsufficientSamplingUnits {
                count: independent_units,
            },
        )?;
        let estimator_variance = sampling_variance / independent_units as f64;
        let pricing_result = PricingResult {
            value: estimate,
            risks: self.build_risk_report(
                &statistics,
                independent_units,
                EstimatorKind::PseudoMonteCarlo,
            )?,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: ReplayMetadata::new(
                SchemaVersion::CURRENT,
                self.request_fingerprint,
                crate::version(),
                format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            ),
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics: self.build_local_vol_risk_diagnostics(),
            independent_sampling_units: independent_units,
            evaluated_paths: engine.evaluated_paths(),
            diagnostics: MonteCarloDiagnostics {
                master_seed: engine.master_seed(),
                estimator: EstimatorKind::PseudoMonteCarlo,
                scramble_count: None,
                direction_checksum: None,
                scramble_checksum: None,
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
            },
        })
    }

    fn local_vol_discounted_payoff(
        &self,
        local_volatility: &LocalVolRuntime,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<f64, MonteCarloError> {
        self.local_vol_discounted_payoff_at_spot(local_volatility, self.spot, shocks, path)
    }

    fn local_vol_discounted_payoff_at_spot(
        &self,
        local_volatility: &LocalVolRuntime,
        spot: f64,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<f64, MonteCarloError> {
        let path = if let (Some(dividends), Some(schedule)) = (
            local_volatility.dividends.as_ref(),
            local_volatility.dividend_schedule.as_ref(),
        ) {
            local_volatility.plan.evolve_path_with_dividend_checks(
                &local_volatility.grid,
                spot,
                shocks,
                dividends,
                schedule,
                path,
            )?
        } else {
            local_volatility
                .plan
                .evolve_path(&local_volatility.grid, self.spot, shocks)?
        };
        let terminal_f =
            path.states()
                .last()
                .copied()
                .ok_or(MonteCarloError::UnsupportedModel {
                    model: "local_volatility",
                })?;
        let terminal = local_volatility.terminal_affine_a * spot
            + local_volatility.terminal_affine_b * terminal_f;
        let outputs = self.payoff.evaluate(|underlying, date| {
            (underlying == self.underlying && date == self.expiry).then_some(terminal)
        })?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(pricing_product::GraphError::NoOutputs)?)
    }

    fn local_vol_bump_runtimes(&self) -> Result<Option<LocalVolBumpRuntimes>, MonteCarloError> {
        if !self.request_delta && self.request_gamma.is_none() {
            return Ok(None);
        }
        let local_volatility =
            self.local_volatility
                .as_ref()
                .ok_or(MonteCarloError::UnsupportedModel {
                    model: "local_volatility",
                })?;
        let spot_bump = self.validation_spot_bump;
        let down_spot = self.spot - spot_bump;
        let up_spot = self.spot + spot_bump;
        let down = self.local_vol_runtime_for_spot(local_volatility, down_spot)?;
        let up = self.local_vol_runtime_for_spot(local_volatility, up_spot)?;
        Ok(Some(LocalVolBumpRuntimes {
            down_spot,
            down,
            up_spot,
            up,
            spot_bump,
        }))
    }

    fn local_vol_runtime_for_spot(
        &self,
        local_volatility: &LocalVolRuntime,
        spot: f64,
    ) -> Result<LocalVolRuntime, MonteCarloError> {
        let spot = PositiveF64::new(spot, "spot").map_err(ResultBuildError::from)?;
        let market_forward = self.market_forward.with_spot(spot)?;
        let forward_evaluation = market_forward.evaluate(self.time)?;
        compile_local_vol_runtime(
            local_volatility.grid.clone(),
            &market_forward,
            self.time,
            forward_evaluation.affine_coordinate.a(),
            forward_evaluation.affine_coordinate.b(),
        )
    }

    fn local_vol_pathwise_values(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let price = self.local_vol_discounted_payoff(local_volatility, shocks, path)?;
        let mut values = [0.0; PATHWISE_COMPONENTS];
        values[PRICE] = price;
        if let Some(bumps) = bump_runtimes {
            let down = self.local_vol_discounted_payoff_at_spot(
                &bumps.down,
                bumps.down_spot,
                shocks,
                path,
            )?;
            let up =
                self.local_vol_discounted_payoff_at_spot(&bumps.up, bumps.up_spot, shocks, path)?;
            let delta = (up - down) / (2.0 * bumps.spot_bump);
            values[DELTA] = delta;
            values[BUMP_DELTA] = delta;
            if self.request_gamma.is_some() {
                let gamma = (up - 2.0 * price + down) / bumps.spot_bump.powi(2);
                values[GAMMA] = gamma;
                values[BUMP_GAMMA] = gamma;
            }
        }
        Ok(values)
    }

    fn execute_local_vol_rqmc(
        &self,
        engine: RqmcConfig,
        local_volatility: &LocalVolRuntime,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let step_count = u32::try_from(local_volatility.plan.time_grid().step_count())
            .map_err(|_| MonteCarloError::RqmcPlan(RqmcPlanError::TableSizeOverflow))?;
        let qmc = RqmcPlan::compile(engine, step_count)?;
        let bridge = if engine.variance_reduction().brownian_bridge() {
            Some(
                BrownianBridgePlan::compile(local_volatility.plan.time_grid().nodes().to_vec(), 1)
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?,
            )
        } else {
            None
        };
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let antithetic = engine.variance_reduction().antithetic();
        let points = engine.points_per_scramble().get();
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        let mut replicate_values: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        for scramble in 0..engine.scramble_count().get() {
            let within = executor.try_map_reduce_statistics_array_tiled(
                points,
                self.aad_tile_policy.resolved_capacity(),
                |point| {
                    let path_index =
                        PathIndex::new(u64::from(scramble).saturating_mul(points) + point);
                    let shocks = local_vol_rqmc_shocks(&qmc, bridge.as_ref(), scramble, point)?;
                    let primary = self.local_vol_pathwise_values(
                        local_volatility,
                        bump_runtimes.as_ref(),
                        &shocks,
                        path_index,
                    )?;
                    if antithetic {
                        let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                        let mate = self.local_vol_pathwise_values(
                            local_volatility,
                            bump_runtimes.as_ref(),
                            &mate_shocks,
                            path_index,
                        )?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                    }
                },
            )?;
            replicate_values.push(std::array::from_fn(|component| {
                within[component].sum().total() / points as f64
            }));
        }

        let statistics: [DeterministicStatistics; PATHWISE_COMPONENTS] =
            std::array::from_fn(|component| {
                let values = replicate_values
                    .iter()
                    .map(|replicate| replicate[component])
                    .collect::<Vec<_>>();
                DeterministicStatistics::from_ordered_values_two_pass(&values)
            });
        let independent_units = u64::from(engine.scramble_count().get());
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let sampling_variance = statistics[PRICE].moments().sample_variance().ok_or(
            MonteCarloError::InsufficientSamplingUnits {
                count: independent_units,
            },
        )?;
        let estimator_variance = sampling_variance / independent_units as f64;
        let pricing_result = PricingResult {
            value: estimate,
            risks: self.build_risk_report(
                &statistics,
                independent_units,
                EstimatorKind::RandomizedQuasiMonteCarlo,
            )?,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: ReplayMetadata::new(
                SchemaVersion::CURRENT,
                self.request_fingerprint,
                crate::version(),
                format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            ),
        };
        let multiplier = if antithetic { 2_u128 } else { 1_u128 };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics: self.build_local_vol_risk_diagnostics(),
            independent_sampling_units: independent_units,
            evaluated_paths: u128::from(points)
                * u128::from(engine.scramble_count().get())
                * multiplier,
            diagnostics: MonteCarloDiagnostics {
                master_seed: engine.master_scramble_seed(),
                estimator: EstimatorKind::RandomizedQuasiMonteCarlo,
                scramble_count: Some(engine.scramble_count().get()),
                direction_checksum: Some(qmc.direction_checksum()),
                scramble_checksum: Some(qmc.scramble_checksum()),
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
            },
        })
    }

    fn normal(&self, generator: &Philox4x32, sampling_unit: u64) -> f64 {
        if self.total_variance == 0.0 {
            0.0
        } else {
            generator.standard_normal(RandomCoordinate::new(
                sampling_unit,
                0,
                RandomDomain::Valuation,
            ))
        }
    }

    fn rqmc_normal(
        &self,
        plan: &RqmcPlan,
        scramble: u32,
        point: u64,
    ) -> Result<f64, MonteCarloError> {
        if self.total_variance == 0.0 {
            return Ok(0.0);
        }
        let probability = plan
            .uniform(scramble, point, 0)
            .expect("scramble, point, and dimension originate from the compiled plan");
        Ok(inverse_standard_normal(probability)
            .expect("the Sobol midpoint mapping is strictly inside the unit interval"))
    }

    fn discounted_payoff(&self, normal: f64) -> Result<f64, pricing_product::GraphError> {
        let log_return = -0.5 * self.total_variance + self.standard_deviation * normal;
        let terminal = self.forward * log_return.exp();
        let outputs = self.payoff.evaluate(|underlying, date| {
            (underlying == self.underlying && date == self.expiry).then_some(terminal)
        })?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(pricing_product::GraphError::NoOutputs)?)
    }

    fn pathwise_values(
        &self,
        normal: f64,
        lane: usize,
        workspace: &mut SoaWorkspace,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let base = self.pathwise_aad(normal, self.spot, self.volatility)?;
        let gamma = if let Some(gamma) = self.request_gamma {
            let bump = resolve_spot_bump(gamma, self.spot);
            let delta_down = self
                .pathwise_aad(normal, self.spot - bump, self.volatility)?
                .delta;
            let delta_up = self
                .pathwise_aad(normal, self.spot + bump, self.volatility)?
                .delta;
            (delta_up - delta_down) / (2.0 * bump)
        } else {
            0.0
        };
        let spot_bump = self.validation_spot_bump;
        let down_spot = self.pathwise_aad(normal, self.spot - spot_bump, self.volatility)?;
        let up_spot = self.pathwise_aad(normal, self.spot + spot_bump, self.volatility)?;
        let bump_delta = (up_spot.price - down_spot.price) / (2.0 * spot_bump);
        let bump_gamma = (up_spot.price - 2.0 * base.price + down_spot.price) / spot_bump.powi(2);
        let bump_vega = if self.validation_volatility_bump == 0.0 {
            0.0
        } else {
            let bump = self.validation_volatility_bump;
            let down = self
                .pathwise_aad(normal, self.spot, self.volatility - bump)?
                .price;
            let up = self
                .pathwise_aad(normal, self.spot, self.volatility + bump)?
                .price;
            (up - down) / (2.0 * bump)
        };
        workspace.primal_mut(0)?.set(lane, base.terminal)?;
        workspace.adjoint_mut(0)?.set(lane, base.terminal_adjoint)?;
        workspace.primal_mut(1)?.set(lane, base.price)?;
        workspace.primal_mut(2)?.set(lane, base.delta)?;
        workspace.primal_mut(3)?.set(lane, base.vega)?;
        workspace.primal_mut(4)?.set(lane, gamma)?;
        Ok([
            workspace.primal(1)?.get(lane)?,
            workspace.primal(2)?.get(lane)?,
            workspace.primal(3)?.get(lane)?,
            workspace.primal(4)?.get(lane)?,
            bump_delta,
            bump_delta - base.delta,
            bump_vega,
            bump_vega - base.vega,
            bump_gamma,
            bump_gamma - gamma,
        ])
    }

    fn pathwise_aad(
        &self,
        normal: f64,
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, pricing_product::GraphError> {
        let total_variance = volatility * volatility * self.time;
        let standard_deviation = total_variance.sqrt();
        let log_return = -0.5 * total_variance + standard_deviation * normal;
        let bumped_forward = self.forward * (spot / self.spot);
        let terminal = bumped_forward * log_return.exp();
        let payoff = self
            .payoff
            .evaluate_single_with_terminal_adjoint(|underlying, date| {
                (underlying == self.underlying && date == self.expiry).then_some(terminal)
            })?;
        let terminal_adjoint = payoff
            .terminal_adjoints
            .iter()
            .filter(|adjoint| {
                adjoint.underlying == self.underlying && adjoint.observation_date == self.expiry
            })
            .fold(0.0, |total, adjoint| total + adjoint.value);
        let price = self.discount * payoff.value;
        let delta = self.discount * terminal_adjoint * terminal / spot;
        let terminal_vega = terminal * (-volatility * self.time + self.time.sqrt() * normal);
        let vega = self.discount * terminal_adjoint * terminal_vega;
        Ok(PathwiseAad {
            price,
            delta,
            vega,
            terminal,
            terminal_adjoint,
        })
    }

    fn build_risk_report(
        &self,
        statistics: &[DeterministicStatistics; PATHWISE_COMPONENTS],
        independent_units: u64,
        estimator: EstimatorKind,
    ) -> Result<RiskReport, MonteCarloError> {
        let delta = self
            .request_delta
            .then(|| {
                risk_estimate(
                    statistics[DELTA],
                    independent_units,
                    self.spot * 0.01,
                    RiskUnit::DeltaRaw,
                    RiskUnit::DeltaOnePercentSpot,
                    estimator,
                )
            })
            .transpose()?;
        let gamma = self
            .request_gamma
            .map(|_| {
                risk_estimate(
                    statistics[GAMMA],
                    independent_units,
                    (self.spot * 0.01).powi(2),
                    RiskUnit::GammaRaw,
                    RiskUnit::GammaOnePercentSpotSquared,
                    estimator,
                )
            })
            .transpose()?;
        let vega = self
            .request_vega
            .then(|| {
                risk_estimate(
                    statistics[VEGA],
                    independent_units,
                    0.01,
                    RiskUnit::VegaRaw,
                    RiskUnit::VegaOneVolPoint,
                    estimator,
                )
            })
            .transpose()?;
        Ok(RiskReport {
            delta,
            gamma,
            vega,
            vega_kt: None,
        })
    }

    fn build_risk_diagnostics(
        &self,
        statistics: &[DeterministicStatistics; PATHWISE_COMPONENTS],
        independent_units: u64,
        estimator: EstimatorKind,
    ) -> Result<RiskDiagnostics, MonteCarloError> {
        let delta_validation = self
            .request_delta
            .then(|| {
                risk_validation(
                    statistics[BUMP_DELTA],
                    statistics[DELTA_DIFFERENCE],
                    independent_units,
                    estimator,
                )
            })
            .transpose()?;
        let gamma_validation = self
            .request_gamma
            .map(|_| {
                risk_validation(
                    statistics[BUMP_GAMMA],
                    statistics[GAMMA_DIFFERENCE],
                    independent_units,
                    estimator,
                )
            })
            .transpose()?;
        let vega_validation = self
            .request_vega
            .then(|| {
                risk_validation(
                    statistics[BUMP_VEGA],
                    statistics[VEGA_DIFFERENCE],
                    independent_units,
                    estimator,
                )
            })
            .transpose()?;
        Ok(RiskDiagnostics {
            methods: RiskMethodMetadata {
                delta: self.request_delta.then_some(RiskMethod::AadReverse),
                gamma: self
                    .request_gamma
                    .map(|_| RiskMethod::CentralBumpOfAadDelta),
                vega: self.request_vega.then_some(RiskMethod::AadReverse),
                smile_dynamics: self.smile_dynamics,
                gamma_spot_bump: self
                    .request_gamma
                    .map(|gamma| resolve_spot_bump(gamma, self.spot)),
                validation_spot_bump: self.risk_enabled().then_some(self.validation_spot_bump),
                validation_volatility_bump: self
                    .request_vega
                    .then_some(self.validation_volatility_bump),
                bump_policy_version: BumpValidationPolicy::VERSION,
            },
            delta_validation,
            gamma_validation,
            vega_validation,
        })
    }

    fn build_local_vol_risk_diagnostics(&self) -> RiskDiagnostics {
        RiskDiagnostics {
            methods: RiskMethodMetadata {
                delta: self.request_delta.then_some(RiskMethod::CentralBump),
                gamma: self.request_gamma.map(|_| RiskMethod::CentralBump),
                vega: None,
                smile_dynamics: self.smile_dynamics,
                gamma_spot_bump: self
                    .request_gamma
                    .map(|gamma| resolve_spot_bump(gamma, self.spot)),
                validation_spot_bump: (self.request_delta || self.request_gamma.is_some())
                    .then_some(self.validation_spot_bump),
                validation_volatility_bump: None,
                bump_policy_version: BumpValidationPolicy::VERSION,
            },
            delta_validation: None,
            gamma_validation: None,
            vega_validation: None,
        }
    }
}

fn build_plan_fingerprint(
    request_fingerprint: [u8; 32],
    payoff_fingerprint: GraphFingerprint,
    execution_policy: ExecutionPolicy,
    aad_tile_policy: AadTilePolicy,
    checkpoint_policy: CheckpointPolicy,
) -> Fingerprint {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pricing/plan\0");
    hasher.update(&PLAN_FINGERPRINT_VERSION.to_be_bytes());
    hasher.update(&request_fingerprint);
    hasher.update(payoff_fingerprint.as_bytes());
    hasher.update(&execution_policy.version().to_be_bytes());
    hasher.update(&execution_policy.worker_threads().get().to_be_bytes());
    hasher.update(&execution_policy.reduction_block_size().get().to_be_bytes());
    hasher.update(&aad_tile_policy.version().to_be_bytes());
    hasher.update(&aad_tile_policy.resolved_capacity().get().to_be_bytes());
    hasher.update(&checkpoint_policy.version().to_be_bytes());
    hasher.update(&checkpoint_policy.resolved_interval().get().to_be_bytes());
    Fingerprint::from_bytes(*hasher.finalize().as_bytes())
}

#[derive(Clone, Copy, Debug)]
struct PathwiseAad {
    price: f64,
    delta: f64,
    vega: f64,
    terminal: f64,
    terminal_adjoint: f64,
}

fn resolve_spot_bump(gamma: GammaConfig, spot: f64) -> f64 {
    match gamma.bump() {
        SpotBump::Absolute(value) => value.get(),
        SpotBump::Relative(value) => spot * value.get(),
    }
}

fn estimate_from_statistics(
    statistics: DeterministicStatistics,
    independent_units: u64,
    scale: f64,
    estimator: EstimatorKind,
) -> Result<Estimate, ResultBuildError> {
    let inverse_count = 1.0 / independent_units as f64;
    let value = statistics.sum().total() * inverse_count * scale;
    let sampling_variance = statistics.moments().sample_variance().unwrap_or(0.0);
    let standard_error = (sampling_variance * inverse_count).sqrt() * scale;
    let half_width = NORMAL_95 * standard_error;
    Estimate::new(
        value,
        standard_error,
        value - half_width,
        value + half_width,
        estimator,
        independent_units,
    )
}

fn risk_estimate(
    statistics: DeterministicStatistics,
    independent_units: u64,
    market_scale: f64,
    raw_unit: RiskUnit,
    market_scaled_unit: RiskUnit,
    estimator: EstimatorKind,
) -> Result<RiskEstimate, ResultBuildError> {
    Ok(RiskEstimate::new(
        estimate_from_statistics(statistics, independent_units, 1.0, estimator)?,
        estimate_from_statistics(statistics, independent_units, market_scale, estimator)?,
        raw_unit,
        market_scaled_unit,
    ))
}

fn risk_validation(
    bump: DeterministicStatistics,
    bump_minus_primary: DeterministicStatistics,
    independent_units: u64,
    estimator: EstimatorKind,
) -> Result<RiskValidation, ResultBuildError> {
    Ok(RiskValidation {
        bump_and_revalue: estimate_from_statistics(bump, independent_units, 1.0, estimator)?,
        bump_minus_primary: estimate_from_statistics(
            bump_minus_primary,
            independent_units,
            1.0,
            estimator,
        )?,
    })
}

fn local_vol_rqmc_shocks(
    qmc: &RqmcPlan,
    bridge: Option<&BrownianBridgePlan>,
    scramble: u32,
    point: u64,
) -> Result<Vec<f64>, MonteCarloError> {
    let mut shocks =
        Vec::with_capacity(usize::try_from(qmc.effective_dimension()).expect("u32 fits usize"));
    for dimension in 0..qmc.effective_dimension() {
        let probability = qmc
            .uniform(scramble, point, dimension)
            .expect("scramble, point, and dimension originate from the compiled plan");
        shocks.push(
            inverse_standard_normal(probability)
                .expect("the Sobol midpoint mapping is strictly inside the unit interval"),
        );
    }
    if let Some(bridge) = bridge {
        bridge
            .apply_one_factor(&shocks)
            .map_err(|error| MonteCarloError::LocalVol(error.into()))
    } else {
        Ok(shocks)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RiskMethod {
    AadReverse,
    CentralBump,
    CentralBumpOfAadDelta,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskMethodMetadata {
    pub delta: Option<RiskMethod>,
    pub gamma: Option<RiskMethod>,
    pub vega: Option<RiskMethod>,
    pub smile_dynamics: SmileDynamics,
    pub gamma_spot_bump: Option<f64>,
    pub validation_spot_bump: Option<f64>,
    pub validation_volatility_bump: Option<f64>,
    pub bump_policy_version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskValidation {
    pub bump_and_revalue: Estimate,
    /// Common-random-number pathwise difference: bump estimator minus the
    /// primary AAD or bumped-AAD estimator.
    pub bump_minus_primary: Estimate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiskDiagnostics {
    pub methods: RiskMethodMetadata,
    pub delta_validation: Option<RiskValidation>,
    pub gamma_validation: Option<RiskValidation>,
    pub vega_validation: Option<RiskValidation>,
}

pub struct BumpValidationPolicy;

impl BumpValidationPolicy {
    pub const VERSION: u32 = 1;
    pub const DEFAULT_RELATIVE_SPOT_BUMP: f64 = DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP;
    pub const DEFAULT_ABSOLUTE_VOLATILITY_BUMP: f64 = DEFAULT_VALIDATION_VOLATILITY_BUMP;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MonteCarloDiagnostics {
    pub master_seed: u64,
    pub estimator: EstimatorKind,
    pub scramble_count: Option<u32>,
    pub direction_checksum: Option<[u8; 32]>,
    pub scramble_checksum: Option<[u8; 32]>,
    pub policy_version: u32,
    pub worker_threads: u32,
    pub reduction_block_size: u64,
    pub aad_tile_policy_version: u32,
    pub aad_tile_capacity: u32,
    pub checkpoint_policy_version: u32,
    pub checkpoint_interval: u32,
    pub antithetic: bool,
    pub discount_region: CurveRegion,
    pub dividend_region: CurveRegion,
    pub payoff_fingerprint: GraphFingerprint,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarloPrice {
    pub pricing_result: PricingResult,
    pub sampling_variance: f64,
    pub estimator_variance: f64,
    pub risk_diagnostics: RiskDiagnostics,
    pub independent_sampling_units: u64,
    pub evaluated_paths: u128,
    pub diagnostics: MonteCarloDiagnostics,
}

pub fn price_pseudo_monte_carlo(
    request: &PricingRequest,
    execution_policy: ExecutionPolicy,
) -> Result<MonteCarloPrice, MonteCarloError> {
    if !matches!(request.engine(), EngineConfig::PseudoMonteCarlo(_)) {
        return Err(MonteCarloError::UnsupportedEngine);
    }
    SimulationPlan::compile(request, execution_policy)?.execute()
}

pub fn price_monte_carlo(
    request: &PricingRequest,
    execution_policy: ExecutionPolicy,
) -> Result<MonteCarloPrice, MonteCarloError> {
    SimulationPlan::compile(request, execution_policy)?.execute()
}

fn extrapolation_warnings(
    discount_region: CurveRegion,
    dividend_region: CurveRegion,
) -> Vec<PricingWarning> {
    let mut warnings = Vec::new();
    if discount_region.is_extrapolated() {
        warnings.push(PricingWarning::new(
            "discount_curve_extrapolation",
            "expiry uses flat-forward discount-curve extrapolation",
        ));
    }
    if dividend_region.is_extrapolated() {
        warnings.push(PricingWarning::new(
            "dividend_curve_extrapolation",
            "expiry uses flat-forward dividend-curve extrapolation",
        ));
    }
    warnings
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pricing_core::{CurrencyId, CurveId, PositiveF64};
    use pricing_market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
    use pricing_mc::{PseudoMcConfig, RqmcConfig, VarianceReduction};
    use pricing_models::{BlackScholesSpec, LocalVolatilitySpec};
    use pricing_product::{EuropeanVanillaSpec, OptionSide};
    use pricing_risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump};

    use super::*;
    use crate::analytical::black_scholes_oracle;

    fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
                .expect("curve"),
        )
    }

    fn request(
        side: OptionSide,
        strike: f64,
        volatility: f64,
        sampling_units: u64,
        antithetic: bool,
    ) -> PricingRequest {
        request_with_spot_and_risk(
            side,
            strike,
            volatility,
            sampling_units,
            antithetic,
            100.0,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn request_with_spot_and_risk(
        side: OptionSide,
        strike: f64,
        volatility: f64,
        sampling_units: u64,
        antithetic: bool,
        spot: f64,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                strike,
                1.0,
                side,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(spot, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.02),
            ),
        ));
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                sampling_units,
                VarianceReduction::new(antithetic, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("request")
    }

    fn all_risks() -> RiskRequest {
        RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(13),
            Some(64),
        )
        .expect("risk request")
    }

    fn rqmc_request(
        scramble_seed: u64,
        points: u64,
        scrambles: u32,
        antithetic: bool,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.02),
            ),
        ));
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                points,
                scrambles,
                scramble_seed,
                VarianceReduction::new(antithetic, true),
            )
            .expect("RQMC engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("request")
    }

    fn policy(workers: u32) -> ExecutionPolicy {
        ExecutionPolicy::new(workers, Some(1024)).expect("policy")
    }

    fn local_vol_price_only_request(
        sampling_units: u64,
        antithetic: bool,
        risk: RiskRequest,
    ) -> PricingRequest {
        price_only_request_with_model(
            ModelSpec::LocalVolatility(
                LocalVolatilitySpec::from_explicit_grid(
                    vec![0.0, 1.0],
                    vec![-1.0, 1.0],
                    vec![0.04, 0.04, 0.04, 0.04],
                    1.0e-8,
                    1.0,
                )
                .expect("local volatility"),
            ),
            sampling_units,
            antithetic,
            risk,
        )
    }

    fn zero_carry_black_scholes_request(sampling_units: u64, antithetic: bool) -> PricingRequest {
        price_only_request_with_model(
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            sampling_units,
            antithetic,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn zero_carry_rqmc_request(model: ModelSpec) -> PricingRequest {
        zero_carry_rqmc_request_with_risk(
            model,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn zero_carry_rqmc_request_with_risk(model: ModelSpec, risk: RiskRequest) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                256,
                16,
                0xfedc_ba98_7654_3210,
                VarianceReduction::new(true, true),
            )
            .expect("RQMC engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("request")
    }

    fn constant_local_vol_model() -> ModelSpec {
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                vec![0.0, 1.0],
                vec![-1.0, 1.0],
                vec![0.04, 0.04, 0.04, 0.04],
                1.0e-8,
                1.0,
            )
            .expect("local volatility"),
        )
    }

    fn price_only_request_with_model(
        model: ModelSpec,
        sampling_units: u64,
        antithetic: bool,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                sampling_units,
                VarianceReduction::new(antithetic, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("request")
    }

    #[test]
    fn mc_converges_to_the_analytical_oracle_with_reported_error() {
        let request = request(OptionSide::Call, 100.0, 0.2, 131_072, true);
        let oracle = black_scholes_oracle(&request).expect("oracle").price;
        let result = price_pseudo_monte_carlo(&request, policy(4)).expect("MC");
        let estimate = result.pricing_result.value;
        let error = (estimate.value().get() - oracle).abs();
        assert!(error <= 6.0 * estimate.standard_error().get());
        assert_eq!(result.independent_sampling_units, 131_072);
        assert_eq!(result.evaluated_paths, 262_144);
        assert_eq!(
            result.estimator_variance.sqrt().to_bits(),
            estimate.standard_error().get().to_bits()
        );
    }

    #[test]
    fn local_vol_price_only_matches_constant_variance_black_scholes_limit() {
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        let local_vol = local_vol_price_only_request(4096, true, risk);
        let black_scholes = zero_carry_black_scholes_request(4096, true);
        let local_result = price_pseudo_monte_carlo(&local_vol, policy(2)).expect("local vol");
        let bs_result = price_pseudo_monte_carlo(&black_scholes, policy(2)).expect("black scholes");
        assert_eq!(
            local_result.pricing_result.value.value().to_bits(),
            bs_result.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            local_result
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits(),
            bs_result
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits()
        );
        assert_eq!(local_result.evaluated_paths, 8192);
    }

    #[test]
    fn local_vol_rqmc_price_only_matches_constant_variance_black_scholes_limit() {
        let local_vol = zero_carry_rqmc_request(constant_local_vol_model());
        let black_scholes = zero_carry_rqmc_request(ModelSpec::BlackScholes(
            BlackScholesSpec::new(0.2).expect("model"),
        ));
        let local_result = price_monte_carlo(&local_vol, policy(2)).expect("local vol");
        let bs_result = price_monte_carlo(&black_scholes, policy(2)).expect("black scholes");
        assert!(
            (local_result.pricing_result.value.value().get()
                - bs_result.pricing_result.value.value().get())
            .abs()
                < 1.0e-12
        );
        assert!(
            (local_result.pricing_result.value.standard_error().get()
                - bs_result.pricing_result.value.standard_error().get())
            .abs()
                < 1.0e-12
        );
        assert_eq!(local_result.independent_sampling_units, 16);
        assert_eq!(local_result.evaluated_paths, 8192);
        assert!(local_result.diagnostics.direction_checksum.is_some());
        assert!(local_result.diagnostics.scramble_checksum.is_some());
    }

    #[test]
    fn local_vol_delta_and_gamma_use_common_random_number_bumps() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            false,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = local_vol_price_only_request(4096, true, risk);
        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("local vol risks");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_none());
        assert_eq!(
            result.risk_diagnostics.methods.delta,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(
            result.risk_diagnostics.methods.gamma,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(result.risk_diagnostics.methods.vega, None);
        assert_eq!(result.risk_diagnostics.delta_validation, None);
        assert_eq!(result.risk_diagnostics.gamma_validation, None);
    }

    #[test]
    fn local_vol_rqmc_delta_and_gamma_use_between_scramble_uncertainty() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            false,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = zero_carry_rqmc_request_with_risk(constant_local_vol_model(), risk);
        let result = price_monte_carlo(&request, policy(2)).expect("local vol rqmc risks");
        let delta = result.pricing_result.risks.delta.expect("delta").raw();
        let gamma = result.pricing_result.risks.gamma.expect("gamma").raw();
        assert_eq!(delta.effective_sampling_units().get(), 16);
        assert_eq!(gamma.effective_sampling_units().get(), 16);
        assert_eq!(
            result.risk_diagnostics.methods.delta,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(
            result.risk_diagnostics.methods.gamma,
            Some(RiskMethod::CentralBump)
        );
    }

    #[test]
    fn local_vol_vega_requests_are_explicitly_unsupported() {
        let request = local_vol_price_only_request(4096, true, all_risks());
        assert!(matches!(
            SimulationPlan::compile(&request, policy(2)),
            Err(MonteCarloError::UnsupportedRiskForModel {
                model: "local_volatility"
            })
        ));
    }

    #[test]
    fn seeded_replay_is_bitwise_equal_across_worker_counts() {
        let request = request(OptionSide::Put, 105.0, 0.35, 10_003, true);
        let single = price_pseudo_monte_carlo(&request, policy(1)).expect("single");
        let parallel = price_pseudo_monte_carlo(&request, policy(4)).expect("parallel");
        assert_eq!(
            single.pricing_result.value.value().to_bits(),
            parallel.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            single.pricing_result.value.standard_error().get().to_bits(),
            parallel
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits()
        );
        assert_eq!(
            single.estimator_variance.to_bits(),
            parallel.estimator_variance.to_bits()
        );
    }

    #[test]
    fn zero_volatility_obeys_exact_forward_discounting() {
        let request = request(OptionSide::Call, 90.0, 0.0, 1, false);
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let expected = plan.discount() * (plan.forward() - 90.0);
        let result = plan.execute().expect("execution");
        assert_eq!(result.pricing_result.value.value().get(), expected);
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
        assert_eq!(result.pricing_result.value.standard_error().get(), 0.0);
    }

    #[test]
    fn higher_call_strike_cannot_increase_seeded_pathwise_price() {
        let low = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 90.0, 0.25, 8192, true),
            policy(3),
        )
        .expect("low strike");
        let high = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 110.0, 0.25, 8192, true),
            policy(3),
        )
        .expect("high strike");
        assert!(low.pricing_result.value.value().get() >= high.pricing_result.value.value().get());
    }

    #[test]
    fn aad_delta_vega_and_bumped_aad_gamma_match_the_oracle() {
        let request = request_with_spot_and_risk(
            OptionSide::Call,
            100.0,
            0.2,
            131_072,
            true,
            100.0,
            all_risks(),
        );
        let oracle = black_scholes_oracle(&request).expect("oracle");
        let result = price_pseudo_monte_carlo(&request, policy(4)).expect("risk MC");
        let risks = &result.pricing_result.risks;
        for (estimate, expected) in [
            (risks.delta.expect("delta").raw(), oracle.delta),
            (risks.gamma.expect("gamma").raw(), oracle.gamma),
            (risks.vega.expect("vega").raw(), oracle.vega),
        ] {
            let error = (estimate.value().get() - expected).abs();
            assert!(error <= 6.0 * estimate.standard_error().get() + 2.0e-5);
        }
        assert_eq!(
            risks.delta.expect("delta").market_scaled().value().get(),
            risks.delta.expect("delta").raw().value().get()
        );
        assert_eq!(result.diagnostics.aad_tile_capacity, 64);
        assert_eq!(result.diagnostics.checkpoint_interval, 13);
        let diagnostics = &result.risk_diagnostics;
        assert_eq!(diagnostics.methods.delta, Some(RiskMethod::AadReverse));
        assert_eq!(
            diagnostics.methods.gamma,
            Some(RiskMethod::CentralBumpOfAadDelta)
        );
        assert_eq!(diagnostics.methods.vega, Some(RiskMethod::AadReverse));
        assert_eq!(diagnostics.methods.gamma_spot_bump, Some(1.0));
        assert_eq!(diagnostics.methods.validation_spot_bump, Some(1.0));
        assert_eq!(diagnostics.methods.bump_policy_version, 1);
        for validation in [
            diagnostics.delta_validation.expect("delta validation"),
            diagnostics.gamma_validation.expect("gamma validation"),
            diagnostics.vega_validation.expect("vega validation"),
        ] {
            assert!(validation.bump_and_revalue.value().get().is_finite());
            assert!(validation.bump_minus_primary.value().get().is_finite());
            assert!(
                validation
                    .bump_minus_primary
                    .standard_error()
                    .get()
                    .is_finite()
            );
        }
    }

    #[test]
    fn adding_risks_does_not_change_seeded_price_bits() {
        let price_only = request(OptionSide::Put, 105.0, 0.35, 10_003, true);
        let with_risks = request_with_spot_and_risk(
            OptionSide::Put,
            105.0,
            0.35,
            10_003,
            true,
            100.0,
            all_risks(),
        );
        let price = price_pseudo_monte_carlo(&price_only, policy(3)).expect("price");
        let risk = price_pseudo_monte_carlo(&with_risks, policy(3)).expect("risk");
        assert_eq!(
            price.pricing_result.value.value().to_bits(),
            risk.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            price.pricing_result.value.standard_error().get().to_bits(),
            risk.pricing_result.value.standard_error().get().to_bits()
        );
    }

    #[test]
    fn price_only_result_has_no_risk_methods_or_validations() {
        let request = request(OptionSide::Call, 100.0, 0.2, 1024, true);
        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("price");
        assert_eq!(result.risk_diagnostics.methods.delta, None);
        assert_eq!(result.risk_diagnostics.methods.gamma, None);
        assert_eq!(result.risk_diagnostics.methods.vega, None);
        assert_eq!(result.risk_diagnostics.delta_validation, None);
        assert_eq!(result.risk_diagnostics.gamma_validation, None);
        assert_eq!(result.risk_diagnostics.vega_validation, None);
    }

    #[test]
    fn risk_replay_is_bitwise_equal_across_worker_counts() {
        let request = request_with_spot_and_risk(
            OptionSide::Call,
            97.0,
            0.31,
            10_003,
            true,
            100.0,
            all_risks(),
        );
        let single = price_pseudo_monte_carlo(&request, policy(1)).expect("single");
        let parallel = price_pseudo_monte_carlo(&request, policy(4)).expect("parallel");
        for (left, right) in [
            (
                single.pricing_result.risks.delta.expect("delta").raw(),
                parallel.pricing_result.risks.delta.expect("delta").raw(),
            ),
            (
                single.pricing_result.risks.gamma.expect("gamma").raw(),
                parallel.pricing_result.risks.gamma.expect("gamma").raw(),
            ),
            (
                single.pricing_result.risks.vega.expect("vega").raw(),
                parallel.pricing_result.risks.vega.expect("vega").raw(),
            ),
        ] {
            assert_eq!(left.value().to_bits(), right.value().to_bits());
            assert_eq!(
                left.standard_error().get().to_bits(),
                right.standard_error().get().to_bits()
            );
        }
    }

    #[test]
    fn aad_delta_and_vega_reconcile_with_common_random_number_bumps() {
        let units = 32_768;
        let base = request_with_spot_and_risk(
            OptionSide::Call,
            100.0,
            0.2,
            units,
            true,
            100.0,
            all_risks(),
        );
        let aad = price_pseudo_monte_carlo(&base, policy(4)).expect("AAD");
        let spot_bump = 0.01;
        let down_spot = price_pseudo_monte_carlo(
            &request_with_spot_and_risk(
                OptionSide::Call,
                100.0,
                0.2,
                units,
                true,
                100.0 - spot_bump,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            ),
            policy(4),
        )
        .expect("down spot");
        let up_spot = price_pseudo_monte_carlo(
            &request_with_spot_and_risk(
                OptionSide::Call,
                100.0,
                0.2,
                units,
                true,
                100.0 + spot_bump,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            ),
            policy(4),
        )
        .expect("up spot");
        let bump_delta = (up_spot.pricing_result.value.value().get()
            - down_spot.pricing_result.value.value().get())
            / (2.0 * spot_bump);
        let volatility_bump = 0.0001;
        let down_vol = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 100.0, 0.2 - volatility_bump, units, true),
            policy(4),
        )
        .expect("down volatility");
        let up_vol = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 100.0, 0.2 + volatility_bump, units, true),
            policy(4),
        )
        .expect("up volatility");
        let bump_vega = (up_vol.pricing_result.value.value().get()
            - down_vol.pricing_result.value.value().get())
            / (2.0 * volatility_bump);
        let risks = &aad.pricing_result.risks;
        assert!((risks.delta.expect("delta").raw().value().get() - bump_delta).abs() < 5.0e-4);
        assert!((risks.vega.expect("vega").raw().value().get() - bump_vega).abs() < 5.0e-3);
    }

    #[test]
    fn gamma_bump_must_leave_a_positive_down_spot() {
        let risk = RiskRequest::new(
            false,
            Some(GammaConfig::new(
                SpotBump::absolute(100.0).expect("positive"),
            )),
            false,
            None,
            SmileDynamics::StickyStrike,
            None,
            None,
        )
        .expect("risk");
        let request =
            request_with_spot_and_risk(OptionSide::Call, 100.0, 0.2, 1024, true, 100.0, risk);
        assert!(matches!(
            SimulationPlan::compile(&request, policy(2)),
            Err(MonteCarloError::InvalidGammaBump { .. })
        ));
    }

    #[test]
    fn rqmc_price_and_error_use_independent_scrambles() {
        let request = rqmc_request(
            0x8877_6655_4433_2211,
            4096,
            16,
            true,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        );
        let oracle = black_scholes_oracle(&request).expect("oracle").price;
        let result = price_monte_carlo(&request, policy(4)).expect("RQMC");
        let estimate = result.pricing_result.value;
        assert_eq!(
            estimate.estimator(),
            EstimatorKind::RandomizedQuasiMonteCarlo
        );
        assert_eq!(estimate.effective_sampling_units().get(), 16);
        assert_eq!(result.independent_sampling_units, 16);
        assert_eq!(result.evaluated_paths, 4096 * 16 * 2);
        assert_eq!(result.diagnostics.scramble_count, Some(16));
        assert!(result.diagnostics.direction_checksum.is_some());
        assert!(result.diagnostics.scramble_checksum.is_some());
        assert!(
            (estimate.value().get() - oracle).abs()
                <= 8.0 * estimate.standard_error().get() + 2.0e-5
        );
    }

    #[test]
    fn rqmc_risks_replay_across_worker_counts() {
        let request = rqmc_request(42, 1024, 8, true, all_risks());
        let single = price_monte_carlo(&request, policy(1)).expect("single");
        let parallel = price_monte_carlo(&request, policy(4)).expect("parallel");
        assert_eq!(single.pricing_result, parallel.pricing_result);
        assert_eq!(
            single.estimator_variance.to_bits(),
            parallel.estimator_variance.to_bits()
        );
        assert_eq!(
            single.diagnostics.scramble_checksum,
            parallel.diagnostics.scramble_checksum
        );
        for estimate in [
            single.pricing_result.risks.delta.expect("delta").raw(),
            single.pricing_result.risks.gamma.expect("gamma").raw(),
            single.pricing_result.risks.vega.expect("vega").raw(),
        ] {
            assert_eq!(
                estimate.estimator(),
                EstimatorKind::RandomizedQuasiMonteCarlo
            );
            assert_eq!(estimate.effective_sampling_units().get(), 8);
        }
    }

    #[test]
    fn rqmc_seed_changes_randomization_but_replays_exactly() {
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        let first = price_monte_carlo(&rqmc_request(1, 1024, 4, false, risk.clone()), policy(2))
            .expect("first");
        let replay = price_monte_carlo(&rqmc_request(1, 1024, 4, false, risk.clone()), policy(2))
            .expect("replay");
        let changed = price_monte_carlo(&rqmc_request(2, 1024, 4, false, risk), policy(2))
            .expect("changed seed");
        assert_eq!(first.pricing_result, replay.pricing_result);
        assert_eq!(
            first.diagnostics.scramble_checksum,
            replay.diagnostics.scramble_checksum
        );
        assert_ne!(
            first.diagnostics.scramble_checksum,
            changed.diagnostics.scramble_checksum
        );
        assert_ne!(
            first.pricing_result.value.value().to_bits(),
            changed.pricing_result.value.value().to_bits()
        );
    }

    #[test]
    fn pseudo_only_entry_point_rejects_rqmc() {
        let request = rqmc_request(
            7,
            8,
            2,
            false,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        );
        assert!(matches!(
            price_pseudo_monte_carlo(&request, policy(1)),
            Err(MonteCarloError::UnsupportedEngine)
        ));
    }
}
