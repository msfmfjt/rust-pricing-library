use pricing_aad::{AadTilePolicy, CheckpointPolicy, SoaWorkspace};
use pricing_core::{Date, DayCountConvention, PathIndex, PositiveF64, SchemaVersion, UnderlyingId};
use pricing_market::{
    AffineDividendTransform, CurveRegion, DiscountCurve, EquityForward, ImpliedVarianceSurface,
    LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing_mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolDividendCheckpointSchedule, LocalVolLogEulerPlan, LocalVolTimeGrid,
    Philox4x32, PseudoMcConfig, RandomCoordinate, RandomDomain, RqmcConfig, RqmcPlan,
    RqmcPlanError, inverse_standard_normal,
};
use pricing_models::{LocalVolatilityReportingBasis, ModelSpec};
use pricing_product::{CompiledPayoff, GraphFingerprint, GraphLimitPolicy};
use pricing_risk::{
    AnalyticCallDensityRow, GammaConfig, ReportingIvBasis, SmileDynamics, SpotBump, VegaKtConfig,
    analytic_call_density_rows_from_surface, local_vega_density_from_node_adjoints,
    project_local_vega_nodes_to_reporting_iv, vega_kt_bucket_estimates,
    vega_kt_full_bucket_covariance, vega_kt_projection_from_parts, vega_kt_report,
};

use crate::{
    Diagnostics, Estimate, EstimatorKind, Fingerprint, MonteCarloError, PricingRequest,
    PricingResult, PricingWarning, ReplayMetadata, ResultBuildError, RiskEstimate, RiskReport,
    RiskUnit, VegaKtResult, fingerprint_request,
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
    payoff: CompiledPayoff,
    observation_dates: Box<[Date]>,
    observation_times: Box<[f64]>,
    observation_forwards: Box<[f64]>,
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
    vega_kt: Option<LocalVolVegaKtRuntime>,
}

#[derive(Clone, Debug)]
struct LocalVolVegaKtRuntime {
    basis: ReportingIvBasis,
    density_rows: Vec<AnalyticCallDensityRow>,
    full_bucket_covariance: bool,
}

#[derive(Clone, Debug)]
struct LocalVolPathwise {
    values: [f64; PATHWISE_COMPONENTS],
    raw_buckets: Option<Vec<f64>>,
}

struct LocalVolRuntimeInputs<'a> {
    grid: LocalVarianceGrid,
    reporting_iv_basis: Option<&'a LocalVolatilityReportingBasis>,
    vega_kt: Option<&'a VegaKtConfig>,
    market_forward: &'a EquityForward,
    valuation_date: Date,
    expiry_time: f64,
    terminal_affine_a: f64,
    terminal_affine_b: f64,
}

struct ReportingIvSurface<'a> {
    basis: &'a LocalVolatilityReportingBasis,
}

impl<'a> ReportingIvSurface<'a> {
    const fn new(basis: &'a LocalVolatilityReportingBasis) -> Self {
        Self { basis }
    }

    fn total_variance_value(&self, time_index: usize, x_index: usize) -> f64 {
        let x_count = self.basis.log_forward_moneyness_nodes().len();
        let maturity = self.basis.maturity_nodes()[time_index];
        let volatility = self.basis.implied_volatilities()[time_index * x_count + x_index];
        maturity * volatility * volatility
    }
}

impl ImpliedVarianceSurface for ReportingIvSurface<'_> {
    fn total_variance_derivatives(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        if !time.is_finite()
            || time < self.basis.maturity_nodes()[0]
            || time > self.basis.maturity_nodes()[self.basis.maturity_nodes().len() - 1]
        {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "time",
                bits: time.to_bits(),
            });
        }
        if !log_moneyness.is_finite() {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "log_moneyness",
                bits: log_moneyness.to_bits(),
            });
        }
        let time_index = lower_cell(self.basis.maturity_nodes(), time);
        let x = log_moneyness
            .max(self.basis.log_forward_moneyness_nodes()[0])
            .min(
                self.basis.log_forward_moneyness_nodes()
                    [self.basis.log_forward_moneyness_nodes().len() - 1],
            );
        let x_index = lower_cell(self.basis.log_forward_moneyness_nodes(), x);
        let time_left = self.basis.maturity_nodes()[time_index];
        let time_right = self.basis.maturity_nodes()[time_index + 1];
        let x_left = self.basis.log_forward_moneyness_nodes()[x_index];
        let x_right = self.basis.log_forward_moneyness_nodes()[x_index + 1];
        let time_weight = interpolation_weight(time_left, time_right, time);
        let x_weight = interpolation_weight(x_left, x_right, x);
        let w00 = self.total_variance_value(time_index, x_index);
        let w01 = self.total_variance_value(time_index, x_index + 1);
        let w10 = self.total_variance_value(time_index + 1, x_index);
        let w11 = self.total_variance_value(time_index + 1, x_index + 1);
        let lower = w00 * (1.0 - x_weight) + w01 * x_weight;
        let upper = w10 * (1.0 - x_weight) + w11 * x_weight;
        let total_variance = lower * (1.0 - time_weight) + upper * time_weight;
        let time_derivative = (upper - lower) / (time_right - time_left);
        let log_moneyness_derivative =
            ((w01 - w00) * (1.0 - time_weight) + (w11 - w10) * time_weight) / (x_right - x_left);
        let log_moneyness_second_derivative = 0.0;
        Ok(TotalVarianceDerivatives {
            total_variance,
            log_moneyness_derivative,
            log_moneyness_second_derivative,
            time_derivative,
            theta: total_variance,
            theta_derivative: time_derivative,
            theta_region: ThetaRegion::Interpolated,
        })
    }
}

fn lower_cell(nodes: &[f64], value: f64) -> usize {
    match nodes.binary_search_by(|node| node.total_cmp(&value)) {
        Ok(index) => index.min(nodes.len() - 2),
        Err(index) => index.saturating_sub(1).min(nodes.len() - 2),
    }
}

fn interpolation_weight(left: f64, right: f64, value: f64) -> f64 {
    if value.to_bits() == right.to_bits() {
        1.0
    } else {
        (value - left) / (right - left)
    }
}

fn average_local_vol_pathwise(
    primary: LocalVolPathwise,
    mate: LocalVolPathwise,
) -> Result<LocalVolPathwise, MonteCarloError> {
    let values =
        std::array::from_fn(|component| (primary.values[component] + mate.values[component]) * 0.5);
    let raw_buckets = match (primary.raw_buckets, mate.raw_buckets) {
        (Some(primary), Some(mate)) => {
            if primary.len() != mate.len() {
                return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis);
            }
            Some(
                primary
                    .into_iter()
                    .zip(mate)
                    .map(|(left, right)| (left + right) * 0.5)
                    .collect(),
            )
        }
        (None, None) => None,
        _ => return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis),
    };
    Ok(LocalVolPathwise {
        values,
        raw_buckets,
    })
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
    inputs: LocalVolRuntimeInputs<'_>,
) -> Result<LocalVolRuntime, MonteCarloError> {
    let LocalVolRuntimeInputs {
        grid,
        reporting_iv_basis,
        vega_kt,
        market_forward,
        valuation_date,
        expiry_time,
        terminal_affine_a,
        terminal_affine_b,
    } = inputs;
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
    let vega_kt = vega_kt
        .map(|config| {
            compile_local_vol_vega_kt_runtime(
                config,
                reporting_iv_basis,
                market_forward,
                valuation_date,
            )
        })
        .transpose()?;
    Ok(LocalVolRuntime {
        grid,
        plan,
        dividends,
        dividend_schedule,
        terminal_affine_a,
        terminal_affine_b,
        vega_kt,
    })
}

fn compile_local_vol_vega_kt_runtime(
    config: &VegaKtConfig,
    reporting_iv_basis: Option<&LocalVolatilityReportingBasis>,
    market_forward: &EquityForward,
    valuation_date: Date,
) -> Result<LocalVolVegaKtRuntime, MonteCarloError> {
    let reporting_iv_basis =
        reporting_iv_basis.ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
    let maturity_nodes = config
        .maturity_nodes()
        .iter()
        .map(|maturity| DayCountConvention::Act365F.year_fraction(valuation_date, *maturity))
        .collect::<Vec<_>>();
    let log_moneyness_nodes = config
        .log_forward_moneyness_nodes()
        .iter()
        .map(|node| node.get())
        .collect::<Vec<_>>();
    if reporting_iv_basis.maturity_nodes() != maturity_nodes
        || reporting_iv_basis.log_forward_moneyness_nodes() != log_moneyness_nodes
    {
        return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis);
    }
    let basis = ReportingIvBasis::new(
        maturity_nodes.clone(),
        log_moneyness_nodes.clone(),
        reporting_iv_basis.implied_volatilities().to_vec(),
    )?;
    let surface = ReportingIvSurface::new(reporting_iv_basis);
    let mut forwards = Vec::with_capacity(maturity_nodes.len());
    for maturity in maturity_nodes.iter().copied() {
        forwards.push(market_forward.evaluate(maturity)?.forward);
    }
    let density_rows = analytic_call_density_rows_from_surface(
        &surface,
        &maturity_nodes,
        &forwards,
        log_moneyness_nodes,
        config.relative_density_threshold().get(),
    )?;
    Ok(LocalVolVegaKtRuntime {
        basis,
        density_rows,
        full_bucket_covariance: config.full_bucket_covariance(),
    })
}

impl SimulationPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let engine = request.engine();
        let product = request.product();
        let time =
            DayCountConvention::Act365F.year_fraction(request.valuation_date(), product.expiry());
        let payment_time = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), product.payment_date());
        let market_forward = request.market().equity().forward();
        let forward_evaluation = market_forward.evaluate(time)?;
        let discount_evaluation = market_forward.discount_curve().evaluate(payment_time)?;
        let discount = discount_evaluation.discount;
        let spot = market_forward.spot().get();
        let (volatility, total_variance, local_volatility) = match request.model() {
            ModelSpec::BlackScholes(model) => {
                let volatility = model.volatility().get();
                (volatility, volatility * volatility * time, None)
            }
            ModelSpec::Black76(model) => {
                let volatility = model.volatility().get();
                (volatility, volatility * volatility * time, None)
            }
            ModelSpec::LocalVolatility(model) => {
                let runtime = compile_local_vol_runtime(LocalVolRuntimeInputs {
                    grid: model.local_variance_grid().clone(),
                    reporting_iv_basis: model.reporting_iv_basis(),
                    vega_kt: request.risk().vega_kt(),
                    market_forward,
                    valuation_date: request.valuation_date(),
                    expiry_time: time,
                    terminal_affine_a: forward_evaluation.affine_coordinate.a(),
                    terminal_affine_b: forward_evaluation.affine_coordinate.b(),
                })?;
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
        let payoff = product
            .source_graph(request.valuation_date())?
            .compile(GraphLimitPolicy::DEFAULT)?;
        let observations = payoff.terminal_observations();
        if observations.is_empty() {
            return Err(MonteCarloError::Graph(
                pricing_product::GraphError::NoOutputs,
            ));
        }
        for (underlying, _) in &observations {
            if *underlying != product.underlying() {
                return Err(MonteCarloError::UnsupportedObservationUnderlying {
                    product: *underlying,
                    market: product.underlying(),
                });
            }
        }
        let observation_dates = observations
            .iter()
            .map(|(_, date)| *date)
            .collect::<Vec<_>>();
        let mut observation_times = Vec::with_capacity(observation_dates.len());
        let mut observation_forwards = Vec::with_capacity(observation_dates.len());
        for date in &observation_dates {
            let observation_time =
                DayCountConvention::Act365F.year_fraction(request.valuation_date(), *date);
            observation_times.push(observation_time);
            observation_forwards.push(market_forward.evaluate(observation_time)?.forward);
        }
        if local_volatility.is_some()
            && observation_dates
                .iter()
                .any(|observation_date| *observation_date != product.expiry())
        {
            return Err(MonteCarloError::UnsupportedModel {
                model: "local_volatility_with_non_terminal_observations",
            });
        }
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
            payoff,
            observation_dates: observation_dates.into_boxed_slice(),
            observation_times: observation_times.into_boxed_slice(),
            observation_forwards: observation_forwards.into_boxed_slice(),
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
            discount_region: discount_evaluation.region,
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
                    let normals = self.normals(&generator, sampling_unit);
                    let primary = self.discounted_payoff_from_normals(&normals)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.discounted_payoff_from_normals(&mate_normals)?;
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
            None,
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
        let qmc = RqmcPlan::compile(
            engine,
            u32::try_from(self.observation_times.len())
                .map_err(|_| MonteCarloError::RqmcPlan(RqmcPlanError::TableSizeOverflow))?,
        )?;
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
                    let normals = self.rqmc_normals(&qmc, scramble, point)?;
                    let primary = self.discounted_payoff_from_normals(&normals)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.discounted_payoff_from_normals(&mate_normals)?;
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
            None,
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
        if local_volatility.vega_kt.is_some() {
            return self.execute_local_vol_pseudo_with_vega_kt(
                engine,
                local_volatility,
                bump_runtimes.as_ref(),
            );
        }
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
                None,
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

    fn execute_local_vol_pseudo_with_vega_kt(
        &self,
        engine: PseudoMcConfig,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let antithetic = engine.variance_reduction().antithetic();
        let brownian_bridge = engine.variance_reduction().brownian_bridge();
        let independent_units = engine.independent_sampling_units().get();
        let bucket_count = vega_kt.basis.bucket_count();
        let mut values = Vec::with_capacity(
            usize::try_from(independent_units).expect("sampling-unit count fits usize"),
        );
        let mut price_samples = Vec::with_capacity(values.capacity());
        let mut raw_bucket_samples = Vec::with_capacity(values.capacity() * bucket_count);
        for sampling_unit in 0..independent_units {
            let shocks = local_volatility.plan.path_shocks(
                engine.master_seed(),
                sampling_unit,
                RandomDomain::Valuation,
                brownian_bridge,
            )?;
            let primary = self.local_vol_pathwise_values_and_buckets(
                local_volatility,
                bump_runtimes,
                &shocks,
                PathIndex::new(sampling_unit),
            )?;
            let pathwise = if antithetic {
                let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                let mate = self.local_vol_pathwise_values_and_buckets(
                    local_volatility,
                    bump_runtimes,
                    &mate_shocks,
                    PathIndex::new(sampling_unit),
                )?;
                average_local_vol_pathwise(primary, mate)?
            } else {
                primary
            };
            price_samples.push(pathwise.values[PRICE]);
            raw_bucket_samples.extend(
                pathwise
                    .raw_buckets
                    .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?,
            );
            values.push(pathwise.values);
        }

        let statistics: [DeterministicStatistics; PATHWISE_COMPONENTS] =
            std::array::from_fn(|component| {
                let component_values = values
                    .iter()
                    .map(|pathwise| pathwise[component])
                    .collect::<Vec<_>>();
                DeterministicStatistics::from_ordered_values_two_pass(&component_values)
            });
        let estimates =
            vega_kt_bucket_estimates(&price_samples, &raw_bucket_samples, bucket_count)?;
        let raw_bucket_means = estimates
            .iter()
            .map(|estimate| estimate.raw_mean())
            .collect::<Vec<_>>();
        let mean_vega = statistics[VEGA].sum().total() / independent_units as f64;
        let bucket_sum = raw_bucket_means
            .iter()
            .copied()
            .collect::<pricing_numerics::NeumaierSum>()
            .total();
        let projection = vega_kt_projection_from_parts(
            raw_bucket_means,
            mean_vega - bucket_sum,
            mean_vega,
            Default::default(),
        )?;
        let full_bucket_covariance = if vega_kt.full_bucket_covariance {
            Some(vega_kt_full_bucket_covariance(
                &raw_bucket_samples,
                bucket_count,
            )?)
        } else {
            None
        };
        let density_row = vega_kt
            .density_rows
            .last()
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        let vega_kt = VegaKtResult::try_from(&vega_kt_report(
            &vega_kt.basis,
            density_row,
            projection,
            estimates,
            full_bucket_covariance,
        )?)?;
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
                Some(vega_kt),
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
                .evolve_path(&local_volatility.grid, spot, shocks)?
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
        compile_local_vol_runtime(LocalVolRuntimeInputs {
            grid: local_volatility.grid.clone(),
            reporting_iv_basis: None,
            vega_kt: None,
            market_forward: &market_forward,
            valuation_date: self.valuation_date,
            expiry_time: self.time,
            terminal_affine_a: forward_evaluation.affine_coordinate.a(),
            terminal_affine_b: forward_evaluation.affine_coordinate.b(),
        })
    }

    fn local_vol_pathwise_values(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let pathwise = self.local_vol_pathwise_values_and_buckets(
            local_volatility,
            bump_runtimes,
            shocks,
            path,
        )?;
        Ok(pathwise.values)
    }

    fn local_vol_pathwise_values_and_buckets(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<LocalVolPathwise, MonteCarloError> {
        let path_state = if let (Some(dividends), Some(schedule)) = (
            local_volatility.dividends.as_ref(),
            local_volatility.dividend_schedule.as_ref(),
        ) {
            local_volatility.plan.evolve_path_with_dividend_checks(
                &local_volatility.grid,
                self.spot,
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
            path_state
                .states()
                .last()
                .copied()
                .ok_or(MonteCarloError::UnsupportedModel {
                    model: "local_volatility",
                })?;
        let terminal = local_volatility.terminal_affine_a * self.spot
            + local_volatility.terminal_affine_b * terminal_f;
        let payoff = self
            .payoff
            .evaluate_single_with_terminal_adjoint(|underlying, date| {
                (underlying == self.underlying && date == self.expiry).then_some(terminal)
            })?;
        let terminal_spot_adjoint = payoff
            .terminal_adjoints
            .iter()
            .filter(|adjoint| {
                adjoint.underlying == self.underlying && adjoint.observation_date == self.expiry
            })
            .fold(0.0, |total, adjoint| total + adjoint.value);
        let price = self.discount * payoff.value;
        let mut values = [0.0; PATHWISE_COMPONENTS];
        let mut raw_buckets = None;
        values[PRICE] = price;
        if self.request_vega || local_volatility.vega_kt.is_some() {
            let terminal_f_adjoint =
                self.discount * terminal_spot_adjoint * local_volatility.terminal_affine_b;
            let reverse = path_state.reverse_terminal(
                terminal_f_adjoint,
                local_volatility.grid.values().len(),
                local_volatility.grid.log_moneyness_nodes().len(),
            )?;
            let local_vol_node_adjoints = reverse
                .local_variance_value_adjoints()
                .iter()
                .zip(local_volatility.grid.values())
                .map(|(adjoint, variance)| adjoint * 2.0 * variance.sqrt())
                .collect::<Vec<_>>();
            let vega = local_vol_node_adjoints
                .iter()
                .copied()
                .collect::<pricing_numerics::NeumaierSum>()
                .total();
            values[VEGA] = vega;
            if let Some(vega_kt) = &local_volatility.vega_kt {
                let x_count = local_volatility.grid.log_moneyness_nodes().len();
                let mut bucket_values = vec![0.0; vega_kt.basis.bucket_count()];
                for (time_index, maturity) in local_volatility
                    .grid
                    .time_nodes()
                    .iter()
                    .copied()
                    .enumerate()
                {
                    if maturity == 0.0 {
                        continue;
                    }
                    let row_start = time_index * x_count;
                    let row_end = row_start + x_count;
                    let density = local_vega_density_from_node_adjoints(
                        &local_vol_node_adjoints[row_start..row_end],
                        local_volatility.grid.log_moneyness_nodes(),
                    )?;
                    let density_row = vega_kt
                        .density_rows
                        .iter()
                        .find(|row| row.maturity().get().to_bits() == maturity.to_bits())
                        .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
                    let projection = project_local_vega_nodes_to_reporting_iv(
                        &vega_kt.basis,
                        maturity,
                        local_volatility.grid.log_moneyness_nodes(),
                        &density,
                        density_row.active_domain(),
                    )?;
                    for (bucket, projected) in
                        bucket_values.iter_mut().zip(projection.raw_buckets())
                    {
                        *bucket += *projected;
                    }
                }
                raw_buckets = Some(bucket_values);
            }
        }
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
        Ok(LocalVolPathwise {
            values,
            raw_buckets,
        })
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
        if local_volatility.vega_kt.is_some() {
            return self.execute_local_vol_rqmc_with_vega_kt(
                engine,
                local_volatility,
                bump_runtimes.as_ref(),
                &qmc,
                bridge.as_ref(),
            );
        }
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
                None,
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

    fn execute_local_vol_rqmc_with_vega_kt(
        &self,
        engine: RqmcConfig,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        qmc: &RqmcPlan,
        bridge: Option<&BrownianBridgePlan>,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let antithetic = engine.variance_reduction().antithetic();
        let points = engine.points_per_scramble().get();
        let bucket_count = vega_kt.basis.bucket_count();
        let mut replicate_values: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        let mut price_samples = Vec::with_capacity(replicate_values.capacity());
        let mut raw_bucket_samples = Vec::with_capacity(replicate_values.capacity() * bucket_count);

        for scramble in 0..engine.scramble_count().get() {
            let mut component_sums = [pricing_numerics::NeumaierSum::new(); PATHWISE_COMPONENTS];
            let mut bucket_sums = vec![pricing_numerics::NeumaierSum::new(); bucket_count];
            for point in 0..points {
                let path_index = PathIndex::new(u64::from(scramble).saturating_mul(points) + point);
                let shocks = local_vol_rqmc_shocks(qmc, bridge, scramble, point)?;
                let primary = self.local_vol_pathwise_values_and_buckets(
                    local_volatility,
                    bump_runtimes,
                    &shocks,
                    path_index,
                )?;
                let pathwise = if antithetic {
                    let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                    let mate = self.local_vol_pathwise_values_and_buckets(
                        local_volatility,
                        bump_runtimes,
                        &mate_shocks,
                        path_index,
                    )?;
                    average_local_vol_pathwise(primary, mate)?
                } else {
                    primary
                };
                for (sum, value) in component_sums.iter_mut().zip(pathwise.values) {
                    sum.add(value);
                }
                for (sum, value) in bucket_sums.iter_mut().zip(
                    pathwise
                        .raw_buckets
                        .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?,
                ) {
                    sum.add(value);
                }
            }
            let replicate =
                std::array::from_fn(|component| component_sums[component].total() / points as f64);
            price_samples.push(replicate[PRICE]);
            raw_bucket_samples.extend(
                bucket_sums
                    .into_iter()
                    .map(|sum| sum.total() / points as f64),
            );
            replicate_values.push(replicate);
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
        let estimates =
            vega_kt_bucket_estimates(&price_samples, &raw_bucket_samples, bucket_count)?;
        let raw_bucket_means = estimates
            .iter()
            .map(|estimate| estimate.raw_mean())
            .collect::<Vec<_>>();
        let mean_vega = statistics[VEGA].sum().total() / independent_units as f64;
        let bucket_sum = raw_bucket_means
            .iter()
            .copied()
            .collect::<pricing_numerics::NeumaierSum>()
            .total();
        let projection = vega_kt_projection_from_parts(
            raw_bucket_means,
            mean_vega - bucket_sum,
            mean_vega,
            Default::default(),
        )?;
        let full_bucket_covariance = if vega_kt.full_bucket_covariance {
            Some(vega_kt_full_bucket_covariance(
                &raw_bucket_samples,
                bucket_count,
            )?)
        } else {
            None
        };
        let density_row = vega_kt
            .density_rows
            .last()
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        let vega_kt = VegaKtResult::try_from(&vega_kt_report(
            &vega_kt.basis,
            density_row,
            projection,
            estimates,
            full_bucket_covariance,
        )?)?;
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
                Some(vega_kt),
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

    fn normals(&self, generator: &Philox4x32, sampling_unit: u64) -> Vec<f64> {
        (0..self.observation_times.len())
            .map(|dimension| {
                if self.total_variance == 0.0 {
                    0.0
                } else {
                    generator.standard_normal(RandomCoordinate::new(
                        sampling_unit,
                        u32::try_from(dimension).expect("observation dimension fits u32"),
                        RandomDomain::Valuation,
                    ))
                }
            })
            .collect()
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

    fn rqmc_normals(
        &self,
        plan: &RqmcPlan,
        scramble: u32,
        point: u64,
    ) -> Result<Vec<f64>, MonteCarloError> {
        if self.total_variance == 0.0 {
            return Ok(vec![0.0; self.observation_times.len()]);
        }
        (0..self.observation_times.len())
            .map(|dimension| {
                let probability = plan
                    .uniform(
                        scramble,
                        point,
                        u32::try_from(dimension).expect("observation dimension fits u32"),
                    )
                    .expect("scramble, point, and dimension originate from the compiled plan");
                Ok(inverse_standard_normal(probability)
                    .expect("the Sobol midpoint mapping is strictly inside the unit interval"))
            })
            .collect()
    }

    fn discounted_payoff_from_normals(
        &self,
        normals: &[f64],
    ) -> Result<f64, pricing_product::GraphError> {
        let spots = self.spots_from_normals(normals);
        let outputs = self.payoff.evaluate(|underlying, date| {
            if underlying != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|observation_date| *observation_date == date)
                .map(|index| spots[index])
        })?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(pricing_product::GraphError::NoOutputs)?)
    }

    fn spots_from_normals(&self, normals: &[f64]) -> Vec<f64> {
        let mut previous_time = 0.0;
        let mut brownian = 0.0;
        self.observation_times
            .iter()
            .zip(self.observation_forwards.iter())
            .zip(normals.iter())
            .map(|((&time, &forward), &normal)| {
                let step = (time - previous_time).max(0.0);
                brownian += step.sqrt() * normal;
                previous_time = time;
                let total_variance = self.volatility * self.volatility * time;
                forward * (-0.5 * total_variance + self.volatility * brownian).exp()
            })
            .collect()
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
        vega_kt: Option<VegaKtResult>,
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
            vega_kt,
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
                vega: self.request_vega.then_some(RiskMethod::AadReverse),
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
    use pricing_models::{Black76Spec, BlackScholesSpec, LocalVolatilitySpec};
    use pricing_product::{
        ArithmeticAsianSpec, AsianObservation, BarrierDirection, BarrierSpec, BarrierStyle,
        DigitalPayout, DigitalSpec, EuropeanVanillaSpec, FixedLookbackSpec, OptionSide,
        ProductSpec,
    };
    use pricing_risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};

    use super::*;
    use crate::analytical::{black_76_oracle, black_scholes_oracle};

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

    fn zero_carry_black_76_request(sampling_units: u64, antithetic: bool) -> PricingRequest {
        price_only_request_with_model(
            ModelSpec::Black76(Black76Spec::new(0.2).expect("model")),
            sampling_units,
            antithetic,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn digital_zero_vol_request(
        side: OptionSide,
        strike: f64,
        payout_kind: DigitalPayout,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::Digital(
            DigitalSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                strike,
                10.0,
                side,
                payout_kind,
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn barrier_zero_vol_request(barrier: f64) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                barrier,
                2.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                vec![
                    "2027-03-05".parse().expect("first"),
                    "2027-09-04".parse().expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn asian_zero_vol_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                2.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown("2027-03-05".parse().expect("first"), 0.25)
                        .expect("first"),
                    AsianObservation::unknown("2027-09-04".parse().expect("second"), 0.75)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn lookback_zero_vol_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                100.0,
                2.0,
                OptionSide::Call,
                vec![
                    "2027-03-05".parse().expect("first"),
                    "2027-09-04".parse().expect("second"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
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

    fn constant_local_vol_model_with_reporting_basis() -> ModelSpec {
        let valuation: Date = "2026-09-04".parse().expect("valuation");
        let first_maturity: Date = "2027-03-05".parse().expect("first maturity");
        let second_maturity: Date = "2027-09-04".parse().expect("second maturity");
        let maturity_nodes = vec![
            DayCountConvention::Act365F.year_fraction(valuation, first_maturity),
            DayCountConvention::Act365F.year_fraction(valuation, second_maturity),
        ];
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                vec![0.0, maturity_nodes[0], maturity_nodes[1]],
                vec![-1.0, 0.0, 1.0],
                vec![0.04; 9],
                1.0e-8,
                1.0,
            )
            .expect("local volatility")
            .with_reporting_iv_basis(
                LocalVolatilityReportingBasis::new(
                    maturity_nodes,
                    vec![-1.0, 0.0, 1.0],
                    vec![0.2; 6],
                )
                .expect("reporting basis"),
            ),
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
    fn black_76_mc_converges_to_the_analytical_oracle_with_reported_error() {
        let request = zero_carry_black_76_request(131_072, true);
        let oracle = black_76_oracle(&request).expect("oracle").price;
        let result = price_pseudo_monte_carlo(&request, policy(4)).expect("MC");
        let estimate = result.pricing_result.value;
        let error = (estimate.value().get() - oracle).abs();
        assert!(error <= 6.0 * estimate.standard_error().get());
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
    fn local_vol_vega_and_vega_kt_are_reported() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            Some(
                VegaKtConfig::new(
                    vec![
                        "2027-03-05".parse().expect("first maturity"),
                        "2027-09-04".parse().expect("second maturity"),
                    ],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    true,
                )
                .expect("vega kt"),
            ),
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = price_only_request_with_model(
            constant_local_vol_model_with_reporting_basis(),
            1024,
            true,
            risk,
        );
        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("local vol vega kt");
        let vega = result.pricing_result.risks.vega.expect("vega").raw();
        assert!(vega.value().get().is_finite());
        let report = result.pricing_result.risks.vega_kt.expect("vega kt");
        assert_eq!(report.coordinates().len(), 6);
        assert_eq!(report.estimates().len(), 6);
        assert_eq!(report.raw_buckets().len(), 6);
        assert_eq!(
            report.full_bucket_covariance().expect("covariance").len(),
            36
        );
        assert!(report.projection().scalar_vega().get().is_finite());
        assert_eq!(
            result.risk_diagnostics.methods.vega,
            Some(RiskMethod::AadReverse)
        );
    }

    #[test]
    fn local_vol_rqmc_vega_kt_uses_between_scramble_bucket_uncertainty() {
        let risk = RiskRequest::new(
            false,
            None,
            true,
            Some(
                VegaKtConfig::new(
                    vec![
                        "2027-03-05".parse().expect("first maturity"),
                        "2027-09-04".parse().expect("second maturity"),
                    ],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    false,
                )
                .expect("vega kt"),
            ),
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = zero_carry_rqmc_request_with_risk(
            constant_local_vol_model_with_reporting_basis(),
            risk,
        );
        let result = price_monte_carlo(&request, policy(2)).expect("local vol rqmc vega kt");
        let vega = result.pricing_result.risks.vega.expect("vega").raw();
        assert_eq!(vega.effective_sampling_units().get(), 16);
        let report = result.pricing_result.risks.vega_kt.expect("vega kt");
        assert!(report.estimates()[0].raw_mean().get().is_finite());
        assert!(report.full_bucket_covariance().is_none());
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
    fn digital_zero_volatility_obeys_exact_indicator_payoff_discounting() {
        let cash_request = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Cash);
        let cash_plan = SimulationPlan::compile(&cash_request, policy(2)).expect("cash plan");
        let cash_expected = cash_plan.discount() * 10.0;
        let cash_result = cash_plan.execute().expect("cash execution");
        assert_eq!(
            cash_result.pricing_result.value.value().get(),
            cash_expected
        );
        assert_eq!(cash_result.sampling_variance.to_bits(), 0.0_f64.to_bits());

        let asset_request = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Asset);
        let asset_plan = SimulationPlan::compile(&asset_request, policy(2)).expect("asset plan");
        let asset_expected = asset_plan.discount() * asset_plan.forward() * 10.0;
        let asset_result = asset_plan.execute().expect("asset execution");
        assert!(
            (asset_result.pricing_result.value.value().get() - asset_expected).abs() <= 1.0e-12
        );
        assert_eq!(asset_result.sampling_variance.to_bits(), 0.0_f64.to_bits());

        let out_request = digital_zero_vol_request(OptionSide::Put, 100.0, DigitalPayout::Cash);
        let out_result = price_pseudo_monte_carlo(&out_request, policy(2)).expect("out");
        assert_eq!(out_result.pricing_result.value.value().get(), 0.0);
        assert_eq!(out_result.sampling_variance.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn barrier_zero_volatility_uses_declared_monitoring_knock_out() {
        let live =
            SimulationPlan::compile(&barrier_zero_vol_request(200.0), policy(2)).expect("live");
        assert_eq!(live.observation_dates.len(), 2);
        let live_expected = live.discount() * (live.observation_forwards[1] - 100.0) * 2.0;
        let live_result = live.execute().expect("live execution");
        assert!((live_result.pricing_result.value.value().get() - live_expected).abs() <= 1.0e-12);
        assert_eq!(live_result.sampling_variance.to_bits(), 0.0_f64.to_bits());

        let knocked =
            SimulationPlan::compile(&barrier_zero_vol_request(50.0), policy(2)).expect("knocked");
        let knocked_result = knocked.execute().expect("knocked execution");
        assert_eq!(
            knocked_result.pricing_result.value.value().get().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            knocked_result.sampling_variance.to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn arithmetic_asian_zero_volatility_uses_all_declared_observation_dates() {
        let request = asian_zero_vol_request();
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        assert_eq!(plan.observation_dates.len(), 2);
        let average = 0.25 * plan.observation_forwards[0] + 0.75 * plan.observation_forwards[1];
        let expected = plan.discount() * (average - 100.0) * 2.0;
        let result = plan.execute().expect("execution");
        assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn fixed_lookback_zero_volatility_uses_declared_monitoring_extremum() {
        let request = lookback_zero_vol_request();
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        assert_eq!(plan.observation_dates.len(), 2);
        let maximum = plan
            .observation_forwards
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let expected = plan.discount() * (maximum - 100.0) * 2.0;
        let result = plan.execute().expect("execution");
        assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
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
