use pricing_aad::{AadTilePolicy, CheckpointPolicy, SoaWorkspace};
use pricing_core::{Date, DayCountConvention, PathIndex, PositiveF64, SchemaVersion, UnderlyingId};
use pricing_market::{
    AffineDividendCoordinate, AffineDividendTransform, CurveRegion, DiscountCurve, EquityForward,
    ImpliedVarianceSurface, LocalVarianceGrid, LocalVarianceInterpolation, MarketError,
    ThetaRegion, TotalVarianceDerivatives,
};
use pricing_mc::{
    BARRIER_BRIDGE_ABI, BarrierBridgeDirection, BarrierBridgeError, BarrierBridgeIntervalInput,
    BarrierBridgePath, BarrierBridgeStatus, BrownianBridgePlan, DeterministicExecutor,
    DeterministicStatistics, EngineConfig, ExecutionPolicy, LocalVolDividendCheckpointSchedule,
    LocalVolLogEulerPlan, LocalVolPath, LocalVolTimeGrid, Philox4x32, PseudoMcConfig,
    RandomCoordinate, RandomDomain, RqmcConfig, RqmcPlan, RqmcPlanError,
    SmoothedBarrierBridgeEndpoint, SmoothedBarrierBridgeEndpointInput,
    SmoothedBarrierBridgeInterval, SmoothedBarrierBridgeIntervalAdjoints,
    SmoothedBarrierBridgeIntervalInput, inverse_standard_normal, transformed_barrier,
};
use pricing_models::{LocalVolatilityReportingBasis, ModelSpec};
use pricing_product::{
    AsianObservationValue, BarrierDirection, BarrierMonitoring, BarrierStyle, CompactC2Smoothing,
    CompiledPayoff, GraphFingerprint, GraphLimitPolicy, OptionSide, ProductSpec,
};
use pricing_risk::{
    AnalyticCallDensityRow, GammaConfig, PayoffSmoothing, ReportingIvBasis, SmileDynamics,
    SpotBump, VegaKtConfig, analytic_call_density_rows_from_surface,
    local_vega_density_from_node_adjoints, project_local_vega_nodes_to_reporting_iv,
    vega_kt_bucket_estimates, vega_kt_full_bucket_covariance, vega_kt_projection_from_parts,
    vega_kt_report,
};

use crate::{
    Diagnostics, Estimate, EstimatorKind, Fingerprint, MigrationProvenance, MonteCarloError,
    PricingRequest, PricingResult, PricingWarning, ReplayMetadata, ResultBuildError, RiskEstimate,
    RiskReport, RiskUnit, VegaKtResult, fingerprint_request,
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
const BARRIER_ENDPOINT_HIT: usize = 10;
const BARRIER_DIVIDEND_JUMP_HIT: usize = 11;
const BARRIER_BRIDGE_HIT_WEIGHT: usize = 12;
const BARRIER_INTERVAL_COUNT: usize = 13;
const BARRIER_FINITE_CORRECTION_COUNT: usize = 14;
const BARRIER_ZERO_VARIANCE_COUNT: usize = 15;
const BARRIER_SURVIVAL_UNDERFLOW_COUNT: usize = 16;
const BARRIER_CERTAIN_SURVIVAL_COUNT: usize = 17;
const PATHWISE_COMPONENTS: usize = 18;
const AAD_WORKSPACE_SLOTS: usize = 5;
const DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP: f64 = 1.0e-4;
const DEFAULT_VALIDATION_VOLATILITY_BUMP: f64 = 1.0e-4;
const PLAN_FINGERPRINT_VERSION: u32 = 1;
const SMOOTHED_BARRIER_BRIDGE_ABI: &str = "continuous-barrier-bridge-log-survival-v2";

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
    observation_dates: Box<[Option<Date>]>,
    observation_times: Box<[f64]>,
    observation_forwards: Box<[f64]>,
    observation_affine_coordinates: Box<[AffineDividendCoordinate]>,
    observation_pre_dividend_coordinates: Box<[Option<AffineDividendCoordinate>]>,
    observation_local_vol_node_indices: Option<Box<[usize]>>,
    engine: EngineConfig,
    execution_policy: ExecutionPolicy,
    aad_tile_policy: AadTilePolicy,
    checkpoint_policy: CheckpointPolicy,
    request_delta: bool,
    request_gamma: Option<GammaConfig>,
    request_vega: bool,
    payoff_smoothing: Option<PayoffSmoothing>,
    payoff_smoothing_endpoint_count: u32,
    payoff_smoothing_dividend_jump_count: u32,
    path_state_diagnostics: Option<PathStateDiagnostics>,
    smile_dynamics: SmileDynamics,
    validation_spot_bump: f64,
    validation_volatility_bump: f64,
    request_fingerprint: [u8; 32],
    request_migration: MigrationProvenance,
    plan_fingerprint: Fingerprint,
    discount_region: CurveRegion,
    dividend_region: CurveRegion,
    market_forward: EquityForward,
    local_volatility: Option<LocalVolRuntime>,
    continuous_barrier: Option<ContinuousBarrierRuntime>,
}

#[derive(Clone, Debug)]
struct LocalVolRuntime {
    grid: LocalVarianceGrid,
    plan: LocalVolLogEulerPlan,
    node_affine_coordinates: Box<[AffineDividendCoordinate]>,
    node_pre_dividend_coordinates: Box<[Option<AffineDividendCoordinate>]>,
    dividends: Option<AffineDividendTransform>,
    dividend_schedule: Option<LocalVolDividendCheckpointSchedule>,
    vega_kt: Option<LocalVolVegaKtRuntime>,
}

#[derive(Clone, Debug)]
struct LocalVolVegaKtRuntime {
    basis: ReportingIvBasis,
    density_rows: Vec<AnalyticCallDensityRow>,
    full_bucket_covariance: bool,
}

#[derive(Clone, Debug)]
struct ContinuousBarrierRuntime {
    strike: f64,
    barrier: f64,
    notional: f64,
    rebate: f64,
    side: OptionSide,
    direction: BarrierDirection,
    style: BarrierStyle,
    monitoring_end_time: f64,
    bridge_observation_indices: Box<[usize]>,
    expiry_observation_index: usize,
}

#[derive(Clone, Debug)]
struct ExactContinuousBarrierBridgeEvaluation {
    path: BarrierBridgePath,
    interval_observation_indices: Box<[(Option<usize>, usize)]>,
    endpoint_touched: bool,
    dividend_jump_touched: bool,
}

impl ExactContinuousBarrierBridgeEvaluation {
    fn survival(&self) -> f64 {
        if self.endpoint_touched || self.dividend_jump_touched {
            0.0
        } else {
            self.path.survival()
        }
    }

    fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        BarrierPathDiagnosticValues::from_bridge(
            &self.path,
            self.endpoint_touched,
            self.dividend_jump_touched,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SmoothedBarrierHitKind {
    Endpoint,
    DividendJump,
}

#[derive(Clone, Copy, Debug)]
struct SmoothedBarrierHitFactor {
    kind: SmoothedBarrierHitKind,
    state_index: Option<usize>,
    hit_weight: f64,
    state_derivative: f64,
}

#[derive(Clone, Debug)]
struct SmoothedContinuousBarrierPath {
    intervals: Box<[SmoothedBarrierBridgeInterval]>,
    hit_factors: Box<[SmoothedBarrierHitFactor]>,
    log_bridge_survival: f64,
    log_hit_survival: f64,
    finite_correction_count: u32,
    zero_variance_count: u32,
    survival_underflow_count: u32,
    certain_survival_count: u32,
}

impl SmoothedContinuousBarrierPath {
    fn evaluate(
        intervals: Vec<SmoothedBarrierBridgeInterval>,
        hit_factors: Vec<SmoothedBarrierHitFactor>,
    ) -> Self {
        let mut log_bridge_survival = 0.0;
        let mut finite_correction_count = 0_u32;
        let mut zero_variance_count = 0_u32;
        let mut survival_underflow_count = 0_u32;
        let mut certain_survival_count = 0_u32;
        for interval in &intervals {
            log_bridge_survival += interval.log_survival();
            match interval.status() {
                BarrierBridgeStatus::FiniteCorrection => {
                    finite_correction_count = finite_correction_count.saturating_add(1);
                }
                BarrierBridgeStatus::TouchedEndpoint => {}
                BarrierBridgeStatus::ZeroVariance => {
                    zero_variance_count = zero_variance_count.saturating_add(1);
                }
                BarrierBridgeStatus::SurvivalUnderflow => {
                    survival_underflow_count = survival_underflow_count.saturating_add(1);
                }
                BarrierBridgeStatus::CertainSurvival => {
                    certain_survival_count = certain_survival_count.saturating_add(1);
                }
            }
        }
        let log_hit_survival = hit_factors
            .iter()
            .fold(0.0, |total, factor| total + factor.safety_weight().ln());
        Self {
            intervals: intervals.into_boxed_slice(),
            hit_factors: hit_factors.into_boxed_slice(),
            log_bridge_survival,
            log_hit_survival,
            finite_correction_count,
            zero_variance_count,
            survival_underflow_count,
            certain_survival_count,
        }
    }

    fn survival(&self) -> f64 {
        (self.log_bridge_survival + self.log_hit_survival).exp()
    }

    fn reverse(
        &self,
        survival_adjoint: f64,
    ) -> (
        Vec<SmoothedBarrierBridgeIntervalAdjoints>,
        Vec<(Option<usize>, f64)>,
    ) {
        let log_survival_adjoint = survival_adjoint * self.survival();
        let intervals = self
            .intervals
            .iter()
            .map(|interval| interval.reverse(log_survival_adjoint))
            .collect();
        let hit_factors = self
            .hit_factors
            .iter()
            .map(|factor| {
                let derivative = if factor.safety_weight() == 0.0 {
                    0.0
                } else {
                    -log_survival_adjoint * factor.state_derivative / factor.safety_weight()
                };
                (factor.state_index, derivative)
            })
            .collect();
        (intervals, hit_factors)
    }

    fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        let component_hit = |kind| {
            1.0 - self
                .hit_factors
                .iter()
                .filter(|factor| factor.kind == kind)
                .fold(1.0, |survival, factor| survival * factor.safety_weight())
        };
        BarrierPathDiagnosticValues {
            endpoint_hit: component_hit(SmoothedBarrierHitKind::Endpoint),
            dividend_jump_hit: component_hit(SmoothedBarrierHitKind::DividendJump),
            bridge_hit_weight: 1.0 - self.log_bridge_survival.exp(),
            interval_count: self.intervals.len() as f64,
            finite_correction_count: f64::from(self.finite_correction_count),
            zero_variance_count: f64::from(self.zero_variance_count),
            survival_underflow_count: f64::from(self.survival_underflow_count),
            certain_survival_count: f64::from(self.certain_survival_count),
        }
    }
}

impl SmoothedBarrierHitFactor {
    fn safety_weight(self) -> f64 {
        1.0 - self.hit_weight
    }
}

#[derive(Clone, Debug)]
enum ContinuousBarrierBridgeEvaluation {
    Exact(ExactContinuousBarrierBridgeEvaluation),
    Smoothed {
        path: SmoothedContinuousBarrierPath,
        interval_observation_indices: Box<[(Option<usize>, usize)]>,
    },
}

impl ContinuousBarrierBridgeEvaluation {
    fn survival(&self) -> f64 {
        match self {
            Self::Exact(evaluation) => evaluation.survival(),
            Self::Smoothed { path, .. } => path.survival(),
        }
    }

    fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        match self {
            Self::Exact(evaluation) => evaluation.diagnostic_values(),
            Self::Smoothed { path, .. } => path.diagnostic_values(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ContinuousBarrierPayoffTerms {
    value: f64,
    terminal_derivative: f64,
    survival_derivative: f64,
}

#[derive(Clone, Debug)]
struct ExactLocalVolContinuousBarrierBridgeEvaluation {
    path: BarrierBridgePath,
    interval_node_indices: Box<[(usize, usize)]>,
    node_interpolations: Box<[LocalVarianceInterpolation]>,
    endpoint_touched: bool,
    dividend_jump_touched: bool,
}

impl ExactLocalVolContinuousBarrierBridgeEvaluation {
    fn survival(&self) -> f64 {
        if self.endpoint_touched || self.dividend_jump_touched {
            0.0
        } else {
            self.path.survival()
        }
    }

    fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        BarrierPathDiagnosticValues::from_bridge(
            &self.path,
            self.endpoint_touched,
            self.dividend_jump_touched,
        )
    }
}

#[derive(Clone, Debug)]
enum LocalVolContinuousBarrierBridgeEvaluation {
    Exact(ExactLocalVolContinuousBarrierBridgeEvaluation),
    Smoothed {
        path: SmoothedContinuousBarrierPath,
        interval_node_indices: Box<[(usize, usize)]>,
        node_interpolations: Box<[LocalVarianceInterpolation]>,
    },
}

impl LocalVolContinuousBarrierBridgeEvaluation {
    fn survival(&self) -> f64 {
        match self {
            Self::Exact(evaluation) => evaluation.survival(),
            Self::Smoothed { path, .. } => path.survival(),
        }
    }

    fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        match self {
            Self::Exact(evaluation) => evaluation.diagnostic_values(),
            Self::Smoothed { path, .. } => path.diagnostic_values(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BarrierPathDiagnosticValues {
    endpoint_hit: f64,
    dividend_jump_hit: f64,
    bridge_hit_weight: f64,
    interval_count: f64,
    finite_correction_count: f64,
    zero_variance_count: f64,
    survival_underflow_count: f64,
    certain_survival_count: f64,
}

impl BarrierPathDiagnosticValues {
    fn from_bridge(
        path: &BarrierBridgePath,
        endpoint_touched: bool,
        dividend_jump_touched: bool,
    ) -> Self {
        let diagnostics = path.diagnostics();
        Self {
            endpoint_hit: if endpoint_touched { 1.0 } else { 0.0 },
            dividend_jump_hit: if dividend_jump_touched { 1.0 } else { 0.0 },
            bridge_hit_weight: if endpoint_touched || dividend_jump_touched {
                0.0
            } else {
                1.0 - path.survival()
            },
            interval_count: f64::from(diagnostics.interval_count),
            finite_correction_count: f64::from(diagnostics.finite_correction_count),
            zero_variance_count: f64::from(diagnostics.zero_variance_count),
            survival_underflow_count: f64::from(diagnostics.survival_underflow_count),
            certain_survival_count: f64::from(diagnostics.certain_survival_count),
        }
    }

    fn write_to(self, values: &mut [f64; PATHWISE_COMPONENTS]) {
        values[BARRIER_ENDPOINT_HIT] = self.endpoint_hit;
        values[BARRIER_DIVIDEND_JUMP_HIT] = self.dividend_jump_hit;
        values[BARRIER_BRIDGE_HIT_WEIGHT] = self.bridge_hit_weight;
        values[BARRIER_INTERVAL_COUNT] = self.interval_count;
        values[BARRIER_FINITE_CORRECTION_COUNT] = self.finite_correction_count;
        values[BARRIER_ZERO_VARIANCE_COUNT] = self.zero_variance_count;
        values[BARRIER_SURVIVAL_UNDERFLOW_COUNT] = self.survival_underflow_count;
        values[BARRIER_CERTAIN_SURVIVAL_COUNT] = self.certain_survival_count;
    }
}

#[derive(Clone, Debug)]
struct LocalVolPathwise {
    values: [f64; PATHWISE_COMPONENTS],
    raw_buckets: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug)]
struct LocalVolObservation {
    post_spot: f64,
    pre_dividend_spot: Option<f64>,
    node_index: usize,
}

struct LocalVolRuntimeInputs<'a> {
    grid: LocalVarianceGrid,
    reporting_iv_basis: Option<&'a LocalVolatilityReportingBasis>,
    vega_kt: Option<&'a VegaKtConfig>,
    market_forward: &'a EquityForward,
    valuation_date: Date,
    expiry_time: f64,
    event_times: &'a [f64],
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
        let horizon = self.plan.time_grid().nodes().last().copied().unwrap_or(0.0);
        horizon * self.grid.cap()
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
        event_times,
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
    let mut required_times = grid.time_nodes().to_vec();
    required_times.extend_from_slice(event_times);
    let time_grid = if let Some(dividends) = market_forward.discrete_dividends() {
        LocalVolTimeGrid::compile_with_dividends(required_times, dividends, maximum_step)?
    } else {
        LocalVolTimeGrid::compile(required_times, maximum_step)?
    };
    let dividends = market_forward.discrete_dividends().cloned();
    let dividend_timeline = dividends
        .as_ref()
        .map(AffineDividendTransform::event_timeline)
        .transpose()?
        .unwrap_or_default();
    let mut forwards = Vec::with_capacity(time_grid.nodes().len());
    let mut node_affine_coordinates = Vec::with_capacity(time_grid.nodes().len());
    let mut node_pre_dividend_coordinates = Vec::with_capacity(time_grid.nodes().len());
    for time in time_grid.nodes().iter().copied() {
        let evaluation = market_forward.evaluate(time)?;
        forwards.push(evaluation.forward);
        node_affine_coordinates.push(evaluation.affine_coordinate);
        node_pre_dividend_coordinates.push(
            dividend_timeline
                .iter()
                .find(|entry| entry.ex_time().to_bits() == time.to_bits())
                .map(|entry| entry.before()),
        );
    }
    let plan = LocalVolLogEulerPlan::new(time_grid, forwards)?;
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
        node_affine_coordinates: node_affine_coordinates.into_boxed_slice(),
        node_pre_dividend_coordinates: node_pre_dividend_coordinates.into_boxed_slice(),
        dividends,
        dividend_schedule,
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
        let path_state_diagnostics =
            PathStateDiagnostics::from_product(product, request.valuation_date());
        let continuous_barrier_spec = match product {
            ProductSpec::Barrier(barrier)
                if barrier.monitoring() == BarrierMonitoring::Continuous =>
            {
                Some(barrier)
            }
            _ => None,
        };
        let market_forward = request.market().equity().forward();
        let dividend_timeline = market_forward
            .discrete_dividends()
            .map(AffineDividendTransform::event_timeline)
            .transpose()?
            .map_or_else(Vec::new, |entries| entries.into_vec());
        let jump_dates = match product {
            ProductSpec::Barrier(barrier) => barrier
                .monitoring_dates()
                .iter()
                .copied()
                .filter(|date| {
                    let time =
                        DayCountConvention::Act365F.year_fraction(request.valuation_date(), *date);
                    dividend_timeline
                        .iter()
                        .any(|entry| entry.ex_time().to_bits() == time.to_bits())
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        let payoff_graph = match (product, request.risk().payoff_smoothing()) {
            (ProductSpec::Digital(digital), Some(PayoffSmoothing::CompactC2 { half_width })) => {
                digital.smoothed_source_graph(CompactC2Smoothing::from_positive(half_width))?
            }
            (ProductSpec::Barrier(barrier), Some(PayoffSmoothing::CompactC2 { half_width })) => {
                barrier.smoothed_source_graph_with_dividend_jumps(
                    CompactC2Smoothing::from_positive(half_width),
                    &jump_dates,
                )?
            }
            (ProductSpec::Barrier(barrier), None) => {
                barrier.source_graph_with_dividend_jumps(&jump_dates)?
            }
            _ => product.source_graph(request.valuation_date())?,
        };
        let payoff = payoff_graph.compile(GraphLimitPolicy::DEFAULT)?;
        let observations = payoff.terminal_observations();
        for (underlying, _) in &observations {
            if *underlying != product.underlying() {
                return Err(MonteCarloError::UnsupportedObservationUnderlying {
                    product: *underlying,
                    market: product.underlying(),
                });
            }
        }
        let contractual_observation_dates = observations
            .iter()
            .map(|(_, date)| *date)
            .collect::<Vec<_>>();
        let mut observation_points = contractual_observation_dates
            .iter()
            .map(|date| {
                (
                    DayCountConvention::Act365F.year_fraction(request.valuation_date(), *date),
                    Some(*date),
                )
            })
            .collect::<Vec<_>>();
        if let Some(barrier) = continuous_barrier_spec {
            let monitoring_end = DayCountConvention::Act365F.year_fraction(
                request.valuation_date(),
                *barrier
                    .monitoring_dates()
                    .last()
                    .expect("Barrier monitoring dates are non-empty"),
            );
            observation_points
                .retain(|(time, date)| *time > 0.0 || *date == Some(barrier.expiry()));
            for entry in &dividend_timeline {
                let time = entry.ex_time();
                if time > 0.0
                    && time <= monitoring_end
                    && !observation_points
                        .iter()
                        .any(|(point_time, _)| point_time.to_bits() == time.to_bits())
                {
                    observation_points.push((time, None));
                }
            }
            observation_points.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .expect("observation times are finite")
            });
        }
        let observation_times = observation_points
            .iter()
            .map(|(time, _)| *time)
            .collect::<Vec<_>>();
        let observation_dates = observation_points
            .iter()
            .map(|(_, date)| *date)
            .collect::<Vec<_>>();
        let time = if observations.is_empty() {
            0.0
        } else {
            DayCountConvention::Act365F.year_fraction(request.valuation_date(), product.expiry())
        };
        let payment_time = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), product.payment_date());
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
                    event_times: &observation_times,
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
        let pre_dividend_observations = payoff.pre_dividend_observations();
        let mut observation_forwards = Vec::with_capacity(observation_dates.len());
        let mut observation_affine_coordinates = Vec::with_capacity(observation_dates.len());
        let mut observation_pre_dividend_coordinates = Vec::with_capacity(observation_dates.len());
        for (&date, &observation_time) in observation_dates.iter().zip(&observation_times) {
            let evaluation = market_forward.evaluate(observation_time)?;
            observation_forwards.push(evaluation.forward);
            observation_affine_coordinates.push(evaluation.affine_coordinate);
            let dividend_entry = dividend_timeline
                .iter()
                .find(|entry| entry.ex_time().to_bits() == observation_time.to_bits());
            let needs_pre_dividend = date.is_some_and(|date| {
                pre_dividend_observations.contains(&(product.underlying(), date))
            });
            observation_pre_dividend_coordinates.push(
                dividend_entry
                    .filter(|_| continuous_barrier_spec.is_some() || needs_pre_dividend)
                    .map(|entry| entry.before()),
            );
        }
        if let (Some(barrier), Some(PayoffSmoothing::CompactC2 { half_width })) =
            (continuous_barrier_spec, request.risk().payoff_smoothing())
            && barrier.direction() == BarrierDirection::Up
        {
            let invalid_coordinate = std::iter::once(AffineDividendCoordinate::identity())
                .chain(observation_affine_coordinates.iter().copied())
                .chain(
                    observation_pre_dividend_coordinates
                        .iter()
                        .flatten()
                        .copied(),
                )
                .find_map(|coordinate| {
                    let transformed_spot_barrier = barrier.barrier().get() - coordinate.a() * spot;
                    (half_width.get() >= transformed_spot_barrier)
                        .then_some(transformed_spot_barrier)
                });
            if let Some(transformed_spot_barrier) = invalid_coordinate {
                return Err(BarrierBridgeError::InvalidSmoothedSafeDistance {
                    safe_distance_bits: half_width.get().to_bits(),
                    transformed_spot_barrier_bits: transformed_spot_barrier.to_bits(),
                }
                .into());
            }
        }
        let observation_local_vol_node_indices = local_volatility.as_ref().map(|runtime| {
            observation_times
                .iter()
                .map(|time| {
                    runtime
                        .plan
                        .time_grid()
                        .node_index_for_time(*time)
                        .expect("contractual observations are Local Volatility event nodes")
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });
        let continuous_barrier = continuous_barrier_spec.map(|barrier| {
            let monitoring_end = DayCountConvention::Act365F.year_fraction(
                request.valuation_date(),
                *barrier
                    .monitoring_dates()
                    .last()
                    .expect("Barrier monitoring dates are non-empty"),
            );
            let bridge_observation_indices = observation_times
                .iter()
                .enumerate()
                .filter_map(|(index, time)| (*time <= monitoring_end).then_some(index))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let expiry_observation_index = observation_dates
                .iter()
                .position(|date| *date == Some(barrier.expiry()))
                .expect("Barrier graph retains expiry");
            ContinuousBarrierRuntime {
                strike: barrier.strike().get(),
                barrier: barrier.barrier().get(),
                notional: barrier.notional().get(),
                rebate: barrier.rebate().map_or(0.0, PositiveF64::get),
                side: barrier.side(),
                direction: barrier.direction(),
                style: barrier.style(),
                monitoring_end_time: monitoring_end,
                bridge_observation_indices,
                expiry_observation_index,
            }
        });
        let request_fingerprint = *fingerprint_request(request)?.as_bytes();
        let request_migration = request
            .wire_migration()
            .cloned()
            .unwrap_or_else(|| MigrationProvenance::current(request_fingerprint));
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
        let (payoff_smoothing_endpoint_count, payoff_smoothing_dividend_jump_count) =
            match (&continuous_barrier, &local_volatility) {
                (Some(barrier), Some(local_volatility)) => {
                    let node_count = local_volatility
                        .plan
                        .time_grid()
                        .nodes()
                        .iter()
                        .take_while(|time| **time <= barrier.monitoring_end_time)
                        .count();
                    let jump_count = local_volatility
                        .node_pre_dividend_coordinates
                        .iter()
                        .take(node_count)
                        .filter(|coordinate| coordinate.is_some())
                        .count();
                    (
                        u32::try_from(node_count.saturating_sub(jump_count)).unwrap_or(u32::MAX),
                        u32::try_from(jump_count).unwrap_or(u32::MAX),
                    )
                }
                (Some(barrier), None) => {
                    let jump_count = barrier
                        .bridge_observation_indices
                        .iter()
                        .filter(|index| observation_pre_dividend_coordinates[**index].is_some())
                        .count();
                    (
                        u32::try_from(
                            1_usize
                                .saturating_add(barrier.bridge_observation_indices.len())
                                .saturating_sub(jump_count),
                        )
                        .unwrap_or(u32::MAX),
                        u32::try_from(jump_count).unwrap_or(u32::MAX),
                    )
                }
                (None, _) => match product {
                    ProductSpec::Digital(_) => (1, 0),
                    ProductSpec::Barrier(barrier) => (
                        u32::try_from(barrier.monitoring_dates().len()).unwrap_or(u32::MAX),
                        u32::try_from(jump_dates.len()).unwrap_or(u32::MAX),
                    ),
                    _ => (0, 0),
                },
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
            observation_affine_coordinates: observation_affine_coordinates.into_boxed_slice(),
            observation_pre_dividend_coordinates: observation_pre_dividend_coordinates
                .into_boxed_slice(),
            observation_local_vol_node_indices,
            engine,
            execution_policy,
            aad_tile_policy,
            checkpoint_policy,
            request_delta: request.risk().delta(),
            request_gamma: request.risk().gamma(),
            request_vega: request.risk().vega(),
            payoff_smoothing: request.risk().payoff_smoothing(),
            payoff_smoothing_endpoint_count,
            payoff_smoothing_dividend_jump_count,
            path_state_diagnostics,
            smile_dynamics: request.risk().smile_dynamics(),
            validation_spot_bump,
            validation_volatility_bump,
            request_fingerprint,
            request_migration,
            plan_fingerprint,
            discount_region: discount_evaluation.region,
            dividend_region: forward_evaluation.dividend_region,
            market_forward: market_forward.clone(),
            local_volatility,
            continuous_barrier,
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

    fn payoff_smoothing_diagnostics(&self) -> Option<PayoffSmoothingDiagnostics> {
        self.payoff_smoothing.map(|smoothing| {
            let mut diagnostics = PayoffSmoothingDiagnostics::from(smoothing);
            diagnostics.endpoint_count = self.payoff_smoothing_endpoint_count;
            diagnostics.dividend_jump_count = self.payoff_smoothing_dividend_jump_count;
            diagnostics
        })
    }

    fn barrier_bridge_diagnostics(
        &self,
        statistics: &[DeterministicStatistics; PATHWISE_COMPONENTS],
        independent_units: u64,
    ) -> Option<BarrierBridgeDiagnostics> {
        self.continuous_barrier.as_ref()?;
        let mean =
            |component: usize| statistics[component].sum().total() / independent_units as f64;
        let smoothed = self.payoff_smoothing.is_some();
        Some(BarrierBridgeDiagnostics {
            abi: if smoothed {
                SMOOTHED_BARRIER_BRIDGE_ABI
            } else {
                BARRIER_BRIDGE_ABI
            },
            policy_version: if smoothed {
                BarrierBridgeDiagnostics::SMOOTHED_POLICY_VERSION
            } else {
                BarrierBridgeDiagnostics::EXACT_POLICY_VERSION
            },
            indicator_mode: if smoothed {
                BarrierHitIndicatorMode::CompactC2
            } else {
                BarrierHitIndicatorMode::Exact
            },
            endpoint_hit_fraction: mean(BARRIER_ENDPOINT_HIT),
            dividend_jump_hit_fraction: mean(BARRIER_DIVIDEND_JUMP_HIT),
            mean_conditional_bridge_hit_weight: mean(BARRIER_BRIDGE_HIT_WEIGHT),
            mean_interval_count: mean(BARRIER_INTERVAL_COUNT),
            mean_finite_correction_count: mean(BARRIER_FINITE_CORRECTION_COUNT),
            mean_zero_variance_count: mean(BARRIER_ZERO_VARIANCE_COUNT),
            mean_survival_underflow_count: mean(BARRIER_SURVIVAL_UNDERFLOW_COUNT),
            mean_certain_survival_count: mean(BARRIER_CERTAIN_SURVIVAL_COUNT),
        })
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

    #[must_use]
    pub const fn request_migration(&self) -> &MigrationProvenance {
        &self.request_migration
    }

    fn replay_metadata(&self) -> ReplayMetadata {
        ReplayMetadata::with_migration(
            SchemaVersion::CURRENT,
            self.request_fingerprint,
            crate::version(),
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            self.request_migration.clone(),
        )
    }

    pub fn execute(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        if self.observation_dates.is_empty() {
            return self.execute_fixed_payoff();
        }
        match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => self.execute_pseudo(engine),
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => self.execute_rqmc(engine),
        }
    }

    fn execute_fixed_payoff(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        let value = self.discounted_payoff_from_normals(&[])?;
        let (
            estimator,
            master_seed,
            independent_units,
            evaluated_paths,
            scramble_count,
            antithetic,
        ) = match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => (
                EstimatorKind::PseudoMonteCarlo,
                engine.master_seed(),
                engine.independent_sampling_units().get(),
                engine.evaluated_paths(),
                None,
                engine.variance_reduction().antithetic(),
            ),
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => {
                let multiplier = if engine.variance_reduction().antithetic() {
                    2_u128
                } else {
                    1_u128
                };
                (
                    EstimatorKind::RandomizedQuasiMonteCarlo,
                    engine.master_scramble_seed(),
                    u64::from(engine.scramble_count().get()),
                    u128::from(engine.points_per_scramble().get())
                        * u128::from(engine.scramble_count().get())
                        * multiplier,
                    Some(engine.scramble_count().get()),
                    engine.variance_reduction().antithetic(),
                )
            }
        };
        let estimate = Estimate::new(value, 0.0, value, value, estimator, independent_units)?;
        let risks = self.fixed_payoff_risk_report(estimator, independent_units)?;
        let risk_diagnostics = self.fixed_payoff_risk_diagnostics(estimator, independent_units)?;
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: self.replay_metadata(),
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance: 0.0,
            estimator_variance: 0.0,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths,
            diagnostics: MonteCarloDiagnostics {
                master_seed,
                estimator,
                scramble_count,
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: None,
            },
        })
    }

    fn fixed_payoff_risk_report(
        &self,
        estimator: EstimatorKind,
        independent_units: u64,
    ) -> Result<RiskReport, MonteCarloError> {
        let zero_delta = self
            .request_delta
            .then(|| {
                zero_risk_estimate(
                    estimator,
                    independent_units,
                    RiskUnit::DeltaRaw,
                    RiskUnit::DeltaOnePercentSpot,
                )
            })
            .transpose()?;
        let zero_gamma = self
            .request_gamma
            .map(|_| {
                zero_risk_estimate(
                    estimator,
                    independent_units,
                    RiskUnit::GammaRaw,
                    RiskUnit::GammaOnePercentSpotSquared,
                )
            })
            .transpose()?;
        let zero_vega = self
            .request_vega
            .then(|| {
                zero_risk_estimate(
                    estimator,
                    independent_units,
                    RiskUnit::VegaRaw,
                    RiskUnit::VegaOneVolPoint,
                )
            })
            .transpose()?;
        Ok(RiskReport {
            delta: zero_delta,
            gamma: zero_gamma,
            vega: zero_vega,
            vega_kt: None,
        })
    }

    fn fixed_payoff_risk_diagnostics(
        &self,
        estimator: EstimatorKind,
        independent_units: u64,
    ) -> Result<RiskDiagnostics, MonteCarloError> {
        let zero_statistics =
            [DeterministicStatistics::from_ordered_values_two_pass(&[0.0]); PATHWISE_COMPONENTS];
        self.build_risk_diagnostics(&zero_statistics, independent_units, estimator)
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
                    let normals = self.normals(&generator, sampling_unit);
                    let primary = self.pathwise_values(&normals, lane, workspace)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.pathwise_values(&mate_normals, lane, workspace)?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                    }
                },
            )?
        } else if self.continuous_barrier.is_some() {
            executor.try_map_reduce_statistics_array_tiled(
                engine.independent_sampling_units().get(),
                self.aad_tile_policy.resolved_capacity(),
                |sampling_unit| {
                    let normals = self.normals(&generator, sampling_unit);
                    let primary = self.continuous_barrier_path_values_from_normals(&normals)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate =
                            self.continuous_barrier_path_values_from_normals(&mate_normals)?;
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
                        Ok::<f64, MonteCarloError>((primary + mate) * 0.5)
                    } else {
                        Ok::<f64, MonteCarloError>(primary)
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
            replay: self.replay_metadata(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
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
                        let normals = self.rqmc_normals(&qmc, scramble, point)?;
                        let primary = self.pathwise_values(&normals, lane, workspace)?;
                        if antithetic {
                            let mate_normals =
                                normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                            let mate = self.pathwise_values(&mate_normals, lane, workspace)?;
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                                |component| (primary[component] + mate[component]) * 0.5,
                            ))
                        } else {
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                        }
                    },
                )?
            } else if self.continuous_barrier.is_some() {
                executor.try_map_reduce_statistics_array_tiled(
                    points,
                    self.aad_tile_policy.resolved_capacity(),
                    |point| {
                        let normals = self.rqmc_normals(&qmc, scramble, point)?;
                        let primary = self.continuous_barrier_path_values_from_normals(&normals)?;
                        if antithetic {
                            let mate_normals =
                                normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                            let mate =
                                self.continuous_barrier_path_values_from_normals(&mate_normals)?;
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
            replay: self.replay_metadata(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
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
            replay: self.replay_metadata(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
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
        let mut raw_bucket_samples =
            Vec::with_capacity(bucket_sample_capacity(values.capacity(), bucket_count));
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
            replay: self.replay_metadata(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
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
        let observations = self.local_vol_path_observations(&path)?;
        if let Some(barrier) = &self.continuous_barrier {
            let bridge =
                self.local_vol_continuous_barrier_bridge(local_volatility, barrier, &path, spot)?;
            let terminal = observations[barrier.expiry_observation_index].post_spot;
            let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
            return Ok(self.discount * payoff.value);
        }
        let outputs = self.payoff.evaluate_with_pre_dividend_spots(
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .map(|index| observations[index].post_spot)
            },
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .and_then(|index| observations[index].pre_dividend_spot)
            },
        )?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(pricing_product::GraphError::NoOutputs)?)
    }

    fn local_vol_path_observations(
        &self,
        path: &LocalVolPath,
    ) -> Result<Vec<LocalVolObservation>, MonteCarloError> {
        let node_indices = self.observation_local_vol_node_indices.as_ref().ok_or(
            MonteCarloError::UnsupportedModel {
                model: "local_volatility",
            },
        )?;
        Ok(node_indices
            .iter()
            .enumerate()
            .map(|(index, &node_index)| {
                let canonical_f = path.states()[node_index];
                let post_coordinate = self.observation_affine_coordinates[index];
                let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * canonical_f;
                let pre_dividend_spot = self.observation_pre_dividend_coordinates[index]
                    .map(|coordinate| coordinate.a() * self.spot + coordinate.b() * canonical_f);
                LocalVolObservation {
                    post_spot,
                    pre_dividend_spot,
                    node_index,
                }
            })
            .collect())
    }

    fn local_vol_continuous_barrier_bridge(
        &self,
        local_volatility: &LocalVolRuntime,
        barrier: &ContinuousBarrierRuntime,
        path: &LocalVolPath,
        spot: f64,
    ) -> Result<LocalVolContinuousBarrierBridgeEvaluation, MonteCarloError> {
        let nodes = local_volatility.plan.time_grid().nodes();
        let end_node = nodes
            .iter()
            .rposition(|time| *time <= barrier.monitoring_end_time)
            .expect("Local Volatility time grids start at valuation");
        let mut node_interpolations = Vec::with_capacity(end_node + 1);
        for (node, (&time, &state)) in nodes
            .iter()
            .zip(path.states())
            .take(end_node + 1)
            .enumerate()
        {
            let forward = local_volatility.plan.forward_normalizers()[node];
            node_interpolations.push(
                local_volatility
                    .grid
                    .interpolate(time, (state / forward).ln())?,
            );
        }

        if let Some(PayoffSmoothing::CompactC2 { half_width }) = self.payoff_smoothing {
            return self.smoothed_local_vol_continuous_barrier_bridge(
                local_volatility,
                barrier,
                path,
                spot,
                end_node,
                node_interpolations,
                CompactC2Smoothing::from_positive(half_width),
            );
        }

        let mut intervals = Vec::with_capacity(end_node);
        let mut interval_node_indices = Vec::with_capacity(end_node);
        let initial_touched = barrier_touched(barrier.direction, spot, barrier.barrier);
        let mut dividend_jump_touched = false;
        for right_node in 1..=end_node {
            if initial_touched {
                break;
            }
            let left_node = right_node - 1;
            let left_coordinate = local_volatility.node_affine_coordinates[left_node];
            let right_post_coordinate = local_volatility.node_affine_coordinates[right_node];
            let right_pre_coordinate = local_volatility.node_pre_dividend_coordinates[right_node]
                .unwrap_or(right_post_coordinate);
            let left_barrier = transformed_barrier(barrier.barrier, spot, left_coordinate)?;
            let right_barrier = transformed_barrier(barrier.barrier, spot, right_pre_coordinate)?;
            intervals.push(BarrierBridgeIntervalInput {
                direction: bridge_direction(barrier.direction),
                left_state: path.states()[left_node],
                right_state: path.states()[right_node],
                left_barrier,
                right_barrier,
                left_local_variance: node_interpolations[left_node].value,
                right_local_variance: node_interpolations[right_node].value,
                dt: nodes[right_node] - nodes[left_node],
            });
            interval_node_indices.push((left_node, right_node));

            if local_volatility.node_pre_dividend_coordinates[right_node].is_some() {
                let state = path.states()[right_node];
                let pre_spot = right_pre_coordinate.a() * spot + right_pre_coordinate.b() * state;
                let post_spot =
                    right_post_coordinate.a() * spot + right_post_coordinate.b() * state;
                let pre_touched = barrier_touched(barrier.direction, pre_spot, barrier.barrier);
                let post_touched = barrier_touched(barrier.direction, post_spot, barrier.barrier);
                if pre_touched || post_touched {
                    dividend_jump_touched = !pre_touched && post_touched;
                    break;
                }
            }
        }
        let path = BarrierBridgePath::evaluate(&intervals)?;
        let endpoint_touched = initial_touched || path.diagnostics().touched_endpoint_count != 0;
        Ok(LocalVolContinuousBarrierBridgeEvaluation::Exact(
            ExactLocalVolContinuousBarrierBridgeEvaluation {
                path,
                interval_node_indices: interval_node_indices.into_boxed_slice(),
                node_interpolations: node_interpolations.into_boxed_slice(),
                endpoint_touched,
                dividend_jump_touched,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn smoothed_local_vol_continuous_barrier_bridge(
        &self,
        local_volatility: &LocalVolRuntime,
        barrier: &ContinuousBarrierRuntime,
        path: &LocalVolPath,
        spot: f64,
        end_node: usize,
        node_interpolations: Vec<LocalVarianceInterpolation>,
        smoothing: CompactC2Smoothing,
    ) -> Result<LocalVolContinuousBarrierBridgeEvaluation, MonteCarloError> {
        let direction = bridge_direction(barrier.direction);
        let nodes = local_volatility.plan.time_grid().nodes();
        let initial_coordinate = local_volatility.node_affine_coordinates[0];
        let initial_barrier = transformed_barrier(barrier.barrier, spot, initial_coordinate)?;
        let initial_endpoint =
            SmoothedBarrierBridgeEndpoint::evaluate(SmoothedBarrierBridgeEndpointInput {
                direction,
                state: path.states()[0],
                transformed_barrier: initial_barrier,
                affine_scale: initial_coordinate.b(),
                smoothing,
            })?;
        let mut hit_factors = vec![smoothed_endpoint_hit_factor(
            initial_endpoint,
            Some(0),
            SmoothedBarrierHitKind::Endpoint,
        )];
        let mut intervals = Vec::with_capacity(end_node);
        let mut interval_node_indices = Vec::with_capacity(end_node);
        for right_node in 1..=end_node {
            let left_node = right_node - 1;
            let left_coordinate = local_volatility.node_affine_coordinates[left_node];
            let right_post_coordinate = local_volatility.node_affine_coordinates[right_node];
            let right_pre_coordinate = local_volatility.node_pre_dividend_coordinates[right_node]
                .unwrap_or(right_post_coordinate);
            let left_barrier = transformed_barrier(barrier.barrier, spot, left_coordinate)?;
            let right_barrier = transformed_barrier(barrier.barrier, spot, right_pre_coordinate)?;
            let interval =
                SmoothedBarrierBridgeInterval::evaluate(SmoothedBarrierBridgeIntervalInput {
                    left_endpoint: SmoothedBarrierBridgeEndpointInput {
                        direction,
                        state: path.states()[left_node],
                        transformed_barrier: left_barrier,
                        affine_scale: left_coordinate.b(),
                        smoothing,
                    },
                    right_endpoint: SmoothedBarrierBridgeEndpointInput {
                        direction,
                        state: path.states()[right_node],
                        transformed_barrier: right_barrier,
                        affine_scale: right_pre_coordinate.b(),
                        smoothing,
                    },
                    left_local_variance: node_interpolations[left_node].value,
                    right_local_variance: node_interpolations[right_node].value,
                    dt: nodes[right_node] - nodes[left_node],
                })?;
            if local_volatility.node_pre_dividend_coordinates[right_node].is_some() {
                let state = path.states()[right_node];
                let pre_spot = right_pre_coordinate.a() * spot + right_pre_coordinate.b() * state;
                let post_spot =
                    right_post_coordinate.a() * spot + right_post_coordinate.b() * state;
                hit_factors.push(smoothed_jump_hit_factor(
                    barrier.direction,
                    pre_spot,
                    post_spot,
                    right_pre_coordinate.b(),
                    right_post_coordinate.b(),
                    barrier.barrier,
                    smoothing,
                    Some(right_node),
                ));
            } else {
                hit_factors.push(smoothed_endpoint_hit_factor(
                    interval.right_endpoint(),
                    Some(right_node),
                    SmoothedBarrierHitKind::Endpoint,
                ));
            }
            intervals.push(interval);
            interval_node_indices.push((left_node, right_node));
        }
        Ok(LocalVolContinuousBarrierBridgeEvaluation::Smoothed {
            path: SmoothedContinuousBarrierPath::evaluate(intervals, hit_factors),
            interval_node_indices: interval_node_indices.into_boxed_slice(),
            node_interpolations: node_interpolations.into_boxed_slice(),
        })
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
        compile_local_vol_runtime(LocalVolRuntimeInputs {
            grid: local_volatility.grid.clone(),
            reporting_iv_basis: None,
            vega_kt: None,
            market_forward: &market_forward,
            valuation_date: self.valuation_date,
            expiry_time: self.time,
            event_times: &self.observation_times,
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
        let observations = self.local_vol_path_observations(&path_state)?;
        let continuous = if let Some(barrier) = &self.continuous_barrier {
            let bridge = self.local_vol_continuous_barrier_bridge(
                local_volatility,
                barrier,
                &path_state,
                self.spot,
            )?;
            let terminal = observations[barrier.expiry_observation_index].post_spot;
            let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
            Some((bridge, payoff))
        } else {
            None
        };
        let graph_payoff = if continuous.is_none() {
            Some(self.payoff.evaluate_single_with_observation_adjoints(
                |underlying, date| {
                    if underlying != self.underlying {
                        return None;
                    }
                    self.observation_dates
                        .iter()
                        .position(|observation_date| *observation_date == Some(date))
                        .map(|index| observations[index].post_spot)
                },
                |underlying, date| {
                    if underlying != self.underlying {
                        return None;
                    }
                    self.observation_dates
                        .iter()
                        .position(|observation_date| *observation_date == Some(date))
                        .and_then(|index| observations[index].pre_dividend_spot)
                },
            )?)
        } else {
            None
        };
        let price = self.discount
            * continuous.as_ref().map_or_else(
                || graph_payoff.as_ref().expect("graph payoff").value,
                |(_, payoff)| payoff.value,
            );
        let mut values = [0.0; PATHWISE_COMPONENTS];
        let mut raw_buckets = None;
        values[PRICE] = price;
        if let Some((bridge, _)) = &continuous {
            bridge.diagnostic_values().write_to(&mut values);
        }
        if self.request_vega || local_volatility.vega_kt.is_some() {
            let mut state_seeds = vec![0.0; path_state.states().len()];
            let mut bridge_grid_adjoints = vec![0.0; local_volatility.grid.values().len()];
            if let Some((bridge, payoff)) = &continuous {
                let barrier = self
                    .continuous_barrier
                    .as_ref()
                    .expect("continuous Barrier");
                let terminal_observation = observations[barrier.expiry_observation_index];
                let terminal_coordinate =
                    self.observation_affine_coordinates[barrier.expiry_observation_index];
                state_seeds[terminal_observation.node_index] +=
                    self.discount * payoff.terminal_derivative * terminal_coordinate.b();
                if let LocalVolContinuousBarrierBridgeEvaluation::Exact(bridge) = bridge
                    && !bridge.endpoint_touched
                    && !bridge.dividend_jump_touched
                {
                    let interval_adjoints = bridge
                        .path
                        .reverse(self.discount * payoff.survival_derivative * bridge.survival());
                    let mut variance_seeds = vec![0.0; bridge.node_interpolations.len()];
                    for ((left_node, right_node), adjoints) in bridge
                        .interval_node_indices
                        .iter()
                        .copied()
                        .zip(interval_adjoints)
                    {
                        state_seeds[left_node] += adjoints.left_state;
                        state_seeds[right_node] += adjoints.right_state;
                        variance_seeds[left_node] += adjoints.left_local_variance;
                        variance_seeds[right_node] += adjoints.right_local_variance;
                    }
                    accumulate_bridge_local_variance_adjoints(
                        local_volatility,
                        path_state.states(),
                        &bridge.node_interpolations,
                        variance_seeds,
                        &mut state_seeds,
                        &mut bridge_grid_adjoints,
                    );
                }
                if let LocalVolContinuousBarrierBridgeEvaluation::Smoothed {
                    path,
                    interval_node_indices,
                    node_interpolations,
                } = bridge
                {
                    let (interval_adjoints, hit_factor_adjoints) =
                        path.reverse(self.discount * payoff.survival_derivative);
                    let mut variance_seeds = vec![0.0; node_interpolations.len()];
                    for ((left_node, right_node), adjoints) in
                        interval_node_indices.iter().copied().zip(interval_adjoints)
                    {
                        state_seeds[left_node] += adjoints.left_endpoint.state;
                        state_seeds[right_node] += adjoints.right_endpoint.state;
                        variance_seeds[left_node] += adjoints.left_local_variance;
                        variance_seeds[right_node] += adjoints.right_local_variance;
                    }
                    for (state_index, adjoint) in hit_factor_adjoints {
                        state_seeds
                            [state_index.expect("Local Vol hit factors use node indices")] +=
                            adjoint;
                    }
                    accumulate_bridge_local_variance_adjoints(
                        local_volatility,
                        path_state.states(),
                        node_interpolations,
                        variance_seeds,
                        &mut state_seeds,
                        &mut bridge_grid_adjoints,
                    );
                }
            } else {
                let payoff = graph_payoff.as_ref().expect("graph payoff");
                for adjoint in &payoff.terminal_adjoints {
                    if adjoint.underlying != self.underlying {
                        continue;
                    }
                    if let Some(index) = self
                        .observation_dates
                        .iter()
                        .position(|date| *date == Some(adjoint.observation_date))
                    {
                        let observation = observations[index];
                        let coordinate = self.observation_affine_coordinates[index];
                        state_seeds[observation.node_index] +=
                            self.discount * adjoint.value * coordinate.b();
                    }
                }
                for adjoint in &payoff.pre_dividend_adjoints {
                    if adjoint.underlying != self.underlying {
                        continue;
                    }
                    if let Some(index) = self
                        .observation_dates
                        .iter()
                        .position(|date| *date == Some(adjoint.observation_date))
                    {
                        let observation = observations[index];
                        let coordinate = self.observation_pre_dividend_coordinates[index]
                            .expect("pre-dividend adjoints have a matching coordinate");
                        state_seeds[observation.node_index] +=
                            self.discount * adjoint.value * coordinate.b();
                    }
                }
            }
            let reverse = path_state.reverse_state_adjoints(
                &state_seeds,
                local_volatility.grid.values().len(),
                local_volatility.grid.log_moneyness_nodes().len(),
            )?;
            let local_vol_node_adjoints = reverse
                .local_variance_value_adjoints()
                .iter()
                .zip(&bridge_grid_adjoints)
                .zip(local_volatility.grid.values())
                .map(|((path_adjoint, bridge_adjoint), variance)| {
                    (path_adjoint + bridge_adjoint) * 2.0 * variance.sqrt()
                })
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
            replay: self.replay_metadata(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
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
        let mut raw_bucket_samples = Vec::with_capacity(bucket_sample_capacity(
            replicate_values.capacity(),
            bucket_count,
        ));

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
            replay: self.replay_metadata(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
            },
        })
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

    fn discounted_payoff_from_normals(&self, normals: &[f64]) -> Result<f64, MonteCarloError> {
        let observations = self.path_observations_from_normals(normals, self.spot, self.volatility);
        if let Some(barrier) = &self.continuous_barrier {
            return self.continuous_barrier_discounted_payoff(barrier, &observations);
        }
        let outputs = self.payoff.evaluate_with_pre_dividend_spots(
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .map(|index| observations[index].post_spot)
            },
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .and_then(|index| observations[index].pre_dividend_spot)
            },
        )?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(pricing_product::GraphError::NoOutputs)?)
    }

    fn continuous_barrier_path_values_from_normals(
        &self,
        normals: &[f64],
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let barrier = self
            .continuous_barrier
            .as_ref()
            .expect("continuous Barrier path values require a compiled Barrier");
        let observations = self.path_observations_from_normals(normals, self.spot, self.volatility);
        let bridge =
            self.continuous_barrier_bridge(barrier, &observations, self.spot, self.volatility)?;
        let terminal = observations[barrier.expiry_observation_index].post_spot;
        let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
        let mut values = [0.0; PATHWISE_COMPONENTS];
        values[PRICE] = self.discount * payoff.value;
        bridge.diagnostic_values().write_to(&mut values);
        Ok(values)
    }

    fn continuous_barrier_discounted_payoff(
        &self,
        barrier: &ContinuousBarrierRuntime,
        observations: &[PathObservation],
    ) -> Result<f64, MonteCarloError> {
        let bridge =
            self.continuous_barrier_bridge(barrier, observations, self.spot, self.volatility)?;
        let terminal = observations[barrier.expiry_observation_index].post_spot;
        let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
        Ok(self.discount * payoff.value)
    }

    fn continuous_barrier_bridge(
        &self,
        barrier: &ContinuousBarrierRuntime,
        observations: &[PathObservation],
        spot: f64,
        volatility: f64,
    ) -> Result<ContinuousBarrierBridgeEvaluation, MonteCarloError> {
        if let Some(PayoffSmoothing::CompactC2 { half_width }) = self.payoff_smoothing {
            return self.smoothed_continuous_barrier_bridge(
                barrier,
                observations,
                spot,
                volatility,
                CompactC2Smoothing::from_positive(half_width),
            );
        }
        let mut previous_state = spot;
        let mut previous_time = 0.0;
        let mut previous_index = None;
        let mut intervals = Vec::with_capacity(barrier.bridge_observation_indices.len());
        let mut interval_observation_indices =
            Vec::with_capacity(barrier.bridge_observation_indices.len());
        let initial_touched = barrier_touched(barrier.direction, spot, barrier.barrier);
        let mut dividend_jump_touched = false;
        let mut previous_barrier = barrier.barrier;
        let variance = volatility * volatility;
        for &index in &barrier.bridge_observation_indices {
            if initial_touched {
                break;
            }
            let time = self.observation_times[index];
            let state = observations[index].canonical_f;
            let post_coordinate = self.observation_affine_coordinates[index];
            let pre_coordinate =
                self.observation_pre_dividend_coordinates[index].unwrap_or(post_coordinate);
            let pre_barrier = transformed_barrier(barrier.barrier, self.spot, pre_coordinate)?;
            let post_barrier = transformed_barrier(barrier.barrier, self.spot, post_coordinate)?;
            let dt = time - previous_time;
            if dt > 0.0 {
                intervals.push(BarrierBridgeIntervalInput {
                    direction: bridge_direction(barrier.direction),
                    left_state: previous_state,
                    right_state: state,
                    left_barrier: previous_barrier,
                    right_barrier: pre_barrier,
                    left_local_variance: variance,
                    right_local_variance: variance,
                    dt,
                });
                interval_observation_indices.push((previous_index, index));
            }
            if self.observation_pre_dividend_coordinates[index].is_some() {
                let pre_spot = pre_coordinate.a() * self.spot + pre_coordinate.b() * state;
                let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * state;
                let pre_touched = barrier_touched(barrier.direction, pre_spot, barrier.barrier);
                let post_touched = barrier_touched(barrier.direction, post_spot, barrier.barrier);
                if pre_touched || post_touched {
                    dividend_jump_touched = !pre_touched && post_touched;
                    break;
                }
            }
            previous_state = state;
            previous_barrier = post_barrier;
            previous_time = time;
            previous_index = Some(index);
        }
        let path = BarrierBridgePath::evaluate(&intervals)?;
        let endpoint_touched = initial_touched || path.diagnostics().touched_endpoint_count != 0;
        Ok(ContinuousBarrierBridgeEvaluation::Exact(
            ExactContinuousBarrierBridgeEvaluation {
                path,
                interval_observation_indices: interval_observation_indices.into_boxed_slice(),
                endpoint_touched,
                dividend_jump_touched,
            },
        ))
    }

    fn smoothed_continuous_barrier_bridge(
        &self,
        barrier: &ContinuousBarrierRuntime,
        observations: &[PathObservation],
        spot: f64,
        volatility: f64,
        smoothing: CompactC2Smoothing,
    ) -> Result<ContinuousBarrierBridgeEvaluation, MonteCarloError> {
        let direction = bridge_direction(barrier.direction);
        let initial_endpoint =
            SmoothedBarrierBridgeEndpoint::evaluate(SmoothedBarrierBridgeEndpointInput {
                direction,
                state: spot,
                transformed_barrier: barrier.barrier,
                affine_scale: 1.0,
                smoothing,
            })?;
        let mut hit_factors = vec![smoothed_endpoint_hit_factor(
            initial_endpoint,
            None,
            SmoothedBarrierHitKind::Endpoint,
        )];
        let mut intervals = Vec::with_capacity(barrier.bridge_observation_indices.len());
        let mut interval_observation_indices =
            Vec::with_capacity(barrier.bridge_observation_indices.len());
        let mut previous_state = spot;
        let mut previous_time = 0.0;
        let mut previous_index = None;
        let mut previous_barrier = barrier.barrier;
        let mut previous_scale = 1.0;
        let variance = volatility * volatility;
        for &index in &barrier.bridge_observation_indices {
            let time = self.observation_times[index];
            let state = observations[index].canonical_f;
            let post_coordinate = self.observation_affine_coordinates[index];
            let pre_coordinate =
                self.observation_pre_dividend_coordinates[index].unwrap_or(post_coordinate);
            let pre_barrier = transformed_barrier(barrier.barrier, self.spot, pre_coordinate)?;
            let post_barrier = transformed_barrier(barrier.barrier, self.spot, post_coordinate)?;
            let dt = time - previous_time;
            if dt > 0.0 {
                let interval =
                    SmoothedBarrierBridgeInterval::evaluate(SmoothedBarrierBridgeIntervalInput {
                        left_endpoint: SmoothedBarrierBridgeEndpointInput {
                            direction,
                            state: previous_state,
                            transformed_barrier: previous_barrier,
                            affine_scale: previous_scale,
                            smoothing,
                        },
                        right_endpoint: SmoothedBarrierBridgeEndpointInput {
                            direction,
                            state,
                            transformed_barrier: pre_barrier,
                            affine_scale: pre_coordinate.b(),
                            smoothing,
                        },
                        left_local_variance: variance,
                        right_local_variance: variance,
                        dt,
                    })?;
                if self.observation_pre_dividend_coordinates[index].is_some() {
                    let pre_spot = pre_coordinate.a() * self.spot + pre_coordinate.b() * state;
                    let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * state;
                    hit_factors.push(smoothed_jump_hit_factor(
                        barrier.direction,
                        pre_spot,
                        post_spot,
                        pre_coordinate.b(),
                        post_coordinate.b(),
                        barrier.barrier,
                        smoothing,
                        Some(index),
                    ));
                } else {
                    hit_factors.push(smoothed_endpoint_hit_factor(
                        interval.right_endpoint(),
                        Some(index),
                        SmoothedBarrierHitKind::Endpoint,
                    ));
                }
                intervals.push(interval);
                interval_observation_indices.push((previous_index, index));
            }
            previous_state = state;
            previous_barrier = post_barrier;
            previous_scale = post_coordinate.b();
            previous_time = time;
            previous_index = Some(index);
        }
        Ok(ContinuousBarrierBridgeEvaluation::Smoothed {
            path: SmoothedContinuousBarrierPath::evaluate(intervals, hit_factors),
            interval_observation_indices: interval_observation_indices.into_boxed_slice(),
        })
    }

    fn path_observations_from_normals(
        &self,
        normals: &[f64],
        spot: f64,
        volatility: f64,
    ) -> Vec<PathObservation> {
        let spot_scale = spot / self.spot;
        if self.observation_times.len() == 1 {
            let time = self.observation_times[0];
            let normal = normals[0];
            let total_variance = volatility * volatility * time;
            let standard_deviation = total_variance.sqrt();
            let brownian = time.sqrt() * normal;
            let log_return = -0.5 * total_variance + standard_deviation * normal;
            let canonical_f = self.observation_forwards[0] * spot_scale * log_return.exp();
            return vec![self.path_observation(0, canonical_f, brownian)];
        }
        let mut previous_time = 0.0;
        let mut brownian = 0.0;
        self.observation_times
            .iter()
            .zip(self.observation_forwards.iter())
            .zip(normals.iter())
            .enumerate()
            .map(|(index, ((&time, &forward), &normal))| {
                let step = (time - previous_time).max(0.0);
                brownian += step.sqrt() * normal;
                previous_time = time;
                let total_variance = volatility * volatility * time;
                let canonical_f =
                    forward * spot_scale * (-0.5 * total_variance + volatility * brownian).exp();
                self.path_observation(index, canonical_f, brownian)
            })
            .collect()
    }

    fn path_observation(&self, index: usize, canonical_f: f64, brownian: f64) -> PathObservation {
        let post_coordinate = self.observation_affine_coordinates[index];
        let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * canonical_f;
        let pre_dividend_spot = self.observation_pre_dividend_coordinates[index]
            .map(|coordinate| coordinate.a() * self.spot + coordinate.b() * canonical_f);
        PathObservation {
            post_spot,
            pre_dividend_spot,
            canonical_f,
            brownian,
        }
    }

    fn pathwise_values(
        &self,
        normals: &[f64],
        lane: usize,
        workspace: &mut SoaWorkspace,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let base = self.pathwise_aad(normals, self.spot, self.volatility)?;
        let gamma = if let Some(gamma) = self.request_gamma {
            let bump = resolve_spot_bump(gamma, self.spot);
            let delta_down = self
                .pathwise_aad(normals, self.spot - bump, self.volatility)?
                .delta;
            let delta_up = self
                .pathwise_aad(normals, self.spot + bump, self.volatility)?
                .delta;
            (delta_up - delta_down) / (2.0 * bump)
        } else {
            0.0
        };
        let spot_bump = self.validation_spot_bump;
        let down_spot = self.pathwise_aad(normals, self.spot - spot_bump, self.volatility)?;
        let up_spot = self.pathwise_aad(normals, self.spot + spot_bump, self.volatility)?;
        let bump_delta = (up_spot.price - down_spot.price) / (2.0 * spot_bump);
        let bump_gamma = (up_spot.price - 2.0 * base.price + down_spot.price) / spot_bump.powi(2);
        let bump_vega = if self.validation_volatility_bump == 0.0 {
            0.0
        } else {
            let bump = self.validation_volatility_bump;
            let down = self
                .pathwise_aad(normals, self.spot, self.volatility - bump)?
                .price;
            let up = self
                .pathwise_aad(normals, self.spot, self.volatility + bump)?
                .price;
            (up - down) / (2.0 * bump)
        };
        workspace.primal_mut(0)?.set(lane, base.price)?;
        workspace.primal_mut(1)?.set(lane, base.delta)?;
        workspace.primal_mut(2)?.set(lane, base.vega)?;
        workspace.primal_mut(3)?.set(lane, gamma)?;
        let mut values = [0.0; PATHWISE_COMPONENTS];
        values[PRICE] = workspace.primal(0)?.get(lane)?;
        values[DELTA] = workspace.primal(1)?.get(lane)?;
        values[VEGA] = workspace.primal(2)?.get(lane)?;
        values[GAMMA] = workspace.primal(3)?.get(lane)?;
        values[BUMP_DELTA] = bump_delta;
        values[DELTA_DIFFERENCE] = bump_delta - base.delta;
        values[BUMP_VEGA] = bump_vega;
        values[VEGA_DIFFERENCE] = bump_vega - base.vega;
        values[BUMP_GAMMA] = bump_gamma;
        values[GAMMA_DIFFERENCE] = bump_gamma - gamma;
        if let Some(diagnostics) = base.barrier_diagnostics {
            diagnostics.write_to(&mut values);
        }
        Ok(values)
    }

    fn pathwise_aad(
        &self,
        normals: &[f64],
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        if let Some(barrier) = &self.continuous_barrier {
            return self.continuous_barrier_pathwise_aad(barrier, normals, spot, volatility);
        }
        if self.observation_dates.len() == 1
            && self.observation_dates[0] == Some(self.expiry)
            && self
                .observation_pre_dividend_coordinates
                .iter()
                .all(Option::is_none)
        {
            return self.pathwise_aad_single_terminal(normals[0], spot, volatility);
        }
        let observations = self.path_observations_from_normals(normals, spot, volatility);
        let payoff = self.payoff.evaluate_single_with_observation_adjoints(
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .map(|index| observations[index].post_spot)
            },
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .and_then(|index| observations[index].pre_dividend_spot)
            },
        )?;
        let price = self.discount * payoff.value;
        let mut delta = 0.0;
        let mut vega = 0.0;
        for adjoint in &payoff.terminal_adjoints {
            if adjoint.underlying != self.underlying {
                continue;
            }
            if let Some(index) = self
                .observation_dates
                .iter()
                .position(|observation_date| *observation_date == Some(adjoint.observation_date))
            {
                let observation = observations[index];
                let coordinate = self.observation_affine_coordinates[index];
                delta += adjoint.value * coordinate.b() * observation.canonical_f / spot;
                vega += adjoint.value
                    * coordinate.b()
                    * observation.canonical_f
                    * (-volatility * self.observation_times[index] + observation.brownian);
            }
        }
        for adjoint in &payoff.pre_dividend_adjoints {
            if adjoint.underlying != self.underlying {
                continue;
            }
            if let Some(index) = self
                .observation_dates
                .iter()
                .position(|observation_date| *observation_date == Some(adjoint.observation_date))
            {
                let observation = observations[index];
                let coordinate = self.observation_pre_dividend_coordinates[index]
                    .expect("pre-dividend adjoints have a matching coordinate");
                delta += adjoint.value * coordinate.b() * observation.canonical_f / spot;
                vega += adjoint.value
                    * coordinate.b()
                    * observation.canonical_f
                    * (-volatility * self.observation_times[index] + observation.brownian);
            }
        }
        Ok(PathwiseAad {
            price,
            delta: self.discount * delta,
            vega: self.discount * vega,
            barrier_diagnostics: None,
        })
    }

    fn continuous_barrier_pathwise_aad(
        &self,
        barrier: &ContinuousBarrierRuntime,
        normals: &[f64],
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        let observations = self.path_observations_from_normals(normals, spot, volatility);
        let bridge = self.continuous_barrier_bridge(barrier, &observations, spot, volatility)?;
        let survival = bridge.survival();
        let terminal = observations[barrier.expiry_observation_index].post_spot;
        let payoff = continuous_barrier_payoff_terms(barrier, terminal, survival);
        let mut state_adjoints = vec![0.0; observations.len()];
        let terminal_coordinate =
            self.observation_affine_coordinates[barrier.expiry_observation_index];
        state_adjoints[barrier.expiry_observation_index] +=
            payoff.terminal_derivative * terminal_coordinate.b();

        let mut initial_state_adjoint = 0.0;
        let mut variance_adjoint = 0.0;
        match &bridge {
            ContinuousBarrierBridgeEvaluation::Exact(bridge)
                if !bridge.endpoint_touched && !bridge.dividend_jump_touched =>
            {
                let interval_adjoints = bridge.path.reverse(payoff.survival_derivative * survival);
                for ((left_index, right_index), adjoints) in bridge
                    .interval_observation_indices
                    .iter()
                    .copied()
                    .zip(interval_adjoints)
                {
                    if let Some(left_index) = left_index {
                        state_adjoints[left_index] += adjoints.left_state;
                    } else {
                        initial_state_adjoint += adjoints.left_state;
                    }
                    state_adjoints[right_index] += adjoints.right_state;
                    variance_adjoint +=
                        adjoints.left_local_variance + adjoints.right_local_variance;
                }
            }
            ContinuousBarrierBridgeEvaluation::Smoothed {
                path,
                interval_observation_indices,
            } => {
                let (interval_adjoints, hit_factor_adjoints) =
                    path.reverse(payoff.survival_derivative);
                for ((left_index, right_index), adjoints) in interval_observation_indices
                    .iter()
                    .copied()
                    .zip(interval_adjoints)
                {
                    if let Some(left_index) = left_index {
                        state_adjoints[left_index] += adjoints.left_endpoint.state;
                    } else {
                        initial_state_adjoint += adjoints.left_endpoint.state;
                    }
                    state_adjoints[right_index] += adjoints.right_endpoint.state;
                    variance_adjoint +=
                        adjoints.left_local_variance + adjoints.right_local_variance;
                }
                for (state_index, adjoint) in hit_factor_adjoints {
                    if let Some(index) = state_index {
                        state_adjoints[index] += adjoint;
                    } else {
                        initial_state_adjoint += adjoint;
                    }
                }
            }
            ContinuousBarrierBridgeEvaluation::Exact(_) => {}
        }

        let mut delta = initial_state_adjoint;
        let mut vega = 2.0 * volatility * variance_adjoint;
        for (index, (state_adjoint, observation)) in state_adjoints
            .iter()
            .copied()
            .zip(observations.iter().copied())
            .enumerate()
        {
            delta += state_adjoint * observation.canonical_f / spot;
            vega += state_adjoint
                * observation.canonical_f
                * (-volatility * self.observation_times[index] + observation.brownian);
        }
        Ok(PathwiseAad {
            price: self.discount * payoff.value,
            delta: self.discount * delta,
            vega: self.discount * vega,
            barrier_diagnostics: Some(bridge.diagnostic_values()),
        })
    }

    fn pathwise_aad_single_terminal(
        &self,
        normal: f64,
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        let total_variance = volatility * volatility * self.time;
        let standard_deviation = total_variance.sqrt();
        let log_return = -0.5 * total_variance + standard_deviation * normal;
        let bumped_forward = self.forward * (spot / self.spot);
        let canonical_f = bumped_forward * log_return.exp();
        let coordinate = self.observation_affine_coordinates[0];
        let terminal = coordinate.a() * self.spot + coordinate.b() * canonical_f;
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
        let delta = self.discount * terminal_adjoint * coordinate.b() * canonical_f / spot;
        let terminal_vega =
            coordinate.b() * canonical_f * (-volatility * self.time + self.time.sqrt() * normal);
        let vega = self.discount * terminal_adjoint * terminal_vega;
        Ok(PathwiseAad {
            price,
            delta,
            vega,
            barrier_diagnostics: None,
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
    barrier_diagnostics: Option<BarrierPathDiagnosticValues>,
}

#[derive(Clone, Copy, Debug)]
struct PathObservation {
    post_spot: f64,
    pre_dividend_spot: Option<f64>,
    canonical_f: f64,
    brownian: f64,
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

fn zero_risk_estimate(
    estimator: EstimatorKind,
    independent_units: u64,
    raw_unit: RiskUnit,
    market_scaled_unit: RiskUnit,
) -> Result<RiskEstimate, ResultBuildError> {
    let raw = Estimate::new(0.0, 0.0, 0.0, 0.0, estimator, independent_units)?;
    let market_scaled = Estimate::new(0.0, 0.0, 0.0, 0.0, estimator, independent_units)?;
    Ok(RiskEstimate::new(
        raw,
        market_scaled,
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

fn bucket_sample_capacity(row_capacity: usize, bucket_count: usize) -> usize {
    row_capacity.checked_mul(bucket_count).unwrap_or(0)
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
pub enum PayoffSmoothingKernel {
    CompactC2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayoffValuationKind {
    ExactContractual,
    SmoothedSurrogate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayoffSmoothingWidthUnit {
    Spot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayoffSmoothingDiagnostics {
    pub kernel: PayoffSmoothingKernel,
    pub policy_version: u32,
    pub half_width: PositiveF64,
    pub full_transition_width: PositiveF64,
    pub width_unit: PayoffSmoothingWidthUnit,
    pub price_and_greeks_share_payoff: bool,
    pub endpoint_count: u32,
    pub dividend_jump_count: u32,
}

impl From<PayoffSmoothing> for PayoffSmoothingDiagnostics {
    fn from(value: PayoffSmoothing) -> Self {
        match value {
            PayoffSmoothing::CompactC2 { half_width } => Self {
                kernel: PayoffSmoothingKernel::CompactC2,
                policy_version: PayoffSmoothing::POLICY_VERSION,
                half_width,
                full_transition_width: PositiveF64::new(
                    half_width.get() * 2.0,
                    "payoff_smoothing_full_transition_width",
                )
                .expect("validated smoothing width has a finite double"),
                width_unit: PayoffSmoothingWidthUnit::Spot,
                price_and_greeks_share_payoff: true,
                endpoint_count: 0,
                dividend_jump_count: 0,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathStateDiagnostics {
    ArithmeticAsian {
        known_observation_count: u32,
        unknown_observation_count: u32,
        known_weight_sum: f64,
        unknown_weight_sum: f64,
        weighted_known_fixing_sum: f64,
    },
    FixedLookback {
        past_monitoring_count: u32,
        future_monitoring_count: u32,
        historical_extremum: Option<f64>,
    },
}

impl PathStateDiagnostics {
    fn from_product(product: &ProductSpec, valuation_date: Date) -> Option<Self> {
        match product {
            ProductSpec::ArithmeticAsian(asian) => {
                let mut known_observation_count = 0_u32;
                let mut unknown_observation_count = 0_u32;
                let mut known_weight_sum = 0.0;
                let mut unknown_weight_sum = 0.0;
                let mut weighted_known_fixing_sum = 0.0;
                for observation in asian.observations() {
                    let weight = observation.weight().get();
                    match observation.value() {
                        AsianObservationValue::Known(fixing) => {
                            known_observation_count = known_observation_count.saturating_add(1);
                            known_weight_sum += weight;
                            weighted_known_fixing_sum += weight * fixing.get();
                        }
                        AsianObservationValue::Unknown => {
                            unknown_observation_count = unknown_observation_count.saturating_add(1);
                            unknown_weight_sum += weight;
                        }
                    }
                }
                Some(Self::ArithmeticAsian {
                    known_observation_count,
                    unknown_observation_count,
                    known_weight_sum,
                    unknown_weight_sum,
                    weighted_known_fixing_sum,
                })
            }
            ProductSpec::FixedLookback(lookback) => {
                let past_monitoring_count = u32::try_from(
                    lookback
                        .monitoring_dates()
                        .iter()
                        .filter(|date| **date < valuation_date)
                        .count(),
                )
                .unwrap_or(u32::MAX);
                let future_monitoring_count = u32::try_from(
                    lookback
                        .monitoring_dates()
                        .iter()
                        .filter(|date| **date >= valuation_date)
                        .count(),
                )
                .unwrap_or(u32::MAX);
                Some(Self::FixedLookback {
                    past_monitoring_count,
                    future_monitoring_count,
                    historical_extremum: lookback.historical_extremum().map(PositiveF64::get),
                })
            }
            ProductSpec::EuropeanVanilla(_) | ProductSpec::Digital(_) | ProductSpec::Barrier(_) => {
                None
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierHitIndicatorMode {
    Exact,
    CompactC2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarrierBridgeDiagnostics {
    pub abi: &'static str,
    pub policy_version: u32,
    pub indicator_mode: BarrierHitIndicatorMode,
    pub endpoint_hit_fraction: f64,
    pub dividend_jump_hit_fraction: f64,
    pub mean_conditional_bridge_hit_weight: f64,
    pub mean_interval_count: f64,
    pub mean_finite_correction_count: f64,
    pub mean_zero_variance_count: f64,
    pub mean_survival_underflow_count: f64,
    pub mean_certain_survival_count: f64,
}

impl BarrierBridgeDiagnostics {
    pub const EXACT_POLICY_VERSION: u32 = 1;
    pub const SMOOTHED_POLICY_VERSION: u32 = 2;
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub valuation_kind: PayoffValuationKind,
    pub payoff_smoothing: Option<PayoffSmoothingDiagnostics>,
    pub path_state: Option<PathStateDiagnostics>,
    pub barrier_bridge: Option<BarrierBridgeDiagnostics>,
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

const fn bridge_direction(direction: BarrierDirection) -> BarrierBridgeDirection {
    match direction {
        BarrierDirection::Up => BarrierBridgeDirection::Up,
        BarrierDirection::Down => BarrierBridgeDirection::Down,
    }
}

fn barrier_touched(direction: BarrierDirection, state: f64, barrier: f64) -> bool {
    match direction {
        BarrierDirection::Up => state >= barrier,
        BarrierDirection::Down => state <= barrier,
    }
}

fn smoothed_endpoint_hit_factor(
    endpoint: SmoothedBarrierBridgeEndpoint,
    state_index: Option<usize>,
    kind: SmoothedBarrierHitKind,
) -> SmoothedBarrierHitFactor {
    SmoothedBarrierHitFactor {
        kind,
        state_index,
        hit_weight: endpoint.hit_weight(),
        state_derivative: endpoint.reverse(1.0, 0.0).state,
    }
}

#[allow(clippy::too_many_arguments)]
fn smoothed_jump_hit_factor(
    direction: BarrierDirection,
    pre_spot: f64,
    post_spot: f64,
    pre_affine_scale: f64,
    post_affine_scale: f64,
    barrier: f64,
    smoothing: CompactC2Smoothing,
    state_index: Option<usize>,
) -> SmoothedBarrierHitFactor {
    let direction_sign = match direction {
        BarrierDirection::Up => 1.0,
        BarrierDirection::Down => -1.0,
    };
    let pre_distance = direction_sign * (pre_spot - barrier);
    let post_distance = direction_sign * (post_spot - barrier);
    let score = smoothing.maximum(pre_distance, post_distance);
    let hit = smoothing.indicator(score.value);
    let score_state_derivative = direction_sign
        * (score.left_first * pre_affine_scale + score.right_first * post_affine_scale);
    SmoothedBarrierHitFactor {
        kind: SmoothedBarrierHitKind::DividendJump,
        state_index,
        hit_weight: hit.value,
        state_derivative: hit.first * score_state_derivative,
    }
}

fn accumulate_bridge_local_variance_adjoints(
    local_volatility: &LocalVolRuntime,
    states: &[f64],
    node_interpolations: &[LocalVarianceInterpolation],
    variance_seeds: Vec<f64>,
    state_seeds: &mut [f64],
    bridge_grid_adjoints: &mut [f64],
) {
    let x_count = local_volatility.grid.log_moneyness_nodes().len();
    for (node, (interpolation, variance_seed)) in node_interpolations
        .iter()
        .copied()
        .zip(variance_seeds)
        .enumerate()
    {
        state_seeds[node] += variance_seed
            * local_volatility
                .grid
                .interpolation_log_moneyness_derivative(interpolation)
            / states[node];
        interpolation.transpose_accumulate(variance_seed, bridge_grid_adjoints, x_count);
    }
}

fn continuous_barrier_payoff_terms(
    barrier: &ContinuousBarrierRuntime,
    terminal: f64,
    survival: f64,
) -> ContinuousBarrierPayoffTerms {
    let (signed_intrinsic, terminal_sign) = match barrier.side {
        OptionSide::Call => (terminal - barrier.strike, 1.0),
        OptionSide::Put => (barrier.strike - terminal, -1.0),
    };
    let vanilla = signed_intrinsic.max(0.0) * barrier.notional;
    let vanilla_derivative = if signed_intrinsic >= 0.0 {
        terminal_sign * barrier.notional
    } else {
        0.0
    };
    match barrier.style {
        BarrierStyle::KnockOut => ContinuousBarrierPayoffTerms {
            value: barrier.rebate + survival * (vanilla - barrier.rebate),
            terminal_derivative: survival * vanilla_derivative,
            survival_derivative: vanilla - barrier.rebate,
        },
        BarrierStyle::KnockIn => ContinuousBarrierPayoffTerms {
            value: vanilla + survival * (barrier.rebate - vanilla),
            terminal_derivative: (1.0 - survival) * vanilla_derivative,
            survival_derivative: barrier.rebate - vanilla,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pricing_core::{CurrencyId, CurveId, EventId, PositiveF64};
    use pricing_market::{
        DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
        MarketContext,
    };
    use pricing_mc::{PseudoMcConfig, RqmcConfig, VarianceReduction};
    use pricing_models::{Black76Spec, BlackScholesSpec, LocalVolatilitySpec};
    use pricing_product::{
        ArithmeticAsianSpec, AsianObservation, BarrierDirection, BarrierMonitoring, BarrierSpec,
        BarrierStyle, DigitalPayout, DigitalSpec, EuropeanVanillaSpec, FixedLookbackSpec,
        OptionSide, ProductSpec,
    };
    use pricing_risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};

    use super::*;
    use crate::analytical::{black_76_oracle, black_scholes_oracle};

    #[test]
    fn bucket_sample_capacity_rejects_overflow() {
        assert_eq!(bucket_sample_capacity(3, 4), 12);
        assert_eq!(bucket_sample_capacity(usize::MAX, 2), 0);
    }

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
        barrier_zero_vol_request_with_monitoring(barrier, BarrierMonitoring::Discrete)
    }

    fn barrier_zero_vol_request_with_monitoring(
        barrier: f64,
        monitoring: BarrierMonitoring,
    ) -> PricingRequest {
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
                monitoring,
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

    fn barrier_dividend_jump_request(
        direction: BarrierDirection,
        style: BarrierStyle,
        volatility: f64,
        risk: RiskRequest,
    ) -> PricingRequest {
        barrier_dividend_jump_request_with_monitoring(
            direction,
            style,
            volatility,
            risk,
            BarrierMonitoring::Discrete,
            None,
        )
    }

    fn barrier_dividend_jump_request_with_monitoring(
        direction: BarrierDirection,
        style: BarrierStyle,
        volatility: f64,
        risk: RiskRequest,
        monitoring: BarrierMonitoring,
        barrier_override: Option<f64>,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let valuation_date: Date = "2026-09-04".parse().expect("valuation");
        let dividend_date: Date = "2027-03-05".parse().expect("dividend date");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let barrier = barrier_override.unwrap_or(match direction {
            BarrierDirection::Up => 95.0,
            BarrierDirection::Down => 90.0,
        });
        let monitoring_dates = match monitoring {
            BarrierMonitoring::Discrete => vec![dividend_date, expiry],
            BarrierMonitoring::Continuous => vec![expiry],
        };
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                expiry,
                80.0,
                barrier,
                1.0,
                OptionSide::Call,
                direction,
                style,
                monitoring,
                monitoring_dates,
                None,
                expiry,
            )
            .expect("product"),
        );
        let ex_time = DayCountConvention::Act365F.year_fraction(valuation_date, dividend_date);
        let event = EventId::new(1);
        let dividend_quote = match monitoring {
            BarrierMonitoring::Discrete => {
                DividendQuote::fixed_cash(15.0, event).expect("cash dividend")
            }
            BarrierMonitoring::Continuous => {
                DividendQuote::fixed_cash_and_proportional(15.0, 0.1, event)
                    .expect("affine dividend")
            }
        };
        let forward = EquityForward::with_discrete_dividends(
            underlying,
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(1, 0.0),
            curve(2, 0.0),
            vec![DividendEvent::new(event, ex_time, dividend_quote).expect("dividend")],
        )
        .expect("forward");
        let market = MarketContext::Equity(EquityMarket::new(currency, forward));
        let sampling_units = if volatility == 0.0 { 1 } else { 4_096 };
        PricingRequest::new(
            valuation_date,
            product,
            market,
            ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(
                    0x0123_4567_89ab_cdef,
                    sampling_units,
                    VarianceReduction::new(volatility != 0.0, false),
                )
                .expect("engine"),
            ),
            risk,
        )
        .expect("request")
    }

    fn continuous_barrier_conformance_request(
        direction: BarrierDirection,
        style: BarrierStyle,
        rebate: Option<f64>,
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> PricingRequest {
        let barrier = match direction {
            BarrierDirection::Up => 130.0,
            BarrierDirection::Down => 70.0,
        };
        let base = barrier_dividend_jump_request_with_monitoring(
            direction,
            style,
            0.2,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            BarrierMonitoring::Continuous,
            Some(barrier),
        );
        let source = match base.product() {
            ProductSpec::Barrier(source) => source,
            _ => unreachable!("helper constructs a Barrier"),
        };
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                source.underlying(),
                source.currency(),
                source.expiry(),
                source.strike().get(),
                source.barrier().get(),
                source.notional().get(),
                source.side(),
                source.direction(),
                source.style(),
                source.monitoring(),
                source.monitoring_dates().to_vec(),
                rebate,
                source.payment_date(),
            )
            .expect("continuous Barrier product"),
        );
        PricingRequest::new(
            base.valuation_date(),
            product,
            base.market().clone(),
            base.model().clone(),
            engine,
            risk,
        )
        .expect("continuous Barrier conformance request")
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

    fn asian_risk_request(risk: RiskRequest) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                1.5,
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.25).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                16_384,
                VarianceReduction::new(true, false),
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

    fn asian_rqmc_risk_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                1.5,
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.25).expect("model"));
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                256,
                8,
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
            all_risks(),
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

    fn lookback_risk_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                100.0,
                1.5,
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.25).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                16_384,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            all_risks(),
        )
        .expect("request")
    }

    fn partially_fixed_asian_request(fixing: f64, model: ModelSpec) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                1.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::known("2026-06-04".parse().expect("known date"), 0.4, fixing)
                        .expect("known fixing"),
                    AsianObservation::unknown(expiry, 0.6).expect("unknown fixing"),
                ],
                expiry,
            )
            .expect("Asian"),
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
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4_096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            all_risks(),
        )
        .expect("partially fixed Asian request")
    }

    fn partially_fixed_lookback_request(
        historical_extremum: f64,
        model: ModelSpec,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                200.0,
                1.0,
                OptionSide::Put,
                vec!["2026-06-04".parse().expect("past date"), expiry],
                Some(historical_extremum),
                expiry,
            )
            .expect("Lookback"),
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
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4_096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            all_risks(),
        )
        .expect("partially fixed Lookback request")
    }

    fn fully_fixed_asian_request(engine: EngineConfig) -> PricingRequest {
        fully_fixed_asian_request_with_risk(
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn fully_fixed_asian_request_with_risk(
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> PricingRequest {
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
                    AsianObservation::known("2026-03-04".parse().expect("first"), 0.25, 95.0)
                        .expect("first"),
                    AsianObservation::known("2026-06-04".parse().expect("second"), 0.75, 115.0)
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
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

    fn fully_fixed_lookback_request(engine: EngineConfig) -> PricingRequest {
        fully_fixed_lookback_request_with_risk(
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn fully_fixed_lookback_request_with_risk(
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> PricingRequest {
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
                    "2026-03-04".parse().expect("first"),
                    "2026-06-04".parse().expect("second"),
                ],
                Some(120.0),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
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

    fn fixed_payoff_pseudo_engine() -> EngineConfig {
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                16,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        )
    }

    fn fixed_payoff_rqmc_engine() -> EngineConfig {
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                16,
                4,
                0xfedc_ba98_7654_3210,
                VarianceReduction::new(true, false),
            )
            .expect("RQMC engine"),
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
        constant_local_vol_model_with_volatility(0.2)
    }

    fn constant_local_vol_model_with_volatility(volatility: f64) -> ModelSpec {
        let variance = volatility * volatility;
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                vec![0.0, 1.0],
                vec![-1.0, 1.0],
                vec![variance; 4],
                1.0e-8,
                1.0,
            )
            .expect("local volatility"),
        )
    }

    fn skewed_local_vol_model(time_nodes: Vec<f64>) -> ModelSpec {
        skewed_local_vol_model_with_parallel_shift(time_nodes, 0.0)
    }

    fn skewed_local_vol_model_with_parallel_shift(
        time_nodes: Vec<f64>,
        volatility_shift: f64,
    ) -> ModelSpec {
        let mut values = Vec::with_capacity(time_nodes.len() * 3);
        for _ in &time_nodes {
            values.extend(
                [0.15_f64, 0.2, 0.25].map(|volatility| (volatility + volatility_shift).powi(2)),
            );
        }
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                time_nodes,
                vec![-1.0, 0.0, 1.0],
                values,
                1.0e-8,
                1.0,
            )
            .expect("skewed local volatility"),
        )
    }

    fn continuous_barrier_local_vol_request(
        model: ModelSpec,
        monitoring_dates: Vec<Date>,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                expiry,
                100.0,
                130.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Continuous,
                monitoring_dates,
                None,
                expiry,
            )
            .expect("continuous Barrier"),
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
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(
                    0x0123_4567_89ab_cdef,
                    4_096,
                    VarianceReduction::new(true, true),
                )
                .expect("engine"),
            ),
            risk,
        )
        .expect("request")
    }

    fn local_vol_barrier_dividend_jump_request(
        style: BarrierStyle,
        risk: RiskRequest,
    ) -> PricingRequest {
        let base = barrier_dividend_jump_request(BarrierDirection::Up, style, 0.2, risk);
        PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            constant_local_vol_model(),
            base.engine(),
            base.risk().clone(),
        )
        .expect("Local Volatility Barrier request")
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
            cash_result.diagnostics.valuation_kind,
            PayoffValuationKind::ExactContractual
        );
        assert!(cash_result.diagnostics.payoff_smoothing.is_none());
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
    fn smoothed_digital_reports_aad_risk_and_crn_validation() {
        let base = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Cash);
        let smoothing = PayoffSmoothing::compact_c2(2.0).expect("smoothing");
        let risk = RiskRequest::new(
            true,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(smoothing);
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            risk,
        )
        .expect("request");

        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("price");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        assert!(result.risk_diagnostics.delta_validation.is_some());
        assert!(result.risk_diagnostics.vega_validation.is_some());
        assert_eq!(
            result.diagnostics.valuation_kind,
            PayoffValuationKind::SmoothedSurrogate
        );
        let diagnostics = result.diagnostics.payoff_smoothing.expect("diagnostics");
        assert_eq!(diagnostics.kernel, PayoffSmoothingKernel::CompactC2);
        assert_eq!(diagnostics.policy_version, PayoffSmoothing::POLICY_VERSION);
        assert_eq!(diagnostics.half_width.get(), 2.0);
        assert_eq!(diagnostics.full_transition_width.get(), 4.0);
        assert_eq!(diagnostics.width_unit, PayoffSmoothingWidthUnit::Spot);
        assert!(diagnostics.price_and_greeks_share_payoff);
        assert_eq!(diagnostics.endpoint_count, 1);
        assert_eq!(diagnostics.dividend_jump_count, 0);
    }

    #[test]
    fn digital_zero_volatility_discounts_to_explicit_payment_date() {
        let mut request = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Cash);
        request = PricingRequest::new(
            request.valuation_date(),
            ProductSpec::Digital(
                DigitalSpec::with_payment_date(
                    UnderlyingId::new(1),
                    CurrencyId::new(1),
                    "2027-09-04".parse().expect("expiry"),
                    100.0,
                    10.0,
                    OptionSide::Call,
                    DigitalPayout::Cash,
                    "2027-09-05".parse().expect("payment"),
                )
                .expect("digital"),
            ),
            request.market().clone(),
            request.model().clone(),
            request.engine(),
            request.risk().clone(),
        )
        .expect("request");
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let result = plan.execute().expect("execution");
        assert_eq!(
            result.pricing_result.value.value().get(),
            plan.discount() * 10.0
        );
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
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
    fn continuous_barrier_zero_variance_has_deterministic_bridge_limits() {
        let live = SimulationPlan::compile(
            &barrier_zero_vol_request_with_monitoring(120.0, BarrierMonitoring::Continuous),
            policy(2),
        )
        .expect("continuous plan");
        assert!(live.continuous_barrier.is_some());
        let live_result = live.execute().expect("live execution");
        let expected = live.discount() * (live.observation_forwards[1] - 100.0) * 2.0;
        assert!((live_result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
        let live_diagnostics = live_result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(live_diagnostics.endpoint_hit_fraction, 0.0);
        assert_eq!(live_diagnostics.dividend_jump_hit_fraction, 0.0);
        assert_eq!(live_diagnostics.mean_interval_count, 2.0);
        assert_eq!(live_diagnostics.mean_zero_variance_count, 2.0);

        let touched = SimulationPlan::compile(
            &barrier_zero_vol_request_with_monitoring(50.0, BarrierMonitoring::Continuous),
            policy(2),
        )
        .expect("touched plan");
        let touched_result = touched.execute().expect("touched execution");
        assert_eq!(touched_result.pricing_result.value.value().get(), 0.0);
        let touched_diagnostics = touched_result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(touched_diagnostics.endpoint_hit_fraction, 1.0);
        assert_eq!(touched_diagnostics.mean_interval_count, 0.0);
    }

    #[test]
    fn continuous_barrier_bridge_matches_independent_quadrature_reference() {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                expiry,
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Continuous,
                vec![expiry],
                None,
                expiry,
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
        let request = PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(
                    8192,
                    16,
                    0x1234_5678_9abc_def0,
                    VarianceReduction::new(true, true),
                )
                .expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request");
        let risk_request = PricingRequest::new(
            request.valuation_date(),
            request.product().clone(),
            request.market().clone(),
            request.model().clone(),
            request.engine(),
            all_risks(),
        )
        .expect("continuous Barrier risk request");
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        assert_eq!(plan.observation_times.len(), 1);
        let knock_out = plan.continuous_barrier.clone().expect("continuous Barrier");
        let knock_in = ContinuousBarrierRuntime {
            style: BarrierStyle::KnockIn,
            ..knock_out.clone()
        };
        for normal in [-1.0, 0.0, 0.5] {
            let observations = plan.path_observations_from_normals(&[normal], plan.spot, 0.2);
            let out = plan
                .continuous_barrier_discounted_payoff(&knock_out, &observations)
                .expect("knock out");
            let entered = plan
                .continuous_barrier_discounted_payoff(&knock_in, &observations)
                .expect("knock in");
            let vanilla = plan.discount
                * (observations[knock_out.expiry_observation_index].post_spot - 100.0).max(0.0);
            assert!((out + entered - vanilla).abs() < 1.0e-14);
        }
        for normal in [-1.0, -0.5, 0.0] {
            let normals = [normal];
            let analytic = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0, 0.2)
                .expect("analytic");
            let spot_bump = 1.0e-4;
            let spot_down = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0 - spot_bump, 0.2)
                .expect("spot down")
                .price;
            let spot_up = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0 + spot_bump, 0.2)
                .expect("spot up")
                .price;
            let delta = (spot_up - spot_down) / (2.0 * spot_bump);
            let volatility_bump = 1.0e-5;
            let volatility_down = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0, 0.2 - volatility_bump)
                .expect("volatility down")
                .price;
            let volatility_up = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0, 0.2 + volatility_bump)
                .expect("volatility up")
                .price;
            let vega = (volatility_up - volatility_down) / (2.0 * volatility_bump);
            assert!((analytic.delta - delta).abs() < 2.0e-8);
            assert!((analytic.vega - vega).abs() < 2.0e-7);
        }
        let result = plan.execute().expect("execution");

        let volatility = 0.2;
        let time = plan.time;
        let standard_deviation = volatility * time.sqrt();
        let upper_normal = ((120.0 / plan.forward).ln() + 0.5 * volatility * volatility * time)
            / standard_deviation;
        let lower_normal = -10.0;
        let slices = 200_000_u32;
        let dz = (upper_normal - lower_normal) / f64::from(slices);
        let mut reference = pricing_numerics::NeumaierSum::default();
        for index in 0..slices {
            let normal = lower_normal + (f64::from(index) + 0.5) * dz;
            let terminal = plan.forward
                * (-0.5 * volatility * volatility * time + standard_deviation * normal).exp();
            let survival = 1.0
                - (-2.0 * (120.0_f64 / 100.0).ln() * (120.0 / terminal).ln()
                    / (volatility * volatility * time))
                    .exp();
            let density = (-0.5 * normal * normal).exp() / std::f64::consts::TAU.sqrt();
            reference.add((terminal - 100.0).max(0.0) * survival * density * dz);
        }
        let reference = plan.discount * reference.total();
        let estimate = &result.pricing_result.value;
        assert!(
            (estimate.value().get() - reference).abs()
                <= 8.0 * estimate.standard_error().get() + 5.0e-5,
            "estimate={}, standard_error={}, reference={reference}",
            estimate.value().get(),
            estimate.standard_error().get()
        );

        let risk_result = SimulationPlan::compile(&risk_request, policy(2))
            .expect("risk plan")
            .execute()
            .expect("risk execution");
        assert_eq!(
            risk_result.pricing_result.value.value().get().to_bits(),
            result.pricing_result.value.value().get().to_bits()
        );
        assert!(risk_result.pricing_result.risks.delta.is_some());
        assert!(risk_result.pricing_result.risks.gamma.is_some());
        assert!(risk_result.pricing_result.risks.vega.is_some());
        assert!(risk_result.risk_diagnostics.delta_validation.is_some());
        assert!(risk_result.risk_diagnostics.gamma_validation.is_some());
        assert!(risk_result.risk_diagnostics.vega_validation.is_some());
        let diagnostics = result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(diagnostics.abi, BARRIER_BRIDGE_ABI);
        assert_eq!(diagnostics.indicator_mode, BarrierHitIndicatorMode::Exact);
        assert_eq!(diagnostics.mean_interval_count, 1.0);
        assert!(diagnostics.endpoint_hit_fraction > 0.0);
        assert!(diagnostics.mean_conditional_bridge_hit_weight > 0.0);
        assert_eq!(risk_result.diagnostics.barrier_bridge, Some(diagnostics));
    }

    #[test]
    fn smoothed_continuous_barrier_reports_matched_aad_and_diagnostics() {
        let risk = all_risks().with_payoff_smoothing(
            PayoffSmoothing::compact_c2(2.0).expect("continuous Barrier smoothing"),
        );
        let request = continuous_barrier_conformance_request(
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            Some(7.5),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(29, 4_096, VarianceReduction::new(true, true)).expect("engine"),
            ),
            risk,
        );
        let plan = SimulationPlan::compile(&request, policy(2)).expect("smoothed plan");
        let barrier = plan
            .continuous_barrier
            .as_ref()
            .expect("continuous Barrier");
        let normals = [0.35, -0.2];
        let analytic = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2)
            .expect("analytic");
        let spot_bump = 1.0e-4;
        let spot_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 - spot_bump, 0.2)
            .expect("spot down")
            .price;
        let spot_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 + spot_bump, 0.2)
            .expect("spot up")
            .price;
        let volatility_bump = 1.0e-5;
        let volatility_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 - volatility_bump)
            .expect("volatility down")
            .price;
        let volatility_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 + volatility_bump)
            .expect("volatility up")
            .price;
        assert!((analytic.delta - (spot_up - spot_down) / (2.0 * spot_bump)).abs() < 2.0e-8);
        assert!(
            (analytic.vega - (volatility_up - volatility_down) / (2.0 * volatility_bump)).abs()
                < 3.0e-7
        );

        let result = plan.execute().expect("execution");
        assert_eq!(
            result.diagnostics.valuation_kind,
            PayoffValuationKind::SmoothedSurrogate
        );
        let smoothing = result
            .diagnostics
            .payoff_smoothing
            .expect("smoothing diagnostics");
        assert!(smoothing.price_and_greeks_share_payoff);
        assert_eq!(smoothing.half_width.get(), 2.0);
        assert_eq!(smoothing.endpoint_count, 2);
        assert_eq!(smoothing.dividend_jump_count, 1);
        let bridge = result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(bridge.abi, SMOOTHED_BARRIER_BRIDGE_ABI);
        assert_eq!(
            bridge.policy_version,
            BarrierBridgeDiagnostics::SMOOTHED_POLICY_VERSION
        );
        assert_eq!(bridge.indicator_mode, BarrierHitIndicatorMode::CompactC2);
        assert!(bridge.mean_conditional_bridge_hit_weight > 0.0);
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        assert!(result.risk_diagnostics.delta_validation.is_some());
        assert!(result.risk_diagnostics.gamma_validation.is_some());
        assert!(result.risk_diagnostics.vega_validation.is_some());
    }

    #[test]
    fn smoothed_continuous_up_barrier_rejects_width_outside_log_domain() {
        let request =
            barrier_zero_vol_request_with_monitoring(120.0, BarrierMonitoring::Continuous)
                .replace_risk(
                    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
                        .with_payoff_smoothing(
                            PayoffSmoothing::compact_c2(120.0).expect("smoothing"),
                        ),
                );
        let error = SimulationPlan::compile(&request, policy(2)).expect_err("invalid width");
        assert!(matches!(
            error,
            MonteCarloError::BarrierBridge(BarrierBridgeError::InvalidSmoothedSafeDistance { .. })
        ));
    }

    #[test]
    fn smoothed_continuous_barrier_local_vol_matches_constant_variance_limit() {
        let risk = all_risks().with_payoff_smoothing(
            PayoffSmoothing::compact_c2(2.0).expect("continuous Barrier smoothing"),
        );
        let black_scholes = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.2,
            risk,
            BarrierMonitoring::Continuous,
            Some(70.0),
        );
        let local_vol = PricingRequest::new(
            black_scholes.valuation_date(),
            black_scholes.product().clone(),
            black_scholes.market().clone(),
            constant_local_vol_model(),
            black_scholes.engine(),
            black_scholes.risk().clone(),
        )
        .expect("Local Volatility smoothed continuous Barrier request");
        let black_scholes_result = SimulationPlan::compile(&black_scholes, policy(2))
            .expect("Black-Scholes plan")
            .execute()
            .expect("Black-Scholes execution");
        let local_vol_result = SimulationPlan::compile(&local_vol, policy(2))
            .expect("Local Volatility plan")
            .execute()
            .expect("Local Volatility execution");
        assert!(
            (local_vol_result.pricing_result.value.value().get()
                - black_scholes_result.pricing_result.value.value().get())
            .abs()
                < 1.0e-12
        );
        let black_scholes_vega = black_scholes_result
            .pricing_result
            .risks
            .vega
            .expect("Black-Scholes Vega")
            .raw()
            .value()
            .get();
        let local_vol_vega = local_vol_result
            .pricing_result
            .risks
            .vega
            .expect("Local Volatility Vega")
            .raw()
            .value()
            .get();
        assert!((local_vol_vega - black_scholes_vega).abs() < 1.0e-10);
        assert_eq!(
            local_vol_result
                .diagnostics
                .barrier_bridge
                .expect("bridge diagnostics")
                .indicator_mode,
            BarrierHitIndicatorMode::CompactC2
        );
    }

    #[test]
    fn continuous_barrier_in_out_parity_covers_directions_and_rebates() {
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 16, VarianceReduction::new(true, true)).expect("engine"),
        );
        for smoothing_width in [None, Some(2.0)] {
            for direction in [BarrierDirection::Up, BarrierDirection::Down] {
                for rebate in [None, Some(7.5)] {
                    let risk = smoothing_width.map_or_else(
                        || RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
                        |width| {
                            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
                                .with_payoff_smoothing(
                                    PayoffSmoothing::compact_c2(width).expect("smoothing"),
                                )
                        },
                    );
                    let request = continuous_barrier_conformance_request(
                        direction,
                        BarrierStyle::KnockOut,
                        rebate,
                        engine,
                        risk,
                    );
                    let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
                    let knock_out = plan.continuous_barrier.clone().expect("continuous Barrier");
                    let knock_in = ContinuousBarrierRuntime {
                        style: BarrierStyle::KnockIn,
                        ..knock_out.clone()
                    };

                    for normals in [[-1.25, 0.5], [0.0, 0.0], [0.75, -0.25], [1.5, 1.0]] {
                        let observations = plan.path_observations_from_normals(
                            &normals,
                            plan.spot,
                            plan.volatility,
                        );
                        let out = plan
                            .continuous_barrier_discounted_payoff(&knock_out, &observations)
                            .expect("knock out");
                        let entered = plan
                            .continuous_barrier_discounted_payoff(&knock_in, &observations)
                            .expect("knock in");
                        let terminal = observations[knock_out.expiry_observation_index].post_spot;
                        let vanilla = (terminal - knock_out.strike).max(0.0) * knock_out.notional;
                        let expected = plan.discount * (vanilla + rebate.unwrap_or(0.0));
                        assert!(
                            (out + entered - expected).abs() < 2.0e-13,
                            "smoothing_width={smoothing_width:?}, direction={direction:?}, rebate={rebate:?}, normals={normals:?}, out={out}, in={entered}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn continuous_barrier_replays_across_worker_counts_for_pseudo_mc_and_rqmc() {
        let engines = [
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(0x1234_5678, 2_048, VarianceReduction::new(true, true))
                    .expect("pseudo-MC engine"),
            ),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(512, 8, 0x8765_4321, VarianceReduction::new(true, true))
                    .expect("RQMC engine"),
            ),
        ];
        for smoothing_width in [None, Some(2.0)] {
            for engine in engines {
                let risk = smoothing_width.map_or_else(all_risks, |width| {
                    all_risks().with_payoff_smoothing(
                        PayoffSmoothing::compact_c2(width).expect("smoothing"),
                    )
                });
                let request = continuous_barrier_conformance_request(
                    BarrierDirection::Down,
                    BarrierStyle::KnockIn,
                    Some(7.5),
                    engine,
                    risk,
                );
                let plan =
                    SimulationPlan::compile(&request, policy(1)).expect("single-worker plan");
                assert_eq!(plan.observation_times.len(), 2);
                let mut single = plan.execute().expect("single-worker execution");
                let parallel = SimulationPlan::compile(&request, policy(4))
                    .expect("parallel plan")
                    .execute()
                    .expect("parallel execution");
                assert_ne!(
                    single.diagnostics.worker_threads,
                    parallel.diagnostics.worker_threads
                );
                single.diagnostics.worker_threads = parallel.diagnostics.worker_threads;
                assert_eq!(single, parallel, "smoothing_width={smoothing_width:?}");
            }
        }
    }

    #[test]
    fn continuous_barrier_splits_affine_dividend_jump_without_extra_coordinate() {
        let deterministic = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.0,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            BarrierMonitoring::Continuous,
            Some(90.0),
        );
        let deterministic_plan =
            SimulationPlan::compile(&deterministic, policy(2)).expect("deterministic plan");
        assert_eq!(
            deterministic_plan.observation_dates.as_ref(),
            [None, Some(deterministic.product().expiry())]
        );
        assert_eq!(deterministic_plan.observation_times.len(), 2);
        assert_eq!(
            deterministic_plan
                .continuous_barrier
                .as_ref()
                .expect("continuous Barrier")
                .bridge_observation_indices
                .len(),
            2
        );
        assert!(deterministic_plan.observation_pre_dividend_coordinates[0].is_some());
        let deterministic_result = deterministic_plan
            .execute()
            .expect("deterministic execution");
        assert_eq!(deterministic_result.pricing_result.value.value().get(), 0.0);
        let deterministic_diagnostics = deterministic_result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(deterministic_diagnostics.endpoint_hit_fraction, 0.0);
        assert_eq!(deterministic_diagnostics.dividend_jump_hit_fraction, 1.0);
        assert_eq!(
            deterministic_diagnostics.mean_conditional_bridge_hit_weight,
            0.0
        );

        let source = match deterministic.product() {
            ProductSpec::Barrier(source) => source,
            _ => unreachable!("helper constructs a Barrier"),
        };
        let valuation_monitoring_product = ProductSpec::Barrier(
            BarrierSpec::new(
                source.underlying(),
                source.currency(),
                source.expiry(),
                source.strike().get(),
                source.barrier().get(),
                source.notional().get(),
                source.side(),
                source.direction(),
                source.style(),
                BarrierMonitoring::Continuous,
                vec![deterministic.valuation_date(), source.expiry()],
                source.rebate().map(PositiveF64::get),
                source.payment_date(),
            )
            .expect("valuation-date monitoring product"),
        );
        let valuation_monitoring_request = PricingRequest::new(
            deterministic.valuation_date(),
            valuation_monitoring_product,
            deterministic.market().clone(),
            deterministic.model().clone(),
            deterministic.engine(),
            deterministic.risk().clone(),
        )
        .expect("valuation-date monitoring request");
        let valuation_monitoring_plan =
            SimulationPlan::compile(&valuation_monitoring_request, policy(2))
                .expect("valuation-date monitoring plan");
        assert_eq!(valuation_monitoring_plan.observation_times.len(), 2);
        assert_eq!(
            valuation_monitoring_plan
                .execute()
                .expect("valuation-date execution")
                .pricing_result
                .value
                .value()
                .get()
                .to_bits(),
            deterministic_result
                .pricing_result
                .value
                .value()
                .get()
                .to_bits()
        );

        let request = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.2,
            all_risks(),
            BarrierMonitoring::Continuous,
            Some(70.0),
        );
        let plan = SimulationPlan::compile(&request, policy(2)).expect("stochastic plan");
        let barrier = plan
            .continuous_barrier
            .as_ref()
            .expect("continuous Barrier");
        let normals = [0.75, 0.25];
        let observations = plan.path_observations_from_normals(&normals, 100.0, 0.2);
        let ex_time = plan.observation_times[0];
        let first_state = observations[0].canonical_f;
        let terminal_state = observations[1].canonical_f;
        let transformed_post_dividend_barrier = 85.0 / 0.9;
        let first_survival = 1.0
            - (-2.0 * (100.0_f64 / 70.0).ln() * (first_state / 70.0).ln()
                / (0.2_f64.powi(2) * ex_time))
                .exp();
        let second_survival = 1.0
            - (-2.0
                * (first_state / transformed_post_dividend_barrier).ln()
                * (terminal_state / transformed_post_dividend_barrier).ln()
                / (0.2_f64.powi(2) * (plan.time - ex_time)))
                .exp();
        let expected = plan.discount
            * (observations[1].post_spot - 80.0).max(0.0)
            * first_survival
            * second_survival;
        let actual = plan
            .continuous_barrier_discounted_payoff(barrier, &observations)
            .expect("payoff");
        assert!((actual - expected).abs() < 1.0e-13);

        let analytic = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2)
            .expect("analytic");
        let spot_bump = 1.0e-4;
        let spot_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 - spot_bump, 0.2)
            .expect("spot down")
            .price;
        let spot_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 + spot_bump, 0.2)
            .expect("spot up")
            .price;
        let volatility_bump = 1.0e-5;
        let volatility_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 - volatility_bump)
            .expect("volatility down")
            .price;
        let volatility_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 + volatility_bump)
            .expect("volatility up")
            .price;
        assert!((analytic.delta - (spot_up - spot_down) / (2.0 * spot_bump)).abs() < 2.0e-8);
        assert!(
            (analytic.vega - (volatility_up - volatility_down) / (2.0 * volatility_bump)).abs()
                < 2.0e-7
        );

        let result = plan.execute().expect("risk execution");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
    }

    #[test]
    fn continuous_barrier_local_vol_matches_constant_variance_limit() {
        let black_scholes = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.2,
            all_risks(),
            BarrierMonitoring::Continuous,
            Some(70.0),
        );
        let local_vol = PricingRequest::new(
            black_scholes.valuation_date(),
            black_scholes.product().clone(),
            black_scholes.market().clone(),
            constant_local_vol_model(),
            black_scholes.engine(),
            black_scholes.risk().clone(),
        )
        .expect("Local Volatility continuous Barrier request");
        let black_scholes_result = SimulationPlan::compile(&black_scholes, policy(2))
            .expect("Black-Scholes plan")
            .execute()
            .expect("Black-Scholes execution");
        let local_vol_result = SimulationPlan::compile(&local_vol, policy(2))
            .expect("Local Volatility plan")
            .execute()
            .expect("Local Volatility execution");

        let black_scholes_price = black_scholes_result.pricing_result.value.value().get();
        let local_vol_price = local_vol_result.pricing_result.value.value().get();
        assert!((local_vol_price - black_scholes_price).abs() < 1.0e-12);
        let black_scholes_vega = black_scholes_result
            .pricing_result
            .risks
            .vega
            .as_ref()
            .expect("Black-Scholes Vega")
            .raw()
            .value()
            .get();
        let local_vol_vega = local_vol_result
            .pricing_result
            .risks
            .vega
            .as_ref()
            .expect("Local Volatility Vega")
            .raw()
            .value()
            .get();
        assert!((local_vol_vega - black_scholes_vega).abs() < 1.0e-10);
        assert!(local_vol_result.pricing_result.risks.delta.is_some());
        assert!(local_vol_result.pricing_result.risks.gamma.is_some());
    }

    #[test]
    fn continuous_barrier_local_vol_time_step_refinement_converges() {
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let value = |time_nodes: Vec<f64>| {
            let request = continuous_barrier_local_vol_request(
                skewed_local_vol_model(time_nodes),
                vec![expiry],
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            );
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            let local_volatility = plan.local_volatility.as_ref().expect("Local Volatility");
            let shocks = local_volatility
                .plan
                .time_grid()
                .nodes()
                .windows(2)
                .map(|times| 0.35 * (times[1] - times[0]).sqrt())
                .collect::<Vec<_>>();
            plan.local_vol_discounted_payoff_at_spot(
                local_volatility,
                plan.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("path value")
        };
        let coarse = value(vec![0.0, 1.0]);
        let medium = value(vec![0.0, 0.5, 1.0]);
        let fine = value(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let reference = value((0..=16).map(|index| f64::from(index) / 16.0).collect());
        let coarse_error = (coarse - reference).abs();
        let medium_error = (medium - reference).abs();
        let fine_error = (fine - reference).abs();
        assert!(
            medium_error < coarse_error && fine_error < medium_error,
            "coarse={coarse}, medium={medium}, fine={fine}, reference={reference}"
        );
    }

    #[test]
    fn continuous_barrier_local_vol_monitoring_refinement_converges() {
        let first: Date = "2026-12-04".parse().expect("first");
        let second: Date = "2027-03-05".parse().expect("second");
        let third: Date = "2027-06-04".parse().expect("third");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let value = |monitoring_dates: Vec<Date>, model: ModelSpec| {
            let request = continuous_barrier_local_vol_request(
                model,
                monitoring_dates,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            );
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            let local_volatility = plan.local_volatility.as_ref().expect("Local Volatility");
            let shocks = local_volatility
                .plan
                .time_grid()
                .nodes()
                .windows(2)
                .map(|times| 0.35 * (times[1] - times[0]).sqrt())
                .collect::<Vec<_>>();
            plan.local_vol_discounted_payoff_at_spot(
                local_volatility,
                plan.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("path value")
        };
        let coarse = value(vec![expiry], skewed_local_vol_model(vec![0.0, 1.0]));
        let medium = value(vec![second, expiry], skewed_local_vol_model(vec![0.0, 1.0]));
        let fine = value(
            vec![first, second, third, expiry],
            skewed_local_vol_model(vec![0.0, 1.0]),
        );
        let reference = value(
            vec![expiry],
            skewed_local_vol_model((0..=16).map(|index| f64::from(index) / 16.0).collect()),
        );
        let coarse_error = (coarse - reference).abs();
        let medium_error = (medium - reference).abs();
        let fine_error = (fine - reference).abs();
        assert!(
            medium_error < coarse_error && fine_error < medium_error,
            "coarse={coarse}, medium={medium}, fine={fine}, reference={reference}"
        );
    }

    #[test]
    fn continuous_barrier_local_vol_reverse_matches_parallel_volatility_bump() {
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let risk = RiskRequest::new(
            false,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");
        let time_nodes = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let request = |volatility_shift| {
            continuous_barrier_local_vol_request(
                skewed_local_vol_model_with_parallel_shift(time_nodes.clone(), volatility_shift),
                vec![expiry],
                risk.clone(),
            )
        };
        let base = SimulationPlan::compile(&request(0.0), policy(2)).expect("base plan");
        let bump = 1.0e-5;
        let down = SimulationPlan::compile(&request(-bump), policy(2)).expect("down plan");
        let up = SimulationPlan::compile(&request(bump), policy(2)).expect("up plan");
        let base_runtime = base.local_volatility.as_ref().expect("base runtime");
        let shocks = base_runtime
            .plan
            .time_grid()
            .nodes()
            .windows(2)
            .map(|times| 0.35 * (times[1] - times[0]).sqrt())
            .collect::<Vec<_>>();
        let analytic = base
            .local_vol_pathwise_values_and_buckets(base_runtime, None, &shocks, PathIndex::new(0))
            .expect("analytic")
            .values[VEGA];
        let down_runtime = down.local_volatility.as_ref().expect("down runtime");
        let down_value = down
            .local_vol_discounted_payoff_at_spot(
                down_runtime,
                down.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("down value");
        let up_runtime = up.local_volatility.as_ref().expect("up runtime");
        let up_value = up
            .local_vol_discounted_payoff_at_spot(up_runtime, up.spot, &shocks, PathIndex::new(0))
            .expect("up value");
        let finite_difference = (up_value - down_value) / (2.0 * bump);
        assert!(
            (analytic - finite_difference).abs() < 2.0e-6,
            "analytic={analytic}, finite_difference={finite_difference}"
        );
    }

    #[test]
    fn continuous_barrier_local_vol_reports_vega_kt() {
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let risk = RiskRequest::new(
            false,
            None,
            true,
            Some(
                VegaKtConfig::new(
                    vec!["2027-03-05".parse().expect("first maturity"), expiry],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    false,
                )
                .expect("VegaKT"),
            ),
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");
        let request = continuous_barrier_local_vol_request(
            constant_local_vol_model_with_reporting_basis(),
            vec![expiry],
            risk,
        );
        let result = SimulationPlan::compile(&request, policy(2))
            .expect("plan")
            .execute()
            .expect("execution");
        assert!(result.pricing_result.risks.vega.is_some());
        let vega_kt = result.pricing_result.risks.vega_kt.expect("VegaKT");
        assert_eq!(vega_kt.coordinates().len(), 6);
        assert_eq!(vega_kt.raw_buckets().len(), 6);
        assert!(vega_kt.projection().scalar_vega().get().is_finite());
    }

    #[test]
    fn barrier_dividend_collision_uses_pre_and_post_spot_without_an_extra_dimension() {
        for direction in [BarrierDirection::Up, BarrierDirection::Down] {
            let knock_in = SimulationPlan::compile(
                &barrier_dividend_jump_request(
                    direction,
                    BarrierStyle::KnockIn,
                    0.0,
                    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
                ),
                policy(2),
            )
            .expect("knock-in plan");
            let knock_out = SimulationPlan::compile(
                &barrier_dividend_jump_request(
                    direction,
                    BarrierStyle::KnockOut,
                    0.0,
                    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
                ),
                policy(2),
            )
            .expect("knock-out plan");

            assert_eq!(knock_in.observation_dates.len(), 2);
            assert_eq!(knock_in.payoff.pre_dividend_observations().len(), 1);
            let observations = knock_in.path_observations_from_normals(&[0.0, 0.0], 100.0, 0.0);
            assert!(
                (observations[0].pre_dividend_spot.expect("pre spot") - 100.0).abs() <= 1.0e-12
            );
            assert!((observations[0].post_spot - 85.0).abs() <= 1.0e-12);

            let knock_in_value = knock_in
                .execute()
                .expect("knock-in execution")
                .pricing_result
                .value
                .value()
                .get();
            let knock_out_value = knock_out
                .execute()
                .expect("knock-out execution")
                .pricing_result
                .value
                .value()
                .get();
            assert!((knock_in_value - 5.0).abs() <= 1.0e-12);
            assert_eq!(knock_out_value.to_bits(), 0.0_f64.to_bits());
            assert!((knock_in_value + knock_out_value - 5.0).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn smoothed_barrier_dividend_jump_aad_matches_common_random_number_bumps() {
        let risk = RiskRequest::new(
            true,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(5.0).expect("smoothing"));
        let request =
            barrier_dividend_jump_request(BarrierDirection::Up, BarrierStyle::KnockIn, 0.2, risk);
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let normals = [0.0, 0.0];
        let base = plan
            .pathwise_aad(&normals, plan.spot, plan.volatility)
            .expect("base AAD");
        let spot_bump = 1.0e-4;
        let delta_fd = (plan
            .pathwise_aad(&normals, plan.spot + spot_bump, plan.volatility)
            .expect("up spot")
            .price
            - plan
                .pathwise_aad(&normals, plan.spot - spot_bump, plan.volatility)
                .expect("down spot")
                .price)
            / (2.0 * spot_bump);
        let volatility_bump = 1.0e-5;
        let vega_fd = (plan
            .pathwise_aad(&normals, plan.spot, plan.volatility + volatility_bump)
            .expect("up volatility")
            .price
            - plan
                .pathwise_aad(&normals, plan.spot, plan.volatility - volatility_bump)
                .expect("down volatility")
                .price)
            / (2.0 * volatility_bump);
        assert!((base.delta - delta_fd).abs() <= 1.0e-7);
        assert!((base.vega - vega_fd).abs() <= 1.0e-6);

        let result = plan.execute().expect("execution");
        let diagnostics = result.diagnostics.payoff_smoothing.expect("smoothing");
        assert_eq!(diagnostics.endpoint_count, 2);
        assert_eq!(diagnostics.dividend_jump_count, 1);
    }

    #[test]
    fn local_vol_exact_barrier_detects_pre_dividend_hit() {
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        let knock_in = SimulationPlan::compile(
            &local_vol_barrier_dividend_jump_request(BarrierStyle::KnockIn, risk.clone()),
            policy(2),
        )
        .expect("knock-in plan");
        let knock_out = SimulationPlan::compile(
            &local_vol_barrier_dividend_jump_request(BarrierStyle::KnockOut, risk),
            policy(2),
        )
        .expect("knock-out plan");
        let local_volatility = knock_in
            .local_volatility
            .as_ref()
            .expect("Local Volatility");
        let shocks = vec![0.0; local_volatility.plan.time_grid().step_count()];
        assert_eq!(shocks.len(), 2);
        let path = local_volatility
            .plan
            .evolve_path_with_dividend_checks(
                &local_volatility.grid,
                knock_in.spot,
                &shocks,
                local_volatility.dividends.as_ref().expect("dividends"),
                local_volatility
                    .dividend_schedule
                    .as_ref()
                    .expect("dividend schedule"),
                PathIndex::new(0),
            )
            .expect("path");
        let observations = knock_in
            .local_vol_path_observations(&path)
            .expect("observations");
        assert!(observations[0].pre_dividend_spot.expect("pre spot") >= 95.0);
        assert!(
            observations
                .iter()
                .all(|observation| observation.post_spot < 95.0)
        );

        let knock_in_value = knock_in
            .local_vol_discounted_payoff_at_spot(
                local_volatility,
                knock_in.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("knock-in value");
        let knock_out_runtime = knock_out
            .local_volatility
            .as_ref()
            .expect("Local Volatility");
        let knock_out_value = knock_out
            .local_vol_discounted_payoff_at_spot(
                knock_out_runtime,
                knock_out.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("knock-out value");
        assert!(knock_in_value > 0.0);
        assert_eq!(knock_out_value.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn local_vol_barrier_dividend_jump_uses_event_node_and_reports_risks() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(5.0).expect("smoothing"));
        let request = local_vol_barrier_dividend_jump_request(BarrierStyle::KnockIn, risk);
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let local_volatility = plan.local_volatility.as_ref().expect("Local Volatility");
        let dividend_time = plan.observation_times[0];
        assert_eq!(local_volatility.grid.time_nodes(), [0.0, 1.0]);
        assert!(
            local_volatility
                .plan
                .time_grid()
                .node_index_for_time(dividend_time)
                .is_some()
        );
        assert_eq!(local_volatility.plan.time_grid().step_count(), 2);
        assert_eq!(plan.payoff.pre_dividend_observations().len(), 1);

        let result = plan.execute().expect("execution");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        let diagnostics = result.diagnostics.payoff_smoothing.expect("smoothing");
        assert_eq!(diagnostics.endpoint_count, 2);
        assert_eq!(diagnostics.dividend_jump_count, 1);
    }

    #[test]
    fn smoothed_discrete_barrier_reports_all_risks_and_endpoint_diagnostics() {
        let base = barrier_zero_vol_request(120.0);
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(2.0).expect("smoothing"));
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            risk,
        )
        .expect("request");

        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("price");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        assert!(result.risk_diagnostics.delta_validation.is_some());
        assert!(result.risk_diagnostics.gamma_validation.is_some());
        assert!(result.risk_diagnostics.vega_validation.is_some());
        let diagnostics = result.diagnostics.payoff_smoothing.expect("smoothing");
        assert_eq!(diagnostics.endpoint_count, 2);
        assert_eq!(diagnostics.dividend_jump_count, 0);
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
    fn arithmetic_asian_price_respects_geometric_lower_and_convexity_upper_bounds() {
        let request =
            asian_risk_request(RiskRequest::price_only(SmileDynamics::StickyLogMoneyness));
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let weights = [0.25, 0.75];
        let weighted_log_mean = weights
            .iter()
            .zip(plan.observation_forwards.iter())
            .zip(plan.observation_times.iter())
            .map(|((&weight, &forward), &time)| {
                weight * (forward.ln() - 0.5 * plan.volatility.powi(2) * time)
            })
            .sum::<f64>();
        let mut weighted_minimum_time = 0.0;
        for (left, &left_weight) in weights.iter().enumerate() {
            for (right, &right_weight) in weights.iter().enumerate() {
                weighted_minimum_time += left_weight
                    * right_weight
                    * plan.observation_times[left].min(plan.observation_times[right]);
            }
        }
        let geometric_variance = plan.volatility.powi(2) * weighted_minimum_time;
        let geometric_forward = (weighted_log_mean + 0.5 * geometric_variance).exp();
        let geometric_lower = crate::analytical::evaluate_black_forward(
            crate::analytical::BlackForwardOracleInputs {
                side: OptionSide::Call,
                forward: geometric_forward,
                strike: 100.0,
                notional: 1.5,
                discount: plan.discount,
                volatility: geometric_variance.sqrt(),
                time: 1.0,
            },
        )
        .expect("geometric Asian lower bound")
        .price;
        let convexity_upper = weights
            .iter()
            .zip(plan.observation_forwards.iter())
            .zip(plan.observation_times.iter())
            .map(|((&weight, &forward), &time)| {
                weight
                    * crate::analytical::evaluate_black_forward(
                        crate::analytical::BlackForwardOracleInputs {
                            side: OptionSide::Call,
                            forward,
                            strike: 100.0,
                            notional: 1.5,
                            discount: plan.discount,
                            volatility: plan.volatility,
                            time,
                        },
                    )
                    .expect("European convexity upper bound")
                    .price
            })
            .sum::<f64>();

        let result = plan.execute().expect("Asian result");
        let estimate = result.pricing_result.value;
        let tolerance = 6.0 * estimate.standard_error().get() + 1.0e-12;
        assert!(estimate.value().get() + tolerance >= geometric_lower);
        assert!(estimate.value().get() - tolerance <= convexity_upper);
    }

    #[test]
    fn asian_and_lookback_dividend_collisions_observe_post_jump_spot() {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let valuation_date: Date = "2026-09-04".parse().expect("valuation");
        let dividend_date: Date = "2027-03-05".parse().expect("dividend date");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let event = EventId::new(1);
        let ex_time = DayCountConvention::Act365F.year_fraction(valuation_date, dividend_date);
        let forward = EquityForward::with_discrete_dividends(
            underlying,
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(1, 0.0),
            curve(2, 0.0),
            vec![
                DividendEvent::new(
                    event,
                    ex_time,
                    DividendQuote::fixed_cash(15.0, event).expect("cash dividend"),
                )
                .expect("dividend"),
            ],
        )
        .expect("forward");
        let market = MarketContext::Equity(EquityMarket::new(currency, forward));
        let products = [
            ProductSpec::ArithmeticAsian(
                ArithmeticAsianSpec::new(
                    underlying,
                    currency,
                    80.0,
                    1.0,
                    OptionSide::Call,
                    vec![
                        AsianObservation::unknown(dividend_date, 0.5).expect("first"),
                        AsianObservation::unknown(expiry, 0.5).expect("second"),
                    ],
                    expiry,
                )
                .expect("Asian"),
            ),
            ProductSpec::FixedLookback(
                FixedLookbackSpec::new(
                    underlying,
                    currency,
                    80.0,
                    1.0,
                    OptionSide::Call,
                    vec![dividend_date, expiry],
                    None,
                    expiry,
                )
                .expect("Lookback"),
            ),
        ];
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1, VarianceReduction::new(false, false)).expect("engine"),
        );
        for product in products {
            let request = PricingRequest::new(
                valuation_date,
                product,
                market.clone(),
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model")),
                engine,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            )
            .expect("request");
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            let observations = plan.path_observations_from_normals(&[0.0, 0.0], 100.0, 0.0);
            assert_eq!(observations.len(), 2);
            assert!((observations[0].post_spot - 85.0).abs() <= 1.0e-12);
            assert!(observations[0].pre_dividend_spot.is_none());
            let result = plan.execute().expect("execution");
            assert!((result.pricing_result.value.value().get() - 5.0).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn fully_fixed_arithmetic_asian_prices_as_discounted_cashflow() {
        for engine in [fixed_payoff_pseudo_engine(), fixed_payoff_rqmc_engine()] {
            let request = fully_fixed_asian_request(engine);
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            assert!(plan.observation_dates.is_empty());
            let result = plan.execute().expect("execution");
            let expected = plan.discount() * 20.0;
            assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
            assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
            assert_eq!(
                result.pricing_result.value.standard_error().get().to_bits(),
                0.0_f64.to_bits()
            );
        }
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
    fn fully_fixed_lookback_prices_as_discounted_cashflow() {
        for engine in [fixed_payoff_pseudo_engine(), fixed_payoff_rqmc_engine()] {
            let request = fully_fixed_lookback_request(engine);
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            assert!(plan.observation_dates.is_empty());
            let result = plan.execute().expect("execution");
            let expected = plan.discount() * 40.0;
            assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
            assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
            assert_eq!(
                result.pricing_result.value.standard_error().get().to_bits(),
                0.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn fully_fixed_asian_and_lookback_report_exact_zero_market_risks() {
        for engine in [fixed_payoff_pseudo_engine(), fixed_payoff_rqmc_engine()] {
            for request in [
                fully_fixed_asian_request_with_risk(engine, all_risks()),
                fully_fixed_lookback_request_with_risk(engine, all_risks()),
            ] {
                let result = price_monte_carlo(&request, policy(2)).expect("execution");
                assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
                for estimate in [
                    result.pricing_result.risks.delta.expect("Delta").raw(),
                    result.pricing_result.risks.gamma.expect("Gamma").raw(),
                    result.pricing_result.risks.vega.expect("Vega").raw(),
                ] {
                    assert_eq!(estimate.value().get().to_bits(), 0.0_f64.to_bits());
                    assert_eq!(estimate.standard_error().get().to_bits(), 0.0_f64.to_bits());
                }
            }
        }
    }

    #[test]
    fn path_state_diagnostics_retain_fixed_asian_and_lookback_state() {
        let asian = price_monte_carlo(
            &partially_fixed_asian_request(
                90.0,
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            ),
            policy(1),
        )
        .expect("Asian");
        assert_eq!(
            asian.diagnostics.path_state,
            Some(PathStateDiagnostics::ArithmeticAsian {
                known_observation_count: 1,
                unknown_observation_count: 1,
                known_weight_sum: 0.4,
                unknown_weight_sum: 0.6,
                weighted_known_fixing_sum: 36.0,
            })
        );

        let lookback = price_monte_carlo(
            &partially_fixed_lookback_request(
                92.0,
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            ),
            policy(1),
        )
        .expect("Lookback");
        assert_eq!(
            lookback.diagnostics.path_state,
            Some(PathStateDiagnostics::FixedLookback {
                past_monitoring_count: 1,
                future_monitoring_count: 1,
                historical_extremum: Some(92.0),
            })
        );
    }

    #[test]
    fn partially_fixed_history_is_invariant_under_all_market_risks() {
        for model in [
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            constant_local_vol_model(),
        ] {
            let request_pairs = [
                (
                    partially_fixed_asian_request(90.0, model.clone()),
                    partially_fixed_asian_request(110.0, model.clone()),
                    8.0,
                ),
                (
                    partially_fixed_lookback_request(1.0, model.clone()),
                    partially_fixed_lookback_request(2.0, model.clone()),
                    -1.0,
                ),
            ];
            for (first_request, second_request, expected_price_difference) in request_pairs {
                let first_plan =
                    SimulationPlan::compile(&first_request, policy(2)).expect("first plan");
                let second_plan =
                    SimulationPlan::compile(&second_request, policy(2)).expect("second plan");
                assert_eq!(first_plan.observation_dates.len(), 1);
                assert_eq!(second_plan.observation_dates.len(), 1);
                let first = first_plan.execute().expect("first execution");
                let second = second_plan.execute().expect("second execution");
                let price_difference = second.pricing_result.value.value().get()
                    - first.pricing_result.value.value().get();
                assert!((price_difference - expected_price_difference).abs() < 2.0e-12);

                let first_risks = [
                    first.pricing_result.risks.delta.expect("Delta").raw(),
                    first.pricing_result.risks.gamma.expect("Gamma").raw(),
                    first.pricing_result.risks.vega.expect("Vega").raw(),
                ];
                let second_risks = [
                    second.pricing_result.risks.delta.expect("Delta").raw(),
                    second.pricing_result.risks.gamma.expect("Gamma").raw(),
                    second.pricing_result.risks.vega.expect("Vega").raw(),
                ];
                for (first_risk, second_risk) in first_risks.into_iter().zip(second_risks) {
                    assert!((first_risk.value().get() - second_risk.value().get()).abs() < 2.0e-12);
                    assert!(
                        (first_risk.standard_error().get() - second_risk.standard_error().get())
                            .abs()
                            < 2.0e-14
                    );
                }
            }
        }
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
    fn asian_risks_do_not_change_seeded_price_bits() {
        let price_only =
            asian_risk_request(RiskRequest::price_only(SmileDynamics::StickyLogMoneyness));
        let with_risks = asian_risk_request(all_risks());
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
    fn asian_and_lookback_pathwise_risks_are_reported() {
        for request in [asian_risk_request(all_risks()), lookback_risk_request()] {
            let result = price_pseudo_monte_carlo(&request, policy(4)).expect("risk");
            let risks = &result.pricing_result.risks;
            for estimate in [
                risks.delta.expect("delta").raw(),
                risks.gamma.expect("gamma").raw(),
                risks.vega.expect("vega").raw(),
            ] {
                assert!(estimate.value().get().is_finite());
                assert!(estimate.standard_error().get().is_finite());
            }
            let diagnostics = &result.risk_diagnostics;
            assert_eq!(diagnostics.methods.delta, Some(RiskMethod::AadReverse));
            assert_eq!(
                diagnostics.methods.gamma,
                Some(RiskMethod::CentralBumpOfAadDelta)
            );
            assert_eq!(diagnostics.methods.vega, Some(RiskMethod::AadReverse));
            assert!(
                diagnostics
                    .delta_validation
                    .expect("delta validation")
                    .bump_minus_primary
                    .value()
                    .get()
                    .abs()
                    < 2.0e-3
            );
            assert!(
                diagnostics
                    .vega_validation
                    .expect("vega validation")
                    .bump_minus_primary
                    .value()
                    .get()
                    .abs()
                    < 2.0e-2
            );
        }
    }

    #[test]
    fn asian_and_lookback_local_vol_risks_match_constant_variance_limit() {
        for black_scholes in [asian_risk_request(all_risks()), lookback_risk_request()] {
            let local_vol = PricingRequest::new(
                black_scholes.valuation_date(),
                black_scholes.product().clone(),
                black_scholes.market().clone(),
                constant_local_vol_model_with_volatility(0.25),
                black_scholes.engine(),
                black_scholes.risk().clone(),
            )
            .expect("Local Volatility request");
            let black_scholes_result =
                price_monte_carlo(&black_scholes, policy(4)).expect("Black-Scholes execution");
            let local_vol_result =
                price_monte_carlo(&local_vol, policy(4)).expect("Local Volatility execution");
            let black_scholes_values = [
                black_scholes_result.pricing_result.value.value().get(),
                black_scholes_result
                    .pricing_result
                    .risks
                    .delta
                    .expect("Delta")
                    .raw()
                    .value()
                    .get(),
                black_scholes_result
                    .pricing_result
                    .risks
                    .gamma
                    .expect("Gamma")
                    .raw()
                    .value()
                    .get(),
                black_scholes_result
                    .pricing_result
                    .risks
                    .vega
                    .expect("Vega")
                    .raw()
                    .value()
                    .get(),
            ];
            let local_vol_values = [
                local_vol_result.pricing_result.value.value().get(),
                local_vol_result
                    .pricing_result
                    .risks
                    .delta
                    .expect("Delta")
                    .raw()
                    .value()
                    .get(),
                local_vol_result
                    .pricing_result
                    .risks
                    .gamma
                    .expect("Gamma")
                    .raw()
                    .value()
                    .get(),
                local_vol_result
                    .pricing_result
                    .risks
                    .vega
                    .expect("Vega")
                    .raw()
                    .value()
                    .get(),
            ];
            for ((black_scholes_value, local_vol_value), tolerance) in black_scholes_values
                .into_iter()
                .zip(local_vol_values)
                .zip([2.0e-10, 2.0e-3, 2.0e-3, 2.0e-8])
            {
                assert!(
                    (local_vol_value - black_scholes_value).abs() < tolerance,
                    "Black-Scholes={black_scholes_value}, Local Volatility={local_vol_value}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn asian_pathwise_risks_work_under_rqmc() {
        let result = price_monte_carlo(&asian_rqmc_risk_request(), policy(4)).expect("RQMC risk");
        assert_eq!(
            result.pricing_result.value.estimator(),
            EstimatorKind::RandomizedQuasiMonteCarlo
        );
        assert_eq!(result.independent_sampling_units, 8);
        let risks = &result.pricing_result.risks;
        for estimate in [
            risks.delta.expect("delta").raw(),
            risks.gamma.expect("gamma").raw(),
            risks.vega.expect("vega").raw(),
        ] {
            assert!(estimate.value().get().is_finite());
            assert!(estimate.standard_error().get().is_finite());
            assert_eq!(
                estimate.estimator(),
                EstimatorKind::RandomizedQuasiMonteCarlo
            );
        }
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
