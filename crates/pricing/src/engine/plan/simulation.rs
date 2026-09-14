//! plan / simulation implementation.

use crate::api::mc_result::PathStateDiagnostics;
use crate::core::{Date, UnderlyingId};
use crate::engine::aad::{AadTilePolicy, CheckpointPolicy};
use crate::market::{
    AffineDividendCoordinate, AffineDividendTransform, CurveRegion, EquityForward,
    LocalVarianceGrid, LocalVarianceInterpolation,
};
use crate::mc::{
    BarrierBridgePath, EngineConfig, ExecutionPolicy, LocalVolDividendCheckpointSchedule,
    LocalVolLogEulerPlan, LsmConfig, SmoothedBarrierBridgeInterval,
};
use crate::models::LocalVolatilityReportingBasis;
use crate::product::{
    BarrierDirection, BarrierStyle, CompiledPayoff, GraphFingerprint, OptionSide,
};
use crate::risk::{
    AnalyticCallDensityRow, GammaConfig, PayoffSmoothing, ReportingIvBasis, SmileDynamics,
    VegaKtConfig,
};
use crate::{Fingerprint, MigrationProvenance};

pub(in crate::engine) const NORMAL_95: f64 = 1.959_963_984_540_054;

pub(in crate::engine) const PRICE: usize = 0;

pub(in crate::engine) const DELTA: usize = 1;

pub(in crate::engine) const VEGA: usize = 2;

pub(in crate::engine) const GAMMA: usize = 3;

pub(in crate::engine) const BUMP_DELTA: usize = 4;

pub(in crate::engine) const DELTA_DIFFERENCE: usize = 5;

pub(in crate::engine) const BUMP_VEGA: usize = 6;

pub(in crate::engine) const VEGA_DIFFERENCE: usize = 7;

pub(in crate::engine) const BUMP_GAMMA: usize = 8;

pub(in crate::engine) const GAMMA_DIFFERENCE: usize = 9;

pub(in crate::engine) const BARRIER_ENDPOINT_HIT: usize = 10;

pub(in crate::engine) const BARRIER_DIVIDEND_JUMP_HIT: usize = 11;

pub(in crate::engine) const BARRIER_BRIDGE_HIT_WEIGHT: usize = 12;

pub(in crate::engine) const BARRIER_INTERVAL_COUNT: usize = 13;

pub(in crate::engine) const BARRIER_FINITE_CORRECTION_COUNT: usize = 14;

pub(in crate::engine) const BARRIER_ZERO_VARIANCE_COUNT: usize = 15;

pub(in crate::engine) const BARRIER_SURVIVAL_UNDERFLOW_COUNT: usize = 16;

pub(in crate::engine) const BARRIER_CERTAIN_SURVIVAL_COUNT: usize = 17;

pub(in crate::engine) const PATHWISE_COMPONENTS: usize = 18;

pub(in crate::engine) const AAD_WORKSPACE_SLOTS: usize = 5;

pub(in crate::engine) const PLAN_FINGERPRINT_VERSION: u32 = 1;

pub(in crate::engine) const SMOOTHED_BARRIER_BRIDGE_ABI: &str =
    "continuous-barrier-bridge-log-survival-v2";

/// Immutable one-expiry plan for the European Black-Scholes MC/RQMC slice.
#[derive(Clone, Debug)]
pub struct SimulationPlan {
    pub(in crate::engine) valuation_date: Date,
    pub(in crate::engine) expiry: Date,
    pub(in crate::engine) underlying: UnderlyingId,
    pub(in crate::engine) time: f64,
    pub(in crate::engine) forward: f64,
    pub(in crate::engine) spot: f64,
    pub(in crate::engine) discount: f64,
    pub(in crate::engine) volatility: f64,
    pub(in crate::engine) total_variance: f64,
    pub(in crate::engine) payoff: CompiledPayoff,
    pub(in crate::engine) observation_dates: Box<[Option<Date>]>,
    pub(in crate::engine) observation_times: Box<[f64]>,
    pub(in crate::engine) observation_forwards: Box<[f64]>,
    pub(in crate::engine) observation_affine_coordinates: Box<[AffineDividendCoordinate]>,
    pub(in crate::engine) observation_pre_dividend_coordinates:
        Box<[Option<AffineDividendCoordinate>]>,
    pub(in crate::engine) observation_local_vol_node_indices: Option<Box<[usize]>>,
    pub(in crate::engine) engine: EngineConfig,
    pub(in crate::engine) execution_policy: ExecutionPolicy,
    pub(in crate::engine) aad_tile_policy: AadTilePolicy,
    pub(in crate::engine) checkpoint_policy: CheckpointPolicy,
    pub(in crate::engine) request_delta: bool,
    pub(in crate::engine) request_gamma: Option<GammaConfig>,
    pub(in crate::engine) request_vega: bool,
    pub(in crate::engine) payoff_smoothing: Option<PayoffSmoothing>,
    pub(in crate::engine) payoff_smoothing_endpoint_count: u32,
    pub(in crate::engine) payoff_smoothing_dividend_jump_count: u32,
    pub(in crate::engine) path_state_diagnostics: Option<PathStateDiagnostics>,
    pub(in crate::engine) smile_dynamics: SmileDynamics,
    pub(in crate::engine) validation_spot_bump: f64,
    pub(in crate::engine) validation_volatility_bump: f64,
    pub(in crate::engine) request_fingerprint: [u8; 32],
    pub(in crate::engine) request_migration: MigrationProvenance,
    pub(in crate::engine) plan_fingerprint: Fingerprint,
    pub(in crate::engine) discount_region: CurveRegion,
    pub(in crate::engine) dividend_region: CurveRegion,
    pub(in crate::engine) market_forward: EquityForward,
    pub(in crate::engine) local_volatility: Option<LocalVolRuntime>,
    pub(in crate::engine) continuous_barrier: Option<ContinuousBarrierRuntime>,
    pub(in crate::engine) early_exercise: Option<EarlyExerciseRuntime>,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct LocalVolRuntime {
    pub(in crate::engine) grid: LocalVarianceGrid,
    pub(in crate::engine) plan: LocalVolLogEulerPlan,
    pub(in crate::engine) node_affine_coordinates: Box<[AffineDividendCoordinate]>,
    pub(in crate::engine) node_pre_dividend_coordinates: Box<[Option<AffineDividendCoordinate>]>,
    pub(in crate::engine) dividends: Option<AffineDividendTransform>,
    pub(in crate::engine) dividend_schedule: Option<LocalVolDividendCheckpointSchedule>,
    pub(in crate::engine) vega_kt: Option<LocalVolVegaKtRuntime>,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct LocalVolVegaKtRuntime {
    pub(in crate::engine) basis: ReportingIvBasis,
    pub(in crate::engine) density_rows: Vec<AnalyticCallDensityRow>,
    pub(in crate::engine) full_bucket_covariance: bool,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct ContinuousBarrierRuntime {
    pub(in crate::engine) strike: f64,
    pub(in crate::engine) barrier: f64,
    pub(in crate::engine) notional: f64,
    pub(in crate::engine) rebate: f64,
    pub(in crate::engine) side: OptionSide,
    pub(in crate::engine) direction: BarrierDirection,
    pub(in crate::engine) style: BarrierStyle,
    pub(in crate::engine) monitoring_end_time: f64,
    pub(in crate::engine) bridge_observation_indices: Box<[usize]>,
    pub(in crate::engine) expiry_observation_index: usize,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct EarlyExerciseRuntime {
    pub(in crate::engine) config: LsmConfig,
    pub(in crate::engine) exercise_dates: Box<[Date]>,
    pub(in crate::engine) observation_indices: Box<[usize]>,
    pub(in crate::engine) discount_factors: Box<[f64]>,
    pub(in crate::engine) dividend_collisions: Box<[bool]>,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct ExactContinuousBarrierBridgeEvaluation {
    pub(in crate::engine) path: BarrierBridgePath,
    pub(in crate::engine) interval_observation_indices: Box<[(Option<usize>, usize)]>,
    pub(in crate::engine) endpoint_touched: bool,
    pub(in crate::engine) dividend_jump_touched: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::engine) enum SmoothedBarrierHitKind {
    Endpoint,
    DividendJump,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct SmoothedBarrierHitFactor {
    pub(in crate::engine) kind: SmoothedBarrierHitKind,
    pub(in crate::engine) state_index: Option<usize>,
    pub(in crate::engine) hit_weight: f64,
    pub(in crate::engine) state_derivative: f64,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct SmoothedContinuousBarrierPath {
    pub(in crate::engine) intervals: Box<[SmoothedBarrierBridgeInterval]>,
    pub(in crate::engine) hit_factors: Box<[SmoothedBarrierHitFactor]>,
    pub(in crate::engine) log_bridge_survival: f64,
    pub(in crate::engine) log_hit_survival: f64,
    pub(in crate::engine) finite_correction_count: u32,
    pub(in crate::engine) zero_variance_count: u32,
    pub(in crate::engine) survival_underflow_count: u32,
    pub(in crate::engine) certain_survival_count: u32,
}

#[derive(Clone, Debug)]
pub(in crate::engine) enum ContinuousBarrierBridgeEvaluation {
    Exact(ExactContinuousBarrierBridgeEvaluation),
    Smoothed {
        path: SmoothedContinuousBarrierPath,
        interval_observation_indices: Box<[(Option<usize>, usize)]>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct ContinuousBarrierPayoffTerms {
    pub(in crate::engine) value: f64,
    pub(in crate::engine) terminal_derivative: f64,
    pub(in crate::engine) survival_derivative: f64,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct ExactLocalVolContinuousBarrierBridgeEvaluation {
    pub(in crate::engine) path: BarrierBridgePath,
    pub(in crate::engine) interval_node_indices: Box<[(usize, usize)]>,
    pub(in crate::engine) node_interpolations: Box<[LocalVarianceInterpolation]>,
    pub(in crate::engine) endpoint_touched: bool,
    pub(in crate::engine) dividend_jump_touched: bool,
}

#[derive(Clone, Debug)]
pub(in crate::engine) enum LocalVolContinuousBarrierBridgeEvaluation {
    Exact(ExactLocalVolContinuousBarrierBridgeEvaluation),
    Smoothed {
        path: SmoothedContinuousBarrierPath,
        interval_node_indices: Box<[(usize, usize)]>,
        node_interpolations: Box<[LocalVarianceInterpolation]>,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::engine) struct BarrierPathDiagnosticValues {
    pub(in crate::engine) endpoint_hit: f64,
    pub(in crate::engine) dividend_jump_hit: f64,
    pub(in crate::engine) bridge_hit_weight: f64,
    pub(in crate::engine) interval_count: f64,
    pub(in crate::engine) finite_correction_count: f64,
    pub(in crate::engine) zero_variance_count: f64,
    pub(in crate::engine) survival_underflow_count: f64,
    pub(in crate::engine) certain_survival_count: f64,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct LocalVolPathwise {
    pub(in crate::engine) values: [f64; PATHWISE_COMPONENTS],
    pub(in crate::engine) raw_buckets: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct ExerciseCashflowSelection {
    pub(in crate::engine) output_index: usize,
    pub(in crate::engine) discount: f64,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct LocalVolObservation {
    pub(in crate::engine) post_spot: f64,
    pub(in crate::engine) pre_dividend_spot: Option<f64>,
    pub(in crate::engine) node_index: usize,
}

pub(in crate::engine) struct LocalVolRuntimeInputs<'a> {
    pub(in crate::engine) grid: LocalVarianceGrid,
    pub(in crate::engine) reporting_iv_basis: Option<&'a LocalVolatilityReportingBasis>,
    pub(in crate::engine) vega_kt: Option<&'a VegaKtConfig>,
    pub(in crate::engine) market_forward: &'a EquityForward,
    pub(in crate::engine) valuation_date: Date,
    pub(in crate::engine) expiry_time: f64,
    pub(in crate::engine) event_times: &'a [f64],
}

pub(in crate::engine) struct ReportingIvSurface<'a> {
    pub(in crate::engine) basis: &'a LocalVolatilityReportingBasis,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct LocalVolBumpRuntimes {
    pub(in crate::engine) down_spot: f64,
    pub(in crate::engine) down: LocalVolRuntime,
    pub(in crate::engine) up_spot: f64,
    pub(in crate::engine) up: LocalVolRuntime,
    pub(in crate::engine) spot_bump: f64,
}

impl SimulationPlan {
    #[must_use]
    pub const fn valuation_date(&self) -> Date {
        self.valuation_date
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn forward(&self) -> f64 {
        self.forward
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn discount(&self) -> f64 {
        self.discount
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn total_variance(&self) -> f64 {
        self.total_variance
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn payoff_fingerprint(&self) -> GraphFingerprint {
        self.payoff.tape_fingerprint()
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn request_fingerprint(&self) -> Fingerprint {
        Fingerprint::from_bytes(self.request_fingerprint)
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.plan_fingerprint
    }
}

impl SimulationPlan {
    #[must_use]
    pub const fn request_migration(&self) -> &MigrationProvenance {
        &self.request_migration
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn risk_enabled(&self) -> bool {
        self.request_delta || self.request_gamma.is_some() || self.request_vega
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct PathwiseAad {
    pub(in crate::engine) price: f64,
    pub(in crate::engine) delta: f64,
    pub(in crate::engine) vega: f64,
    pub(in crate::engine) barrier_diagnostics: Option<BarrierPathDiagnosticValues>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct PathObservation {
    pub(in crate::engine) post_spot: f64,
    pub(in crate::engine) pre_dividend_spot: Option<f64>,
    pub(in crate::engine) canonical_f: f64,
    pub(in crate::engine) brownian: f64,
}

pub(in crate::engine) struct LsmPathMatrices {
    pub(in crate::engine) path_count: usize,
    pub(in crate::engine) immediate_values: Box<[f64]>,
    pub(in crate::engine) features: Box<[f64]>,
}

pub(in crate::engine) fn bucket_sample_capacity(row_capacity: usize, bucket_count: usize) -> usize {
    row_capacity.checked_mul(bucket_count).unwrap_or(0)
}

use crate::risk::SpotBump;
pub(in crate::engine) fn resolve_spot_bump(gamma: GammaConfig, spot: f64) -> f64 {
    match gamma.bump() {
        SpotBump::Absolute(value) => value.get(),
        SpotBump::Relative(value) => spot * value.get(),
    }
}
