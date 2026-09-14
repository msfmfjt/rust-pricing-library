//! api / mc result implementation.

use crate::core::{Date, PositiveF64};
use crate::market::CurveRegion;
use crate::mc::{
    ExerciseDecisionModel, ExercisePolicyFingerprint, ExerciseRegressionDiagnostics,
    PolynomialBasisSpec, RandomDomain,
};
use crate::product::{AsianObservationValue, GraphFingerprint, ProductSpec};
use crate::risk::{PayoffSmoothing, SmileDynamics};
use crate::{Estimate, EstimatorKind, PricingResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RiskMethod {
    AadReverse,
    CentralBump,
    CentralBumpOfAadDelta,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExerciseStrategyRisk {
    FixedExerciseStrategy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoppingIndexRisk {
    FrozenStoppingIndices,
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
    pub exercise_strategy: Option<ExerciseStrategyRisk>,
    pub stopping_indices: Option<StoppingIndexRisk>,
    pub exercise_policy_fingerprint: Option<ExercisePolicyFingerprint>,
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
    pub(crate) fn from_product(product: &ProductSpec, valuation_date: Date) -> Option<Self> {
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
            ProductSpec::EuropeanVanilla(_)
            | ProductSpec::AmericanVanilla(_)
            | ProductSpec::Digital(_)
            | ProductSpec::Barrier(_) => None,
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
pub struct EarlyExerciseDiagnostics {
    pub policy_fingerprint: ExercisePolicyFingerprint,
    pub training_random_domain: RandomDomain,
    pub valuation_random_domain: RandomDomain,
    pub training_direction_checksum: Option<[u8; 32]>,
    pub training_scramble_checksum: Option<[u8; 32]>,
    pub valuation_direction_checksum: Option<[u8; 32]>,
    pub valuation_scramble_checksum: Option<[u8; 32]>,
    pub training_sampling_units: u64,
    pub training_trajectories: u64,
    pub valuation_sampling_units: u64,
    pub valuation_trajectories: u64,
    pub in_sample_value: f64,
    pub exercise_dates: Box<[Date]>,
    pub exercise_counts: Box<[usize]>,
    pub exercise_probabilities: Box<[f64]>,
    pub stopping_indices: Box<[usize]>,
    pub dividend_collisions: Box<[bool]>,
    pub regression_diagnostics: Box<[ExerciseRegressionDiagnostics]>,
    pub policy_basis: PolynomialBasisSpec,
    pub itm_abs_tolerance: f64,
    pub cpqr_config: crate::mc::CpqrConfig,
    pub max_matrix_elements: usize,
    pub decision_models: Box<[ExerciseDecisionModel]>,
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
    pub early_exercise_diagnostics: Option<EarlyExerciseDiagnostics>,
}

pub(crate) const DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP: f64 = 1.0e-4;
pub(crate) const DEFAULT_VALIDATION_VOLATILITY_BUMP: f64 = 1.0e-4;
