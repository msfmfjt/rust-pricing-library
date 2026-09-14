//! risk / report implementation.

use crate::api::mc_result::BarrierBridgeDiagnostics;
use crate::api::mc_result::BarrierHitIndicatorMode;
use crate::api::mc_result::BumpValidationPolicy;
use crate::api::mc_result::ExerciseStrategyRisk;
use crate::api::mc_result::PayoffSmoothingDiagnostics;
use crate::api::mc_result::RiskDiagnostics;
use crate::api::mc_result::RiskMethod;
use crate::api::mc_result::RiskMethodMetadata;
use crate::api::mc_result::RiskValidation;
use crate::api::mc_result::StoppingIndexRisk;
use crate::core::SchemaVersion;
use crate::engine::plan::simulation::BARRIER_BRIDGE_HIT_WEIGHT;
use crate::engine::plan::simulation::BARRIER_CERTAIN_SURVIVAL_COUNT;
use crate::engine::plan::simulation::BARRIER_DIVIDEND_JUMP_HIT;
use crate::engine::plan::simulation::BARRIER_ENDPOINT_HIT;
use crate::engine::plan::simulation::BARRIER_FINITE_CORRECTION_COUNT;
use crate::engine::plan::simulation::BARRIER_INTERVAL_COUNT;
use crate::engine::plan::simulation::BARRIER_SURVIVAL_UNDERFLOW_COUNT;
use crate::engine::plan::simulation::BARRIER_ZERO_VARIANCE_COUNT;
use crate::engine::plan::simulation::BUMP_DELTA;
use crate::engine::plan::simulation::BUMP_GAMMA;
use crate::engine::plan::simulation::BUMP_VEGA;
use crate::engine::plan::simulation::DELTA;
use crate::engine::plan::simulation::DELTA_DIFFERENCE;
use crate::engine::plan::simulation::GAMMA;
use crate::engine::plan::simulation::GAMMA_DIFFERENCE;
use crate::engine::plan::simulation::LocalVolRuntime;
use crate::engine::plan::simulation::NORMAL_95;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::SMOOTHED_BARRIER_BRIDGE_ABI;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::plan::simulation::VEGA;
use crate::engine::plan::simulation::VEGA_DIFFERENCE;
use crate::engine::plan::simulation::resolve_spot_bump;
use crate::market::CurveRegion;
use crate::mc::{BARRIER_BRIDGE_ABI, DeterministicStatistics, ExercisePolicyFingerprint};
use crate::risk::{
    SmileDynamics, vega_kt_bucket_estimates, vega_kt_full_bucket_covariance,
    vega_kt_projection_from_parts, vega_kt_report,
};
use crate::{
    Estimate, EstimatorKind, MonteCarloError, PricingWarning, ReplayMetadata, ResultBuildError,
    RiskEstimate, RiskReport, RiskUnit, VegaKtResult,
};

impl SimulationPlan {
    pub(in crate::engine) fn payoff_smoothing_diagnostics(
        &self,
    ) -> Option<PayoffSmoothingDiagnostics> {
        self.payoff_smoothing.map(|smoothing| {
            let mut diagnostics = PayoffSmoothingDiagnostics::from(smoothing);
            diagnostics.endpoint_count = self.payoff_smoothing_endpoint_count;
            diagnostics.dividend_jump_count = self.payoff_smoothing_dividend_jump_count;
            diagnostics
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn barrier_bridge_diagnostics(
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
}

impl SimulationPlan {
    pub(in crate::engine) fn replay_metadata(&self) -> ReplayMetadata {
        ReplayMetadata::with_migration(
            SchemaVersion::CURRENT,
            self.request_fingerprint,
            crate::version(),
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            self.request_migration.clone(),
        )
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn fixed_payoff_risk_report(
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
}

impl SimulationPlan {
    pub(in crate::engine) fn fixed_payoff_risk_diagnostics(
        &self,
        estimator: EstimatorKind,
        independent_units: u64,
    ) -> Result<RiskDiagnostics, MonteCarloError> {
        let zero_statistics =
            [DeterministicStatistics::from_ordered_values_two_pass(&[0.0]); PATHWISE_COMPONENTS];
        self.build_risk_diagnostics(&zero_statistics, independent_units, estimator)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn build_vega_kt_result(
        &self,
        local_volatility: &LocalVolRuntime,
        statistics: &[DeterministicStatistics; PATHWISE_COMPONENTS],
        independent_units: u64,
        price_samples: &[f64],
        raw_bucket_samples: &[f64],
    ) -> Result<VegaKtResult, MonteCarloError> {
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let bucket_count = vega_kt.basis.bucket_count();
        let estimates = vega_kt_bucket_estimates(price_samples, raw_bucket_samples, bucket_count)?;
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
                raw_bucket_samples,
                bucket_count,
            )?)
        } else {
            None
        };
        let density_row = vega_kt
            .density_rows
            .last()
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        Ok(VegaKtResult::try_from(&vega_kt_report(
            &vega_kt.basis,
            density_row,
            projection,
            estimates,
            full_bucket_covariance,
        )?)?)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn build_risk_report(
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
}

impl SimulationPlan {
    pub(in crate::engine) fn build_risk_diagnostics(
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
                exercise_strategy: None,
                stopping_indices: None,
                exercise_policy_fingerprint: None,
            },
            delta_validation,
            gamma_validation,
            vega_validation,
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn build_local_vol_risk_diagnostics(&self) -> RiskDiagnostics {
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
                exercise_strategy: None,
                stopping_indices: None,
                exercise_policy_fingerprint: None,
            },
            delta_validation: None,
            gamma_validation: None,
            vega_validation: None,
        }
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn build_fixed_policy_risk_diagnostics(
        &self,
        statistics: &[DeterministicStatistics; PATHWISE_COMPONENTS],
        independent_units: u64,
        estimator: EstimatorKind,
        policy_fingerprint: ExercisePolicyFingerprint,
    ) -> Result<RiskDiagnostics, MonteCarloError> {
        let mut diagnostics =
            self.build_risk_diagnostics(statistics, independent_units, estimator)?;
        diagnostics.methods.exercise_strategy = Some(ExerciseStrategyRisk::FixedExerciseStrategy);
        diagnostics.methods.stopping_indices = Some(StoppingIndexRisk::FrozenStoppingIndices);
        diagnostics.methods.exercise_policy_fingerprint = Some(policy_fingerprint);
        Ok(diagnostics)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn build_local_vol_fixed_policy_risk_diagnostics(
        &self,
        policy_fingerprint: ExercisePolicyFingerprint,
    ) -> RiskDiagnostics {
        let mut diagnostics = self.build_local_vol_risk_diagnostics();
        diagnostics.methods.exercise_strategy = Some(ExerciseStrategyRisk::FixedExerciseStrategy);
        diagnostics.methods.stopping_indices = Some(StoppingIndexRisk::FrozenStoppingIndices);
        diagnostics.methods.exercise_policy_fingerprint = Some(policy_fingerprint);
        diagnostics
    }
}

pub(in crate::engine) fn empty_risk_diagnostics(smile_dynamics: SmileDynamics) -> RiskDiagnostics {
    RiskDiagnostics {
        methods: RiskMethodMetadata {
            delta: None,
            gamma: None,
            vega: None,
            smile_dynamics,
            gamma_spot_bump: None,
            validation_spot_bump: None,
            validation_volatility_bump: None,
            bump_policy_version: BumpValidationPolicy::VERSION,
            exercise_strategy: None,
            stopping_indices: None,
            exercise_policy_fingerprint: None,
        },
        delta_validation: None,
        gamma_validation: None,
        vega_validation: None,
    }
}

pub(in crate::engine) fn estimate_from_statistics(
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

pub(in crate::engine) fn risk_estimate(
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

pub(in crate::engine) fn zero_risk_estimate(
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

pub(in crate::engine) fn risk_validation(
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

pub(in crate::engine) fn extrapolation_warnings(
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
