use pricing::market::CurveRegion;
use pricing::{
    BarrierHitIndicatorMode, Estimate, EstimatorKind, ExerciseStrategyRisk, MonteCarloPrice,
    PathStateDiagnostics, PayoffSmoothingKernel, PayoffSmoothingWidthUnit, PayoffValuationKind,
    RiskMethod, RiskValidation, StoppingIndexRisk,
};
use pyo3::prelude::*;

use crate::format_fingerprint;

/// A deterministic warning emitted by a completed valuation.
#[pyclass(frozen, name = "PricingWarning", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PyPricingWarning {
    code: String,
    message: String,
}

#[pymethods]
impl PyPricingWarning {
    /// Stable machine-readable warning code.
    #[getter]
    fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable warning detail.
    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingWarning(code={:?}, message={:?})",
            self.code, self.message
        )
    }
}

/// One statistical estimate used by risk validation diagnostics.
#[pyclass(frozen, name = "DiagnosticEstimate", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyDiagnosticEstimate {
    estimate: Estimate,
}

impl PyDiagnosticEstimate {
    pub(crate) fn from_estimate(estimate: Estimate) -> Self {
        Self { estimate }
    }
}

#[pymethods]
impl PyDiagnosticEstimate {
    /// Mean estimate value.
    #[getter]
    fn value(&self) -> f64 {
        self.estimate.value().get()
    }

    /// Standard error of the estimate.
    #[getter]
    fn standard_error(&self) -> f64 {
        self.estimate.standard_error().get()
    }

    /// Two-sided confidence interval as `(lower, upper)`.
    #[getter]
    fn confidence_interval(&self) -> (f64, f64) {
        let interval = self.estimate.confidence_interval();
        (interval.lower().get(), interval.upper().get())
    }

    /// Estimator kind used for this estimate.
    #[getter]
    fn estimator(&self) -> &str {
        estimator_name(self.estimate.estimator())
    }

    /// Number of statistically independent sampling units.
    #[getter]
    fn effective_sampling_units(&self) -> u64 {
        self.estimate.effective_sampling_units().get()
    }

    fn __repr__(&self) -> String {
        format!("DiagnosticEstimate(value={:?})", self.value())
    }
}

/// CRN bump validation diagnostics for one requested risk.
#[pyclass(frozen, name = "RiskValidation", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyRiskValidation {
    validation: RiskValidation,
}

impl PyRiskValidation {
    fn from_validation(validation: RiskValidation) -> Self {
        Self { validation }
    }
}

#[pymethods]
impl PyRiskValidation {
    /// Independent bump-and-revalue estimate.
    #[getter]
    fn bump_and_revalue(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.validation.bump_and_revalue)
    }

    /// CRN bump estimate minus the primary AAD or bumped-AAD estimate.
    #[getter]
    fn bump_minus_primary(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.validation.bump_minus_primary)
    }

    fn __repr__(&self) -> String {
        format!(
            "RiskValidation(bump_and_revalue={:?}, bump_minus_primary={:?})",
            self.bump_and_revalue().value(),
            self.bump_minus_primary().value()
        )
    }
}

/// Immutable replay and numerical diagnostics for a Monte Carlo result.
#[pyclass(frozen, name = "Diagnostics", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDiagnostics {
    master_seed: u64,
    estimator: &'static str,
    scramble_count: Option<u32>,
    direction_checksum: Option<String>,
    scramble_checksum: Option<String>,
    policy_version: u32,
    worker_threads: u32,
    reduction_block_size: u64,
    aad_tile_policy_version: u32,
    aad_tile_capacity: u32,
    checkpoint_policy_version: u32,
    checkpoint_interval: u32,
    antithetic: bool,
    discount_region: &'static str,
    dividend_region: &'static str,
    payoff_fingerprint: String,
    valuation_kind: &'static str,
    payoff_smoothing_kernel: Option<&'static str>,
    payoff_smoothing_policy_version: Option<u32>,
    payoff_smoothing_half_width: Option<f64>,
    payoff_smoothing_full_transition_width: Option<f64>,
    payoff_smoothing_width_unit: Option<&'static str>,
    payoff_smoothing_price_and_greeks_share_payoff: Option<bool>,
    payoff_smoothing_endpoint_count: Option<u32>,
    payoff_smoothing_dividend_jump_count: Option<u32>,
    path_state_kind: Option<&'static str>,
    asian_known_observation_count: Option<u32>,
    asian_unknown_observation_count: Option<u32>,
    asian_known_weight_sum: Option<f64>,
    asian_unknown_weight_sum: Option<f64>,
    asian_weighted_known_fixing_sum: Option<f64>,
    lookback_past_monitoring_count: Option<u32>,
    lookback_future_monitoring_count: Option<u32>,
    lookback_historical_extremum: Option<f64>,
    barrier_bridge_abi: Option<&'static str>,
    barrier_bridge_policy_version: Option<u32>,
    barrier_hit_indicator_mode: Option<&'static str>,
    barrier_endpoint_hit_fraction: Option<f64>,
    barrier_dividend_jump_hit_fraction: Option<f64>,
    barrier_mean_conditional_bridge_hit_weight: Option<f64>,
    barrier_mean_interval_count: Option<f64>,
    barrier_mean_finite_correction_count: Option<f64>,
    barrier_mean_zero_variance_count: Option<f64>,
    barrier_mean_survival_underflow_count: Option<f64>,
    barrier_mean_certain_survival_count: Option<f64>,
    delta_method: Option<&'static str>,
    gamma_method: Option<&'static str>,
    vega_method: Option<&'static str>,
    gamma_spot_bump: Option<f64>,
    validation_spot_bump: Option<f64>,
    validation_volatility_bump: Option<f64>,
    bump_policy_version: u32,
    exercise_strategy_risk: Option<&'static str>,
    stopping_index_risk: Option<&'static str>,
    exercise_policy_fingerprint: Option<String>,
    delta_validation: Option<PyRiskValidation>,
    gamma_validation: Option<PyRiskValidation>,
    vega_validation: Option<PyRiskValidation>,
    warnings: Vec<PyPricingWarning>,
}

impl PyDiagnostics {
    pub(crate) fn from_price(price: &MonteCarloPrice) -> Self {
        let diagnostics = price.diagnostics;
        let methods = price.risk_diagnostics.methods;
        Self {
            master_seed: diagnostics.master_seed,
            estimator: estimator_name(diagnostics.estimator),
            scramble_count: diagnostics.scramble_count,
            direction_checksum: diagnostics.direction_checksum.map(hex_32),
            scramble_checksum: diagnostics.scramble_checksum.map(hex_32),
            policy_version: diagnostics.policy_version,
            worker_threads: diagnostics.worker_threads,
            reduction_block_size: diagnostics.reduction_block_size,
            aad_tile_policy_version: diagnostics.aad_tile_policy_version,
            aad_tile_capacity: diagnostics.aad_tile_capacity,
            checkpoint_policy_version: diagnostics.checkpoint_policy_version,
            checkpoint_interval: diagnostics.checkpoint_interval,
            antithetic: diagnostics.antithetic,
            discount_region: curve_region_name(diagnostics.discount_region),
            dividend_region: curve_region_name(diagnostics.dividend_region),
            payoff_fingerprint: diagnostics.payoff_fingerprint.to_string(),
            valuation_kind: payoff_valuation_kind_name(diagnostics.valuation_kind),
            payoff_smoothing_kernel: diagnostics
                .payoff_smoothing
                .map(|smoothing| payoff_smoothing_kernel_name(smoothing.kernel)),
            payoff_smoothing_policy_version: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.policy_version),
            payoff_smoothing_half_width: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.half_width.get()),
            payoff_smoothing_full_transition_width: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.full_transition_width.get()),
            payoff_smoothing_width_unit: diagnostics
                .payoff_smoothing
                .map(|smoothing| payoff_smoothing_width_unit_name(smoothing.width_unit)),
            payoff_smoothing_price_and_greeks_share_payoff: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.price_and_greeks_share_payoff),
            payoff_smoothing_endpoint_count: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.endpoint_count),
            payoff_smoothing_dividend_jump_count: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.dividend_jump_count),
            path_state_kind: diagnostics.path_state.map(path_state_kind_name),
            asian_known_observation_count: asian_state(diagnostics.path_state).map(|state| state.0),
            asian_unknown_observation_count: asian_state(diagnostics.path_state)
                .map(|state| state.1),
            asian_known_weight_sum: asian_state(diagnostics.path_state).map(|state| state.2),
            asian_unknown_weight_sum: asian_state(diagnostics.path_state).map(|state| state.3),
            asian_weighted_known_fixing_sum: asian_state(diagnostics.path_state)
                .map(|state| state.4),
            lookback_past_monitoring_count: lookback_state(diagnostics.path_state)
                .map(|state| state.0),
            lookback_future_monitoring_count: lookback_state(diagnostics.path_state)
                .map(|state| state.1),
            lookback_historical_extremum: lookback_state(diagnostics.path_state)
                .and_then(|state| state.2),
            barrier_bridge_abi: diagnostics.barrier_bridge.map(|bridge| bridge.abi),
            barrier_bridge_policy_version: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.policy_version),
            barrier_hit_indicator_mode: diagnostics
                .barrier_bridge
                .map(|bridge| barrier_hit_indicator_mode_name(bridge.indicator_mode)),
            barrier_endpoint_hit_fraction: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.endpoint_hit_fraction),
            barrier_dividend_jump_hit_fraction: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.dividend_jump_hit_fraction),
            barrier_mean_conditional_bridge_hit_weight: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.mean_conditional_bridge_hit_weight),
            barrier_mean_interval_count: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.mean_interval_count),
            barrier_mean_finite_correction_count: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.mean_finite_correction_count),
            barrier_mean_zero_variance_count: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.mean_zero_variance_count),
            barrier_mean_survival_underflow_count: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.mean_survival_underflow_count),
            barrier_mean_certain_survival_count: diagnostics
                .barrier_bridge
                .map(|bridge| bridge.mean_certain_survival_count),
            delta_method: methods.delta.map(risk_method_name),
            gamma_method: methods.gamma.map(risk_method_name),
            vega_method: methods.vega.map(risk_method_name),
            gamma_spot_bump: methods.gamma_spot_bump,
            validation_spot_bump: methods.validation_spot_bump,
            validation_volatility_bump: methods.validation_volatility_bump,
            bump_policy_version: methods.bump_policy_version,
            exercise_strategy_risk: methods.exercise_strategy.map(exercise_strategy_risk_name),
            stopping_index_risk: methods.stopping_indices.map(stopping_index_risk_name),
            exercise_policy_fingerprint: methods
                .exercise_policy_fingerprint
                .map(|fingerprint| format_fingerprint(fingerprint.as_bytes())),
            delta_validation: price
                .risk_diagnostics
                .delta_validation
                .map(PyRiskValidation::from_validation),
            gamma_validation: price
                .risk_diagnostics
                .gamma_validation
                .map(PyRiskValidation::from_validation),
            vega_validation: price
                .risk_diagnostics
                .vega_validation
                .map(PyRiskValidation::from_validation),
            warnings: price
                .pricing_result
                .diagnostics
                .warnings()
                .iter()
                .map(|warning| PyPricingWarning {
                    code: warning.code().to_owned(),
                    message: warning.message().to_owned(),
                })
                .collect(),
        }
    }
}

#[pymethods]
impl PyDiagnostics {
    #[getter]
    fn master_seed(&self) -> u64 {
        self.master_seed
    }

    #[getter]
    fn estimator(&self) -> &str {
        self.estimator
    }

    #[getter]
    fn scramble_count(&self) -> Option<u32> {
        self.scramble_count
    }

    #[getter]
    fn direction_checksum(&self) -> Option<&str> {
        self.direction_checksum.as_deref()
    }

    #[getter]
    fn scramble_checksum(&self) -> Option<&str> {
        self.scramble_checksum.as_deref()
    }

    #[getter]
    fn policy_version(&self) -> u32 {
        self.policy_version
    }

    #[getter]
    fn worker_threads(&self) -> u32 {
        self.worker_threads
    }

    #[getter]
    fn reduction_block_size(&self) -> u64 {
        self.reduction_block_size
    }

    #[getter]
    fn aad_tile_policy_version(&self) -> u32 {
        self.aad_tile_policy_version
    }

    #[getter]
    fn aad_tile_capacity(&self) -> u32 {
        self.aad_tile_capacity
    }

    #[getter]
    fn checkpoint_policy_version(&self) -> u32 {
        self.checkpoint_policy_version
    }

    #[getter]
    fn checkpoint_interval(&self) -> u32 {
        self.checkpoint_interval
    }

    #[getter]
    fn antithetic(&self) -> bool {
        self.antithetic
    }

    #[getter]
    fn discount_region(&self) -> &str {
        self.discount_region
    }

    #[getter]
    fn dividend_region(&self) -> &str {
        self.dividend_region
    }

    #[getter]
    fn payoff_fingerprint(&self) -> &str {
        &self.payoff_fingerprint
    }

    #[getter]
    fn valuation_kind(&self) -> &str {
        self.valuation_kind
    }

    #[getter]
    fn payoff_smoothing_kernel(&self) -> Option<&str> {
        self.payoff_smoothing_kernel
    }

    #[getter]
    fn payoff_smoothing_policy_version(&self) -> Option<u32> {
        self.payoff_smoothing_policy_version
    }

    #[getter]
    fn payoff_smoothing_half_width(&self) -> Option<f64> {
        self.payoff_smoothing_half_width
    }

    #[getter]
    fn payoff_smoothing_full_transition_width(&self) -> Option<f64> {
        self.payoff_smoothing_full_transition_width
    }

    #[getter]
    fn payoff_smoothing_width_unit(&self) -> Option<&str> {
        self.payoff_smoothing_width_unit
    }

    #[getter]
    fn payoff_smoothing_price_and_greeks_share_payoff(&self) -> Option<bool> {
        self.payoff_smoothing_price_and_greeks_share_payoff
    }

    #[getter]
    fn payoff_smoothing_endpoint_count(&self) -> Option<u32> {
        self.payoff_smoothing_endpoint_count
    }

    #[getter]
    fn payoff_smoothing_dividend_jump_count(&self) -> Option<u32> {
        self.payoff_smoothing_dividend_jump_count
    }

    #[getter]
    fn path_state_kind(&self) -> Option<&str> {
        self.path_state_kind
    }

    #[getter]
    fn asian_known_observation_count(&self) -> Option<u32> {
        self.asian_known_observation_count
    }

    #[getter]
    fn asian_unknown_observation_count(&self) -> Option<u32> {
        self.asian_unknown_observation_count
    }

    #[getter]
    fn asian_known_weight_sum(&self) -> Option<f64> {
        self.asian_known_weight_sum
    }

    #[getter]
    fn asian_unknown_weight_sum(&self) -> Option<f64> {
        self.asian_unknown_weight_sum
    }

    #[getter]
    fn asian_weighted_known_fixing_sum(&self) -> Option<f64> {
        self.asian_weighted_known_fixing_sum
    }

    #[getter]
    fn lookback_past_monitoring_count(&self) -> Option<u32> {
        self.lookback_past_monitoring_count
    }

    #[getter]
    fn lookback_future_monitoring_count(&self) -> Option<u32> {
        self.lookback_future_monitoring_count
    }

    #[getter]
    fn lookback_historical_extremum(&self) -> Option<f64> {
        self.lookback_historical_extremum
    }

    #[getter]
    fn barrier_bridge_abi(&self) -> Option<&str> {
        self.barrier_bridge_abi
    }

    #[getter]
    fn barrier_bridge_policy_version(&self) -> Option<u32> {
        self.barrier_bridge_policy_version
    }

    #[getter]
    fn barrier_hit_indicator_mode(&self) -> Option<&str> {
        self.barrier_hit_indicator_mode
    }

    #[getter]
    fn barrier_endpoint_hit_fraction(&self) -> Option<f64> {
        self.barrier_endpoint_hit_fraction
    }

    #[getter]
    fn barrier_dividend_jump_hit_fraction(&self) -> Option<f64> {
        self.barrier_dividend_jump_hit_fraction
    }

    #[getter]
    fn barrier_mean_conditional_bridge_hit_weight(&self) -> Option<f64> {
        self.barrier_mean_conditional_bridge_hit_weight
    }

    #[getter]
    fn barrier_mean_interval_count(&self) -> Option<f64> {
        self.barrier_mean_interval_count
    }

    #[getter]
    fn barrier_mean_finite_correction_count(&self) -> Option<f64> {
        self.barrier_mean_finite_correction_count
    }

    #[getter]
    fn barrier_mean_zero_variance_count(&self) -> Option<f64> {
        self.barrier_mean_zero_variance_count
    }

    #[getter]
    fn barrier_mean_survival_underflow_count(&self) -> Option<f64> {
        self.barrier_mean_survival_underflow_count
    }

    #[getter]
    fn barrier_mean_certain_survival_count(&self) -> Option<f64> {
        self.barrier_mean_certain_survival_count
    }

    #[getter]
    fn delta_method(&self) -> Option<&str> {
        self.delta_method
    }

    #[getter]
    fn gamma_method(&self) -> Option<&str> {
        self.gamma_method
    }

    #[getter]
    fn vega_method(&self) -> Option<&str> {
        self.vega_method
    }

    #[getter]
    fn gamma_spot_bump(&self) -> Option<f64> {
        self.gamma_spot_bump
    }

    #[getter]
    fn validation_spot_bump(&self) -> Option<f64> {
        self.validation_spot_bump
    }

    #[getter]
    fn validation_volatility_bump(&self) -> Option<f64> {
        self.validation_volatility_bump
    }

    #[getter]
    fn bump_policy_version(&self) -> u32 {
        self.bump_policy_version
    }

    #[getter]
    fn exercise_strategy_risk(&self) -> Option<&str> {
        self.exercise_strategy_risk
    }

    #[getter]
    fn stopping_index_risk(&self) -> Option<&str> {
        self.stopping_index_risk
    }

    #[getter]
    fn exercise_policy_fingerprint(&self) -> Option<&str> {
        self.exercise_policy_fingerprint.as_deref()
    }

    #[getter]
    fn delta_validation(&self) -> Option<PyRiskValidation> {
        self.delta_validation
    }

    #[getter]
    fn gamma_validation(&self) -> Option<PyRiskValidation> {
        self.gamma_validation
    }

    #[getter]
    fn vega_validation(&self) -> Option<PyRiskValidation> {
        self.vega_validation
    }

    /// Warnings in deterministic emission order.
    #[getter]
    pub(crate) fn warnings(&self) -> Vec<PyPricingWarning> {
        self.warnings.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(estimator={:?}, master_seed={}, warnings={})",
            self.estimator,
            self.master_seed,
            self.warnings.len()
        )
    }
}

const fn estimator_name(estimator: EstimatorKind) -> &'static str {
    match estimator {
        EstimatorKind::Analytical => "analytical",
        EstimatorKind::PseudoMonteCarlo => "pseudo_monte_carlo",
        EstimatorKind::RandomizedQuasiMonteCarlo => "randomized_quasi_monte_carlo",
    }
}

const fn curve_region_name(region: CurveRegion) -> &'static str {
    match region {
        CurveRegion::Pillar => "pillar",
        CurveRegion::Interpolated => "interpolated",
        CurveRegion::RightExtrapolated => "right_extrapolated",
    }
}

const fn risk_method_name(method: RiskMethod) -> &'static str {
    match method {
        RiskMethod::AadReverse => "aad_reverse",
        RiskMethod::CentralBump => "central_bump",
        RiskMethod::CentralBumpOfAadDelta => "central_bump_of_aad_delta",
    }
}

const fn exercise_strategy_risk_name(risk: ExerciseStrategyRisk) -> &'static str {
    match risk {
        ExerciseStrategyRisk::FixedExerciseStrategy => "fixed_exercise_strategy",
    }
}

const fn stopping_index_risk_name(risk: StoppingIndexRisk) -> &'static str {
    match risk {
        StoppingIndexRisk::FrozenStoppingIndices => "frozen_stopping_indices",
    }
}

const fn payoff_smoothing_kernel_name(kernel: PayoffSmoothingKernel) -> &'static str {
    match kernel {
        PayoffSmoothingKernel::CompactC2 => "compact_c2",
    }
}

const fn payoff_valuation_kind_name(kind: PayoffValuationKind) -> &'static str {
    match kind {
        PayoffValuationKind::ExactContractual => "exact_contractual",
        PayoffValuationKind::SmoothedSurrogate => "smoothed_surrogate",
    }
}

const fn payoff_smoothing_width_unit_name(unit: PayoffSmoothingWidthUnit) -> &'static str {
    match unit {
        PayoffSmoothingWidthUnit::Spot => "spot",
    }
}

const fn path_state_kind_name(state: PathStateDiagnostics) -> &'static str {
    match state {
        PathStateDiagnostics::ArithmeticAsian { .. } => "arithmetic_asian",
        PathStateDiagnostics::FixedLookback { .. } => "fixed_lookback",
    }
}

const fn asian_state(state: Option<PathStateDiagnostics>) -> Option<(u32, u32, f64, f64, f64)> {
    match state {
        Some(PathStateDiagnostics::ArithmeticAsian {
            known_observation_count,
            unknown_observation_count,
            known_weight_sum,
            unknown_weight_sum,
            weighted_known_fixing_sum,
        }) => Some((
            known_observation_count,
            unknown_observation_count,
            known_weight_sum,
            unknown_weight_sum,
            weighted_known_fixing_sum,
        )),
        Some(PathStateDiagnostics::FixedLookback { .. }) | None => None,
    }
}

const fn lookback_state(state: Option<PathStateDiagnostics>) -> Option<(u32, u32, Option<f64>)> {
    match state {
        Some(PathStateDiagnostics::FixedLookback {
            past_monitoring_count,
            future_monitoring_count,
            historical_extremum,
        }) => Some((
            past_monitoring_count,
            future_monitoring_count,
            historical_extremum,
        )),
        Some(PathStateDiagnostics::ArithmeticAsian { .. }) | None => None,
    }
}

const fn barrier_hit_indicator_mode_name(mode: BarrierHitIndicatorMode) -> &'static str {
    match mode {
        BarrierHitIndicatorMode::Exact => "exact",
        BarrierHitIndicatorMode::CompactC2 => "compact_c2",
    }
}

fn hex_32(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}
