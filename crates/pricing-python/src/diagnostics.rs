use pricing::market::CurveRegion;
use pricing::{
    Estimate, EstimatorKind, MonteCarloPrice, PayoffSmoothingKernel, RiskMethod, RiskValidation,
};
use pyo3::prelude::*;

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
    payoff_smoothing_kernel: Option<&'static str>,
    payoff_smoothing_policy_version: Option<u32>,
    payoff_smoothing_half_width: Option<f64>,
    delta_method: Option<&'static str>,
    gamma_method: Option<&'static str>,
    vega_method: Option<&'static str>,
    gamma_spot_bump: Option<f64>,
    validation_spot_bump: Option<f64>,
    validation_volatility_bump: Option<f64>,
    bump_policy_version: u32,
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
            payoff_smoothing_kernel: diagnostics
                .payoff_smoothing
                .map(|smoothing| payoff_smoothing_kernel_name(smoothing.kernel)),
            payoff_smoothing_policy_version: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.policy_version),
            payoff_smoothing_half_width: diagnostics
                .payoff_smoothing
                .map(|smoothing| smoothing.half_width.get()),
            delta_method: methods.delta.map(risk_method_name),
            gamma_method: methods.gamma.map(risk_method_name),
            vega_method: methods.vega.map(risk_method_name),
            gamma_spot_bump: methods.gamma_spot_bump,
            validation_spot_bump: methods.validation_spot_bump,
            validation_volatility_bump: methods.validation_volatility_bump,
            bump_policy_version: methods.bump_policy_version,
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

const fn payoff_smoothing_kernel_name(kernel: PayoffSmoothingKernel) -> &'static str {
    match kernel {
        PayoffSmoothingKernel::CompactC2 => "compact_c2",
    }
}

fn hex_32(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}
