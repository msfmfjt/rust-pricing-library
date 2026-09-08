use pricing::market::CurveRegion;
use pricing::{EstimatorKind, MonteCarloPrice, RiskMethod};
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

/// Immutable replay and numerical diagnostics for a Monte Carlo result.
#[pyclass(frozen, name = "Diagnostics", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDiagnostics {
    master_seed: u64,
    estimator: &'static str,
    scramble_count: Option<u32>,
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
    delta_method: Option<&'static str>,
    gamma_method: Option<&'static str>,
    vega_method: Option<&'static str>,
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
            delta_method: methods.delta.map(risk_method_name),
            gamma_method: methods.gamma.map(risk_method_name),
            vega_method: methods.vega.map(risk_method_name),
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
        RiskMethod::CentralBumpOfAadDelta => "central_bump_of_aad_delta",
    }
}
