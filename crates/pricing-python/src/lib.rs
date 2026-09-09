//! Stable PyO3 boundary over the `pricing` facade.

#![forbid(unsafe_code)]

mod builders;
mod diagnostics;

use pricing::market::CurveRegion;
use pricing::mc::ExecutionPolicy;
use pricing::{
    Estimate, MonteCarloDiagnostics, MonteCarloError, MonteCarloPrice, PricingPlan, PricingRequest,
    RiskDiagnostics, RiskEstimate, RiskMethodMetadata, VegaKtResult, VegaKtResultBucketEstimate,
    VegaKtResultCoordinate, VegaKtResultCovarianceLayout, VegaKtResultProjection,
    VegaKtResultReportingStats, VegaKtResultResidualDiagnostics, VegaKtResultUnit, WireError,
    fingerprint_request, parse_request_json, parse_result_json, request_to_json, result_to_json,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use builders::{
    PyAsianObservation, PyDiscountCurve, PyDividendEvent, PyEngine, PyEssviSlice, PyMarket,
    PyModel, PyProduct, PyRiskRequest, build_request,
};
use diagnostics::{PyDiagnostics, PyPricingWarning};

create_exception!(rust_pricing, ValidationError, PyValueError);
create_exception!(rust_pricing, PricingError, PyRuntimeError);

/// One immutable, structured validation issue.
#[pyclass(frozen, name = "ValidationIssue", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PyValidationIssue {
    pointer: String,
    phase: String,
    schema_version: u32,
    document_kind: String,
    code: String,
    message: String,
}

impl PyValidationIssue {
    fn request_schema_version() -> u32 {
        pricing::core::SchemaVersion::CURRENT.get()
    }

    fn request_document_kind() -> &'static str {
        pricing::core::DocumentKind::PricingRequest.as_str()
    }

    fn result_document_kind() -> &'static str {
        pricing::core::DocumentKind::PricingResult.as_str()
    }

    fn request_wire(error: &WireError) -> Self {
        Self::wire(error, Self::request_document_kind())
    }

    fn result_wire(error: &WireError) -> Self {
        Self::wire(error, Self::result_document_kind())
    }

    fn wire(error: &WireError, document_kind: &'static str) -> Self {
        let (phase, code) = match error {
            WireError::Json(_) | WireError::Utf8Bom => ("syntax_and_limits", "invalid_json"),
            WireError::ResourceLimit { .. } | WireError::LimitOverrideExceedsHardCap => {
                ("syntax_and_limits", "resource_limit")
            }
            WireError::UnsupportedSchemaVersion(_) => {
                ("declared_schema", "unsupported_schema_version")
            }
            WireError::WrongDocumentKind { .. } => ("declared_schema", "wrong_document_kind"),
            WireError::Domain(_) | WireError::DomainAt { .. } => ("domain", "invalid_domain_value"),
            WireError::InvalidFingerprint(_) => ("declared_schema", "invalid_fingerprint"),
        };
        let schema_version = match error {
            WireError::UnsupportedSchemaVersion(version) => *version,
            _ => Self::request_schema_version(),
        };
        let pointer = match error {
            WireError::DomainAt { pointer, .. } => pointer.clone(),
            _ => String::new(),
        };
        Self {
            pointer,
            phase: phase.into(),
            schema_version,
            document_kind: document_kind.into(),
            code: code.into(),
            message: error.to_string(),
        }
    }

    fn compile(error: &MonteCarloError) -> Self {
        Self {
            pointer: String::new(),
            phase: "domain".into(),
            schema_version: Self::request_schema_version(),
            document_kind: Self::request_document_kind().into(),
            code: "plan_compile_error".into(),
            message: error.to_string(),
        }
    }

    fn domain(
        pointer: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            pointer: pointer.into(),
            phase: "domain".into(),
            schema_version: Self::request_schema_version(),
            document_kind: Self::request_document_kind().into(),
            code: code.into(),
            message: message.into(),
        }
    }
}

#[pymethods]
impl PyValidationIssue {
    #[getter]
    fn pointer(&self) -> &str {
        &self.pointer
    }

    #[getter]
    fn instance_path(&self) -> &str {
        &self.pointer
    }

    #[getter]
    fn phase(&self) -> &str {
        &self.phase
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[getter]
    fn document_kind(&self) -> &str {
        &self.document_kind
    }

    #[getter]
    fn code(&self) -> &str {
        &self.code
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("pointer", &self.pointer)?;
        dict.set_item("instance_path", &self.pointer)?;
        dict.set_item("phase", &self.phase)?;
        dict.set_item("schema_version", self.schema_version)?;
        dict.set_item("document_kind", &self.document_kind)?;
        dict.set_item("code", &self.code)?;
        dict.set_item("message", &self.message)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationIssue(pointer={:?}, phase={:?}, schema_version={}, document_kind={:?}, code={:?}, message={:?})",
            self.pointer,
            self.phase,
            self.schema_version,
            self.document_kind,
            self.code,
            self.message
        )
    }
}

/// Validated, immutable valuation request and replay serialization boundary.
#[pyclass(frozen, name = "PricingRequest", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPricingRequest {
    inner: PricingRequest,
}

#[pymethods]
impl PyPricingRequest {
    #[new]
    fn new(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &PyProduct,
        market: &PyMarket,
        model: &PyModel,
        engine: &PyEngine,
        risk: &PyRiskRequest,
    ) -> PyResult<Self> {
        build_request(py, valuation_date, product, market, model, engine, risk)
            .map(|inner| Self { inner })
    }

    /// Parse and validate a versioned pricing-request JSON document.
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        parse_request_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(|error| validation_exception(py, PyValidationIssue::request_wire(&error)))
    }

    fn to_json(&self) -> PyResult<String> {
        request_to_json(&self.inner).map_err(pricing_exception)
    }

    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        fingerprint_request(&self.inner)
            .map(|fingerprint| fingerprint.to_string())
            .map_err(pricing_exception)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PricingRequest(fingerprint={:?})",
            self.fingerprint()?
        ))
    }
}

/// Compiled immutable execution plan that can be evaluated repeatedly.
#[pyclass(frozen, name = "PricingPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPricingPlan {
    inner: PricingPlan,
}

#[pymethods]
impl PyPricingPlan {
    /// Compile a request while releasing the Python GIL.
    #[staticmethod]
    #[pyo3(signature = (request, *, worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        request: &PyPricingRequest,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|error| validation_exception(py, PyValidationIssue::compile(&error.into())))?;
        let request = request.inner.clone();
        py.detach(|| PricingPlan::compile(&request, policy))
            .map(|inner| Self { inner })
            .map_err(|error| validation_exception(py, PyValidationIssue::compile(&error)))
    }

    /// Evaluate the plan while releasing the Python GIL.
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyPricingResult> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyPricingResult { inner })
            .map_err(pricing_exception)
    }

    #[getter]
    fn request_fingerprint(&self) -> String {
        self.inner.request_fingerprint().to_string()
    }

    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }

    #[getter]
    fn worker_threads(&self) -> u32 {
        self.inner.execution_policy().worker_threads().get()
    }

    #[getter]
    fn reduction_block_size(&self) -> u64 {
        self.inner.execution_policy().reduction_block_size().get()
    }

    fn __repr__(&self) -> String {
        format!("PricingPlan(fingerprint={:?})", self.plan_fingerprint())
    }
}

/// Immutable Price, Greeks, uncertainty, replay, and diagnostic result.
#[pyclass(frozen, name = "PricingResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPricingResult {
    inner: MonteCarloPrice,
}

/// One VegaKT bucket coordinate in maturity/log-moneyness space.
#[pyclass(frozen, name = "VegaKtCoordinate", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtCoordinate {
    inner: VegaKtResultCoordinate,
}

#[pymethods]
impl PyVegaKtCoordinate {
    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity().get()
    }

    #[getter]
    fn log_moneyness(&self) -> f64 {
        self.inner.log_moneyness().get()
    }

    #[getter]
    fn implied_volatility(&self) -> f64 {
        self.inner.implied_volatility().get()
    }
}

/// One VegaKT bucket estimate with raw and market-scaled units.
#[pyclass(frozen, name = "VegaKtBucketEstimate", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtBucketEstimate {
    inner: VegaKtResultBucketEstimate,
}

#[pymethods]
impl PyVegaKtBucketEstimate {
    #[getter]
    fn raw_mean(&self) -> f64 {
        self.inner.raw_mean().get()
    }

    #[getter]
    fn market_scaled_mean(&self) -> f64 {
        self.inner.market_scaled_mean().get()
    }

    #[getter]
    fn sample_variance(&self) -> Option<f64> {
        self.inner.sample_variance().map(|value| value.get())
    }

    #[getter]
    fn price_covariance(&self) -> Option<f64> {
        self.inner.price_covariance().map(|value| value.get())
    }
}

/// Reporting-edge sensitivity diagnostics for VegaKT projection.
#[pyclass(frozen, name = "VegaKtReportingStats", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtReportingStats {
    inner: VegaKtResultReportingStats,
}

#[pymethods]
impl PyVegaKtReportingStats {
    #[getter]
    fn left_edge_count(&self) -> u64 {
        self.inner.left_edge_count()
    }

    #[getter]
    fn right_edge_count(&self) -> u64 {
        self.inner.right_edge_count()
    }

    #[getter]
    fn left_edge_sensitivity(&self) -> f64 {
        self.inner.left_edge_sensitivity().get()
    }

    #[getter]
    fn right_edge_sensitivity(&self) -> f64 {
        self.inner.right_edge_sensitivity().get()
    }
}

/// Projection summary used to reconcile VegaKT buckets with scalar vega.
#[pyclass(frozen, name = "VegaKtProjection", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtProjection {
    inner: VegaKtResultProjection,
}

#[pymethods]
impl PyVegaKtProjection {
    #[getter]
    fn scalar_vega(&self) -> f64 {
        self.inner.scalar_vega().get()
    }

    #[getter]
    fn signed_residual(&self) -> f64 {
        self.inner.signed_residual().get()
    }

    #[getter]
    fn pre_projection(&self) -> f64 {
        self.inner.pre_projection().get()
    }

    #[getter]
    fn reporting_stats(&self) -> PyVegaKtReportingStats {
        PyVegaKtReportingStats {
            inner: self.inner.reporting_stats(),
        }
    }
}

/// Residual diagnostics for the VegaKT active reporting domain.
#[pyclass(frozen, name = "VegaKtResidualDiagnostics", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtResidualDiagnostics {
    inner: VegaKtResultResidualDiagnostics,
}

#[pymethods]
impl PyVegaKtResidualDiagnostics {
    #[getter]
    fn active_domain_start_index(&self) -> usize {
        self.inner.active_domain_start_index()
    }

    #[getter]
    fn active_domain_end_index(&self) -> usize {
        self.inner.active_domain_end_index()
    }

    #[getter]
    fn active_domain_forward_index(&self) -> usize {
        self.inner.active_domain_forward_index()
    }

    #[getter]
    fn excluded_probability_mass(&self) -> f64 {
        self.inner.excluded_probability_mass().get()
    }

    #[getter]
    fn signed_residual(&self) -> f64 {
        self.inner.signed_residual().get()
    }

    #[getter]
    fn pre_projection(&self) -> f64 {
        self.inner.pre_projection().get()
    }

    #[getter]
    fn reporting_stats(&self) -> PyVegaKtReportingStats {
        PyVegaKtReportingStats {
            inner: self.inner.reporting_stats(),
        }
    }
}

/// Full VegaKT bucket report attached to a pricing result when requested.
#[pyclass(frozen, name = "VegaKtResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtResult {
    inner: VegaKtResult,
}

#[pymethods]
impl PyVegaKtResult {
    #[getter]
    fn coordinates(&self) -> Vec<PyVegaKtCoordinate> {
        self.inner
            .coordinates()
            .iter()
            .copied()
            .map(|inner| PyVegaKtCoordinate { inner })
            .collect()
    }

    #[getter]
    fn estimates(&self) -> Vec<PyVegaKtBucketEstimate> {
        self.inner
            .estimates()
            .iter()
            .copied()
            .map(|inner| PyVegaKtBucketEstimate { inner })
            .collect()
    }

    #[getter]
    fn raw_buckets(&self) -> Vec<f64> {
        self.inner
            .raw_buckets()
            .iter()
            .map(|value| value.get())
            .collect()
    }

    #[getter]
    fn full_bucket_covariance(&self) -> Option<Vec<Option<f64>>> {
        self.inner.full_bucket_covariance().map(|values| {
            values
                .iter()
                .map(|value| value.map(|value| value.get()))
                .collect()
        })
    }

    #[getter]
    fn covariance_layout(&self) -> &'static str {
        vega_kt_covariance_layout_name(self.inner.covariance_layout())
    }

    #[getter]
    fn projection(&self) -> PyVegaKtProjection {
        PyVegaKtProjection {
            inner: self.inner.projection(),
        }
    }

    #[getter]
    fn residual_diagnostics(&self) -> PyVegaKtResidualDiagnostics {
        PyVegaKtResidualDiagnostics {
            inner: self.inner.residual_diagnostics(),
        }
    }

    #[getter]
    fn raw_unit(&self) -> &'static str {
        vega_kt_unit_name(self.inner.raw_unit())
    }

    #[getter]
    fn market_scaled_unit(&self) -> &'static str {
        vega_kt_unit_name(self.inner.market_scaled_unit())
    }

    #[getter]
    fn policy_label(&self) -> &str {
        self.inner.policy_label()
    }

    #[getter]
    fn truncation_order(&self) -> &str {
        self.inner.truncation_order()
    }
}

#[pymethods]
impl PyPricingResult {
    /// Parse and validate a versioned pricing-result JSON document.
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        parse_result_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(monte_carlo_price_from_result)
            .map(|inner| Self { inner })
            .map_err(|error| validation_exception(py, PyValidationIssue::result_wire(&error)))
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.pricing_result.value.value().get()
    }

    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.pricing_result.value.standard_error().get()
    }

    #[getter]
    fn confidence_interval(&self) -> (f64, f64) {
        let interval = self.inner.pricing_result.value.confidence_interval();
        (interval.lower().get(), interval.upper().get())
    }

    #[getter]
    fn delta_raw(&self) -> Option<f64> {
        risk_value(self.inner.pricing_result.risks.delta, false)
    }

    #[getter]
    fn delta_market_scaled(&self) -> Option<f64> {
        risk_value(self.inner.pricing_result.risks.delta, true)
    }

    #[getter]
    fn gamma_raw(&self) -> Option<f64> {
        risk_value(self.inner.pricing_result.risks.gamma, false)
    }

    #[getter]
    fn gamma_market_scaled(&self) -> Option<f64> {
        risk_value(self.inner.pricing_result.risks.gamma, true)
    }

    #[getter]
    fn vega_raw(&self) -> Option<f64> {
        risk_value(self.inner.pricing_result.risks.vega, false)
    }

    #[getter]
    fn vega_market_scaled(&self) -> Option<f64> {
        risk_value(self.inner.pricing_result.risks.vega, true)
    }

    #[getter]
    fn vega_kt(&self) -> Option<PyVegaKtResult> {
        self.inner
            .pricing_result
            .risks
            .vega_kt
            .clone()
            .map(|inner| PyVegaKtResult { inner })
    }

    #[getter]
    fn sampling_variance(&self) -> f64 {
        self.inner.sampling_variance
    }

    #[getter]
    fn estimator_variance(&self) -> f64 {
        self.inner.estimator_variance
    }

    #[getter]
    fn independent_sampling_units(&self) -> u64 {
        self.inner.independent_sampling_units
    }

    #[getter]
    fn evaluated_paths(&self) -> u128 {
        self.inner.evaluated_paths
    }

    /// Replay, numerical-method, and warning metadata.
    #[getter]
    fn diagnostics(&self) -> PyDiagnostics {
        PyDiagnostics::from_price(&self.inner)
    }

    /// Valuation warnings in deterministic emission order.
    #[getter]
    fn warnings(&self) -> Vec<PyPricingWarning> {
        PyDiagnostics::from_price(&self.inner).warnings()
    }

    fn to_json(&self) -> PyResult<String> {
        result_to_json(&self.inner.pricing_result).map_err(pricing_exception)
    }

    fn __repr__(&self) -> String {
        format!("PricingResult(value={:?})", self.value())
    }
}

fn risk_value(risk: Option<RiskEstimate>, market_scaled: bool) -> Option<f64> {
    risk.map(|estimate| {
        let estimate: Estimate = if market_scaled {
            estimate.market_scaled()
        } else {
            estimate.raw()
        };
        estimate.value().get()
    })
}

fn monte_carlo_price_from_result(pricing_result: pricing::PricingResult) -> MonteCarloPrice {
    let effective_units = pricing_result.value.effective_sampling_units().get();
    let estimator_variance = pricing_result.value.standard_error().get().powi(2);
    let estimator = pricing_result.value.estimator();
    MonteCarloPrice {
        sampling_variance: estimator_variance * effective_units as f64,
        estimator_variance,
        risk_diagnostics: RiskDiagnostics {
            methods: RiskMethodMetadata {
                delta: None,
                gamma: None,
                vega: None,
                smile_dynamics: pricing::risk::SmileDynamics::StickyLogMoneyness,
                gamma_spot_bump: None,
                validation_spot_bump: None,
                validation_volatility_bump: None,
                bump_policy_version: 0,
            },
            delta_validation: None,
            gamma_validation: None,
            vega_validation: None,
        },
        pricing_result,
        independent_sampling_units: effective_units,
        evaluated_paths: u128::from(effective_units),
        diagnostics: MonteCarloDiagnostics {
            master_seed: 0,
            estimator,
            scramble_count: None,
            direction_checksum: None,
            scramble_checksum: None,
            policy_version: 0,
            worker_threads: 1,
            reduction_block_size: 1,
            aad_tile_policy_version: 0,
            aad_tile_capacity: 1,
            checkpoint_policy_version: 0,
            checkpoint_interval: 1,
            antithetic: false,
            discount_region: CurveRegion::Pillar,
            dividend_region: CurveRegion::Pillar,
            payoff_fingerprint: pricing::product::GraphFingerprint::from_bytes([0; 32]),
        },
    }
}

fn vega_kt_covariance_layout_name(value: VegaKtResultCovarianceLayout) -> &'static str {
    match value {
        VegaKtResultCovarianceLayout::PriceAndBucketVarianceOnly => {
            "price_and_bucket_variance_only"
        }
        VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor => "full_bucket_matrix_row_major",
    }
}

fn vega_kt_unit_name(value: VegaKtResultUnit) -> &'static str {
    match value {
        VegaKtResultUnit::CurrencyPerUnitAbsoluteVolatility => {
            "currency_per_unit_absolute_volatility"
        }
        VegaKtResultUnit::CurrencyPerVolatilityPoint => "currency_per_volatility_point",
    }
}

pub(crate) fn validation_exception(py: Python<'_>, issue: PyValidationIssue) -> PyErr {
    let message = issue.message.clone();
    let error = ValidationError::new_err(message);
    if let Ok(issue) = Py::new(py, issue) {
        let _ = error.value(py).setattr("issues", vec![issue]);
    }
    error
}

fn pricing_exception(error: impl ToString) -> PyErr {
    PricingError::new_err(error.to_string())
}

/// Returns the Rust facade version exposed by the Python module.
#[must_use]
pub const fn facade_version() -> &'static str {
    pricing::version()
}

#[pyfunction]
fn version() -> &'static str {
    facade_version()
}

#[pymodule]
fn rust_pricing(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", facade_version())?;
    module.add("ValidationError", module.py().get_type::<ValidationError>())?;
    module.add("PricingError", module.py().get_type::<PricingError>())?;
    module.add_class::<PyValidationIssue>()?;
    module.add_class::<PyPricingWarning>()?;
    module.add_class::<PyDiagnostics>()?;
    module.add_class::<PyDiscountCurve>()?;
    module.add_class::<PyDividendEvent>()?;
    module.add_class::<PyAsianObservation>()?;
    module.add_class::<PyEssviSlice>()?;
    module.add_class::<PyProduct>()?;
    module.add_class::<PyMarket>()?;
    module.add_class::<PyModel>()?;
    module.add_class::<PyEngine>()?;
    module.add_class::<PyRiskRequest>()?;
    module.add_class::<PyPricingRequest>()?;
    module.add_class::<PyPricingPlan>()?;
    module.add_class::<PyPricingResult>()?;
    module.add_class::<PyVegaKtCoordinate>()?;
    module.add_class::<PyVegaKtBucketEstimate>()?;
    module.add_class::<PyVegaKtReportingStats>()?;
    module.add_class::<PyVegaKtProjection>()?;
    module.add_class::<PyVegaKtResidualDiagnostics>()?;
    module.add_class::<PyVegaKtResult>()?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_uses_facade_version() {
        assert_eq!(crate::facade_version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn wire_errors_have_stable_issue_codes() {
        let issue = PyValidationIssue::request_wire(&WireError::UnsupportedSchemaVersion(99));
        assert_eq!(issue.phase, "declared_schema");
        assert_eq!(issue.code, "unsupported_schema_version");
    }
}
