//! Stable PyO3 boundary over the `pricing` facade.

#![forbid(unsafe_code)]

mod builders;

use std::collections::BTreeMap;

use pricing::mc::ExecutionPolicy;
use pricing::{
    Estimate, MonteCarloError, MonteCarloPrice, PricingPlan, PricingRequest, RiskEstimate,
    WireError, fingerprint_request, parse_request_json, request_to_json, result_to_json,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use builders::{
    PyDiscountCurve, PyEngine, PyMarket, PyModel, PyProduct, PyRiskRequest, build_request,
};

create_exception!(rust_pricing, ValidationError, PyValueError);
create_exception!(rust_pricing, PricingError, PyRuntimeError);

#[pyclass(frozen, name = "ValidationIssue", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PyValidationIssue {
    pointer: String,
    phase: String,
    code: String,
    message: String,
}

impl PyValidationIssue {
    fn wire(error: &WireError) -> Self {
        let (phase, code) = match error {
            WireError::Json(_) | WireError::Utf8Bom => ("syntax_and_limits", "invalid_json"),
            WireError::ResourceLimit { .. } | WireError::LimitOverrideExceedsHardCap => {
                ("syntax_and_limits", "resource_limit")
            }
            WireError::UnsupportedSchemaVersion(_) => {
                ("declared_schema", "unsupported_schema_version")
            }
            WireError::WrongDocumentKind { .. } => ("declared_schema", "wrong_document_kind"),
            WireError::Domain(_) => ("domain", "invalid_domain_value"),
            WireError::InvalidFingerprint(_) => ("declared_schema", "invalid_fingerprint"),
        };
        Self {
            pointer: String::new(),
            phase: phase.into(),
            code: code.into(),
            message: error.to_string(),
        }
    }

    fn compile(error: &MonteCarloError) -> Self {
        Self {
            pointer: String::new(),
            phase: "domain".into(),
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
    fn phase(&self) -> &str {
        &self.phase
    }

    #[getter]
    fn code(&self) -> &str {
        &self.code
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn to_dict(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("pointer".into(), self.pointer.clone()),
            ("phase".into(), self.phase.clone()),
            ("code".into(), self.code.clone()),
            ("message".into(), self.message.clone()),
        ])
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationIssue(pointer={:?}, phase={:?}, code={:?}, message={:?})",
            self.pointer, self.phase, self.code, self.message
        )
    }
}

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

    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        parse_request_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(|error| validation_exception(py, PyValidationIssue::wire(&error)))
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

#[pyclass(frozen, name = "PricingPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPricingPlan {
    inner: PricingPlan,
}

#[pymethods]
impl PyPricingPlan {
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

#[pyclass(frozen, name = "PricingResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPricingResult {
    inner: MonteCarloPrice,
}

#[pymethods]
impl PyPricingResult {
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
    module.add_class::<PyDiscountCurve>()?;
    module.add_class::<PyProduct>()?;
    module.add_class::<PyMarket>()?;
    module.add_class::<PyModel>()?;
    module.add_class::<PyEngine>()?;
    module.add_class::<PyRiskRequest>()?;
    module.add_class::<PyPricingRequest>()?;
    module.add_class::<PyPricingPlan>()?;
    module.add_class::<PyPricingResult>()?;
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
        let issue = PyValidationIssue::wire(&WireError::UnsupportedSchemaVersion(99));
        assert_eq!(issue.phase, "declared_schema");
        assert_eq!(issue.code, "unsupported_schema_version");
    }
}
