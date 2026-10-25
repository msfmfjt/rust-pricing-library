//! Python shares Rust's validated source compiler and numerical dispatch.
use crate::*;
use pricing::{
    CompiledPricingSpecification, PricingSpecification, SpecificationOutput, SpecificationResult,
};

fn wire_exception(py: Python<'_>, context: &pricing::WireParseError) -> PyErr {
    let error = context.wire_error();
    let mut issue = PyValidationIssue::wire(error, "pricing_specification");
    issue.schema_version = match error {
        WireError::UnsupportedSchemaVersion(version)
        | WireError::UnsupportedSchemaFeature {
            schema_version: version,
            ..
        } => *version,
        _ => 2,
    };
    validation_exception(py, issue.with_input_version(context.schema_version()))
}

#[pyclass(frozen, name = "PricingSpecification")]
#[derive(Clone, Debug)]
pub struct PyPricingSpecification {
    pub(super) inner: PricingSpecification,
}

#[pymethods]
impl PyPricingSpecification {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &builders::PyProduct,
        market: &builders::PyMarket,
        model: &source_config::PyPricingModelSpec,
        engine: &builders::PyEngine,
        risk: &builders::PyRiskRequest,
        numerics: &source_config::PyPricingNumerics,
    ) -> PyResult<Self> {
        PricingSpecification::new(
            builders::date_from_python(py, valuation_date, "/valuation_date")?,
            product.inner.clone(),
            market.inner.clone(),
            model.inner.clone(),
            engine.inner,
            risk.inner.clone(),
            numerics.inner.clone(),
        )
        .map(|inner| Self { inner })
        .map_err(source_config::invalid)
    }
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        pricing::parse_specification_json_with_context(
            json.as_bytes(),
            pricing::JsonLimits::DEFAULT,
        )
        .map(|inner| Self { inner })
        .map_err(|e| wire_exception(py, &e))
    }
    #[staticmethod]
    #[pyo3(signature = (request, *, worker_threads, reduction_block_size=None))]
    fn from_request(
        py: Python<'_>,
        request: &PyPricingRequest,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size).map_err(|e| {
            let error = WireError::DomainAt {
                pointer: "/numerics".into(),
                message: e.to_string(),
            };
            let mut issue = PyValidationIssue::wire(&error, "pricing_specification");
            issue.schema_version = 2;
            issue.error_metadata = errors::ErrorMetadata::from_error(&e, errors::Stage::Validation);
            validation_exception(py, issue)
        })?;
        Ok(Self {
            inner: PricingSpecification::from_request(request.inner.clone(), policy),
        })
    }
    fn to_json(&self) -> PyResult<String> {
        pricing::specification_to_json(&self.inner).map_err(crate::errors::serialize_exception)
    }
    fn to_pretty_json(&self) -> PyResult<String> {
        pricing::specification_to_pretty_json(&self.inner)
            .map_err(crate::errors::serialize_exception)
    }
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.inner
            .fingerprint()
            .map(|f| f.to_string())
            .map_err(crate::errors::serialize_exception)
    }
    fn compile(&self, py: Python<'_>) -> PyResult<PyCompiledPricingSpecification> {
        py.detach(|| self.inner.compile())
            .map(|inner| PyCompiledPricingSpecification { inner })
            .map_err(|e| {
                let mut issue = PyValidationIssue::compile(&e);
                issue.document_kind = "pricing_specification".into();
                issue.schema_version = 2;
                validation_exception(py, issue)
            })
    }
}

#[pyclass(frozen, name = "CompiledPricingSpecification")]
#[derive(Clone, Debug)]
pub struct PyCompiledPricingSpecification {
    inner: CompiledPricingSpecification,
}
#[pymethods]
impl PyCompiledPricingSpecification {
    #[getter]
    fn specification_fingerprint(&self) -> String {
        self.inner.specification_fingerprint().to_string()
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PySpecificationResult> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PySpecificationResult { inner })
            .map_err(pricing_exception)
    }
}

#[pyclass(frozen, name = "SpecificationResult")]
#[derive(Clone, Debug)]
pub struct PySpecificationResult {
    pub(super) inner: SpecificationResult,
}
#[pymethods]
impl PySpecificationResult {
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error()
    }
    #[getter]
    fn specification_fingerprint(&self) -> String {
        self.inner.specification_fingerprint.to_string()
    }
    /// Retain the existing typed result, including all backend diagnostics.
    #[getter]
    fn native_result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.inner.output {
            SpecificationOutput::Standard(p) => Py::new(
                py,
                PyPricingResult {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            SpecificationOutput::Hybrid(p) | SpecificationOutput::RoughVolatility(p) => {
                Py::new(py, hull_white::PyHullWhitePrice { inner: p.clone() })?.into_any()
            }
            SpecificationOutput::LocalStochasticVolatility(p) => {
                Py::new(py, lsv::PyLsvPrice { inner: p.clone() })?.into_any()
            }
            _ => {
                return Err(crate::errors::internal_exception(
                    "unknown specification result variant",
                ));
            }
        })
    }
}

#[pyfunction]
fn specification_json_schema() -> &'static str {
    pricing::current_specification_schema()
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPricingSpecification>()?;
    module.add_class::<PyCompiledPricingSpecification>()?;
    module.add_class::<PySpecificationResult>()?;
    module.add_function(wrap_pyfunction!(specification_json_schema, module)?)?;
    Ok(())
}
