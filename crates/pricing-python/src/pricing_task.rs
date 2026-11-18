//! Thin adapters over complete Rust tasks and common reports.
use crate::*;
use pricing::{
    CalibrationMetric, CapabilityStatus, PricingCapability, PricingOperation, PricingTask,
    PricingTaskOutput, ReportedCalibration, ReportedRisk, ValuationReport,
};

type Measures<'py> = Option<Vec<PyRef<'py, measures::PyMeasureRequest>>>;

pub(crate) fn operation(
    name: &str,
    bump: Option<f64>,
    measures: &Measures<'_>,
) -> PyResult<PricingOperation> {
    match (name, bump, measures) {
        ("evaluate", None, None) => Ok(PricingOperation::Evaluate),
        ("aad", gamma_relative_bump, None) => Ok(PricingOperation::Aad {
            gamma_relative_bump,
        }),
        ("local_variance_risk", None, None) => Ok(PricingOperation::LocalVarianceRisk),
        ("measures", None, Some(list)) => {
            Ok(PricingOperation::Measures(measures::measure_set(list)?))
        }
        _ => Err(Python::attach(|py| {
            errors::ErrorMetadata::domain("invalid_operation", Some("operation".into())).attach(
                py, PyValueError::new_err("operation must be evaluate, aad, local_variance_risk or measures; Gamma bump belongs to aad and a measure list belongs to measures"), Some(name))
        })),
    }
}
fn wire_error(py: Python<'_>, error: &WireError, kind: &'static str, schema_version: u32) -> PyErr {
    let mut issue = PyValidationIssue::wire(error, kind);
    issue.schema_version = match error {
        WireError::UnsupportedSchemaVersion(version)
        | WireError::UnsupportedSchemaFeature {
            schema_version: version,
            ..
        } => *version,
        _ => schema_version,
    };
    validation_exception(py, issue)
}
#[pyclass(frozen, name = "PricingCapability")]
#[derive(Clone, Debug)]
pub struct PyPricingCapability {
    inner: PricingCapability,
}
#[pymethods]
impl PyPricingCapability {
    #[getter]
    fn operation(&self) -> &'static str {
        errors::operation_name(self.inner.operation.clone())
    }
    #[getter]
    fn gamma_relative_bump(&self) -> Option<f64> {
        match self.inner.operation {
            PricingOperation::Aad {
                gamma_relative_bump,
            } => gamma_relative_bump,
            _ => None,
        }
    }
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            CapabilityStatus::Supported => "supported",
            CapabilityStatus::Conditional => "conditional",
            CapabilityStatus::Unsupported => "unsupported",
        }
    }
    #[getter]
    fn reason(&self) -> Option<&'static str> {
        self.inner.reason.map(errors::capability_reason)
    }
    #[getter]
    fn detail(&self) -> &str {
        &self.inner.detail
    }
}
#[pyclass(frozen, name = "PricingTask")]
#[derive(Clone, Debug)]
pub struct PyPricingTask {
    pub(crate) inner: PricingTask,
}
#[pymethods]
impl PyPricingTask {
    #[staticmethod]
    #[pyo3(signature=(source, *, operation="evaluate", gamma_relative_bump=None, measures=None))]
    fn from_source(
        source: &Bound<'_, PyAny>,
        operation: &str,
        gamma_relative_bump: Option<f64>,
        measures: Measures<'_>,
    ) -> PyResult<Self> {
        PricingTask::new(
            pricing_source::extract_source(source)?,
            self::operation(operation, gamma_relative_bump, &measures)?,
        )
        .map(|inner| Self { inner })
        .map_err(|e| errors::exception(e, errors::Stage::Validation, Some(operation)))
    }
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        pricing::parse_pricing_task_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(|e| wire_error(py, &e, "pricing_task", 1))
    }
    #[staticmethod]
    #[pyo3(signature=(specification, *, operation="evaluate", gamma_relative_bump=None, measures=None))]
    fn from_specification(
        specification: &specification::PyPricingSpecification,
        operation: &str,
        gamma_relative_bump: Option<f64>,
        measures: Measures<'_>,
    ) -> PyResult<Self> {
        PricingTask::new(
            specification.inner.clone(),
            self::operation(operation, gamma_relative_bump, &measures)?,
        )
        .map(|inner| Self { inner })
        .map_err(|e| errors::exception(e, errors::Stage::Validation, Some(operation)))
    }
    #[pyo3(signature=(operation, *, gamma_relative_bump=None, measures=None))]
    fn with_operation(
        &self,
        operation: &str,
        gamma_relative_bump: Option<f64>,
        measures: Measures<'_>,
    ) -> PyResult<Self> {
        self.inner
            .with_operation(self::operation(operation, gamma_relative_bump, &measures)?)
            .map(|inner| Self { inner })
            .map_err(|e| errors::exception(e, errors::Stage::Validation, Some(operation)))
    }
    #[pyo3(signature=(operation, *, gamma_relative_bump=None, measures=None))]
    fn capability(
        &self,
        operation: &str,
        gamma_relative_bump: Option<f64>,
        measures: Measures<'_>,
    ) -> PyResult<PyPricingCapability> {
        Ok(PyPricingCapability {
            inner: self.inner.source().capability(self::operation(
                operation,
                gamma_relative_bump,
                &measures,
            )?),
        })
    }
    /// Capabilities of `evaluate`, first-order `aad` and `local_variance_risk`,
    /// in that order, from pure source inspection.
    fn available_operations(&self) -> Vec<PyPricingCapability> {
        self.inner
            .source()
            .available_operations()
            .into_iter()
            .map(|inner| PyPricingCapability { inner })
            .collect()
    }
    /// The requested measures, or `None` for the three legacy operations.
    #[getter]
    fn measures(&self) -> Option<Vec<measures::PyMeasureRequest>> {
        self.inner.measures().map(|set| {
            set.requests()
                .iter()
                .map(|inner| measures::PyMeasureRequest {
                    inner: inner.clone(),
                })
                .collect()
        })
    }
    /// Measures the `measures` operation does not statically refuse for this
    /// source, each mapped to the methods that apply.
    fn available_measures<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let available = PyDict::new(py);
        for row in self.inner.source().available_measures() {
            available.set_item(
                row.measure.as_str(),
                row.methods.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
            )?;
        }
        Ok(available)
    }
    /// One answer per request, in request order, from pure source inspection.
    fn measure_capabilities(
        &self,
        measures: Vec<PyRef<'_, measures::PyMeasureRequest>>,
    ) -> Vec<measures::PyMeasureCapability> {
        self.inner
            .source()
            .measure_capabilities(&measures::requests(&measures))
            .into_iter()
            .map(|inner| measures::PyMeasureCapability { inner })
            .collect()
    }
    /// Calibrations and evaluations the bump-and-revalue requests among
    /// `measures` would run beyond the base price. Nothing is compiled.
    fn bump_cost<'py>(
        &self,
        py: Python<'py>,
        measures: Vec<PyRef<'_, measures::PyMeasureRequest>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cost = self
            .inner
            .source()
            .bump_cost(&measures::requests(&measures))
            .map_err(|e| errors::exception(e, errors::Stage::Validation, Some("measures")))?;
        measures::cost(py, cost)
    }
    /// The source composition's row in the common operation matrix.
    #[getter]
    fn source_family(&self) -> &'static str {
        self.inner.source().family().as_str()
    }
    fn to_json(&self) -> PyResult<String> {
        pricing::pricing_task_to_json(&self.inner).map_err(crate::errors::serialize_exception)
    }
    fn to_pretty_json(&self) -> PyResult<String> {
        pricing::pricing_task_to_pretty_json(&self.inner)
            .map_err(crate::errors::serialize_exception)
    }
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.inner
            .fingerprint()
            .map(|f| f.to_string())
            .map_err(crate::errors::serialize_exception)
    }
    fn compile(&self, py: Python<'_>) -> PyResult<PyCompiledPricingTask> {
        py.detach(|| self.inner.compile())
            .map(|inner| PyCompiledPricingTask { inner })
            .map_err(|e| {
                errors::exception(
                    e,
                    errors::Stage::Compile,
                    Some(errors::operation_name(self.inner.operation())),
                )
            })
    }
}
#[pyclass(frozen, name = "CompiledPricingTask")]
#[derive(Clone, Debug)]
pub struct PyCompiledPricingTask {
    inner: pricing::CompiledPricingTask,
}
#[pymethods]
impl PyCompiledPricingTask {
    #[getter]
    fn task(&self) -> PyPricingTask {
        PyPricingTask {
            inner: self.inner.task().clone(),
        }
    }
    /// Select an operation using this plan's existing compilation and calibration.
    #[pyo3(signature=(operation, *, gamma_relative_bump=None, measures=None))]
    fn with_operation(
        &self,
        py: Python<'_>,
        operation: &str,
        gamma_relative_bump: Option<f64>,
        measures: Measures<'_>,
    ) -> PyResult<Self> {
        let selected = self::operation(operation, gamma_relative_bump, &measures)?;
        py.detach(|| self.inner.with_operation(selected))
            .map(|inner| Self { inner })
            .map_err(|e| errors::exception(e, errors::Stage::Compile, Some(operation)))
    }
    /// Inspect source prerequisites and the realized plan's native preflight.
    #[pyo3(signature=(operation, *, gamma_relative_bump=None, measures=None))]
    fn capability_for(
        &self,
        py: Python<'_>,
        operation: &str,
        gamma_relative_bump: Option<f64>,
        measures: Measures<'_>,
    ) -> PyResult<PyPricingCapability> {
        let selected = self::operation(operation, gamma_relative_bump, &measures)?;
        Ok(PyPricingCapability {
            inner: py.detach(|| self.inner.capability_for(selected)),
        })
    }
    /// Calibrations realized for bumped sides and side evaluations run so far.
    fn bump_cost<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        measures::cost(py, self.inner.bump_cost())
    }
    /// One answer per request against the source and the realized plan.
    fn measure_capabilities(
        &self,
        measures: Vec<PyRef<'_, measures::PyMeasureRequest>>,
    ) -> Vec<measures::PyMeasureCapability> {
        self.inner
            .measure_capabilities(&measures::requests(&measures))
            .into_iter()
            .map(|inner| measures::PyMeasureCapability { inner })
            .collect()
    }
    #[getter]
    fn task_fingerprint(&self) -> String {
        self.inner.task_fingerprint().to_string()
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint()
    }
    fn capability(&self) -> PyPricingCapability {
        PyPricingCapability {
            inner: self.inner.capability(),
        }
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyPricingTaskResult> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyPricingTaskResult { inner })
            .map_err(|e| {
                errors::exception(
                    e,
                    errors::Stage::Evaluate,
                    Some(errors::operation_name(self.inner.operation())),
                )
            })
    }
}
#[pyclass(frozen, name = "PricingTaskResult")]
#[derive(Clone, Debug)]
pub struct PyPricingTaskResult {
    inner: pricing::PricingTaskResult,
}
#[pymethods]
impl PyPricingTaskResult {
    #[getter]
    fn report(&self) -> PyValuationReport {
        PyValuationReport {
            inner: self.inner.report.clone(),
        }
    }
    #[getter]
    fn native_result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.inner.native_result {
            PricingTaskOutput::SingleAsset(p) => Py::new(
                py,
                specification::PySpecificationResult { inner: p.clone() },
            )?
            .into_any(),
            PricingTaskOutput::MultiAsset(p) => Py::new(
                py,
                multi_asset::PyMultiAssetPrice {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            PricingTaskOutput::StochasticDividend(p) => Py::new(
                py,
                stochastic_dividends::PyStochasticDividendPrice { inner: p.clone() },
            )?
            .into_any(),
            PricingTaskOutput::HullWhiteRisk(p) => Py::new(
                py,
                hull_white::PyHullWhiteAadRisk {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            PricingTaskOutput::LocalVarianceRisk(p) => Py::new(
                py,
                lsv::PyLsvLocalVarianceRisk {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            PricingTaskOutput::StochasticDividendRisk(p) => Py::new(
                py,
                stochastic_dividends::PyStochasticDividendAadRisk {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            PricingTaskOutput::StochasticDividendLocalVarianceRisk(p) => Py::new(
                py,
                stochastic_dividends::PyStochasticDividendLocalVarianceRisk {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            PricingTaskOutput::RoughFamilyLsvMeasureRisk(p) => Py::new(
                py,
                lsv::PyRoughFamilyLsvMeasureRisk {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            PricingTaskOutput::RoughVolatilityMeasureRisk(p) => Py::new(
                py,
                rough_volatility::PyRoughVolatilityMeasureRisk {
                    inner: (**p).clone(),
                },
            )?
            .into_any(),
            _ => return Err(crate::errors::internal_exception("unknown task output")),
        })
    }
}
#[pyclass(frozen, name = "ReportedRisk")]
#[derive(Clone, Debug)]
pub struct PyReportedRisk {
    inner: ReportedRisk,
}
#[pymethods]
impl PyReportedRisk {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlying_ids.clone()
    }
    #[getter]
    fn axes(&self) -> Vec<(String, Vec<f64>)> {
        self.inner
            .axes
            .iter()
            .map(|a| (a.name.clone(), a.values.clone()))
            .collect()
    }
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values.clone()
    }
    #[getter]
    fn standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.standard_errors.clone()
    }
    #[getter]
    fn method(&self) -> &str {
        &self.inner.method
    }
    #[getter]
    fn convention(&self) -> &str {
        &self.inner.convention
    }
    #[getter]
    fn unit(&self) -> &'static str {
        self.inner.unit.as_str()
    }
    #[getter]
    fn calibration(&self) -> &'static str {
        use pricing::CalibrationTreatment as C;
        match self.inner.calibration {
            C::NotApplicable => "not_applicable",
            C::Recalibrated => "recalibrated",
            C::FixedCalibration => "fixed_calibration",
            C::FixedExercisePolicy => "fixed_exercise_policy",
        }
    }
    /// The requested measure this row answers; `None` on rows of the three
    /// legacy operations.
    #[getter]
    fn measure(&self) -> Option<&'static str> {
        self.inner.measure.map(|m| m.as_str())
    }
    /// The method the task requested for `measure`, with any bump.
    #[getter]
    fn method_detail<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .method_detail
            .map(|m| measures::method_detail(py, m))
            .transpose()
    }
    /// The parameter label of a `model_parameters` row.
    #[getter]
    fn model_parameter(&self) -> Option<String> {
        self.inner.model_parameter.map(|p| p.label())
    }
}
fn calibration_name(c: pricing::CalibrationTreatment) -> &'static str {
    use pricing::CalibrationTreatment as C;
    match c {
        C::NotApplicable => "not_applicable",
        C::Recalibrated => "recalibrated",
        C::FixedCalibration => "fixed_calibration",
        C::FixedExercisePolicy => "fixed_exercise_policy",
    }
}
fn scope_name(s: pricing::UncertaintyScope) -> &'static str {
    match s {
        pricing::UncertaintyScope::PricingSampling => "pricing_sampling",
        pricing::UncertaintyScope::ConditionalOnCalibration => "conditional_on_calibration",
        pricing::UncertaintyScope::ConditionalOnExercisePolicy => "conditional_on_exercise_policy",
    }
}
#[pyclass(frozen, name = "CalibrationMetric")]
#[derive(Clone, Debug)]
pub struct PyCalibrationMetric {
    inner: CalibrationMetric,
}
#[pymethods]
impl PyCalibrationMetric {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values.clone()
    }
    #[getter]
    fn unit(&self) -> &'static str {
        self.inner.unit.as_str()
    }
    #[getter]
    fn convention(&self) -> &str {
        &self.inner.convention
    }
}
#[pyclass(frozen, name = "ReportedCalibration")]
#[derive(Clone, Debug)]
pub struct PyReportedCalibration {
    inner: ReportedCalibration,
}
#[pymethods]
impl PyReportedCalibration {
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind.as_str()
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlying_ids.clone()
    }
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }
    #[getter]
    fn axes(&self) -> Vec<(String, Vec<f64>)> {
        self.inner
            .axes
            .iter()
            .map(|a| (a.name.clone(), a.values.clone()))
            .collect()
    }
    #[getter]
    fn metrics(&self) -> Vec<PyCalibrationMetric> {
        self.inner
            .metrics
            .iter()
            .map(|m| PyCalibrationMetric { inner: m.clone() })
            .collect()
    }
}
#[pyclass(frozen, name = "ValuationReport")]
#[derive(Clone, Debug)]
pub struct PyValuationReport {
    inner: ValuationReport,
}
#[pymethods]
impl PyValuationReport {
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        pricing::parse_valuation_report_json_with_context(
            json.as_bytes(),
            pricing::JsonLimits::DEFAULT,
        )
        .map(|inner| Self { inner })
        .map_err(|e| {
            wire_error(
                py,
                e.wire_error(),
                "valuation_report",
                e.schema_version().unwrap_or(2),
            )
        })
    }
    fn to_json(&self) -> PyResult<String> {
        pricing::valuation_report_to_json(&self.inner).map_err(crate::errors::serialize_exception)
    }
    fn to_pretty_json(&self) -> PyResult<String> {
        pricing::valuation_report_to_pretty_json(&self.inner)
            .map_err(crate::errors::serialize_exception)
    }
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error()
    }
    #[getter]
    fn currency_id(&self) -> u16 {
        self.inner.currency_id()
    }
    #[getter]
    fn task_fingerprint(&self) -> String {
        self.inner.task_fingerprint().to_string()
    }
    #[getter]
    fn plan_fingerprint(&self) -> &str {
        self.inner.plan_fingerprint()
    }
    #[getter]
    fn library_version(&self) -> &str {
        self.inner.library_version()
    }
    #[getter]
    fn platform(&self) -> &str {
        self.inner.platform()
    }
    #[getter]
    fn independent_sampling_units(&self) -> u64 {
        self.inner.independent_sampling_units()
    }
    #[getter]
    fn evaluated_paths(&self) -> u128 {
        self.inner.evaluated_paths()
    }
    #[getter]
    fn estimator(&self) -> &'static str {
        match self.inner.estimator() {
            pricing::EstimatorKind::Analytical => "analytical",
            pricing::EstimatorKind::PseudoMonteCarlo => "pseudo_monte_carlo",
            pricing::EstimatorKind::RandomizedQuasiMonteCarlo => "randomized_quasi_monte_carlo",
        }
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        match self.inner.uncertainty_scope() {
            pricing::UncertaintyScope::PricingSampling => "pricing_sampling",
            pricing::UncertaintyScope::ConditionalOnCalibration => "conditional_on_calibration",
            pricing::UncertaintyScope::ConditionalOnExercisePolicy => {
                "conditional_on_exercise_policy"
            }
        }
    }
    #[getter]
    fn calibration_seeds(&self) -> Vec<u64> {
        self.inner.calibration_seeds().to_vec()
    }
    #[getter]
    fn calibration_diagnostics(&self) -> Option<Vec<PyReportedCalibration>> {
        self.inner.calibration_diagnostics().map(|blocks| {
            blocks
                .iter()
                .map(|b| PyReportedCalibration { inner: b.clone() })
                .collect()
        })
    }
    #[getter]
    fn risks(&self) -> Vec<PyReportedRisk> {
        self.inner
            .risks()
            .iter()
            .map(|r| PyReportedRisk { inner: r.clone() })
            .collect()
    }
    #[getter]
    fn vega_kt(&self) -> Option<PyVegaKtResult> {
        self.inner
            .vega_kt()
            .map(|v| PyVegaKtResult { inner: v.clone() })
    }
    /// The raw-unit row answering `measure`, if the task requested it.
    fn risk(&self, py: Python<'_>, measure: &str) -> PyResult<Option<PyReportedRisk>> {
        let kind = pricing::MeasureKind::ALL
            .into_iter()
            .find(|k| k.as_str() == measure)
            .ok_or_else(|| {
                errors::ErrorMetadata::domain("invalid_measure", Some("measure".into())).attach(
                    py,
                    PyValueError::new_err("unknown measure name"),
                    Some("measures"),
                )
            })?;
        Ok(self
            .inner
            .risk(kind)
            .map(|r| PyReportedRisk { inner: r.clone() }))
    }
    /// The `model_parameters` row of one parameter label, if requested.
    fn model_parameter_risk(
        &self,
        py: Python<'_>,
        parameter: &str,
    ) -> PyResult<Option<PyReportedRisk>> {
        let id = pricing::ModelParameterId::from_label(parameter).ok_or_else(|| {
            errors::ErrorMetadata::domain("invalid_model_parameter", Some("parameter".into()))
                .attach(
                    py,
                    PyValueError::new_err("unknown model parameter label"),
                    Some("measures"),
                )
        })?;
        Ok(self
            .inner
            .model_parameter_risk(id)
            .map(|r| PyReportedRisk { inner: r.clone() }))
    }
    /// The price and every risk coordinate as flat rows of plain data.
    fn to_records<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .to_records()
            .into_iter()
            .map(|r| {
                let row = PyDict::new(py);
                row.set_item("measure", r.measure)?;
                row.set_item("name", r.name)?;
                row.set_item("underlying_ids", r.underlying_ids)?;
                let coordinate = PyDict::new(py);
                for (axis, value) in r.coordinate {
                    coordinate.set_item(axis, value)?;
                }
                row.set_item("coordinate", coordinate)?;
                row.set_item("value", r.value)?;
                row.set_item("standard_error", r.standard_error)?;
                row.set_item("unit", r.unit)?;
                row.set_item("method", r.method)?;
                row.set_item("calibration", calibration_name(r.calibration))?;
                row.set_item("uncertainty_scope", scope_name(r.uncertainty_scope))?;
                row.set_item(
                    "task_fingerprint",
                    self.inner.task_fingerprint().to_string(),
                )?;
                Ok(row)
            })
            .collect()
    }
}
#[pyfunction]
fn pricing_task_json_schema() -> &'static str {
    pricing::current_pricing_task_schema()
}
#[pyfunction]
fn valuation_report_json_schema() -> &'static str {
    pricing::current_valuation_report_schema()
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPricingCapability>()?;
    module.add_class::<PyPricingTask>()?;
    module.add_class::<PyCompiledPricingTask>()?;
    module.add_class::<PyPricingTaskResult>()?;
    module.add_class::<PyReportedRisk>()?;
    module.add_class::<PyCalibrationMetric>()?;
    module.add_class::<PyReportedCalibration>()?;
    module.add_class::<PyValuationReport>()?;
    module.add_function(wrap_pyfunction!(pricing_task_json_schema, module)?)?;
    module.add_function(wrap_pyfunction!(valuation_report_json_schema, module)?)?;
    Ok(())
}
