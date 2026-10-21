//! Thin adapters over complete Rust tasks and common reports.
use crate::*;
use pricing::{
    CapabilityReason, CapabilityStatus, PricingCapability, PricingOperation, PricingTask,
    PricingTaskOutput, ReportedRisk, ValuationReport,
};

fn operation(name: &str, bump: Option<f64>) -> PyResult<PricingOperation> {
    match (name, bump) {
        ("evaluate", None) => Ok(PricingOperation::Evaluate),
        ("aad", gamma_relative_bump) => Ok(PricingOperation::Aad {
            gamma_relative_bump,
        }),
        ("local_variance_risk", None) => Ok(PricingOperation::LocalVarianceRisk),
        _ => Err(PyValueError::new_err(
            "operation must be evaluate, aad or local_variance_risk; Gamma bump belongs to aad",
        )),
    }
}
fn wire_error(py: Python<'_>, error: &WireError, kind: &'static str) -> PyErr {
    let mut issue = PyValidationIssue::wire(error, kind);
    issue.schema_version = match error {
        WireError::UnsupportedSchemaVersion(version)
        | WireError::UnsupportedSchemaFeature {
            schema_version: version,
            ..
        } => *version,
        _ => 1,
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
        match self.inner.operation {
            PricingOperation::Evaluate => "evaluate",
            PricingOperation::Aad { .. } => "aad",
            PricingOperation::LocalVarianceRisk => "local_variance_risk",
            _ => "unknown",
        }
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
        self.inner.reason.map(|r| match r {
            CapabilityReason::CompilationRequired => "compilation_required",
            CapabilityReason::NativePreflightRequired => "native_preflight_required",
            CapabilityReason::DedicatedRiskApi => "dedicated_risk_api",
            CapabilityReason::ModelCombination => "model_combination",
            CapabilityReason::DiscontinuousPayoff => "discontinuous_payoff",
            CapabilityReason::ReverseTraceRequired => "reverse_trace_required",
            CapabilityReason::InvalidGammaBump => "invalid_gamma_bump",
            CapabilityReason::InvalidSource => "invalid_source",
            _ => "unknown",
        })
    }
    #[getter]
    fn detail(&self) -> &str {
        &self.inner.detail
    }
}
#[pyclass(frozen, name = "PricingTask")]
#[derive(Clone, Debug)]
pub struct PyPricingTask {
    inner: PricingTask,
}
#[pymethods]
impl PyPricingTask {
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        pricing::parse_pricing_task_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(|e| wire_error(py, &e, "pricing_task"))
    }
    #[staticmethod]
    #[pyo3(signature=(specification, *, operation="evaluate", gamma_relative_bump=None))]
    fn from_specification(
        specification: &specification::PyPricingSpecification,
        operation: &str,
        gamma_relative_bump: Option<f64>,
    ) -> PyResult<Self> {
        PricingTask::new(
            specification.inner.clone(),
            self::operation(operation, gamma_relative_bump)?,
        )
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    #[pyo3(signature=(operation, *, gamma_relative_bump=None))]
    fn with_operation(&self, operation: &str, gamma_relative_bump: Option<f64>) -> PyResult<Self> {
        PricingTask::new(
            self.inner.source().clone(),
            self::operation(operation, gamma_relative_bump)?,
        )
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    #[pyo3(signature=(operation, *, gamma_relative_bump=None))]
    fn capability(
        &self,
        operation: &str,
        gamma_relative_bump: Option<f64>,
    ) -> PyResult<PyPricingCapability> {
        Ok(PyPricingCapability {
            inner: self
                .inner
                .source()
                .capability(self::operation(operation, gamma_relative_bump)?),
        })
    }
    fn to_json(&self) -> PyResult<String> {
        pricing::pricing_task_to_json(&self.inner).map_err(pricing_exception)
    }
    fn to_pretty_json(&self) -> PyResult<String> {
        pricing::pricing_task_to_pretty_json(&self.inner).map_err(pricing_exception)
    }
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.inner
            .fingerprint()
            .map(|f| f.to_string())
            .map_err(pricing_exception)
    }
    fn compile(&self, py: Python<'_>) -> PyResult<PyCompiledPricingTask> {
        py.detach(|| self.inner.compile())
            .map(|inner| PyCompiledPricingTask { inner })
            .map_err(pricing_exception)
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
            .map_err(pricing_exception)
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
            _ => return Err(pricing_exception("unknown task output")),
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
        use pricing::SensitivityUnit as U;
        match self.inner.unit {
            U::CurrencyPerSpot => "currency_per_spot",
            U::CurrencyPerSpotSquared => "currency_per_spot_squared",
            U::CurrencyPerUnitVolatility => "currency_per_unit_volatility",
            U::CurrencyPerVariance => "currency_per_variance",
            U::CurrencyPerLogDiscountFactor => "currency_per_log_discount_factor",
            U::CurrencyPerLogDensity => "currency_per_log_density",
            U::CurrencyPerUnitParameter => "currency_per_unit_parameter",
            U::CurrencyPerOnePercentSpot => "currency_per_one_percent_spot",
            U::CurrencyPerOnePercentSpotSquared => "currency_per_one_percent_spot_squared",
            U::CurrencyPerVolatilityPoint => "currency_per_volatility_point",
            U::CurrencyPerBasisPoint => "currency_per_basis_point",
            _ => "unknown",
        }
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
        pricing::parse_valuation_report_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(|e| wire_error(py, &e, "valuation_report"))
    }
    fn to_json(&self) -> PyResult<String> {
        pricing::valuation_report_to_json(&self.inner).map_err(pricing_exception)
    }
    fn to_pretty_json(&self) -> PyResult<String> {
        pricing::valuation_report_to_pretty_json(&self.inner).map_err(pricing_exception)
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
    module.add_class::<PyValuationReport>()?;
    module.add_function(wrap_pyfunction!(pricing_task_json_schema, module)?)?;
    module.add_function(wrap_pyfunction!(valuation_report_json_schema, module)?)?;
    Ok(())
}
