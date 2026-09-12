use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::lsv::{BergomiLsvPricingPlan, LsvLocalVarianceRisk, LsvPrice};
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::Bergomi1Factor;
use pyo3::prelude::*;

#[pyclass(frozen, name = "BergomiLsvPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBergomiLsvPlan {
    inner: BergomiLsvPricingPlan,
}

#[pymethods]
impl PyBergomiLsvPlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(target_request, *, mean_reversion, vol_of_vol, correlation, particle_count,
        calibration_seed, log_bandwidth, minimum_effective_samples, retain_reverse_trace,
        worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        target_request: &PyPricingRequest,
        mean_reversion: f64,
        vol_of_vol: f64,
        correlation: f64,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        retain_reverse_trace: bool,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let issue = |message: String| {
            validation_exception(
                py,
                PyValidationIssue::domain("/lsv", "invalid_lsv_configuration", message),
            )
        };
        let factor = Bergomi1Factor::new(mean_reversion, vol_of_vol, correlation)
            .map_err(|e| issue(e.to_string()))?;
        let particles = LsvParticleConfig::new(
            particle_count,
            calibration_seed,
            log_bandwidth,
            minimum_effective_samples,
            retain_reverse_trace,
        )
        .map_err(|e| issue(e.to_string()))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| issue(e.to_string()))?;
        let request = target_request.inner.clone();
        py.detach(|| BergomiLsvPricingPlan::compile(&request, factor, particles, policy))
            .map(|inner| Self { inner })
            .map_err(pricing_exception)
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyLsvPrice> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyLsvPrice { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_local_variance_risk(&self, py: Python<'_>) -> PyResult<PyLsvLocalVarianceRisk> {
        py.detach(|| self.inner.evaluate_local_variance_risk())
            .map(|inner| PyLsvLocalVarianceRisk { inner })
            .map_err(pricing_exception)
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.calibration().surface().times().to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.calibration().surface().log_nodes().to_vec()
    }
    #[getter]
    fn squared_leverage(&self) -> Vec<f64> {
        self.inner
            .calibration()
            .surface()
            .squared_leverage()
            .to_vec()
    }
    /// Per-time count of moment nodes using a supported neighbouring estimate.
    #[getter]
    fn extrapolated_moment_nodes(&self) -> Vec<usize> {
        self.inner
            .calibration()
            .diagnostics()
            .iter()
            .map(|r| r.extrapolated_nodes)
            .collect()
    }
    #[getter]
    fn minimum_effective_samples(&self) -> Vec<f64> {
        self.inner
            .calibration()
            .diagnostics()
            .iter()
            .map(|r| r.minimum_effective_samples)
            .collect()
    }
}

#[pyclass(frozen, name = "LsvPrice", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyLsvPrice {
    inner: LsvPrice,
}
#[pymethods]
impl PyLsvPrice {
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error
    }
    #[getter]
    fn independent_sampling_units(&self) -> u64 {
        self.inner.independent_sampling_units
    }
    #[getter]
    fn evaluated_paths(&self) -> u128 {
        self.inner.evaluated_paths
    }
    #[getter]
    fn calibration_seed(&self) -> u64 {
        self.inner.calibration_seed
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint.to_string()
    }
    #[getter]
    fn scheme(&self) -> &'static str {
        self.inner.scheme
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_conditional_on_calibration"
    }
}

#[pyclass(frozen, name = "LsvLocalVarianceRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyLsvLocalVarianceRisk {
    inner: LsvLocalVarianceRisk,
}
#[pymethods]
impl PyLsvLocalVarianceRisk {
    #[getter]
    fn price(&self) -> PyLsvPrice {
        PyLsvPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes.to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.log_moneyness_nodes.to_vec()
    }
    #[getter]
    fn node_adjoints(&self) -> Vec<f64> {
        self.inner.node_adjoints.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.standard_errors.as_ref().map(|v| v.to_vec())
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "relative_dupire_variance_nodes_in_f"
    }
}
