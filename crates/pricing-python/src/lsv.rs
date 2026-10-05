use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::lsv::{
    BergomiLsvPricingPlan, LsvLocalVarianceRisk, LsvPrice, RoughBergomiLsvPricingPlan,
};
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::{Bergomi1Factor, Bergomi2Factor, RoughBergomi};
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

#[pyclass(frozen, name = "Bergomi2FactorLsvPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBergomi2FactorLsvPlan {
    inner: BergomiLsvPricingPlan<Bergomi2Factor>,
}

#[pymethods]
impl PyBergomi2FactorLsvPlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(target_request, *, mean_reversions, vol_of_vol, mixing_weight, spot_correlations, factor_correlation, particle_count,
        calibration_seed, log_bandwidth, minimum_effective_samples, retain_reverse_trace,
        worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        target_request: &PyPricingRequest,
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
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
        let factor = Bergomi2Factor::new(
            mean_reversions,
            vol_of_vol,
            mixing_weight,
            spot_correlations,
            factor_correlation,
        )
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

/// Deterministic-rate rough Bergomi LSV. `vol_of_vol` is eta, the coefficient
/// of log variance, as for `RoughBergomiModel`.
#[pyclass(frozen, name = "RoughBergomiLsvPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughBergomiLsvPlan {
    inner: RoughBergomiLsvPricingPlan,
}

#[pymethods]
impl PyRoughBergomiLsvPlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(target_request, *, hurst, vol_of_vol, correlation, particle_count,
        calibration_seed, log_bandwidth, minimum_effective_samples, retain_reverse_trace,
        worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        target_request: &PyPricingRequest,
        hurst: f64,
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
        let model =
            RoughBergomi::new(hurst, vol_of_vol, correlation).map_err(|e| issue(e.to_string()))?;
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
        py.detach(|| RoughBergomiLsvPricingPlan::compile(&request, model, particles, policy))
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

#[pyclass(frozen, name = "RoughFamilyLsvPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughFamilyLsvPlan {
    inner: pricing::rough_volatility::RoughFamilyLsvPricingPlan,
}

#[pymethods]
impl PyRoughFamilyLsvPlan {
    fn evaluate_mixed_bergomi_parameter_risk(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyMixedBergomiLsvParameterRisk> {
        py.detach(|| self.inner.evaluate_mixed_bergomi_parameter_risk())
            .map(|inner| PyMixedBergomiLsvParameterRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_mixed_bergomi_parameter_risk_with_hurst(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyMixedBergomiLsvParameterRisk> {
        py.detach(|| {
            self.inner
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
        })
        .map(|inner| PyMixedBergomiLsvParameterRisk { inner })
        .map_err(pricing_exception)
    }

    #[pyo3(signature=(*, include_hurst=false))]
    fn evaluate_heston_parameter_risk(
        &self,
        py: Python<'_>,
        include_hurst: bool,
    ) -> PyResult<PyHestonLsvParameterRisk> {
        py.detach(|| self.inner.evaluate_heston_parameter_risk(include_hurst))
            .map(|inner| PyHestonLsvParameterRisk { inner })
            .map_err(pricing_exception)
    }

    fn evaluate_frozen_leverage_gamma_bump(
        &self,
        py: Python<'_>,
        spot_bump: f64,
    ) -> PyResult<PyRoughFamilyLsvGamma> {
        py.detach(|| self.inner.evaluate_frozen_leverage_gamma_bump(spot_bump))
            .map(|inner| PyRoughFamilyLsvGamma { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_sticky_moneyness_gamma_bump(
        &self,
        py: Python<'_>,
        spot_bump: f64,
    ) -> PyResult<PyRoughFamilyLsvGamma> {
        py.detach(|| self.inner.evaluate_sticky_moneyness_gamma_bump(spot_bump))
            .map(|inner| PyRoughFamilyLsvGamma { inner })
            .map_err(pricing_exception)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(target_request, model, *, particle_count,
        calibration_seed, log_bandwidth, minimum_effective_samples, retain_reverse_trace,
        worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        target_request: &PyPricingRequest,
        model: &super::rough_volatility::PyRoughVolatilityModel,
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
        let model = model.inner.clone();
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
        py.detach(|| {
            pricing::rough_volatility::RoughFamilyLsvPricingPlan::compile(
                &request, model, particles, policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    fn market_iv_risk_plan(
        &self,
        py: Python<'_>,
        surface: &crate::market_iv::PyMarketIvSurface,
    ) -> PyResult<PyRoughFamilyLsvMarketIvRiskPlan> {
        let source = surface.inner.clone();
        py.detach(|| self.inner.market_iv_risk_plan(source))
            .map(|inner| PyRoughFamilyLsvMarketIvRiskPlan { inner })
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
    fn evaluate_frozen_leverage_delta(&self, py: Python<'_>) -> PyResult<PyRoughFamilyLsvDelta> {
        py.detach(|| self.inner.evaluate_frozen_leverage_delta())
            .map(|inner| PyRoughFamilyLsvDelta { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_sticky_moneyness_delta(&self, py: Python<'_>) -> PyResult<PyRoughFamilyLsvDelta> {
        py.detach(|| self.inner.evaluate_sticky_moneyness_delta())
            .map(|inner| PyRoughFamilyLsvDelta { inner })
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

#[pyclass(frozen, name = "RoughFamilyLsvDelta", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughFamilyLsvDelta {
    inner: pricing::rough_volatility::RoughFamilyLsvDelta,
}
#[pymethods]
impl PyRoughFamilyLsvDelta {
    #[getter]
    fn price(&self) -> PyLsvPrice {
        PyLsvPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta
    }
    #[getter]
    fn delta_standard_error(&self) -> f64 {
        self.inner.delta_standard_error
    }
    #[getter]
    fn convention(&self) -> &'static str {
        self.inner.convention.as_str()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "physical_spot_fixed_curves_and_cash_dividends"
    }
}

#[pyclass(frozen, name = "RoughFamilyLsvMarketIvRiskPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughFamilyLsvMarketIvRiskPlan {
    inner: pricing::rough_volatility::RoughFamilyLsvMarketIvRiskPlan,
}
#[pymethods]
impl PyRoughFamilyLsvMarketIvRiskPlan {
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyRoughFamilyLsvMarketIvRisk> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyRoughFamilyLsvMarketIvRisk { inner })
            .map_err(pricing_exception)
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint().to_string()
    }
}

#[pyclass(frozen, name = "RoughFamilyLsvMarketIvRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughFamilyLsvMarketIvRisk {
    inner: pricing::rough_volatility::RoughFamilyLsvMarketIvRisk,
}
#[pymethods]
impl PyRoughFamilyLsvMarketIvRisk {
    #[getter]
    fn price(&self) -> PyLsvPrice {
        PyLsvPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn maturity_nodes(&self) -> Vec<f64> {
        self.inner.maturity_nodes.to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.log_moneyness_nodes.to_vec()
    }
    #[getter]
    fn implied_volatilities(&self) -> Vec<f64> {
        self.inner.implied_volatilities.to_vec()
    }
    #[getter]
    fn quote_adjoints(&self) -> Vec<f64> {
        self.inner.quote_adjoints.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.standard_errors.as_ref().map(|v| v.to_vec())
    }
    #[getter]
    fn parallel_vega(&self) -> f64 {
        self.inner.parallel_vega
    }
    #[getter]
    fn parallel_standard_error(&self) -> Option<f64> {
        self.inner.parallel_standard_error
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        pricing::rough_volatility::RoughFamilyLsvMarketIvRiskPlan::COORDINATE
    }
    #[getter]
    fn interpolation(&self) -> &'static str {
        pricing::market::MARKET_IV_INTERPOLATION
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_conditional_on_calibration"
    }
}

#[pyclass(frozen, name = "RoughFamilyLsvGamma", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughFamilyLsvGamma {
    inner: pricing::rough_volatility::RoughFamilyLsvGamma,
}
#[pymethods]
impl PyRoughFamilyLsvGamma {
    #[getter]
    fn price(&self) -> PyLsvPrice {
        PyLsvPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }
    #[getter]
    fn gamma_standard_error(&self) -> f64 {
        self.inner.gamma_standard_error
    }
    #[getter]
    fn half_bump_gamma(&self) -> f64 {
        self.inner.half_bump_gamma
    }
    #[getter]
    fn half_bump_standard_error(&self) -> f64 {
        self.inner.half_bump_standard_error
    }
    #[getter]
    fn bump_difference(&self) -> f64 {
        self.inner.bump_difference
    }
    #[getter]
    fn bump_difference_standard_error(&self) -> f64 {
        self.inner.bump_difference_standard_error
    }
    #[getter]
    fn spot_bump(&self) -> f64 {
        self.inner.spot_bump
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn convention(&self) -> &'static str {
        self.inner.convention
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}

#[pyclass(frozen, name = "HestonLsvParameterRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonLsvParameterRisk {
    inner: pricing::rough_volatility::HestonLsvParameterRisk,
}
#[pymethods]
impl PyHestonLsvParameterRisk {
    #[getter]
    fn price(&self) -> PyLsvPrice {
        PyLsvPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner
            .parameter_names
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
    #[getter]
    fn parameter_adjoints(&self) -> Vec<f64> {
        self.inner.parameter_adjoints.to_vec()
    }
    #[getter]
    fn direct_adjoints(&self) -> Vec<f64> {
        self.inner.direct_adjoints.to_vec()
    }
    #[getter]
    fn calibration_adjoints(&self) -> Vec<f64> {
        self.inner.calibration_adjoints.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.standard_errors.as_ref().map(|v| v.to_vec())
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "heston_parameters_fixed_relative_local_variance_target"
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}

#[pyclass(frozen, name = "MixedBergomiLsvParameterRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMixedBergomiLsvParameterRisk {
    inner: pricing::rough_volatility::MixedBergomiLsvParameterRisk,
}
#[pymethods]
impl PyMixedBergomiLsvParameterRisk {
    #[getter]
    fn price(&self) -> PyLsvPrice {
        PyLsvPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner
            .parameter_names
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
    #[getter]
    fn parameter_adjoints(&self) -> Vec<f64> {
        self.inner.parameter_adjoints.to_vec()
    }
    #[getter]
    fn direct_adjoints(&self) -> Vec<f64> {
        self.inner.direct_adjoints.to_vec()
    }
    #[getter]
    fn calibration_adjoints(&self) -> Vec<f64> {
        self.inner.calibration_adjoints.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.standard_errors.as_ref().map(|v| v.to_vec())
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        if self
            .inner
            .parameter_names
            .last()
            .is_some_and(|name| name == "hurst")
        {
            "mixed_bergomi_eta_rho_hurst_fixed_relative_local_variance_target"
        } else {
            "mixed_bergomi_eta_rho_fixed_relative_local_variance_target"
        }
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}
