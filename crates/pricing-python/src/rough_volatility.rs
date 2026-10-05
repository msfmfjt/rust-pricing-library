//! Additive bindings for the six rough-volatility model families.

use super::hull_white::PyHullWhitePrice;
use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::{
    ForwardVarianceCurve, LiftedHeston, MixedRoughBergomi, QuadraticRoughHeston, Rfsv, RoughHeston,
    RoughSabr, RoughVolatilityModel, RoughVolatilityPath, RoughVolatilityPathPlan,
    RoughVolatilityPricingPlan,
};
use pyo3::prelude::*;

fn invalid(py: Python<'_>, error: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/rough_volatility",
            "invalid_rough_volatility_configuration",
            error.to_string(),
        ),
    )
}

#[pyclass(frozen, name = "ForwardVarianceCurve", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyForwardVarianceCurve {
    inner: ForwardVarianceCurve,
}
#[pymethods]
impl PyForwardVarianceCurve {
    #[staticmethod]
    fn constant(py: Python<'_>, variance: f64) -> PyResult<Self> {
        ForwardVarianceCurve::constant(variance)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    fn piecewise_linear(py: Python<'_>, times: Vec<f64>, values: Vec<f64>) -> PyResult<Self> {
        ForwardVarianceCurve::piecewise_linear(times, values)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    fn exponential(py: Python<'_>, initial: f64, growth: f64) -> PyResult<Self> {
        ForwardVarianceCurve::exponential(initial, growth)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    fn value(&self, py: Python<'_>, time: f64) -> PyResult<f64> {
        self.inner.value(time).map_err(|e| invalid(py, e))
    }
}

#[pyclass(frozen, name = "RoughVolatilityModel", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughVolatilityModel {
    pub(super) inner: RoughVolatilityModel,
}
#[pymethods]
impl PyRoughVolatilityModel {
    #[staticmethod]
    #[pyo3(signature=(*, hurst, initial_variance, mean_reversion, long_run_variance, vol_of_vol, correlation))]
    fn rough_heston(
        py: Python<'_>,
        hurst: f64,
        initial_variance: f64,
        mean_reversion: f64,
        long_run_variance: f64,
        vol_of_vol: f64,
        correlation: f64,
    ) -> PyResult<Self> {
        RoughHeston::new(
            hurst,
            initial_variance,
            mean_reversion,
            long_run_variance,
            vol_of_vol,
            correlation,
        )
        .map(|m| Self { inner: m.into() })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*, initial_variance, mean_reversion, long_run_variance, vol_of_vol, correlation, weights, rates))]
    fn lifted_heston(
        py: Python<'_>,
        initial_variance: f64,
        mean_reversion: f64,
        long_run_variance: f64,
        vol_of_vol: f64,
        correlation: f64,
        weights: Vec<f64>,
        rates: Vec<f64>,
    ) -> PyResult<Self> {
        LiftedHeston::new(
            initial_variance,
            mean_reversion,
            long_run_variance,
            vol_of_vol,
            correlation,
            weights,
            rates,
        )
        .map(|m| Self { inner: m.into() })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[pyo3(signature=(rough_heston, *, factors=20, ratio=2.5))]
    fn lifted_heston_from_rough(
        py: Python<'_>,
        rough_heston: &Self,
        factors: usize,
        ratio: f64,
    ) -> PyResult<Self> {
        let RoughVolatilityModel::RoughHeston(model) = &rough_heston.inner else {
            return Err(invalid(
                py,
                "lifted_heston_from_rough requires a Rough Heston model",
            ));
        };
        LiftedHeston::from_rough(model, factors, ratio)
            .map(|m| Self { inner: m.into() })
            .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*, hurst, initial_state, mean_reversion, vol_of_vol, quadratic, shift, variance_floor))]
    fn quadratic_rough_heston(
        py: Python<'_>,
        hurst: f64,
        initial_state: f64,
        mean_reversion: f64,
        vol_of_vol: f64,
        quadratic: f64,
        shift: f64,
        variance_floor: f64,
    ) -> PyResult<Self> {
        QuadraticRoughHeston::new(
            hurst,
            initial_state,
            mean_reversion,
            vol_of_vol,
            quadratic,
            shift,
            variance_floor,
        )
        .map(|m| Self { inner: m.into() })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[pyo3(signature=(*, hurst, correlation, weights, vol_of_vols, forward_variance))]
    fn mixed_rough_bergomi(
        py: Python<'_>,
        hurst: f64,
        correlation: f64,
        weights: Vec<f64>,
        vol_of_vols: Vec<f64>,
        forward_variance: &PyForwardVarianceCurve,
    ) -> PyResult<Self> {
        MixedRoughBergomi::new(
            hurst,
            correlation,
            weights,
            vol_of_vols,
            forward_variance.inner.clone(),
        )
        .map(|m| Self { inner: m.into() })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[pyo3(signature=(*, hurst, vol_of_vol, correlation, beta, forward_variance))]
    fn rough_sabr(
        py: Python<'_>,
        hurst: f64,
        vol_of_vol: f64,
        correlation: f64,
        beta: f64,
        forward_variance: &PyForwardVarianceCurve,
    ) -> PyResult<Self> {
        RoughSabr::new(
            hurst,
            vol_of_vol,
            correlation,
            beta,
            forward_variance.inner.clone(),
        )
        .map(|m| Self { inner: m.into() })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[pyo3(signature=(*, hurst, mean_reversion, vol_of_log_vol, mean_log_vol, initial_log_vol=None))]
    fn rfsv(
        py: Python<'_>,
        hurst: f64,
        mean_reversion: f64,
        vol_of_log_vol: f64,
        mean_log_vol: f64,
        initial_log_vol: Option<f64>,
    ) -> PyResult<Self> {
        Rfsv::new(
            hurst,
            mean_reversion,
            vol_of_log_vol,
            mean_log_vol,
            initial_log_vol,
        )
        .map(|m| Self { inner: m.into() })
        .map_err(|e| invalid(py, e))
    }
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

#[pyclass(frozen, name = "RoughVolatilityPath", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughVolatilityPath {
    inner: RoughVolatilityPath,
}
#[pymethods]
impl PyRoughVolatilityPath {
    #[getter]
    fn forwards(&self) -> Vec<f64> {
        self.inner.forwards.clone()
    }
    #[getter]
    fn variances(&self) -> Vec<f64> {
        self.inner.variances.clone()
    }
    #[getter]
    fn latent_states(&self) -> Vec<f64> {
        self.inner.latent_states.clone()
    }
    #[getter]
    fn negative_variance_nodes(&self) -> u64 {
        self.inner.negative_variance_nodes
    }
    #[getter]
    fn absorbed_forward_steps(&self) -> u64 {
        self.inner.absorbed_forward_steps
    }
}

#[pyclass(frozen, name = "RoughVolatilityPathPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughVolatilityPathPlan {
    inner: RoughVolatilityPathPlan,
}
#[pymethods]
impl PyRoughVolatilityPathPlan {
    #[staticmethod]
    fn compile(
        py: Python<'_>,
        model: &PyRoughVolatilityModel,
        time_nodes: Vec<f64>,
    ) -> PyResult<Self> {
        let model = model.inner.clone();
        py.detach(|| RoughVolatilityPathPlan::compile(model, time_nodes))
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    fn evolve_path(
        &self,
        py: Python<'_>,
        initial_forward: f64,
        normals: Vec<f64>,
    ) -> PyResult<PyRoughVolatilityPath> {
        py.detach(|| self.inner.evolve_path(initial_forward, &normals))
            .map(|inner| PyRoughVolatilityPath { inner })
            .map_err(pricing_exception)
    }
    fn reverse_initial_forward(
        &self,
        py: Python<'_>,
        initial_forward: f64,
        normals: Vec<f64>,
        state_seeds: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner
                .evolve_recorded_path(initial_forward, &normals)?
                .reverse_initial_forward(&state_seeds)
        })
        .map_err(pricing_exception)
    }
    fn pseudo_shocks(&self, py: Python<'_>, seed: u64, path: u64) -> Vec<f64> {
        py.detach(|| {
            self.inner
                .pseudo_shocks(seed, path, RandomDomain::Valuation)
        })
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn random_dimension(&self) -> u32 {
        self.inner.random_dimension()
    }
    #[getter]
    fn scheme(&self) -> &'static str {
        self.inner.scheme()
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
}

#[pyclass(frozen, name = "RoughVolatilityPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughVolatilityPlan {
    inner: RoughVolatilityPricingPlan,
}
#[pymethods]
impl PyRoughVolatilityPlan {
    fn evaluate_gamma_bump(
        &self,
        py: Python<'_>,
        spot_bump: f64,
    ) -> PyResult<PyRoughVolatilityGamma> {
        py.detach(|| self.inner.evaluate_gamma_bump(spot_bump))
            .map(|inner| PyRoughVolatilityGamma { inner })
            .map_err(pricing_exception)
    }

    #[staticmethod]
    #[pyo3(signature=(request, model, *, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        request: &PyPricingRequest,
        model: &PyRoughVolatilityModel,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        let model = model.inner.clone();
        py.detach(|| RoughVolatilityPricingPlan::compile(&request, model, maximum_step, policy))
            .map(|inner| Self { inner })
            .map_err(pricing_exception)
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyHullWhitePrice> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyHullWhitePrice { inner })
            .map_err(pricing_exception)
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn random_dimension(&self) -> u32 {
        self.inner.random_dimension()
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
    fn evaluate_hurst_risk(&self, py: Python<'_>) -> PyResult<PyRoughHestonMcHurstRisk> {
        py.detach(|| self.inner.evaluate_hurst_risk())
            .map(|inner| PyRoughHestonMcHurstRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_mixed_bergomi_parameter_risk(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyMixedBergomiMcParameterRisk> {
        py.detach(|| self.inner.evaluate_mixed_bergomi_parameter_risk())
            .map(|inner| PyMixedBergomiMcParameterRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_mixed_bergomi_parameter_risk_with_hurst(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyMixedBergomiMcParameterRisk> {
        py.detach(|| {
            self.inner
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
        })
        .map(|inner| PyMixedBergomiMcParameterRisk { inner })
        .map_err(pricing_exception)
    }
    fn evaluate_heston_parameter_risk(&self, py: Python<'_>) -> PyResult<PyHestonMcParameterRisk> {
        py.detach(|| self.inner.evaluate_heston_parameter_risk())
            .map(|inner| PyHestonMcParameterRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_delta(&self, py: Python<'_>) -> PyResult<PyRoughVolatilityDelta> {
        py.detach(|| self.inner.evaluate_delta())
            .map(|inner| PyRoughVolatilityDelta { inner })
            .map_err(pricing_exception)
    }
    #[getter]
    fn risky_spot(&self) -> f64 {
        self.inner.risky_spot()
    }
    #[getter]
    fn path_plan(&self) -> PyRoughVolatilityPathPlan {
        PyRoughVolatilityPathPlan {
            inner: self.inner.path_plan().clone(),
        }
    }
}

#[pyclass(frozen, name = "RoughVolatilityDelta", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughVolatilityDelta {
    inner: pricing::rough_volatility::RoughVolatilityDelta,
}
#[pymethods]
impl PyRoughVolatilityDelta {
    #[getter]
    fn price(&self) -> PyHullWhitePrice {
        PyHullWhitePrice {
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
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "physical_spot_fixed_model_fixed_cash_dividends"
    }
}

#[pyclass(frozen, name = "RoughVolatilityGamma", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughVolatilityGamma {
    inner: pricing::rough_volatility::RoughVolatilityGamma,
}
#[pymethods]
impl PyRoughVolatilityGamma {
    #[getter]
    fn price(&self) -> PyHullWhitePrice {
        PyHullWhitePrice {
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

#[pyclass(frozen, name = "HestonMcParameterRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonMcParameterRisk {
    inner: pricing::rough_volatility::HestonMcParameterRisk,
}
#[pymethods]
impl PyHestonMcParameterRisk {
    #[getter]
    fn price(&self) -> PyHullWhitePrice {
        PyHullWhitePrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn parameter_names(&self) -> Vec<&'static str> {
        pricing::rough_volatility::HESTON_MC_PARAMETER_NAMES.to_vec()
    }
    #[getter]
    fn parameter_adjoints(&self) -> Vec<f64> {
        self.inner.parameter_adjoints.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Vec<f64> {
        self.inner.standard_errors.to_vec()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "fixed_kernel_heston_scalar_parameters"
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}

#[pyclass(frozen, name = "RoughHestonMcHurstRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRoughHestonMcHurstRisk {
    inner: pricing::rough_volatility::RoughHestonMcHurstRisk,
}
#[pymethods]
impl PyRoughHestonMcHurstRisk {
    #[getter]
    fn price(&self) -> PyHullWhitePrice {
        PyHullWhitePrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn hurst_sensitivity(&self) -> f64 {
        self.inner.hurst_sensitivity
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "rough_heston_hurst_fixed_scalar_parameters"
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}

#[pyclass(frozen, name = "MixedBergomiMcParameterRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMixedBergomiMcParameterRisk {
    inner: pricing::rough_volatility::MixedBergomiMcParameterRisk,
}
#[pymethods]
impl PyMixedBergomiMcParameterRisk {
    #[getter]
    fn price(&self) -> PyHullWhitePrice {
        PyHullWhitePrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.to_vec()
    }
    #[getter]
    fn parameter_adjoints(&self) -> Vec<f64> {
        self.inner.parameter_adjoints.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Vec<f64> {
        self.inner.standard_errors.to_vec()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        if self
            .inner
            .parameter_names
            .last()
            .is_some_and(|name| name == "hurst")
        {
            "mixed_bergomi_eta_rho_hurst_fixed_weights_and_xi"
        } else {
            "mixed_bergomi_eta_rho_fixed_kernel_weights_and_xi"
        }
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}
