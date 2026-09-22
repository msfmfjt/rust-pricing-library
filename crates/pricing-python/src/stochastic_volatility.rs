use super::hull_white::{PyHullWhiteAadRisk, PyHullWhitePrice, PyRoughBergomiModel};
use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::mc::ExecutionPolicy;
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::stochastic_volatility::StochasticVolatilityPricingPlan;
use pyo3::prelude::*;

fn invalid(py: Python<'_>, e: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/stochastic_volatility",
            "invalid_stochastic_volatility_configuration",
            e.to_string(),
        ),
    )
}

/// Pure SV with deterministic market curves and escrowed discrete dividends.
#[pyclass(frozen, name = "StochasticVolatilityPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticVolatilityPlan {
    inner: StochasticVolatilityPricingPlan,
}

#[pymethods]
impl PyStochasticVolatilityPlan {
    /// vol_of_vol is nu, the coefficient of log volatility (rough eta = 2*nu).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversion, vol_of_vol, correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bergomi(
        py: Python<'_>,
        request: &PyPricingRequest,
        mean_reversion: f64,
        vol_of_vol: f64,
        correlation: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let factor = Bergomi1Factor::new(mean_reversion, vol_of_vol, correlation)
            .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticVolatilityPricingPlan::compile_bergomi(&request, factor, maximum_step, policy)
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversions, vol_of_vol, mixing_weight, spot_correlations, factor_correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bergomi_two_factor(
        py: Python<'_>,
        request: &PyPricingRequest,
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let factor = Bergomi2Factor::new(
            mean_reversions,
            vol_of_vol,
            mixing_weight,
            spot_correlations,
            factor_correlation,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticVolatilityPricingPlan::compile_bergomi_two_factor(
                &request,
                factor,
                maximum_step,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    #[staticmethod]
    #[pyo3(signature=(request, rough_model, *, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_rough_bergomi(
        py: Python<'_>,
        request: &PyPricingRequest,
        rough_model: &PyRoughBergomiModel,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let factor = rough_model.inner;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticVolatilityPricingPlan::compile_rough_bergomi(
                &request,
                factor,
                maximum_step,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyHullWhitePrice> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyHullWhitePrice { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_aad(&self, py: Python<'_>) -> PyResult<PyHullWhiteAadRisk> {
        py.detach(|| self.inner.evaluate_aad())
            .map(|inner| PyHullWhiteAadRisk { inner })
            .map_err(pricing_exception)
    }
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn random_factor_count(&self) -> usize {
        self.inner.random_factor_count()
    }
    #[getter]
    fn risky_spot(&self) -> f64 {
        self.inner.risky_spot()
    }
    #[getter]
    fn cash_dividend_model(&self) -> Option<&'static str> {
        self.inner.cash_dividend_model()
    }
}
