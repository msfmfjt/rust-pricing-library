use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::mc::ExecutionPolicy;
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::stochastic_dividends::{
    BuehlerDividendModel, StochasticDividendAadRisk, StochasticDividendPrice,
    StochasticDividendPricingPlan,
};
use pyo3::prelude::*;

fn invalid(py: Python<'_>, e: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/stochastic_dividends",
            "invalid_stochastic_dividend_configuration",
            e.to_string(),
        ),
    )
}

/// Buehler discrete cash dividends; deterministic rates and price-only requests.
#[pyclass(frozen, name = "StochasticDividendPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendPlan {
    inner: StochasticDividendPricingPlan,
}

#[pymethods]
impl PyStochasticDividendPlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bs(
        py: Python<'_>,
        request: &PyPricingRequest,
        mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let model = BuehlerDividendModel::new(
            mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendPricingPlan::compile_bs(&request, model, maximum_step, policy)
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversion, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bergomi(
        py: Python<'_>,
        request: &PyPricingRequest,
        mean_reversion: f64,
        vol_of_vol: f64,
        correlation: f64,
        dividend_mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
        dividend_volatility_correlation: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let factor = Bergomi1Factor::new(mean_reversion, vol_of_vol, correlation)
            .map_err(|e| invalid(py, e))?;
        let model = BuehlerDividendModel::new(
            dividend_mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendPricingPlan::compile_bergomi(
                &request,
                model,
                factor,
                dividend_volatility_correlation,
                maximum_step,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversions, vol_of_vol, mixing_weight, spot_correlations, factor_correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlations, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bergomi_two_factor(
        py: Python<'_>,
        request: &PyPricingRequest,
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
        dividend_mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
        dividend_volatility_correlations: [f64; 2],
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
        let model = BuehlerDividendModel::new(
            dividend_mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendPricingPlan::compile_bergomi_two_factor(
                &request,
                model,
                factor,
                dividend_volatility_correlations,
                maximum_step,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyStochasticDividendPrice> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyStochasticDividendPrice { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_aad(&self, py: Python<'_>) -> PyResult<PyStochasticDividendAadRisk> {
        py.detach(|| self.inner.evaluate_aad())
            .map(|inner| PyStochasticDividendAadRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_correlation_aad(&self, py: Python<'_>) -> PyResult<PyStochasticDividendAadRisk> {
        py.detach(|| self.inner.evaluate_correlation_aad())
            .map(|inner| PyStochasticDividendAadRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_bergomi_aad(&self, py: Python<'_>) -> PyResult<PyStochasticDividendAadRisk> {
        py.detach(|| self.inner.evaluate_bergomi_aad())
            .map(|inner| PyStochasticDividendAadRisk { inner })
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
    fn scheme(&self) -> &'static str {
        self.inner.scheme()
    }
}

#[pyclass(frozen, name = "StochasticDividendPrice", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendPrice {
    inner: StochasticDividendPrice,
}

#[pymethods]
impl PyStochasticDividendPrice {
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
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint.to_string()
    }
    #[getter]
    fn scheme(&self) -> &'static str {
        self.inner.scheme
    }
    /// Sampling error only; does not include timestep or parameter uncertainty.
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_only"
    }
}

/// First-order reverse at fixed correlations/grid; labels identify active parameters.
#[pyclass(frozen, name = "StochasticDividendAadRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendAadRisk {
    inner: StochasticDividendAadRisk,
}
#[pymethods]
impl PyStochasticDividendAadRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn parameter_labels(&self) -> Vec<String> {
        self.inner.parameter_labels.to_vec()
    }
    #[getter]
    fn derivatives(&self) -> Vec<f64> {
        self.inner.derivatives.to_vec()
    }
    #[getter]
    fn standard_errors(&self) -> Vec<f64> {
        self.inner.standard_errors.to_vec()
    }
    #[getter]
    fn cash_times(&self) -> Vec<f64> {
        self.inner.cash_times.to_vec()
    }
    #[getter]
    fn discount_times(&self) -> Vec<f64> {
        self.inner.discount_times.to_vec()
    }
    #[getter]
    fn repo_spread_times(&self) -> Vec<f64> {
        self.inner.repo_spread_times.to_vec()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "sampling_only_fixed_parameters_and_grid"
    }
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta()
    }
    #[getter]
    fn initial_volatility_vega(&self) -> f64 {
        self.inner.initial_volatility_vega()
    }
    #[getter]
    fn initial_volatility_vega_per_vol_point(&self) -> f64 {
        self.inner.initial_volatility_vega_per_vol_point()
    }
    #[getter]
    fn dividend_volatility_vega_per_vol_point(&self) -> f64 {
        self.inner.dividend_volatility_vega_per_vol_point()
    }
    #[getter]
    fn cash_mean_adjoints(&self) -> Vec<f64> {
        self.inner.cash_mean_adjoints().to_vec()
    }
    #[getter]
    fn discount_node_dv01(&self) -> Vec<f64> {
        self.inner.discount_node_dv01()
    }
    #[getter]
    fn repo_spread_node_dv01(&self) -> Vec<f64> {
        self.inner.repo_spread_node_dv01()
    }
}
