//! Dedicated HW API; it never dispatches deterministic-rate AAD.
use super::stochastic_dividends::{PyStochasticDividendAadRisk, PyStochasticDividendPrice};
use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::mc::ExecutionPolicy;
use pricing::models::{BuehlerDividendModel, HullWhite1Factor};
use pricing::stochastic_dividends::StochasticDividendHullWhitePricingPlan;
use pyo3::prelude::*;

fn invalid(py: Python<'_>, e: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/stochastic_dividend_hull_white",
            "invalid_stochastic_dividend_hw_configuration",
            e.to_string(),
        ),
    )
}
#[pyclass(frozen, name = "StochasticDividendHullWhitePlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendHullWhitePlan {
    inner: StochasticDividendHullWhitePricingPlan,
}
#[pymethods]
impl PyStochasticDividendHullWhitePlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, rate_mean_reversion, rate_volatility_times, rate_volatilities, equity_rate_correlation, dividend_rate_correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bs(
        py: Python<'_>,
        request: &PyPricingRequest,
        dividend_mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
        rate_mean_reversion: f64,
        rate_volatility_times: Vec<f64>,
        rate_volatilities: Vec<f64>,
        equity_rate_correlation: f64,
        dividend_rate_correlation: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let model = BuehlerDividendModel::new(
            dividend_mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        )
        .map_err(|e| invalid(py, e))?;
        let rates = HullWhite1Factor::new(
            rate_mean_reversion,
            rate_volatility_times,
            rate_volatilities,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendHullWhitePricingPlan::compile_bs(
                &request,
                model,
                rates,
                equity_rate_correlation,
                dividend_rate_correlation,
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
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
    #[getter]
    fn scheme(&self) -> &'static str {
        self.inner.scheme()
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
    fn cash_times(&self) -> Vec<f64> {
        self.inner.cash_times().to_vec()
    }
    #[getter]
    fn initial_dividend_claim_values(&self) -> Vec<f64> {
        self.inner.initial_dividend_claim_values().to_vec()
    }
    #[getter]
    fn initial_dividend_forwards(&self) -> Vec<f64> {
        self.inner.initial_dividend_forwards().to_vec()
    }
}
