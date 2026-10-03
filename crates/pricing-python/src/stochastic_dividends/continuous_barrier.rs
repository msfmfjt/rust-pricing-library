use super::*;
use pricing::stochastic_dividends::StochasticDividendContinuousBarrierPlan;

/// Price-only rough-LSV continuous Barrier approximation. Uses left-frozen
/// physical log-Spot variance, stochastic reserve and separate cash jumps.
/// Sampling errors exclude calibration uncertainty and time-grid bias.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierPlan",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierPlan {
    inner: StochasticDividendContinuousBarrierPlan,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierPlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, hurst, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_rough_bergomi_lsv(
        py: Python<'_>,
        request: &PyPricingRequest,
        hurst: f64,
        vol_of_vol: f64,
        correlation: f64,
        dividend_mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
        dividend_volatility_correlation: f64,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let factor =
            RoughBergomi::new(hurst, vol_of_vol, correlation).map_err(|e| invalid(py, e))?;
        let model = BuehlerDividendModel::new(
            dividend_mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        )
        .map_err(|e| invalid(py, e))?;
        let particles = LsvParticleConfig::new(
            particle_count,
            calibration_seed,
            log_bandwidth,
            minimum_effective_samples,
            false,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendContinuousBarrierPlan::compile_rough_bergomi_lsv(
                &request,
                model,
                factor,
                dividend_volatility_correlation,
                particles,
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
    #[getter]
    fn lsv_time_nodes(&self) -> Vec<f64> {
        self.inner.lsv_time_nodes().to_vec()
    }
    #[getter]
    fn lsv_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.lsv_log_moneyness_nodes().to_vec()
    }
    #[getter]
    fn lsv_squared_leverage(&self) -> Vec<f64> {
        self.inner.lsv_squared_leverage().to_vec()
    }
    #[getter]
    fn lsv_initial_residual_equity(&self) -> f64 {
        self.inner.lsv_initial_residual_equity()
    }
}
