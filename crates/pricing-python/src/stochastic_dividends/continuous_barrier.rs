use super::*;
use pricing::core::PositiveF64;
use pricing::stochastic_dividends::{
    StochasticDividendContinuousBarrierGammaRisk, StochasticDividendContinuousBarrierPlan,
    StochasticDividendContinuousBarrierSpotRisk,
};

/// Rough-LSV continuous Barrier approximation. Uses left-frozen
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
    /// Finite central price differences on a half/base/double Spot-bump ladder.
    #[pyo3(signature=(*, spot_absolute_bump=None, spot_relative_bump=None))]
    fn evaluate_spot_bump_risk(
        &self,
        py: Python<'_>,
        spot_absolute_bump: Option<f64>,
        spot_relative_bump: Option<f64>,
    ) -> PyResult<PyStochasticDividendContinuousBarrierSpotRisk> {
        let bump = spot_bump(py, spot_absolute_bump, spot_relative_bump)?;
        py.detach(|| self.inner.evaluate_spot_bump_risk(bump))
            .map(|inner| PyStochasticDividendContinuousBarrierSpotRisk { inner })
            .map_err(pricing_exception)
    }
    /// Paired second price differences, with half/base/double Gamma diagnostics.
    #[pyo3(signature=(*, spot_absolute_bump=None, spot_relative_bump=None))]
    fn evaluate_gamma_bump_risk(
        &self,
        py: Python<'_>,
        spot_absolute_bump: Option<f64>,
        spot_relative_bump: Option<f64>,
    ) -> PyResult<PyStochasticDividendContinuousBarrierGammaRisk> {
        let bump = spot_bump(py, spot_absolute_bump, spot_relative_bump)?;
        py.detach(|| self.inner.evaluate_gamma_bump_risk(bump))
            .map(|inner| PyStochasticDividendContinuousBarrierGammaRisk { inner })
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

/// Finite-bump Spot risk of the bridge approximation, with paired sampling errors.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierSpotRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierSpotRisk {
    inner: StochasticDividendContinuousBarrierSpotRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierSpotRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }
    #[getter]
    fn spot_bumps(&self) -> Vec<f64> {
        self.inner.spot_bumps.to_vec()
    }
    #[getter]
    fn delta_estimates(&self) -> Vec<f64> {
        self.inner.delta_estimates.to_vec()
    }
    #[getter]
    fn delta_standard_errors(&self) -> Vec<f64> {
        self.inner.delta_standard_errors.to_vec()
    }
    #[getter]
    fn bump_differences(&self) -> Vec<f64> {
        self.inner.bump_differences.to_vec()
    }
    #[getter]
    fn bump_difference_standard_errors(&self) -> Vec<f64> {
        self.inner.bump_difference_standard_errors.to_vec()
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error()
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        self.inner.uncertainty_scope()
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}

fn spot_bump(
    py: Python<'_>,
    spot_absolute_bump: Option<f64>,
    spot_relative_bump: Option<f64>,
) -> PyResult<SpotBump> {
    match (spot_absolute_bump, spot_relative_bump) {
        (Some(h), None) => PositiveF64::new(h, "spot_absolute_bump").map(SpotBump::Absolute),
        (None, Some(h)) => PositiveF64::new(h, "spot_relative_bump").map(SpotBump::Relative),
        _ => {
            return Err(invalid(
                py,
                "specify exactly one continuous Barrier Spot bump",
            ));
        }
    }
    .map_err(|e| invalid(py, e))
}

/// Finite-bump Gamma of the bridge approximation, with paired sampling errors.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierGammaRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierGammaRisk {
    inner: StochasticDividendContinuousBarrierGammaRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierGammaRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }
    #[getter]
    fn spot_bumps(&self) -> Vec<f64> {
        self.inner.spot_bumps.to_vec()
    }
    #[getter]
    fn gamma_estimates(&self) -> Vec<f64> {
        self.inner.gamma_estimates.to_vec()
    }
    #[getter]
    fn gamma_standard_errors(&self) -> Vec<f64> {
        self.inner.gamma_standard_errors.to_vec()
    }
    #[getter]
    fn bump_differences(&self) -> Vec<f64> {
        self.inner.bump_differences.to_vec()
    }
    #[getter]
    fn bump_difference_standard_errors(&self) -> Vec<f64> {
        self.inner.bump_difference_standard_errors.to_vec()
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
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
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error()
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        self.inner.uncertainty_scope()
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
}
