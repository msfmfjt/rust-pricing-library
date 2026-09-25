use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::{Bergomi1Factor, Bergomi2Factor, RoughBergomi};
use pricing::risk::{GammaConfig, SpotBump};
use pricing::stochastic_dividends::{
    BuehlerDividendModel, StochasticDividendAadRisk, StochasticDividendGammaRisk,
    StochasticDividendPrice, StochasticDividendPricingPlan,
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
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversion, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bergomi_lsv(
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
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
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
            StochasticDividendPricingPlan::compile_bergomi_lsv(
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

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, mean_reversions, vol_of_vol, mixing_weight, spot_correlations, factor_correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlations, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bergomi_two_factor_lsv(
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
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
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
            StochasticDividendPricingPlan::compile_bergomi_two_factor_lsv(
                &request,
                model,
                factor,
                dividend_volatility_correlations,
                particles,
                maximum_step,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, *, hurst, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_rough_bergomi(
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
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendPricingPlan::compile_rough_bergomi(
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
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyStochasticDividendPrice> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyStochasticDividendPrice { inner })
            .map_err(pricing_exception)
    }
    /// Exactly one absolute or relative Spot bump is required. A half/base/double
    /// ladder is evaluated with common normals; all other inputs remain fixed.
    #[pyo3(signature=(*, gamma_absolute_bump=None, gamma_relative_bump=None))]
    fn evaluate_gamma(
        &self,
        py: Python<'_>,
        gamma_absolute_bump: Option<f64>,
        gamma_relative_bump: Option<f64>,
    ) -> PyResult<PyStochasticDividendGammaRisk> {
        let bump = match (gamma_absolute_bump, gamma_relative_bump) {
            (Some(h), None) => SpotBump::absolute(h),
            (None, Some(h)) => SpotBump::relative(h),
            _ => return Err(invalid(py, "specify exactly one Gamma Spot bump")),
        }
        .map_err(|e| invalid(py, e))?;
        py.detach(|| self.inner.evaluate_gamma(GammaConfig::new(bump)))
            .map(|inner| PyStochasticDividendGammaRisk { inner })
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
    fn evaluate_rough_aad(&self, py: Python<'_>) -> PyResult<PyStochasticDividendAadRisk> {
        py.detach(|| self.inner.evaluate_rough_aad())
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
    #[getter]
    fn lsv_time_nodes(&self) -> Option<Vec<f64>> {
        self.inner.lsv_time_nodes().map(<[f64]>::to_vec)
    }
    #[getter]
    fn lsv_log_moneyness_nodes(&self) -> Option<Vec<f64>> {
        self.inner
            .lsv_log_moneyness_nodes()
            .map(<[f64]>::to_vec)
    }
    #[getter]
    fn lsv_squared_leverage(&self) -> Option<Vec<f64>> {
        self.inner.lsv_squared_leverage().map(<[f64]>::to_vec)
    }
    #[getter]
    fn lsv_initial_residual_equity(&self) -> Option<f64> {
        self.inner.lsv_initial_residual_equity()
    }
}

#[pyclass(frozen, name = "StochasticDividendPrice", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendPrice {
    pub(super) inner: StochasticDividendPrice,
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

/// First-order reverse at fixed grid; method and labels identify active parameters/correlations.
#[pyclass(frozen, name = "StochasticDividendAadRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendAadRisk {
    pub(super) inner: StochasticDividendAadRisk,
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

/// Paired finite-bump Gamma; sampling error excludes bump/grid/smoothing bias.
#[pyclass(frozen, name = "StochasticDividendGammaRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendGammaRisk {
    pub(super) inner: StochasticDividendGammaRisk,
}
#[pymethods]
impl PyStochasticDividendGammaRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
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
    fn spot(&self) -> f64 {
        self.inner.spot
    }
    #[getter]
    fn spot_bumps(&self) -> Vec<f64> {
        self.inner.spot_bumps.to_vec()
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
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn delta_change_per_one_percent_spot(&self) -> f64 {
        self.inner.delta_change_per_one_percent_spot()
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "sampling_only_fixed_bump_grid_and_smoothing"
    }
}
