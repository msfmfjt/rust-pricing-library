use super::{
    PyPricingRequest, PyValidationIssue, PyVegaKtResult, pricing_exception, validation_exception,
};
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::{Bergomi1Factor, Bergomi2Factor, RoughBergomi};
use pricing::risk::{GammaConfig, SpotBump};
use pricing::stochastic_dividends::{
    BuehlerDividendModel, StochasticDividendAadRisk, StochasticDividendGammaRisk,
    StochasticDividendLocalVarianceRisk, StochasticDividendLsvBergomi2FactorCorrelationRisk,
    StochasticDividendLsvBergomi2FactorRisk, StochasticDividendLsvBergomiRisk,
    StochasticDividendLsvCorrelationRisk, StochasticDividendLsvMarketRisk,
    StochasticDividendLsvSpotRisk, StochasticDividendPrice, StochasticDividendPricingPlan,
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
    #[pyo3(signature=(request, *, mean_reversion, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None, retain_reverse_trace=false))]
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
        retain_reverse_trace: bool,
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
            retain_reverse_trace,
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
    #[pyo3(signature=(request, *, mean_reversions, vol_of_vol, mixing_weight, spot_correlations, factor_correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlations, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None, retain_reverse_trace=false))]
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
        retain_reverse_trace: bool,
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
            retain_reverse_trace,
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
    #[pyo3(signature=(request, *, hurst, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None, retain_reverse_trace=false))]
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
        retain_reverse_trace: bool,
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
            retain_reverse_trace,
        )
        .map_err(|e| invalid(py, e))?;
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        py.detach(|| {
            StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
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
    fn evaluate_local_variance_risk(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyStochasticDividendLocalVarianceRisk> {
        py.detach(|| self.inner.evaluate_local_variance_risk())
            .map(|inner| PyStochasticDividendLocalVarianceRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_lsv_spot_risk(&self, py: Python<'_>) -> PyResult<PyStochasticDividendLsvSpotRisk> {
        py.detach(|| self.inner.evaluate_lsv_spot_risk())
            .map(|inner| PyStochasticDividendLsvSpotRisk { inner })
            .map_err(pricing_exception)
    }
    fn evaluate_lsv_market_risk(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyStochasticDividendLsvMarketRisk> {
        py.detach(|| self.inner.evaluate_lsv_market_risk())
            .map(|inner| PyStochasticDividendLsvMarketRisk { inner })
            .map_err(pricing_exception)
    }
    #[pyo3(signature=(*, dividend_mean_reversion_bump, equity_linkage_bump, dividend_volatility_bump, equity_dividend_correlation_bump))]
    fn evaluate_lsv_dividend_model_risk(
        &self,
        py: Python<'_>,
        dividend_mean_reversion_bump: f64,
        equity_linkage_bump: f64,
        dividend_volatility_bump: f64,
        equity_dividend_correlation_bump: f64,
    ) -> PyResult<PyStochasticDividendLsvDividendModelRisk> {
        py.detach(|| {
            self.inner.evaluate_lsv_dividend_model_risk(
                dividend_mean_reversion_bump,
                equity_linkage_bump,
                dividend_volatility_bump,
                equity_dividend_correlation_bump,
            )
        })
        .map(|inner| PyStochasticDividendLsvDividendModelRisk { inner })
        .map_err(pricing_exception)
    }
    #[pyo3(signature=(*, mean_reversion_bump, vol_of_vol_bump))]
    fn evaluate_lsv_bergomi_parameter_risk(
        &self,
        py: Python<'_>,
        mean_reversion_bump: f64,
        vol_of_vol_bump: f64,
    ) -> PyResult<PyStochasticDividendLsvBergomiRisk> {
        py.detach(|| {
            self.inner
                .evaluate_lsv_bergomi_parameter_risk(mean_reversion_bump, vol_of_vol_bump)
        })
        .map(|inner| PyStochasticDividendLsvBergomiRisk { inner })
        .map_err(pricing_exception)
    }
    #[pyo3(signature=(*, equity_volatility_correlation_bump, dividend_volatility_correlation_bump))]
    fn evaluate_lsv_correlation_risk(
        &self,
        py: Python<'_>,
        equity_volatility_correlation_bump: f64,
        dividend_volatility_correlation_bump: f64,
    ) -> PyResult<PyStochasticDividendLsvCorrelationRisk> {
        py.detach(|| {
            self.inner.evaluate_lsv_correlation_risk(
                equity_volatility_correlation_bump,
                dividend_volatility_correlation_bump,
            )
        })
        .map(|inner| PyStochasticDividendLsvCorrelationRisk { inner })
        .map_err(pricing_exception)
    }
    #[pyo3(signature=(*, mean_reversion_bumps, vol_of_vol_bump, mixing_weight_bump))]
    fn evaluate_lsv_bergomi_two_factor_parameter_risk(
        &self,
        py: Python<'_>,
        mean_reversion_bumps: Vec<f64>,
        vol_of_vol_bump: f64,
        mixing_weight_bump: f64,
    ) -> PyResult<PyStochasticDividendLsvBergomi2FactorRisk> {
        let bumps: [f64; 2] = mean_reversion_bumps
            .try_into()
            .map_err(|_| invalid(py, "mean_reversion_bumps must contain exactly two values"))?;
        py.detach(|| {
            self.inner.evaluate_lsv_bergomi_two_factor_parameter_risk(
                bumps,
                vol_of_vol_bump,
                mixing_weight_bump,
            )
        })
        .map(|inner| PyStochasticDividendLsvBergomi2FactorRisk { inner })
        .map_err(pricing_exception)
    }
    #[pyo3(signature=(*, spot_volatility_correlation_bumps, factor_correlation_bump, dividend_volatility_correlation_bumps))]
    fn evaluate_lsv_bergomi_two_factor_correlation_risk(
        &self,
        py: Python<'_>,
        spot_volatility_correlation_bumps: Vec<f64>,
        factor_correlation_bump: f64,
        dividend_volatility_correlation_bumps: Vec<f64>,
    ) -> PyResult<PyStochasticDividendLsvBergomi2FactorCorrelationRisk> {
        let spot_bumps: [f64; 2] = spot_volatility_correlation_bumps.try_into().map_err(|_| {
            invalid(
                py,
                "spot_volatility_correlation_bumps must contain exactly two values",
            )
        })?;
        let dividend_bumps: [f64; 2] =
            dividend_volatility_correlation_bumps
                .try_into()
                .map_err(|_| {
                    invalid(
                        py,
                        "dividend_volatility_correlation_bumps must contain exactly two values",
                    )
                })?;
        py.detach(|| {
            self.inner.evaluate_lsv_bergomi_two_factor_correlation_risk(
                spot_bumps,
                factor_correlation_bump,
                dividend_bumps,
            )
        })
        .map(|inner| PyStochasticDividendLsvBergomi2FactorCorrelationRisk { inner })
        .map_err(pricing_exception)
    }
    fn evaluate_vega_kt(&self, py: Python<'_>) -> PyResult<PyVegaKtResult> {
        py.detach(|| self.inner.evaluate_vega_kt())
            .map(|inner| PyVegaKtResult { inner })
            .map_err(pricing_exception)
    }
    /// Residual-LSV Gamma re-anchors the leverage surface under every Spot bump.
    #[pyo3(signature=(*, gamma_absolute_bump=None, gamma_relative_bump=None))]
    fn evaluate_lsv_gamma(
        &self,
        py: Python<'_>,
        gamma_absolute_bump: Option<f64>,
        gamma_relative_bump: Option<f64>,
    ) -> PyResult<PyStochasticDividendGammaRisk> {
        let bump = match (gamma_absolute_bump, gamma_relative_bump) {
            (Some(h), None) => SpotBump::absolute(h),
            (None, Some(h)) => SpotBump::relative(h),
            _ => return Err(invalid(py, "specify exactly one LSV Gamma Spot bump")),
        }
        .map_err(|e| invalid(py, e))?;
        py.detach(|| self.inner.evaluate_lsv_gamma(GammaConfig::new(bump)))
            .map(|inner| PyStochasticDividendGammaRisk { inner })
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
        self.inner.lsv_log_moneyness_nodes().map(<[f64]>::to_vec)
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
    /// Sampling error only; LSV conditions on the finite particle calibration.
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        self.inner.uncertainty_scope()
    }
}

/// 2F Bergomi correlation risk with selective residual-LSV recalibration.
#[pyclass(
    frozen,
    name = "StochasticDividendLsvBergomi2FactorCorrelationRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvBergomi2FactorCorrelationRisk {
    pub(super) inner: StochasticDividendLsvBergomi2FactorCorrelationRisk,
}
#[pymethods]
impl PyStochasticDividendLsvBergomi2FactorCorrelationRisk {
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
    fn correlation_bumps(&self) -> Vec<f64> {
        self.inner.correlation_bumps.to_vec()
    }
    #[getter]
    fn spot_volatility_correlation_derivatives(&self) -> Vec<f64> {
        self.inner
            .spot_volatility_correlation_derivatives()
            .to_vec()
    }
    #[getter]
    fn factor_correlation_derivative(&self) -> f64 {
        self.inner.factor_correlation_derivative()
    }
    #[getter]
    fn dividend_volatility_correlation_derivatives(&self) -> Vec<f64> {
        self.inner
            .dividend_volatility_correlation_derivatives()
            .to_vec()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "two_factor_bergomi_correlations_with_selective_residual_lsv_recalibration"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_fixed_calibration_seed_and_correlation_bumps"
    }
}

/// Full-recalibration 2F Bergomi parameter risk for residual-equity LSV.
#[pyclass(
    frozen,
    name = "StochasticDividendLsvBergomi2FactorRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvBergomi2FactorRisk {
    pub(super) inner: StochasticDividendLsvBergomi2FactorRisk,
}
#[pymethods]
impl PyStochasticDividendLsvBergomi2FactorRisk {
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
    fn parameter_bumps(&self) -> Vec<f64> {
        self.inner.parameter_bumps.to_vec()
    }
    #[getter]
    fn mean_reversion_derivatives(&self) -> Vec<f64> {
        self.inner.mean_reversion_derivatives().to_vec()
    }
    #[getter]
    fn vol_of_vol_derivative(&self) -> f64 {
        self.inner.vol_of_vol_derivative()
    }
    #[getter]
    fn mixing_weight_derivative(&self) -> f64 {
        self.inner.mixing_weight_derivative()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "two_factor_bergomi_parameters_with_full_residual_lsv_recalibration"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_fixed_calibration_seed_and_parameter_bumps"
    }
}

/// 1F Bergomi correlation risk with selective residual-LSV recalibration.
#[pyclass(
    frozen,
    name = "StochasticDividendLsvCorrelationRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvCorrelationRisk {
    pub(super) inner: StochasticDividendLsvCorrelationRisk,
}
#[pymethods]
impl PyStochasticDividendLsvCorrelationRisk {
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
    fn correlation_bumps(&self) -> Vec<f64> {
        self.inner.correlation_bumps.to_vec()
    }
    #[getter]
    fn equity_volatility_correlation_derivative(&self) -> f64 {
        self.inner.equity_volatility_correlation_derivative()
    }
    #[getter]
    fn dividend_volatility_correlation_derivative(&self) -> f64 {
        self.inner.dividend_volatility_correlation_derivative()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "one_factor_bergomi_correlations_with_selective_residual_lsv_recalibration"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_fixed_calibration_seed_and_correlation_bumps"
    }
}

/// Full-recalibration 1F Bergomi parameter risk for residual-equity LSV.
#[pyclass(frozen, name = "StochasticDividendLsvBergomiRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvBergomiRisk {
    pub(super) inner: StochasticDividendLsvBergomiRisk,
}
#[pymethods]
impl PyStochasticDividendLsvBergomiRisk {
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
    fn parameter_bumps(&self) -> Vec<f64> {
        self.inner.parameter_bumps.to_vec()
    }
    #[getter]
    fn mean_reversion_derivative(&self) -> f64 {
        self.inner.mean_reversion_derivative()
    }
    #[getter]
    fn vol_of_vol_derivative(&self) -> f64 {
        self.inner.vol_of_vol_derivative()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "one_factor_bergomi_parameters_with_full_residual_lsv_recalibration"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_fixed_calibration_seed_and_parameter_bumps"
    }
}

/// Buehler dividend-model parameter risk with fixed residual-LSV calibration.
#[pyclass(frozen, name = "StochasticDividendLsvDividendModelRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvDividendModelRisk {
    pub(super) inner: StochasticDividendLsvDividendModelRisk,
}
#[pymethods]
impl PyStochasticDividendLsvDividendModelRisk {
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
    fn parameter_bumps(&self) -> Vec<f64> {
        self.inner.parameter_bumps.to_vec()
    }
    #[getter]
    fn dividend_mean_reversion_derivative(&self) -> f64 {
        self.inner.dividend_mean_reversion_derivative()
    }
    #[getter]
    fn equity_linkage_derivative(&self) -> f64 {
        self.inner.equity_linkage_derivative()
    }
    #[getter]
    fn dividend_volatility_derivative(&self) -> f64 {
        self.inner.dividend_volatility_derivative()
    }
    #[getter]
    fn equity_dividend_correlation_derivative(&self) -> f64 {
        self.inner.equity_dividend_correlation_derivative()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "buehler_dividend_model_parameters_with_fixed_residual_lsv_calibration"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_fixed_calibration_and_parameter_bumps"
    }
}

/// Spot/cash/curve risk with scale-invariant residual-LSV re-anchoring.
#[pyclass(frozen, name = "StochasticDividendLsvMarketRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvMarketRisk {
    pub(super) inner: StochasticDividendLsvMarketRisk,
}
#[pymethods]
impl PyStochasticDividendLsvMarketRisk {
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
    fn cash_times(&self) -> Vec<f64> {
        self.inner.cash_times.to_vec()
    }
    #[getter]
    fn cash_mean_adjoints(&self) -> Vec<f64> {
        self.inner.cash_mean_adjoints.to_vec()
    }
    #[getter]
    fn cash_mean_standard_errors(&self) -> Vec<f64> {
        self.inner.cash_mean_standard_errors.to_vec()
    }
    #[getter]
    fn discount_times(&self) -> Vec<f64> {
        self.inner.discount_times.to_vec()
    }
    #[getter]
    fn discount_log_df_adjoints(&self) -> Vec<f64> {
        self.inner.discount_log_df_adjoints.to_vec()
    }
    #[getter]
    fn discount_log_df_standard_errors(&self) -> Vec<f64> {
        self.inner.discount_log_df_standard_errors.to_vec()
    }
    #[getter]
    fn discount_node_dv01(&self) -> Vec<f64> {
        self.inner.discount_node_dv01()
    }
    #[getter]
    fn repo_spread_times(&self) -> Vec<f64> {
        self.inner.repo_spread_times.to_vec()
    }
    #[getter]
    fn repo_spread_log_df_adjoints(&self) -> Vec<f64> {
        self.inner.repo_spread_log_df_adjoints.to_vec()
    }
    #[getter]
    fn repo_spread_log_df_standard_errors(&self) -> Vec<f64> {
        self.inner.repo_spread_log_df_standard_errors.to_vec()
    }
    #[getter]
    fn repo_spread_node_dv01(&self) -> Vec<f64> {
        self.inner.repo_spread_node_dv01()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "spot_cash_and_log_df_curves_with_residual_lsv_reanchoring"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_scale_invariant_calibration"
    }
}

/// Physical-Spot Delta with the residual-equity LSV surface re-anchored.
#[pyclass(frozen, name = "StochasticDividendLsvSpotRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLsvSpotRisk {
    pub(super) inner: StochasticDividendLsvSpotRisk,
}
#[pymethods]
impl PyStochasticDividendLsvSpotRisk {
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
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn coordinate(&self) -> &'static str {
        "physical_spot_with_residual_lsv_reanchoring"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_scale_invariant_calibration"
    }
}

/// Recalibration-aware risk to residual-equity Dupire Local-variance nodes.
#[pyclass(
    frozen,
    name = "StochasticDividendLocalVarianceRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendLocalVarianceRisk {
    pub(super) inner: StochasticDividendLocalVarianceRisk,
}
#[pymethods]
impl PyStochasticDividendLocalVarianceRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
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
        "relative_dupire_variance_nodes_in_residual_equity"
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        "pricing_sampling_only_fixed_calibration_seed_and_particle_count"
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
