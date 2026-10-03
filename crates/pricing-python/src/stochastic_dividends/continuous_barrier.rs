use super::*;
use crate::market_iv::PyMarketIvSurface;
use pricing::core::PositiveF64;
use pricing::stochastic_dividends::{
    StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk,
    StochasticDividendContinuousBarrierBucketedMarketIvRisk,
    StochasticDividendContinuousBarrierGammaRisk,
    StochasticDividendContinuousBarrierLocalVolatilityRisk,
    StochasticDividendContinuousBarrierMarketIvRisk, StochasticDividendContinuousBarrierPlan,
    StochasticDividendContinuousBarrierReportingIvRisk,
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
    #[pyo3(signature=(request, *, hurst, vol_of_vol, correlation, dividend_mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation, dividend_volatility_correlation, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, maximum_step, worker_threads, reduction_block_size=None, market_iv_surface=None))]
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
        market_iv_surface: Option<&PyMarketIvSurface>,
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
        let surface = market_iv_surface.map(|s| s.inner.clone());
        py.detach(|| {
            let plan = StochasticDividendContinuousBarrierPlan::compile_rough_bergomi_lsv(
                &request,
                model,
                factor,
                dividend_volatility_correlation,
                particles,
                maximum_step,
                policy,
            )?;
            match surface {
                Some(surface) => plan.with_market_iv_surface(surface),
                None => Ok(plan),
            }
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
    fn supports_market_iv_risk(&self) -> bool {
        self.inner.supports_market_iv_risk()
    }

    /// Parallel retained-quote IV bumps, with Dupire rebuild and LSV recalibration.
    #[pyo3(signature=(*, implied_volatility_bump))]
    fn evaluate_parallel_market_iv_risk(
        &self,
        py: Python<'_>,
        implied_volatility_bump: f64,
    ) -> PyResult<PyStochasticDividendContinuousBarrierMarketIvRisk> {
        py.detach(|| {
            self.inner
                .evaluate_parallel_market_iv_risk(implied_volatility_bump)
        })
        .map(|inner| PyStochasticDividendContinuousBarrierMarketIvRisk { inner })
        .map_err(pricing_exception)
    }
    /// Parallel absolute shifts of original residual Local volatility, with full recalibration.
    #[pyo3(signature=(*, local_volatility_bump))]
    fn evaluate_parallel_local_volatility_risk(
        &self,
        py: Python<'_>,
        local_volatility_bump: f64,
    ) -> PyResult<PyStochasticDividendContinuousBarrierLocalVolatilityRisk> {
        py.detach(|| {
            self.inner
                .evaluate_parallel_local_volatility_risk(local_volatility_bump)
        })
        .map(|inner| PyStochasticDividendContinuousBarrierLocalVolatilityRisk { inner })
        .map_err(pricing_exception)
    }
    /// Selected retained residual-forward IV quotes, recalibrated independently.
    #[pyo3(signature=(*, implied_volatility_bump, quote_indices))]
    fn evaluate_bucketed_market_iv_risk(
        &self,
        py: Python<'_>,
        implied_volatility_bump: f64,
        quote_indices: Vec<usize>,
    ) -> PyResult<PyStochasticDividendContinuousBarrierBucketedMarketIvRisk> {
        py.detach(|| {
            self.inner
                .evaluate_bucketed_market_iv_risk(implied_volatility_bump, &quote_indices)
        })
        .map(|inner| PyStochasticDividendContinuousBarrierBucketedMarketIvRisk { inner })
        .map_err(pricing_exception)
    }
    /// Selected original residual Local-volatility nodes, recalibrated independently.
    #[pyo3(signature=(*, local_volatility_bump, node_indices))]
    fn evaluate_bucketed_local_volatility_risk(
        &self,
        py: Python<'_>,
        local_volatility_bump: f64,
        node_indices: Vec<usize>,
    ) -> PyResult<PyStochasticDividendContinuousBarrierBucketedLocalVolatilityRisk> {
        py.detach(|| {
            self.inner
                .evaluate_bucketed_local_volatility_risk(local_volatility_bump, &node_indices)
        })
        .map(|inner| PyStochasticDividendContinuousBarrierBucketedLocalVolatilityRisk { inner })
        .map_err(pricing_exception)
    }
    /// Reporting-only map of all recalibrated Local-volatility node risks.
    #[pyo3(signature=(*, local_volatility_bump, relative_density_threshold, full_covariance=false))]
    fn evaluate_reporting_iv_projection(
        &self,
        py: Python<'_>,
        local_volatility_bump: f64,
        relative_density_threshold: f64,
        full_covariance: bool,
    ) -> PyResult<PyStochasticDividendContinuousBarrierReportingIvRisk> {
        py.detach(|| {
            if full_covariance {
                self.inner.evaluate_reporting_iv_projection_with_covariance(
                    local_volatility_bump,
                    relative_density_threshold,
                )
            } else {
                self.inner.evaluate_reporting_iv_projection(
                    local_volatility_bump,
                    relative_density_threshold,
                )
            }
        })
        .map(|inner| PyStochasticDividendContinuousBarrierReportingIvRisk { inner })
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

/// Paired parallel residual Local-volatility risk, including particle recalibration.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierLocalVolatilityRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierLocalVolatilityRisk {
    inner: StochasticDividendContinuousBarrierLocalVolatilityRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierLocalVolatilityRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn local_volatility_bumps(&self) -> Vec<f64> {
        self.inner.local_volatility_bumps.to_vec()
    }
    #[getter]
    fn vega_estimates(&self) -> Vec<f64> {
        self.inner.vega_estimates.to_vec()
    }
    #[getter]
    fn vega_standard_errors(&self) -> Vec<f64> {
        self.inner.vega_standard_errors.to_vec()
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
    fn scenario_evaluated_paths(&self) -> u128 {
        self.inner.scenario_evaluated_paths
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn recalibration_count(&self) -> usize {
        self.inner.recalibration_count
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn vega(&self) -> f64 {
        self.inner.vega()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error()
    }
    #[getter]
    fn vega_per_vol_point(&self) -> f64 {
        self.inner.vega_per_vol_point()
    }
    #[getter]
    fn standard_error_per_vol_point(&self) -> f64 {
        self.inner.standard_error_per_vol_point()
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

/// Selected original target node risks with paired sum errors.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierBucketedLocalVolatilityRisk {
    inner: StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierBucketedLocalVolatilityRisk {
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn scenario_evaluated_paths(&self) -> u128 {
        self.inner.scenario_evaluated_paths
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn recalibration_count(&self) -> usize {
        self.inner.recalibration_count
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn node_indices(&self) -> Vec<usize> {
        self.inner.node_indices.to_vec()
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
    fn local_volatility_bumps(&self) -> Vec<f64> {
        self.inner.local_volatility_bumps.to_vec()
    }
    #[getter]
    fn sum_vega_estimates(&self) -> Vec<f64> {
        self.inner.sum_vega_estimates.to_vec()
    }
    #[getter]
    fn sum_vega_standard_errors(&self) -> Vec<f64> {
        self.inner.sum_vega_standard_errors.to_vec()
    }
    #[getter]
    fn sum_bump_differences(&self) -> Vec<f64> {
        self.inner.sum_bump_differences.to_vec()
    }
    #[getter]
    fn sum_bump_difference_standard_errors(&self) -> Vec<f64> {
        self.inner.sum_bump_difference_standard_errors.to_vec()
    }
    #[getter]
    fn vega_estimates(&self) -> Vec<Vec<f64>> {
        self.inner
            .vega_estimates
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn vega_standard_errors(&self) -> Vec<Vec<f64>> {
        self.inner
            .vega_standard_errors
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bump_differences(&self) -> Vec<Vec<f64>> {
        self.inner
            .bump_differences
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bump_difference_standard_errors(&self) -> Vec<Vec<f64>> {
        self.inner
            .bump_difference_standard_errors
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        self.inner.uncertainty_scope()
    }
}

/// Reporting-only IV basis projection, with paired sampling errors.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierReportingIvRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierReportingIvRisk {
    inner: StochasticDividendContinuousBarrierReportingIvRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierReportingIvRisk {
    #[getter]
    fn estimator_covariance(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.estimator_covariance.clone()
    }
    #[getter]
    fn covariance_labels(&self) -> Vec<String> {
        self.inner.covariance_labels()
    }
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn local_volatility_bumps(&self) -> Vec<f64> {
        self.inner.local_volatility_bumps.to_vec()
    }
    #[getter]
    fn reporting_maturity_nodes(&self) -> Vec<f64> {
        self.inner.reporting_maturity_nodes.to_vec()
    }
    #[getter]
    fn reporting_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.reporting_log_moneyness_nodes.to_vec()
    }
    #[getter]
    fn reporting_implied_volatilities(&self) -> Vec<f64> {
        self.inner.reporting_implied_volatilities.to_vec()
    }
    #[getter]
    fn pre_projection_estimates(&self) -> Vec<f64> {
        self.inner.pre_projection_estimates.to_vec()
    }
    #[getter]
    fn pre_projection_standard_errors(&self) -> Vec<f64> {
        self.inner.pre_projection_standard_errors.to_vec()
    }
    #[getter]
    fn projected_sum_estimates(&self) -> Vec<f64> {
        self.inner.projected_sum_estimates.to_vec()
    }
    #[getter]
    fn projected_sum_standard_errors(&self) -> Vec<f64> {
        self.inner.projected_sum_standard_errors.to_vec()
    }
    #[getter]
    fn residual_estimates(&self) -> Vec<f64> {
        self.inner.residual_estimates.to_vec()
    }
    #[getter]
    fn residual_standard_errors(&self) -> Vec<f64> {
        self.inner.residual_standard_errors.to_vec()
    }
    #[getter]
    fn target_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.target_log_moneyness_nodes.to_vec()
    }
    #[getter]
    fn positive_target_time_nodes(&self) -> Vec<f64> {
        self.inner.positive_target_time_nodes.to_vec()
    }
    #[getter]
    fn excluded_probability_masses(&self) -> Vec<f64> {
        self.inner.excluded_probability_masses.to_vec()
    }
    #[getter]
    fn active_domain_start_indices(&self) -> Vec<usize> {
        self.inner.active_domain_start_indices.to_vec()
    }
    #[getter]
    fn active_domain_end_indices(&self) -> Vec<usize> {
        self.inner.active_domain_end_indices.to_vec()
    }
    #[getter]
    fn bucket_estimates(&self) -> Vec<Vec<f64>> {
        self.inner
            .bucket_estimates
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bucket_standard_errors(&self) -> Vec<Vec<f64>> {
        self.inner
            .bucket_standard_errors
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bump_differences(&self) -> Vec<Vec<f64>> {
        self.inner
            .bump_differences
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bump_difference_standard_errors(&self) -> Vec<Vec<f64>> {
        self.inner
            .bump_difference_standard_errors
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn relative_density_threshold(&self) -> f64 {
        self.inner.relative_density_threshold
    }
    #[getter]
    fn scenario_evaluated_paths(&self) -> u128 {
        self.inner.scenario_evaluated_paths
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn recalibration_count(&self) -> usize {
        self.inner.recalibration_count
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
    #[getter]
    fn projection_policy(&self) -> &'static str {
        self.inner.projection_policy()
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        self.inner.uncertainty_scope()
    }
}

/// Paired parallel residual-forward quote IV risk, with full Dupire/LSV rebuild.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierMarketIvRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierMarketIvRisk {
    inner: StochasticDividendContinuousBarrierMarketIvRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierMarketIvRisk {
    #[getter]
    fn quote_maturity_nodes(&self) -> Vec<f64> {
        self.inner.quote_maturity_nodes.clone()
    }
    #[getter]
    fn quote_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.quote_log_moneyness_nodes.clone()
    }
    #[getter]
    fn implied_volatilities(&self) -> Vec<f64> {
        self.inner.implied_volatilities.clone()
    }
    #[getter]
    fn interpolation(&self) -> &'static str {
        self.inner.interpolation()
    }

    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn implied_volatility_bumps(&self) -> Vec<f64> {
        self.inner.implied_volatility_bumps.to_vec()
    }
    #[getter]
    fn vega_estimates(&self) -> Vec<f64> {
        self.inner.vega_estimates.to_vec()
    }
    #[getter]
    fn vega_standard_errors(&self) -> Vec<f64> {
        self.inner.vega_standard_errors.to_vec()
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
    fn scenario_evaluated_paths(&self) -> u128 {
        self.inner.scenario_evaluated_paths
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn recalibration_count(&self) -> usize {
        self.inner.recalibration_count
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn vega(&self) -> f64 {
        self.inner.vega()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error()
    }
    #[getter]
    fn vega_per_vol_point(&self) -> f64 {
        self.inner.vega_per_vol_point()
    }
    #[getter]
    fn standard_error_per_vol_point(&self) -> f64 {
        self.inner.standard_error_per_vol_point()
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

/// Selected retained quote-IV risks with paired sum errors.
#[pyclass(
    frozen,
    name = "StochasticDividendContinuousBarrierBucketedMarketIvRisk",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendContinuousBarrierBucketedMarketIvRisk {
    inner: StochasticDividendContinuousBarrierBucketedMarketIvRisk,
}
#[pymethods]
impl PyStochasticDividendContinuousBarrierBucketedMarketIvRisk {
    #[getter]
    fn implied_volatilities(&self) -> Vec<f64> {
        self.inner.implied_volatilities.clone()
    }
    #[getter]
    fn interpolation(&self) -> &'static str {
        self.inner.interpolation()
    }
    #[getter]
    fn price(&self) -> PyStochasticDividendPrice {
        PyStochasticDividendPrice {
            inner: self.inner.price.clone(),
        }
    }
    #[getter]
    fn scenario_evaluated_paths(&self) -> u128 {
        self.inner.scenario_evaluated_paths
    }
    #[getter]
    fn payoff_evaluations(&self) -> u128 {
        self.inner.payoff_evaluations
    }
    #[getter]
    fn recalibration_count(&self) -> usize {
        self.inner.recalibration_count
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
    #[getter]
    fn quote_indices(&self) -> Vec<usize> {
        self.inner.quote_indices.to_vec()
    }
    #[getter]
    fn quote_maturity_nodes(&self) -> Vec<f64> {
        self.inner.quote_maturity_nodes.to_vec()
    }
    #[getter]
    fn quote_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.quote_log_moneyness_nodes.to_vec()
    }
    #[getter]
    fn implied_volatility_bumps(&self) -> Vec<f64> {
        self.inner.implied_volatility_bumps.to_vec()
    }
    #[getter]
    fn sum_vega_estimates(&self) -> Vec<f64> {
        self.inner.sum_vega_estimates.to_vec()
    }
    #[getter]
    fn sum_vega_standard_errors(&self) -> Vec<f64> {
        self.inner.sum_vega_standard_errors.to_vec()
    }
    #[getter]
    fn sum_bump_differences(&self) -> Vec<f64> {
        self.inner.sum_bump_differences.to_vec()
    }
    #[getter]
    fn sum_bump_difference_standard_errors(&self) -> Vec<f64> {
        self.inner.sum_bump_difference_standard_errors.to_vec()
    }
    #[getter]
    fn vega_estimates(&self) -> Vec<Vec<f64>> {
        self.inner
            .vega_estimates
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn vega_standard_errors(&self) -> Vec<Vec<f64>> {
        self.inner
            .vega_standard_errors
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bump_differences(&self) -> Vec<Vec<f64>> {
        self.inner
            .bump_differences
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn bump_difference_standard_errors(&self) -> Vec<Vec<f64>> {
        self.inner
            .bump_difference_standard_errors
            .iter()
            .map(|row| row.to_vec())
            .collect()
    }
    #[getter]
    fn risk_fingerprint(&self) -> String {
        self.inner.risk_fingerprint.to_string()
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        self.inner.uncertainty_scope()
    }
}
