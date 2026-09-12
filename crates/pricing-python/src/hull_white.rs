use super::builders::{PyDiscountCurve, PyEssviSlice, PyModel};
use super::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};
use pricing::hull_white::{HullWhiteEquityPricingPlan, HullWhitePrice};
use pricing::market::{EssviSurface, SurfaceValidationTolerance};
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::{
    Bergomi1Factor, HullWhite1Factor, HybridCorrelation, LocalVolatilitySpec, ModelSpec,
};
use pyo3::prelude::*;

fn invalid(py: Python<'_>, e: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/hull_white",
            "invalid_hull_white_configuration",
            e.to_string(),
        ),
    )
}

#[pyclass(frozen, name = "HullWhiteModel", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHullWhiteModel {
    inner: HullWhite1Factor,
}
#[pymethods]
impl PyHullWhiteModel {
    /// Constant mean reversion and left-constant rate volatility, flat in the tail.
    #[new]
    fn new(
        py: Python<'_>,
        mean_reversion: f64,
        volatility_times: Vec<f64>,
        volatilities: Vec<f64>,
    ) -> PyResult<Self> {
        HullWhite1Factor::new(mean_reversion, volatility_times, volatilities)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    #[getter]
    fn mean_reversion(&self) -> f64 {
        self.inner.mean_reversion()
    }
    #[getter]
    fn volatility_times(&self) -> Vec<f64> {
        self.inner.volatility_times().to_vec()
    }
    #[getter]
    fn volatilities(&self) -> Vec<f64> {
        self.inner.volatilities().to_vec()
    }
    /// Zero-coupon bond price conditional on the shifted OU state x(t).
    fn bond_price(
        &self,
        discount_curve: &PyDiscountCurve,
        time: f64,
        maturity: f64,
        rate_factor: f64,
    ) -> PyResult<f64> {
        self.inner
            .bond_price(&discount_curve.inner, time, maturity, rate_factor)
            .map_err(pricing_exception)
    }
    /// Time-zero European option on a zero-coupon bond, with unit notional.
    #[pyo3(signature=(discount_curve, expiry, maturity, strike, *, is_call=true))]
    fn bond_option(
        &self,
        discount_curve: &PyDiscountCurve,
        expiry: f64,
        maturity: f64,
        strike: f64,
        is_call: bool,
    ) -> PyResult<f64> {
        self.inner
            .bond_option(&discount_curve.inner, expiry, maturity, strike, is_call)
            .map_err(pricing_exception)
    }
}

#[pyclass(frozen, name = "HullWhiteLsvTarget", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHullWhiteLsvTarget {
    inner: HullWhiteLsvTarget,
}
#[pymethods]
impl PyHullWhiteLsvTarget {
    /// Flat market-IV smile with matching Dupire variance and T-forward density.
    #[staticmethod]
    #[pyo3(signature=(volatility, time_nodes, log_moneyness_nodes, *, floor=1e-8, cap=4.0))]
    fn flat(
        py: Python<'_>,
        volatility: f64,
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> PyResult<Self> {
        HullWhiteLsvTarget::flat(volatility, time_nodes, log_moneyness_nodes, floor, cap)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    /// Construct both calibration inputs from one eSSVI surface. Grid repairs
    /// are rejected because they would break variance/density consistency.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(slices, terminal_theta_slope, time_nodes, log_moneyness_nodes, *, floor=1e-8, cap=4.0))]
    fn from_essvi(
        py: Python<'_>,
        slices: Vec<Py<PyEssviSlice>>,
        terminal_theta_slope: f64,
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> PyResult<Self> {
        let slices = slices.iter().map(|s| s.borrow(py).inner).collect();
        let surface = EssviSurface::new(
            slices,
            terminal_theta_slope,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .map_err(|e| invalid(py, e))?;
        py.detach(|| {
            HullWhiteLsvTarget::from_surface(&surface, time_nodes, log_moneyness_nodes, floor, cap)
        })
        .map(|inner| Self { inner })
        .map_err(|e| invalid(py, e))
    }
    /// Explicit row-major K*C_KK/P(0,T) samples paired with an LV target model.
    #[staticmethod]
    fn from_grid(
        py: Python<'_>,
        model: &PyModel,
        forward_log_densities: Vec<f64>,
    ) -> PyResult<Self> {
        let ModelSpec::LocalVolatility(lv) = &model.inner else {
            return Err(invalid(py, "target model must be LocalVolatility"));
        };
        HullWhiteLsvTarget::new(lv.local_variance_grid().clone(), forward_log_densities)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    #[getter]
    fn model(&self) -> PyModel {
        let grid = self.inner.grid();
        let lv = LocalVolatilitySpec::from_explicit_grid(
            grid.time_nodes().to_vec(),
            grid.log_moneyness_nodes().to_vec(),
            grid.values().to_vec(),
            grid.floor(),
            grid.cap(),
        )
        .expect("validated hybrid target");
        PyModel {
            inner: ModelSpec::LocalVolatility(lv),
        }
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.grid().time_nodes().to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.grid().log_moneyness_nodes().to_vec()
    }
    #[getter]
    fn forward_log_densities(&self) -> Vec<f64> {
        self.inner.log_densities().to_vec()
    }
}

#[pyclass(frozen, name = "HullWhiteEquityPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHullWhiteEquityPlan {
    inner: HullWhiteEquityPricingPlan,
}
#[pymethods]
impl PyHullWhiteEquityPlan {
    /// Compile price-only BS+HW; cash dividends are currently unsupported.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, rate_model, *, equity_rate_correlation, maximum_step, worker_threads, reduction_block_size=None))]
    fn compile_bs(
        py: Python<'_>,
        request: &PyPricingRequest,
        rate_model: &PyHullWhiteModel,
        equity_rate_correlation: f64,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let request = request.inner.clone();
        let rates = rate_model.inner.clone();
        py.detach(|| {
            HullWhiteEquityPricingPlan::compile_bs(
                &request,
                rates,
                equity_rate_correlation,
                maximum_step,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(pricing_exception)
    }
    /// Calibrate LSV with stochastic-rate correction, then price independently.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(request, target, rate_model, *, vol_mean_reversion, vol_of_vol, equity_vol_correlation, equity_rate_correlation, vol_rate_correlation, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, worker_threads, reduction_block_size=None))]
    fn compile_lsv(
        py: Python<'_>,
        request: &PyPricingRequest,
        target: &PyHullWhiteLsvTarget,
        rate_model: &PyHullWhiteModel,
        vol_mean_reversion: f64,
        vol_of_vol: f64,
        equity_vol_correlation: f64,
        equity_rate_correlation: f64,
        vol_rate_correlation: f64,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| invalid(py, e))?;
        let factor = Bergomi1Factor::new(vol_mean_reversion, vol_of_vol, equity_vol_correlation)
            .map_err(|e| invalid(py, e))?;
        let correlation = HybridCorrelation::new(
            equity_vol_correlation,
            equity_rate_correlation,
            vol_rate_correlation,
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
        let request = request.inner.clone();
        let rates = rate_model.inner.clone();
        let target = target.inner.clone();
        py.detach(|| {
            HullWhiteEquityPricingPlan::compile_lsv(
                &request,
                &target,
                factor,
                rates,
                correlation,
                particles,
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
    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint().to_string()
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn squared_leverage(&self) -> Option<Vec<f64>> {
        self.inner
            .calibration()
            .map(|c| c.surface.squared_leverage().to_vec())
    }
    #[getter]
    fn minimum_effective_samples(&self) -> Vec<f64> {
        self.inner.calibration().map_or_else(Vec::new, |c| {
            c.diagnostics
                .iter()
                .map(|d| d.minimum_effective_samples)
                .collect()
        })
    }
    #[getter]
    fn fallback_nodes(&self) -> Vec<usize> {
        self.inner.calibration().map_or_else(Vec::new, |c| {
            c.diagnostics.iter().map(|d| d.fallback_nodes).collect()
        })
    }
    #[getter]
    fn calibration_discount_means(&self) -> Vec<f64> {
        self.inner.calibration().map_or_else(Vec::new, |c| {
            c.diagnostics
                .iter()
                .map(|d| d.mean_relative_discount)
                .collect()
        })
    }
    #[getter]
    fn calibration_discounted_equity_means(&self) -> Vec<f64> {
        self.inner.calibration().map_or_else(Vec::new, |c| {
            c.diagnostics
                .iter()
                .map(|d| d.mean_discounted_normalized_equity)
                .collect()
        })
    }
}

#[pyclass(frozen, name = "HullWhitePrice", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHullWhitePrice {
    inner: HullWhitePrice,
}
#[pymethods]
impl PyHullWhitePrice {
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
    #[getter]
    fn calibration_method(&self) -> Option<&'static str> {
        self.inner.calibration_method
    }
    #[getter]
    fn calibration_seed(&self) -> Option<u64> {
        self.inner.calibration_seed
    }
    #[getter]
    fn uncertainty_scope(&self) -> &'static str {
        if self.inner.calibration_seed.is_some() {
            "pricing_conditional_on_calibration"
        } else {
            "pricing_only"
        }
    }
}
