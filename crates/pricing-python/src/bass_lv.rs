use crate::builders::PyEssviSlice;
use crate::{PricingError, PyValidationIssue, validation_exception};
use pricing::bass_lv::{
    BassCalibrationDiagnostics, BassError, BassEstimate, BassLvConfig, BassLvModel, BassMarginal,
    BassMarginalProjection, BassSimulationPlan, BassSurfaceDiagnostics,
    BassSurfaceProjectionConfig,
};
use pricing::market::{EssviSurface, SurfaceValidationTolerance};
use pyo3::prelude::*;

mod market_risk;
mod risk;
pub use market_risk::{
    PyBassMarketIvModel, PyBassVegaKtRisk, PyBassVegaKtRiskPlan, PyBassVegaKtScenarioDiagnostics,
};
pub use risk::{
    PyBassDeterministicMappingRisk, PyBassMappingBump, PyBassMappingRisk, PyBassMappingRiskPlan,
};

fn error(py: Python<'_>, e: BassError) -> PyErr {
    match e {
        BassError::InvalidInput(_) | BassError::ConvexOrder { .. } => validation_exception(
            py,
            PyValidationIssue::domain("/bass_lv", "invalid_bass_configuration", e.to_string()),
        ),
        _ => PricingError::new_err(e.to_string()),
    }
}

#[pyclass(frozen, name = "BassMarginal", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassMarginal {
    inner: BassMarginal,
}
#[pymethods]
impl PyBassMarginal {
    #[new]
    fn new(
        py: Python<'_>,
        expiry: f64,
        spots: Vec<f64>,
        probabilities: Vec<f64>,
    ) -> PyResult<Self> {
        BassMarginal::new(expiry, spots, probabilities)
            .map(|inner| Self { inner })
            .map_err(|e| error(py, e))
    }
    #[staticmethod]
    #[pyo3(signature=(expiry,spot,volatility,*,nodes=1601,tail_std=7.0))]
    fn lognormal(
        py: Python<'_>,
        expiry: f64,
        spot: f64,
        volatility: f64,
        nodes: usize,
        tail_std: f64,
    ) -> PyResult<Self> {
        BassMarginal::lognormal(expiry, spot, volatility, nodes, tail_std)
            .map(|inner| Self { inner })
            .map_err(|e| error(py, e))
    }
    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry()
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(expiry,spot,slices,log_moneyness_nodes,*,terminal_theta_slope=0.0,tail_probability_tolerance=1e-7,relative_mean_tolerance=1e-4))]
    fn from_essvi(
        py: Python<'_>,
        expiry: f64,
        spot: f64,
        slices: Vec<Py<PyEssviSlice>>,
        log_moneyness_nodes: Vec<f64>,
        terminal_theta_slope: f64,
        tail_probability_tolerance: f64,
        relative_mean_tolerance: f64,
    ) -> PyResult<PyBassMarginalProjection> {
        let slices = slices.iter().map(|s| s.borrow(py).inner).collect();
        let surface = EssviSurface::new(
            slices,
            terminal_theta_slope,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .map_err(|e| error(py, BassError::InvalidInput(e.to_string())))?;
        let config = BassSurfaceProjectionConfig {
            tail_probability_tolerance,
            relative_mean_tolerance,
        };
        py.detach(|| {
            BassMarginal::from_surface(expiry, spot, &surface, &log_moneyness_nodes, config)
        })
        .map(|inner| PyBassMarginalProjection { inner })
        .map_err(|e| error(py, e))
    }
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()
    }
    #[getter]
    fn spots(&self) -> Vec<f64> {
        self.inner.spots().to_vec()
    }
    #[getter]
    fn probabilities(&self) -> Vec<f64> {
        self.inner.probabilities().to_vec()
    }
    fn cdf(&self, spot: f64) -> f64 {
        self.inner.cdf(spot)
    }
    fn quantile(&self, py: Python<'_>, probability: f64) -> PyResult<f64> {
        self.inner.quantile(probability).map_err(|e| error(py, e))
    }
    fn call_price(&self, py: Python<'_>, strike: f64) -> PyResult<f64> {
        self.inner.call_price(strike).map_err(|e| error(py, e))
    }
}

#[pyclass(frozen, name = "BassSurfaceDiagnostics", skip_from_py_object, get_all)]
#[derive(Clone, Debug)]
pub struct PyBassSurfaceDiagnostics {
    lower_tail_probability: f64,
    upper_tail_probability: f64,
    retained_probability: f64,
    unscaled_mean: f64,
    mean_scale: f64,
    max_call_price_error: f64,
    retained_nodes: usize,
}
impl From<&BassSurfaceDiagnostics> for PyBassSurfaceDiagnostics {
    fn from(d: &BassSurfaceDiagnostics) -> Self {
        Self {
            lower_tail_probability: d.lower_tail_probability,
            upper_tail_probability: d.upper_tail_probability,
            retained_probability: d.retained_probability,
            unscaled_mean: d.unscaled_mean,
            mean_scale: d.mean_scale,
            max_call_price_error: d.max_call_price_error,
            retained_nodes: d.retained_nodes,
        }
    }
}
#[pyclass(frozen, name = "BassMarginalProjection", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassMarginalProjection {
    inner: BassMarginalProjection,
}
#[pymethods]
impl PyBassMarginalProjection {
    #[getter]
    fn marginal(&self) -> PyBassMarginal {
        PyBassMarginal {
            inner: self.inner.marginal.clone(),
        }
    }
    #[getter]
    fn diagnostics(&self) -> PyBassSurfaceDiagnostics {
        (&self.inner.diagnostics).into()
    }
}

#[pyclass(frozen, name = "BassLvConfig", skip_from_py_object, get_all)]
#[derive(Clone, Debug)]
pub struct PyBassLvConfig {
    grid_points: usize,
    grid_width: f64,
    max_iterations: usize,
    cdf_tolerance: f64,
    tail_tolerance: f64,
}
impl PyBassLvConfig {
    pub(crate) fn config(&self) -> BassLvConfig {
        BassLvConfig {
            grid_points: self.grid_points,
            grid_width: self.grid_width,
            max_iterations: self.max_iterations,
            cdf_tolerance: self.cdf_tolerance,
            tail_tolerance: self.tail_tolerance,
        }
    }
}
#[pymethods]
impl PyBassLvConfig {
    #[new]
    #[pyo3(signature=(*,grid_points=801,grid_width=8.0,max_iterations=2000,cdf_tolerance=1e-6,tail_tolerance=1e-7))]
    fn new(
        grid_points: usize,
        grid_width: f64,
        max_iterations: usize,
        cdf_tolerance: f64,
        tail_tolerance: f64,
    ) -> Self {
        Self {
            grid_points,
            grid_width,
            max_iterations,
            cdf_tolerance,
            tail_tolerance,
        }
    }
}

#[pyclass(
    frozen,
    name = "BassCalibrationDiagnostics",
    skip_from_py_object,
    get_all
)]
#[derive(Clone, Debug)]
pub struct PyBassCalibrationDiagnostics {
    start_time: f64,
    end_time: f64,
    iterations: usize,
    cdf_residual: f64,
    marginal_cdf_error: f64,
    boundary_tail_probability: f64,
    brownian_min: f64,
    brownian_max: f64,
    grid_spacing: f64,
}
impl From<&BassCalibrationDiagnostics> for PyBassCalibrationDiagnostics {
    fn from(d: &BassCalibrationDiagnostics) -> Self {
        Self {
            start_time: d.start_time,
            end_time: d.end_time,
            iterations: d.iterations,
            cdf_residual: d.cdf_residual,
            marginal_cdf_error: d.marginal_cdf_error,
            boundary_tail_probability: d.boundary_tail_probability,
            brownian_min: d.brownian_min,
            brownian_max: d.brownian_max,
            grid_spacing: d.grid_spacing,
        }
    }
}
#[pyclass(frozen, name = "BassEstimate", skip_from_py_object, get_all)]
#[derive(Clone, Debug)]
pub struct PyBassEstimate {
    price: f64,
    standard_error: f64,
    paths: usize,
    seed: u64,
}
impl From<BassEstimate> for PyBassEstimate {
    fn from(e: BassEstimate) -> Self {
        Self {
            price: e.price,
            standard_error: e.standard_error,
            paths: e.paths,
            seed: e.seed,
        }
    }
}

#[pyclass(frozen, name = "BassLvModel", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassLvModel {
    inner: BassLvModel,
}
#[pymethods]
impl PyBassLvModel {
    #[staticmethod]
    #[pyo3(signature=(spot,marginals,*,config=None))]
    fn calibrate(
        py: Python<'_>,
        spot: f64,
        marginals: Vec<Py<PyBassMarginal>>,
        config: Option<PyRef<'_, PyBassLvConfig>>,
    ) -> PyResult<Self> {
        let config = config.map(|c| c.config()).unwrap_or_default();
        let marginals = marginals
            .iter()
            .map(|m| m.borrow(py).inner.clone())
            .collect();
        py.detach(|| BassLvModel::calibrate(spot, marginals, config))
            .map(|inner| Self { inner })
            .map_err(|e| error(py, e))
    }
    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot()
    }
    #[getter]
    fn initial_spot_error(&self) -> f64 {
        self.inner.initial_spot_error()
    }
    #[getter]
    fn initial_brownian_state(&self) -> f64 {
        self.inner.initial_brownian_state()
    }
    #[getter]
    fn diagnostics(&self) -> Vec<PyBassCalibrationDiagnostics> {
        self.inner.diagnostics().iter().map(Into::into).collect()
    }
    fn mapping(&self, py: Python<'_>, time: f64, brownian_state: f64) -> PyResult<f64> {
        self.inner
            .mapping(time, brownian_state)
            .map_err(|e| error(py, e))
    }
    fn local_volatility(&self, py: Python<'_>, time: f64, spot: f64) -> PyResult<f64> {
        self.inner
            .local_volatility(time, spot)
            .map_err(|e| error(py, e))
    }
    fn compile_simulation(
        &self,
        py: Python<'_>,
        observation_times: Vec<f64>,
    ) -> PyResult<PyBassSimulationPlan> {
        py.detach(|| self.inner.compile_simulation(observation_times))
            .map(|inner| PyBassSimulationPlan { inner })
            .map_err(|e| error(py, e))
    }
    fn compile_mapping_risk(
        &self,
        py: Python<'_>,
        observation_times: Vec<f64>,
        bumps: Vec<Py<PyBassMappingBump>>,
    ) -> PyResult<PyBassMappingRiskPlan> {
        let bumps = bumps.iter().map(|b| b.borrow(py).inner).collect();
        py.detach(|| self.inner.compile_mapping_risk(observation_times, bumps))
            .map(|inner| PyBassMappingRiskPlan { inner })
            .map_err(|e| error(py, e))
    }
}

#[pyclass(frozen, name = "BassSimulationPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassSimulationPlan {
    inner: BassSimulationPlan,
}
#[pymethods]
impl PyBassSimulationPlan {
    #[getter]
    fn observation_times(&self) -> Vec<f64> {
        self.inner.observation_times().to_vec()
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn normal_count(&self) -> usize {
        self.inner.normal_count()
    }
    fn path_from_normals(&self, py: Python<'_>, normals: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .path_from_normals(&normals)
            .map_err(|e| error(py, e))
    }
    #[pyo3(signature=(paths,*,seed=0))]
    fn sample_paths(&self, py: Python<'_>, paths: usize, seed: u64) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| self.inner.sample_paths(paths, seed))
            .map_err(|e| error(py, e))
    }
    #[pyo3(signature=(strike,*,is_call=true,paths=100_000,seed=0,discount_factor=1.0))]
    fn price_european(
        &self,
        py: Python<'_>,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> PyResult<PyBassEstimate> {
        py.detach(|| {
            self.inner
                .price_european(strike, is_call, paths, seed, discount_factor)
        })
        .map(Into::into)
        .map_err(|e| error(py, e))
    }
    #[pyo3(signature=(strike,*,is_call=true,paths=100_000,seed=0,discount_factor=1.0))]
    fn price_asian(
        &self,
        py: Python<'_>,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> PyResult<PyBassEstimate> {
        py.detach(|| {
            self.inner
                .price_asian(strike, is_call, paths, seed, discount_factor)
        })
        .map(Into::into)
        .map_err(|e| error(py, e))
    }
}
