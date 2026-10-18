use super::*;
use pricing::bass_lv::{
    BassMarketIvModel, BassVegaKtRisk, BassVegaKtRiskPlan, BassVegaKtScenarioDiagnostics,
};
use pricing::market::MarketIvSurface;

#[pyclass(frozen, name = "BassMarketIvModel", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassMarketIvModel {
    inner: BassMarketIvModel,
}
#[pymethods]
impl PyBassMarketIvModel {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(spot,maturity_nodes,log_moneyness_nodes,implied_volatilities,projection_nodes,*,config=None,tail_probability_tolerance=1e-7,relative_mean_tolerance=1e-4))]
    fn calibrate(
        py: Python<'_>,
        spot: f64,
        maturity_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        implied_volatilities: Vec<f64>,
        projection_nodes: Vec<f64>,
        config: Option<PyRef<'_, PyBassLvConfig>>,
        tail_probability_tolerance: f64,
        relative_mean_tolerance: f64,
    ) -> PyResult<Self> {
        let config = config.map(|c| c.config()).unwrap_or_default();
        py.detach(|| {
            let surface =
                MarketIvSurface::new(maturity_nodes, log_moneyness_nodes, implied_volatilities)
                    .map_err(|e| BassError::InvalidInput(e.to_string()))?;
            BassMarketIvModel::calibrate(
                spot,
                surface,
                projection_nodes,
                config,
                BassSurfaceProjectionConfig {
                    tail_probability_tolerance,
                    relative_mean_tolerance,
                },
            )
        })
        .map(|inner| Self { inner })
        .map_err(|e| error(py, e))
    }
    #[getter]
    fn model(&self) -> PyBassLvModel {
        PyBassLvModel {
            inner: self.inner.model().clone(),
        }
    }
    #[getter]
    fn maturity_nodes(&self) -> Vec<f64> {
        self.inner.surface().maturity_nodes().to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.surface().log_moneyness_nodes().to_vec()
    }
    #[getter]
    fn implied_volatilities(&self) -> Vec<f64> {
        self.inner.surface().implied_volatilities().to_vec()
    }
    #[getter]
    fn projection_diagnostics(&self) -> Vec<PyBassSurfaceDiagnostics> {
        self.inner
            .projection_diagnostics()
            .iter()
            .map(Into::into)
            .collect()
    }
    fn bumped(&self, py: Python<'_>, shifts: Vec<f64>) -> PyResult<Self> {
        py.detach(|| self.inner.bumped(&shifts))
            .map(|inner| Self { inner })
            .map_err(|e| error(py, e))
    }
    #[pyo3(signature=(observation_times,*,bump_size=1e-4))]
    fn compile_vega_kt(
        &self,
        py: Python<'_>,
        observation_times: Vec<f64>,
        bump_size: f64,
    ) -> PyResult<PyBassVegaKtRiskPlan> {
        py.detach(|| self.inner.compile_vega_kt(observation_times, bump_size))
            .map(|inner| PyBassVegaKtRiskPlan { inner })
            .map_err(|e| error(py, e))
    }
}

#[pyclass(frozen, name = "BassVegaKtScenarioDiagnostics", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassVegaKtScenarioDiagnostics {
    inner: BassVegaKtScenarioDiagnostics,
}
impl From<&BassVegaKtScenarioDiagnostics> for PyBassVegaKtScenarioDiagnostics {
    fn from(d: &BassVegaKtScenarioDiagnostics) -> Self {
        Self { inner: d.clone() }
    }
}
#[pymethods]
impl PyBassVegaKtScenarioDiagnostics {
    #[getter]
    fn quote_index(&self) -> Option<usize> {
        self.inner.quote_index
    }
    #[getter]
    fn shift(&self) -> f64 {
        self.inner.shift
    }
    #[getter]
    fn initial_spot_error(&self) -> f64 {
        self.inner.initial_spot_error
    }
    #[getter]
    fn calibration(&self) -> Vec<PyBassCalibrationDiagnostics> {
        self.inner.calibration.iter().map(Into::into).collect()
    }
    #[getter]
    fn projection(&self) -> Vec<PyBassSurfaceDiagnostics> {
        self.inner.projection.iter().map(Into::into).collect()
    }
}

#[pyclass(frozen, name = "BassVegaKtRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassVegaKtRisk {
    inner: BassVegaKtRisk,
}
#[pymethods]
impl PyBassVegaKtRisk {
    #[getter]
    fn estimate(&self) -> PyBassEstimate {
        self.inner.estimate.clone().into()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method()
    }
    #[getter]
    fn sensitivities(&self) -> Vec<f64> {
        self.inner.sensitivities.clone()
    }
    #[getter]
    fn standard_errors(&self) -> Vec<f64> {
        self.inner.standard_errors.clone()
    }
    #[getter]
    fn vega_per_vol_point(&self) -> Vec<f64> {
        self.inner.vega_per_vol_point()
    }
    #[getter]
    fn standard_errors_per_vol_point(&self) -> Vec<f64> {
        self.inner.standard_errors_per_vol_point()
    }
    #[getter]
    fn parallel_sensitivity(&self) -> f64 {
        self.inner.parallel_sensitivity
    }
    #[getter]
    fn parallel_standard_error(&self) -> f64 {
        self.inner.parallel_standard_error
    }
    #[getter]
    fn bucket_sum(&self) -> f64 {
        self.inner.bucket_sum
    }
    #[getter]
    fn bucket_sum_standard_error(&self) -> f64 {
        self.inner.bucket_sum_standard_error
    }
    #[getter]
    fn bump_size(&self) -> f64 {
        self.inner.bump_size
    }
    #[getter]
    fn maturity_nodes(&self) -> Vec<f64> {
        self.inner.maturity_nodes.clone()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.log_moneyness_nodes.clone()
    }
    #[getter]
    fn implied_volatilities(&self) -> Vec<f64> {
        self.inner.implied_volatilities.clone()
    }
}

#[pyclass(frozen, name = "BassVegaKtRiskPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassVegaKtRiskPlan {
    inner: BassVegaKtRiskPlan,
}
#[pymethods]
impl PyBassVegaKtRiskPlan {
    #[getter]
    fn simulation(&self) -> PyBassSimulationPlan {
        PyBassSimulationPlan {
            inner: self.inner.simulation().clone(),
        }
    }
    #[getter]
    fn diagnostics(&self) -> Vec<PyBassVegaKtScenarioDiagnostics> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| PyBassVegaKtScenarioDiagnostics { inner: d.clone() })
            .collect()
    }
    #[getter]
    fn bump_size(&self) -> f64 {
        self.inner.bump_size()
    }
    #[getter]
    fn maturity_nodes(&self) -> Vec<f64> {
        self.inner.maturity_nodes().to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.log_moneyness_nodes().to_vec()
    }
    #[getter]
    fn implied_volatilities(&self) -> Vec<f64> {
        self.inner.implied_volatilities().to_vec()
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
    ) -> PyResult<PyBassVegaKtRisk> {
        py.detach(|| {
            self.inner
                .price_european(strike, is_call, paths, seed, discount_factor)
        })
        .map(|inner| PyBassVegaKtRisk { inner })
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
    ) -> PyResult<PyBassVegaKtRisk> {
        py.detach(|| {
            self.inner
                .price_asian(strike, is_call, paths, seed, discount_factor)
        })
        .map(|inner| PyBassVegaKtRisk { inner })
        .map_err(|e| error(py, e))
    }
}
