use super::*;
use pricing::bass_lv::{
    BassDeterministicMappingRisk, BassMappingBump, BassMappingRisk, BassMappingRiskPlan,
};

#[pyclass(frozen, name = "BassMappingBump", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassMappingBump {
    pub(super) inner: BassMappingBump,
}
#[pymethods]
impl PyBassMappingBump {
    #[new]
    fn new(py: Python<'_>, interval: usize, left: f64, center: f64, right: f64) -> PyResult<Self> {
        BassMappingBump::new(interval, left, center, right)
            .map(|inner| Self { inner })
            .map_err(|e| error(py, e))
    }
    #[getter]
    fn interval(&self) -> usize {
        self.inner.interval()
    }
    #[getter]
    fn left(&self) -> f64 {
        self.inner.left()
    }
    #[getter]
    fn center(&self) -> f64 {
        self.inner.center()
    }
    #[getter]
    fn right(&self) -> f64 {
        self.inner.right()
    }
}

#[pyclass(frozen, name = "BassMappingRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassMappingRisk {
    inner: BassMappingRisk,
}
#[pymethods]
impl PyBassMappingRisk {
    #[getter]
    fn estimate(&self) -> PyBassEstimate {
        self.inner.estimate.clone().into()
    }
    #[getter]
    fn sensitivities(&self) -> Vec<f64> {
        self.inner.sensitivities.clone()
    }
    #[getter]
    fn standard_errors(&self) -> Vec<f64> {
        self.inner.standard_errors.clone()
    }
}

#[pyclass(frozen, name = "BassDeterministicMappingRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassDeterministicMappingRisk {
    inner: BassDeterministicMappingRisk,
}
#[pymethods]
impl PyBassDeterministicMappingRisk {
    #[getter]
    fn price(&self) -> f64 {
        self.inner.price
    }
    #[getter]
    fn sensitivities(&self) -> Vec<f64> {
        self.inner.sensitivities.clone()
    }
}

#[pyclass(frozen, name = "BassMappingRiskPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBassMappingRiskPlan {
    pub(super) inner: BassMappingRiskPlan,
}
#[pymethods]
impl PyBassMappingRiskPlan {
    #[getter]
    fn simulation(&self) -> PyBassSimulationPlan {
        PyBassSimulationPlan {
            inner: self.inner.simulation().clone(),
        }
    }
    #[getter]
    fn bumps(&self) -> Vec<PyBassMappingBump> {
        self.inner
            .bumps()
            .iter()
            .map(|b| PyBassMappingBump { inner: *b })
            .collect()
    }
    fn path_sensitivities(
        &self,
        py: Python<'_>,
        normals: Vec<f64>,
        payoff_partials: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .path_sensitivities(&normals, &payoff_partials)
            .map_err(|e| error(py, e))
    }
    fn bumped_simulation(
        &self,
        py: Python<'_>,
        amplitudes: Vec<f64>,
    ) -> PyResult<PyBassSimulationPlan> {
        py.detach(|| self.inner.bumped_simulation(&amplitudes))
            .map(|inner| PyBassSimulationPlan { inner })
            .map_err(|e| error(py, e))
    }
    #[pyo3(signature=(interval,strike,*,discount_factor=1.0))]
    fn vanilla_call(
        &self,
        py: Python<'_>,
        interval: usize,
        strike: f64,
        discount_factor: f64,
    ) -> PyResult<PyBassDeterministicMappingRisk> {
        py.detach(|| self.inner.vanilla_call(interval, strike, discount_factor))
            .map(|inner| PyBassDeterministicMappingRisk { inner })
            .map_err(|e| error(py, e))
    }
    #[pyo3(signature=(amplitudes,interval,strike,*,discount_factor=1.0))]
    fn bumped_vanilla_call(
        &self,
        py: Python<'_>,
        amplitudes: Vec<f64>,
        interval: usize,
        strike: f64,
        discount_factor: f64,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner
                .bumped_vanilla_call(&amplitudes, interval, strike, discount_factor)
        })
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
    ) -> PyResult<PyBassMappingRisk> {
        py.detach(|| {
            self.inner
                .price_european(strike, is_call, paths, seed, discount_factor)
        })
        .map(|inner| PyBassMappingRisk { inner })
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
    ) -> PyResult<PyBassMappingRisk> {
        py.detach(|| {
            self.inner
                .price_asian(strike, is_call, paths, seed, discount_factor)
        })
        .map(|inner| PyBassMappingRisk { inner })
        .map_err(|e| error(py, e))
    }
}
