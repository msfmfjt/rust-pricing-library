//! Immutable rough-LSV configuration for the common-rate multi-asset adapter.
use crate::multi_asset::invalid;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::models::RoughBergomi;
use pricing::multi_asset::MultiAssetRoughLsvConfig;
use pyo3::prelude::*;

#[pyclass(frozen, name = "MultiAssetRoughLsvConfig", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetRoughLsvConfig {
    pub(crate) inner: MultiAssetRoughLsvConfig,
}
#[pymethods]
impl PyMultiAssetRoughLsvConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*,hurst,vol_of_vol,correlation,particle_count,calibration_seed,log_bandwidth,minimum_effective_samples,retain_reverse_trace=false))]
    fn new(
        py: Python<'_>,
        hurst: f64,
        vol_of_vol: f64,
        correlation: f64,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        retain_reverse_trace: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultiAssetRoughLsvConfig {
                factor: RoughBergomi::new(hurst, vol_of_vol, correlation)
                    .map_err(|e| invalid(py, e))?,
                particles: LsvParticleConfig::new(
                    particle_count,
                    calibration_seed,
                    log_bandwidth,
                    minimum_effective_samples,
                    retain_reverse_trace,
                )
                .map_err(|e| invalid(py, e))?,
            },
        })
    }
    #[getter]
    fn hurst(&self) -> f64 {
        self.inner.factor.hurst()
    }
    /// Eta, the log-variance coefficient (twice the Markovian log-vol coefficient).
    #[getter]
    fn vol_of_vol(&self) -> f64 {
        self.inner.factor.vol_of_vol()
    }
    #[getter]
    fn correlation(&self) -> f64 {
        self.inner.factor.correlation()
    }
    #[getter]
    fn particle_count(&self) -> usize {
        self.inner.particles.particle_count()
    }
    #[getter]
    fn calibration_seed(&self) -> u64 {
        self.inner.particles.seed()
    }
    #[getter]
    fn log_bandwidth(&self) -> f64 {
        self.inner.particles.log_bandwidth()
    }
    #[getter]
    fn minimum_effective_samples(&self) -> f64 {
        self.inner.particles.minimum_effective_samples()
    }
    #[getter]
    fn retain_reverse_trace(&self) -> bool {
        self.inner.particles.retain_reverse_trace()
    }
}
