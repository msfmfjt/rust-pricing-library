//! Immutable configuration, calibration diagnostics and target-risk snapshots.
use crate::multi_asset::invalid;
use pricing::mc::lsv::{CalibratedBergomiLsv, LSV_PARTICLE_CALIBRATION, LsvParticleConfig};
use pricing::models::{Bergomi1Factor, Bergomi2Factor, BergomiDynamics};
use pricing::multi_asset::{
    MultiAssetBergomiLsvConfig, MultiAssetLsv2FactorConfig, MultiAssetLsvConfig, MultiAssetLsvRisk,
};
use pyo3::prelude::*;

#[pyclass(frozen, name = "MultiAssetLsvConfig", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetLsvConfig {
    pub(crate) inner: MultiAssetLsvConfig,
}
#[pymethods]
impl PyMultiAssetLsvConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*,mean_reversion,vol_of_vol,correlation,particle_count,calibration_seed,log_bandwidth,minimum_effective_samples,retain_reverse_trace=false))]
    fn new(
        py: Python<'_>,
        mean_reversion: f64,
        vol_of_vol: f64,
        correlation: f64,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        retain_reverse_trace: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultiAssetLsvConfig {
                factor: Bergomi1Factor::new(mean_reversion, vol_of_vol, correlation)
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
    fn mean_reversion(&self) -> f64 {
        self.inner.factor.mean_reversion()
    }
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

#[pyclass(frozen, name = "MultiAssetLsv2FactorConfig", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetLsv2FactorConfig {
    pub(crate) inner: MultiAssetLsv2FactorConfig,
}
#[pymethods]
impl PyMultiAssetLsv2FactorConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*,mean_reversions,vol_of_vol,mixing_weight,spot_correlations,factor_correlation,particle_count,calibration_seed,log_bandwidth,minimum_effective_samples,retain_reverse_trace=false))]
    fn new(
        py: Python<'_>,
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        retain_reverse_trace: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultiAssetLsv2FactorConfig {
                factor: Bergomi2Factor::new(
                    mean_reversions,
                    vol_of_vol,
                    mixing_weight,
                    spot_correlations,
                    factor_correlation,
                )
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
    fn mean_reversions(&self) -> [f64; 2] {
        self.inner.factor.mean_reversions()
    }
    #[getter]
    fn vol_of_vol(&self) -> f64 {
        self.inner.factor.vol_of_vol()
    }
    #[getter]
    fn spot_correlations(&self) -> [f64; 2] {
        self.inner.factor.spot_correlations()
    }
    #[getter]
    fn mixing_weight(&self) -> f64 {
        self.inner.factor.mixing_weight()
    }
    #[getter]
    fn factor_correlation(&self) -> f64 {
        self.inner.factor.factor_correlation()
    }
    #[getter]
    fn normalized_weights(&self) -> [f64; 2] {
        self.inner.factor.normalized_weights()
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

#[pyclass(frozen, name = "MultiAssetLsvCalibration", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetLsvCalibration {
    #[pyo3(get)]
    calibration_seed: u64,
    #[pyo3(get)]
    volatility_factor_count: usize,
    #[pyo3(get)]
    time_nodes: Vec<f64>,
    #[pyo3(get)]
    log_moneyness_nodes: Vec<f64>,
    #[pyo3(get)]
    squared_leverage: Vec<f64>,
    #[pyo3(get)]
    effective_samples: Vec<f64>,
    #[pyo3(get)]
    donor_nodes: Vec<usize>,
    #[pyo3(get)]
    extrapolated: Vec<bool>,
    #[pyo3(get)]
    minimum_effective_samples: Vec<f64>,
    #[pyo3(get)]
    extrapolated_nodes: Vec<usize>,
    #[pyo3(get)]
    particle_mean_normalized_f: Vec<f64>,
}
impl<F: BergomiDynamics> From<&CalibratedBergomiLsv<F>> for PyMultiAssetLsvCalibration {
    fn from(c: &CalibratedBergomiLsv<F>) -> Self {
        Self {
            calibration_seed: c.config().seed(),
            volatility_factor_count: F::FACTOR_COUNT,
            time_nodes: c.surface().times().to_vec(),
            log_moneyness_nodes: c.surface().log_nodes().to_vec(),
            squared_leverage: c.surface().squared_leverage().to_vec(),
            effective_samples: c
                .conditional_moments()
                .iter()
                .map(|m| m.effective_samples)
                .collect(),
            donor_nodes: c
                .conditional_moments()
                .iter()
                .map(|m| m.source_node)
                .collect(),
            extrapolated: c
                .conditional_moments()
                .iter()
                .map(|m| m.extrapolated)
                .collect(),
            minimum_effective_samples: c
                .diagnostics()
                .iter()
                .map(|r| r.minimum_effective_samples)
                .collect(),
            extrapolated_nodes: c
                .diagnostics()
                .iter()
                .map(|r| r.extrapolated_nodes)
                .collect(),
            particle_mean_normalized_f: c.diagnostics().iter().map(|r| r.particle_mean_f).collect(),
        }
    }
}
#[pymethods]
impl PyMultiAssetLsvCalibration {
    #[getter]
    fn method(&self) -> &'static str {
        LSV_PARTICLE_CALIBRATION
    }
}

#[pyclass(frozen, name = "MultiAssetLsvRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetLsvRisk {
    pub(crate) inner: MultiAssetLsvRisk,
}
#[pymethods]
impl PyMultiAssetLsvRisk {
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes.clone()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.log_moneyness_nodes.clone()
    }
    #[getter]
    fn node_adjoints(&self) -> Vec<f64> {
        self.inner.node_adjoints.clone()
    }
    #[getter]
    fn standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.standard_errors.clone()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
}

pub(crate) fn extract_lsv_config(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<MultiAssetBergomiLsvConfig> {
    if let Ok(c) = value.extract::<PyRef<'_, PyMultiAssetLsvConfig>>() {
        return Ok(c.inner.clone().into());
    }
    if let Ok(c) = value.extract::<PyRef<'_, PyMultiAssetLsv2FactorConfig>>() {
        return Ok(c.inner.clone().into());
    }
    Err(invalid(
        py,
        "lsv_configs requires MultiAssetLsvConfig, MultiAssetLsv2FactorConfig or None",
    ))
}
