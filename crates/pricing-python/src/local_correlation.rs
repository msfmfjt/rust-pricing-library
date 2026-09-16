//! Immutable particle local-correlation configuration, diagnostics and joint risk.
use crate::builders::PyModel;
use crate::multi_asset::{PyCorrelationSchedule, invalid};
use pricing::mc::lsv::LsvParticleConfig;
use pricing::models::ModelSpec;
use pricing::multi_asset::{
    LocalCorrelationCalibration, LocalCorrelationConfig, LocalCorrelationFeasibility,
    LocalCorrelationRisk,
};
use pyo3::prelude::*;

#[pyclass(frozen, name = "LocalCorrelationConfig", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyLocalCorrelationConfig {
    pub(crate) inner: LocalCorrelationConfig,
    target: ModelSpec,
}
#[pymethods]
impl PyLocalCorrelationConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*,basket_weights,target_model,second_correlations,particle_count,calibration_seed,log_bandwidth,minimum_effective_samples,feasibility="reject",minimum_variance_span=1e-12,retain_reverse_trace=false))]
    fn new(
        py: Python<'_>,
        basket_weights: Vec<f64>,
        target_model: &PyModel,
        second_correlations: &PyCorrelationSchedule,
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        feasibility: &str,
        minimum_variance_span: f64,
        retain_reverse_trace: bool,
    ) -> PyResult<Self> {
        let ModelSpec::LocalVolatility(target) = &target_model.inner else {
            return Err(invalid(
                py,
                "local correlation target_model must contain an effective local variance grid",
            ));
        };
        if basket_weights.len() < 2
            || basket_weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
            || (basket_weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
            || !minimum_variance_span.is_finite()
            || minimum_variance_span <= 0.0
        {
            return Err(invalid(
                py,
                "local correlation requires positive normalized weights and a positive variance-span tolerance",
            ));
        }
        let feasibility = match feasibility {
            "reject" => LocalCorrelationFeasibility::Reject,
            "project_and_report" => LocalCorrelationFeasibility::ProjectAndReport,
            _ => {
                return Err(invalid(
                    py,
                    "feasibility must be reject or project_and_report",
                ));
            }
        };
        Ok(Self {
            target: target_model.inner.clone(),
            inner: LocalCorrelationConfig {
                basket_weights,
                target: target.local_variance_grid().clone(),
                second_correlation: second_correlations.inner.clone(),
                particles: LsvParticleConfig::new(
                    particle_count,
                    calibration_seed,
                    log_bandwidth,
                    minimum_effective_samples,
                    retain_reverse_trace,
                )
                .map_err(|e| invalid(py, e))?,
                feasibility,
                minimum_variance_span,
            },
        })
    }
    #[getter]
    fn basket_weights(&self) -> Vec<f64> {
        self.inner.basket_weights.clone()
    }
    #[getter]
    fn target_model(&self) -> PyModel {
        PyModel {
            inner: self.target.clone(),
        }
    }
    #[getter]
    fn second_correlations(&self) -> PyCorrelationSchedule {
        PyCorrelationSchedule {
            inner: self.inner.second_correlation.clone(),
        }
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
    #[getter]
    fn minimum_variance_span(&self) -> f64 {
        self.inner.minimum_variance_span
    }
    #[getter]
    fn feasibility(&self) -> &'static str {
        match self.inner.feasibility {
            LocalCorrelationFeasibility::Reject => "reject",
            LocalCorrelationFeasibility::ProjectAndReport => "project_and_report",
        }
    }
}

#[pyclass(frozen, name = "LocalCorrelationCalibration", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyLocalCorrelationCalibration {
    pub(crate) inner: LocalCorrelationCalibration,
}
#[pymethods]
impl PyLocalCorrelationCalibration {
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn log_nodes(&self) -> Vec<f64> {
        self.inner.log_nodes().to_vec()
    }
    #[getter]
    fn mixing_coefficients(&self) -> Vec<f64> {
        self.inner.mixing_coefficients().to_vec()
    }
    #[getter]
    fn endpoint_variances(&self) -> Vec<Vec<f64>> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.endpoint_variances.to_vec())
            .collect()
    }
    #[getter]
    fn target_variances(&self) -> Vec<f64> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.target_variance)
            .collect()
    }
    #[getter]
    fn attained_variances(&self) -> Vec<f64> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.attained_variance)
            .collect()
    }
    #[getter]
    fn variance_residuals(&self) -> Vec<f64> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.attained_variance - d.target_variance)
            .collect()
    }
    #[getter]
    fn raw_mixing(&self) -> Vec<f64> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.raw_mixing)
            .collect()
    }
    #[getter]
    fn effective_samples(&self) -> Vec<f64> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.effective_samples)
            .collect()
    }
    #[getter]
    fn source_nodes(&self) -> Vec<usize> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.source_node)
            .collect()
    }
    #[getter]
    fn fallback_nodes(&self) -> Vec<bool> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.fallback)
            .collect()
    }
    #[getter]
    fn projected_nodes(&self) -> Vec<bool> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.projected)
            .collect()
    }
    #[getter]
    fn unidentifiable_nodes(&self) -> Vec<bool> {
        self.inner
            .diagnostics()
            .iter()
            .map(|d| d.unidentifiable)
            .collect()
    }
    #[getter]
    fn particle_means(&self) -> Vec<Vec<f64>> {
        self.inner.particle_means().to_vec()
    }
    #[getter]
    fn retains_reverse_trace(&self) -> bool {
        self.inner.retains_reverse_trace()
    }
    fn correlation_at(
        &self,
        py: Python<'_>,
        time: f64,
        log_basket: f64,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.inner
            .correlation_at(time, log_basket)
            .map_err(|e| invalid(py, e))
    }
}

#[pyclass(frozen, name = "LocalCorrelationRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyLocalCorrelationRisk {
    pub(crate) inner: LocalCorrelationRisk,
}
#[pymethods]
impl PyLocalCorrelationRisk {
    #[getter]
    fn basket_time_nodes(&self) -> Vec<f64> {
        self.inner.basket_time_nodes.clone()
    }
    #[getter]
    fn basket_log_nodes(&self) -> Vec<f64> {
        self.inner.basket_log_nodes.clone()
    }
    #[getter]
    fn basket_variance_adjoints(&self) -> Vec<f64> {
        self.inner.basket_variance_adjoints.clone()
    }
    #[getter]
    fn basket_standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.basket_standard_errors.clone()
    }
    #[getter]
    fn asset_adjoints(&self) -> Vec<Vec<f64>> {
        self.inner.asset_adjoints.clone()
    }
    #[getter]
    fn asset_standard_errors(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.asset_standard_errors.clone()
    }
    #[getter]
    fn asset_time_nodes(&self) -> Vec<Vec<f64>> {
        self.inner.asset_time_nodes.clone()
    }
    #[getter]
    fn asset_log_nodes(&self) -> Vec<Vec<f64>> {
        self.inner.asset_log_nodes.clone()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
}
