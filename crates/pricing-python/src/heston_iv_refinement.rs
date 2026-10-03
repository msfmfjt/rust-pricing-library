//! Immutable finite-grid calibration diagnostics and stage history.
use super::heston_calibration::invalid;
use super::heston_iv_calibration::PyHestonIvCalibrationResult;
use pricing::rough_volatility::{
    HestonIvGridValidation, HestonIvRefinementOptions, HestonIvRefinementResult,
    HestonIvRefinementStage,
};
use pyo3::prelude::*;

// Invalid grid/point inputs must remain validation failures, not pricing failures.
pub(super) fn refinement_failure(
    py: Python<'_>,
    e: pricing::rough_volatility::HestonCalibrationError,
) -> PyErr {
    match e {
        pricing::rough_volatility::HestonCalibrationError::Fourier(
            pricing::rough_volatility::FourierError::InvalidInput(_),
        ) => invalid(py, e),
        _ => super::heston_calibration::failure(py, e),
    }
}

#[pyclass(frozen, name = "HestonIvRefinementOptions", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvRefinementOptions {
    pub(super) inner: HestonIvRefinementOptions,
}
#[pymethods]
impl PyHestonIvRefinementOptions {
    #[staticmethod]
    #[pyo3(signature=(*,fit_tolerance=5e-4,grid_tolerance=1e-4,max_stages=2))]
    fn create(
        py: Python<'_>,
        fit_tolerance: f64,
        grid_tolerance: f64,
        max_stages: usize,
    ) -> PyResult<Self> {
        let inner = HestonIvRefinementOptions {
            fit_tolerance,
            grid_tolerance,
            max_stages,
        };
        inner.validate().map_err(|e| invalid(py, e))?;
        Ok(Self { inner })
    }
    #[getter]
    fn fit_tolerance(&self) -> f64 {
        self.inner.fit_tolerance
    }
    #[getter]
    fn grid_tolerance(&self) -> f64 {
        self.inner.grid_tolerance
    }
    #[getter]
    fn max_stages(&self) -> usize {
        self.inner.max_stages
    }
}
#[pyclass(frozen, name = "HestonIvGridValidation", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvGridValidation {
    pub(super) inner: HestonIvGridValidation,
}
#[pymethods]
impl PyHestonIvGridValidation {
    #[getter]
    fn grid_names(&self) -> Vec<&str> {
        vec!["base", "time", "frequency", "cutoff", "joint"]
    }
    #[getter]
    fn configurations(&self) -> Vec<(usize, usize, f64)> {
        self.inner
            .configurations
            .iter()
            .map(|c| (c.time_steps(), c.integration_intervals(), c.cutoff()))
            .collect()
    }
    #[getter]
    fn model_implied_volatilities(&self) -> Vec<Vec<f64>> {
        self.inner
            .model_implied_volatilities
            .iter()
            .map(|r| r.to_vec())
            .collect()
    }
    #[getter]
    fn iv_residuals(&self) -> Vec<Vec<f64>> {
        self.inner.iv_residuals.iter().map(|r| r.to_vec()).collect()
    }
    #[getter]
    fn iv_differences(&self) -> Vec<Vec<f64>> {
        self.inner
            .iv_differences
            .iter()
            .map(|r| r.to_vec())
            .collect()
    }
    #[getter]
    fn fit_tolerance(&self) -> f64 {
        self.inner.fit_tolerance
    }
    #[getter]
    fn grid_tolerance(&self) -> f64 {
        self.inner.grid_tolerance
    }
    #[getter]
    fn max_abs_iv_residual(&self) -> f64 {
        self.inner.max_abs_iv_residual
    }
    #[getter]
    fn max_abs_iv_difference(&self) -> f64 {
        self.inner.max_abs_iv_difference
    }
    #[getter]
    fn fit_within_tolerance(&self) -> bool {
        self.inner.fit_within_tolerance
    }
    #[getter]
    fn grid_stable(&self) -> bool {
        self.inner.grid_stable
    }
    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted
    }
    #[getter]
    fn plan_compilations(&self) -> usize {
        self.inner.plan_compilations
    }
}
#[pyclass(frozen, name = "HestonIvRefinementStage", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvRefinementStage {
    inner: HestonIvRefinementStage,
    names: Vec<String>,
}
#[pymethods]
impl PyHestonIvRefinementStage {
    #[getter]
    fn initial_parameters(&self) -> Vec<f64> {
        self.inner.initial_parameters.clone()
    }
    #[getter]
    fn calibration(&self) -> PyHestonIvCalibrationResult {
        PyHestonIvCalibrationResult {
            inner: self.inner.calibration.clone(),
            names: self.names.clone(),
        }
    }
    #[getter]
    fn validation(&self) -> PyHestonIvGridValidation {
        PyHestonIvGridValidation {
            inner: self.inner.validation.clone(),
        }
    }
}
#[pyclass(frozen, name = "HestonIvRefinementResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvRefinementResult {
    pub(super) inner: HestonIvRefinementResult,
    pub(super) names: Vec<String>,
}
#[pymethods]
impl PyHestonIvRefinementResult {
    #[getter]
    fn stages(&self) -> Vec<PyHestonIvRefinementStage> {
        self.inner
            .stages
            .iter()
            .map(|s| PyHestonIvRefinementStage {
                inner: s.clone(),
                names: self.names.clone(),
            })
            .collect()
    }
    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted
    }
    #[getter]
    fn termination(&self) -> &str {
        if self.inner.accepted {
            "validation_accepted"
        } else {
            "stage_limit"
        }
    }
    #[getter]
    fn optimizer_evaluations(&self) -> usize {
        self.inner.optimizer_evaluations
    }
    #[getter]
    fn validation_plan_compilations(&self) -> usize {
        self.inner.validation_plan_compilations
    }
}
