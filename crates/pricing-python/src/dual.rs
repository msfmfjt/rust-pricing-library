//! Python boundary for policy-based primal-dual bounds on a finite exercise grid.

use pricing::dual::{
    AndersenBroadieConfig, AndersenBroadieError, AndersenBroadiePlan, AndersenBroadieResult,
};
use pricing::mc::ExecutionPolicy;
use pyo3::prelude::*;

use crate::diagnostics::PyDiagnosticEstimate;
use crate::{PyPricingRequest, PyValidationIssue, pricing_exception, validation_exception};

/// Inner rollout counts and seed, separate from LSM training and outer evaluation.
#[pyclass(frozen, name = "AndersenBroadieConfig", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyAndersenBroadieConfig {
    inner: AndersenBroadieConfig,
}

#[pymethods]
impl PyAndersenBroadieConfig {
    #[new]
    fn new(
        py: Python<'_>,
        continuation_inner_paths: u32,
        exercise_inner_paths: u32,
        inner_seed: u64,
    ) -> PyResult<Self> {
        AndersenBroadieConfig::new(continuation_inner_paths, exercise_inner_paths, inner_seed)
            .map(|inner| Self { inner })
            .map_err(|error| dual_validation_exception(py, &error))
    }

    #[getter]
    fn continuation_inner_paths(&self) -> u32 {
        self.inner.continuation_inner_paths()
    }

    #[getter]
    fn exercise_inner_paths(&self) -> u32 {
        self.inner.exercise_inner_paths()
    }

    #[getter]
    fn inner_seed(&self) -> u64 {
        self.inner.inner_seed()
    }

    fn __repr__(&self) -> String {
        format!(
            "AndersenBroadieConfig(continuation_inner_paths={}, exercise_inner_paths={}, inner_seed={})",
            self.continuation_inner_paths(),
            self.exercise_inner_paths(),
            self.inner_seed(),
        )
    }
}

/// Immutable, repeatable dual plan for a price-only AmericanVanilla request.
#[pyclass(frozen, name = "AndersenBroadiePlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAndersenBroadiePlan {
    inner: AndersenBroadiePlan,
}

#[pymethods]
impl PyAndersenBroadiePlan {
    /// Compile using the existing request and an independent inner simulation config.
    #[staticmethod]
    #[pyo3(signature = (request, config, *, worker_threads, reduction_block_size=None))]
    fn compile(
        py: Python<'_>,
        request: &PyPricingRequest,
        config: &PyAndersenBroadieConfig,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        let execution = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|error| validation_exception(py, PyValidationIssue::compile(&error.into())))?;
        let request = request.inner.clone();
        let config = config.inner;
        py.detach(|| AndersenBroadiePlan::compile(&request, execution, config))
            .map(|inner| Self { inner })
            .map_err(|error| dual_validation_exception(py, &error))
    }

    /// Release the GIL while training the policy and evaluating the nested simulations.
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyAndersenBroadieResult> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyAndersenBroadieResult { inner })
            .map_err(pricing_exception)
    }

    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.fingerprint().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "AndersenBroadiePlan(fingerprint={:?})",
            self.plan_fingerprint()
        )
    }
}

/// Statistical bounds conditional on the fitted policy and declared exercise grid.
#[pyclass(frozen, name = "AndersenBroadieResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAndersenBroadieResult {
    inner: AndersenBroadieResult,
}

#[pymethods]
impl PyAndersenBroadieResult {
    #[getter]
    fn lower_bound(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.inner.lower_bound)
    }

    #[getter]
    fn upper_bound(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.inner.upper_bound)
    }

    #[getter]
    fn duality_gap(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.inner.duality_gap)
    }

    /// Asymptotic 95% bracket; excludes continuous-exercise and model error.
    #[getter]
    fn price_confidence_interval_95(&self) -> (f64, f64) {
        let [lower, upper] = self.inner.price_confidence_interval_95;
        (lower, upper)
    }

    #[getter]
    fn policy_fingerprint(&self) -> String {
        self.inner.policy_fingerprint.to_string()
    }

    #[getter]
    fn plan_fingerprint(&self) -> String {
        self.inner.plan_fingerprint.to_string()
    }

    #[getter]
    fn config(&self) -> PyAndersenBroadieConfig {
        PyAndersenBroadieConfig {
            inner: self.inner.config,
        }
    }

    /// Includes both lanes of every antithetic pair.
    #[getter]
    fn outer_trajectories(&self) -> u64 {
        self.inner.outer_trajectories
    }

    #[getter]
    fn exercise_date_count(&self) -> usize {
        self.inner.exercise_date_count
    }

    fn __repr__(&self) -> String {
        format!(
            "AndersenBroadieResult(lower_bound={:?}, upper_bound={:?})",
            self.inner.lower_bound.value().get(),
            self.inner.upper_bound.value().get(),
        )
    }
}

fn dual_validation_exception(py: Python<'_>, error: &AndersenBroadieError) -> PyErr {
    let issue = match error {
        AndersenBroadieError::ZeroInnerPaths { region } => PyValidationIssue::domain(
            format!("/{region}_inner_paths"),
            "zero_inner_paths",
            error.to_string(),
        ),
        AndersenBroadieError::Unsupported { .. } => {
            PyValidationIssue::domain("", "dual_unsupported", error.to_string())
        }
        AndersenBroadieError::RandomCoordinateOverflow => {
            PyValidationIssue::domain("", "dual_random_coordinate_overflow", error.to_string())
        }
        AndersenBroadieError::MonteCarlo(error) => PyValidationIssue::compile(error),
        _ => PyValidationIssue::domain("", "dual_compile_error", error.to_string()),
    };
    validation_exception(py, issue)
}
