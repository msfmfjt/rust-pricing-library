//! Multi-expiry Fourier price calibration; no implicit Spot/dividend conversion.
use super::rough_volatility::PyRoughVolatilityModel;
use super::{PyValidationIssue, pricing_exception, validation_exception};
use pricing::rough_volatility::{
    HestonCalibrationError, HestonCalibrationEvaluation, HestonCalibrationParameter as P,
    HestonCalibrationProblem, HestonCalibrationQuote, HestonCalibrationResult,
    HestonCalibrationVariable, HestonFourierConfig, LeastSquaresOptions, LeastSquaresTermination,
};
use pyo3::prelude::*;
pub(super) fn invalid(py: Python<'_>, message: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/heston_calibration",
            "invalid_calibration_input",
            message.to_string(),
        ),
    )
}
pub(super) fn failure(py: Python<'_>, e: HestonCalibrationError) -> PyErr {
    match e {
        HestonCalibrationError::InvalidInput(_) => invalid(py, e),
        _ => pricing_exception(e),
    }
}
fn parameter(py: Python<'_>, name: &str) -> PyResult<P> {
    match name {
        "initial_variance" => Ok(P::InitialVariance),
        "mean_reversion" => Ok(P::MeanReversion),
        "long_run_variance" => Ok(P::LongRunVariance),
        "vol_of_vol" => Ok(P::VolOfVol),
        "correlation" => Ok(P::Correlation),
        "hurst" => Ok(P::Hurst),
        _ => Err(invalid(py, "unknown Heston calibration parameter")),
    }
}
#[pyclass(frozen, name = "HestonCalibrationQuote", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonCalibrationQuote {
    inner: HestonCalibrationQuote,
}
#[pymethods]
impl PyHestonCalibrationQuote {
    /// Quote data are fully validated when the problem is compiled.
    #[staticmethod]
    #[pyo3(signature=(maturity,forward,strike,discount,target_price,price_scale=1.0,*,is_call=true))]
    fn create(
        maturity: f64,
        forward: f64,
        strike: f64,
        discount: f64,
        target_price: f64,
        price_scale: f64,
        is_call: bool,
    ) -> Self {
        Self {
            inner: HestonCalibrationQuote {
                maturity,
                forward,
                strike,
                discount,
                target_price,
                price_scale,
                is_call,
            },
        }
    }
    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity
    }
    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward
    }
    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike
    }
    #[getter]
    fn discount(&self) -> f64 {
        self.inner.discount
    }
    #[getter]
    fn target_price(&self) -> f64 {
        self.inner.target_price
    }
    #[getter]
    fn price_scale(&self) -> f64 {
        self.inner.price_scale
    }
    #[getter]
    fn is_call(&self) -> bool {
        self.inner.is_call
    }
}
#[pyclass(frozen, name = "HestonCalibrationVariable", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonCalibrationVariable {
    pub(super) inner: HestonCalibrationVariable,
}
#[pymethods]
impl PyHestonCalibrationVariable {
    #[staticmethod]
    fn create(py: Python<'_>, name: &str, lower: f64, upper: f64, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: HestonCalibrationVariable {
                parameter: parameter(py, name)?,
                lower,
                upper,
                scale,
            },
        })
    }
    #[getter]
    fn name(&self) -> &str {
        self.inner.parameter.name()
    }
    #[getter]
    fn lower(&self) -> f64 {
        self.inner.lower
    }
    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper
    }
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }
}
#[pyclass(frozen, name = "HestonCalibrationEvaluation", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonCalibrationEvaluation {
    inner: HestonCalibrationEvaluation,
    width: usize,
}
#[pymethods]
impl PyHestonCalibrationEvaluation {
    #[getter]
    fn model_prices(&self) -> Vec<f64> {
        self.inner.model_prices.clone()
    }
    #[getter]
    fn scaled_residuals(&self) -> Vec<f64> {
        self.inner.scaled_residuals.clone()
    }
    #[getter]
    fn jacobian(&self) -> Vec<Vec<f64>> {
        self.inner
            .jacobian
            .chunks_exact(self.width)
            .map(|r| r.to_vec())
            .collect()
    }
    #[getter]
    fn quadrature_differences(&self) -> Vec<f64> {
        self.inner.quadrature_differences.clone()
    }
    #[getter]
    fn tail_indicators(&self) -> Vec<f64> {
        self.inner.tail_indicators.clone()
    }
}
#[pyclass(frozen, name = "HestonCalibrationResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonCalibrationResult {
    inner: HestonCalibrationResult,
    names: Vec<String>,
}
#[pymethods]
impl PyHestonCalibrationResult {
    #[getter]
    fn model(&self) -> PyRoughVolatilityModel {
        PyRoughVolatilityModel {
            inner: self.inner.model.clone(),
        }
    }
    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.inner.optimizer.parameters.clone()
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.names.clone()
    }
    #[getter]
    fn evaluation(&self) -> PyHestonCalibrationEvaluation {
        PyHestonCalibrationEvaluation {
            inner: self.inner.evaluation.clone(),
            width: self.names.len(),
        }
    }
    #[getter]
    fn fit_achieved(&self) -> bool {
        self.inner.fit_achieved
    }
    #[getter]
    fn termination(&self) -> &str {
        match self.inner.optimizer.termination {
            LeastSquaresTermination::ResidualTolerance => "residual_tolerance",
            LeastSquaresTermination::ProjectedGradientTolerance => "projected_gradient_tolerance",
            LeastSquaresTermination::StepTolerance => "step_tolerance",
            LeastSquaresTermination::MaxIterations => "max_iterations",
            LeastSquaresTermination::MaxEvaluations => "max_evaluations",
            LeastSquaresTermination::NoProgress => "no_progress",
        }
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.optimizer.objective
    }
    #[getter]
    fn projected_gradient_norm(&self) -> f64 {
        self.inner.optimizer.projected_gradient_norm
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.optimizer.iterations
    }
    #[getter]
    fn evaluations(&self) -> usize {
        self.inner.evaluations
    }
    #[getter]
    fn rejected_evaluations(&self) -> usize {
        self.inner.optimizer.rejected_evaluations
    }
    #[getter]
    fn invalid_evaluations(&self) -> usize {
        self.inner.optimizer.invalid_evaluations
    }
    #[getter]
    fn active_bounds(&self) -> Vec<bool> {
        self.inner.active_bounds.clone()
    }
    #[getter]
    fn accepted_objectives(&self) -> Vec<f64> {
        self.inner.optimizer.accepted_objectives.clone()
    }
}
#[pyclass(frozen, name = "HestonCalibrationProblem", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonCalibrationProblem {
    inner: HestonCalibrationProblem,
}
#[pymethods]
impl PyHestonCalibrationProblem {
    #[staticmethod]
    #[pyo3(signature=(model,quotes,variables,*,time_steps=512,integration_intervals=512,cutoff=128.0))]
    fn compile(
        py: Python<'_>,
        model: &PyRoughVolatilityModel,
        quotes: Vec<Py<PyHestonCalibrationQuote>>,
        variables: Vec<Py<PyHestonCalibrationVariable>>,
        time_steps: usize,
        integration_intervals: usize,
        cutoff: f64,
    ) -> PyResult<Self> {
        let fourier = HestonFourierConfig::new(time_steps, integration_intervals, cutoff)
            .map_err(|e| invalid(py, e))?;
        let quotes = quotes.iter().map(|q| q.borrow(py).inner).collect();
        let variables = variables.iter().map(|v| v.borrow(py).inner).collect();
        HestonCalibrationProblem::new(model.inner.clone(), quotes, variables, fourier)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    #[getter]
    fn initial_parameters(&self) -> Vec<f64> {
        self.inner.initial_parameters()
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner
            .variables()
            .iter()
            .map(|v| v.parameter.name().into())
            .collect()
    }
    #[getter]
    fn maturity_count(&self) -> usize {
        self.inner.maturity_count()
    }
    fn evaluate(
        &self,
        py: Python<'_>,
        parameters: Vec<f64>,
    ) -> PyResult<PyHestonCalibrationEvaluation> {
        py.detach(|| self.inner.evaluate(&parameters))
            .map(|inner| PyHestonCalibrationEvaluation {
                inner,
                width: self.inner.variables().len(),
            })
            .map_err(|e| match e {
                pricing::rough_volatility::FourierError::InvalidInput(_) => invalid(py, e),
                _ => pricing_exception(e),
            })
    }
    #[pyo3(signature=(*,max_iterations=100,max_evaluations=150,residual_tolerance=1e-6,gradient_tolerance=1e-10,step_tolerance=1e-12))]
    fn calibrate(
        &self,
        py: Python<'_>,
        max_iterations: usize,
        max_evaluations: usize,
        residual_tolerance: f64,
        gradient_tolerance: f64,
        step_tolerance: f64,
    ) -> PyResult<PyHestonCalibrationResult> {
        let options = LeastSquaresOptions {
            max_iterations,
            max_evaluations,
            residual_tolerance,
            gradient_tolerance,
            step_tolerance,
            initial_damping: LeastSquaresOptions::default().initial_damping,
        };
        py.detach(|| self.inner.calibrate(options))
            .map(|inner| PyHestonCalibrationResult {
                inner,
                names: self.parameter_names(),
            })
            .map_err(|e| failure(py, e))
    }
}
