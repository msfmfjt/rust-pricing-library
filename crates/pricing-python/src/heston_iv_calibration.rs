//! Black-IV objective and immutable SSVI target sampling.
use super::heston_calibration::{PyHestonCalibrationVariable, failure, invalid};
use super::pricing_exception;
use super::rough_volatility::PyRoughVolatilityModel;
use pricing::market::{PhiSpec, StandardSsvi, SurfaceValidationTolerance, ThetaPchip};
use pricing::rough_volatility::{
    HestonFourierConfig, HestonIvCalibrationEvaluation, HestonIvCalibrationProblem,
    HestonIvCalibrationQuote, HestonIvCalibrationResult, LeastSquaresOptions,
    LeastSquaresTermination,
};
use pyo3::prelude::*;
#[pyclass(frozen, name = "HestonIvCalibrationQuote", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvCalibrationQuote {
    inner: HestonIvCalibrationQuote,
}
#[pymethods]
impl PyHestonIvCalibrationQuote {
    /// Positive Black IV; rejects numerically unresolvable prices and low Vega.
    #[staticmethod]
    #[pyo3(signature=(maturity,forward,strike,discount,target_volatility,iv_scale=1.0,*,is_call=true))]
    fn create(
        maturity: f64,
        forward: f64,
        strike: f64,
        discount: f64,
        target_volatility: f64,
        iv_scale: f64,
        is_call: bool,
    ) -> PyResult<Self> {
        let inner = HestonIvCalibrationQuote {
            maturity,
            forward,
            strike,
            discount,
            target_volatility,
            iv_scale,
            is_call,
        };
        Python::attach(|py| inner.to_price_quote().map_err(|e| invalid(py, e)))?;
        Ok(Self { inner })
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
    fn target_volatility(&self) -> f64 {
        self.inner.target_volatility
    }
    #[getter]
    fn iv_scale(&self) -> f64 {
        self.inner.iv_scale
    }
    #[getter]
    fn is_call(&self) -> bool {
        self.inner.is_call
    }
}
#[pyclass(frozen, name = "HestonIvCalibrationEvaluation", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvCalibrationEvaluation {
    inner: HestonIvCalibrationEvaluation,
    width: usize,
}
#[pymethods]
impl PyHestonIvCalibrationEvaluation {
    #[getter]
    fn model_prices(&self) -> Vec<f64> {
        self.inner.model_prices.clone()
    }
    #[getter]
    fn model_implied_volatilities(&self) -> Vec<f64> {
        self.inner.model_implied_volatilities.clone()
    }
    #[getter]
    fn model_vegas(&self) -> Vec<f64> {
        self.inner.model_vegas.clone()
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
    fn price_quadrature_differences(&self) -> Vec<f64> {
        self.inner.price_quadrature_differences.clone()
    }
    #[getter]
    fn price_tail_indicators(&self) -> Vec<f64> {
        self.inner.price_tail_indicators.clone()
    }
}
#[pyclass(frozen, name = "HestonIvCalibrationResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvCalibrationResult {
    pub(super) inner: HestonIvCalibrationResult,
    pub(super) names: Vec<String>,
}
#[pymethods]
impl PyHestonIvCalibrationResult {
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
    fn evaluation(&self) -> PyHestonIvCalibrationEvaluation {
        PyHestonIvCalibrationEvaluation {
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
#[pyclass(frozen, name = "HestonIvCalibrationProblem", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonIvCalibrationProblem {
    pub(super) inner: HestonIvCalibrationProblem,
}
#[pymethods]
impl PyHestonIvCalibrationProblem {
    #[staticmethod]
    #[pyo3(signature=(model,quotes,variables,*,time_steps=512,integration_intervals=512,cutoff=128.0))]
    fn compile(
        py: Python<'_>,
        model: &PyRoughVolatilityModel,
        quotes: Vec<Py<PyHestonIvCalibrationQuote>>,
        variables: Vec<Py<PyHestonCalibrationVariable>>,
        time_steps: usize,
        integration_intervals: usize,
        cutoff: f64,
    ) -> PyResult<Self> {
        let fourier = HestonFourierConfig::new(time_steps, integration_intervals, cutoff)
            .map_err(|e| invalid(py, e))?;
        let quotes = quotes.iter().map(|q| q.borrow(py).inner).collect();
        let variables = variables.iter().map(|v| v.borrow(py).inner).collect();
        HestonIvCalibrationProblem::new(model.inner.clone(), quotes, variables, fourier)
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
    ) -> PyResult<PyHestonIvCalibrationEvaluation> {
        py.detach(|| self.inner.evaluate(&parameters))
            .map(|inner| PyHestonIvCalibrationEvaluation {
                inner,
                width: self.inner.variables().len(),
            })
            .map_err(|e| match e {
                pricing::rough_volatility::FourierError::InvalidInput(_) => invalid(py, e),
                _ => pricing_exception(e),
            })
    }
    fn validate_grid(
        &self,
        py: Python<'_>,
        parameters: Vec<f64>,
        policy: &super::heston_iv_refinement::PyHestonIvRefinementOptions,
    ) -> PyResult<super::heston_iv_refinement::PyHestonIvGridValidation> {
        let policy = policy.inner;
        py.detach(|| self.inner.validate_grid(&parameters, policy))
            .map(|inner| super::heston_iv_refinement::PyHestonIvGridValidation { inner })
            .map_err(|e| super::heston_iv_refinement::refinement_failure(py, e))
    }
    #[pyo3(signature=(policy,*,max_iterations=100,max_evaluations=150,residual_tolerance=1e-6,gradient_tolerance=1e-10,step_tolerance=1e-12))]
    fn calibrate_refined(
        &self,
        policy: &super::heston_iv_refinement::PyHestonIvRefinementOptions,
        max_iterations: usize,
        max_evaluations: usize,
        residual_tolerance: f64,
        gradient_tolerance: f64,
        step_tolerance: f64,
    ) -> PyResult<super::heston_iv_refinement::PyHestonIvRefinementResult> {
        let options = LeastSquaresOptions {
            max_iterations,
            max_evaluations,
            residual_tolerance,
            gradient_tolerance,
            step_tolerance,
            ..LeastSquaresOptions::default()
        };
        let policy = policy.inner;
        Python::attach(|py| {
            py.detach(|| self.inner.calibrate_refined(options, policy))
                .map(
                    |inner| super::heston_iv_refinement::PyHestonIvRefinementResult {
                        inner,
                        names: self.parameter_names(),
                    },
                )
                .map_err(|e| super::heston_iv_refinement::refinement_failure(py, e))
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
    ) -> PyResult<PyHestonIvCalibrationResult> {
        let options = LeastSquaresOptions {
            max_iterations,
            max_evaluations,
            residual_tolerance,
            gradient_tolerance,
            step_tolerance,
            initial_damping: LeastSquaresOptions::default().initial_damping,
        };
        py.detach(|| self.inner.calibrate(options))
            .map(|inner| PyHestonIvCalibrationResult {
                inner,
                names: self.parameter_names(),
            })
            .map_err(|e| failure(py, e))
    }
}

/// Immutable standard SSVI surface for generating OTM IV calibration targets.
#[pyclass(frozen, name = "HestonCalibrationSsviSurface", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonCalibrationSsviSurface {
    inner: StandardSsvi,
}
impl PyHestonCalibrationSsviSurface {
    fn build(
        py: Python<'_>,
        theta_times: Vec<f64>,
        theta_values: Vec<f64>,
        terminal_theta_slope: f64,
        rho: f64,
        phi: PhiSpec,
    ) -> PyResult<Self> {
        let theta = ThetaPchip::new(theta_times, theta_values, terminal_theta_slope)
            .map_err(|e| invalid(py, e))?;
        let inner = StandardSsvi::new(
            theta,
            rho,
            phi,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .map_err(|e| invalid(py, e))?;
        Ok(Self { inner })
    }
}
#[pymethods]
impl PyHestonCalibrationSsviSurface {
    #[staticmethod]
    fn power_law(
        py: Python<'_>,
        theta_times: Vec<f64>,
        theta_values: Vec<f64>,
        terminal_theta_slope: f64,
        rho: f64,
        eta: f64,
        gamma: f64,
    ) -> PyResult<Self> {
        Self::build(
            py,
            theta_times,
            theta_values,
            terminal_theta_slope,
            rho,
            PhiSpec::PowerLaw { eta, gamma },
        )
    }
    #[staticmethod]
    fn heston_like(
        py: Python<'_>,
        theta_times: Vec<f64>,
        theta_values: Vec<f64>,
        terminal_theta_slope: f64,
        rho: f64,
        lambda_: f64,
    ) -> PyResult<Self> {
        Self::build(
            py,
            theta_times,
            theta_values,
            terminal_theta_slope,
            rho,
            PhiSpec::HestonLike { lambda: lambda_ },
        )
    }
    #[pyo3(signature=(maturity,forward,strike,discount,iv_scale=1.0))]
    fn quote(
        &self,
        py: Python<'_>,
        maturity: f64,
        forward: f64,
        strike: f64,
        discount: f64,
        iv_scale: f64,
    ) -> PyResult<PyHestonIvCalibrationQuote> {
        HestonIvCalibrationQuote::from_ssvi(
            &self.inner,
            maturity,
            forward,
            strike,
            discount,
            iv_scale,
        )
        .map(|inner| PyHestonIvCalibrationQuote { inner })
        .map_err(|e| invalid(py, e))
    }
}
