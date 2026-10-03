//! Deterministic continuous-time Heston-family transform and price bindings.
use super::rough_volatility::PyRoughVolatilityModel;
use super::{PyValidationIssue, pricing_exception, validation_exception};
use pricing::rough_volatility::{
    Complex64, FourierError, HestonFourierConfig, HestonFourierGreeks, HestonFourierPlan,
    HestonFourierPrice,
};
use pyo3::prelude::*;

fn failure(py: Python<'_>, error: FourierError) -> PyErr {
    match error {
        FourierError::InvalidInput(_) => validation_exception(
            py,
            PyValidationIssue::domain(
                "/heston_fourier",
                "invalid_fourier_input",
                error.to_string(),
            ),
        ),
        _ => pricing_exception(error),
    }
}
#[pyclass(frozen, name = "HestonFourierPrice", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonFourierPrice {
    inner: HestonFourierPrice,
}
#[pymethods]
impl PyHestonFourierPrice {
    #[getter]
    fn call(&self) -> f64 {
        self.inner.call
    }
    #[getter]
    fn put(&self) -> f64 {
        self.inner.put
    }
    #[getter]
    fn quadrature_difference(&self) -> f64 {
        self.inner.quadrature_difference
    }
    #[getter]
    fn tail_indicator(&self) -> f64 {
        self.inner.tail_indicator
    }
}
/// Frozen forward-only Greeks; the model, maturity, strike and discount stay fixed.
#[pyclass(frozen, name = "HestonFourierGreeks", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonFourierGreeks {
    inner: HestonFourierGreeks,
}
#[pymethods]
impl PyHestonFourierGreeks {
    #[getter]
    fn price(&self) -> PyHestonFourierPrice {
        PyHestonFourierPrice {
            inner: self.inner.price,
        }
    }
    #[getter]
    fn call_forward_delta(&self) -> f64 {
        self.inner.call_forward_delta
    }
    #[getter]
    fn put_forward_delta(&self) -> f64 {
        self.inner.put_forward_delta
    }
    #[getter]
    fn forward_gamma(&self) -> f64 {
        self.inner.forward_gamma
    }
    #[getter]
    fn delta_quadrature_difference(&self) -> f64 {
        self.inner.delta_quadrature_difference
    }
    #[getter]
    fn gamma_quadrature_difference(&self) -> f64 {
        self.inner.gamma_quadrature_difference
    }
    #[getter]
    fn delta_tail_indicator(&self) -> f64 {
        self.inner.delta_tail_indicator
    }
    #[getter]
    fn gamma_tail_indicator(&self) -> f64 {
        self.inner.gamma_tail_indicator
    }
}
#[pyclass(frozen, name = "HestonFourierPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHestonFourierPlan {
    inner: HestonFourierPlan,
}
#[pymethods]
impl PyHestonFourierPlan {
    #[staticmethod]
    #[pyo3(signature=(model,maturity,*,time_steps=512,integration_intervals=512,cutoff=128.0))]
    fn compile(
        py: Python<'_>,
        model: &PyRoughVolatilityModel,
        maturity: f64,
        time_steps: usize,
        integration_intervals: usize,
        cutoff: f64,
    ) -> PyResult<Self> {
        let config = HestonFourierConfig::new(time_steps, integration_intervals, cutoff)
            .map_err(|e| failure(py, e))?;
        let model = model.inner.clone();
        py.detach(|| HestonFourierPlan::compile(model, maturity, config))
            .map(|inner| Self { inner })
            .map_err(|e| failure(py, e))
    }
    fn price(
        &self,
        py: Python<'_>,
        forward: f64,
        strike: f64,
        discount: f64,
    ) -> PyResult<PyHestonFourierPrice> {
        py.detach(|| self.inner.price(forward, strike, discount))
            .map(|inner| PyHestonFourierPrice { inner })
            .map_err(|e| failure(py, e))
    }
    fn price_and_greeks(
        &self,
        py: Python<'_>,
        forward: f64,
        strike: f64,
        discount: f64,
    ) -> PyResult<PyHestonFourierGreeks> {
        py.detach(|| self.inner.price_and_greeks(forward, strike, discount))
            .map(|inner| PyHestonFourierGreeks { inner })
            .map_err(|e| failure(py, e))
    }
    fn log_transform(&self, py: Python<'_>, real: f64, imag: f64) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.log_transform(Complex64::new(real, imag)))
            .map(|z| (z.re, z.im))
            .map_err(|e| failure(py, e))
    }
    fn characteristic_function(&self, py: Python<'_>, frequency: f64) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.characteristic_function(frequency))
            .map(|z| (z.re, z.im))
            .map_err(|e| failure(py, e))
    }
    #[getter]
    fn time_steps(&self) -> usize {
        self.inner.config().time_steps()
    }
    #[getter]
    fn integration_intervals(&self) -> usize {
        self.inner.config().integration_intervals()
    }
    #[getter]
    fn cutoff(&self) -> f64 {
        self.inner.config().cutoff()
    }
}
