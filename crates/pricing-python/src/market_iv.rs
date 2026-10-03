//! Explicit quote source shared by strict Dupire target construction and risks.
use crate::{PyModel, PyValidationIssue, validation_exception};
use pricing::market::{MARKET_IV_INTERPOLATION, MarketIvSurface};
use pricing::models::{LocalVolatilitySpec, ModelSpec};
use pyo3::prelude::*;

#[pyclass(frozen, name = "MarketIvSurface", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMarketIvSurface {
    pub(crate) inner: MarketIvSurface,
}
fn invalid(py: Python<'_>, error: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain("/market_iv", "invalid_market_iv_surface", error.to_string()),
    )
}
#[pymethods]
impl PyMarketIvSurface {
    #[new]
    fn new(
        py: Python<'_>,
        maturity_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        implied_volatilities: Vec<f64>,
    ) -> PyResult<Self> {
        py.detach(|| {
            MarketIvSurface::new(maturity_nodes, log_moneyness_nodes, implied_volatilities)
        })
        .map(|inner| Self { inner })
        .map_err(|e| invalid(py, e))
    }
    /// Natural-cubic total variance in log moneyness; linear total variance in
    /// time, with constant-IV time tails. Target samples must need no repair.
    #[pyo3(signature=(time_nodes, log_moneyness_nodes, *, floor=1e-8, cap=4.0))]
    fn local_volatility_model(
        &self,
        py: Python<'_>,
        time_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> PyResult<PyModel> {
        py.detach(|| {
            let grid =
                self.inner
                    .local_variance_grid(time_nodes, log_moneyness_nodes, floor, cap)?;
            LocalVolatilitySpec::from_explicit_grid(
                grid.time_nodes().to_vec(),
                grid.log_moneyness_nodes().to_vec(),
                grid.values().to_vec(),
                floor,
                cap,
            )
        })
        .map(|lv| PyModel {
            inner: ModelSpec::LocalVolatility(lv),
        })
        .map_err(|e| invalid(py, e))
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
    #[getter]
    fn interpolation(&self) -> &'static str {
        MARKET_IV_INTERPOLATION
    }
}
