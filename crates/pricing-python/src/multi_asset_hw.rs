//! Immutable multi-asset HW calibration and paired-risk snapshots.
use crate::diagnostics::PyDiagnosticEstimate;
use pricing::mc::hull_white::CalibratedHullWhiteLsv;
use pricing::multi_asset::{MultiAssetHullWhiteCurveRisk, MultiAssetHullWhiteLsvRisk};
use pyo3::prelude::*;

#[pyclass(frozen, name = "MultiAssetHullWhiteCalibration", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetHullWhiteCalibration {
    #[pyo3(get)]
    volatility_factor_count: usize,
    #[pyo3(get)]
    time_nodes: Vec<f64>,
    #[pyo3(get)]
    log_moneyness_nodes: Vec<f64>,
    #[pyo3(get)]
    squared_leverage: Vec<f64>,
    #[pyo3(get)]
    minimum_effective_samples: Vec<f64>,
    #[pyo3(get)]
    fallback_nodes: Vec<usize>,
    #[pyo3(get)]
    mean_relative_discount: Vec<f64>,
    #[pyo3(get)]
    mean_discounted_normalized_equity: Vec<f64>,
    #[pyo3(get)]
    conditional_second_moments: Vec<f64>,
    #[pyo3(get)]
    rate_corrections: Vec<f64>,
    #[pyo3(get)]
    retains_reverse_trace: bool,
}
impl PyMultiAssetHullWhiteCalibration {
    pub(crate) fn new(c: &CalibratedHullWhiteLsv, count: usize) -> Self {
        Self {
            volatility_factor_count: count,
            time_nodes: c.surface.times().to_vec(),
            log_moneyness_nodes: c.surface.log_nodes().to_vec(),
            squared_leverage: c.surface.squared_leverage().to_vec(),
            minimum_effective_samples: c
                .diagnostics
                .iter()
                .map(|d| d.minimum_effective_samples)
                .collect(),
            fallback_nodes: c.diagnostics.iter().map(|d| d.fallback_nodes).collect(),
            mean_relative_discount: c
                .diagnostics
                .iter()
                .map(|d| d.mean_relative_discount)
                .collect(),
            mean_discounted_normalized_equity: c
                .diagnostics
                .iter()
                .map(|d| d.mean_discounted_normalized_equity)
                .collect(),
            conditional_second_moments: c.conditional_second_moments.to_vec(),
            rate_corrections: c.rate_corrections.to_vec(),
            retains_reverse_trace: c.retains_reverse_trace(),
        }
    }
}

#[pyclass(frozen, name = "MultiAssetHullWhiteLsvRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetHullWhiteLsvRisk {
    pub(crate) inner: MultiAssetHullWhiteLsvRisk,
}
#[pymethods]
impl PyMultiAssetHullWhiteLsvRisk {
    #[getter]
    fn forward_log_density_adjoints(&self) -> Vec<f64> {
        self.inner.forward_log_density_adjoints.clone()
    }
    #[getter]
    fn density_standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.density_standard_errors.clone()
    }
    #[getter]
    fn vega_kt_raw(&self) -> Option<Vec<f64>> {
        self.inner.vega_kt_raw.clone()
    }
    #[getter]
    fn vega_kt_market_scaled(&self) -> Option<Vec<f64>> {
        self.inner.vega_kt_market_scaled.clone()
    }
    #[getter]
    fn vega_kt_standard_errors(&self) -> Option<Vec<f64>> {
        self.inner.vega_kt_standard_errors.clone()
    }
    #[getter]
    fn vega_kt_maturity_nodes(&self) -> Vec<f64> {
        self.inner.vega_kt_maturity_nodes.clone()
    }
    #[getter]
    fn vega_kt_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.vega_kt_log_moneyness_nodes.clone()
    }
    #[getter]
    fn parallel_vega(&self) -> Option<f64> {
        self.inner.parallel_vega
    }
    #[getter]
    fn parallel_vega_standard_error(&self) -> Option<f64> {
        self.inner.parallel_vega_standard_error
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }
}

#[pyclass(frozen, name = "MultiAssetHullWhiteCurveRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetHullWhiteCurveRisk {
    pub(crate) inner: MultiAssetHullWhiteCurveRisk,
}
#[pymethods]
impl PyMultiAssetHullWhiteCurveRisk {
    #[getter]
    fn discount_time_nodes(&self) -> Vec<f64> {
        self.inner.discount_time_nodes.clone()
    }
    #[getter]
    fn discount_log_df_adjoints(&self) -> Vec<PyDiagnosticEstimate> {
        self.inner
            .discount_log_df_adjoints
            .iter()
            .copied()
            .map(PyDiagnosticEstimate::from_estimate)
            .collect()
    }
    #[getter]
    fn discount_node_dv01(&self) -> Vec<PyDiagnosticEstimate> {
        self.inner
            .discount_node_dv01
            .iter()
            .copied()
            .map(PyDiagnosticEstimate::from_estimate)
            .collect()
    }
    #[getter]
    fn dividend_time_nodes(&self) -> Vec<Vec<f64>> {
        self.inner.dividend_time_nodes.clone()
    }
    #[getter]
    fn dividend_log_df_adjoints(&self) -> Vec<Vec<PyDiagnosticEstimate>> {
        self.inner
            .dividend_log_df_adjoints
            .iter()
            .map(|row| {
                row.iter()
                    .copied()
                    .map(PyDiagnosticEstimate::from_estimate)
                    .collect()
            })
            .collect()
    }
}
