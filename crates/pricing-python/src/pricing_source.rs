//! Source adapters own their inputs and never retain a compiled workspace.
use crate::builders::{PyEngine, PyMarket, PyModel, date_from_python};
use crate::hull_white::{PyHullWhiteLsvTarget, PyHullWhiteModel};
use crate::local_correlation::PyLocalCorrelationConfig;
use crate::multi_asset::PyCorrelationSchedule;
use crate::multi_asset_lsv::extract_lsv_config;
use crate::product_graph::PyGraphProduct;
use crate::source_config::{PyBuehlerDividendModel, invalid};
use crate::specification::PyPricingSpecification;
use pricing::mc::ExecutionPolicy;
use pricing::multi_asset::MultiAssetHullWhiteConfig;
use pricing::{
    MultiAssetModelSpec, MultiAssetSpecification, PricingSource, StochasticDividendSpecification,
};
use pyo3::prelude::*;

#[pyclass(frozen, name = "MultiAssetHullWhiteConfig")]
#[derive(Clone, Debug)]
pub struct PyMultiAssetHullWhiteConfig {
    // Missing targets expand using the asset count at composition time, never
    // the number of rate correlations (which includes volatility drivers).
    rate_model: pricing::models::HullWhite1Factor,
    rate_correlations: Vec<f64>,
    lsv_targets: Option<Vec<Option<pricing::mc::hull_white::HullWhiteLsvTarget>>>,
}
#[pymethods]
impl PyMultiAssetHullWhiteConfig {
    #[new]
    #[pyo3(signature=(rate_model, rate_correlations, *, lsv_targets=None))]
    fn new(
        py: Python<'_>,
        rate_model: &PyHullWhiteModel,
        rate_correlations: Vec<f64>,
        lsv_targets: Option<Vec<Option<Py<PyHullWhiteLsvTarget>>>>,
    ) -> Self {
        Self {
            rate_model: rate_model.inner.clone(),
            rate_correlations,
            lsv_targets: lsv_targets.map(|targets| {
                targets
                    .into_iter()
                    .map(|t| t.map(|t| t.borrow(py).inner.clone()))
                    .collect()
            }),
        }
    }
}

#[pyclass(frozen, name = "MultiAssetModelSpec")]
#[derive(Clone, Debug)]
pub struct PyMultiAssetModelSpec {
    inner: MultiAssetModelSpec,
}
#[pymethods]
impl PyMultiAssetModelSpec {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(models, correlation, *, lsv=None, driver_correlations=None, hull_white=None, local_correlation=None))]
    fn new(
        py: Python<'_>,
        models: Vec<Py<PyModel>>,
        correlation: &PyCorrelationSchedule,
        lsv: Option<Vec<Option<Py<PyAny>>>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
        hull_white: Option<&PyMultiAssetHullWhiteConfig>,
        local_correlation: Option<&PyLocalCorrelationConfig>,
    ) -> PyResult<Self> {
        let models: Vec<_> = models
            .into_iter()
            .map(|m| m.borrow(py).inner.clone())
            .collect();
        let mut inner = MultiAssetModelSpec::new(models, correlation.inner.clone());
        if let Some(configs) = lsv {
            inner.lsv = configs
                .into_iter()
                .map(|c| c.map(|c| extract_lsv_config(py, c.bind(py))).transpose())
                .collect::<PyResult<_>>()?;
        }
        inner.driver_correlations = driver_correlations;
        inner.hull_white = hull_white.map(|c| MultiAssetHullWhiteConfig {
            rate_model: c.rate_model.clone(),
            rate_correlations: c.rate_correlations.clone(),
            lsv_targets: c
                .lsv_targets
                .clone()
                .unwrap_or_else(|| vec![None; inner.models.len()]),
        });
        inner.local_correlation =
            local_correlation.map(|c| (c.inner.clone(), c.extensions.clone()));
        Ok(Self { inner })
    }
}

#[pyclass(frozen, name = "MultiAssetSpecification")]
#[derive(Clone, Debug)]
pub struct PyMultiAssetSpecification {
    inner: MultiAssetSpecification,
}
#[pymethods]
impl PyMultiAssetSpecification {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(valuation_date, product, markets, model, engine, *, maximum_step, worker_threads, reduction_block_size=None))]
    fn new(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &PyGraphProduct,
        markets: Vec<Py<PyMarket>>,
        model: &PyMultiAssetModelSpec,
        engine: &PyEngine,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<Self> {
        MultiAssetSpecification::new(
            date_from_python(py, valuation_date, "/valuation_date")?,
            product.inner.clone(),
            markets
                .into_iter()
                .map(|m| m.borrow(py).inner.equity().clone())
                .collect(),
            model.inner.clone(),
            engine.inner,
            ExecutionPolicy::new(worker_threads, reduction_block_size).map_err(invalid)?,
            maximum_step,
        )
        .map(|inner| Self { inner })
        .map_err(invalid)
    }
}

#[pyclass(frozen, name = "StochasticDividendSpecification")]
#[derive(Clone, Debug)]
pub struct PyStochasticDividendSpecification {
    inner: StochasticDividendSpecification,
}
#[pymethods]
impl PyStochasticDividendSpecification {
    #[new]
    #[pyo3(signature=(equity, dividends, *, maximum_step, volatility_correlations=None, rate_correlation=None))]
    fn new(
        equity: &PyPricingSpecification,
        dividends: &PyBuehlerDividendModel,
        maximum_step: f64,
        volatility_correlations: Option<Vec<f64>>,
        rate_correlation: Option<f64>,
    ) -> PyResult<Self> {
        StochasticDividendSpecification::new(
            equity.inner.clone(),
            dividends.inner,
            volatility_correlations.unwrap_or_default(),
            rate_correlation,
            maximum_step,
        )
        .map(|inner| Self { inner })
        .map_err(invalid)
    }
}

pub(super) fn extract_source(source: &Bound<'_, PyAny>) -> PyResult<PricingSource> {
    if let Ok(s) = source.extract::<PyRef<'_, PyPricingSpecification>>() {
        return Ok(s.inner.clone().into());
    }
    if let Ok(s) = source.extract::<PyRef<'_, PyMultiAssetSpecification>>() {
        return Ok(s.inner.clone().into());
    }
    if let Ok(s) = source.extract::<PyRef<'_, PyStochasticDividendSpecification>>() {
        return Ok(s.inner.clone().into());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "source must be PricingSpecification, MultiAssetSpecification or StochasticDividendSpecification",
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMultiAssetHullWhiteConfig>()?;
    module.add_class::<PyMultiAssetModelSpec>()?;
    module.add_class::<PyMultiAssetSpecification>()?;
    module.add_class::<PyStochasticDividendSpecification>()?;
    Ok(())
}
