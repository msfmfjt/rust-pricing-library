//! Immutable source configuration; validation and numerical policy belong to Rust.
use crate::builders::{PyLsmConfig, PyModel};
use crate::errors;
use crate::hull_white::{PyHullWhiteLsvTarget, PyHullWhiteModel, PyRoughBergomiModel};
use crate::rough_volatility::PyRoughVolatilityModel;
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::{Bergomi1Factor, Bergomi2Factor, BuehlerDividendModel, ModelSpec};
use pricing::{
    EquityModelSpec, PricingModelSpec, PricingNumerics, RateModelSpec, VolatilityFactorSpec,
};
use pyo3::prelude::*;

pub(super) fn invalid(error: impl std::error::Error + 'static) -> PyErr {
    errors::exception(error, errors::Stage::Validation, None)
}

fn local_volatility(model: &PyModel) -> PyResult<pricing::models::LocalVolatilitySpec> {
    match &model.inner {
        ModelSpec::LocalVolatility(target) => Ok(target.clone()),
        _ => Err(invalid(pricing::SpecificationValidationError::Invalid {
            field: "model.equity.target",
            message: "a local-volatility Model is required".into(),
        })),
    }
}

#[pyclass(frozen, name = "VolatilityFactorSpec")]
#[derive(Clone, Debug)]
pub struct PyVolatilityFactorSpec {
    inner: VolatilityFactorSpec,
}
#[pymethods]
impl PyVolatilityFactorSpec {
    #[staticmethod]
    fn bergomi(mean_reversion: f64, vol_of_vol: f64, correlation: f64) -> PyResult<Self> {
        Bergomi1Factor::new(mean_reversion, vol_of_vol, correlation)
            .map(|factor| Self {
                inner: VolatilityFactorSpec::Bergomi(factor),
            })
            .map_err(invalid)
    }
    #[staticmethod]
    fn bergomi_two_factor(
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
    ) -> PyResult<Self> {
        Bergomi2Factor::new(
            mean_reversions,
            vol_of_vol,
            mixing_weight,
            spot_correlations,
            factor_correlation,
        )
        .map(|factor| Self {
            inner: VolatilityFactorSpec::BergomiTwoFactor(factor),
        })
        .map_err(invalid)
    }
    #[staticmethod]
    fn rough_bergomi(model: &PyRoughBergomiModel) -> Self {
        Self {
            inner: VolatilityFactorSpec::RoughBergomi(model.inner),
        }
    }
}

#[pyclass(frozen, name = "LsvParticleConfig")]
#[derive(Clone, Debug)]
pub struct PyLsvParticleConfig {
    inner: LsvParticleConfig,
}
#[pymethods]
impl PyLsvParticleConfig {
    #[new]
    #[pyo3(signature=(*, particle_count, calibration_seed, log_bandwidth, minimum_effective_samples, retain_reverse_trace=false))]
    fn new(
        particle_count: usize,
        calibration_seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        retain_reverse_trace: bool,
    ) -> PyResult<Self> {
        LsvParticleConfig::new(
            particle_count,
            calibration_seed,
            log_bandwidth,
            minimum_effective_samples,
            retain_reverse_trace,
        )
        .map(|inner| Self { inner })
        .map_err(invalid)
    }
}

#[pyclass(frozen, name = "EquityModelSpec")]
#[derive(Clone, Debug)]
pub struct PyEquityModelSpec {
    inner: EquityModelSpec,
}
#[pymethods]
impl PyEquityModelSpec {
    #[staticmethod]
    fn standard(model: &PyModel) -> Self {
        Self {
            inner: EquityModelSpec::Standard(model.inner.clone()),
        }
    }
    #[staticmethod]
    fn stochastic_volatility(
        initial_volatility: f64,
        factor: &PyVolatilityFactorSpec,
    ) -> PyResult<Self> {
        // The same scalar constructor is used by PricingSpecification::new.
        pricing::models::BlackScholesSpec::new(initial_volatility).map_err(invalid)?;
        Ok(Self {
            inner: EquityModelSpec::StochasticVolatility {
                initial_volatility,
                factor: factor.inner.clone(),
            },
        })
    }
    #[staticmethod]
    #[pyo3(signature=(target, factor, particles, *, forward_density=None))]
    fn local_stochastic_volatility(
        target: &PyModel,
        factor: &PyVolatilityFactorSpec,
        particles: &PyLsvParticleConfig,
        forward_density: Option<&PyHullWhiteLsvTarget>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: EquityModelSpec::LocalStochasticVolatility {
                target: local_volatility(target)?,
                factor: factor.inner.clone(),
                particles: particles.inner.clone(),
                forward_density: forward_density.map(|t| t.inner.clone()),
            },
        })
    }
    #[staticmethod]
    fn rough_volatility(model: &PyRoughVolatilityModel) -> Self {
        Self {
            inner: EquityModelSpec::RoughVolatility(model.inner.clone()),
        }
    }
    #[staticmethod]
    fn rough_local_stochastic_volatility(
        target: &PyModel,
        model: &PyRoughVolatilityModel,
        particles: &PyLsvParticleConfig,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: EquityModelSpec::RoughLocalStochasticVolatility {
                target: local_volatility(target)?,
                model: model.inner.clone(),
                particles: particles.inner.clone(),
            },
        })
    }
}

#[pyclass(frozen, name = "RateModelSpec")]
#[derive(Clone, Debug)]
pub struct PyRateModelSpec {
    inner: RateModelSpec,
}
#[pymethods]
impl PyRateModelSpec {
    #[staticmethod]
    fn deterministic() -> Self {
        Self {
            inner: RateModelSpec::Deterministic,
        }
    }
    #[staticmethod]
    #[pyo3(signature=(model, *, equity_rate_correlation, vol_rate_correlations=None))]
    fn hull_white(
        model: &PyHullWhiteModel,
        equity_rate_correlation: f64,
        vol_rate_correlations: Option<Vec<f64>>,
    ) -> Self {
        Self {
            inner: RateModelSpec::HullWhite {
                model: model.inner.clone(),
                equity_rate_correlation,
                vol_rate_correlations: vol_rate_correlations.unwrap_or_default(),
            },
        }
    }
}

#[pyclass(frozen, name = "PricingModelSpec")]
#[derive(Clone, Debug)]
pub struct PyPricingModelSpec {
    pub(super) inner: PricingModelSpec,
}
#[pymethods]
impl PyPricingModelSpec {
    #[new]
    #[pyo3(signature=(equity, *, rates=None))]
    fn new(equity: &PyEquityModelSpec, rates: Option<&PyRateModelSpec>) -> Self {
        Self {
            inner: PricingModelSpec::new(
                equity.inner.clone(),
                rates
                    .map(|r| r.inner.clone())
                    .unwrap_or(RateModelSpec::Deterministic),
            ),
        }
    }
}

#[pyclass(frozen, name = "PricingNumerics")]
#[derive(Clone, Debug)]
pub struct PyPricingNumerics {
    pub(super) inner: PricingNumerics,
}
#[pymethods]
impl PyPricingNumerics {
    #[new]
    #[pyo3(signature=(*, worker_threads, reduction_block_size=None, maximum_step=None, lsm=None))]
    fn new(
        worker_threads: u32,
        reduction_block_size: Option<u64>,
        maximum_step: Option<f64>,
        lsm: Option<&PyLsmConfig>,
    ) -> PyResult<Self> {
        let mut inner = PricingNumerics::new(
            ExecutionPolicy::new(worker_threads, reduction_block_size).map_err(invalid)?,
        );
        inner.maximum_step = maximum_step;
        inner.lsm = lsm.map(|l| l.inner.clone());
        Ok(Self { inner })
    }
}

#[pyclass(frozen, name = "BuehlerDividendModel")]
#[derive(Clone, Debug)]
pub struct PyBuehlerDividendModel {
    pub(super) inner: BuehlerDividendModel,
}
#[pymethods]
impl PyBuehlerDividendModel {
    #[new]
    #[pyo3(signature=(*, mean_reversion, equity_linkage, dividend_volatility, equity_dividend_correlation))]
    fn new(
        mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
    ) -> PyResult<Self> {
        BuehlerDividendModel::new(
            mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        )
        .map(|inner| Self { inner })
        .map_err(invalid)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyVolatilityFactorSpec>()?;
    module.add_class::<PyLsvParticleConfig>()?;
    module.add_class::<PyEquityModelSpec>()?;
    module.add_class::<PyRateModelSpec>()?;
    module.add_class::<PyPricingModelSpec>()?;
    module.add_class::<PyPricingNumerics>()?;
    module.add_class::<PyBuehlerDividendModel>()?;
    Ok(())
}
