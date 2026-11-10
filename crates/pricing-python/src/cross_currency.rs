use crate::builders::{
    PyDiscountCurve, PyEngine, PyLsmConfig, PyMarket, PyModel, PyProduct, date_from_python,
    domain_error,
};
use crate::source_config::PyEquityModelSpec;
use crate::{diagnostics::PyDiagnosticEstimate, pricing_exception};
use pricing::EquityModelSpec;
use pricing::core::PositiveF64;
use pricing::cross_currency::{CrossCurrencyEquityOption, EquityFxSettlement};
use pricing::mc::ExecutionPolicy;
use pricing::models::{BlackScholesSpec, ModelSpec};
use pyo3::prelude::*;

/// A single-equity option or payoff graph settled in a different currency.
#[pyclass(frozen, name = "CrossCurrencyEquityOption")]
pub struct PyCrossCurrencyEquityOption {
    inner: CrossCurrencyEquityOption,
}

#[pymethods]
impl PyCrossCurrencyEquityOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(valuation_date, product, foreign_market, settlement_curve, *, settlement, fx_spot,
        equity_volatility=None, fx_volatility, correlation, fixed_fx_rate=None, payment_date=None,
        equity_model=None, maximum_step=0.02))]
    fn new(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &PyProduct,
        foreign_market: &PyMarket,
        settlement_curve: &PyDiscountCurve,
        settlement: &str,
        fx_spot: f64,
        equity_volatility: Option<f64>,
        fx_volatility: f64,
        correlation: f64,
        fixed_fx_rate: Option<f64>,
        payment_date: Option<&Bound<'_, PyAny>>,
        equity_model: Option<&Bound<'_, PyAny>>,
        maximum_step: f64,
    ) -> PyResult<Self> {
        let invalid = |message: &str| {
            domain_error(
                py,
                "invalid_cross_currency_option",
                "/cross_currency",
                message,
            )
        };
        let settlement = match (settlement, fixed_fx_rate) {
            ("quanto", Some(rate)) => EquityFxSettlement::Quanto {
                fixed_fx_rate: PositiveF64::new(rate, "fixed_fx_rate")
                    .map_err(|e| invalid(&e.to_string()))?,
            },
            ("quanto", None) => return Err(invalid("quanto settlement requires fixed_fx_rate")),
            ("floating_fx", None) => EquityFxSettlement::FloatingFx,
            ("composite", None) => EquityFxSettlement::Composite,
            ("floating_fx" | "composite", Some(_)) => {
                return Err(invalid("fixed_fx_rate applies only to quanto settlement"));
            }
            _ => {
                return Err(invalid(
                    "settlement must be quanto, floating_fx or composite",
                ));
            }
        };
        let model = match (equity_volatility, equity_model) {
            (Some(sigma), None) => EquityModelSpec::Standard(ModelSpec::BlackScholes(
                BlackScholesSpec::new(sigma).map_err(|e| invalid(&e.to_string()))?,
            )),
            (None, Some(m)) => {
                if let Ok(m) = m.extract::<PyRef<PyModel>>() {
                    EquityModelSpec::Standard(m.inner.clone())
                } else {
                    m.extract::<PyRef<PyEquityModelSpec>>()?.inner.clone()
                }
            }
            _ => {
                return Err(invalid(
                    "supply exactly one of equity_model and equity_volatility",
                ));
            }
        };
        let as_of = date_from_python(py, valuation_date, "/cross_currency/valuation_date")?;
        let payment = payment_date
            .map(|d| date_from_python(py, d, "/cross_currency/payment_date"))
            .transpose()?;
        CrossCurrencyEquityOption::from_model(
            as_of,
            product.inner.clone(),
            foreign_market.inner.equity(),
            settlement_curve.inner.clone(),
            fx_spot,
            model,
            fx_volatility,
            correlation,
            settlement,
            payment,
            maximum_step,
        )
        .map(|inner| Self { inner })
        .map_err(|e| invalid(&e.to_string()))
    }

    /// Closed-form present value in settlement currency.
    fn analytical_price(&self) -> PyResult<f64> {
        self.inner.analytical_price().map_err(pricing_exception)
    }

    /// Brownian correlation order is equity, quoted FX, domestic rate, foreign rate.
    #[pyo3(signature=(domestic, foreign, *, correlations=None))]
    fn with_hull_white_rates(
        &self,
        py: Python<'_>,
        domestic: &crate::hull_white::PyHullWhiteModel,
        foreign: &crate::hull_white::PyHullWhiteModel,
        correlations: Option<[[f64; 4]; 4]>,
    ) -> PyResult<Self> {
        self.inner
            .clone()
            .with_hull_white_rates(domestic.inner.clone(), foreign.inner.clone(), correlations)
            .map(|inner| Self { inner })
            .map_err(|e| {
                domain_error(
                    py,
                    "invalid_cross_currency_rates",
                    "/cross_currency/rates",
                    e,
                )
            })
    }

    /// Return a floating-FX graph option with the given (payment, FX fixing) dates.
    fn with_payment_fx_fixings(
        &self,
        py: Python<'_>,
        schedule: Vec<(Py<PyAny>, Py<PyAny>)>,
    ) -> PyResult<Self> {
        let path = "/cross_currency/payment_fx_fixings";
        let dates = schedule
            .iter()
            .map(|(payment, fixing)| {
                Ok((
                    date_from_python(py, payment.bind(py), path)?,
                    date_from_python(py, fixing.bind(py), path)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .clone()
            .with_payment_fx_fixings(&dates)
            .map(|inner| Self { inner })
            .map_err(|e| domain_error(py, "invalid_payment_fx_fixings", path, e))
    }

    /// Monte Carlo estimate in settlement currency, with native LSM for scheduled exercise.
    #[pyo3(signature=(engine, *, worker_threads=1, reduction_block_size=None, lsm=None))]
    fn price_mc(
        &self,
        py: Python<'_>,
        engine: &PyEngine,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
        lsm: Option<&PyLsmConfig>,
    ) -> PyResult<PyDiagnosticEstimate> {
        let execution = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| domain_error(py, "invalid_execution_policy", "/execution", e))?;
        let lsm = lsm.map(|config| config.inner.clone());
        py.detach(|| self.inner.price_mc_with_lsm(engine.inner, execution, lsm))
            .map(PyDiagnosticEstimate::from_estimate)
            .map_err(pricing_exception)
    }
}
