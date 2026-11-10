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

fn parse_settlement(
    py: Python<'_>,
    settlement: &str,
    fixed_fx_rate: Option<f64>,
) -> PyResult<EquityFxSettlement> {
    let invalid = |message: &str| {
        domain_error(
            py,
            "invalid_cross_currency_option",
            "/cross_currency",
            message,
        )
    };
    Ok(match (settlement, fixed_fx_rate) {
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
    })
}

fn parse_model(m: &Bound<'_, PyAny>) -> PyResult<EquityModelSpec> {
    if let Ok(m) = m.extract::<PyRef<PyModel>>() {
        Ok(EquityModelSpec::Standard(m.inner.clone()))
    } else {
        Ok(m.extract::<PyRef<PyEquityModelSpec>>()?.inner.clone())
    }
}

fn payment_dates(
    py: Python<'_>,
    schedule: Vec<(Py<PyAny>, Py<PyAny>)>,
) -> PyResult<Vec<(pricing::core::Date, pricing::core::Date)>> {
    let path = "/cross_currency/payment_fx_fixings";
    schedule
        .iter()
        .map(|(payment, fixing)| {
            Ok((
                date_from_python(py, payment.bind(py), path)?,
                date_from_python(py, fixing.bind(py), path)?,
            ))
        })
        .collect()
}

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
        equity_model=None, maximum_step=0.02, dividend_model=None))]
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
        dividend_model: Option<&crate::source_config::PyBuehlerDividendModel>,
    ) -> PyResult<Self> {
        let invalid = |message: &str| {
            domain_error(
                py,
                "invalid_cross_currency_option",
                "/cross_currency",
                message,
            )
        };
        let settlement = parse_settlement(py, settlement, fixed_fx_rate)?;
        let model = match (equity_volatility, equity_model) {
            (Some(sigma), None) => EquityModelSpec::Standard(ModelSpec::BlackScholes(
                BlackScholesSpec::new(sigma).map_err(|e| invalid(&e.to_string()))?,
            )),
            (None, Some(m)) => parse_model(m)?,
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
        CrossCurrencyEquityOption::from_model_with_dividends(
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
            dividend_model.map(|d| d.inner),
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
        let dates = payment_dates(py, schedule)?;
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

/// Foreign stocks in one currency, observed jointly against a shared FX rate.
#[pyclass(frozen, name = "CrossCurrencyEquityBasket")]
pub struct PyCrossCurrencyEquityBasket {
    inner: pricing::cross_currency::CrossCurrencyEquityBasket,
}
#[pymethods]
impl PyCrossCurrencyEquityBasket {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(valuation_date,product,foreign_markets,settlement_curve,*,equity_models,settlement,fx_spot,fx_volatility,correlations,fixed_fx_rate=None,rates=None,maximum_step=0.02,dividend_models=None))]
    fn new(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &crate::product_graph::PyGraphProduct,
        foreign_markets: Vec<Py<PyMarket>>,
        settlement_curve: &PyDiscountCurve,
        equity_models: Vec<Py<PyAny>>,
        settlement: &str,
        fx_spot: f64,
        fx_volatility: f64,
        correlations: Vec<Vec<f64>>,
        fixed_fx_rate: Option<f64>,
        rates: Option<(
            Py<crate::hull_white::PyHullWhiteModel>,
            Py<crate::hull_white::PyHullWhiteModel>,
        )>,
        maximum_step: f64,
        dividend_models: Option<Vec<Option<Py<crate::source_config::PyBuehlerDividendModel>>>>,
    ) -> PyResult<Self> {
        let invalid = |message: String| {
            domain_error(
                py,
                "invalid_cross_currency_basket",
                "/cross_currency/basket",
                message,
            )
        };
        if equity_models.len() != foreign_markets.len()
            || dividend_models
                .as_ref()
                .is_some_and(|d| d.len() != foreign_markets.len())
        {
            return Err(invalid("markets, equity models and supplied dividend models must follow graph underlying order".into()));
        }
        let stocks = foreign_markets
            .iter()
            .zip(equity_models)
            .enumerate()
            .map(|(i, (market, m))| {
                Ok(pricing::cross_currency::ForeignEquity {
                    market: market.borrow(py).inner.equity().clone(),
                    model: parse_model(m.bind(py))?,
                    dividend_model: dividend_models
                        .as_ref()
                        .and_then(|ds| ds[i].as_ref())
                        .map(|d| d.borrow(py).inner),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let rate_models =
            rates.map(|(d, f)| (d.borrow(py).inner.clone(), f.borrow(py).inner.clone()));
        pricing::cross_currency::CrossCurrencyEquityBasket::new(
            date_from_python(py, valuation_date, "/cross_currency/valuation_date")?,
            product.inner.clone(),
            stocks,
            settlement_curve.inner.clone(),
            parse_settlement(py, settlement, fixed_fx_rate)?,
            fx_spot,
            fx_volatility,
            correlations,
            rate_models,
            maximum_step,
        )
        .map(|inner| Self { inner })
        .map_err(|e| invalid(e.to_string()))
    }

    fn with_payment_fx_fixings(
        &self,
        py: Python<'_>,
        schedule: Vec<(Py<PyAny>, Py<PyAny>)>,
    ) -> PyResult<Self> {
        self.inner
            .clone()
            .with_payment_fx_fixings(&payment_dates(py, schedule)?)
            .map(|inner| Self { inner })
            .map_err(|e| {
                domain_error(
                    py,
                    "invalid_payment_fx_fixings",
                    "/cross_currency/payment_fx_fixings",
                    e,
                )
            })
    }

    #[pyo3(signature=(engine,*,worker_threads=1,reduction_block_size=None))]
    fn price_mc(
        &self,
        py: Python<'_>,
        engine: &PyEngine,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<PyDiagnosticEstimate> {
        let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| domain_error(py, "invalid_execution_policy", "/execution", e))?;
        py.detach(|| self.inner.price_mc(engine.inner, policy))
            .map(PyDiagnosticEstimate::from_estimate)
            .map_err(pricing_exception)
    }
}
