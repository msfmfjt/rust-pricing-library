use crate::builders::{
    PyDiscountCurve, PyEngine, PyMarket, PyProduct, date_from_python, domain_error,
};
use crate::{diagnostics::PyDiagnosticEstimate, pricing_exception};
use pricing::core::PositiveF64;
use pricing::cross_currency::{CrossCurrencyEquityOption, EquityFxDynamics, EquityFxSettlement};
use pricing::mc::ExecutionPolicy;
use pricing::product::ProductSpec;
use pyo3::prelude::*;

/// European call or put fixing both equity and FX at expiry, paid in product currency.
#[pyclass(frozen, name = "CrossCurrencyEquityOption")]
pub struct PyCrossCurrencyEquityOption {
    inner: CrossCurrencyEquityOption,
}

#[pymethods]
impl PyCrossCurrencyEquityOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(valuation_date, product, foreign_market, settlement_curve, *, settlement, fx_spot,
        equity_volatility, fx_volatility, correlation, fixed_fx_rate=None, payment_date=None))]
    fn new(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &PyProduct,
        foreign_market: &PyMarket,
        settlement_curve: &PyDiscountCurve,
        settlement: &str,
        fx_spot: f64,
        equity_volatility: f64,
        fx_volatility: f64,
        correlation: f64,
        fixed_fx_rate: Option<f64>,
        payment_date: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let invalid = |message: &str| {
            domain_error(
                py,
                "invalid_cross_currency_option",
                "/cross_currency",
                message,
            )
        };
        let ProductSpec::EuropeanVanilla(product) = &product.inner else {
            return Err(invalid(
                "cross-currency pricing requires a European vanilla option",
            ));
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
        let dynamics = EquityFxDynamics::new(equity_volatility, fx_volatility, correlation)
            .map_err(|e| invalid(&e.to_string()))?;
        let as_of = date_from_python(py, valuation_date, "/cross_currency/valuation_date")?;
        let payment = payment_date
            .map(|d| date_from_python(py, d, "/cross_currency/payment_date"))
            .transpose()?
            .unwrap_or(product.expiry());
        CrossCurrencyEquityOption::new(
            as_of,
            product.clone(),
            foreign_market.inner.equity(),
            settlement_curve.inner.clone(),
            fx_spot,
            dynamics,
            settlement,
            payment,
        )
        .map(|inner| Self { inner })
        .map_err(|e| invalid(&e.to_string()))
    }

    /// Closed-form present value in settlement currency.
    fn analytical_price(&self) -> PyResult<f64> {
        self.inner.analytical_price().map_err(pricing_exception)
    }

    /// Joint equity/FX Monte Carlo estimate in settlement currency.
    #[pyo3(signature=(engine, *, worker_threads=1, reduction_block_size=None))]
    fn price_mc(
        &self,
        py: Python<'_>,
        engine: &PyEngine,
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> PyResult<PyDiagnosticEstimate> {
        let execution = ExecutionPolicy::new(worker_threads, reduction_block_size)
            .map_err(|e| domain_error(py, "invalid_execution_policy", "/execution", e))?;
        py.detach(|| self.inner.price_mc(engine.inner, execution))
            .map(PyDiagnosticEstimate::from_estimate)
            .map_err(pricing_exception)
    }
}
