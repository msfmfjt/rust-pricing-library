use std::sync::Arc;

use pricing::PricingRequest;
use pricing::core::{CurrencyId, CurveId, Date, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump};
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::{PyValidationIssue, validation_exception};

/// Immutable log-linear discount-factor curve with flat-forward extrapolation.
#[pyclass(frozen, name = "DiscountCurve", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDiscountCurve {
    pub(crate) inner: Arc<LogLinearDiscountCurve>,
}

#[pymethods]
impl PyDiscountCurve {
    #[new]
    fn new(
        py: Python<'_>,
        curve_id: u32,
        times: &Bound<'_, PyAny>,
        discount_factors: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let times = copied_f64_array(py, times, "/times")?;
        let discount_factors = copied_f64_array(py, discount_factors, "/discount_factors")?;
        LogLinearDiscountCurve::new(CurveId::new(curve_id), times, discount_factors)
            .map(|curve| Self {
                inner: Arc::new(curve),
            })
            .map_err(|error| domain_error(py, "invalid_discount_curve", "", error))
    }

    #[getter]
    fn curve_id(&self) -> u32 {
        self.inner.id().get()
    }

    fn __repr__(&self) -> String {
        format!("DiscountCurve(curve_id={})", self.curve_id())
    }
}

/// Immutable product specification built through named factory methods.
#[pyclass(frozen, name = "Product", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyProduct {
    pub(crate) inner: ProductSpec,
}

#[pymethods]
impl PyProduct {
    /// Build a European vanilla call or put.
    #[staticmethod]
    fn european_vanilla(
        py: Python<'_>,
        underlying_id: u32,
        currency_id: u16,
        expiry: &Bound<'_, PyAny>,
        strike: f64,
        notional: f64,
        side: &str,
    ) -> PyResult<Self> {
        let expiry = date_from_python(py, expiry, "/product/expiry")?;
        let side = option_side(py, side)?;
        EuropeanVanillaSpec::new(
            UnderlyingId::new(underlying_id),
            CurrencyId::new(currency_id),
            expiry,
            strike,
            notional,
            side,
        )
        .map(|spec| Self {
            inner: ProductSpec::EuropeanVanilla(spec),
        })
        .map_err(|error| domain_error(py, "invalid_european_vanilla", "/product", error))
    }

    fn __repr__(&self) -> String {
        "Product(type='european_vanilla')".into()
    }
}

/// Immutable market context built through named factory methods.
#[pyclass(frozen, name = "Market", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMarket {
    pub(crate) inner: MarketContext,
}

#[pymethods]
impl PyMarket {
    /// Build a single-currency equity market with deterministic carry curves.
    #[staticmethod]
    fn equity(
        py: Python<'_>,
        currency_id: u16,
        underlying_id: u32,
        spot: f64,
        discount_curve: &PyDiscountCurve,
        dividend_curve: &PyDiscountCurve,
    ) -> PyResult<Self> {
        let spot = PositiveF64::new(spot, "spot")
            .map_err(|error| domain_error(py, "invalid_spot", "/market/spot", error))?;
        let forward = EquityForward::new(
            UnderlyingId::new(underlying_id),
            spot,
            Arc::clone(&discount_curve.inner),
            Arc::clone(&dividend_curve.inner),
        );
        Ok(Self {
            inner: MarketContext::Equity(EquityMarket::new(CurrencyId::new(currency_id), forward)),
        })
    }

    fn __repr__(&self) -> String {
        "Market(type='equity')".into()
    }
}

/// Immutable pricing-model specification.
#[pyclass(frozen, name = "Model", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyModel {
    pub(crate) inner: ModelSpec,
}

#[pymethods]
impl PyModel {
    /// Build a constant-volatility Black--Scholes model.
    #[staticmethod]
    fn black_scholes(py: Python<'_>, volatility: f64) -> PyResult<Self> {
        BlackScholesSpec::new(volatility)
            .map(|spec| Self {
                inner: ModelSpec::BlackScholes(spec),
            })
            .map_err(|error| domain_error(py, "invalid_volatility", "/model/volatility", error))
    }

    fn __repr__(&self) -> String {
        "Model(type='black_scholes')".into()
    }
}

/// Immutable Monte Carlo engine configuration.
#[pyclass(frozen, name = "Engine", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyEngine {
    pub(crate) inner: EngineConfig,
}

#[pymethods]
impl PyEngine {
    /// Build a counter-based Philox pseudo-Monte Carlo engine.
    #[staticmethod]
    #[pyo3(signature = (master_seed, independent_sampling_units, *, antithetic=false, brownian_bridge=false))]
    fn pseudo_monte_carlo(
        py: Python<'_>,
        master_seed: u64,
        independent_sampling_units: u64,
        antithetic: bool,
        brownian_bridge: bool,
    ) -> PyResult<Self> {
        PseudoMcConfig::new(
            master_seed,
            independent_sampling_units,
            VarianceReduction::new(antithetic, brownian_bridge),
        )
        .map(|config| Self {
            inner: EngineConfig::PseudoMonteCarlo(config),
        })
        .map_err(|error| domain_error(py, "invalid_pseudo_mc", "/engine", error))
    }

    /// Build a randomized Sobol QMC engine with independent scrambles.
    #[staticmethod]
    #[pyo3(signature = (points_per_scramble, master_scramble_seed, *, scramble_count=RqmcConfig::DEFAULT_SCRAMBLE_COUNT, antithetic=false, brownian_bridge=true))]
    fn randomized_quasi_monte_carlo(
        py: Python<'_>,
        points_per_scramble: u64,
        master_scramble_seed: u64,
        scramble_count: u32,
        antithetic: bool,
        brownian_bridge: bool,
    ) -> PyResult<Self> {
        RqmcConfig::new(
            points_per_scramble,
            scramble_count,
            master_scramble_seed,
            VarianceReduction::new(antithetic, brownian_bridge),
        )
        .map(|config| Self {
            inner: EngineConfig::RandomizedQuasiMonteCarlo(config),
        })
        .map_err(|error| domain_error(py, "invalid_rqmc", "/engine", error))
    }

    fn __repr__(&self) -> String {
        let name = match self.inner {
            EngineConfig::PseudoMonteCarlo(_) => "pseudo_monte_carlo",
            EngineConfig::RandomizedQuasiMonteCarlo(_) => "randomized_quasi_monte_carlo",
        };
        format!("Engine(type={name:?})")
    }
}

/// Requested Greeks and deterministic AAD/bump execution controls.
#[pyclass(frozen, name = "RiskRequest", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRiskRequest {
    pub(crate) inner: RiskRequest,
}

#[pymethods]
impl PyRiskRequest {
    #[new]
    #[pyo3(signature = (*, delta=false, gamma_relative_bump=None, gamma_absolute_bump=None, vega=false, smile_dynamics="sticky_log_moneyness", checkpoint_interval=None, aad_tile_capacity=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        delta: bool,
        gamma_relative_bump: Option<f64>,
        gamma_absolute_bump: Option<f64>,
        vega: bool,
        smile_dynamics: &str,
        checkpoint_interval: Option<u32>,
        aad_tile_capacity: Option<u32>,
    ) -> PyResult<Self> {
        if gamma_relative_bump.is_some() && gamma_absolute_bump.is_some() {
            return Err(domain_error(
                py,
                "ambiguous_gamma_bump",
                "/risk/gamma",
                "specify only one of gamma_relative_bump and gamma_absolute_bump",
            ));
        }
        let gamma = gamma_relative_bump
            .map(SpotBump::relative)
            .or_else(|| gamma_absolute_bump.map(SpotBump::absolute))
            .transpose()
            .map_err(|error| domain_error(py, "invalid_gamma_bump", "/risk/gamma", error))?
            .map(GammaConfig::new);
        let smile_dynamics = smile_dynamics_from_str(py, smile_dynamics)?;
        RiskRequest::new(
            delta,
            gamma,
            vega,
            None,
            smile_dynamics,
            checkpoint_interval,
            aad_tile_capacity,
        )
        .map(|inner| Self { inner })
        .map_err(|error| domain_error(py, "invalid_risk_request", "/risk", error))
    }

    fn __repr__(&self) -> String {
        "RiskRequest()".into()
    }
}

pub(crate) fn build_request(
    py: Python<'_>,
    valuation_date: &Bound<'_, PyAny>,
    product: &PyProduct,
    market: &PyMarket,
    model: &PyModel,
    engine: &PyEngine,
    risk: &PyRiskRequest,
) -> PyResult<PricingRequest> {
    let valuation_date = date_from_python(py, valuation_date, "/valuation_date")?;
    PricingRequest::new(
        valuation_date,
        product.inner.clone(),
        market.inner.clone(),
        model.inner.clone(),
        engine.inner,
        risk.inner.clone(),
    )
    .map_err(|error| domain_error(py, "invalid_pricing_request", "", error))
}

fn copied_f64_array(py: Python<'_>, value: &Bound<'_, PyAny>, pointer: &str) -> PyResult<Vec<f64>> {
    value.extract::<Vec<f64>>().map_err(|error| {
        domain_error(
            py,
            "invalid_dense_f64_array",
            pointer,
            format!("expected a one-dimensional numeric sequence: {error}"),
        )
    })
}

fn date_from_python(py: Python<'_>, value: &Bound<'_, PyAny>, pointer: &str) -> PyResult<Date> {
    let text = if value.cast::<PyString>().is_ok() {
        value.extract::<String>()?
    } else {
        let date_type = py.import("datetime")?.getattr("date")?;
        if !value.is_instance(&date_type)? {
            return Err(domain_error(
                py,
                "invalid_date_type",
                pointer,
                "expected datetime.date or an ISO YYYY-MM-DD string",
            ));
        }
        value.call_method0("isoformat")?.extract::<String>()?
    };
    text.parse::<Date>()
        .map_err(|error| domain_error(py, "invalid_date", pointer, error))
}

fn option_side(py: Python<'_>, value: &str) -> PyResult<OptionSide> {
    match value {
        "call" => Ok(OptionSide::Call),
        "put" => Ok(OptionSide::Put),
        _ => Err(domain_error(
            py,
            "invalid_option_side",
            "/product/side",
            format!("expected 'call' or 'put', received {value:?}"),
        )),
    }
}

fn smile_dynamics_from_str(py: Python<'_>, value: &str) -> PyResult<SmileDynamics> {
    match value {
        "sticky_log_moneyness" => Ok(SmileDynamics::StickyLogMoneyness),
        "sticky_strike" => Ok(SmileDynamics::StickyStrike),
        "sticky_delta" => Ok(SmileDynamics::StickyDelta),
        _ => Err(domain_error(
            py,
            "invalid_smile_dynamics",
            "/risk/smile_dynamics",
            format!(
                "expected 'sticky_log_moneyness', 'sticky_strike', or 'sticky_delta'; received {value:?}"
            ),
        )),
    }
}

fn domain_error(py: Python<'_>, code: &str, pointer: &str, error: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(pointer, code, error.to_string()),
    )
}
