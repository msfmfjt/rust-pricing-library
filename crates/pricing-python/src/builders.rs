use std::sync::Arc;

use pricing::PricingRequest;
use pricing::core::{CurrencyId, CurveId, Date, EventId, PositiveF64, UnderlyingId};
use pricing::market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, EssviSlice, EssviSurface,
    LogLinearDiscountCurve, MarketContext, PhiSpec, StandardSsvi, SurfaceValidationTolerance,
    ThetaPchip,
};
use pricing::mc::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing::models::{
    Black76Spec, BlackScholesSpec, LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec,
};
use pricing::product::{DigitalPayout, DigitalSpec, EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};
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

/// Immutable discrete dividend event.
#[pyclass(frozen, name = "DividendEvent", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyDividendEvent {
    pub(crate) inner: DividendEvent,
}

#[pymethods]
impl PyDividendEvent {
    /// Build a fixed-cash dividend event.
    #[staticmethod]
    fn fixed_cash(py: Python<'_>, event_id: u32, ex_time: f64, amount: f64) -> PyResult<Self> {
        let event = EventId::new(event_id);
        let quote = DividendQuote::fixed_cash(amount, event).map_err(|error| {
            domain_error(
                py,
                "invalid_dividend_cash",
                "/market/discrete_dividends",
                error,
            )
        })?;
        DividendEvent::new(event, ex_time, quote)
            .map(|inner| Self { inner })
            .map_err(|error| {
                domain_error(
                    py,
                    "invalid_dividend_event",
                    "/market/discrete_dividends",
                    error,
                )
            })
    }

    /// Build a proportional dividend event.
    #[staticmethod]
    fn proportional(py: Python<'_>, event_id: u32, ex_time: f64, beta: f64) -> PyResult<Self> {
        let event = EventId::new(event_id);
        let quote = DividendQuote::proportional(beta, event).map_err(|error| {
            domain_error(
                py,
                "invalid_dividend_proportion",
                "/market/discrete_dividends",
                error,
            )
        })?;
        DividendEvent::new(event, ex_time, quote)
            .map(|inner| Self { inner })
            .map_err(|error| {
                domain_error(
                    py,
                    "invalid_dividend_event",
                    "/market/discrete_dividends",
                    error,
                )
            })
    }

    /// Build a fixed-cash plus proportional dividend event.
    #[staticmethod]
    fn fixed_cash_and_proportional(
        py: Python<'_>,
        event_id: u32,
        ex_time: f64,
        fixed_cash: f64,
        beta: f64,
    ) -> PyResult<Self> {
        let event = EventId::new(event_id);
        let quote = DividendQuote::fixed_cash_and_proportional(fixed_cash, beta, event).map_err(
            |error| {
                domain_error(
                    py,
                    "invalid_dividend_quote",
                    "/market/discrete_dividends",
                    error,
                )
            },
        )?;
        DividendEvent::new(event, ex_time, quote)
            .map(|inner| Self { inner })
            .map_err(|error| {
                domain_error(
                    py,
                    "invalid_dividend_event",
                    "/market/discrete_dividends",
                    error,
                )
            })
    }

    #[getter]
    fn event_id(&self) -> u32 {
        self.inner.event().get()
    }

    #[getter]
    fn ex_time(&self) -> f64 {
        self.inner.ex_time()
    }

    fn __repr__(&self) -> String {
        format!(
            "DividendEvent(event_id={}, ex_time={})",
            self.event_id(),
            self.ex_time()
        )
    }
}

/// Immutable eSSVI slice used to materialize Local Volatility grids.
#[pyclass(frozen, name = "EssviSlice", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyEssviSlice {
    inner: EssviSlice,
}

#[pymethods]
impl PyEssviSlice {
    #[new]
    fn new(py: Python<'_>, time: f64, theta: f64, psi: f64, rho_psi: f64) -> PyResult<Self> {
        EssviSlice::new(time, theta, psi, rho_psi)
            .map(|inner| Self { inner })
            .map_err(|error| domain_error(py, "invalid_essvi_slice", "/model/essvi/slices", error))
    }

    #[getter]
    fn time(&self) -> f64 {
        self.inner.time()
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    #[getter]
    fn psi(&self) -> f64 {
        self.inner.psi()
    }

    #[getter]
    fn rho_psi(&self) -> f64 {
        self.inner.rho_psi()
    }

    fn __repr__(&self) -> String {
        format!(
            "EssviSlice(time={}, theta={}, psi={}, rho_psi={})",
            self.time(),
            self.theta(),
            self.psi(),
            self.rho_psi()
        )
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

    /// Build a cash-or-nothing or asset-or-nothing digital call or put.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn digital(
        py: Python<'_>,
        underlying_id: u32,
        currency_id: u16,
        expiry: &Bound<'_, PyAny>,
        strike: f64,
        payout: f64,
        side: &str,
        payout_kind: &str,
    ) -> PyResult<Self> {
        let expiry = date_from_python(py, expiry, "/product/expiry")?;
        let side = option_side(py, side)?;
        let payout_kind = digital_payout(py, payout_kind)?;
        DigitalSpec::new(
            UnderlyingId::new(underlying_id),
            CurrencyId::new(currency_id),
            expiry,
            strike,
            payout,
            side,
            payout_kind,
        )
        .map(|spec| Self {
            inner: ProductSpec::Digital(spec),
        })
        .map_err(|error| domain_error(py, "invalid_digital", "/product", error))
    }

    fn __repr__(&self) -> String {
        format!("Product(type={:?})", product_name(&self.inner))
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
    #[pyo3(signature = (currency_id, underlying_id, spot, discount_curve, dividend_curve, *, discrete_dividends=None))]
    fn equity(
        py: Python<'_>,
        currency_id: u16,
        underlying_id: u32,
        spot: f64,
        discount_curve: &PyDiscountCurve,
        dividend_curve: &PyDiscountCurve,
        discrete_dividends: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let spot = PositiveF64::new(spot, "spot")
            .map_err(|error| domain_error(py, "invalid_spot", "/market/spot", error))?;
        let underlying = UnderlyingId::new(underlying_id);
        let discount = Arc::clone(&discount_curve.inner);
        let dividend = Arc::clone(&dividend_curve.inner);
        let forward = if let Some(discrete_dividends) = discrete_dividends {
            let dividends = dividend_events_from_python(py, discrete_dividends)?;
            EquityForward::with_discrete_dividends(underlying, spot, discount, dividend, dividends)
                .map_err(|error| {
                    domain_error(
                        py,
                        "invalid_discrete_dividends",
                        "/market/discrete_dividends",
                        error,
                    )
                })?
        } else {
            EquityForward::new(underlying, spot, discount, dividend)
        };
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

    /// Build a constant-volatility Black--76 model on the market forward.
    #[staticmethod]
    fn black_76(py: Python<'_>, volatility: f64) -> PyResult<Self> {
        Black76Spec::new(volatility)
            .map(|spec| Self {
                inner: ModelSpec::Black76(spec),
            })
            .map_err(|error| domain_error(py, "invalid_volatility", "/model/volatility", error))
    }

    /// Build a Local Volatility model from a row-major local-variance grid.
    #[staticmethod]
    fn local_volatility_from_grid(
        py: Python<'_>,
        time_nodes: &Bound<'_, PyAny>,
        log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        local_variances: &Bound<'_, PyAny>,
        floor: f64,
        cap: f64,
    ) -> PyResult<Self> {
        let time_nodes = copied_f64_array(py, time_nodes, "/model/local_variance_grid/time_nodes")?;
        let log_forward_moneyness_nodes = copied_f64_array(
            py,
            log_forward_moneyness_nodes,
            "/model/local_variance_grid/log_forward_moneyness_nodes",
        )?;
        let local_variances =
            copied_f64_array(py, local_variances, "/model/local_variance_grid/values")?;
        LocalVolatilitySpec::from_explicit_grid(
            time_nodes,
            log_forward_moneyness_nodes,
            local_variances,
            floor,
            cap,
        )
        .map(|spec| Self {
            inner: ModelSpec::LocalVolatility(spec),
        })
        .map_err(|error| {
            domain_error(
                py,
                "invalid_local_variance_grid",
                "/model/local_variance_grid",
                error,
            )
        })
    }

    /// Build a Local Volatility model from explicit local-variance and reporting-IV grids.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn local_volatility_from_grid_with_reporting_basis(
        py: Python<'_>,
        time_nodes: &Bound<'_, PyAny>,
        log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        local_variances: &Bound<'_, PyAny>,
        floor: f64,
        cap: f64,
        reporting_maturity_nodes: &Bound<'_, PyAny>,
        reporting_log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        reporting_implied_volatilities: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let time_nodes = copied_f64_array(py, time_nodes, "/model/local_variance_grid/time_nodes")?;
        let log_forward_moneyness_nodes = copied_f64_array(
            py,
            log_forward_moneyness_nodes,
            "/model/local_variance_grid/log_forward_moneyness_nodes",
        )?;
        let local_variances =
            copied_f64_array(py, local_variances, "/model/local_variance_grid/values")?;
        let reporting_maturity_nodes = copied_f64_array(
            py,
            reporting_maturity_nodes,
            "/model/reporting_iv_basis/maturity_nodes",
        )?;
        let reporting_log_forward_moneyness_nodes = copied_f64_array(
            py,
            reporting_log_forward_moneyness_nodes,
            "/model/reporting_iv_basis/log_forward_moneyness_nodes",
        )?;
        let reporting_implied_volatilities = copied_f64_array(
            py,
            reporting_implied_volatilities,
            "/model/reporting_iv_basis/implied_volatilities",
        )?;
        let basis = LocalVolatilityReportingBasis::new(
            reporting_maturity_nodes,
            reporting_log_forward_moneyness_nodes,
            reporting_implied_volatilities,
        )
        .map_err(|error| {
            domain_error(
                py,
                "invalid_reporting_iv_basis",
                "/model/reporting_iv_basis",
                error,
            )
        })?;
        LocalVolatilitySpec::from_explicit_grid(
            time_nodes,
            log_forward_moneyness_nodes,
            local_variances,
            floor,
            cap,
        )
        .map(|spec| Self {
            inner: ModelSpec::LocalVolatility(spec.with_reporting_iv_basis(basis)),
        })
        .map_err(|error| {
            domain_error(
                py,
                "invalid_local_variance_grid",
                "/model/local_variance_grid",
                error,
            )
        })
    }

    /// Build a Local Volatility model by sampling an eSSVI implied-volatility surface.
    #[staticmethod]
    fn local_volatility_from_essvi(
        py: Python<'_>,
        slices: &Bound<'_, PyAny>,
        terminal_theta_slope: f64,
        time_nodes: &Bound<'_, PyAny>,
        log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        floor: f64,
        cap: f64,
    ) -> PyResult<Self> {
        let surface = EssviSurface::new(
            essvi_slices_from_python(py, slices)?,
            terminal_theta_slope,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .map_err(|error| domain_error(py, "invalid_essvi_surface", "/model/essvi", error))?;
        let time_nodes = copied_f64_array(py, time_nodes, "/model/local_variance_grid/time_nodes")?;
        let log_forward_moneyness_nodes = copied_f64_array(
            py,
            log_forward_moneyness_nodes,
            "/model/local_variance_grid/log_forward_moneyness_nodes",
        )?;
        LocalVolatilitySpec::from_surface_with_reporting_basis(
            &surface,
            time_nodes.clone(),
            log_forward_moneyness_nodes.clone(),
            floor,
            cap,
            time_nodes,
            log_forward_moneyness_nodes,
        )
        .map(|spec| Self {
            inner: ModelSpec::LocalVolatility(spec),
        })
        .map_err(|error| {
            domain_error(
                py,
                "invalid_local_variance_grid",
                "/model/local_variance_grid",
                error,
            )
        })
    }

    /// Build a Local Volatility model by sampling a standard SSVI power-law surface.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn local_volatility_from_standard_ssvi_power_law(
        py: Python<'_>,
        theta_times: &Bound<'_, PyAny>,
        theta_values: &Bound<'_, PyAny>,
        terminal_theta_slope: f64,
        rho: f64,
        eta: f64,
        gamma: f64,
        time_nodes: &Bound<'_, PyAny>,
        log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        floor: f64,
        cap: f64,
    ) -> PyResult<Self> {
        local_volatility_from_standard_ssvi(
            py,
            theta_times,
            theta_values,
            terminal_theta_slope,
            rho,
            PhiSpec::PowerLaw { eta, gamma },
            time_nodes,
            log_forward_moneyness_nodes,
            floor,
            cap,
        )
    }

    /// Build a Local Volatility model by sampling a standard SSVI Heston-like surface.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn local_volatility_from_standard_ssvi_heston_like(
        py: Python<'_>,
        theta_times: &Bound<'_, PyAny>,
        theta_values: &Bound<'_, PyAny>,
        terminal_theta_slope: f64,
        rho: f64,
        lambda_: f64,
        time_nodes: &Bound<'_, PyAny>,
        log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        floor: f64,
        cap: f64,
    ) -> PyResult<Self> {
        local_volatility_from_standard_ssvi(
            py,
            theta_times,
            theta_values,
            terminal_theta_slope,
            rho,
            PhiSpec::HestonLike { lambda: lambda_ },
            time_nodes,
            log_forward_moneyness_nodes,
            floor,
            cap,
        )
    }

    fn __repr__(&self) -> String {
        format!("Model(type={:?})", self.inner.name())
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
    #[pyo3(signature = (*, delta=false, gamma_relative_bump=None, gamma_absolute_bump=None, vega=false, vega_kt_maturity_nodes=None, vega_kt_log_forward_moneyness_nodes=None, vega_kt_relative_density_threshold=None, vega_kt_full_bucket_covariance=false, smile_dynamics="sticky_log_moneyness", checkpoint_interval=None, aad_tile_capacity=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        delta: bool,
        gamma_relative_bump: Option<f64>,
        gamma_absolute_bump: Option<f64>,
        vega: bool,
        vega_kt_maturity_nodes: Option<&Bound<'_, PyAny>>,
        vega_kt_log_forward_moneyness_nodes: Option<&Bound<'_, PyAny>>,
        vega_kt_relative_density_threshold: Option<f64>,
        vega_kt_full_bucket_covariance: bool,
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
        let vega_kt = vega_kt_from_python(
            py,
            vega_kt_maturity_nodes,
            vega_kt_log_forward_moneyness_nodes,
            vega_kt_relative_density_threshold,
            vega_kt_full_bucket_covariance,
        )?;
        let smile_dynamics = smile_dynamics_from_str(py, smile_dynamics)?;
        RiskRequest::new(
            delta,
            gamma,
            vega,
            vega_kt,
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

fn copied_date_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    pointer: &str,
) -> PyResult<Vec<Date>> {
    let mut dates = Vec::new();
    let iter = value.try_iter().map_err(|error| {
        domain_error(
            py,
            "invalid_date_sequence",
            pointer,
            format!("expected a sequence of datetime.date or ISO date values: {error}"),
        )
    })?;
    for item in iter {
        dates.push(date_from_python(py, &item?, pointer)?);
    }
    Ok(dates)
}

fn vega_kt_from_python(
    py: Python<'_>,
    maturity_nodes: Option<&Bound<'_, PyAny>>,
    log_forward_moneyness_nodes: Option<&Bound<'_, PyAny>>,
    relative_density_threshold: Option<f64>,
    full_bucket_covariance: bool,
) -> PyResult<Option<VegaKtConfig>> {
    let requested = maturity_nodes.is_some()
        || log_forward_moneyness_nodes.is_some()
        || relative_density_threshold.is_some()
        || full_bucket_covariance;
    if !requested {
        return Ok(None);
    }
    let maturity_nodes = maturity_nodes.ok_or_else(|| {
        domain_error(
            py,
            "missing_vega_kt_maturity_nodes",
            "/risk/vega_kt/maturity_nodes",
            "VegaKT requires maturity nodes",
        )
    })?;
    let log_forward_moneyness_nodes = log_forward_moneyness_nodes.ok_or_else(|| {
        domain_error(
            py,
            "missing_vega_kt_log_forward_moneyness_nodes",
            "/risk/vega_kt/log_forward_moneyness_nodes",
            "VegaKT requires log-forward-moneyness nodes",
        )
    })?;
    let relative_density_threshold = relative_density_threshold.ok_or_else(|| {
        domain_error(
            py,
            "missing_vega_kt_relative_density_threshold",
            "/risk/vega_kt/relative_density_threshold",
            "VegaKT requires a relative density threshold",
        )
    })?;
    VegaKtConfig::new(
        copied_date_array(py, maturity_nodes, "/risk/vega_kt/maturity_nodes")?,
        copied_f64_array(
            py,
            log_forward_moneyness_nodes,
            "/risk/vega_kt/log_forward_moneyness_nodes",
        )?,
        relative_density_threshold,
        full_bucket_covariance,
    )
    .map(Some)
    .map_err(|error| domain_error(py, "invalid_vega_kt", "/risk/vega_kt", error))
}

#[allow(clippy::too_many_arguments)]
fn local_volatility_from_standard_ssvi(
    py: Python<'_>,
    theta_times: &Bound<'_, PyAny>,
    theta_values: &Bound<'_, PyAny>,
    terminal_theta_slope: f64,
    rho: f64,
    phi: PhiSpec,
    time_nodes: &Bound<'_, PyAny>,
    log_forward_moneyness_nodes: &Bound<'_, PyAny>,
    floor: f64,
    cap: f64,
) -> PyResult<PyModel> {
    let theta_curve = ThetaPchip::new(
        copied_f64_array(py, theta_times, "/model/standard_ssvi/theta_times")?,
        copied_f64_array(py, theta_values, "/model/standard_ssvi/theta_values")?,
        terminal_theta_slope,
    )
    .map_err(|error| domain_error(py, "invalid_theta_curve", "/model/standard_ssvi", error))?;
    let surface = StandardSsvi::new(
        theta_curve,
        rho,
        phi,
        SurfaceValidationTolerance::local_vol_vegakt_v1(),
    )
    .map_err(|error| domain_error(py, "invalid_standard_ssvi", "/model/standard_ssvi", error))?;
    let time_nodes = copied_f64_array(py, time_nodes, "/model/local_variance_grid/time_nodes")?;
    let log_forward_moneyness_nodes = copied_f64_array(
        py,
        log_forward_moneyness_nodes,
        "/model/local_variance_grid/log_forward_moneyness_nodes",
    )?;
    LocalVolatilitySpec::from_surface_with_reporting_basis(
        &surface,
        time_nodes.clone(),
        log_forward_moneyness_nodes.clone(),
        floor,
        cap,
        time_nodes,
        log_forward_moneyness_nodes,
    )
    .map(|spec| PyModel {
        inner: ModelSpec::LocalVolatility(spec),
    })
    .map_err(|error| {
        domain_error(
            py,
            "invalid_local_variance_grid",
            "/model/local_variance_grid",
            error,
        )
    })
}

fn essvi_slices_from_python(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<EssviSlice>> {
    let mut slices = Vec::new();
    let iter = value.try_iter().map_err(|error| {
        domain_error(
            py,
            "invalid_essvi_slice_sequence",
            "/model/essvi/slices",
            format!("expected a sequence of EssviSlice objects: {error}"),
        )
    })?;
    for item in iter {
        let item = item?;
        let slice = item.extract::<PyRef<'_, PyEssviSlice>>().map_err(|error| {
            domain_error(
                py,
                "invalid_essvi_slice",
                "/model/essvi/slices",
                format!("expected EssviSlice: {error}"),
            )
        })?;
        slices.push(slice.inner);
    }
    Ok(slices)
}

fn dividend_events_from_python(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<DividendEvent>> {
    let mut events = Vec::new();
    let iter = value.try_iter().map_err(|error| {
        domain_error(
            py,
            "invalid_dividend_event_sequence",
            "/market/discrete_dividends",
            format!("expected a sequence of DividendEvent objects: {error}"),
        )
    })?;
    for item in iter {
        let item = item?;
        let dividend = item
            .extract::<PyRef<'_, PyDividendEvent>>()
            .map_err(|error| {
                domain_error(
                    py,
                    "invalid_dividend_event",
                    "/market/discrete_dividends",
                    format!("expected DividendEvent: {error}"),
                )
            })?;
        events.push(dividend.inner);
    }
    Ok(events)
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

fn digital_payout(py: Python<'_>, value: &str) -> PyResult<DigitalPayout> {
    match value {
        "cash" => Ok(DigitalPayout::Cash),
        "asset" => Ok(DigitalPayout::Asset),
        _ => Err(domain_error(
            py,
            "invalid_digital_payout",
            "/product/payout_kind",
            format!("expected 'cash' or 'asset', received {value:?}"),
        )),
    }
}

const fn product_name(product: &ProductSpec) -> &'static str {
    match product {
        ProductSpec::EuropeanVanilla(_) => "european_vanilla",
        ProductSpec::Digital(_) => "digital",
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
