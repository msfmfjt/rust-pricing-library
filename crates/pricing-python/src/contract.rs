//! Python construction and replay delegate to the canonical Rust contract.
use crate::builders::date_from_python;
use crate::pricing_task::PyPricingTask;
use crate::product_graph::PyGraphProduct;
use pricing::contract::{self as c, Contract, ContractError, InstrumentBuilder};
use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::product::{BarrierStyle, OptionSide};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
fn error(e: ContractError) -> PyErr {
    let code = match &e {
        ContractError::Invalid(_) => "invalid_contract",
        ContractError::MissingFixing(_) => "missing_fixing",
        ContractError::Unsupported(_) => "unsupported_contract",
        ContractError::EventConflict(_) => "contract_event_conflict",
        ContractError::Graph(_) => "contract_graph",
        ContractError::Wire(_) => "contract_wire",
        _ => "contract_error",
    };
    Python::attach(|py| {
        crate::errors::ErrorMetadata::domain(code, Some("contract".into())).attach(
            py,
            PyValueError::new_err(e.to_string()),
            None,
        )
    })
}
fn invalid(s: &str) -> PyErr {
    error(ContractError::Invalid(s.into()))
}
fn date(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Date> {
    date_from_python(py, value, "/contract/date")
}
fn dates(py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<Vec<Date>> {
    values.iter().map(|v| date(py, v.bind(py))).collect()
}
fn side(value: &str) -> PyResult<OptionSide> {
    match value {
        "call" => Ok(OptionSide::Call),
        "put" => Ok(OptionSide::Put),
        _ => Err(invalid("side must be call or put")),
    }
}
fn aggregate(value: &str) -> PyResult<c::Aggregate> {
    match value {
        "arithmetic_average" => Ok(c::Aggregate::ArithmeticAverage),
        "geometric_average" => Ok(c::Aggregate::GeometricAverage),
        "running_minimum" => Ok(c::Aggregate::RunningMinimum),
        "running_maximum" => Ok(c::Aggregate::RunningMaximum),
        _ => Err(invalid("unsupported path aggregation")),
    }
}
#[pyclass(frozen, name = "Contract")]
#[derive(Clone, Debug)]
pub struct PyContract {
    inner: Contract,
}
#[pymethods]
impl PyContract {
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Contract::from_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(error)
    }
    #[staticmethod]
    fn json_schema() -> &'static str {
        Contract::json_schema()
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(error)
    }
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.inner
            .fingerprint()
            .map(|v| v.to_string())
            .map_err(error)
    }
    fn analysis_json(&self) -> PyResult<String> {
        self.inner.analysis_json().map_err(error)
    }
    fn observations_json(&self) -> PyResult<String> {
        self.inner.observations_json().map_err(error)
    }
    #[staticmethod]
    fn bundle(py: Python<'_>, id: String, contracts: Vec<Py<PyContract>>) -> PyResult<Self> {
        Contract::bundle(
            id,
            contracts
                .iter()
                .map(|c| c.borrow(py).inner.clone())
                .collect(),
        )
        .map(|inner| Self { inner })
        .map_err(error)
    }
    fn reversed(&self, id: String) -> PyResult<Self> {
        self.inner
            .reversed(id)
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn scaled(&self, id: String, factor: f64) -> PyResult<Self> {
        self.inner
            .scaled(id, factor)
            .map(|inner| Self { inner })
            .map_err(error)
    }
    #[pyo3(signature=(event_id, observation_id, value, *, effective_date=None))]
    fn fixing_event_json(
        &self,
        py: Python<'_>,
        event_id: String,
        observation_id: String,
        value: f64,
        effective_date: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        if !value.is_finite() {
            return Err(invalid("fixing must be finite"));
        }
        let observation = self
            .inner
            .observations()
            .iter()
            .find(|o| o.id == observation_id)
            .ok_or_else(|| invalid("unknown observation ID"))?
            .clone();
        let effective_date = effective_date
            .map(|v| date(py, v))
            .transpose()?
            .unwrap_or(observation.date);
        c::LifecycleEvent {
            id: event_id,
            effective_date,
            event: c::Event::Fixing { observation, value },
        }
        .to_json()
        .map_err(error)
    }
    fn advance(
        &self,
        py: Python<'_>,
        events_json: &str,
        as_of: &Bound<'_, PyAny>,
    ) -> PyResult<PyContractLifecycle> {
        let events = c::parse_events_json(events_json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map_err(error)?;
        let as_of = date(py, as_of)?;
        let source = self.inner.clone();
        py.detach(move || source.advance(&events, as_of))
            .map(|inner| PyContractLifecycle { inner })
            .map_err(error)
    }
    fn to_graphs(
        &self,
        py: Python<'_>,
        as_of: &Bound<'_, PyAny>,
        underlying_ids: Vec<u32>,
    ) -> PyResult<Vec<PyGraphProduct>> {
        let as_of = date(py, as_of)?;
        let source = self.inner.clone();
        let lowered = py
            .detach(move || {
                source.compile_mc(
                    as_of,
                    underlying_ids.into_iter().map(UnderlyingId::new).collect(),
                )
            })
            .map_err(error)?;
        Ok(lowered
            .groups()
            .iter()
            .map(|g| PyGraphProduct {
                inner: g.product().clone(),
            })
            .collect())
    }
    #[pyo3(signature=(source, *, operation="evaluate", gamma_relative_bump=None))]
    fn pricing_task(
        &self,
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        operation: &str,
        gamma_relative_bump: Option<f64>,
    ) -> PyResult<PyPricingTask> {
        let source = crate::pricing_source::extract_source(source)?;
        let operation = crate::pricing_task::operation(operation, gamma_relative_bump)?;
        let contract = self.inner.clone();
        let task = py
            .detach(move || match source {
                pricing::PricingSource::SingleAsset(s) => contract
                    .compile_mc(s.valuation_date(), vec![s.product().underlying()])?
                    .pricing_task(&s, operation),
                pricing::PricingSource::MultiAsset(s) => contract
                    .compile_mc(s.valuation_date(), s.product().underlyings().to_vec())?
                    .multi_asset_task(&s, operation),
                pricing::PricingSource::StochasticDividend(s) => contract
                    .compile_mc(
                        s.equity().valuation_date(),
                        vec![s.equity().product().underlying()],
                    )?
                    .stochastic_dividend_task(&s, operation),
                _ => Err(ContractError::Unsupported("pricing source".into())),
            })
            .map_err(error)?;
        Ok(PyPricingTask { inner: task })
    }
}
#[pyclass(frozen, name = "ContractLifecycle")]
#[derive(Clone, Debug)]
pub struct PyContractLifecycle {
    inner: c::LifecycleResult,
}
#[pymethods]
impl PyContractLifecycle {
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        c::LifecycleResult::from_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(error)
    }
    #[getter]
    fn residual_contract(&self) -> PyContract {
        PyContract {
            inner: self.inner.residual_contract.clone(),
        }
    }
    #[getter]
    fn as_of(&self) -> String {
        self.inner.as_of().to_string()
    }
    #[getter]
    fn applied_event_ids(&self) -> Vec<String> {
        self.inner.applied_event_ids.clone()
    }
    fn determined_cashflows_json(&self) -> PyResult<String> {
        self.inner.determined_cashflows_json().map_err(error)
    }
    fn outstanding_payments_json(&self) -> PyResult<String> {
        self.inner.outstanding_payments_json().map_err(error)
    }
    fn remaining_observations_json(&self) -> PyResult<String> {
        self.inner.remaining_observations_json().map_err(error)
    }
    fn barrier_states_json(&self) -> PyResult<String> {
        self.inner.barrier_states_json().map_err(error)
    }
    fn advance(
        &self,
        py: Python<'_>,
        events_json: &str,
        as_of: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let events = c::parse_events_json(events_json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map_err(error)?;
        let as_of = date(py, as_of)?;
        let result = self.inner.clone();
        py.detach(move || result.advance(&events, as_of))
            .map(|inner| Self { inner })
            .map_err(error)
    }
}
#[pyclass(frozen, name = "ContractBuilder")]
#[derive(Clone, Debug)]
pub struct PyContractBuilder {
    inner: InstrumentBuilder,
}
#[pymethods]
impl PyContractBuilder {
    #[new]
    fn new(
        id: String,
        currency_id: u16,
        underlying_id: u32,
        fixing_source: String,
    ) -> PyResult<Self> {
        InstrumentBuilder::new(
            id,
            CurrencyId::new(currency_id),
            UnderlyingId::new(underlying_id),
            fixing_source,
        )
        .map(|inner| Self { inner })
        .map_err(error)
    }
    fn forward(
        &self,
        py: Python<'_>,
        strike: f64,
        notional: f64,
        observation_date: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
    ) -> PyResult<PyContract> {
        self.inner
            .forward(
                strike,
                notional,
                date(py, observation_date)?,
                date(py, payment_date)?,
            )
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
    #[allow(clippy::too_many_arguments)]
    fn european(
        &self,
        py: Python<'_>,
        side: &str,
        strike: f64,
        notional: f64,
        expiry: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
    ) -> PyResult<PyContract> {
        self.inner
            .european(
                self::side(side)?,
                strike,
                notional,
                date(py, expiry)?,
                date(py, payment_date)?,
            )
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
    #[allow(clippy::too_many_arguments)]
    fn digital(
        &self,
        py: Python<'_>,
        side: &str,
        strike: f64,
        payout: f64,
        expiry: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
    ) -> PyResult<PyContract> {
        self.inner
            .digital(
                self::side(side)?,
                strike,
                payout,
                date(py, expiry)?,
                date(py, payment_date)?,
            )
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
    #[allow(clippy::too_many_arguments)]
    fn asian(
        &self,
        py: Python<'_>,
        aggregation: &str,
        side: &str,
        strike: f64,
        notional: f64,
        observation_dates: Vec<Py<PyAny>>,
        payment_date: &Bound<'_, PyAny>,
    ) -> PyResult<PyContract> {
        self.inner
            .asian(
                aggregate(aggregation)?,
                self::side(side)?,
                strike,
                notional,
                dates(py, observation_dates)?,
                date(py, payment_date)?,
            )
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(notional, fixed_rate, observation_dates, payment_dates, year_fractions, *, pay_fixed=true))]
    fn swap(
        &self,
        py: Python<'_>,
        notional: f64,
        fixed_rate: f64,
        observation_dates: Vec<Py<PyAny>>,
        payment_dates: Vec<Py<PyAny>>,
        year_fractions: Vec<f64>,
        pay_fixed: bool,
    ) -> PyResult<PyContract> {
        if observation_dates.len() != payment_dates.len()
            || observation_dates.len() != year_fractions.len()
        {
            return Err(invalid("swap schedule lengths differ"));
        }
        let obs = dates(py, observation_dates)?;
        let pay = dates(py, payment_dates)?;
        let periods = obs
            .into_iter()
            .zip(pay)
            .zip(year_fractions)
            .map(
                |((observation_date, payment_date), year_fraction)| c::SwapPeriod {
                    observation_date,
                    payment_date,
                    year_fraction,
                },
            )
            .collect();
        self.inner
            .swap(notional, fixed_rate, periods, pay_fixed)
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
    #[allow(clippy::too_many_arguments)]
    fn discrete_barrier(
        &self,
        py: Python<'_>,
        side: &str,
        strike: f64,
        notional: f64,
        expiry: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
        monitoring_dates: Vec<Py<PyAny>>,
        direction: &str,
        style: &str,
        level: f64,
        rebate: f64,
    ) -> PyResult<PyContract> {
        let direction = match direction {
            "up" => c::BarrierDirection::Up,
            "down" => c::BarrierDirection::Down,
            _ => return Err(invalid("direction must be up or down")),
        };
        let style = match style {
            "knock_in" => BarrierStyle::KnockIn,
            "knock_out" => BarrierStyle::KnockOut,
            _ => return Err(invalid("style must be knock_in or knock_out")),
        };
        self.inner
            .discrete_barrier(c::DiscreteBarrierTerms {
                side: self::side(side)?,
                strike,
                notional,
                expiry: date(py, expiry)?,
                payment_date: date(py, payment_date)?,
                monitoring_dates: dates(py, monitoring_dates)?,
                direction,
                style,
                level,
                rebate,
            })
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
    #[allow(clippy::too_many_arguments)]
    fn autocall(
        &self,
        py: Python<'_>,
        notional: f64,
        initial_spot: f64,
        protection_level: f64,
        observation_dates: Vec<Py<PyAny>>,
        payment_dates: Vec<Py<PyAny>>,
        call_levels: Vec<f64>,
        coupon_levels: Vec<f64>,
        coupon_rates: Vec<f64>,
    ) -> PyResult<PyContract> {
        let n = observation_dates.len();
        if [
            payment_dates.len(),
            call_levels.len(),
            coupon_levels.len(),
            coupon_rates.len(),
        ]
        .iter()
        .any(|m| *m != n)
        {
            return Err(invalid("autocall schedule lengths differ"));
        }
        let obs = dates(py, observation_dates)?;
        let pay = dates(py, payment_dates)?;
        let periods = (0..n)
            .map(|i| c::AutocallPeriod {
                observation_date: obs[i],
                payment_date: pay[i],
                call_level: call_levels[i],
                coupon_level: coupon_levels[i],
                coupon_rate: coupon_rates[i],
            })
            .collect();
        self.inner
            .autocall(notional, initial_spot, protection_level, periods)
            .map(|inner| PyContract { inner })
            .map_err(error)
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyContract>()?;
    module.add_class::<PyContractLifecycle>()?;
    module.add_class::<PyContractBuilder>()?;
    Ok(())
}
