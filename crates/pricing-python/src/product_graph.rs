//! Python graph construction delegates every validation to the Rust domain.
use crate::builders::date_from_python;
use crate::*;
use pricing::core::{CurrencyId, NodeId, UnderlyingId};
use pricing::product::{CompactC2Smoothing, GraphProduct, SourceGraphBuilder, SourceOpcode};

fn invalid(py: Python<'_>, error: impl std::fmt::Display) -> PyErr {
    let issue = PyValidationIssue::wire(
        &WireError::DomainAt {
            pointer: "/product".into(),
            message: error.to_string(),
        },
        "product_graph",
    );
    let mut issue = issue;
    issue.schema_version = 1;
    validation_exception(py, issue)
}
fn wire_error(py: Python<'_>, error: WireError) -> PyErr {
    let mut issue = PyValidationIssue::wire(&error, "product_graph");
    issue.schema_version = match error {
        WireError::UnsupportedSchemaVersion(version) => version,
        _ => 1,
    };
    validation_exception(py, issue)
}

fn graph_date(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    pointer: &str,
) -> PyResult<pricing::core::Date> {
    date_from_python(py, value, pointer).map_err(|error| {
        let mut issue = PyValidationIssue::domain(pointer, "invalid_date", error.to_string());
        issue.document_kind = "product_graph".into();
        issue.schema_version = 1;
        validation_exception(py, issue)
    })
}
#[pyclass(frozen, name = "GraphProduct", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGraphProduct {
    pub(crate) inner: GraphProduct,
}
#[pymethods]
impl PyGraphProduct {
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        pricing::parse_product_graph_json(json.as_bytes(), pricing::JsonLimits::DEFAULT)
            .map(|inner| Self { inner })
            .map_err(|e| wire_error(py, e))
    }
    fn to_json(&self) -> PyResult<String> {
        pricing::product_graph_to_json(&self.inner).map_err(pricing_exception)
    }
    fn to_pretty_json(&self) -> PyResult<String> {
        pricing::product_graph_to_pretty_json(&self.inner).map_err(pricing_exception)
    }
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        pricing::fingerprint_product_graph(&self.inner)
            .map(|f| f.to_string())
            .map_err(pricing_exception)
    }
    #[getter]
    fn currency_id(&self) -> u16 {
        self.inner.currency().get()
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlyings().iter().map(|u| u.get()).collect()
    }
    #[getter]
    fn payment_dates(&self) -> Vec<String> {
        self.inner
            .payment_dates()
            .iter()
            .map(ToString::to_string)
            .collect()
    }
}

#[pyclass(name = "SourceGraphBuilder", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PySourceGraphBuilder {
    inner: SourceGraphBuilder,
}
#[pymethods]
impl PySourceGraphBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: SourceGraphBuilder::new(),
        }
    }
    fn literal(&mut self, py: Python<'_>, value: f64) -> PyResult<u32> {
        self.inner
            .literal(value)
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    #[pyo3(signature=(underlying_id, observation_date, *, pre_dividend=false))]
    fn spot(
        &mut self,
        py: Python<'_>,
        underlying_id: u32,
        observation_date: &Bound<'_, PyAny>,
        pre_dividend: bool,
    ) -> PyResult<u32> {
        let underlying = UnderlyingId::new(underlying_id);
        let observation_date = graph_date(py, observation_date, "/product/observation_date")?;
        let op = if pre_dividend {
            SourceOpcode::PreDividendSpot {
                underlying,
                observation_date,
            }
        } else {
            SourceOpcode::TerminalSpot {
                underlying,
                observation_date,
            }
        };
        self.inner
            .push(op)
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    fn add(&mut self, py: Python<'_>, left: u32, right: u32) -> PyResult<u32> {
        self.inner
            .push(SourceOpcode::Add {
                left: NodeId::new(left),
                right: NodeId::new(right),
            })
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    fn subtract(&mut self, py: Python<'_>, left: u32, right: u32) -> PyResult<u32> {
        self.inner
            .push(SourceOpcode::Subtract {
                left: NodeId::new(left),
                right: NodeId::new(right),
            })
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    fn multiply(&mut self, py: Python<'_>, left: u32, right: u32) -> PyResult<u32> {
        self.inner
            .push(SourceOpcode::Multiply {
                left: NodeId::new(left),
                right: NodeId::new(right),
            })
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    fn divide(&mut self, py: Python<'_>, numerator: u32, denominator: u32) -> PyResult<u32> {
        self.inner
            .push(SourceOpcode::Divide {
                numerator: NodeId::new(numerator),
                denominator: NodeId::new(denominator),
            })
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    fn negate(&mut self, py: Python<'_>, input: u32) -> PyResult<u32> {
        self.inner
            .push(SourceOpcode::Negate {
                input: NodeId::new(input),
            })
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    fn barrier_hit_state(
        &mut self,
        py: Python<'_>,
        previous: u32,
        hit_weight: u32,
    ) -> PyResult<u32> {
        self.inner
            .push(SourceOpcode::BarrierHitState {
                previous: NodeId::new(previous),
                hit_weight: NodeId::new(hit_weight),
            })
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    #[pyo3(signature=(left, right, *, smoothing_half_width=None))]
    fn minimum(
        &mut self,
        py: Python<'_>,
        left: u32,
        right: u32,
        smoothing_half_width: Option<f64>,
    ) -> PyResult<u32> {
        let op = match smoothing_half_width {
            None => SourceOpcode::Minimum {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Some(width) => SourceOpcode::SmoothMinimum {
                left: NodeId::new(left),
                right: NodeId::new(right),
                smoothing: CompactC2Smoothing::new(width).map_err(|e| invalid(py, e))?,
            },
        };
        self.inner
            .push(op)
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    #[pyo3(signature=(left, right, *, smoothing_half_width=None))]
    fn maximum(
        &mut self,
        py: Python<'_>,
        left: u32,
        right: u32,
        smoothing_half_width: Option<f64>,
    ) -> PyResult<u32> {
        let op = match smoothing_half_width {
            None => SourceOpcode::Maximum {
                left: NodeId::new(left),
                right: NodeId::new(right),
            },
            Some(width) => SourceOpcode::SmoothMaximum {
                left: NodeId::new(left),
                right: NodeId::new(right),
                smoothing: CompactC2Smoothing::new(width).map_err(|e| invalid(py, e))?,
            },
        };
        self.inner
            .push(op)
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    #[pyo3(signature=(input, *, smoothing_half_width=None))]
    fn indicator(
        &mut self,
        py: Python<'_>,
        input: u32,
        smoothing_half_width: Option<f64>,
    ) -> PyResult<u32> {
        let op = match smoothing_half_width {
            None => SourceOpcode::Indicator {
                input: NodeId::new(input),
            },
            Some(width) => SourceOpcode::SmoothIndicator {
                input: NodeId::new(input),
                smoothing: CompactC2Smoothing::new(width).map_err(|e| invalid(py, e))?,
            },
        };
        self.inner
            .push(op)
            .map(|id| id.get())
            .map_err(|e| invalid(py, e))
    }
    #[pyo3(signature=(outputs, payment_dates, *, currency_id, underlying_ids))]
    fn finish(
        &self,
        py: Python<'_>,
        outputs: Vec<u32>,
        payment_dates: Vec<Bound<'_, PyAny>>,
        currency_id: u16,
        underlying_ids: Vec<u32>,
    ) -> PyResult<PyGraphProduct> {
        let dates = payment_dates
            .iter()
            .enumerate()
            .map(|(index, date)| {
                graph_date(
                    py,
                    date,
                    &format!("/product/cashflows/{index}/payment_date"),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let graph = self
            .inner
            .clone()
            .finish(outputs.into_iter().map(NodeId::new).collect());
        GraphProduct::new(
            CurrencyId::new(currency_id),
            underlying_ids.into_iter().map(UnderlyingId::new).collect(),
            graph,
            dates,
        )
        .map(|inner| PyGraphProduct { inner })
        .map_err(|e| invalid(py, e))
    }
}
#[pyfunction]
fn product_graph_json_schema() -> &'static str {
    pricing::current_product_graph_schema()
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGraphProduct>()?;
    module.add_class::<PySourceGraphBuilder>()?;
    module.add_function(wrap_pyfunction!(product_graph_json_schema, module)?)?;
    Ok(())
}
