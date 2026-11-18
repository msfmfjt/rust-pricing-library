//! Measure requests for the `measures` task operation (ADR 0043).
use crate::*;
use pricing::risk::SpotBump;
use pricing::{
    BumpCalibration, BumpSize, CapabilityStatus, CorrelationId, CorrelationSelection, Measure,
    MeasureCapability, MeasureMethod, MeasureRequest, MeasureSet, ModelParameterId,
    ModelParameterSelection,
};
use pyo3::types::PyDict;

#[pyclass(frozen, name = "MeasureRequest")]
#[derive(Clone, Debug)]
pub struct PyMeasureRequest {
    pub(crate) inner: MeasureRequest,
}
/// A stated calibration treatment, or the structured error for an unknown one.
fn parse_calibration(py: Python<'_>, calibration: &str, field: &str) -> PyResult<BumpCalibration> {
    match calibration {
        "not_applicable" => Ok(BumpCalibration::NotApplicable),
        "fixed" => Ok(BumpCalibration::Fixed),
        "recalibrated" => Ok(BumpCalibration::Recalibrated),
        _ => Err(builders::domain_error(
            py,
            "invalid_bump_calibration",
            field,
            "calibration must be not_applicable, fixed or recalibrated",
        )),
    }
}

#[pymethods]
impl PyMeasureRequest {
    #[staticmethod]
    fn price() -> Self {
        Self {
            inner: MeasureRequest::native(Measure::Price),
        }
    }
    /// Spot Delta by the backend's own estimator. A calibrated source with
    /// native Spot risk (rough-family LSV) needs the calibration treatment.
    #[staticmethod]
    #[pyo3(signature=(*, calibration=None))]
    fn spot_delta(py: Python<'_>, calibration: Option<&str>) -> PyResult<Self> {
        let method = match calibration {
            None => MeasureMethod::Native,
            Some(c) => MeasureMethod::NativeCalibrated {
                calibration: parse_calibration(py, c, "/operation/measures/method/calibration")?,
            },
        };
        Ok(Self {
            inner: MeasureRequest::new(Measure::SpotDelta, method),
        })
    }
    /// Spot Gamma. With a bump it is the backend's central difference of
    /// native Deltas; without one it asks for a bump-free native estimator.
    /// A calibration treatment selects the calibrated native estimator and
    /// needs a bump.
    #[staticmethod]
    #[pyo3(signature=(*, relative_bump=None, absolute_bump=None, calibration=None))]
    fn spot_gamma(
        py: Python<'_>,
        relative_bump: Option<f64>,
        absolute_bump: Option<f64>,
        calibration: Option<&str>,
    ) -> PyResult<Self> {
        let bump = match (relative_bump, absolute_bump) {
            (Some(_), Some(_)) => {
                return Err(builders::domain_error(
                    py,
                    "ambiguous_gamma_bump",
                    "/operation/measures/method/bump",
                    "specify only one of relative_bump and absolute_bump",
                ));
            }
            (Some(h), None) => Some(SpotBump::relative(h)),
            (None, Some(h)) => Some(SpotBump::absolute(h)),
            (None, None) => None,
        }
        .transpose()
        .map_err(|error| {
            builders::domain_error(
                py,
                "invalid_gamma_bump",
                "/operation/measures/method/bump",
                error,
            )
        })?;
        let calibration = calibration
            .map(|c| parse_calibration(py, c, "/operation/measures/method/calibration"))
            .transpose()?;
        Ok(Self {
            inner: match (bump, calibration) {
                (Some(bump), Some(calibration)) => MeasureRequest::new(
                    Measure::SpotGamma,
                    MeasureMethod::NativeSpotBumpCalibrated { bump, calibration },
                ),
                (Some(bump), None) => MeasureRequest::spot_gamma(bump),
                (None, None) => MeasureRequest::native(Measure::SpotGamma),
                (None, Some(_)) => {
                    return Err(builders::domain_error(
                        py,
                        "invalid_gamma_bump",
                        "/operation/measures/method/bump",
                        "a calibrated native Gamma needs relative_bump or absolute_bump",
                    ));
                }
            },
        })
    }
    /// Parallel market-IV Vega by the backend's own estimator. A calibrated
    /// source with native market-IV risk (rough-family LSV) needs the
    /// calibration treatment, which must be `recalibrated`.
    #[staticmethod]
    #[pyo3(signature=(*, calibration=None))]
    fn parallel_market_iv_vega(py: Python<'_>, calibration: Option<&str>) -> PyResult<Self> {
        let method = match calibration {
            None => MeasureMethod::Native,
            Some(c) => MeasureMethod::NativeCalibrated {
                calibration: parse_calibration(py, c, "/operation/measures/method/calibration")?,
            },
        };
        Ok(Self {
            inner: MeasureRequest::new(Measure::ParallelMarketIvVega, method),
        })
    }
    /// Bucketed quote-node Vega on the stated grid.
    #[staticmethod]
    #[pyo3(signature=(*, maturity_nodes, log_forward_moneyness_nodes, relative_density_threshold, full_bucket_covariance=false))]
    fn vega_kt(
        py: Python<'_>,
        maturity_nodes: &Bound<'_, PyAny>,
        log_forward_moneyness_nodes: &Bound<'_, PyAny>,
        relative_density_threshold: f64,
        full_bucket_covariance: bool,
    ) -> PyResult<Self> {
        let grid = builders::vega_kt_from_python(
            py,
            Some(maturity_nodes),
            Some(log_forward_moneyness_nodes),
            Some(relative_density_threshold),
            full_bucket_covariance,
        )?
        .expect("a stated grid is always requested");
        Ok(Self {
            inner: MeasureRequest::native(Measure::VegaKt(grid)),
        })
    }
    /// A central difference of complete revaluations of `measure`. The bump
    /// size and the calibration treatment are both required.
    #[staticmethod]
    #[pyo3(signature=(measure, *, calibration, relative=None, absolute=None))]
    fn bump_and_revalue(
        py: Python<'_>,
        measure: &str,
        calibration: &str,
        relative: Option<f64>,
        absolute: Option<f64>,
    ) -> PyResult<Self> {
        let field = "/operation/measures/method";
        let measure = match measure {
            "spot_delta" => Measure::SpotDelta,
            "spot_gamma" => Measure::SpotGamma,
            "parallel_market_iv_vega" => Measure::ParallelMarketIvVega,
            _ => {
                return Err(builders::domain_error(
                    py,
                    "invalid_measure",
                    "/operation/measures/measure",
                    "bump_and_revalue estimates spot_delta, spot_gamma or parallel_market_iv_vega",
                ));
            }
        };
        let calibration = parse_calibration(py, calibration, field)?;
        let size = match (relative, absolute) {
            (Some(h), None) => BumpSize::relative(h),
            (None, Some(h)) => BumpSize::absolute(h),
            _ => {
                return Err(builders::domain_error(
                    py,
                    "ambiguous_bump_size",
                    field,
                    "specify exactly one of relative and absolute",
                ));
            }
        }
        .map_err(|error| builders::domain_error(py, "invalid_bump_size", field, error))?;
        Ok(Self {
            inner: MeasureRequest::new(
                measure,
                MeasureMethod::BumpAndRevalue { size, calibration },
            ),
        })
    }
    /// Derivatives with respect to native model parameters, named by label
    /// (`"vol_of_vol"`, `"hurst"`, `"component_vol_of_vol.1"`), one report
    /// row each in this order. Without a calibration treatment it asks for
    /// the fixed-model native estimator (direct rough volatility, stochastic
    /// dividends without a calibration step); a
    /// calibrated source (rough-family LSV) needs `recalibrated`, since its
    /// estimator differentiates through the recalibration.
    #[staticmethod]
    #[pyo3(signature=(parameters, *, calibration=None))]
    fn model_parameters(
        py: Python<'_>,
        parameters: Vec<String>,
        calibration: Option<&str>,
    ) -> PyResult<Self> {
        let field = "/operation/measures/measure/parameters";
        let ids = parameters
            .iter()
            .map(|label| {
                ModelParameterId::from_label(label).ok_or_else(|| {
                    builders::domain_error(
                        py,
                        "invalid_model_parameter",
                        field,
                        format!("unknown model parameter {label:?}"),
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let selection = ModelParameterSelection::new(ids)
            .map_err(|e| builders::domain_error(py, "invalid_model_parameter", field, e))?;
        let method = match calibration {
            None => MeasureMethod::Native,
            Some(c) => MeasureMethod::NativeCalibrated {
                calibration: parse_calibration(py, c, "/operation/measures/method/calibration")?,
            },
        };
        Ok(Self {
            inner: MeasureRequest::new(Measure::ModelParameters(selection), method),
        })
    }
    /// Derivatives with respect to Brownian correlations, named by label
    /// (`"equity_dividend"`, `"equity_rate"`, `"spot_volatility.0"`,
    /// `"dividend_volatility.1"`, `"volatility_factors"`), one report row
    /// each in this order. The fixed-model native estimator varies one
    /// symmetric pair of the driver correlation matrix.
    #[staticmethod]
    fn correlations(py: Python<'_>, correlations: Vec<String>) -> PyResult<Self> {
        let field = "/operation/measures/measure/correlations";
        let ids = correlations
            .iter()
            .map(|label| {
                CorrelationId::from_label(label).ok_or_else(|| {
                    builders::domain_error(
                        py,
                        "invalid_correlation",
                        field,
                        format!("unknown correlation {label:?}"),
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let selection = CorrelationSelection::new(ids)
            .map_err(|e| builders::domain_error(py, "invalid_correlation", field, e))?;
        Ok(Self {
            inner: MeasureRequest::native(Measure::Correlations(selection)),
        })
    }
    #[getter]
    fn measure(&self) -> &'static str {
        self.inner.measure.kind().as_str()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method.kind().as_str()
    }
    /// The parameter labels of a `model_parameters` request, in order.
    #[getter]
    fn parameters(&self) -> Option<Vec<String>> {
        match &self.inner.measure {
            Measure::ModelParameters(selection) => {
                Some(selection.parameters().iter().map(|p| p.label()).collect())
            }
            _ => None,
        }
    }
    /// The correlation labels of a `correlations` request, in order.
    #[getter]
    fn correlation_labels(&self) -> Option<Vec<String>> {
        match &self.inner.measure {
            Measure::Correlations(selection) => {
                Some(selection.correlations().iter().map(|c| c.label()).collect())
            }
            _ => None,
        }
    }
    #[getter]
    fn relative_bump(&self) -> Option<f64> {
        match self.inner.method {
            MeasureMethod::NativeSpotBump {
                bump: SpotBump::Relative(h),
            }
            | MeasureMethod::NativeSpotBumpCalibrated {
                bump: SpotBump::Relative(h),
                ..
            }
            | MeasureMethod::BumpAndRevalue {
                size: BumpSize::Relative(h),
                ..
            } => Some(h.get()),
            _ => None,
        }
    }
    #[getter]
    fn absolute_bump(&self) -> Option<f64> {
        match self.inner.method {
            MeasureMethod::NativeSpotBump {
                bump: SpotBump::Absolute(h),
            }
            | MeasureMethod::NativeSpotBumpCalibrated {
                bump: SpotBump::Absolute(h),
                ..
            }
            | MeasureMethod::BumpAndRevalue {
                size: BumpSize::Absolute(h),
                ..
            } => Some(h.get()),
            _ => None,
        }
    }
    /// The stated calibration treatment, for the methods that carry one.
    #[getter]
    fn calibration(&self) -> Option<&'static str> {
        self.inner.method.calibration().map(BumpCalibration::as_str)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        match (self.relative_bump(), self.absolute_bump()) {
            (Some(h), _) => format!(
                "MeasureRequest({}, {}, relative_bump={h})",
                self.measure(),
                self.method()
            ),
            (_, Some(h)) => format!(
                "MeasureRequest({}, {}, absolute_bump={h})",
                self.measure(),
                self.method()
            ),
            _ => format!("MeasureRequest({}, {})", self.measure(), self.method()),
        }
    }
}

#[pyclass(frozen, name = "MeasureCapability")]
#[derive(Clone, Debug)]
pub struct PyMeasureCapability {
    pub(crate) inner: MeasureCapability,
}
#[pymethods]
impl PyMeasureCapability {
    #[getter]
    fn request(&self) -> PyMeasureRequest {
        PyMeasureRequest {
            inner: self.inner.request.clone(),
        }
    }
    #[getter]
    fn measure(&self) -> &'static str {
        self.inner.request.measure.kind().as_str()
    }
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.request.method.kind().as_str()
    }
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            CapabilityStatus::Supported => "supported",
            CapabilityStatus::Conditional => "conditional",
            CapabilityStatus::Unsupported => "unsupported",
        }
    }
    #[getter]
    fn reason(&self) -> Option<&'static str> {
        self.inner.reason.map(errors::capability_reason)
    }
    #[getter]
    fn detail(&self) -> &str {
        &self.inner.detail
    }
}

pub(crate) fn requests(measures: &[PyRef<'_, PyMeasureRequest>]) -> Vec<MeasureRequest> {
    measures.iter().map(|m| m.inner.clone()).collect()
}

/// A duplicate-free list, or the structured validation error for it.
pub(crate) fn measure_set(measures: &[PyRef<'_, PyMeasureRequest>]) -> PyResult<MeasureSet> {
    MeasureSet::new(requests(measures))
        .map_err(|e| errors::exception(e, errors::Stage::Validation, Some("measures")))
}

/// The requested method as plain data for a report row.
pub(crate) fn method_detail<'py>(
    py: Python<'py>,
    method: MeasureMethod,
) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("type", method.kind().as_str())?;
    match method {
        MeasureMethod::NativeSpotBump { bump }
        | MeasureMethod::NativeSpotBumpCalibrated { bump, .. } => {
            let (kind, size) = match bump {
                SpotBump::Relative(h) => ("relative", h.get()),
                SpotBump::Absolute(h) => ("absolute", h.get()),
            };
            detail.set_item("bump_type", kind)?;
            detail.set_item("bump", size)?;
            if let Some(calibration) = method.calibration() {
                detail.set_item("calibration", calibration.as_str())?;
            }
        }
        MeasureMethod::NativeCalibrated { calibration } => {
            detail.set_item("calibration", calibration.as_str())?;
        }
        MeasureMethod::BumpAndRevalue { size, calibration } => {
            let (kind, size) = match size {
                BumpSize::Relative(h) => ("relative", h.get()),
                BumpSize::Absolute(h) => ("absolute", h.get()),
            };
            detail.set_item("bump_type", kind)?;
            detail.set_item("bump", size)?;
            detail.set_item("calibration", calibration.as_str())?;
        }
        _ => {}
    }
    Ok(detail)
}

/// A bump cost as plain data.
pub(crate) fn cost<'py>(py: Python<'py>, cost: pricing::BumpCost) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("calibrations", cost.calibrations)?;
    out.set_item("evaluations", cost.evaluations)?;
    Ok(out)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMeasureRequest>()?;
    module.add_class::<PyMeasureCapability>()?;
    Ok(())
}
