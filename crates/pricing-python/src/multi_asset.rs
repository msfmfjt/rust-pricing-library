use crate::builders::{PyEngine, PyMarket, PyModel, date_from_python, option_side};
use crate::diagnostics::PyDiagnosticEstimate;
use crate::hull_white::{PyHullWhiteLsvTarget, PyHullWhiteModel};
use crate::multi_asset_hw::{
    PyMultiAssetHullWhiteCalibration, PyMultiAssetHullWhiteCurveRisk, PyMultiAssetHullWhiteLsvRisk,
};
use crate::multi_asset_lsv::{PyMultiAssetLsvCalibration, PyMultiAssetLsvRisk, extract_lsv_config};
use crate::{PyValidationIssue, pricing_exception, validation_exception};
use pricing::core::{CurrencyId, UnderlyingId};
use pricing::market::{CorrelationTermStructure, CorrelationToleranceConfig};
use pricing::mc::ExecutionPolicy;
use pricing::multi_asset::{
    AutocallObservation, AutocallSpec, BasketComponent, MemoryTermination, MultiAssetPrice,
    MultiAssetPricingPlan, MultiAssetProduct, MultiAssetRisk, MultiAssetRiskConfig,
    WorstOfComponent,
};
use pricing::product::CompactC2Smoothing;
use pyo3::prelude::*;
pub(crate) fn invalid(py: Python<'_>, e: impl ToString) -> PyErr {
    validation_exception(
        py,
        PyValidationIssue::domain(
            "/multi_asset",
            "invalid_multi_asset_configuration",
            e.to_string(),
        ),
    )
}
fn smooth(py: Python<'_>, width: Option<f64>) -> PyResult<Option<CompactC2Smoothing>> {
    width
        .map(CompactC2Smoothing::new)
        .transpose()
        .map_err(|e| invalid(py, e))
}
fn termination(py: Python<'_>, s: &str) -> PyResult<MemoryTermination> {
    match s {
        "forfeit" => Ok(MemoryTermination::Forfeit),
        "pay" => Ok(MemoryTermination::Pay),
        _ => Err(invalid(py, "memory termination must be forfeit or pay")),
    }
}

#[pyclass(frozen, name = "CorrelationSchedule", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCorrelationSchedule {
    inner: CorrelationTermStructure,
}
#[pymethods]
impl PyCorrelationSchedule {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(underlying_ids,effective_dates,matrices,*,symmetry_abs_tol,diagonal_abs_tol,psd_abs_tol,psd_rel_tol,zero_pivot_abs_tol,zero_pivot_rel_tol))]
    fn new(
        py: Python<'_>,
        underlying_ids: Vec<u32>,
        effective_dates: Vec<Py<PyAny>>,
        matrices: Vec<Vec<Vec<f64>>>,
        symmetry_abs_tol: f64,
        diagonal_abs_tol: f64,
        psd_abs_tol: f64,
        psd_rel_tol: f64,
        zero_pivot_abs_tol: f64,
        zero_pivot_rel_tol: f64,
    ) -> PyResult<Self> {
        if effective_dates.len() != matrices.len() {
            return Err(invalid(py, "correlation date/matrix counts differ"));
        }
        let dates = effective_dates
            .iter()
            .map(|d| date_from_python(py, d.bind(py), "/multi_asset/correlation_dates"))
            .collect::<PyResult<Vec<_>>>()?;
        let t = CorrelationToleranceConfig {
            symmetry_abs_tol,
            diagonal_abs_tol,
            psd_abs_tol,
            psd_rel_tol,
            zero_pivot_abs_tol,
            zero_pivot_rel_tol,
        };
        CorrelationTermStructure::new(
            underlying_ids.into_iter().map(UnderlyingId::new).collect(),
            dates.into_iter().zip(matrices).collect(),
            t,
        )
        .map(|inner| Self { inner })
        .map_err(|e| invalid(py, e))
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlyings().iter().map(|u| u.get()).collect()
    }
    #[getter]
    fn effective_dates(&self) -> Vec<String> {
        self.inner
            .entries()
            .iter()
            .map(|(d, _)| d.to_string())
            .collect()
    }
    #[getter]
    fn matrices(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| {
                c.canonical()
                    .chunks(c.dimension())
                    .map(|r| r.to_vec())
                    .collect()
            })
            .collect()
    }
    #[getter]
    fn raw_matrices(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| c.raw().chunks(c.dimension()).map(|r| r.to_vec()).collect())
            .collect()
    }
    #[getter]
    fn lower_factors(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| {
                c.lower()
                    .chunks(c.dimension())
                    .map(|r| r.to_vec())
                    .collect()
            })
            .collect()
    }
    #[getter]
    fn ranks(&self) -> Vec<usize> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| c.diagnostics().rank)
            .collect()
    }
    #[getter]
    fn pivots(&self) -> Vec<Vec<f64>> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| c.diagnostics().pivots.clone())
            .collect()
    }
    #[getter]
    fn zero_pivots(&self) -> Vec<Vec<usize>> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| c.diagnostics().zero_pivots.clone())
            .collect()
    }
    #[getter]
    fn maximum_adjustments(&self) -> Vec<f64> {
        self.inner
            .entries()
            .iter()
            .map(|(_, c)| c.diagnostics().maximum_adjustment)
            .collect()
    }
}
#[pyclass(frozen, name = "AutocallObservation", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAutocallObservation {
    inner: AutocallObservation,
}
#[pymethods]
impl PyAutocallObservation {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(date,payment_date,*,coupon_amount,coupon_level,call_level=None,call_coupon_amount=0.0))]
    fn new(
        py: Python<'_>,
        date: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
        coupon_amount: f64,
        coupon_level: f64,
        call_level: Option<f64>,
        call_coupon_amount: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AutocallObservation {
                date: date_from_python(py, date, "/multi_asset/observation")?,
                payment_date: date_from_python(py, payment_date, "/multi_asset/payment")?,
                coupon_amount,
                coupon_level,
                call_level,
                call_coupon_amount,
            },
        })
    }
}
#[pyclass(frozen, name = "MultiAssetProduct", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetProduct {
    inner: MultiAssetProduct,
}
#[pymethods]
impl PyMultiAssetProduct {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(underlying_ids,weights,scales,side,strike,expiry,payment_date,*,currency_id=1,notional=1.0,smoothing_half_width=None))]
    fn basket(
        py: Python<'_>,
        underlying_ids: Vec<u32>,
        weights: Vec<f64>,
        scales: Vec<f64>,
        side: &str,
        strike: f64,
        expiry: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
        currency_id: u16,
        notional: f64,
        smoothing_half_width: Option<f64>,
    ) -> PyResult<Self> {
        if underlying_ids.len() != weights.len() || weights.len() != scales.len() {
            return Err(invalid(py, "basket component lengths differ"));
        }
        let components = underlying_ids
            .into_iter()
            .zip(weights)
            .zip(scales)
            .map(|((u, weight), scale)| BasketComponent {
                underlying: UnderlyingId::new(u),
                weight,
                scale,
            })
            .collect();
        MultiAssetProduct::basket(
            CurrencyId::new(currency_id),
            components,
            option_side(py, side)?,
            strike,
            notional,
            date_from_python(py, expiry, "/multi_asset/expiry")?,
            date_from_python(py, payment_date, "/multi_asset/payment")?,
            smooth(py, smoothing_half_width)?,
        )
        .map(|inner| Self { inner })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(underlying_ids,reference_levels,side,strike,expiry,payment_date,*,currency_id=1,notional=1.0,smoothing_half_width=None))]
    fn worst_of(
        py: Python<'_>,
        underlying_ids: Vec<u32>,
        reference_levels: Vec<f64>,
        side: &str,
        strike: f64,
        expiry: &Bound<'_, PyAny>,
        payment_date: &Bound<'_, PyAny>,
        currency_id: u16,
        notional: f64,
        smoothing_half_width: Option<f64>,
    ) -> PyResult<Self> {
        if underlying_ids.len() != reference_levels.len() {
            return Err(invalid(py, "worst-of component lengths differ"));
        }
        let components = underlying_ids
            .into_iter()
            .zip(reference_levels)
            .map(|(u, reference_level)| WorstOfComponent {
                underlying: UnderlyingId::new(u),
                reference_level,
            })
            .collect();
        MultiAssetProduct::worst_of(
            CurrencyId::new(currency_id),
            components,
            option_side(py, side)?,
            strike,
            notional,
            date_from_python(py, expiry, "/multi_asset/expiry")?,
            date_from_python(py, payment_date, "/multi_asset/payment")?,
            smooth(py, smoothing_half_width)?,
        )
        .map(|inner| Self { inner })
        .map_err(|e| invalid(py, e))
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(underlying_ids,reference_levels,observations,maturity,maturity_payment,*,notional,final_barrier,memory,on_autocall,on_maturity,currency_id=1,smoothing_half_width=None))]
    fn autocallable(
        py: Python<'_>,
        underlying_ids: Vec<u32>,
        reference_levels: Vec<f64>,
        observations: Vec<Py<PyAutocallObservation>>,
        maturity: &Bound<'_, PyAny>,
        maturity_payment: &Bound<'_, PyAny>,
        notional: f64,
        final_barrier: f64,
        memory: bool,
        on_autocall: &str,
        on_maturity: &str,
        currency_id: u16,
        smoothing_half_width: Option<f64>,
    ) -> PyResult<Self> {
        if underlying_ids.len() != reference_levels.len() {
            return Err(invalid(py, "worst-of component lengths differ"));
        }
        let components = underlying_ids
            .into_iter()
            .zip(reference_levels)
            .map(|(u, reference_level)| WorstOfComponent {
                underlying: UnderlyingId::new(u),
                reference_level,
            })
            .collect();
        let spec = AutocallSpec {
            currency: CurrencyId::new(currency_id),
            components,
            observations: observations
                .into_iter()
                .map(|o| o.borrow(py).inner.clone())
                .collect(),
            maturity: date_from_python(py, maturity, "/multi_asset/maturity")?,
            maturity_payment: date_from_python(py, maturity_payment, "/multi_asset/payment")?,
            notional,
            final_barrier,
            memory,
            on_autocall: termination(py, on_autocall)?,
            on_maturity: termination(py, on_maturity)?,
        };
        spec.product(smooth(py, smoothing_half_width)?)
            .map(|inner| Self { inner })
            .map_err(|e| invalid(py, e))
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlyings().iter().map(|u| u.get()).collect()
    }
}
#[pyclass(frozen, name = "MultiAssetPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetPlan {
    inner: MultiAssetPricingPlan,
}
#[pymethods]
impl PyMultiAssetPlan {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(valuation_date,product,markets,models,correlations,engine,*,maximum_step,worker_threads=1,reduction_block_size=4096,lsv_configs=None,driver_correlations=None,rate_model=None,rate_correlations=None,lsv_targets=None))]
    fn compile(
        py: Python<'_>,
        valuation_date: &Bound<'_, PyAny>,
        product: &PyMultiAssetProduct,
        markets: Vec<Py<PyMarket>>,
        models: Vec<Py<PyModel>>,
        correlations: &PyCorrelationSchedule,
        engine: &PyEngine,
        maximum_step: f64,
        worker_threads: u32,
        reduction_block_size: u64,
        lsv_configs: Option<Vec<Option<Py<PyAny>>>>,
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
        rate_model: Option<&PyHullWhiteModel>,
        rate_correlations: Option<Vec<f64>>,
        lsv_targets: Option<Vec<Option<Py<PyHullWhiteLsvTarget>>>>,
    ) -> PyResult<Self> {
        let date = date_from_python(py, valuation_date, "/multi_asset/valuation_date")?;
        let execution = ExecutionPolicy::new(worker_threads, Some(reduction_block_size))
            .map_err(|e| invalid(py, e))?;
        let lsv = lsv_configs
            .map(|v| {
                v.into_iter()
                    .map(|c| c.map(|c| extract_lsv_config(py, c.bind(py))).transpose())
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_else(|| vec![None; models.len()]);
        let hw = if let Some(rates) = rate_model {
            Some(pricing::multi_asset::MultiAssetHullWhiteConfig {
                rate_model: rates.inner.clone(),
                rate_correlations: rate_correlations
                    .ok_or_else(|| invalid(py, "rate_correlations is required with rate_model"))?,
                lsv_targets: lsv_targets
                    .map(|v| {
                        v.into_iter()
                            .map(|t| t.map(|t| t.borrow(py).inner.clone()))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![None; models.len()]),
            })
        } else {
            if rate_correlations.is_some() || lsv_targets.is_some() {
                return Err(invalid(
                    py,
                    "rate_correlations and lsv_targets require rate_model",
                ));
            }
            None
        };
        let product = product.inner.clone();
        let markets = markets
            .into_iter()
            .map(|m| m.borrow(py).inner.equity().clone())
            .collect();
        let models = models
            .into_iter()
            .map(|m| m.borrow(py).inner.clone())
            .collect();
        let correlation = correlations.inner.clone();
        let engine = engine.inner;
        py.detach(|| {
            if let Some(hw) = hw {
                MultiAssetPricingPlan::compile_with_hull_white(
                    date,
                    product,
                    markets,
                    models,
                    correlation,
                    engine,
                    execution,
                    maximum_step,
                    lsv,
                    driver_correlations,
                    hw,
                )
            } else {
                MultiAssetPricingPlan::compile_with_bergomi_lsv(
                    date,
                    product,
                    markets,
                    models,
                    correlation,
                    engine,
                    execution,
                    maximum_step,
                    lsv,
                    driver_correlations,
                )
            }
        })
        .map(|inner| Self { inner })
        .map_err(|e| invalid(py, e))
    }
    fn evaluate(&self, py: Python<'_>) -> PyResult<PyMultiAssetPrice> {
        py.detach(|| self.inner.evaluate())
            .map(|inner| PyMultiAssetPrice { inner })
            .map_err(|e| pricing_exception(e.to_string()))
    }
    #[pyo3(signature=(*,gamma_relative_bump=None))]
    fn evaluate_aad(
        &self,
        py: Python<'_>,
        gamma_relative_bump: Option<f64>,
    ) -> PyResult<PyMultiAssetPrice> {
        py.detach(|| {
            self.inner.evaluate_aad(MultiAssetRiskConfig {
                gamma_relative_bump,
            })
        })
        .map(|inner| PyMultiAssetPrice { inner })
        .map_err(|e| pricing_exception(e.to_string()))
    }
    #[getter]
    fn fingerprint(&self) -> &str {
        self.inner.fingerprint()
    }
    #[getter]
    fn time_nodes(&self) -> Vec<f64> {
        self.inner.time_nodes().to_vec()
    }
    #[getter]
    fn correlation_entry_indices(&self) -> Vec<usize> {
        self.inner.correlation_entry_indices().to_vec()
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlyings().iter().map(|u| u.get()).collect()
    }
    #[getter]
    fn random_factor_count(&self) -> usize {
        self.inner.random_factor_count()
    }
    #[getter]
    fn lsv_calibrations(&self) -> Vec<Option<PyMultiAssetLsvCalibration>> {
        self.inner
            .lsv_calibrations()
            .into_iter()
            .zip(self.inner.lsv_two_factor_calibrations())
            .map(|(one, two)| one.map(Into::into).or_else(|| two.map(Into::into)))
            .collect()
    }
    #[getter]
    fn hull_white_calibrations(&self) -> Vec<Option<PyMultiAssetHullWhiteCalibration>> {
        self.inner
            .hull_white_lsv_calibrations()
            .into_iter()
            .zip(self.inner.lsv_volatility_factor_counts())
            .map(|(c, n)| c.map(|c| PyMultiAssetHullWhiteCalibration::new(c, n)))
            .collect()
    }
    #[getter]
    fn has_hull_white(&self) -> bool {
        self.inner.hull_white_model().is_some()
    }
    #[getter]
    fn lsv_driver_correlations(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner
            .lsv_driver_correlations()
            .iter()
            .map(|c| {
                c.canonical()
                    .chunks(c.dimension())
                    .map(|r| r.to_vec())
                    .collect()
            })
            .collect()
    }
    #[getter]
    fn lsv_transition_covariances(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner.lsv_transition_covariances()
    }
}
#[pyclass(frozen, name = "MultiAssetRisk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetRisk {
    inner: MultiAssetRisk,
}
#[pymethods]
impl PyMultiAssetRisk {
    #[getter]
    fn hull_white_lsv(&self) -> Option<PyMultiAssetHullWhiteLsvRisk> {
        self.inner
            .hull_white_lsv
            .clone()
            .map(|inner| PyMultiAssetHullWhiteLsvRisk { inner })
    }
    #[getter]
    fn lsv_local_variance(&self) -> Option<PyMultiAssetLsvRisk> {
        self.inner
            .lsv_local_variance
            .clone()
            .map(|inner| PyMultiAssetLsvRisk { inner })
    }
    #[getter]
    fn underlying_id(&self) -> u32 {
        self.inner.underlying.get()
    }
    #[getter]
    fn delta(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.inner.delta)
    }
    #[getter]
    fn delta_per_one_percent_spot(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.inner.delta_per_one_percent_spot)
    }
    #[getter]
    fn bs_vega(&self) -> Option<PyDiagnosticEstimate> {
        self.inner.bs_vega.map(PyDiagnosticEstimate::from_estimate)
    }
    #[getter]
    fn bs_vega_per_vol_point(&self) -> Option<PyDiagnosticEstimate> {
        self.inner
            .bs_vega_per_vol_point
            .map(PyDiagnosticEstimate::from_estimate)
    }
    #[getter]
    fn local_variance_time_nodes(&self) -> Vec<f64> {
        self.inner.local_variance_time_nodes.clone()
    }
    #[getter]
    fn local_variance_log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner.local_variance_log_moneyness_nodes.clone()
    }
    #[getter]
    fn local_variance(&self) -> Vec<PyDiagnosticEstimate> {
        self.inner
            .local_variance
            .iter()
            .copied()
            .map(PyDiagnosticEstimate::from_estimate)
            .collect()
    }
}
#[pyclass(frozen, name = "MultiAssetPrice", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiAssetPrice {
    inner: MultiAssetPrice,
}
#[pymethods]
impl PyMultiAssetPrice {
    #[getter]
    fn hull_white_curve_risk(&self) -> Option<PyMultiAssetHullWhiteCurveRisk> {
        self.inner
            .hull_white_curve_risk
            .clone()
            .map(|inner| PyMultiAssetHullWhiteCurveRisk { inner })
    }
    #[getter]
    fn value(&self) -> f64 {
        self.inner.price.value().get()
    }
    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.price.standard_error().get()
    }
    #[getter]
    fn price(&self) -> PyDiagnosticEstimate {
        PyDiagnosticEstimate::from_estimate(self.inner.price)
    }
    #[getter]
    fn risks(&self) -> Vec<PyMultiAssetRisk> {
        self.inner
            .risks
            .iter()
            .cloned()
            .map(|inner| PyMultiAssetRisk { inner })
            .collect()
    }
    #[getter]
    fn gamma(&self) -> Vec<Vec<PyDiagnosticEstimate>> {
        self.inner
            .gamma
            .iter()
            .map(|r| {
                r.iter()
                    .copied()
                    .map(PyDiagnosticEstimate::from_estimate)
                    .collect()
            })
            .collect()
    }
    #[getter]
    fn gamma_relative_bump(&self) -> Option<f64> {
        self.inner.gamma_relative_bump
    }
    #[getter]
    fn underlying_ids(&self) -> Vec<u32> {
        self.inner.underlyings.iter().map(|u| u.get()).collect()
    }
    #[getter]
    fn fingerprint(&self) -> &str {
        &self.inner.fingerprint
    }
    #[getter]
    fn evaluated_paths(&self) -> u128 {
        self.inner.evaluated_paths
    }
    #[getter]
    fn worker_threads(&self) -> u32 {
        self.inner.worker_threads
    }
    #[getter]
    fn reduction_block_size(&self) -> u64 {
        self.inner.reduction_block_size
    }
    #[getter]
    fn local_variance_boundary_counts(&self) -> Vec<f64> {
        self.inner.local_variance_boundary_counts.clone()
    }
    #[getter]
    fn lsv_leverage_boundary_counts(&self) -> Vec<f64> {
        self.inner.lsv_leverage_boundary_counts.clone()
    }
    #[getter]
    fn direction_checksum(&self) -> Option<String> {
        self.inner
            .direction_checksum
            .map(|h| h.iter().map(|b| format!("{b:02x}")).collect())
    }
    #[getter]
    fn scramble_checksum(&self) -> Option<String> {
        self.inner
            .scramble_checksum
            .map(|h| h.iter().map(|b| format!("{b:02x}")).collect())
    }
}
