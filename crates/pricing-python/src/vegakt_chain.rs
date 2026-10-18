//! Diagnostic-only flat-space paper chain over the compiled PricingPlan.
use pricing::risk::{VegaKtChainDiagnostic, VegaKtStepMoment, VegaKtStepSamplingUnit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(frozen, name = "VegaKtChainDiagnostic", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVegaKtChainDiagnostic {
    pub(crate) inner: VegaKtChainDiagnostic,
}
fn moment(py: Python<'_>, m: VegaKtStepMoment) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mean", m.mean())?;
    d.set_item("sample_variance", m.sample_variance())?;
    d.set_item("standard_error", m.standard_error())?;
    Ok(d)
}
fn moments<'py>(py: Python<'py>, ms: &[VegaKtStepMoment]) -> PyResult<Vec<Bound<'py, PyDict>>> {
    ms.iter().map(|&m| moment(py, m)).collect()
}
#[pymethods]
impl PyVegaKtChainDiagnostic {
    #[getter]
    fn policy_label(&self) -> &'static str {
        self.inner.estimate.policy_label
    }
    #[getter]
    fn raw_buckets(&self) -> Vec<f64> {
        self.inner
            .estimate
            .position_iv_buckets
            .iter()
            .map(|v| v.mean())
            .collect()
    }
    #[getter]
    fn standard_errors(&self) -> Vec<Option<f64>> {
        self.inner
            .estimate
            .position_iv_buckets
            .iter()
            .map(|v| v.standard_error())
            .collect()
    }
    #[getter]
    fn hedge_quantities(&self) -> Vec<f64> {
        self.inner
            .estimate
            .offsetting_hedge_quantities
            .iter()
            .map(|v| v.mean())
            .collect()
    }
    #[getter]
    fn scalar_vega(&self) -> f64 {
        self.inner.estimate.total_position_iv_vega.mean()
    }
    #[getter]
    fn scalar_standard_error(&self) -> Option<f64> {
        self.inner.estimate.total_position_iv_vega.standard_error()
    }
    #[getter]
    fn maturity_nodes(&self) -> Vec<f64> {
        self.inner
            .estimate
            .reporting_basis
            .maturity_nodes()
            .to_vec()
    }
    #[getter]
    fn log_moneyness_nodes(&self) -> Vec<f64> {
        self.inner
            .estimate
            .reporting_basis
            .log_moneyness_nodes()
            .to_vec()
    }
    /// JSON-compatible diagnostics; not a pricing_result wire document or acceptance.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let r = &self.inner;
        let e = &r.estimate;
        d.set_item("document_kind", "vega_kt_chain_diagnostic")?;
        d.set_item("schema_version", 1)?;
        d.set_item("diagnostic_only", true)?;
        d.set_item("policy_label", e.policy_label)?;
        d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
        d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
        d.set_item("initial_strikes", r.initial_strikes.to_vec())?;
        d.set_item(
            "requested_initial_strikes",
            r.requested_initial_strikes.to_vec(),
        )?;
        d.set_item(
            "recovery_log_moneyness_nodes",
            r.recovery_log_moneyness_nodes.to_vec(),
        )?;
        d.set_item(
            "maximum_recovery_log_spacing",
            r.maximum_recovery_log_spacing,
        )?;
        d.set_item("relative_bump", r.relative_bump)?;
        d.set_item("coarse_subdivisions", e.coarse_subdivisions)?;
        d.set_item("master_seed", r.master_seed)?;
        d.set_item("brownian_bridge", r.brownian_bridge)?;
        d.set_item("independent_sampling_units", e.independent_sampling_units)?;
        d.set_item("evaluated_paths", r.evaluated_paths)?;
        d.set_item(
            "sampling_unit",
            match e.sampling_unit {
                VegaKtStepSamplingUnit::IndependentPath => "independent_path",
                VegaKtStepSamplingUnit::AntitheticPair => "antithetic_pair",
            },
        )?;
        d.set_item("maturity_nodes", self.maturity_nodes())?;
        d.set_item("log_moneyness_nodes", self.log_moneyness_nodes())?;
        d.set_item(
            "implied_volatilities",
            e.reporting_basis.implied_volatilities().to_vec(),
        )?;
        for (key, ms) in [
            ("position_iv_buckets", &e.position_iv_buckets),
            (
                "offsetting_hedge_quantities",
                &e.offsetting_hedge_quantities,
            ),
            (
                "initial_bump_bucket_differences",
                &e.initial_bump_bucket_differences,
            ),
            (
                "quadrature_bucket_differences",
                &e.quadrature_bucket_differences,
            ),
            ("excluded_local_vegas", &e.excluded_local_vegas),
            ("propagated_tail_iv_vegas", &e.propagated_tail_iv_vegas),
        ] {
            d.set_item(key, moments(py, ms)?)?;
        }
        for (key, m) in [
            ("total_position_iv_vega", e.total_position_iv_vega),
            (
                "total_initial_bump_difference",
                e.total_initial_bump_difference,
            ),
            ("total_quadrature_difference", e.total_quadrature_difference),
            (
                "initial_reconstruction_difference",
                e.initial_reconstruction_difference,
            ),
            (
                "initial_bump_vega_difference",
                e.initial_bump_vega_difference,
            ),
        ] {
            d.set_item(key, moment(py, m)?)?;
        }
        d.set_item("initial_tail_vegas", moments(py, &e.initial_tail_vegas)?)?;
        d.set_item(
            "reporting_edge_iv_vegas",
            e.reporting_edge_iv_vegas
                .iter()
                .map(|row| moments(py, row))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let mut rows = Vec::new();
        for row in &e.rows {
            let d = PyDict::new(py);
            d.set_item("maturity", row.maturity)?;
            d.set_item("strike_bounds", row.strike_bounds.to_vec())?;
            d.set_item(
                "next_gamma_bounds",
                row.next_gamma_bounds.map(|v| v.to_vec()),
            )?;
            d.set_item("recovery_source_maturity", row.recovery_source_maturity)?;
            d.set_item("recovery_density_maturity", row.recovery_density_maturity)?;
            d.set_item(
                "excluded_density_probability",
                row.excluded_density_probability,
            )?;
            rows.push(d);
        }
        d.set_item("rows", rows)?;
        d.set_item("limitations",vec!["Flat spatial variance, constant forward, unit discount and identity dividend coordinates only.","Standard errors are conditional sampling errors; no bias bound, price covariance or full bucket covariance is supplied.","Weak Gamma recovery approximates conditional source hats by next-state hats with no uniform error-order guarantee at payoff or basis kinks. Initial Gamma, finite strike domain and interpolation errors remain.","Equation (9) excludes unknown outside-domain risk; propagated Gamma tails use constant extrapolation.","Ordinary pricing VegaKT and its diagnostic guard are unchanged."])?;
        Ok(d)
    }
}
