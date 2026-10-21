use pricing::risk::{
    VegaKtGammaCandidateAudit, VegaKtGammaRecoveryAudit, VegaKtGammaRecoveryPolicy,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(frozen, name = "VegaKtGammaRecoveryPolicy")]
#[derive(Clone, Debug)]
pub struct PyVegaKtGammaRecoveryPolicy {
    pub(crate) inner: VegaKtGammaRecoveryPolicy,
}
#[pymethods]
impl PyVegaKtGammaRecoveryPolicy {
    #[new]
    #[pyo3(signature=(candidate_regularizations, *, pilot_sampling_units, reconstruction_relative_limit, gamma_relative_standard_error_limit, mc_margin_multiplier, preserve_local_vega_total=false, local_vega_conservation_directions=None))]
    fn new(
        candidate_regularizations: Vec<f64>,
        pilot_sampling_units: u64,
        reconstruction_relative_limit: f64,
        gamma_relative_standard_error_limit: f64,
        mc_margin_multiplier: f64,
        preserve_local_vega_total: bool,
        local_vega_conservation_directions: Option<Vec<Vec<Vec<f64>>>>,
    ) -> PyResult<Self> {
        VegaKtGammaRecoveryPolicy::new(
            candidate_regularizations,
            pilot_sampling_units,
            reconstruction_relative_limit,
            gamma_relative_standard_error_limit,
            mc_margin_multiplier,
        )
        .and_then(|inner| {
            let inner = if preserve_local_vega_total {
                inner.with_local_vega_conservation()
            } else {
                inner
            };
            match local_vega_conservation_directions {
                Some(d) => inner.with_local_vega_direction_conservation(d),
                None => Ok(inner),
            }
        })
        .map(|inner| Self { inner })
        .map_err(|e| {
            Python::attach(|py| {
                crate::validation_exception(
                    py,
                    crate::PyValidationIssue::domain(
                        "/risk",
                        "invalid_risk_configuration",
                        e.to_string(),
                    ),
                )
            })
        })
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        policy_dict(py, &self.inner)
    }
}
pub(crate) fn policy_dict<'py>(
    py: Python<'py>,
    p: &VegaKtGammaRecoveryPolicy,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("candidate_regularizations", p.candidate_regularizations())?;
    d.set_item("pilot_sampling_units", p.pilot_sampling_units())?;
    d.set_item(
        "reconstruction_relative_limit",
        p.reconstruction_relative_limit(),
    )?;
    d.set_item(
        "gamma_relative_standard_error_limit",
        p.gamma_relative_standard_error_limit(),
    )?;
    d.set_item("mc_margin_multiplier", p.mc_margin_multiplier())?;
    if p.preserves_local_vega_total() {
        d.set_item("preserve_local_vega_total", true)?;
    }
    if !p.local_vega_conservation_directions().is_empty() {
        d.set_item(
            "local_vega_conservation_directions",
            p.local_vega_conservation_directions(),
        )?;
    }
    Ok(d)
}
fn candidate<'py>(
    py: Python<'py>,
    c: &VegaKtGammaCandidateAudit,
    start: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("passed", c.passed)?;
    let raw = crate::vegakt_convolution::to_dict(py, &c.diagnostic)?;
    raw.set_item("sampling_index_start", start)?;
    raw.set_item(
        "sampling_index_end_exclusive",
        start + c.diagnostic.independent_sampling_units,
    )?;
    d.set_item("diagnostic", raw)?;
    let mut rows = Vec::new();
    for r in &c.quality {
        let q = PyDict::new(py);
        for (key, value) in [
            ("source_maturity", r.source_maturity),
            ("local_vega_norm", r.local_vega_norm),
            ("local_vega_mc_norm", r.local_vega_mc_norm),
            (
                "reconstruction_difference_norm",
                r.reconstruction_difference_norm,
            ),
            ("reconstruction_mc_norm", r.reconstruction_mc_norm),
            ("gamma_mass_norm", r.gamma_mass_norm),
            ("gamma_mc_norm", r.gamma_mc_norm),
        ] {
            q.set_item(key, value)?;
        }
        q.set_item("reconstruction_margin_ratio", r.reconstruction_margin_ratio)?;
        q.set_item(
            "gamma_relative_standard_error",
            r.gamma_relative_standard_error,
        )?;
        q.set_item("unobserved_source_nodes", &r.unobserved_source_nodes)?;
        q.set_item("passed", r.passed)?;
        rows.push(q);
    }
    d.set_item("quality", rows)?;
    Ok(d)
}
pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtGammaRecoveryAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_gamma_recovery_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("policy_label", r.policy_label)?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item("policy", policy_dict(py, &r.policy)?)?;
    d.set_item("status", r.status.as_str())?;
    d.set_item("validation_sampling_units", r.validation_sampling_units)?;
    d.set_item("pilot_index_start", r.pilot_index_start)?;
    d.set_item("pilot_index_end_exclusive", r.pilot_index_end_exclusive)?;
    d.set_item("selected_regularization", r.selected_regularization)?;
    d.set_item(
        "pilot_candidates",
        r.pilot_candidates
            .iter()
            .map(|c| candidate(py, c, r.pilot_index_start))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    d.set_item(
        "validation",
        r.validation
            .as_ref()
            .map(|c| candidate(py, c, 0))
            .transpose()?,
    )?;
    d.set_item(
        "initial",
        crate::vegakt_initial_measure::to_dict(py, &r.initial)?,
    )?;
    d.set_item("iv_bucket_acceptance", "not_established")?;
    d.set_item("cross_row_iv_covariance", "not_established")?;
    d.set_item(
        "uncertainty_scope",
        "MC_conditional_on_pilot_choice_excludes_regularization_bias_and_unobserved_tails",
    )?;
    d.set_item(
        "mc_margin_interpretation",
        "estimated_norm_margin_not_simultaneous_confidence_bound",
    )?;
    Ok(d)
}
