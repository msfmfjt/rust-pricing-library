use crate::vegakt_convolution::{moment, moments};
use pricing::risk::{VegaKtTwoStepSmileGammaAudit, VegaKtVarianceGammaCandidate};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn candidate<'py>(
    py: Python<'py>,
    c: &VegaKtVarianceGammaCandidate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("passed", c.quality.passed)?;
    d.set_item("relative_regularization", c.relative_regularization)?;
    d.set_item("sampling_index_start", c.sampling_index_start)?;
    d.set_item(
        "sampling_index_end_exclusive",
        c.sampling_index_start + c.independent_sampling_units,
    )?;
    d.set_item("independent_sampling_units", c.independent_sampling_units)?;
    d.set_item("inverse_noise_gain_bound", c.inverse_noise_gain_bound)?;
    d.set_item("resolution_trace", c.resolution_trace)?;
    d.set_item("nonzero_sampling_units", &c.nonzero_sampling_units)?;
    for (key, m) in [
        ("native_price", c.native_price),
        (
            "unrecovered_initial_variance_sensitivity",
            c.unrecovered_initial_variance_sensitivity,
        ),
        ("variance_sensitivity_total", c.variance_sensitivity_total),
        ("reconstruction_total", c.reconstruction_total),
        ("total_difference", c.total_difference),
    ] {
        d.set_item(key, moment(py, m)?)?;
    }
    for (key, m) in [
        ("gamma_masses", &c.gamma_masses),
        ("variance_sensitivities", &c.variance_sensitivities),
        ("reconstruction_differences", &c.reconstruction_differences),
        ("variance_direction_risks", &c.variance_direction_risks),
        ("recovered_direction_risks", &c.recovered_direction_risks),
        ("direction_differences", &c.direction_differences),
    ] {
        d.set_item(key, moments(py, m)?)?;
    }
    let q = PyDict::new(py);
    for (key, value) in [
        ("sensitivity_norm", c.quality.sensitivity_norm),
        ("sensitivity_mc_norm", c.quality.sensitivity_mc_norm),
        ("reconstruction_norm", c.quality.reconstruction_norm),
        ("reconstruction_mc_norm", c.quality.reconstruction_mc_norm),
        ("gamma_norm", c.quality.gamma_norm),
        ("gamma_mc_norm", c.quality.gamma_mc_norm),
    ] {
        q.set_item(key, value)?;
    }
    q.set_item(
        "reconstruction_margin_ratio",
        c.quality.reconstruction_margin_ratio,
    )?;
    q.set_item(
        "gamma_relative_standard_error",
        c.quality.gamma_relative_standard_error,
    )?;
    q.set_item("passed", c.quality.passed)?;
    d.set_item("quality", q)?;
    Ok(d)
}

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtTwoStepSmileGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_two_step_smile_gamma_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("coordinate", "annual_local_variance_nodes")?;
    d.set_item(
        "scope",
        "fixed_gaussian_first_marginal_nonflat_second_transition",
    )?;
    d.set_item("policy_label", "two_step_variance_gamma_split_sample_v1")?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item(
        "policy",
        crate::vegakt_selection::policy_dict(py, &r.policy)?,
    )?;
    d.set_item(
        "variance_conservation_directions",
        &r.variance_conservation_directions,
    )?;
    d.set_item("status", r.status.as_str())?;
    d.set_item(
        "sampling_unit",
        if r.antithetic {
            "antithetic_pair"
        } else {
            "path"
        },
    )?;
    d.set_item("validation_sampling_units", r.validation_sampling_units)?;
    d.set_item("pilot_index_start", r.pilot_index_start)?;
    d.set_item("pilot_index_end_exclusive", r.pilot_index_end_exclusive)?;
    d.set_item("source_maturity", r.source_maturity)?;
    d.set_item("terminal_maturity", r.terminal_maturity)?;
    d.set_item("source_variance", r.source_variance)?;
    d.set_item("source_log_nodes", &r.source_log_nodes)?;
    d.set_item("second_step_variances", &r.second_step_variances)?;
    d.set_item("gamma_strikes", &r.gamma_strikes)?;
    d.set_item("forward_matrix", &r.forward_matrix)?;
    let a = PyDict::new(py);
    a.set_item("quadrature_order", r.operator_accuracy.quadrature_order)?;
    a.set_item(
        "refinement_maximum_difference",
        r.operator_accuracy.refinement_maximum_difference,
    )?;
    a.set_item(
        "omitted_tail_entry_bound",
        r.operator_accuracy.omitted_tail_entry_bound,
    )?;
    a.set_item("entry_tolerance", r.operator_accuracy.entry_tolerance)?;
    a.set_item(
        "refinement_interpretation",
        "convergence_diagnostic_not_rigorous_total_error_bound",
    )?;
    d.set_item("operator_accuracy", a)?;
    d.set_item(
        "pilot_candidates",
        r.pilot_candidates
            .iter()
            .map(|c| candidate(py, c))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    d.set_item("selected_regularization", r.selected_regularization)?;
    d.set_item(
        "validation",
        r.validation
            .as_ref()
            .map(|c| candidate(py, c))
            .transpose()?,
    )?;
    d.set_item(
        "estimate",
        r.estimate.as_ref().map(|v| moments(py, v)).transpose()?,
    )?;
    d.set_item(
        "estimate_scope",
        "supplied_variance_directions_on_heldout_success_only",
    )?;
    d.set_item("initial_gamma_recovery", "not_implemented")?;
    d.set_item("full_market_iv_bucket_acceptance", "not_established")?;
    d.set_item(
        "uncertainty_scope",
        "same_path_direction_MC_conditional_on_pilot_choice_excludes_bias_and_unobserved_tails",
    )?;
    d.set_item(
        "mc_margin_interpretation",
        "estimated_norm_margin_not_simultaneous_confidence_bound",
    )?;
    Ok(d)
}
