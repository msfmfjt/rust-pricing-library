use crate::vegakt_convolution::moments;
use crate::vegakt_smile::candidate;
use pricing::risk::VegaKtThreeStepTerminalGammaAudit;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtThreeStepTerminalGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_three_step_terminal_gamma_audit")?;
    d.set_item("schema_version", 1)?;
    if let Some(readouts) = &r.variance_risk_directions {
        d.set_item("document_kind", "vega_kt_three_step_terminal_readout_audit")?;
        d.set_item("variance_risk_directions", readouts)?;
        d.set_item("readout_inverse_conservation_relative_tolerance", 1e-9)?;
    }
    d.set_item("diagnostic_only", true)?;
    d.set_item("coordinate", "annual_local_variance_nodes")?;
    d.set_item(
        "scope",
        "three_intervals_non_gaussian_second_marginal_terminal_row_only",
    )?;
    d.set_item(
        "policy_label",
        "three_step_terminal_variance_gamma_split_sample_v1",
    )?;
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
    d.set_item("first_step_variance", r.first_step_variance)?;
    d.set_item("source_distribution", "two_log_euler_transition_mixture")?;
    d.set_item("third_step_variances", &r.third_step_variances)?;
    d.set_item("earlier_gamma_recovery", "not_implemented")?;
    d.set_item("recovered_variance_row", 2)?;
    d.set_item("unrecovered_variance_rows", [0, 1])?;
    d.set_item("source_log_nodes", &r.source_log_nodes)?;
    d.set_item("second_step_variances", &r.second_step_variances)?;
    d.set_item("gamma_strikes", &r.gamma_strikes)?;
    d.set_item("forward_matrix", &r.forward_matrix)?;
    let a = PyDict::new(py);
    a.set_item("quadrature_order", r.operator_accuracy.quadrature_order)?;
    a.set_item("source_log_window", r.operator_accuracy.source_log_window)?;
    for (key, value) in [
        ("source_probability", r.operator_accuracy.source_probability),
        (
            "source_first_spot_moment",
            r.operator_accuracy.source_first_spot_moment,
        ),
        (
            "source_probability_tail_bound",
            r.operator_accuracy.source_probability_tail_bound,
        ),
        (
            "source_spot_tail_bound",
            r.operator_accuracy.source_spot_tail_bound,
        ),
        (
            "probability_refinement_difference",
            r.operator_accuracy.probability_refinement_difference,
        ),
        (
            "spot_refinement_difference",
            r.operator_accuracy.spot_refinement_difference,
        ),
        (
            "operator_refinement_difference",
            r.operator_accuracy.operator_refinement_difference,
        ),
        (
            "omitted_operator_entry_bound",
            r.operator_accuracy.omitted_operator_entry_bound,
        ),
        (
            "operator_entry_tolerance",
            r.operator_accuracy.operator_entry_tolerance,
        ),
    ] {
        a.set_item(key, value)?;
    }
    a.set_item("renormalized", false)?;
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
