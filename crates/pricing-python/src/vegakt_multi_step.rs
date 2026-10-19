use crate::vegakt_convolution::moments;
use crate::vegakt_smile::candidate;
use pricing::risk::VegaKtMultiStepTerminalGammaAudit;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtMultiStepTerminalGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_multi_step_terminal_gamma_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item(
        "variance_risk_estimator",
        "analytic_terminal_payoff_conditioning_on_entire_prefix_history",
    )?;
    d.set_item("native_price_estimator", "unchanged_compiled_full_path")?;
    d.set_item("coordinate", "annual_local_variance_nodes")?;
    d.set_item(
        "scope",
        "variable_intervals_recursive_non_gaussian_marginals_terminal_row_only",
    )?;
    d.set_item(
        "policy_label",
        "multi_step_terminal_variance_gamma_split_sample_v1",
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
    let count = r.time_nodes.len() - 1;
    d.set_item("time_nodes", &r.time_nodes)?;
    d.set_item("variance_rows", &r.variance_rows)?;
    d.set_item("interval_count", count)?;
    d.set_item(
        "source_distribution",
        "successively_integrated_log_euler_transition_densities",
    )?;
    d.set_item("source_log_nodes", &r.source_log_nodes)?;
    d.set_item("recovered_variance_row", count - 1)?;
    d.set_item(
        "unrecovered_variance_rows",
        (0..count - 1).collect::<Vec<_>>(),
    )?;
    d.set_item("earlier_gamma_recovery", "not_implemented_by_this_audit")?;
    d.set_item("gamma_strikes", &r.gamma_strikes)?;
    d.set_item("forward_matrix", &r.forward_matrix)?;
    let a = PyDict::new(py);
    a.set_item("quadrature_order", r.operator_accuracy.quadrature_order)?;
    let steps = r
        .operator_accuracy
        .steps
        .iter()
        .map(|v| {
            let d = PyDict::new(py);
            d.set_item("transition_count", v.transition_count)?;
            d.set_item("log_window", v.log_window)?;
            d.set_item("probability", v.probability)?;
            d.set_item("first_spot_moment", v.first_spot_moment)?;
            d.set_item("probability_tail_bound", v.probability_tail_bound)?;
            d.set_item("spot_tail_bound", v.spot_tail_bound)?;
            d.set_item(
                "probability_refinement_difference",
                v.probability_refinement_difference,
            )?;
            d.set_item("spot_refinement_difference", v.spot_refinement_difference)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    a.set_item("steps", steps)?;
    a.set_item(
        "operator_refinement_difference",
        r.operator_accuracy.operator_refinement_difference,
    )?;
    a.set_item(
        "omitted_operator_entry_bound",
        r.operator_accuracy.omitted_operator_entry_bound,
    )?;
    a.set_item(
        "operator_entry_tolerance",
        r.operator_accuracy.operator_entry_tolerance,
    )?;
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
