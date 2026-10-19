use pricing::risk::VegaKtThreeStepInitialGammaAudit;
use pyo3::prelude::*;
use pyo3::types::PyDict;
pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtThreeStepInitialGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = crate::vegakt_initial_smile::to_dict(py, &r.audit)?;
    d.set_item("document_kind", "vega_kt_three_step_initial_gamma_audit")?;
    d.set_item(
        "scope",
        "three_intervals_flat_first_variance_nonflat_middle_and_terminal_variances",
    )?;
    d.set_item(
        "policy_label",
        "three_step_initial_continuation_gamma_refinement_v1",
    )?;
    d.set_item(
        "gamma_representation",
        "signed_two_stage_richardson_slope_jump_measure",
    )?;
    d.set_item(
        "history_interpretation",
        "first_observation_and_both_conditional_future_transitions_move_with_initial_state",
    )?;
    d.set_item("refinement_subdivisions", r.refinement_subdivisions)?;
    d.set_item("continuation_evaluations", r.continuation_evaluations)?;
    d.set_item(
        "continuation_relative_refinement",
        r.continuation_relative_refinement,
    )?;
    d.set_item(
        "continuation_price_refinement_allowance",
        r.continuation_price_refinement_allowance,
    )?;
    d.set_item(
        "continuation_variance_refinement_allowance",
        r.continuation_variance_refinement_allowance,
    )?;
    d.set_item(
        "continuation_price_tail_bound",
        r.continuation_price_tail_bound,
    )?;
    d.set_item(
        "continuation_variance_tail_bound",
        r.continuation_variance_tail_bound,
    )?;
    d.set_item("continuation_source_moment_relative_tolerance", 1e-10)?;
    d.set_item("continuation_source_renormalized", false)?;
    Ok(d)
}
