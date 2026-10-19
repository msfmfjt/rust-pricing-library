use crate::vegakt_convolution::moment;
use pricing::risk::{VegaKtMiddleGammaResolution, VegaKtMiddleSmileGammaAudit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
fn resolution<'py>(
    py: Python<'py>,
    r: &VegaKtMiddleGammaResolution,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("history_order", r.history_order)?;
    d.set_item("subdivisions", r.subdivisions)?;
    d.set_item("histories", r.histories)?;
    d.set_item("continuation_evaluations", r.continuation_evaluations)?;
    d.set_item("recovered_price", r.recovered_price)?;
    d.set_item("quadrature_price", r.quadrature_price)?;
    d.set_item("inner_price_refinement", r.inner_price_refinement)?;
    d.set_item("inner_variance_refinement", r.inner_variance_refinement)?;
    d.set_item("inner_bin_mass_refinement", r.inner_bin_mass_refinement)?;
    d.set_item("score_price_refinement", r.score_price_refinement)?;
    d.set_item("score_variance_refinement", r.score_variance_refinement)?;
    d.set_item("price_tail_bound", r.price_tail_bound)?;
    d.set_item("variance_tail_bound", r.variance_tail_bound)?;
    d.set_item("negative_fine_gamma_atoms", r.negative_fine_gamma_atoms)?;
    d.set_item(
        "variance_node_sensitivities",
        &r.variance_node_sensitivities,
    )?;
    d.set_item(
        "history_averaged_gamma_bin_masses",
        &r.history_averaged_gamma_bin_masses,
    )?;
    d.set_item(
        "gamma_bin_variance_sensitivities",
        &r.gamma_bin_variance_sensitivities,
    )?;
    d.set_item(
        "quadrature_variance_node_sensitivities",
        &r.quadrature_variance_node_sensitivities,
    )?;
    Ok(d)
}
pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtMiddleSmileGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_middle_smile_gamma_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item(
        "scope",
        "three_intervals_flat_first_variance_middle_and_terminal_smiles",
    )?;
    d.set_item("coordinate", "second_interval_annual_variance_nodes")?;
    d.set_item(
        "gamma_representation",
        "history_conditional_signed_two_stage_richardson_slope_jump_measures",
    )?;
    d.set_item(
        "gamma_mass_domain",
        "history_averaged_finite_conditional_windows_only_tail_mass_not_established",
    )?;
    d.set_item(
        "bin_convention",
        "[0,b0),[b0,b1),...,[bn,infinity), intersected with each conditional log window",
    )?;
    d.set_item(
        "uncertainty_scope",
        "deterministic_refinement_not_statistical_SE; native_MC_validation_separate",
    )?;
    d.set_item(
        "quadrature_interpretation",
        "32_64_point_convergence_diagnostic_not_rigorous_total_error_bound",
    )?;
    d.set_item("full_market_iv_bucket_acceptance", "not_established")?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item("gamma_bin_boundaries", &r.gamma_bin_boundaries)?;
    d.set_item("continuation_growth_bound", r.continuation_growth_bound)?;
    d.set_item("relative_tolerance", r.relative_tolerance)?;
    d.set_item("functional_tolerance", r.functional_tolerance)?;
    d.set_item("mass_tolerance", r.mass_tolerance)?;
    d.set_item("outer_price_refinement", r.outer_price_refinement)?;
    d.set_item("outer_variance_refinement", r.outer_variance_refinement)?;
    d.set_item("outer_bin_mass_refinement", r.outer_bin_mass_refinement)?;
    d.set_item("recovery_passed", r.recovery_passed)?;
    d.set_item("validation_passed", r.validation_passed)?;
    d.set_item("independent_sampling_units", r.independent_sampling_units)?;
    d.set_item("coarse_history", resolution(py, &r.coarse_history)?)?;
    d.set_item("fine_history", resolution(py, &r.fine_history)?)?;
    d.set_item(
        "status",
        if !r.recovery_passed {
            "recovery_failed"
        } else if !r.validation_passed {
            "validation_failed"
        } else {
            "middle_variance_functional_passed"
        },
    )?;
    d.set_item("estimate", &r.estimate)?;
    d.set_item(
        "sampling_unit",
        if r.antithetic {
            "antithetic_stratified_sweep"
        } else {
            "stratified_sweep"
        },
    )?;
    d.set_item("validation_mc_multiplier", 6.)?;
    d.set_item(
        "validation_sampling_strategy",
        "uniform_first_state_cell_sweeps_with_original_gaussian_density_weights_no_renormalization",
    )?;
    d.set_item("validation_history_strata", r.validation_history_strata)?;
    d.set_item(
        "validation_history_log_window",
        r.validation_history_log_window,
    )?;
    d.set_item("native_price", moment(py, r.native_price)?)?;
    d.set_item("price_difference", moment(py, r.price_difference)?)?;
    for (name, values) in [
        (
            "native_middle_variance_sensitivities",
            &r.native_middle_variance_sensitivities,
        ),
        ("variance_differences", &r.variance_differences),
    ] {
        let list = PyList::empty(py);
        for value in values {
            list.append(moment(py, *value)?)?;
        }
        d.set_item(name, list)?;
    }
    Ok(d)
}
