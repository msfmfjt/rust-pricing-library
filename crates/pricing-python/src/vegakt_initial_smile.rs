use crate::vegakt_convolution::moment;
use pricing::risk::{VegaKtInitialSmileGammaAudit, VegaKtInitialSmileResolution};
use pyo3::prelude::*;
use pyo3::types::PyDict;
fn resolution<'py>(
    py: Python<'py>,
    r: &VegaKtInitialSmileResolution,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("subdivisions", r.subdivisions)?;
    d.set_item("interpolation_nodes", r.interpolation_nodes)?;
    d.set_item("gamma_bin_masses", &r.gamma_bin_masses)?;
    d.set_item(
        "gamma_bin_variance_sensitivities",
        &r.gamma_bin_variance_sensitivities,
    )?;
    d.set_item("total_gamma_mass", r.total_gamma_mass)?;
    d.set_item(
        "recovered_variance_sensitivity",
        r.recovered_variance_sensitivity,
    )?;
    d.set_item("interpolated_price", r.interpolated_price)?;
    Ok(d)
}
pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtInitialSmileGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_initial_smile_gamma_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item(
        "policy_label",
        "two_step_initial_continuation_gamma_refinement_v1",
    )?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item(
        "scope",
        "two_intervals_flat_first_variance_nonflat_second_variance",
    )?;
    d.set_item("coordinate", "first_interval_annual_variance")?;
    d.set_item(
        "gamma_representation",
        "signed_slope_jump_atoms_of_refined_continuation_interpolant",
    )?;
    d.set_item(
        "gamma_mass_domain",
        "finite_log_window_only_tail_mass_not_established",
    )?;
    d.set_item(
        "bin_convention",
        "[0,b0),[b0,b1),...,[bn,infinity), intersected with log_window",
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
    d.set_item("maturity", r.maturity)?;
    d.set_item("first_step_variance", r.first_step_variance)?;
    d.set_item("gamma_bin_boundaries", &r.gamma_bin_boundaries)?;
    d.set_item("log_window", r.log_window)?;
    d.set_item("continuation_growth_bound", r.continuation_growth_bound)?;
    for (key, v) in [
        ("relative_tolerance", r.relative_tolerance),
        ("functional_tolerance", r.functional_tolerance),
        ("mass_tolerance", r.mass_tolerance),
        ("price_tail_bound", r.price_tail_bound),
        ("variance_tail_bound", r.variance_tail_bound),
        ("quadrature_price", r.quadrature_price),
        (
            "quadrature_variance_sensitivity",
            r.quadrature_variance_sensitivity,
        ),
        ("quadrature_price_difference", r.quadrature_price_difference),
        (
            "quadrature_variance_difference",
            r.quadrature_variance_difference,
        ),
        ("price_refinement_difference", r.price_refinement_difference),
        (
            "variance_refinement_difference",
            r.variance_refinement_difference,
        ),
        (
            "bin_mass_refinement_difference",
            r.bin_mass_refinement_difference,
        ),
        (
            "bin_variance_refinement_difference",
            r.bin_variance_refinement_difference,
        ),
    ] {
        d.set_item(key, v)?;
    }
    d.set_item("coarse", resolution(py, &r.coarse)?)?;
    d.set_item("fine", resolution(py, &r.fine)?)?;
    d.set_item("recovery_passed", r.recovery_passed)?;
    d.set_item("validation_passed", r.validation_passed)?;
    d.set_item(
        "status",
        if !r.recovery_passed {
            "recovery_failed"
        } else if !r.validation_passed {
            "validation_failed"
        } else {
            "initial_variance_functional_passed"
        },
    )?;
    d.set_item("estimate", r.estimate)?;
    d.set_item("independent_sampling_units", r.independent_sampling_units)?;
    d.set_item(
        "sampling_unit",
        if r.antithetic {
            "antithetic_pair"
        } else {
            "path"
        },
    )?;
    d.set_item("validation_mc_multiplier", 6.)?;
    for (key, m) in [
        ("native_price", r.native_price),
        (
            "native_initial_variance_sensitivity",
            r.native_initial_variance_sensitivity,
        ),
        ("price_difference", r.price_difference),
        ("variance_difference", r.variance_difference),
    ] {
        d.set_item(key, moment(py, m)?)?;
    }
    Ok(d)
}
