use crate::vegakt_convolution::{moment, moments};
use pricing::risk::VegaKtPathGammaChainAudit;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtPathGammaChainAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_multi_step_gamma_chain_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item("time_nodes", &r.time_nodes)?;
    d.set_item("gamma_bin_boundaries", &r.gamma_bin_boundaries)?;
    d.set_item("variance_risk_directions", &r.variance_risk_directions)?;
    d.set_item("subdivisions", r.subdivisions)?;
    d.set_item(
        "refinement_subdivisions",
        [r.subdivisions, 2 * r.subdivisions, 4 * r.subdivisions],
    )?;
    d.set_item("relative_tolerance", r.relative_tolerance)?;
    d.set_item("functional_tolerance", r.functional_tolerance)?;
    d.set_item(
        "gamma_relative_standard_error_limit",
        r.gamma_relative_standard_error_limit,
    )?;
    d.set_item("independent_sampling_units", r.independent_sampling_units)?;
    d.set_item(
        "sampling_unit",
        if r.points_per_scramble.is_some() {
            "scramble_mean"
        } else if r.antithetic {
            "antithetic_pair"
        } else {
            "path"
        },
    )?;
    if let Some(points) = r.points_per_scramble {
        d.set_item("points_per_scramble", points)?;
        d.set_item(
            "conditional_ad_points_per_scramble",
            r.conditional_ad_points_per_scramble,
        )?;
        d.set_item(
            "conditional_AD_sampling",
            "same_scramble_nested_prefix_reference_refinement",
        )?;
        d.set_item(
            "conditional_AD_reference_scope",
            "nonterminal_rows; terminal_pathwise_identity_uses_Gamma_points",
        )?;
        d.set_item("antithetic_within_scramble", r.antithetic)?;
        d.set_item(
            "raw_AD_variance_floor",
            "MC_reference_floor_per_physical_pair_not_RQMC_standard_error",
        )?;
    }
    d.set_item("native_price", moment(py, r.native_price)?)?;
    if let Some(samples) = &r.sampling_unit_values {
        d.set_item("sampling_unit_values", samples)?;
        d.set_item(
            "sampling_unit_values_layout",
            "counter_order_price_then_recovered_directions",
        )?;
    }
    d.set_item(
        "gamma_representation",
        "history_and_future_shock_conditional_signed_Richardson_slope_jump_measures",
    )?;
    d.set_item(
        "gamma_mass_domain",
        "finite_conditional_transition_windows_except_exact_terminal_measure",
    )?;
    d.set_item(
        "continuation_method",
        "move_branch_observation_and_all_later_states_keep_prior_history_integrate_terminal_payoff",
    )?;
    d.set_item("coordinate", "annual_local_variance_nodes")?;
    d.set_item(
        "uncertainty_scope",
        "same_sampling_unit_full_direction_covariance; refinement_and_tail_diagnostics_separate",
    )?;
    d.set_item(
        "refinement_interpretation",
        "mean_absolute_path_refinement_plus_6_SE_not_rigorous_total_error_bound",
    )?;
    d.set_item("market_calibration", "not_verified_by_native")?;
    d.set_item("full_market_iv_bucket_acceptance", "not_established")?;
    let gamma_points = r.points_per_scramble;
    let reference_points = r.conditional_ad_points_per_scramble;
    let terminal_row = r.time_nodes.len() - 2;
    let rows = r
        .rows
        .iter()
        .map(|r| {
            let d = PyDict::new(py);
            d.set_item("variance_row", r.variance_row)?;
            if let Some(points) = gamma_points {
                d.set_item(
                    "conditional_ad_points_per_scramble",
                    if r.variance_row == terminal_row {
                        Some(points)
                    } else {
                        reference_points
                    },
                )?;
            }
            d.set_item(
                "terminal_ad_conditional_variances",
                moments(py, &r.terminal_ad_conditional_variances)?,
            )?;
            d.set_item(
                "node_validation_standard_errors",
                &r.node_validation_standard_errors,
            )?;
            for (name, value) in [
                ("recovered_price", &r.recovered_price),
                ("price_difference", &r.price_difference),
                ("variance_refinement", &r.variance_refinement),
                ("price_refinement", &r.price_refinement),
                ("price_tail_bound", &r.price_tail_bound),
                ("variance_tail_bound", &r.variance_tail_bound),
                ("negative_atoms", &r.negative_atoms),
                ("bin_mass_refinement", &r.bin_mass_refinement),
            ] {
                d.set_item(name, moment(py, *value)?)?;
            }
            for (name, values) in [
                ("gamma_bin_masses", &r.gamma_bin_masses),
                (
                    "gamma_bin_variance_sensitivities",
                    &r.gamma_bin_variance_sensitivities,
                ),
                ("recovered_variance_nodes", &r.recovered_variance_nodes),
                ("native_variance_nodes", &r.native_variance_nodes),
                ("node_differences", &r.node_differences),
                (
                    "conditional_native_variance_nodes",
                    &r.conditional_native_variance_nodes,
                ),
                (
                    "conditional_node_differences",
                    &r.conditional_node_differences,
                ),
                (
                    "raw_ad_conditional_variances",
                    &r.raw_ad_conditional_variances,
                ),
                (
                    "raw_conditional_node_differences",
                    &r.raw_conditional_node_differences,
                ),
            ] {
                d.set_item(name, moments(py, values)?)?;
            }
            d.set_item(
                "gamma_relative_standard_error",
                r.gamma_relative_standard_error,
            )?;
            d.set_item(
                "raw_conditional_standard_errors",
                &r.raw_conditional_standard_errors,
            )?;
            d.set_item("raw_node_validation_passed", r.raw_node_validation_passed)?;
            d.set_item(
                "conditional_gamma_validation_passed",
                r.conditional_gamma_validation_passed,
            )?;
            d.set_item(
                "raw_conditional_validation_passed",
                r.raw_conditional_validation_passed,
            )?;
            d.set_item("recovery_passed", r.recovery_passed)?;
            d.set_item("validation_passed", r.validation_passed)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("rows", rows)?;
    d.set_item(
        "direction_relative_standard_error",
        r.direction_relative_standard_error,
    )?;
    d.set_item("joint_sample_covariance", &r.joint_sample_covariance)?;
    d.set_item("covariance_dimension", 3 * r.native_directions.len())?;
    d.set_item(
        "covariance_layout",
        "row_major_native_recovered_recovered_minus_native",
    )?;
    d.set_item("covariance_unit", "per_sampling_unit_divide_by_N_for_means")?;
    d.set_item(
        "covariance_scope",
        "raw_MC_statistics_not_total_numerical_error",
    )?;
    d.set_item("node_validation_method", "Gamma_vs_last_shock_integrated_native_AD_and_raw_vs_conditional_AD; terminal_legacy_gate_retained")?;
    d.set_item(
        "conditional_AD_method",
        "exact_payoff_branch_gradient_and_truncated_Gaussian_exponential_polynomial_moments",
    )?;
    d.set_item("node_validation_uncertainty", "max_empirical_SE_and_analytic_terminal_AD_variance_envelope; antithetic_Cauchy_bound; conditional_variance_mean_plus_6_SE")?;
    for (name, values) in [
        ("native_directions", &r.native_directions),
        ("recovered_directions", &r.recovered_directions),
        ("direction_differences", &r.direction_differences),
        (
            "direction_refinement_allowances",
            &r.direction_refinement_allowances,
        ),
    ] {
        d.set_item(name, moments(py, values)?)?;
    }
    d.set_item("recovery_passed", r.recovery_passed)?;
    d.set_item("validation_passed", r.validation_passed)?;
    d.set_item(
        "status",
        if !r.recovery_passed {
            "recovery_failed"
        } else if !r.validation_passed {
            "validation_failed"
        } else {
            "path_gamma_chain_passed"
        },
    )?;
    d.set_item(
        "estimate",
        r.estimate.as_ref().map(|v| moments(py, v)).transpose()?,
    )?;
    Ok(d)
}
