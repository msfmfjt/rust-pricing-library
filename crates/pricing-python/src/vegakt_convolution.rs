use pricing::risk::{VegaKtConvolutionDiagnostic, VegaKtStepMoment};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn moment(py: Python<'_>, m: VegaKtStepMoment) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mean", m.mean())?;
    d.set_item("sample_variance", m.sample_variance())?;
    d.set_item("standard_error", m.standard_error())?;
    Ok(d)
}
pub(crate) fn moments<'py>(
    py: Python<'py>,
    m: &[VegaKtStepMoment],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    m.iter().map(|&v| moment(py, v)).collect()
}
pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtConvolutionDiagnostic,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_gamma_recovery_diagnostic")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("policy_label", r.policy_label)?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item("gamma_strikes", &r.gamma_strikes)?;
    d.set_item("source_log_nodes", &r.source_log_nodes)?;
    d.set_item("relative_regularization", r.relative_regularization)?;
    if r.preserves_local_vega_total {
        d.set_item(
            "local_vega_total_constraint",
            "equality_constrained_tikhonov_fixed_linear_inverse",
        )?;
    }
    if !r.conservation_directions.is_empty() {
        d.set_item(
            "local_vega_conservation_directions",
            &r.conservation_directions,
        )?;
        d.set_item(
            "direction_constraint_method",
            "simultaneous_equality_constrained_tikhonov_fixed_linear_inverse",
        )?;
    }
    d.set_item("independent_sampling_units", r.independent_sampling_units)?;
    d.set_item(
        "sampling_unit",
        if r.antithetic {
            "antithetic_pair"
        } else {
            "independent_path"
        },
    )?;
    d.set_item(
        "initial_gamma_status",
        "not_identifiable_from_initial_local_vega",
    )?;
    d.set_item(
        "unrecovered_initial_local_vega",
        moment(py, r.unrecovered_initial_local_vega)?,
    )?;
    d.set_item("iv_bucket_acceptance", "not_established")?;
    d.set_item("gamma_unit", "currency_per_unit_spot")?;
    d.set_item("gamma_representation", "signed_atoms_not_nodal_densities")?;
    d.set_item(
        "uncertainty_scope",
        "MC_only_excludes_regularization_and_finite_domain_bias",
    )?;
    let mut rows = Vec::new();
    for r in &r.rows {
        let row = PyDict::new(py);
        row.set_item("source_maturity", r.source_maturity)?;
        row.set_item("recovered_maturity", r.recovered_maturity)?;
        row.set_item("source_total_variance", r.source_total_variance)?;
        row.set_item("step_variance", r.step_variance)?;
        row.set_item("operator_scale", r.operator_scale)?;
        row.set_item("inverse_noise_gain_bound", r.inverse_noise_gain_bound)?;
        row.set_item("resolution_trace", r.resolution_trace)?;
        row.set_item("gamma_masses", moments(py, &r.gamma_masses)?)?;
        row.set_item("local_vega", moments(py, &r.local_vega)?)?;
        row.set_item(
            "local_vega_nonzero_sampling_units",
            &r.local_vega_nonzero_sampling_units,
        )?;
        row.set_item(
            "reconstruction_difference",
            moments(py, &r.reconstruction_difference)?,
        )?;
        row.set_item("local_vega_total", moment(py, r.local_vega_total)?)?;
        row.set_item("reconstructed_total", moment(py, r.reconstructed_total)?)?;
        row.set_item("total_difference", moment(py, r.total_difference)?)?;
        if !r.local_vega_directions.is_empty() {
            row.set_item(
                "local_vega_directions",
                moments(py, &r.local_vega_directions)?,
            )?;
            row.set_item(
                "reconstructed_directions",
                moments(py, &r.reconstructed_directions)?,
            )?;
            row.set_item(
                "direction_differences",
                moments(py, &r.direction_differences)?,
            )?;
        }
        rows.push(row);
    }
    d.set_item("rows", rows)?;
    Ok(d)
}
