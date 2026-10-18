use crate::vegakt_convolution::moments;
use pricing::risk::{VegaKtJointGammaAudit, VegaKtJointGammaStatistics};
use pyo3::prelude::*;
use pyo3::types::PyDict;
fn statistics<'py>(
    py: Python<'py>,
    s: &VegaKtJointGammaStatistics,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("independent_sampling_units", s.independent_sampling_units)?;
    d.set_item(
        "sampling_unit",
        if s.antithetic {
            "antithetic_pair"
        } else {
            "path"
        },
    )?;
    d.set_item("native_directions", moments(py, &s.native_directions)?)?;
    d.set_item(
        "recovered_directions",
        moments(py, &s.recovered_directions)?,
    )?;
    d.set_item("differences", moments(py, &s.differences)?)?;
    d.set_item("joint_sample_covariance", &s.joint_sample_covariance)?;
    d.set_item(
        "covariance_layout",
        ["native_directions", "recovered_directions", "differences"],
    )?;
    d.set_item("covariance_dimension", 3 * s.native_directions.len())?;
    d.set_item("covariance_unit", "per_sampling_unit_divide_by_N_for_means")?;
    d.set_item("initial_error_allowances", &s.initial_error_allowances)?;
    d.set_item("roundoff_allowances", &s.roundoff_allowances)?;
    d.set_item("passed", s.passed)?;
    Ok(d)
}
pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtJointGammaAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_joint_gamma_audit")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("coordinate", "caller_supplied_annual_variance_directions")?;
    d.set_item("market_calibration", "not_verified_by_native")?;
    d.set_item("full_market_iv_bucket_acceptance", "not_established")?;
    d.set_item("uncertainty_scope","same_path_MC_conditional_on_pilot_choice; deterministic_initial_Gamma_error_excluded_from_SE")?;
    d.set_item(
        "variance_conservation_directions",
        &r.directions.conservation_basis,
    )?;
    d.set_item("risk_directions", &r.directions.risk_directions)?;
    d.set_item(
        "initial",
        crate::vegakt_initial_smile::to_dict(py, &r.initial)?,
    )?;
    d.set_item("second", crate::vegakt_smile::to_dict(py, &r.second)?)?;
    d.set_item(
        "joint_validation",
        r.joint_validation
            .as_ref()
            .map(|s| statistics(py, s))
            .transpose()?,
    )?;
    let valid = r.joint_validation.as_ref().filter(|s| s.passed);
    d.set_item("estimate", valid.map(|s| statistics(py, s)).transpose()?)?;
    d.set_item(
        "status",
        if r.initial.estimate.is_none() {
            "initial_recovery_failed"
        } else if r.second.estimate.is_none() {
            "second_recovery_failed"
        } else if valid.is_none() {
            "joint_validation_failed"
        } else {
            "joint_direction_policy_passed"
        },
    )?;
    Ok(d)
}
