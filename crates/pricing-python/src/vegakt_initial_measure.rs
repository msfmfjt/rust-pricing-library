use crate::vegakt_convolution::{moment, moments};
use pricing::risk::VegaKtInitialMeasureDiagnostic;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtInitialMeasureDiagnostic,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_initial_gamma_measure_diagnostic")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("policy_label", r.policy_label)?;
    d.set_item("request_fingerprint", r.request_fingerprint.to_string())?;
    d.set_item("plan_fingerprint", r.plan_fingerprint.to_string())?;
    d.set_item("maturity", r.maturity)?;
    d.set_item("first_step_variance", r.first_step_variance)?;
    d.set_item("gamma_bin_boundaries", &r.gamma_bin_boundaries)?;
    d.set_item("gamma_bin_masses", moments(py, &r.gamma_bin_masses)?)?;
    d.set_item(
        "gamma_bin_first_step_vegas",
        moments(py, &r.gamma_bin_first_step_vegas)?,
    )?;
    d.set_item(
        "gamma_bin_nonzero_sampling_units",
        &r.gamma_bin_nonzero_sampling_units,
    )?;
    d.set_item("independent_sampling_units", r.independent_sampling_units)?;
    d.set_item(
        "sampling_unit",
        if r.antithetic {
            "antithetic_pair"
        } else {
            "independent_path"
        },
    )?;
    d.set_item("initial_gamma_status", "conditional_payoff_measure")?;
    d.set_item("iv_bucket_acceptance", "not_established")?;
    d.set_item(
        "gamma_representation",
        "signed_bin_masses_including_both_tails",
    )?;
    d.set_item("bin_convention", "[0,b0),[b0,b1),...,[bn,infinity)")?;
    d.set_item("gamma_unit", "currency_per_unit_spot")?;
    d.set_item(
        "uncertainty_scope",
        "MC_over_future_shocks_first_interval_integrated_exactly",
    )?;
    for (key, m) in [
        ("cash", r.cash),
        ("slope", r.slope),
        ("total_gamma_mass", r.total_gamma_mass),
        (
            "recovered_initial_local_vega",
            r.recovered_initial_local_vega,
        ),
        ("native_initial_local_vega", r.native_initial_local_vega),
        (
            "initial_local_vega_difference",
            r.initial_local_vega_difference,
        ),
        ("conditional_price", r.conditional_price),
        ("native_price", r.native_price),
        ("price_difference", r.price_difference),
        (
            "payoff_reconstruction_difference",
            r.payoff_reconstruction_difference,
        ),
    ] {
        d.set_item(key, moment(py, m)?)?;
    }
    Ok(d)
}
