use crate::vegakt_convolution::{moment, moments};
use pricing::risk::VegaKtAtomChainDiagnostic;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    r: &VegaKtAtomChainDiagnostic,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("document_kind", "vega_kt_atom_chain_diagnostic")?;
    d.set_item("schema_version", 1)?;
    d.set_item("diagnostic_only", true)?;
    d.set_item("policy_label", r.policy_label)?;
    d.set_item("reporting_maturities", r.reporting_basis.maturity_nodes())?;
    d.set_item(
        "reporting_log_nodes",
        r.reporting_basis.log_moneyness_nodes(),
    )?;
    d.set_item(
        "reporting_implied_volatilities",
        r.reporting_basis.implied_volatilities(),
    )?;
    d.set_item("interval_variances", &r.interval_variances)?;
    d.set_item(
        "recovery",
        crate::vegakt_selection::to_dict(py, &r.recovery)?,
    )?;
    d.set_item("iv_bucket_acceptance", "not_established")?;
    d.set_item(
        "propagation",
        "analytic_gaussian_product_over_all_positive_strikes",
    )?;
    d.set_item("reporting_tails", "clamped_log_hats_no_integration_cutoff")?;
    d.set_item(
        "uncertainty_scope",
        "joint_MC_conditional_on_pilot_choice_excludes_recovery_bias",
    )?;
    d.set_item("sampling_index_start", 0)?;
    d.set_item(
        "sampling_index_end_exclusive",
        r.recovery.validation_sampling_units,
    )?;
    if let Some(e) = &r.estimate {
        let v = PyDict::new(py);
        v.set_item("independent_sampling_units", e.independent_sampling_units)?;
        v.set_item("antithetic", e.antithetic)?;
        v.set_item("raw_buckets", moments(py, &e.raw_buckets)?)?;
        v.set_item("hedge_quantities", moments(py, &e.hedge_quantities)?)?;
        v.set_item("total_iv_vega", moment(py, e.total_iv_vega)?)?;
        v.set_item(
            "native_parallel_local_vega",
            moment(py, e.native_parallel_local_vega)?,
        )?;
        v.set_item(
            "iv_minus_local_parallel",
            moment(py, e.iv_minus_local_parallel)?,
        )?;
        v.set_item(
            "constant_volatility_term_structure",
            e.constant_volatility_term_structure,
        )?;
        v.set_item("native_price", moment(py, e.native_price)?)?;
        v.set_item(
            "bucket_price_sample_covariance",
            &e.bucket_price_sample_covariance,
        )?;
        v.set_item(
            "full_bucket_sample_covariance",
            &e.full_bucket_sample_covariance,
        )?;
        v.set_item(
            "covariance_units",
            "per_sampling_unit_divide_by_N_for_covariance_of_means",
        )?;
        d.set_item("estimate", v)?;
    } else {
        d.set_item("estimate", py.None())?;
    }
    Ok(d)
}
