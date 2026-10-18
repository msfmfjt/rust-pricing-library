use crate::risk::{ReportingIvBasis, VegaKtGammaRecoveryAudit, VegaKtStepMoment};

/// Joint equation-(4)/(9) diagnostic, conditional on the frozen pilot choice.
/// It does not certify market-IV sensitivity or remove recovery bias.
#[derive(Clone, Debug)]
pub struct VegaKtAtomChainDiagnostic {
    pub policy_label: &'static str,
    pub reporting_basis: ReportingIvBasis,
    pub interval_variances: Vec<f64>,
    pub recovery: VegaKtGammaRecoveryAudit,
    /// None when pilot selection or held-out recovery validation failed.
    pub estimate: Option<VegaKtAtomChainEstimate>,
}

#[derive(Clone, Debug)]
pub struct VegaKtAtomChainEstimate {
    pub independent_sampling_units: u64,
    pub antithetic: bool,
    pub raw_buckets: Vec<VegaKtStepMoment>,
    pub hedge_quantities: Vec<VegaKtStepMoment>,
    pub total_iv_vega: VegaKtStepMoment,
    /// Parallel shift of every discrete interval's LOCAL volatility.
    pub native_parallel_local_vega: VegaKtStepMoment,
    /// Compare to total_iv_vega only for constant volatility term structure.
    pub iv_minus_local_parallel: VegaKtStepMoment,
    pub constant_volatility_term_structure: bool,
    pub native_price: VegaKtStepMoment,
    /// Per sampling unit, not covariance of estimated means (divide by N).
    pub bucket_price_sample_covariance: Vec<f64>,
    /// Row-major B x B; includes correlations across all reporting maturities.
    pub full_bucket_sample_covariance: Option<Vec<f64>>,
}
