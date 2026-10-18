use crate::risk::{
    ReportingIvBasis, VegaKtInitialGammaPlan, VegaKtStepMoment, VegaKtStepRecoveryPlan,
    VegaKtStepSamplingUnit,
};
use std::sync::Arc;

pub const VEGA_KT_CHAIN_LABEL: &str = "flat_variance_paper_chain_diagnostic_v1";
#[derive(Clone, Debug)]
pub struct VegaKtChainPlan {
    pub(crate) layout: Arc<ChainLayout>,
}
#[derive(Clone, Debug)]
pub(crate) struct ChainLayout {
    pub initial: VegaKtInitialGammaPlan,
    pub steps: VegaKtStepRecoveryPlan,
    pub basis: ReportingIvBasis,
    pub rows: Vec<ChainRow>,
    pub subdivisions: usize,
    pub components: usize,
    pub integrated_denominators: Option<Vec<Vec<f64>>>,
}
#[derive(Clone, Debug)]
pub(crate) struct ChainRow {
    pub audit: VegaKtChainRowAudit,
    pub fine: Vec<Vec<f64>>,
    pub coarse: Vec<Vec<f64>>,
    pub reporting_edges: Vec<Vec<f64>>,
    pub propagation: Vec<Vec<f64>>,
    pub tails: Vec<Vec<f64>>,
    pub nodal_vegas: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtChainRowAudit {
    pub maturity: f64,
    /// Equation (9) is integrated only over this finite interval.
    pub strike_bounds: [f64; 2],
    /// Next Gamma is extrapolated constantly outside these bounds in equation (4).
    pub next_gamma_bounds: Option<[f64; 2]>,
    /// None for the separately estimated initial Gamma row.
    pub recovery_source_maturity: Option<f64>,
    pub excluded_density_probability: Option<f64>,
    /// Time of the density used in the recovery denominator.
    pub recovery_density_maturity: Option<f64>,
}
#[derive(Clone, Debug)]
pub struct VegaKtChainSample {
    pub(crate) layout: Arc<ChainLayout>,
    pub(crate) values: Box<[f64]>,
    pub(crate) sampling_unit: VegaKtStepSamplingUnit,
}
/// Joint sample statistics, conditional on fixed density, domains and quadrature.
/// Standard errors exclude recovery, finite-difference and truncation biases.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtChainEstimate {
    pub policy_label: &'static str,
    pub reporting_basis: ReportingIvBasis,
    pub independent_sampling_units: u64,
    pub sampling_unit: VegaKtStepSamplingUnit,
    pub coarse_subdivisions: usize,
    pub rows: Vec<VegaKtChainRowAudit>,
    pub position_iv_buckets: Vec<VegaKtStepMoment>,
    pub offsetting_hedge_quantities: Vec<VegaKtStepMoment>,
    /// Fine Gamma minus coarse Gamma, using the fine quadrature in both.
    pub initial_bump_bucket_differences: Vec<VegaKtStepMoment>,
    /// Fine (2q) minus coarse (q) midpoint integration, using fine Gamma in both.
    pub quadrature_bucket_differences: Vec<VegaKtStepMoment>,
    pub total_position_iv_vega: VegaKtStepMoment,
    pub total_initial_bump_difference: VegaKtStepMoment,
    pub total_quadrature_difference: VegaKtStepMoment,
    /// One value per positive source interval, in LOCAL volatility units.
    pub excluded_local_vegas: Vec<VegaKtStepMoment>,
    /// Reconstructed initial Vega minus native first-interval AD, in LOCAL units.
    pub initial_reconstruction_difference: VegaKtStepMoment,
    pub initial_bump_vega_difference: VegaKtStepMoment,
    pub initial_tail_vegas: [VegaKtStepMoment; 2],
    /// Position IV contribution of constant next-Gamma tails, per reporting row.
    /// Not an error bound, nor the omitted tails of equation (9).
    pub propagated_tail_iv_vegas: Vec<VegaKtStepMoment>,
    /// Left/right IV risk allocated to the nearest reporting strike outside its range.
    pub reporting_edge_iv_vegas: Vec<[VegaKtStepMoment; 2]>,
}

/// Explicit diagnostic output, separate from accepted pricing risk reports.
/// Method parameters supplement the ordinary request and plan fingerprints.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtChainDiagnostic {
    pub estimate: VegaKtChainEstimate,
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub initial_strikes: Box<[f64]>,
    pub requested_initial_strikes: Box<[f64]>,
    pub recovery_log_moneyness_nodes: Box<[f64]>,
    pub maximum_recovery_log_spacing: f64,
    pub relative_bump: f64,
    pub master_seed: u64,
    pub brownian_bridge: bool,
    pub evaluated_paths: u128,
}
