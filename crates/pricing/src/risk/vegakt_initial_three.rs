//! Initial Gamma for a continuation with two subsequent local-variance transitions.
use super::VegaKtInitialSmileGammaAudit;
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtThreeStepInitialGammaAudit {
    pub audit: VegaKtInitialSmileGammaAudit,
    pub refinement_subdivisions: [usize; 4],
    pub continuation_evaluations: usize,
    pub continuation_relative_refinement: f64,
    pub continuation_price_refinement_allowance: f64,
    pub continuation_variance_refinement_allowance: f64,
    pub continuation_price_tail_bound: f64,
    pub continuation_variance_tail_bound: f64,
}
