//! Variable interval count; only the last annual-variance row is recovered.
use super::{
    VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus, VegaKtMultiTransitionSourceAccuracy,
    VegaKtStepMoment, VegaKtVarianceGammaCandidate,
};
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtMultiStepTerminalGammaAudit {
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub policy: VegaKtGammaRecoveryPolicy,
    pub variance_conservation_directions: Vec<Vec<f64>>,
    pub status: VegaKtGammaRecoveryStatus,
    pub validation_sampling_units: u64,
    pub antithetic: bool,
    pub pilot_index_start: u64,
    pub pilot_index_end_exclusive: u64,
    pub source_maturity: f64,
    pub terminal_maturity: f64,
    pub time_nodes: Vec<f64>,
    pub variance_rows: Vec<Vec<f64>>,
    pub source_log_nodes: Vec<f64>,

    pub gamma_strikes: Vec<f64>,
    pub operator_accuracy: VegaKtMultiTransitionSourceAccuracy,
    pub forward_matrix: Vec<Vec<f64>>,
    pub pilot_candidates: Vec<VegaKtVarianceGammaCandidate>,
    pub selected_regularization: Option<f64>,
    pub validation: Option<VegaKtVarianceGammaCandidate>,
    /// Present only on held-out success; never manufacture zero risk on failure.
    pub estimate: Option<Vec<VegaKtStepMoment>>,
}
