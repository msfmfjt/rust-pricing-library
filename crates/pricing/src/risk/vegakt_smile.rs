//! Positive-time two-step smile diagnostic in annual variance-node coordinates.
use super::{
    VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus, VegaKtStepMoment,
    VegaKtVarianceOperatorAccuracy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtVarianceGammaQuality {
    pub sensitivity_norm: f64,
    pub sensitivity_mc_norm: f64,
    pub reconstruction_norm: f64,
    pub reconstruction_mc_norm: f64,
    pub gamma_norm: f64,
    pub gamma_mc_norm: f64,
    pub reconstruction_margin_ratio: Option<f64>,
    pub gamma_relative_standard_error: Option<f64>,
    pub passed: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtVarianceGammaCandidate {
    pub relative_regularization: f64,
    pub sampling_index_start: u64,
    pub independent_sampling_units: u64,
    pub inverse_noise_gain_bound: f64,
    pub resolution_trace: f64,
    pub native_price: VegaKtStepMoment,
    pub unrecovered_initial_variance_sensitivity: VegaKtStepMoment,
    pub gamma_masses: Vec<VegaKtStepMoment>,
    pub variance_sensitivities: Vec<VegaKtStepMoment>,
    pub reconstruction_differences: Vec<VegaKtStepMoment>,
    pub nonzero_sampling_units: Vec<u64>,
    pub variance_sensitivity_total: VegaKtStepMoment,
    pub reconstruction_total: VegaKtStepMoment,
    pub total_difference: VegaKtStepMoment,
    pub variance_direction_risks: Vec<VegaKtStepMoment>,
    pub recovered_direction_risks: Vec<VegaKtStepMoment>,
    pub direction_differences: Vec<VegaKtStepMoment>,
    pub quality: VegaKtVarianceGammaQuality,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtTwoStepSmileGammaAudit {
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
    pub source_variance: f64,
    pub source_log_nodes: Vec<f64>,
    pub second_step_variances: Vec<f64>,
    pub gamma_strikes: Vec<f64>,
    pub operator_accuracy: VegaKtVarianceOperatorAccuracy,
    pub forward_matrix: Vec<Vec<f64>>,
    pub pilot_candidates: Vec<VegaKtVarianceGammaCandidate>,
    pub selected_regularization: Option<f64>,
    pub validation: Option<VegaKtVarianceGammaCandidate>,
    /// Present only on held-out success; never manufacture zero risk on failure.
    pub estimate: Option<Vec<VegaKtStepMoment>>,
}
