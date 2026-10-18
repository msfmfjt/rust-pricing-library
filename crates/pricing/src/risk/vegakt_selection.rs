//! Explicit split-sample regularization policy. This is not IV acceptance.
use crate::risk::{RiskConfigError, VegaKtConvolutionDiagnostic, VegaKtInitialMeasureDiagnostic};

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtGammaRecoveryPolicy {
    pub(crate) candidate_regularizations: Vec<f64>,
    pub(crate) pilot_sampling_units: u64,
    pub(crate) reconstruction_relative_limit: f64,
    pub(crate) gamma_relative_standard_error_limit: f64,
    pub(crate) mc_margin_multiplier: f64,
    pub(crate) preserve_local_vega_total: bool,
    pub(crate) local_vega_conservation_directions: Vec<Vec<Vec<f64>>>,
}
impl VegaKtGammaRecoveryPolicy {
    /// Candidates are fixed before sampling, strictly strongest to weakest.
    /// The pilot uses disjoint counter indices after the original pricing range.
    pub fn new(
        candidate_regularizations: Vec<f64>,
        pilot_sampling_units: u64,
        reconstruction_relative_limit: f64,
        gamma_relative_standard_error_limit: f64,
        mc_margin_multiplier: f64,
    ) -> Result<Self, RiskConfigError> {
        if candidate_regularizations.is_empty()
            || candidate_regularizations.len() > 8
            || candidate_regularizations
                .iter()
                .any(|v| !(1e-6..=1.).contains(v))
            || candidate_regularizations.windows(2).any(|p| p[0] <= p[1])
            || !(2..=10_000_000).contains(&pilot_sampling_units)
            || !reconstruction_relative_limit.is_finite()
            || reconstruction_relative_limit <= 0.
            || reconstruction_relative_limit >= 1.
            || !gamma_relative_standard_error_limit.is_finite()
            || gamma_relative_standard_error_limit <= 0.
            || gamma_relative_standard_error_limit >= 1.
            || !(1.0..=10.).contains(&mc_margin_multiplier)
        {
            return Err(RiskConfigError::VegaKtChainInput {
                quantity: "invalid Gamma recovery selection policy",
            });
        }
        Ok(Self {
            candidate_regularizations,
            pilot_sampling_units,
            reconstruction_relative_limit,
            gamma_relative_standard_error_limit,
            mc_margin_multiplier,
            preserve_local_vega_total: false,
            local_vega_conservation_directions: Vec::new(),
        })
    }
    /// Use an equality-constrained inverse for every pilot and validation row.
    /// Spatial reconstruction and Gamma noise limits are still enforced.
    #[must_use]
    pub fn with_local_vega_conservation(mut self) -> Self {
        self.preserve_local_vega_total = true;
        self
    }
    #[must_use]
    pub const fn preserves_local_vega_total(&self) -> bool {
        self.preserve_local_vega_total
    }
    /// One matrix per positive-time recovery row, with a common number of
    /// directions and source nodes. Frozen before pilot/validation sampling.
    pub fn with_local_vega_direction_conservation(
        mut self,
        directions: Vec<Vec<Vec<f64>>>,
    ) -> Result<Self, RiskConfigError> {
        let fail = || RiskConfigError::VegaKtChainInput {
            quantity: "invalid Gamma conservation direction tensor",
        };
        if directions.is_empty() || directions.len() > 64 {
            return Err(fail());
        }
        let count = directions[0].len();
        if !(1..=8).contains(&count) {
            return Err(fail());
        }
        let width = directions[0][0].len();
        if !(2..=129).contains(&width)
            || directions.len() * count * width > 16384
            || directions.iter().any(|row| {
                row.len() != count
                    || row.iter().any(|v| {
                        let norm = v.iter().fold(0., |a, &b| f64::hypot(a, b));
                        v.len() != width
                            || v.iter().any(|x| !x.is_finite())
                            || !norm.is_finite()
                            || norm <= 0.
                    })
            })
        {
            return Err(fail());
        }
        self.local_vega_conservation_directions = directions;
        Ok(self)
    }
    #[must_use]
    pub fn local_vega_conservation_directions(&self) -> &[Vec<Vec<f64>>] {
        &self.local_vega_conservation_directions
    }
    #[must_use]
    pub fn candidate_regularizations(&self) -> &[f64] {
        &self.candidate_regularizations
    }
    #[must_use]
    pub const fn pilot_sampling_units(&self) -> u64 {
        self.pilot_sampling_units
    }
    #[must_use]
    pub const fn reconstruction_relative_limit(&self) -> f64 {
        self.reconstruction_relative_limit
    }
    #[must_use]
    pub const fn gamma_relative_standard_error_limit(&self) -> f64 {
        self.gamma_relative_standard_error_limit
    }
    #[must_use]
    pub const fn mc_margin_multiplier(&self) -> f64 {
        self.mc_margin_multiplier
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtGammaRowQuality {
    pub source_maturity: f64,
    pub local_vega_norm: f64,
    pub local_vega_mc_norm: f64,
    pub reconstruction_difference_norm: f64,
    pub reconstruction_mc_norm: f64,
    pub gamma_mass_norm: f64,
    pub gamma_mc_norm: f64,
    /// (||mean residual|| + m*||SE residual||) / (||mean input|| - m*||SE input||).
    /// An estimated MC margin, not a calibrated simultaneous confidence bound.
    /// None means the denominator does not identify a nonzero input signal.
    pub reconstruction_margin_ratio: Option<f64>,
    pub gamma_relative_standard_error: Option<f64>,
    pub unobserved_source_nodes: Vec<usize>,
    pub passed: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtGammaCandidateAudit {
    pub diagnostic: VegaKtConvolutionDiagnostic,
    pub quality: Vec<VegaKtGammaRowQuality>,
    pub passed: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VegaKtGammaRecoveryStatus {
    NoPilotCandidate,
    ValidationFailed,
    PositiveTimePolicyPassed,
}
impl VegaKtGammaRecoveryStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoPilotCandidate => "no_feasible_pilot_candidate",
            Self::ValidationFailed => "validation_failed",
            Self::PositiveTimePolicyPassed => "positive_time_policy_passed",
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtGammaRecoveryAudit {
    pub policy_label: &'static str,
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub policy: VegaKtGammaRecoveryPolicy,
    pub status: VegaKtGammaRecoveryStatus,
    pub validation_sampling_units: u64,
    pub pilot_index_start: u64,
    pub pilot_index_end_exclusive: u64,
    pub pilot_candidates: Vec<VegaKtGammaCandidateAudit>,
    pub selected_regularization: Option<f64>,
    pub validation: Option<VegaKtGammaCandidateAudit>,
    /// Separate first-boundary measure on validation indices. No cross-row IV
    /// covariance or equation-(4)/(9) hedge is manufactured by this container.
    pub initial: VegaKtInitialMeasureDiagnostic,
}
