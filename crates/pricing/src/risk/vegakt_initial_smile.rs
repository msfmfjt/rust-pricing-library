//! Deterministic, finite-window initial Gamma approximation with separate MC checks.
use super::VegaKtStepMoment;
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtInitialSmileResolution {
    pub subdivisions: usize,
    pub interpolation_nodes: usize,
    pub gamma_bin_masses: Vec<f64>,
    pub gamma_bin_variance_sensitivities: Vec<f64>,
    pub total_gamma_mass: f64,
    pub recovered_variance_sensitivity: f64,
    pub interpolated_price: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtInitialSmileGammaAudit {
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub maturity: f64,
    pub first_step_variance: f64,
    pub gamma_bin_boundaries: Vec<f64>,
    pub log_window: [f64; 2],
    pub continuation_growth_bound: [f64; 2],
    pub relative_tolerance: f64,
    pub functional_tolerance: f64,
    pub mass_tolerance: f64,
    pub price_tail_bound: f64,
    pub variance_tail_bound: f64,
    pub quadrature_price: f64,
    pub quadrature_variance_sensitivity: f64,
    pub quadrature_price_difference: f64,
    pub quadrature_variance_difference: f64,
    pub coarse: VegaKtInitialSmileResolution,
    pub fine: VegaKtInitialSmileResolution,
    pub price_refinement_difference: f64,
    pub variance_refinement_difference: f64,
    pub bin_mass_refinement_difference: f64,
    pub bin_variance_refinement_difference: f64,
    pub recovery_passed: bool,
    pub independent_sampling_units: u64,
    pub antithetic: bool,
    pub native_price: VegaKtStepMoment,
    pub native_initial_variance_sensitivity: VegaKtStepMoment,
    pub price_difference: VegaKtStepMoment,
    pub variance_difference: VegaKtStepMoment,
    pub validation_passed: bool,
    /// Only the first-interval annual-variance functional, not a complete IV chain.
    pub estimate: Option<f64>,
}
