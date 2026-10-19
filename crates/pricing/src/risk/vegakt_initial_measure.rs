use crate::risk::VegaKtStepMoment;

/// Gamma at the first positive time with valuation history fixed. For each
/// future draw the continuous piecewise-affine payoff supplies signed atoms.
/// Bins [0,b0), [b0,b1), ..., [bn,infinity) include the entire positive domain.
/// They are integrated masses, not nodal densities or projected atoms.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtInitialMeasureDiagnostic {
    pub policy_label: &'static str,
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub maturity: f64,
    pub first_step_variance: f64,
    pub gamma_bin_boundaries: Vec<f64>,
    pub gamma_bin_masses: Vec<VegaKtStepMoment>,
    /// Nonzero pair/path mass contributions; zero is not proof of no tail risk.
    pub gamma_bin_nonzero_sampling_units: Vec<u64>,
    pub gamma_bin_first_step_vegas: Vec<VegaKtStepMoment>,
    pub independent_sampling_units: u64,
    pub antithetic: bool,
    pub cash: VegaKtStepMoment,
    pub slope: VegaKtStepMoment,
    pub total_gamma_mass: VegaKtStepMoment,
    pub recovered_initial_local_vega: VegaKtStepMoment,
    pub native_initial_local_vega: VegaKtStepMoment,
    pub initial_local_vega_difference: VegaKtStepMoment,
    pub conditional_price: VegaKtStepMoment,
    pub native_price: VegaKtStepMoment,
    pub price_difference: VegaKtStepMoment,
    pub payoff_reconstruction_difference: VegaKtStepMoment,
}
