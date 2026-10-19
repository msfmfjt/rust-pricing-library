//! Conditional signed Gamma measures for selected rows on a variable time grid.
use super::VegaKtStepMoment;

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtPathGammaRow {
    pub variance_row: usize,
    pub recovered_price: VegaKtStepMoment,
    pub price_difference: VegaKtStepMoment,
    pub variance_refinement: VegaKtStepMoment,
    pub price_refinement: VegaKtStepMoment,
    pub price_tail_bound: VegaKtStepMoment,
    pub variance_tail_bound: VegaKtStepMoment,
    pub negative_atoms: VegaKtStepMoment,
    pub bin_mass_refinement: VegaKtStepMoment,
    pub gamma_bin_masses: Vec<VegaKtStepMoment>,
    pub gamma_bin_variance_sensitivities: Vec<VegaKtStepMoment>,
    pub recovered_variance_nodes: Vec<VegaKtStepMoment>,
    pub native_variance_nodes: Vec<VegaKtStepMoment>,
    pub node_differences: Vec<VegaKtStepMoment>,
    pub terminal_ad_conditional_variances: Vec<VegaKtStepMoment>,
    pub node_validation_standard_errors: Vec<f64>,
    /// Legacy direct Gamma/raw-AD comparison, retained as a diagnostic.
    pub raw_node_validation_passed: bool,
    /// Exact last-shock conditional native AD and paired Gamma difference.
    pub conditional_native_variance_nodes: Vec<VegaKtStepMoment>,
    pub conditional_node_differences: Vec<VegaKtStepMoment>,
    pub raw_ad_conditional_variances: Vec<VegaKtStepMoment>,
    pub raw_conditional_node_differences: Vec<VegaKtStepMoment>,
    pub raw_conditional_standard_errors: Vec<f64>,
    pub conditional_gamma_validation_passed: bool,
    pub raw_conditional_validation_passed: bool,
    pub gamma_relative_standard_error: Option<f64>,
    pub recovery_passed: bool,
    pub validation_passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtPathGammaChainAudit {
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub time_nodes: Vec<f64>,
    pub gamma_bin_boundaries: Vec<f64>,
    pub variance_risk_directions: Vec<Vec<Vec<f64>>>,
    pub subdivisions: usize,
    pub relative_tolerance: f64,
    pub functional_tolerance: f64,
    pub gamma_relative_standard_error_limit: f64,
    pub independent_sampling_units: u64,
    /// RQMC uncertainty is across independent scramble means, never points.
    pub points_per_scramble: Option<u64>,
    /// Higher-resolution conditional-AD reference, paired on the same scrambles.
    pub conditional_ad_points_per_scramble: Option<u64>,
    pub antithetic: bool,
    pub native_price: VegaKtStepMoment,
    pub rows: Vec<VegaKtPathGammaRow>,
    pub native_directions: Vec<VegaKtStepMoment>,
    pub recovered_directions: Vec<VegaKtStepMoment>,
    pub direction_differences: Vec<VegaKtStepMoment>,
    pub direction_refinement_allowances: Vec<VegaKtStepMoment>,
    /// Row-major (3D)x(3D), ordered native/recovered/recovered-minus-native.
    /// Per independent sampling unit (pair average when antithetic); divide by N
    /// for covariance of means. Raw diagnostic retained even if estimate is None.
    pub joint_sample_covariance: Vec<f64>,
    /// Optional counter-ordered [native price, recovered directions...] readouts.
    /// One row per independent unit, after averaging an antithetic pair.
    pub sampling_unit_values: Option<Vec<Vec<f64>>>,
    pub direction_relative_standard_error: Option<f64>,
    pub recovery_passed: bool,
    pub validation_passed: bool,
    pub estimate: Option<Vec<VegaKtStepMoment>>,
}
