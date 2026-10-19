use super::{VegaKtInitialSmileGammaAudit, VegaKtStepMoment, VegaKtTwoStepSmileGammaAudit};

/// Constraint basis acts only on the second variance row. Risk readouts each
/// contain one uniform first-variance coefficient followed by second-row nodes.
/// Dependent readouts are allowed; the constraint basis must remain independent.
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtJointGammaDirections {
    pub conservation_basis: Vec<Vec<f64>>,
    pub risk_directions: Vec<Vec<f64>>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtJointGammaStatistics {
    pub independent_sampling_units: u64,
    pub antithetic: bool,
    pub native_directions: Vec<VegaKtStepMoment>,
    pub recovered_directions: Vec<VegaKtStepMoment>,
    pub differences: Vec<VegaKtStepMoment>,
    /// Row-major (3D)x(3D), ordered native/recovered/difference; per sample,
    /// not covariance of means. Divide by independent_sampling_units for means.
    pub joint_sample_covariance: Vec<f64>,
    pub initial_error_allowances: Vec<f64>,
    pub roundoff_allowances: Vec<f64>,
    pub passed: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtJointGammaAudit {
    pub directions: VegaKtJointGammaDirections,
    pub initial: VegaKtInitialSmileGammaAudit,
    pub second: VegaKtTwoStepSmileGammaAudit,
    /// None if either component gate failed. Raw failed joint statistics remain
    /// available here when the final direction gate fails.
    pub joint_validation: Option<VegaKtJointGammaStatistics>,
}
