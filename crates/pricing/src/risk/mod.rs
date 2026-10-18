//! Risk orchestration, bump validation, and VegaKT transformation.

#![forbid(unsafe_code)]

mod error;
mod request;
mod vegakt;
mod vegakt_atom_chain;
mod vegakt_convolution;
mod vegakt_selection;
mod vegakt_smile;
pub use vegakt_atom_chain::{VegaKtAtomChainDiagnostic, VegaKtAtomChainEstimate};
pub use vegakt_convolution::{
    VegaKtConvolutionDiagnostic, VegaKtConvolutionPlan, VegaKtConvolutionRow,
    VegaKtTwoTransitionSourceAccuracy, VegaKtVarianceOperatorAccuracy,
};
pub use vegakt_selection::{
    VegaKtGammaCandidateAudit, VegaKtGammaRecoveryAudit, VegaKtGammaRecoveryPolicy,
    VegaKtGammaRecoveryStatus, VegaKtGammaRowQuality,
};
pub use vegakt_smile::{
    VegaKtTwoStepSmileGammaAudit, VegaKtVarianceGammaCandidate, VegaKtVarianceGammaQuality,
};
mod vegakt_initial_measure;
pub use vegakt_initial_measure::VegaKtInitialMeasureDiagnostic;
mod vegakt_measure;
pub use crate::engine::risk::vegakt_terminal::terminal_payoff_gamma_measure;
pub use vegakt_measure::{
    VEGA_KT_TERMINAL_MEASURE_LABEL, VegaKtGammaAtom, VegaKtGammaAtoms, VegaKtTerminalDiagnostic,
    VegaKtTerminalLocalRow, VegaKtTerminalPayoff,
};
mod vegakt_chain;
pub(crate) use vegakt_chain::{ChainLayout, ChainRow};
mod vegakt_initial;
pub use vegakt_chain::{
    VEGA_KT_CHAIN_LABEL, VegaKtChainDiagnostic, VegaKtChainEstimate, VegaKtChainPlan,
    VegaKtChainRowAudit, VegaKtChainSample,
};
mod vegakt_steps;
pub(crate) use vegakt_initial::InitialGammaLayout;
pub use vegakt_initial::{
    INITIAL_GAMMA_CONDITIONAL_FD_LABEL, VegaKtGammaBumpEstimate, VegaKtInitialGammaEstimate,
    VegaKtInitialGammaPlan, VegaKtInitialGammaSample,
};
pub(crate) use vegakt_steps::VegaKtStepLayout;
pub use vegakt_steps::{
    VegaKtDirectVarianceRowEstimate, VegaKtStepEstimate, VegaKtStepMoment, VegaKtStepRecoveryPlan,
    VegaKtStepRowEstimate, VegaKtStepSample, VegaKtStepSamplingUnit,
};

pub use crate::engine::risk::vegakt::{
    active_density_domain, analytic_call_density_row_from_surface,
    analytic_call_density_rows_from_surface, equation_11_local_gamma_from_log_moneyness_vega,
    equation_11_local_gamma_from_strike_vega, equation_11_refinement_diagnostics,
    integrate_local_gamma_total_variance_transition,
    integrate_piecewise_linear_local_gamma_transition, local_gamma_transition_cells,
    local_vega_density_from_node_adjoints, mass_lumped_hat_areas,
    project_local_vega_nodes_to_reporting_iv, vega_kt_bucket_estimates,
    vega_kt_full_bucket_covariance, vega_kt_projection_from_parts, vega_kt_report,
    vega_kt_report_from_samples,
};
pub use crate::engine::risk::vegakt_paper::project_vanilla_hedge_measure_to_reporting_iv;
pub use crate::engine::risk::vegakt_recovery::recover_equation11_gamma_row;
pub use error::RiskConfigError;
pub use request::{
    GammaConfig, PayoffSmoothing, PayoffSmoothingWidthLadder, RiskRequest, SmileDynamics, SpotBump,
    VegaKtConfig, gamma_bump_ladder,
};
pub use vegakt::{
    ActiveDensityDomain, AnalyticCallDensityRow, EQUATION_11_FIRST_ORDER_LABEL,
    EQUATION_11_TRUNCATION_ORDER, Equation11GammaRow, Equation11LocalGamma,
    Equation11RefinementDiagnostics, Equation11RefinementLevel, ReportingIvBasis,
    ReportingIvBoundary, ReportingIvInterpolation, ReportingIvProjectionStats,
    TransitionCellIntegral, VEGA_KT_MARKET_SCALE, VegaKtBucketCoordinate, VegaKtBucketEstimate,
    VegaKtBucketUnit, VegaKtCovarianceLayout, VegaKtProjection, VegaKtReport,
    VegaKtResidualDiagnostics, VegaKtVanillaProjection,
};

/// Confirms that the risk layer is connected to AAD-enabled simulation.
#[must_use]
pub const fn aad_simulation_enabled() -> bool {
    crate::mc::aad_enabled()
}

pub use crate::engine::risk::vegakt_density_mass::{
    gaussian_log_hat_probabilities, lognormal_spot_squared_hat_masses, refine_vega_kt_log_nodes,
};

mod vegakt_initial_smile;
pub(crate) use vegakt_convolution::gaussian_legendre;
pub use vegakt_initial_smile::{VegaKtInitialSmileGammaAudit, VegaKtInitialSmileResolution};

mod vegakt_joint;
mod vegakt_middle;
mod vegakt_three_step;
pub use vegakt_joint::{
    VegaKtJointGammaAudit, VegaKtJointGammaDirections, VegaKtJointGammaStatistics,
};
pub use vegakt_middle::{VegaKtMiddleGammaResolution, VegaKtMiddleSmileGammaAudit};
pub use vegakt_three_step::VegaKtThreeStepTerminalGammaAudit;

mod vegakt_initial_three;
pub use vegakt_initial_three::VegaKtThreeStepInitialGammaAudit;

mod vegakt_multi_step;
mod vegakt_path_chain;
pub use vegakt_convolution::{VegaKtMultiTransitionSourceAccuracy, VegaKtSourceStepAccuracy};
pub use vegakt_multi_step::VegaKtMultiStepTerminalGammaAudit;
pub use vegakt_path_chain::{VegaKtPathGammaChainAudit, VegaKtPathGammaRow};
