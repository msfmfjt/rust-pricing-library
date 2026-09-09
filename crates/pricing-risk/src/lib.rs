//! Risk orchestration, bump validation, and VegaKT transformation.

#![forbid(unsafe_code)]

mod error;
mod request;
mod vegakt;

pub use error::RiskConfigError;
pub use request::{
    GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig, gamma_bump_ladder,
};
pub use vegakt::{
    ActiveDensityDomain, AnalyticCallDensityRow, EQUATION_11_FIRST_ORDER_LABEL,
    EQUATION_11_TRUNCATION_ORDER, Equation11LocalGamma, Equation11RefinementDiagnostics,
    Equation11RefinementLevel, ReportingIvBasis, ReportingIvBoundary, ReportingIvInterpolation,
    ReportingIvProjectionStats, TransitionCellIntegral, VEGA_KT_MARKET_SCALE,
    VegaKtBucketCoordinate, VegaKtBucketEstimate, VegaKtBucketUnit, VegaKtCovarianceLayout,
    VegaKtProjection, VegaKtReport, VegaKtResidualDiagnostics, active_density_domain,
    analytic_call_density_row_from_surface, analytic_call_density_rows_from_surface,
    equation_11_local_gamma_from_log_moneyness_vega, equation_11_local_gamma_from_strike_vega,
    equation_11_refinement_diagnostics, integrate_piecewise_linear_local_gamma_transition,
    local_gamma_transition_cells, local_vega_density_from_node_adjoints, mass_lumped_hat_areas,
    project_local_vega_nodes_to_reporting_iv, vega_kt_bucket_estimates,
    vega_kt_full_bucket_covariance, vega_kt_report, vega_kt_report_from_samples,
};

/// Confirms that the risk layer is connected to AAD-enabled simulation.
#[must_use]
pub const fn aad_simulation_enabled() -> bool {
    pricing_mc::aad_enabled()
}
