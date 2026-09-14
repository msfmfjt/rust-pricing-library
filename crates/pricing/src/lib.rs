//! Stable Rust facade for the derivatives-pricing library.

#![forbid(unsafe_code)]

mod api;
mod engine;
use api::{error, request, result};
pub mod hull_white;
pub mod lsv;
mod monte_carlo;
pub mod multi_asset;
mod plan;
mod wire;

#[doc(hidden)]
pub mod analytical;

pub use error::{MonteCarloError, RequestValidationError, ResultBuildError};
pub use monte_carlo::{
    BarrierBridgeDiagnostics, BarrierHitIndicatorMode, BumpValidationPolicy,
    EarlyExerciseDiagnostics, ExerciseStrategyRisk, MonteCarloDiagnostics, MonteCarloPrice,
    PathStateDiagnostics, PayoffSmoothingDiagnostics, PayoffSmoothingKernel,
    PayoffSmoothingWidthUnit, PayoffValuationKind, RiskDiagnostics, RiskMethod, RiskMethodMetadata,
    RiskValidation, SimulationPlan, StoppingIndexRisk, price_monte_carlo, price_pseudo_monte_carlo,
};
pub use plan::{
    PricingPlan, WidthLadderDifference, WidthLadderEntry, WidthLadderResult, compile, evaluate,
};
pub mod core;
pub mod market;
pub mod mc;
pub mod models;
pub mod product;
pub mod risk;
pub use request::PricingRequest;
pub use result::{
    ConfidenceInterval, Diagnostics, Estimate, EstimatorKind, MigrationProvenance, PricingResult,
    PricingWarning, ReplayMetadata, RiskEstimate, RiskReport, RiskUnit, VegaKtResult,
    VegaKtResultBucketEstimate, VegaKtResultCoordinate, VegaKtResultCovarianceLayout,
    VegaKtResultProjection, VegaKtResultReportingStats, VegaKtResultResidualDiagnostics,
    VegaKtResultUnit,
};
pub use wire::{
    Fingerprint, JsonLimits, MigrationRegistry, WireError, current_request_schema,
    current_result_schema, fingerprint_request, monte_carlo_result_to_json,
    monte_carlo_result_to_pretty_json, parse_monte_carlo_result_json, parse_request_json,
    parse_result_json, request_to_json, request_to_pretty_json, result_to_json,
    result_to_pretty_json,
};

/// Returns the public facade version.
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn facade_connects_to_risk_execution() {
        assert!(crate::risk::aad_simulation_enabled());
    }
}
