//! Stable Rust facade for the derivatives-pricing library.

#![forbid(unsafe_code)]

mod error;
mod monte_carlo;
mod plan;
mod request;
mod result;
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
pub use pricing_core as core;
pub use pricing_market as market;
pub use pricing_mc as mc;
pub use pricing_models as models;
pub use pricing_product as product;
pub use pricing_risk as risk;
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
    current_result_schema, fingerprint_request, parse_request_json, parse_result_json,
    request_to_json, request_to_pretty_json, result_to_json, result_to_pretty_json,
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
