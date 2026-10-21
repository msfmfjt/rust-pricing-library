//! Stable Rust facade for the derivatives-pricing library.

#![forbid(unsafe_code)]

mod api;
pub mod bass_lv;
mod engine;
use api::{error, request, result};
pub mod hull_white;
pub mod lsv;
mod monte_carlo;
pub mod multi_asset;
mod plan;
pub mod rough_volatility;
pub mod stochastic_dividends;
pub mod stochastic_volatility;
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
pub mod dual;
pub mod market;
pub mod mc;
pub mod models;
pub mod product;
pub mod risk;
pub use api::calibration_report::{
    CalibrationKind, CalibrationMetric, DiagnosticUnit, ReportedCalibration,
};
pub use api::source::{
    CapabilityReason, CapabilityStatus, MultiAssetModelSpec, MultiAssetSpecification,
    PricingCapability, PricingOperation, PricingSource, PricingTask,
    StochasticDividendSpecification,
};
pub use api::specification::{
    EquityModelSpec, PricingModelSpec, PricingNumerics, PricingSpecification, RateModelSpec,
    SpecificationValidationError, VolatilityFactorSpec,
};
pub use api::valuation_report::{
    CalibrationTreatment, ReportAxis, ReportedRisk, RiskParameterKind, SensitivityUnit,
    UncertaintyScope, ValuationReport,
};
pub use engine::risk::source::{
    CompiledPricingTask, PricingTaskError, PricingTaskOutput, PricingTaskResult,
};
pub use engine::risk::specification::{
    CompiledPricingSpecification, SpecificationOutput, SpecificationResult,
};
pub use request::PricingRequest;
pub use result::{
    ConfidenceInterval, Diagnostics, Estimate, EstimatorKind, MigrationProvenance, PricingResult,
    PricingWarning, ReplayMetadata, RiskEstimate, RiskReport, RiskUnit, VegaKtResult,
    VegaKtResultBucketEstimate, VegaKtResultCoordinate, VegaKtResultCovarianceLayout,
    VegaKtResultProjection, VegaKtResultReportingStats, VegaKtResultResidualDiagnostics,
    VegaKtResultUnit,
};
pub use wire::{
    Fingerprint, JsonLimits, MigrationRegistry, WireError, current_pricing_task_schema,
    current_product_graph_schema, current_request_schema, current_result_schema,
    current_specification_schema, current_valuation_report_schema, fingerprint_pricing_task,
    fingerprint_product_graph, fingerprint_request, fingerprint_specification,
    monte_carlo_result_to_json, monte_carlo_result_to_pretty_json, parse_monte_carlo_result_json,
    parse_pricing_task_json, parse_product_graph_json, parse_request_json, parse_result_json,
    parse_specification_json, parse_valuation_report_json, pricing_task_to_json,
    pricing_task_to_pretty_json, product_graph_to_json, product_graph_to_pretty_json,
    request_to_json, request_to_pretty_json, result_to_json, result_to_pretty_json,
    specification_to_json, specification_to_pretty_json, valuation_report_to_json,
    valuation_report_to_pretty_json,
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
