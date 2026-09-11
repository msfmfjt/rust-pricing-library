//! Immutable market-data objects and compilation support.

#![forbid(unsafe_code)]

mod context;
mod curve;
mod dividend;
mod error;
mod essvi;
mod forward;
mod local_variance;
mod pchip;
mod ssvi;

pub use context::{EquityMarket, MarketContext};
pub use curve::{
    CurveEvaluation, CurveExtrapolationStats, CurveRegion, DiscountCurve, LogLinearDiscountCurve,
};
pub use dividend::{
    AffineDividendCoordinate, AffineDividendTimelineEntry, AffineDividendTransform,
    CompiledDividendEvent, DIVIDEND_EVENT_ORDER, DividendEvent, DividendMatchingTolerance,
    DividendQuote, validate_dividend_call_price_matching,
};
pub use error::MarketError;
pub use essvi::{EssviParameters, EssviSlice, EssviSurface};
pub use forward::{EquityForward, ForwardEvaluation};
pub use local_variance::{
    LocalVarianceBoundary, LocalVarianceBoundaryStats, LocalVarianceGrid,
    LocalVarianceGridSuggestion, LocalVarianceInterpolation, LocalVarianceRepair,
    LocalVarianceRepairReason, piecewise_sinh_log_moneyness_nodes,
    suggest_log_moneyness_nodes_from_density,
};
pub use pchip::{ThetaEvaluation, ThetaPchip, ThetaRegion};
pub use ssvi::{
    ForwardCallEvaluation, ImpliedVarianceSurface, PhiEvaluation, PhiSpec, StandardSsvi,
    SurfaceValidationTolerance, TotalVarianceDerivatives, durrleman_density_factor,
};

/// Returns the lower-level role used by market numerics.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_numerics::foundation_role()
}
