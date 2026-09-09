//! Immutable market-data objects and compilation support.

#![forbid(unsafe_code)]

mod context;
mod curve;
mod error;
mod essvi;
mod forward;
mod pchip;
mod ssvi;

pub use context::{EquityMarket, MarketContext};
pub use curve::{
    CurveEvaluation, CurveExtrapolationStats, CurveRegion, DiscountCurve, LogLinearDiscountCurve,
};
pub use error::MarketError;
pub use essvi::{EssviParameters, EssviSlice, EssviSurface};
pub use forward::{EquityForward, ForwardEvaluation};
pub use pchip::{ThetaEvaluation, ThetaPchip, ThetaRegion};
pub use ssvi::{
    ForwardCallEvaluation, ImpliedVarianceSurface, PhiEvaluation, PhiSpec, StandardSsvi,
    SurfaceValidationTolerance, TotalVarianceDerivatives,
};

/// Returns the lower-level role used by market numerics.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_numerics::foundation_role()
}
