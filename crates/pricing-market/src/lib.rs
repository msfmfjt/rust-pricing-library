//! Immutable market-data objects and compilation support.

#![forbid(unsafe_code)]

mod context;
mod curve;
mod error;
mod forward;

pub use context::{EquityMarket, MarketContext};
pub use curve::{
    CurveEvaluation, CurveExtrapolationStats, CurveRegion, DiscountCurve, LogLinearDiscountCurve,
};
pub use error::MarketError;
pub use forward::{EquityForward, ForwardEvaluation};

/// Returns the lower-level role used by market numerics.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_numerics::foundation_role()
}
