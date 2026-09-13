//! Compiled stochastic-model kernels.

#![forbid(unsafe_code)]

mod bergomi;
pub mod hull_white;
pub mod hull_white_dividends;
mod rough_bergomi;
mod spec;

pub use bergomi::{Bergomi1Factor, BergomiTransition};
pub use hull_white::{
    HullWhite1Factor, HullWhiteError, HullWhiteHybridTransition, HybridCorrelation,
};
pub use rough_bergomi::RoughBergomi;

pub use spec::{
    Black76Spec, BlackScholesSpec, LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec,
};

/// Returns the role of the market layer consumed by model kernels.
#[must_use]
pub const fn market_foundation() -> &'static str {
    pricing_market::foundation_role()
}
