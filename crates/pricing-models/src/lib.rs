//! Compiled stochastic-model kernels.

#![forbid(unsafe_code)]

mod bergomi;
mod spec;

pub use bergomi::{Bergomi1Factor, BergomiTransition};

pub use spec::{
    Black76Spec, BlackScholesSpec, LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec,
};

/// Returns the role of the market layer consumed by model kernels.
#[must_use]
pub const fn market_foundation() -> &'static str {
    pricing_market::foundation_role()
}
