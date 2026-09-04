//! Compiled stochastic-model kernels.

#![forbid(unsafe_code)]

/// Returns the role of the market layer consumed by model kernels.
#[must_use]
pub const fn market_foundation() -> &'static str {
    pricing_market::foundation_role()
}
