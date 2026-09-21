//! Compiled stochastic-model kernels.

#![forbid(unsafe_code)]

mod bergomi;
mod bergomi_dynamics;
mod bergomi_two_factor;
pub mod hull_white;
pub mod hull_white_dividends;
mod rough_bergomi;
mod spec;
mod volatility_inputs;

pub use bergomi::{Bergomi1Factor, BergomiTransition};
pub use bergomi_dynamics::BergomiDynamics;
pub(crate) use bergomi_two_factor::ou_kernel_correlation;
pub use bergomi_two_factor::{
    BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES, Bergomi2Factor, Bergomi2FactorTransition,
    BergomiError,
};
pub use hull_white::{
    HullWhite1Factor, HullWhiteError, HullWhiteHybridTransition, HybridCorrelation,
};
pub use rough_bergomi::RoughBergomi;
pub(crate) use volatility_inputs::{
    HistoryInnovations, InnovationSource, OrthogonalNormals, OuInnovations,
};

pub use spec::{
    Black76Spec, BlackScholesSpec, LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec,
};

/// Returns the role of the market layer consumed by model kernels.
#[must_use]
pub const fn market_foundation() -> &'static str {
    crate::market::foundation_role()
}
