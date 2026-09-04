//! Monte Carlo, randomized QMC, compiled path execution, and LSM.

#![forbid(unsafe_code)]

mod config;
mod error;
mod execution;
mod random;

pub use config::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
pub use error::EngineConfigError;
pub use execution::{DeterministicExecutor, ExecutionError, ExecutionPolicy, ExecutorBuildError};
pub use random::{
    NormalQuantileError, Philox4x32, RandomCoordinate, RandomDomain, antithetic_normal,
    inverse_standard_normal, open_unit_interval,
};

/// Returns the role of the model layer consumed by simulation.
#[must_use]
pub const fn model_foundation() -> &'static str {
    pricing_models::market_foundation()
}

/// Confirms that the optional AAD execution feature is compiled.
#[cfg(feature = "aad")]
#[must_use]
pub const fn aad_enabled() -> bool {
    let _ = pricing_aad::numerical_foundation();
    true
}
