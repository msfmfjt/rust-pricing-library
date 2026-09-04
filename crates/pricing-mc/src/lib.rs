//! Monte Carlo, randomized QMC, compiled path execution, and LSM.

#![forbid(unsafe_code)]

mod config;
mod error;

pub use config::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
pub use error::EngineConfigError;

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
