//! Reverse-mode infrastructure for simulation and compiled payoff execution.

#![forbid(unsafe_code)]

mod workspace;

pub use workspace::{
    AadConfigError, AadTilePolicy, AlignedF64Buffer, CheckpointPolicy, SoaWorkspace,
};

/// Confirms that AAD is built on the deterministic numerical layer.
#[must_use]
pub const fn numerical_foundation() -> &'static str {
    pricing_numerics::foundation_role()
}
