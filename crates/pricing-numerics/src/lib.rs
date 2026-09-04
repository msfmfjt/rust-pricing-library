//! Deterministic numerical foundations.

#![forbid(unsafe_code)]

mod reduction;

pub use reduction::{
    CenteredCovariance, CenteredMoment, NeumaierSum, reduce_covariances,
    reduce_moments, reduce_sums,
};

/// Returns the direct lower-layer dependency role.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_core::role()
}
