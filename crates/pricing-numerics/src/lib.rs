//! Deterministic numerical foundations.

#![forbid(unsafe_code)]

mod normal;
mod reduction;

pub use normal::{standard_normal_cdf, standard_normal_pdf};
pub use reduction::{
    CenteredCovariance, CenteredMoment, NeumaierSum, reduce_covariances, reduce_moments,
    reduce_sums,
};

/// Returns the direct lower-layer dependency role.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_core::role()
}
