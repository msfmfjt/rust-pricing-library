//! Deterministic numerical foundations.

#![forbid(unsafe_code)]

mod correlation;
mod normal;
pub use correlation::{
    CorrelationDiagnostics, CorrelationError, CorrelationFactor, CorrelationToleranceConfig,
};
mod reduction;

pub use normal::{standard_normal_cdf, standard_normal_pdf};
pub use reduction::{
    CenteredCovariance, CenteredMoment, NeumaierSum, reduce_covariances, reduce_moments,
    reduce_sums,
};

/// Legacy foundation marker; the numerical crate has no financial dependencies.
#[must_use]
pub const fn foundation_role() -> &'static str {
    "core"
}

mod fractional;
pub use fractional::{digamma_half_to_two, fractional_ou_correlation, gamma_half_to_two};

mod complex;
pub use complex::Complex64;

pub mod least_squares;
