//! Product specifications and Event/Payoff graph compilation.

#![forbid(unsafe_code)]

mod spec;

pub use spec::{EuropeanVanillaSpec, OptionSide, ProductSpec};

/// Returns the domain foundation role.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_core::role()
}
