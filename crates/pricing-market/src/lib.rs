//! Immutable market-data objects and compilation support.

#![forbid(unsafe_code)]

/// Returns the lower-level role used by market numerics.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_numerics::foundation_role()
}
