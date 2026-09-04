//! Deterministic numerical foundations.

#![forbid(unsafe_code)]

/// Returns the direct lower-layer dependency role.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_core::role()
}
