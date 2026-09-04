//! Fundamental domain types shared by the pricing workspace.

#![forbid(unsafe_code)]

/// Returns the crate's stable workspace role name.
#[must_use]
pub const fn role() -> &'static str {
    "core"
}
