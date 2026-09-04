//! Python-boundary placeholder; PyO3 integration is introduced at Gate G7.

#![forbid(unsafe_code)]

/// Returns the Rust facade version exposed by the future Python module.
#[must_use]
pub const fn facade_version() -> &'static str {
    pricing::version()
}

#[cfg(test)]
mod tests {
    #[test]
    fn boundary_uses_facade_version() {
        assert_eq!(crate::facade_version(), env!("CARGO_PKG_VERSION"));
    }
}
