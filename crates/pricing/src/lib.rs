//! Stable Rust facade for the derivatives-pricing library.

#![forbid(unsafe_code)]

pub use pricing_core as core;
pub use pricing_market as market;
pub use pricing_mc as mc;
pub use pricing_models as models;
pub use pricing_product as product;
pub use pricing_risk as risk;

/// Returns the public facade version.
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn facade_connects_to_risk_execution() {
        assert!(crate::risk::aad_simulation_enabled());
    }
}
