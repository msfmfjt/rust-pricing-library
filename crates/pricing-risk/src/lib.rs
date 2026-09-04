//! Risk orchestration, bump validation, and VegaKT transformation.

#![forbid(unsafe_code)]

/// Confirms that the risk layer is connected to AAD-enabled simulation.
#[must_use]
pub const fn aad_simulation_enabled() -> bool {
    pricing_mc::aad_enabled()
}
