//! Compatibility entry for root-level simulation and result types.
pub use crate::api::mc_result::*;
pub use crate::engine::plan::simulation::SimulationPlan;
pub use crate::engine::risk::valuation::{price_monte_carlo, price_pseudo_monte_carlo};
