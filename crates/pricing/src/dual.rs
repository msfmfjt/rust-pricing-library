//! Andersen–Broadie primal-dual bounds for a finite exercise schedule.
//!
//! The first adapter supports price-only AmericanVanilla requests under
//! Black–Scholes/Black-76 with deterministic curves and affine dividends.
//! Training and outer evaluation use Pseudo-MC; the frozen LSM policy drives
//! independent inner rollouts. Bounds hold in expectation for the declared
//! exercise grid, not for continuous exercise. See `docs/library/american-dual.md`.

pub use crate::engine::risk::dual::{
    ANDERSEN_BROADIE_ABI, AndersenBroadieConfig, AndersenBroadieError, AndersenBroadiePlan,
    AndersenBroadieResult,
};
