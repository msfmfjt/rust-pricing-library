//! Additional rough model families, finite-grid paths and deterministic-rate
//! MC/RQMC pricing. This is an additive, price-only extension API; it does not
//! add stable JSON model tags or claim AAD, calibration, LSV/HW composition.

pub use crate::engine::processes::rough_volatility::{
    RoughVolatilityPath, RoughVolatilityPathPlan,
};
pub use crate::engine::risk::rough_volatility::{RoughVolatilityPrice, RoughVolatilityPricingPlan};
pub use crate::models::{
    ForwardVarianceCurve, LiftedHeston, MixedRoughBergomi, QuadraticRoughHeston, Rfsv, RoughHeston,
    RoughSabr, RoughVolatilityModel,
};

pub use crate::engine::analytic::heston_fourier::{
    FourierError, HestonFourierConfig, HestonFourierPlan, HestonFourierPrice,
};
pub use pricing_numerics::Complex64;
