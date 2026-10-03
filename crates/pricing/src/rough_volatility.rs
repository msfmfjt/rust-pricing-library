//! Additional rough model families, finite-grid paths and deterministic-rate
//! MC/RQMC pricing, plus separate fixed-model Fourier forward sensitivities.
//! This additive API does not add stable JSON model tags or claim AAD,
//! calibration, or LSV/HW composition.

pub use crate::engine::processes::rough_volatility::{
    RoughVolatilityPath, RoughVolatilityPathPlan,
};
pub use crate::engine::risk::rough_volatility::{RoughVolatilityPrice, RoughVolatilityPricingPlan};
pub use crate::models::{
    ForwardVarianceCurve, LiftedHeston, MixedRoughBergomi, QuadraticRoughHeston, Rfsv, RoughHeston,
    RoughSabr, RoughVolatilityModel,
};

pub use crate::engine::analytic::heston_fourier::{
    FourierError, HestonFourierConfig, HestonFourierGreeks, HestonFourierParameterRisk,
    HestonFourierParameterRiskPlan, HestonFourierPlan, HestonFourierPrice,
    HestonParameterSensitivities,
};
pub use pricing_numerics::Complex64;
