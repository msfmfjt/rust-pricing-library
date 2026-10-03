//! Additional rough model families, finite-grid paths and deterministic-rate
//! MC/RQMC pricing, plus separate fixed-model Fourier forward, scalar and Hurst sensitivities.
//! This additive API does not add stable JSON model tags or claim AAD,
//! or LSV/HW composition. Separate experimental Heston price calibration is available.

pub use crate::engine::processes::rough_volatility::{
    RoughVolatilityPath, RoughVolatilityPathPlan,
};
pub use crate::engine::risk::rough_volatility::{RoughVolatilityPrice, RoughVolatilityPricingPlan};
pub use crate::models::{
    ForwardVarianceCurve, LiftedHeston, MixedRoughBergomi, QuadraticRoughHeston, Rfsv, RoughHeston,
    RoughSabr, RoughVolatilityModel,
};

pub use crate::engine::analytic::heston_fourier::{
    FourierError, HestonFourierConfig, HestonFourierGreeks, HestonFourierHurstRisk,
    HestonFourierHurstRiskPlan, HestonFourierParameterRisk, HestonFourierParameterRiskPlan,
    HestonFourierPlan, HestonFourierPrice, HestonParameterSensitivities,
};
pub use pricing_numerics::Complex64;

pub use crate::engine::analytic::heston_fourier::{
    HestonCalibrationError, HestonCalibrationEvaluation, HestonCalibrationParameter,
    HestonCalibrationProblem, HestonCalibrationQuote, HestonCalibrationResult,
    HestonCalibrationVariable,
};
pub use pricing_numerics::least_squares::{LeastSquaresOptions, LeastSquaresTermination};

pub use crate::engine::analytic::heston_fourier::{
    HestonIvCalibrationEvaluation, HestonIvCalibrationProblem, HestonIvCalibrationQuote,
    HestonIvCalibrationResult, HestonIvGridValidation, HestonIvRefinementOptions,
    HestonIvRefinementResult, HestonIvRefinementStage,
};
