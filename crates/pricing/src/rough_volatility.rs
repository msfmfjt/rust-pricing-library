//! Additional rough model families, finite-grid paths and deterministic-rate
//! MC/RQMC pricing, plus separate fixed-model Fourier forward, scalar and Hurst sensitivities.
//! Explicit fixed-model MC Spot Delta and particle LSV local-variance-node
//! discrete adjoints are available. This does not add stable JSON tags, MC
//! Gamma, stochastic-parameter AAD or Hull-White composition. An explicit
//! market-IV source can additionally be bound for discrete quote-node risk.

pub use crate::engine::processes::rough_volatility::{
    RoughFamilyLsvAdjoints, RoughFamilyLsvPath, RoughFamilyLsvPlan, RoughVolatilityPath,
    RoughVolatilityPathPlan, RoughVolatilityRecordedPath,
};
pub use crate::engine::risk::rough_volatility::{
    RoughVolatilityDelta, RoughVolatilityPrice, RoughVolatilityPricingPlan,
};
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

/// Particle LSV in the normalized funded-forward coordinate. Fixed model
/// parameters; the local-variance-node VJP includes discrete recalibration.
pub use crate::engine::calibration::lsv::{
    CalibratedRoughFamilyLsv, calibrate_rough_family_lsv, calibrate_rough_family_lsv_parallel,
};
pub use crate::engine::risk::lsv::{
    RoughFamilyLsvDelta, RoughFamilyLsvDeltaConvention, RoughFamilyLsvMarketIvRisk,
    RoughFamilyLsvMarketIvRiskPlan, RoughFamilyLsvPricingPlan,
};
