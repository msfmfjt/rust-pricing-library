//! Additional rough model families, finite-grid paths and deterministic-rate
//! MC/RQMC pricing, plus separate fixed-model Fourier forward, scalar and Hurst sensitivities.
//! Explicit fixed-model MC Spot Delta and particle LSV local-variance-node
//! discrete adjoints are available. This does not add stable JSON tags,
//! general stochastic-parameter AAD or Hull-White composition. A fixed-kernel
//! pure Heston MC parameter reverse is available separately. Explicit finite-bump MC
//! Gamma with paired h/h2 diagnostics is available separately from AAD. An explicit
//! market-IV source can additionally be bound for discrete quote-node risk.

pub use crate::engine::processes::rough_volatility::{
    HESTON_MC_PARAMETER_NAMES, HestonMcAdjoints, HestonMcRecordedPath, MixedBergomiMcAdjoints,
    MixedBergomiMcHurstPlan, MixedBergomiMcRecordedPath, RoughFamilyLsvAdjoints,
    RoughFamilyLsvPath, RoughFamilyLsvPlan, RoughHestonMcHurstAdjoints, RoughHestonMcHurstPlan,
    RoughHestonMcHurstRecordedPath, RoughVolatilityPath, RoughVolatilityPathPlan,
    RoughVolatilityRecordedPath,
};
pub use crate::engine::risk::rough_volatility::{
    HestonMcParameterRisk, MixedBergomiMcParameterRisk, RoughHestonMcHurstRisk,
    RoughVolatilityDelta, RoughVolatilityGamma, RoughVolatilityPrice, RoughVolatilityPricingPlan,
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
    HestonLsvParameterRisk, MixedBergomiLsvParameterRisk, RoughFamilyLsvDelta,
    RoughFamilyLsvDeltaConvention, RoughFamilyLsvGamma, RoughFamilyLsvMarketIvRisk,
    RoughFamilyLsvMarketIvRiskPlan, RoughFamilyLsvPricingPlan,
};

/// Shared finite-bump Gamma report (not a second-order AAD result).
pub use crate::engine::risk::gamma_bump::RoughGammaBump;
