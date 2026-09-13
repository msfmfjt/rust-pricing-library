//! Monte Carlo, randomized QMC, compiled path execution, and LSM.

#![forbid(unsafe_code)]

pub use crate::engine::mc::config::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
pub use crate::engine::mc::error::EngineConfigError;
pub use crate::engine::mc::execution::{
    DeterministicExecutor, ExecutionError, ExecutionPolicy, ExecutorBuildError,
};
pub use crate::engine::mc::execution::{DeterministicStatistics, TryExecutionError};
pub use crate::engine::mc::lsm::{
    ContinueAllReason, CpqrConfig, CpqrFit, DateLocalExerciseFit, ExerciseDecisionModel,
    ExercisePolicy, ExercisePolicyFingerprint, ExercisePolicyTrainingMetadata,
    ExercisePolicyTrainingOutcome, ExercisePolicyValuationOutcome, ExerciseRegressionDiagnostics,
    FeatureScaling, LSM_BASIS_ABI, LSM_POLICY_ABI, LSM_REGRESSION_ABI, LsmConfig,
    LsmConfigurationFingerprint, LsmNumericalError, LsmStateVariable, LsmWarning,
    PolynomialBasisSpec, PolynomialRegressionModel, fit_cpqr, fit_exercise_decision,
    fit_polynomial_regression, is_training_itm, should_exercise, train_exercise_policy,
    value_exercise_policy,
};
pub use crate::engine::processes::local_vol::{
    LOCAL_VOL_LOG_EULER_SCHEME, LocalVolDividendCheckpoint, LocalVolDividendCheckpointSchedule,
    LocalVolDividendReverseCache, LocalVolError, LocalVolLogEulerPlan, LocalVolPath,
    LocalVolPostDividendSpot, LocalVolReverseAdjoints, LocalVolStepCache, LocalVolTimeGrid,
};
pub use crate::engine::sampling::qmc::{
    JOE_KUO_DIRECTION_SET, RQMC_SCRAMBLE_ABI, RqmcPlan, RqmcPlanError, RqmcPointError, Scramble32,
    Sobol32, SobolDimensionError,
};
pub use crate::engine::sampling::random::{
    NormalQuantileError, Philox4x32, RandomCoordinate, RandomDomain, antithetic_normal,
    inverse_standard_normal, open_unit_interval,
};

/// Returns the role of the model layer consumed by simulation.
#[must_use]
pub const fn model_foundation() -> &'static str {
    crate::models::market_foundation()
}

/// Confirms that the optional AAD execution feature is compiled.
#[must_use]
pub const fn aad_enabled() -> bool {
    let _ = crate::engine::aad::numerical_foundation();
    true
}
pub use crate::engine::payoff::barrier_bridge::{
    BARRIER_BRIDGE_ABI, BarrierBridgeDirection, BarrierBridgeError, BarrierBridgeInterval,
    BarrierBridgeIntervalAdjoints, BarrierBridgeIntervalInput, BarrierBridgePath,
    BarrierBridgePathDiagnostics, BarrierBridgeStatus, SmoothedBarrierBridgeEndpoint,
    SmoothedBarrierBridgeEndpointAdjoints, SmoothedBarrierBridgeEndpointInput,
    SmoothedBarrierBridgeInterval, SmoothedBarrierBridgeIntervalAdjoints,
    SmoothedBarrierBridgeIntervalInput, transformed_barrier,
};
pub use crate::engine::sampling::bridge::{
    BROWNIAN_BRIDGE_ABI, BrownianBridgeError, BrownianBridgeInstruction, BrownianBridgePlan,
};

pub mod hull_white {
    pub use crate::engine::calibration::hull_white::*;
    pub use crate::engine::processes::hull_white::*;
}
pub mod lsv {
    pub use crate::engine::calibration::lsv::*;
    pub use crate::engine::processes::lsv::*;
}

#[doc(hidden)]
pub use crate::engine::aad::{
    AadConfigError, AadTilePolicy, AlignedF64Buffer, CheckpointPolicy, SoaWorkspace,
};
