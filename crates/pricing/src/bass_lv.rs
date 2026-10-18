//! Experimental multi-marginal Bass-LV (Conze and Henry-Labordere, 2021).
//!
//! Marginals use a driftless martingale coordinate and share the initial mean.
//! See `docs/models/bass-local-volatility.md` for numerical contracts.
pub use crate::engine::bass_lv::{
    BASS_VEGA_KT_METHOD, BassCalibrationDiagnostics, BassDeterministicMappingRisk, BassEstimate,
    BassLvModel, BassMappingRisk, BassMappingRiskPlan, BassMarketIvModel, BassSimulationPlan,
    BassVegaKtRisk, BassVegaKtRiskPlan, BassVegaKtScenarioDiagnostics,
};
pub use crate::models::bass_lv::{
    BassError, BassLvConfig, BassLvSpec, BassMappingBump, BassMarginal, BassMarginalProjection,
    BassSurfaceDiagnostics, BassSurfaceProjectionConfig,
};
