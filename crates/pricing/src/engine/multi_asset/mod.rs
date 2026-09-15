//! Shared multi-asset compiler, sampling, path evolution and risk execution.
mod compile;
mod evaluate;
mod hull_white;
mod local_correlation;
mod lsv;
mod path;
use crate::Estimate;
use crate::core::{Date, UnderlyingId};
use crate::market::{AffineDividendCoordinate, CorrelationTermStructure, EquityForward};
use crate::mc::{
    BrownianBridgePlan, EngineConfig, ExecutionPolicy, LocalVolLogEulerPlan, RqmcPlan,
};
use crate::models::ModelSpec;
use crate::product::CompiledPayoff;
pub use hull_white::{
    MultiAssetHullWhiteConfig, MultiAssetHullWhiteCurveRisk, MultiAssetHullWhiteLsvRisk,
};
mod lsv_kernels;
pub use local_correlation::{
    LocalCorrelationCalibration, LocalCorrelationConfig, LocalCorrelationExtensions,
    LocalCorrelationFeasibility, LocalCorrelationNodeDiagnostics, LocalCorrelationRisk,
};
pub use lsv::{
    MultiAssetBergomiLsvConfig, MultiAssetLsv2FactorConfig, MultiAssetLsvConfig, MultiAssetLsvRisk,
    MultiAssetRoughLsvConfig,
};

#[derive(Clone, Debug)]
pub struct MultiAssetPricingPlan {
    valuation_date: Date,
    assets: Vec<Asset>,
    correlation: CorrelationTermStructure,
    interval_correlations: Vec<usize>,
    times: Vec<f64>,
    payoff: CompiledPayoff,
    engine: EngineConfig,
    execution: ExecutionPolicy,
    bridge: Option<BrownianBridgePlan>,
    qmc: Option<RqmcPlan>,
    fingerprint: String,
    lsv_drivers: Option<lsv::LsvDrivers>,
    hull_white: Option<hull_white::HwContext>,
    local_correlation: Option<std::sync::Arc<LocalCorrelationCalibration>>,
}
#[derive(Clone, Debug)]
struct Asset {
    forward: EquityForward,
    model: ModelSpec,
    process: LocalVolLogEulerPlan,
    coordinates: Vec<AffineDividendCoordinate>,
    pre_coordinates: Vec<AffineDividendCoordinate>,
    lsv: Option<std::sync::Arc<lsv::LsvAsset>>,
    hw: Option<std::sync::Arc<hull_white::HwAsset>>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MultiAssetRiskConfig {
    /// Relative central Spot bump for `Gamma[i,j] = d Delta_i / d Spot_j`.
    /// None computes first-order AAD only. References, payouts and grid axes stay fixed.
    pub gamma_relative_bump: Option<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetRisk {
    pub underlying: UnderlyingId,
    pub delta: Estimate,
    pub delta_per_one_percent_spot: Estimate,
    pub bs_vega: Option<Estimate>,
    pub bs_vega_per_vol_point: Option<Estimate>,
    /// dPV/d(effective local variance node), row-major in (time, log-moneyness).
    pub local_variance: Vec<Estimate>,
    pub local_variance_time_nodes: Vec<f64>,
    pub local_variance_log_moneyness_nodes: Vec<f64>,
    /// Effective target-grid risk through particle recalibration, when this asset is LSV.
    pub lsv_local_variance: Option<MultiAssetLsvRisk>,
    pub hull_white_lsv: Option<MultiAssetHullWhiteLsvRisk>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetPrice {
    pub price: Estimate,
    pub local_correlation_risk: Option<LocalCorrelationRisk>,
    pub hull_white_curve_risk: Option<MultiAssetHullWhiteCurveRisk>,
    pub risks: Vec<MultiAssetRisk>,
    /// Rows are Delta underlyings; columns are bumped Spot underlyings.
    /// Unsymmetrized central differences of pathwise Delta.
    pub gamma: Vec<Vec<Estimate>>,
    pub gamma_relative_bump: Option<f64>,
    pub underlyings: Vec<UnderlyingId>,
    pub fingerprint: String,
    pub evaluated_paths: u128,
    pub worker_threads: u32,
    pub reduction_block_size: u64,
    pub direction_checksum: Option<[u8; 32]>,
    pub scramble_checksum: Option<[u8; 32]>,
    /// Mean counts per evaluated path, one per asset. Flat wing interpolation is explicit.
    pub local_variance_boundary_counts: Vec<f64>,
    /// Mean flat-space leverage lookups per path; zero for non-LSV assets.
    pub lsv_leverage_boundary_counts: Vec<f64>,
}

impl Asset {
    fn has_lsv(&self) -> bool {
        self.lsv.is_some() || self.hw.as_ref().is_some_and(|a| a.calibration.is_some())
    }
}
