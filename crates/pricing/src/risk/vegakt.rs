use crate::core::PositiveF64;

pub const EQUATION_11_FIRST_ORDER_LABEL: &str = "equation_11_first_order_v1";

pub const EQUATION_11_TRUNCATION_ORDER: &str = "O(delta_t_k)";

pub const VEGA_KT_MARKET_SCALE: f64 = 0.01;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VegaKtBucketUnit {
    CurrencyPerUnitAbsoluteVolatility,
    CurrencyPerVolatilityPoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveDensityDomain {
    pub(crate) start_index: usize,
    pub(crate) end_index: usize,
    pub(crate) forward_index: usize,
    pub(crate) max_density_bits: u64,
    pub(crate) relative_threshold: PositiveF64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticCallDensityRow {
    pub(crate) maturity: PositiveF64,
    pub(crate) forward: PositiveF64,
    pub(crate) log_moneyness_nodes: Box<[f64]>,
    pub(crate) call_densities: Box<[f64]>,
    pub(crate) active_domain: ActiveDensityDomain,
    pub(crate) excluded_probability_mass: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equation11LocalGamma {
    pub(crate) value: f64,
    pub(crate) local_volatility: PositiveF64,
    pub(crate) policy_label: &'static str,
    pub(crate) truncation_order: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equation11RefinementLevel {
    pub(crate) delta_t: PositiveF64,
    pub(crate) approximation: f64,
    pub(crate) absolute_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Equation11RefinementDiagnostics {
    pub(crate) levels: Box<[Equation11RefinementLevel]>,
    pub(crate) observed_orders: Box<[Option<f64>]>,
    pub(crate) policy_label: &'static str,
    pub(crate) truncation_order: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionCellIntegral {
    pub(crate) lower_strike: f64,
    pub(crate) upper_strike: f64,
    pub(crate) probability: f64,
    pub(crate) first_moment: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportingIvBoundary {
    InRange,
    LeftEdge,
    RightEdge,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReportingIvProjectionStats {
    pub(crate) left_edge_count: u64,
    pub(crate) right_edge_count: u64,
    pub(crate) left_edge_sensitivity: f64,
    pub(crate) right_edge_sensitivity: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReportingIvInterpolation {
    pub(crate) value: f64,
    pub(crate) lower_time_index: usize,
    pub(crate) lower_log_moneyness_index: usize,
    pub(crate) time_weight: f64,
    pub(crate) log_moneyness_weight: f64,
    pub(crate) boundary: ReportingIvBoundary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReportingIvBasis {
    pub(crate) maturity_nodes: Box<[f64]>,
    pub(crate) log_moneyness_nodes: Box<[f64]>,
    pub(crate) implied_volatilities: Box<[f64]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtProjection {
    pub(crate) raw_buckets: Box<[f64]>,
    pub(crate) scalar_vega: f64,
    pub(crate) signed_residual: f64,
    pub(crate) pre_projection: f64,
    pub(crate) reporting_stats: ReportingIvProjectionStats,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtBucketEstimate {
    pub(crate) raw_mean: f64,
    pub(crate) market_scaled_mean: f64,
    pub(crate) sample_variance: Option<f64>,
    pub(crate) price_covariance: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VegaKtCovarianceLayout {
    PriceAndBucketVarianceOnly,
    FullBucketMatrixRowMajor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtBucketCoordinate {
    pub(crate) maturity: f64,
    pub(crate) log_moneyness: f64,
    pub(crate) implied_volatility: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtResidualDiagnostics {
    pub(crate) active_domain: ActiveDensityDomain,
    pub(crate) excluded_probability_mass: f64,
    pub(crate) signed_residual: f64,
    pub(crate) pre_projection: f64,
    pub(crate) reporting_stats: ReportingIvProjectionStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtReport {
    pub(crate) coordinates: Box<[VegaKtBucketCoordinate]>,
    pub(crate) estimates: Box<[VegaKtBucketEstimate]>,
    pub(crate) full_bucket_covariance: Option<Box<[Option<f64>]>>,
    pub(crate) covariance_layout: VegaKtCovarianceLayout,
    pub(crate) projection: VegaKtProjection,
    pub(crate) residual_diagnostics: VegaKtResidualDiagnostics,
    pub(crate) policy_label: &'static str,
    pub(crate) truncation_order: &'static str,
}
