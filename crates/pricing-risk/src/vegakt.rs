use pricing_core::{FiniteF64, PositiveF64};
use pricing_market::ImpliedVarianceSurface;
use pricing_numerics::{CenteredCovariance, CenteredMoment, NeumaierSum, standard_normal_cdf};

use crate::RiskConfigError;

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
    start_index: usize,
    end_index: usize,
    forward_index: usize,
    max_density_bits: u64,
    relative_threshold: PositiveF64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticCallDensityRow {
    maturity: PositiveF64,
    forward: PositiveF64,
    log_moneyness_nodes: Box<[f64]>,
    call_densities: Box<[f64]>,
    active_domain: ActiveDensityDomain,
    excluded_probability_mass: f64,
}

impl ActiveDensityDomain {
    #[must_use]
    pub const fn start_index(self) -> usize {
        self.start_index
    }

    #[must_use]
    pub const fn end_index(self) -> usize {
        self.end_index
    }

    #[must_use]
    pub const fn forward_index(self) -> usize {
        self.forward_index
    }

    #[must_use]
    pub const fn max_density(self) -> f64 {
        f64::from_bits(self.max_density_bits)
    }

    #[must_use]
    pub const fn relative_threshold(self) -> PositiveF64 {
        self.relative_threshold
    }

    #[must_use]
    pub const fn len(self) -> usize {
        self.end_index - self.start_index + 1
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        false
    }

    #[must_use]
    pub const fn contains_index(self, index: usize) -> bool {
        self.start_index <= index && index <= self.end_index
    }
}

impl AnalyticCallDensityRow {
    #[must_use]
    pub const fn maturity(&self) -> PositiveF64 {
        self.maturity
    }

    #[must_use]
    pub const fn forward(&self) -> PositiveF64 {
        self.forward
    }

    #[must_use]
    pub fn log_moneyness_nodes(&self) -> &[f64] {
        &self.log_moneyness_nodes
    }

    #[must_use]
    pub fn call_densities(&self) -> &[f64] {
        &self.call_densities
    }

    #[must_use]
    pub const fn active_domain(&self) -> ActiveDensityDomain {
        self.active_domain
    }

    #[must_use]
    pub const fn excluded_probability_mass(&self) -> f64 {
        self.excluded_probability_mass
    }
}

pub fn analytic_call_density_row_from_surface(
    surface: &dyn ImpliedVarianceSurface,
    maturity: f64,
    forward: f64,
    log_moneyness_nodes: Vec<f64>,
    relative_threshold: f64,
) -> Result<AnalyticCallDensityRow, RiskConfigError> {
    let maturity = PositiveF64::new(maturity, "vega_kt_density_maturity")?;
    let forward = PositiveF64::new(forward, "vega_kt_density_forward")?;
    validate_log_moneyness_nodes(&log_moneyness_nodes)?;
    let mut call_densities = Vec::with_capacity(log_moneyness_nodes.len());
    for log_moneyness in log_moneyness_nodes.iter().copied() {
        let evaluation =
            surface.forward_call_evaluation(maturity.get(), log_moneyness, forward.get())?;
        call_densities.push(evaluation.call_density);
    }
    let active_domain =
        active_density_domain(&log_moneyness_nodes, &call_densities, relative_threshold)?;
    let excluded_probability_mass = excluded_density_probability_mass(
        forward.get(),
        &log_moneyness_nodes,
        &call_densities,
        active_domain,
    )?;

    Ok(AnalyticCallDensityRow {
        maturity,
        forward,
        log_moneyness_nodes: log_moneyness_nodes.into_boxed_slice(),
        call_densities: call_densities.into_boxed_slice(),
        active_domain,
        excluded_probability_mass,
    })
}

pub fn analytic_call_density_rows_from_surface(
    surface: &dyn ImpliedVarianceSurface,
    maturity_nodes: &[f64],
    forwards: &[f64],
    log_moneyness_nodes: Vec<f64>,
    relative_threshold: f64,
) -> Result<Vec<AnalyticCallDensityRow>, RiskConfigError> {
    if maturity_nodes.is_empty() {
        return Err(RiskConfigError::TooFewVegaKtMaturities { count: 0 });
    }
    if forwards.len() != maturity_nodes.len() {
        return Err(RiskConfigError::VegaKtForwardLengthMismatch {
            maturity_count: maturity_nodes.len(),
            forward_count: forwards.len(),
        });
    }
    validate_log_moneyness_nodes(&log_moneyness_nodes)?;

    let mut rows = Vec::with_capacity(maturity_nodes.len());
    for (maturity, forward) in maturity_nodes.iter().copied().zip(forwards.iter().copied()) {
        rows.push(analytic_call_density_row_from_surface(
            surface,
            maturity,
            forward,
            log_moneyness_nodes.clone(),
            relative_threshold,
        )?);
    }
    Ok(rows)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equation11LocalGamma {
    value: f64,
    local_volatility: PositiveF64,
    policy_label: &'static str,
    truncation_order: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equation11RefinementLevel {
    delta_t: PositiveF64,
    approximation: f64,
    absolute_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Equation11RefinementDiagnostics {
    levels: Box<[Equation11RefinementLevel]>,
    observed_orders: Box<[Option<f64>]>,
    policy_label: &'static str,
    truncation_order: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionCellIntegral {
    lower_strike: f64,
    upper_strike: f64,
    probability: f64,
    first_moment: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportingIvBoundary {
    InRange,
    LeftEdge,
    RightEdge,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReportingIvProjectionStats {
    left_edge_count: u64,
    right_edge_count: u64,
    left_edge_sensitivity: f64,
    right_edge_sensitivity: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReportingIvInterpolation {
    value: f64,
    lower_time_index: usize,
    lower_log_moneyness_index: usize,
    time_weight: f64,
    log_moneyness_weight: f64,
    boundary: ReportingIvBoundary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReportingIvBasis {
    maturity_nodes: Box<[f64]>,
    log_moneyness_nodes: Box<[f64]>,
    implied_volatilities: Box<[f64]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtProjection {
    raw_buckets: Box<[f64]>,
    scalar_vega: f64,
    signed_residual: f64,
    pre_projection: f64,
    reporting_stats: ReportingIvProjectionStats,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtBucketEstimate {
    raw_mean: f64,
    market_scaled_mean: f64,
    sample_variance: Option<f64>,
    price_covariance: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VegaKtCovarianceLayout {
    PriceAndBucketVarianceOnly,
    FullBucketMatrixRowMajor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtBucketCoordinate {
    maturity: f64,
    log_moneyness: f64,
    implied_volatility: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtResidualDiagnostics {
    active_domain: ActiveDensityDomain,
    excluded_probability_mass: f64,
    signed_residual: f64,
    pre_projection: f64,
    reporting_stats: ReportingIvProjectionStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtReport {
    coordinates: Box<[VegaKtBucketCoordinate]>,
    estimates: Box<[VegaKtBucketEstimate]>,
    full_bucket_covariance: Option<Box<[Option<f64>]>>,
    covariance_layout: VegaKtCovarianceLayout,
    projection: VegaKtProjection,
    residual_diagnostics: VegaKtResidualDiagnostics,
    policy_label: &'static str,
    truncation_order: &'static str,
}

impl TransitionCellIntegral {
    #[must_use]
    pub const fn lower_strike(self) -> f64 {
        self.lower_strike
    }

    #[must_use]
    pub const fn upper_strike(self) -> f64 {
        self.upper_strike
    }

    #[must_use]
    pub const fn probability(self) -> f64 {
        self.probability
    }

    #[must_use]
    pub const fn first_moment(self) -> f64 {
        self.first_moment
    }
}

impl VegaKtBucketCoordinate {
    #[must_use]
    pub const fn maturity(self) -> f64 {
        self.maturity
    }

    #[must_use]
    pub const fn log_moneyness(self) -> f64 {
        self.log_moneyness
    }

    #[must_use]
    pub const fn implied_volatility(self) -> f64 {
        self.implied_volatility
    }
}

impl VegaKtBucketEstimate {
    #[must_use]
    pub const fn raw_mean(self) -> f64 {
        self.raw_mean
    }

    #[must_use]
    pub const fn market_scaled_mean(self) -> f64 {
        self.market_scaled_mean
    }

    #[must_use]
    pub const fn sample_variance(self) -> Option<f64> {
        self.sample_variance
    }

    #[must_use]
    pub const fn price_covariance(self) -> Option<f64> {
        self.price_covariance
    }

    #[must_use]
    pub const fn raw_unit(self) -> VegaKtBucketUnit {
        VegaKtBucketUnit::CurrencyPerUnitAbsoluteVolatility
    }

    #[must_use]
    pub const fn market_scaled_unit(self) -> VegaKtBucketUnit {
        VegaKtBucketUnit::CurrencyPerVolatilityPoint
    }
}

impl VegaKtResidualDiagnostics {
    #[must_use]
    pub const fn active_domain(self) -> ActiveDensityDomain {
        self.active_domain
    }

    #[must_use]
    pub const fn excluded_probability_mass(self) -> f64 {
        self.excluded_probability_mass
    }

    #[must_use]
    pub const fn signed_residual(self) -> f64 {
        self.signed_residual
    }

    #[must_use]
    pub const fn pre_projection(self) -> f64 {
        self.pre_projection
    }

    #[must_use]
    pub const fn reporting_stats(self) -> ReportingIvProjectionStats {
        self.reporting_stats
    }
}

impl VegaKtReport {
    #[must_use]
    pub fn coordinates(&self) -> &[VegaKtBucketCoordinate] {
        &self.coordinates
    }

    #[must_use]
    pub fn estimates(&self) -> &[VegaKtBucketEstimate] {
        &self.estimates
    }

    #[must_use]
    pub fn full_bucket_covariance(&self) -> Option<&[Option<f64>]> {
        self.full_bucket_covariance.as_deref()
    }

    #[must_use]
    pub const fn covariance_layout(&self) -> VegaKtCovarianceLayout {
        self.covariance_layout
    }

    #[must_use]
    pub const fn projection(&self) -> &VegaKtProjection {
        &self.projection
    }

    #[must_use]
    pub const fn residual_diagnostics(&self) -> VegaKtResidualDiagnostics {
        self.residual_diagnostics
    }

    #[must_use]
    pub const fn raw_unit(&self) -> VegaKtBucketUnit {
        VegaKtBucketUnit::CurrencyPerUnitAbsoluteVolatility
    }

    #[must_use]
    pub const fn market_scaled_unit(&self) -> VegaKtBucketUnit {
        VegaKtBucketUnit::CurrencyPerVolatilityPoint
    }

    #[must_use]
    pub const fn policy_label(&self) -> &'static str {
        self.policy_label
    }

    #[must_use]
    pub const fn truncation_order(&self) -> &'static str {
        self.truncation_order
    }
}

impl ReportingIvProjectionStats {
    #[must_use]
    pub const fn left_edge_count(self) -> u64 {
        self.left_edge_count
    }

    #[must_use]
    pub const fn right_edge_count(self) -> u64 {
        self.right_edge_count
    }

    #[must_use]
    pub const fn left_edge_sensitivity(self) -> f64 {
        self.left_edge_sensitivity
    }

    #[must_use]
    pub const fn right_edge_sensitivity(self) -> f64 {
        self.right_edge_sensitivity
    }

    #[must_use]
    pub const fn total_edge_count(self) -> u64 {
        self.left_edge_count + self.right_edge_count
    }
}

impl ReportingIvInterpolation {
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    #[must_use]
    pub const fn lower_time_index(self) -> usize {
        self.lower_time_index
    }

    #[must_use]
    pub const fn lower_log_moneyness_index(self) -> usize {
        self.lower_log_moneyness_index
    }

    #[must_use]
    pub const fn time_weight(self) -> f64 {
        self.time_weight
    }

    #[must_use]
    pub const fn log_moneyness_weight(self) -> f64 {
        self.log_moneyness_weight
    }

    #[must_use]
    pub const fn boundary(self) -> ReportingIvBoundary {
        self.boundary
    }

    pub fn transpose_accumulate(self, seed: f64, buckets: &mut [f64], x_count: usize) {
        let row = self.lower_time_index * x_count;
        let next_row = row + x_count;
        let left = self.lower_log_moneyness_index;
        let right = (left + 1).min(x_count - 1);
        let time_left = 1.0 - self.time_weight;
        let time_right = self.time_weight;
        let x_left = 1.0 - self.log_moneyness_weight;
        let x_right = self.log_moneyness_weight;
        buckets[row + left] += seed * time_left * x_left;
        buckets[row + right] += seed * time_left * x_right;
        buckets[next_row + left] += seed * time_right * x_left;
        buckets[next_row + right] += seed * time_right * x_right;
    }
}

impl Equation11LocalGamma {
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    #[must_use]
    pub const fn local_volatility(self) -> PositiveF64 {
        self.local_volatility
    }

    #[must_use]
    pub const fn policy_label(self) -> &'static str {
        self.policy_label
    }

    #[must_use]
    pub const fn truncation_order(self) -> &'static str {
        self.truncation_order
    }
}

impl Equation11RefinementLevel {
    #[must_use]
    pub const fn delta_t(self) -> PositiveF64 {
        self.delta_t
    }

    #[must_use]
    pub const fn approximation(self) -> f64 {
        self.approximation
    }

    #[must_use]
    pub const fn absolute_error(self) -> f64 {
        self.absolute_error
    }
}

impl Equation11RefinementDiagnostics {
    #[must_use]
    pub fn levels(&self) -> &[Equation11RefinementLevel] {
        &self.levels
    }

    #[must_use]
    pub fn observed_orders(&self) -> &[Option<f64>] {
        &self.observed_orders
    }

    #[must_use]
    pub const fn policy_label(&self) -> &'static str {
        self.policy_label
    }

    #[must_use]
    pub const fn truncation_order(&self) -> &'static str {
        self.truncation_order
    }
}

impl ReportingIvBasis {
    pub fn from_surface(
        surface: &dyn ImpliedVarianceSurface,
        maturity_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
    ) -> Result<Self, RiskConfigError> {
        validate_reporting_maturity_nodes(&maturity_nodes)?;
        validate_reporting_log_moneyness_nodes(&log_moneyness_nodes)?;

        let capacity = reporting_iv_value_count(maturity_nodes.len(), log_moneyness_nodes.len())?;
        let mut implied_volatilities = Vec::with_capacity(capacity);
        for maturity in maturity_nodes.iter().copied() {
            let maturity = PositiveF64::new(maturity, "vega_kt_reporting_surface_maturity")?.get();
            for log_moneyness in log_moneyness_nodes.iter().copied() {
                let variance = surface.total_variance_derivatives(maturity, log_moneyness)?;
                let volatility = PositiveF64::new(
                    variance.total_variance / maturity,
                    "vega_kt_reporting_variance",
                )?
                .get()
                .sqrt();
                implied_volatilities.push(volatility);
            }
        }

        Self::new(maturity_nodes, log_moneyness_nodes, implied_volatilities)
    }

    pub fn new(
        maturity_nodes: Vec<f64>,
        log_moneyness_nodes: Vec<f64>,
        implied_volatilities: Vec<f64>,
    ) -> Result<Self, RiskConfigError> {
        validate_reporting_maturity_nodes(&maturity_nodes)?;
        validate_reporting_log_moneyness_nodes(&log_moneyness_nodes)?;
        let expected = reporting_iv_value_count(maturity_nodes.len(), log_moneyness_nodes.len())?;
        if implied_volatilities.len() != expected {
            return Err(RiskConfigError::ReportingIvValueLengthMismatch {
                expected,
                actual: implied_volatilities.len(),
            });
        }
        for volatility in implied_volatilities.iter().copied() {
            PositiveF64::new(volatility, "vega_kt_reporting_implied_volatility")?;
        }
        Ok(Self {
            maturity_nodes: maturity_nodes.into_boxed_slice(),
            log_moneyness_nodes: log_moneyness_nodes.into_boxed_slice(),
            implied_volatilities: implied_volatilities.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn maturity_nodes(&self) -> &[f64] {
        &self.maturity_nodes
    }

    #[must_use]
    pub fn log_moneyness_nodes(&self) -> &[f64] {
        &self.log_moneyness_nodes
    }

    #[must_use]
    pub fn implied_volatilities(&self) -> &[f64] {
        &self.implied_volatilities
    }

    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.implied_volatilities.len()
    }

    pub fn interpolate(
        &self,
        maturity: f64,
        log_moneyness: f64,
    ) -> Result<ReportingIvInterpolation, RiskConfigError> {
        validate_finite(maturity, "vega_kt_reporting_query_maturity")?;
        validate_finite(log_moneyness, "vega_kt_reporting_query_log_moneyness")?;
        let time_index =
            reporting_interval(&self.maturity_nodes, maturity, "maturity").map_err(|_| {
                RiskConfigError::ReportingIvQueryTimeOutOfRange {
                    bits: maturity.to_bits(),
                }
            })?;
        let (x_index, x_weight, boundary) =
            reporting_log_moneyness_interval(&self.log_moneyness_nodes, log_moneyness);
        let time_weight = (maturity - self.maturity_nodes[time_index])
            / (self.maturity_nodes[time_index + 1] - self.maturity_nodes[time_index]);
        let value = self.bilinear_value(time_index, x_index, time_weight, x_weight);
        Ok(ReportingIvInterpolation {
            value,
            lower_time_index: time_index,
            lower_log_moneyness_index: x_index,
            time_weight,
            log_moneyness_weight: x_weight,
            boundary,
        })
    }

    pub fn project_weight(
        &self,
        maturity: f64,
        log_moneyness: f64,
        weight: f64,
        buckets: &mut [f64],
        stats: &mut ReportingIvProjectionStats,
    ) -> Result<(), RiskConfigError> {
        validate_finite(weight, "vega_kt_reporting_weight")?;
        if buckets.len() != self.bucket_count() {
            return Err(RiskConfigError::ReportingIvBucketLengthMismatch {
                expected: self.bucket_count(),
                actual: buckets.len(),
            });
        }
        let interpolation = self.interpolate(maturity, log_moneyness)?;
        match interpolation.boundary() {
            ReportingIvBoundary::InRange => {}
            ReportingIvBoundary::LeftEdge => {
                stats.left_edge_count += 1;
                stats.left_edge_sensitivity += weight;
            }
            ReportingIvBoundary::RightEdge => {
                stats.right_edge_count += 1;
                stats.right_edge_sensitivity += weight;
            }
        }
        interpolation.transpose_accumulate(weight, buckets, self.log_moneyness_nodes.len());
        Ok(())
    }

    fn bilinear_value(
        &self,
        time_index: usize,
        x_index: usize,
        time_weight: f64,
        x_weight: f64,
    ) -> f64 {
        let x_count = self.log_moneyness_nodes.len();
        let row = time_index * x_count;
        let next_row = row + x_count;
        let left = x_index;
        let right = (x_index + 1).min(x_count - 1);
        let time_left = 1.0 - time_weight;
        let time_right = time_weight;
        let x_left = 1.0 - x_weight;
        let x_right = x_weight;
        self.implied_volatilities[row + left] * time_left * x_left
            + self.implied_volatilities[row + right] * time_left * x_right
            + self.implied_volatilities[next_row + left] * time_right * x_left
            + self.implied_volatilities[next_row + right] * time_right * x_right
    }
}

impl VegaKtProjection {
    #[must_use]
    pub fn raw_buckets(&self) -> &[f64] {
        &self.raw_buckets
    }

    #[must_use]
    pub fn market_scaled_buckets(&self) -> Vec<f64> {
        self.raw_buckets
            .iter()
            .map(|bucket| bucket * VEGA_KT_MARKET_SCALE)
            .collect()
    }

    #[must_use]
    pub const fn scalar_vega(&self) -> f64 {
        self.scalar_vega
    }

    #[must_use]
    pub const fn signed_residual(&self) -> f64 {
        self.signed_residual
    }

    #[must_use]
    pub const fn pre_projection(&self) -> f64 {
        self.pre_projection
    }

    #[must_use]
    pub const fn reporting_stats(&self) -> ReportingIvProjectionStats {
        self.reporting_stats
    }

    #[must_use]
    pub fn reconciles(&self) -> bool {
        let reconstructed = self.reconstructed_pre_projection();
        (reconstructed - self.pre_projection).abs() <= 1.0e-12 * self.pre_projection.abs().max(1.0)
    }

    #[must_use]
    pub fn reconstructed_pre_projection(&self) -> f64 {
        let bucket_sum = self.raw_buckets.iter().copied().collect::<NeumaierSum>();
        bucket_sum.total() + self.signed_residual
    }
}

pub fn vega_kt_projection_from_parts(
    raw_buckets: Vec<f64>,
    signed_residual: f64,
    pre_projection: f64,
    reporting_stats: ReportingIvProjectionStats,
) -> Result<VegaKtProjection, RiskConfigError> {
    for bucket in &raw_buckets {
        validate_finite(*bucket, "vega_kt_raw_bucket")?;
    }
    validate_finite(signed_residual, "vega_kt_signed_residual")?;
    validate_finite(pre_projection, "vega_kt_pre_projection")?;
    let scalar_vega = raw_buckets.iter().copied().collect::<NeumaierSum>().total();
    let projection = VegaKtProjection {
        raw_buckets: raw_buckets.into_boxed_slice(),
        scalar_vega,
        signed_residual,
        pre_projection,
        reporting_stats,
    };
    if !projection.reconciles() {
        return Err(RiskConfigError::VegaKtProjectionReconciliationFailure {
            reconstructed_bits: projection.reconstructed_pre_projection().to_bits(),
            pre_projection_bits: projection.pre_projection().to_bits(),
        });
    }
    Ok(projection)
}

pub fn vega_kt_bucket_estimates(
    price_samples: &[f64],
    raw_bucket_samples: &[f64],
    bucket_count: usize,
) -> Result<Vec<VegaKtBucketEstimate>, RiskConfigError> {
    if price_samples.is_empty() || bucket_count == 0 {
        return Err(RiskConfigError::EmptyVegaKtBucketSamples);
    }
    let Some(expected) = price_samples.len().checked_mul(bucket_count) else {
        return Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
            expected_multiple: bucket_count,
            actual: raw_bucket_samples.len(),
        });
    };
    if raw_bucket_samples.len() != expected {
        return Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
            expected_multiple: bucket_count,
            actual: raw_bucket_samples.len(),
        });
    }
    for price in price_samples {
        validate_finite(*price, "vega_kt_price_sample")?;
    }
    for sample in raw_bucket_samples {
        validate_finite(*sample, "vega_kt_raw_bucket_sample")?;
    }

    let mut estimates = Vec::with_capacity(bucket_count);
    for bucket_index in 0..bucket_count {
        let mut bucket_values = Vec::with_capacity(price_samples.len());
        let mut covariance = CenteredCovariance::new();
        for (sample_index, price) in price_samples.iter().copied().enumerate() {
            let bucket = raw_bucket_samples[sample_index * bucket_count + bucket_index];
            bucket_values.push(bucket);
            covariance.add(price, bucket);
        }
        let moments = CenteredMoment::from_ordered_values_two_pass(&bucket_values);
        let raw_mean = moments.mean();
        estimates.push(VegaKtBucketEstimate {
            raw_mean,
            market_scaled_mean: raw_mean * VEGA_KT_MARKET_SCALE,
            sample_variance: moments.sample_variance(),
            price_covariance: covariance.sample_covariance(),
        });
    }
    Ok(estimates)
}

pub fn vega_kt_full_bucket_covariance(
    raw_bucket_samples: &[f64],
    bucket_count: usize,
) -> Result<Vec<Option<f64>>, RiskConfigError> {
    if bucket_count == 0 || raw_bucket_samples.is_empty() {
        return Err(RiskConfigError::EmptyVegaKtBucketSamples);
    }
    if !raw_bucket_samples.len().is_multiple_of(bucket_count) {
        return Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
            expected_multiple: bucket_count,
            actual: raw_bucket_samples.len(),
        });
    }
    for sample in raw_bucket_samples {
        validate_finite(*sample, "vega_kt_raw_bucket_sample")?;
    }

    let sample_count = raw_bucket_samples.len() / bucket_count;
    let mut covariance = Vec::with_capacity(bucket_count * bucket_count);
    for left_bucket in 0..bucket_count {
        for right_bucket in 0..bucket_count {
            let mut accumulator = CenteredCovariance::new();
            for sample_index in 0..sample_count {
                let row = sample_index * bucket_count;
                accumulator.add(
                    raw_bucket_samples[row + left_bucket],
                    raw_bucket_samples[row + right_bucket],
                );
            }
            covariance.push(accumulator.sample_covariance());
        }
    }
    Ok(covariance)
}

pub fn vega_kt_report(
    basis: &ReportingIvBasis,
    density_row: &AnalyticCallDensityRow,
    projection: VegaKtProjection,
    estimates: Vec<VegaKtBucketEstimate>,
    full_bucket_covariance: Option<Vec<Option<f64>>>,
) -> Result<VegaKtReport, RiskConfigError> {
    let bucket_count = basis.bucket_count();
    if projection.raw_buckets().len() != bucket_count {
        return Err(RiskConfigError::ReportingIvBucketLengthMismatch {
            expected: bucket_count,
            actual: projection.raw_buckets().len(),
        });
    }
    if estimates.len() != bucket_count {
        return Err(RiskConfigError::VegaKtBucketEstimateLengthMismatch {
            expected: bucket_count,
            actual: estimates.len(),
        });
    }
    if !projection.reconciles() {
        return Err(RiskConfigError::VegaKtProjectionReconciliationFailure {
            reconstructed_bits: projection.reconstructed_pre_projection().to_bits(),
            pre_projection_bits: projection.pre_projection().to_bits(),
        });
    }
    let full_bucket_covariance = match full_bucket_covariance {
        Some(covariance) => {
            let expected = bucket_count
                .checked_mul(bucket_count)
                .expect("bucket matrix size fits usize");
            if covariance.len() != expected {
                return Err(RiskConfigError::VegaKtFullCovarianceLengthMismatch {
                    expected,
                    actual: covariance.len(),
                });
            }
            Some(covariance.into_boxed_slice())
        }
        None => None,
    };
    let covariance_layout = if full_bucket_covariance.is_some() {
        VegaKtCovarianceLayout::FullBucketMatrixRowMajor
    } else {
        VegaKtCovarianceLayout::PriceAndBucketVarianceOnly
    };

    let mut coordinates = Vec::with_capacity(bucket_count);
    let x_count = basis.log_moneyness_nodes().len();
    for (bucket_index, implied_volatility) in
        basis.implied_volatilities().iter().copied().enumerate()
    {
        coordinates.push(VegaKtBucketCoordinate {
            maturity: basis.maturity_nodes()[bucket_index / x_count],
            log_moneyness: basis.log_moneyness_nodes()[bucket_index % x_count],
            implied_volatility,
        });
    }

    Ok(VegaKtReport {
        coordinates: coordinates.into_boxed_slice(),
        estimates: estimates.into_boxed_slice(),
        full_bucket_covariance,
        covariance_layout,
        residual_diagnostics: VegaKtResidualDiagnostics {
            active_domain: density_row.active_domain(),
            excluded_probability_mass: density_row.excluded_probability_mass(),
            signed_residual: projection.signed_residual(),
            pre_projection: projection.pre_projection(),
            reporting_stats: projection.reporting_stats(),
        },
        projection,
        policy_label: EQUATION_11_FIRST_ORDER_LABEL,
        truncation_order: EQUATION_11_TRUNCATION_ORDER,
    })
}

pub fn vega_kt_report_from_samples(
    basis: &ReportingIvBasis,
    density_row: &AnalyticCallDensityRow,
    projection: VegaKtProjection,
    price_samples: &[f64],
    raw_bucket_samples: &[f64],
    include_full_bucket_covariance: bool,
) -> Result<VegaKtReport, RiskConfigError> {
    let bucket_count = basis.bucket_count();
    let estimates = vega_kt_bucket_estimates(price_samples, raw_bucket_samples, bucket_count)?;
    let full_bucket_covariance = if include_full_bucket_covariance {
        Some(vega_kt_full_bucket_covariance(
            raw_bucket_samples,
            bucket_count,
        )?)
    } else {
        None
    };
    vega_kt_report(
        basis,
        density_row,
        projection,
        estimates,
        full_bucket_covariance,
    )
}

pub fn project_local_vega_nodes_to_reporting_iv(
    basis: &ReportingIvBasis,
    maturity: f64,
    log_moneyness_nodes: &[f64],
    local_vega_node_adjoints: &[f64],
    active_domain: ActiveDensityDomain,
) -> Result<VegaKtProjection, RiskConfigError> {
    validate_log_moneyness_nodes(log_moneyness_nodes)?;
    if local_vega_node_adjoints.len() != log_moneyness_nodes.len() {
        return Err(RiskConfigError::VegaKtDensityLengthMismatch {
            node_count: log_moneyness_nodes.len(),
            density_count: local_vega_node_adjoints.len(),
        });
    }
    if active_domain.end_index() >= log_moneyness_nodes.len() {
        return Err(RiskConfigError::VegaKtActiveDomainOutOfRange {
            end_index: active_domain.end_index(),
            node_count: log_moneyness_nodes.len(),
        });
    }

    let mut buckets = vec![0.0; basis.bucket_count()];
    let mut stats = ReportingIvProjectionStats::default();
    let mut residual = NeumaierSum::new();
    let mut pre_projection = NeumaierSum::new();
    for (index, adjoint) in local_vega_node_adjoints.iter().copied().enumerate() {
        validate_finite(adjoint, "vega_kt_local_vega_node_adjoint")?;
        pre_projection.add(adjoint);
        if active_domain.contains_index(index) {
            basis.project_weight(
                maturity,
                log_moneyness_nodes[index],
                adjoint,
                &mut buckets,
                &mut stats,
            )?;
        } else {
            residual.add(adjoint);
        }
    }
    let scalar_vega = buckets.iter().copied().collect::<NeumaierSum>().total();
    Ok(VegaKtProjection {
        raw_buckets: buckets.into_boxed_slice(),
        scalar_vega,
        signed_residual: residual.total(),
        pre_projection: pre_projection.total(),
        reporting_stats: stats,
    })
}

pub fn active_density_domain(
    log_moneyness_nodes: &[f64],
    call_densities: &[f64],
    relative_threshold: f64,
) -> Result<ActiveDensityDomain, RiskConfigError> {
    validate_domain_inputs(log_moneyness_nodes, call_densities)?;
    let relative_threshold =
        PositiveF64::new(relative_threshold, "vega_kt_relative_density_threshold")?;
    if relative_threshold.get() > 1.0 {
        return Err(RiskConfigError::InvalidDensityThreshold {
            bits: relative_threshold.get().to_bits(),
        });
    }

    let max_density = maximum_positive_density(call_densities)?;
    let forward_index = forward_node_index(log_moneyness_nodes);
    let qualifies = |density: f64| density / max_density >= relative_threshold.get();
    if !qualifies(call_densities[forward_index]) {
        return Err(RiskConfigError::ForwardNodeBelowDensityThreshold {
            forward_index,
            density_bits: call_densities[forward_index].to_bits(),
            max_density_bits: max_density.to_bits(),
            threshold_bits: relative_threshold.get().to_bits(),
        });
    }

    let mut start_index = forward_index;
    while start_index > 0 && qualifies(call_densities[start_index - 1]) {
        start_index -= 1;
    }
    let mut end_index = forward_index;
    while end_index + 1 < call_densities.len() && qualifies(call_densities[end_index + 1]) {
        end_index += 1;
    }

    Ok(ActiveDensityDomain {
        start_index,
        end_index,
        forward_index,
        max_density_bits: max_density.to_bits(),
        relative_threshold,
    })
}

pub fn mass_lumped_hat_areas(log_moneyness_nodes: &[f64]) -> Result<Vec<f64>, RiskConfigError> {
    validate_log_moneyness_nodes(log_moneyness_nodes)?;
    let mut areas = Vec::with_capacity(log_moneyness_nodes.len());
    areas.push(0.5 * (log_moneyness_nodes[1] - log_moneyness_nodes[0]));
    for window in log_moneyness_nodes.windows(3) {
        areas.push(0.5 * (window[2] - window[0]));
    }
    let last = log_moneyness_nodes.len() - 1;
    areas.push(0.5 * (log_moneyness_nodes[last] - log_moneyness_nodes[last - 1]));
    Ok(areas)
}

pub fn local_vega_density_from_node_adjoints(
    node_adjoints: &[f64],
    log_moneyness_nodes: &[f64],
) -> Result<Vec<f64>, RiskConfigError> {
    let areas = mass_lumped_hat_areas(log_moneyness_nodes)?;
    if node_adjoints.len() != areas.len() {
        return Err(RiskConfigError::VegaKtDensityLengthMismatch {
            node_count: areas.len(),
            density_count: node_adjoints.len(),
        });
    }
    node_adjoints
        .iter()
        .zip(areas)
        .enumerate()
        .map(|(index, (adjoint, area))| {
            validate_finite(*adjoint, "vega_kt_node_adjoint")?;
            if area <= 0.0 {
                return Err(RiskConfigError::UnsortedVegaKtDomainNodes {
                    left_index: index.saturating_sub(1),
                });
            }
            Ok(adjoint / area)
        })
        .collect()
}

pub fn equation_11_local_gamma_from_strike_vega(
    local_vega_strike_density: f64,
    strike: f64,
    call_density: f64,
    local_variance: f64,
    delta_t: f64,
) -> Result<Equation11LocalGamma, RiskConfigError> {
    validate_finite(
        local_vega_strike_density,
        "vega_kt_local_vega_strike_density",
    )?;
    let strike = PositiveF64::new(strike, "vega_kt_equation_11_strike")?;
    let call_density = PositiveF64::new(call_density, "vega_kt_equation_11_call_density")?;
    let local_variance = PositiveF64::new(local_variance, "vega_kt_equation_11_local_variance")?;
    let delta_t = PositiveF64::new(delta_t, "vega_kt_equation_11_delta_t")?;
    let local_volatility = PositiveF64::new(
        local_variance.get().sqrt(),
        "vega_kt_equation_11_local_volatility",
    )?;
    let denominator =
        strike.get() * strike.get() * call_density.get() * local_volatility.get() * delta_t.get();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(RiskConfigError::InvalidVegaKtEquation11Denominator {
            bits: denominator.to_bits(),
        });
    }
    let value = local_vega_strike_density / denominator;
    if !value.is_finite() {
        return Err(RiskConfigError::InvalidVegaKtEquation11Gamma {
            bits: value.to_bits(),
        });
    }
    Ok(Equation11LocalGamma {
        value,
        local_volatility,
        policy_label: EQUATION_11_FIRST_ORDER_LABEL,
        truncation_order: EQUATION_11_TRUNCATION_ORDER,
    })
}

pub fn equation_11_local_gamma_from_log_moneyness_vega(
    local_vega_log_moneyness_density: f64,
    strike: f64,
    call_density: f64,
    local_variance: f64,
    delta_t: f64,
) -> Result<Equation11LocalGamma, RiskConfigError> {
    validate_finite(
        local_vega_log_moneyness_density,
        "vega_kt_local_vega_log_moneyness_density",
    )?;
    let strike = PositiveF64::new(strike, "vega_kt_equation_11_strike")?;
    equation_11_local_gamma_from_strike_vega(
        local_vega_log_moneyness_density / strike.get(),
        strike.get(),
        call_density,
        local_variance,
        delta_t,
    )
}

pub fn equation_11_refinement_diagnostics(
    delta_t_levels: &[f64],
    approximations: &[f64],
    reference_value: f64,
) -> Result<Equation11RefinementDiagnostics, RiskConfigError> {
    if delta_t_levels.len() < 2 {
        return Err(RiskConfigError::TooFewVegaKtRefinementLevels {
            count: delta_t_levels.len(),
        });
    }
    if approximations.len() != delta_t_levels.len() {
        return Err(RiskConfigError::VegaKtGammaLengthMismatch {
            strike_count: delta_t_levels.len(),
            gamma_count: approximations.len(),
        });
    }
    validate_finite(reference_value, "vega_kt_equation_11_refinement_reference")?;

    let mut levels = Vec::with_capacity(delta_t_levels.len());
    for (index, (delta_t, approximation)) in delta_t_levels
        .iter()
        .copied()
        .zip(approximations.iter().copied())
        .enumerate()
    {
        let delta_t = PositiveF64::new(delta_t, "vega_kt_equation_11_refinement_delta_t")?;
        if index > 0 && delta_t.get() >= delta_t_levels[index - 1] {
            return Err(RiskConfigError::NonDecreasingVegaKtRefinementStep {
                left_index: index - 1,
            });
        }
        validate_finite(
            approximation,
            "vega_kt_equation_11_refinement_approximation",
        )?;
        levels.push(Equation11RefinementLevel {
            delta_t,
            approximation,
            absolute_error: (approximation - reference_value).abs(),
        });
    }

    let mut observed_orders = Vec::with_capacity(levels.len() - 1);
    for pair in levels.windows(2) {
        let coarse = pair[0];
        let fine = pair[1];
        let order = if coarse.absolute_error() > 0.0 && fine.absolute_error() > 0.0 {
            Some(
                (coarse.absolute_error() / fine.absolute_error()).ln()
                    / (coarse.delta_t().get() / fine.delta_t().get()).ln(),
            )
        } else {
            None
        };
        observed_orders.push(order);
    }

    Ok(Equation11RefinementDiagnostics {
        levels: levels.into_boxed_slice(),
        observed_orders: observed_orders.into_boxed_slice(),
        policy_label: EQUATION_11_FIRST_ORDER_LABEL,
        truncation_order: EQUATION_11_TRUNCATION_ORDER,
    })
}

pub fn local_gamma_transition_cells(
    spot: f64,
    local_variance_times_delta_t: f64,
    strike_nodes: &[f64],
) -> Result<Vec<TransitionCellIntegral>, RiskConfigError> {
    let spot = PositiveF64::new(spot, "vega_kt_transition_spot")?;
    let variance = PositiveF64::new(local_variance_times_delta_t, "vega_kt_transition_variance")?;
    validate_transition_strikes(strike_nodes)?;

    let standard_deviation = variance.get().sqrt();
    let mean = spot.get().ln() + 1.5 * variance.get();
    let moment_scale = (mean + 0.5 * variance.get()).exp();
    let mut cells = Vec::with_capacity(strike_nodes.len() + 1);
    let mut probability_sum = NeumaierSum::new();

    for cell_index in 0..=strike_nodes.len() {
        let lower = if cell_index == 0 {
            0.0
        } else {
            strike_nodes[cell_index - 1]
        };
        let upper = if cell_index == strike_nodes.len() {
            f64::INFINITY
        } else {
            strike_nodes[cell_index]
        };
        let probability = if cell_index == strike_nodes.len() {
            let raw = 1.0 - probability_sum.total();
            if !raw.is_finite() {
                return Err(RiskConfigError::InvalidVegaKtTransitionProbability {
                    cell_index,
                    bits: raw.to_bits(),
                });
            }
            if !(-8.0 * f64::EPSILON..=1.0 + 8.0 * f64::EPSILON).contains(&raw) {
                return Err(RiskConfigError::VegaKtTransitionMassDefect {
                    bits: raw.to_bits(),
                });
            }
            raw.clamp(0.0, 1.0)
        } else {
            let probability = lognormal_probability(mean, standard_deviation, lower, upper);
            if !probability.is_finite() {
                return Err(RiskConfigError::InvalidVegaKtTransitionProbability {
                    cell_index,
                    bits: probability.to_bits(),
                });
            }
            probability_sum.add(probability);
            probability
        };
        let first_moment = moment_scale
            * lognormal_probability(mean + variance.get(), standard_deviation, lower, upper);
        cells.push(TransitionCellIntegral {
            lower_strike: lower,
            upper_strike: upper,
            probability,
            first_moment,
        });
    }
    Ok(cells)
}

pub fn integrate_piecewise_linear_local_gamma_transition(
    spot: f64,
    local_variance_times_delta_t: f64,
    strike_nodes: &[f64],
    local_gamma_values: &[f64],
) -> Result<f64, RiskConfigError> {
    let cells = local_gamma_transition_cells(spot, local_variance_times_delta_t, strike_nodes)?;
    if local_gamma_values.len() != strike_nodes.len() {
        return Err(RiskConfigError::VegaKtGammaLengthMismatch {
            strike_count: strike_nodes.len(),
            gamma_count: local_gamma_values.len(),
        });
    }
    for gamma in local_gamma_values {
        validate_finite(*gamma, "vega_kt_local_gamma")?;
    }

    let variance = PositiveF64::new(local_variance_times_delta_t, "vega_kt_transition_variance")?;
    let mut integral = NeumaierSum::new();
    for (cell_index, cell) in cells.iter().copied().enumerate() {
        let contribution = if cell_index == 0 {
            local_gamma_values[0] * cell.probability()
        } else if cell_index == cells.len() - 1 {
            local_gamma_values[local_gamma_values.len() - 1] * cell.probability()
        } else {
            let left_index = cell_index - 1;
            let right_index = cell_index;
            let left_strike = strike_nodes[left_index];
            let right_strike = strike_nodes[right_index];
            let slope = (local_gamma_values[right_index] - local_gamma_values[left_index])
                / (right_strike - left_strike);
            let intercept = local_gamma_values[left_index] - slope * left_strike;
            intercept * cell.probability() + slope * cell.first_moment()
        };
        integral.add(contribution);
    }
    let value = variance.get().exp() * integral.total();
    if !value.is_finite() {
        return Err(RiskConfigError::InvalidVegaKtTransitionValue {
            bits: value.to_bits(),
        });
    }
    Ok(value)
}

fn validate_domain_inputs(
    log_moneyness_nodes: &[f64],
    call_densities: &[f64],
) -> Result<(), RiskConfigError> {
    validate_log_moneyness_nodes(log_moneyness_nodes)?;
    if call_densities.len() != log_moneyness_nodes.len() {
        return Err(RiskConfigError::VegaKtDensityLengthMismatch {
            node_count: log_moneyness_nodes.len(),
            density_count: call_densities.len(),
        });
    }
    for (index, density) in call_densities.iter().copied().enumerate() {
        validate_finite(density, "vega_kt_call_density")?;
        if density < 0.0 {
            return Err(RiskConfigError::NegativeVegaKtDensity {
                index,
                bits: density.to_bits(),
            });
        }
    }
    Ok(())
}

fn validate_transition_strikes(strike_nodes: &[f64]) -> Result<(), RiskConfigError> {
    if strike_nodes.len() < 2 {
        return Err(RiskConfigError::TooFewVegaKtTransitionStrikes {
            count: strike_nodes.len(),
        });
    }
    for (index, strike) in strike_nodes.iter().copied().enumerate() {
        PositiveF64::new(strike, "vega_kt_transition_strike")?;
        if index > 0 && strike <= strike_nodes[index - 1] {
            return Err(RiskConfigError::UnsortedVegaKtTransitionStrikes {
                left_index: index - 1,
            });
        }
    }
    Ok(())
}

fn validate_reporting_maturity_nodes(nodes: &[f64]) -> Result<(), RiskConfigError> {
    if nodes.len() < 2 {
        return Err(RiskConfigError::TooFewReportingIvMaturities { count: nodes.len() });
    }
    for (index, value) in nodes.iter().copied().enumerate() {
        validate_finite(value, "vega_kt_reporting_maturity")?;
        if index > 0 && value <= nodes[index - 1] {
            return Err(RiskConfigError::UnsortedReportingIvMaturities {
                left_index: index - 1,
            });
        }
    }
    Ok(())
}

fn validate_reporting_log_moneyness_nodes(nodes: &[f64]) -> Result<(), RiskConfigError> {
    if nodes.len() < 2 {
        return Err(RiskConfigError::TooFewReportingIvStrikes { count: nodes.len() });
    }
    for (index, value) in nodes.iter().copied().enumerate() {
        validate_finite(value, "vega_kt_reporting_log_moneyness")?;
        if index > 0 && value <= nodes[index - 1] {
            return Err(RiskConfigError::UnsortedReportingIvStrikes {
                left_index: index - 1,
            });
        }
    }
    Ok(())
}

fn reporting_iv_value_count(
    maturity_count: usize,
    log_moneyness_count: usize,
) -> Result<usize, RiskConfigError> {
    maturity_count.checked_mul(log_moneyness_count).ok_or(
        RiskConfigError::ReportingIvValueLengthMismatch {
            expected: usize::MAX,
            actual: 0,
        },
    )
}

fn excluded_density_probability_mass(
    forward: f64,
    log_moneyness_nodes: &[f64],
    call_densities: &[f64],
    active_domain: ActiveDensityDomain,
) -> Result<f64, RiskConfigError> {
    let hat_areas = mass_lumped_hat_areas(log_moneyness_nodes)?;
    let mut excluded_mass = NeumaierSum::new();
    for (index, ((log_moneyness, density), hat_area)) in log_moneyness_nodes
        .iter()
        .copied()
        .zip(call_densities.iter().copied())
        .zip(hat_areas.iter().copied())
        .enumerate()
    {
        if !active_domain.contains_index(index) {
            let strike = forward * log_moneyness.exp();
            excluded_mass.add(density * strike * hat_area);
        }
    }
    let mass = excluded_mass.total();
    validate_finite(mass, "vega_kt_excluded_probability_mass")?;
    Ok(mass.max(0.0))
}

fn reporting_interval(
    nodes: &[f64],
    value: f64,
    axis: &'static str,
) -> Result<usize, &'static str> {
    if value < nodes[0] || value > nodes[nodes.len() - 1] {
        return Err(axis);
    }
    Ok(
        match nodes.binary_search_by(|node| node.total_cmp(&value)) {
            Ok(index) => index.min(nodes.len() - 2),
            Err(index) => index - 1,
        },
    )
}

fn reporting_log_moneyness_interval(
    nodes: &[f64],
    value: f64,
) -> (usize, f64, ReportingIvBoundary) {
    if value <= nodes[0] {
        return (0, 0.0, ReportingIvBoundary::LeftEdge);
    }
    let last = nodes.len() - 1;
    if value >= nodes[last] {
        return (last - 1, 1.0, ReportingIvBoundary::RightEdge);
    }
    let index = match nodes.binary_search_by(|node| node.total_cmp(&value)) {
        Ok(index) => index.min(nodes.len() - 2),
        Err(index) => index - 1,
    };
    let weight = (value - nodes[index]) / (nodes[index + 1] - nodes[index]);
    (index, weight, ReportingIvBoundary::InRange)
}

fn lognormal_probability(mean: f64, standard_deviation: f64, lower: f64, upper: f64) -> f64 {
    let lower_z = if lower == 0.0 {
        f64::NEG_INFINITY
    } else {
        (lower.ln() - mean) / standard_deviation
    };
    let upper_z = if upper.is_infinite() {
        f64::INFINITY
    } else {
        (upper.ln() - mean) / standard_deviation
    };
    normal_interval_probability(lower_z, upper_z)
}

fn normal_interval_probability(lower_z: f64, upper_z: f64) -> f64 {
    let direct = standard_normal_cdf(upper_z) - standard_normal_cdf(lower_z);
    let survival = standard_normal_cdf(-lower_z) - standard_normal_cdf(-upper_z);
    if lower_z > 0.0 || upper_z < 0.0 {
        direct.min(survival).max(0.0)
    } else {
        direct.max(0.0)
    }
}

fn validate_log_moneyness_nodes(log_moneyness_nodes: &[f64]) -> Result<(), RiskConfigError> {
    if log_moneyness_nodes.len() < 2 {
        return Err(RiskConfigError::TooFewVegaKtDomainNodes {
            count: log_moneyness_nodes.len(),
        });
    }
    for (index, value) in log_moneyness_nodes.iter().copied().enumerate() {
        validate_finite(value, "vega_kt_domain_log_moneyness")?;
        if index > 0 && value <= log_moneyness_nodes[index - 1] {
            return Err(RiskConfigError::UnsortedVegaKtDomainNodes {
                left_index: index - 1,
            });
        }
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), RiskConfigError> {
    FiniteF64::new(value, field).map(|_| ()).map_err(Into::into)
}

fn maximum_positive_density(call_densities: &[f64]) -> Result<f64, RiskConfigError> {
    let max_density = call_densities
        .iter()
        .copied()
        .fold(0.0_f64, |left, right| left.max(right));
    if max_density > 0.0 {
        Ok(max_density)
    } else {
        Err(RiskConfigError::NoPositiveVegaKtDensity)
    }
}

fn forward_node_index(log_moneyness_nodes: &[f64]) -> usize {
    let mut forward_index = 0;
    let mut min_abs = log_moneyness_nodes[0].abs();
    for (index, value) in log_moneyness_nodes.iter().copied().enumerate().skip(1) {
        let abs = value.abs();
        if abs < min_abs {
            min_abs = abs;
            forward_index = index;
        }
    }
    forward_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use pricing_market::{MarketError, ThetaRegion, TotalVarianceDerivatives};

    struct ConstantVolSurface {
        volatility: f64,
    }

    impl ImpliedVarianceSurface for ConstantVolSurface {
        fn total_variance_derivatives(
            &self,
            time: f64,
            _log_moneyness: f64,
        ) -> Result<TotalVarianceDerivatives, MarketError> {
            Ok(TotalVarianceDerivatives {
                total_variance: self.volatility * self.volatility * time,
                log_moneyness_derivative: 0.0,
                log_moneyness_second_derivative: 0.0,
                time_derivative: self.volatility * self.volatility,
                theta: self.volatility * self.volatility * time,
                theta_derivative: self.volatility * self.volatility,
                theta_region: ThetaRegion::Interpolated,
            })
        }
    }

    struct FailingSurface;

    impl ImpliedVarianceSurface for FailingSurface {
        fn total_variance_derivatives(
            &self,
            time: f64,
            log_moneyness: f64,
        ) -> Result<TotalVarianceDerivatives, MarketError> {
            Err(MarketError::InvalidSurfaceQuery {
                coordinate: "test",
                bits: (time + log_moneyness).to_bits(),
            })
        }
    }

    fn assert_close(left: f64, right: f64) {
        let tolerance = 1.0e-14_f64.max(1.0e-14 * right.abs().max(1.0));
        assert!(
            (left - right).abs() < tolerance,
            "expected {left} to be close to {right}"
        );
    }

    fn test_density_row() -> AnalyticCallDensityRow {
        analytic_call_density_row_from_surface(
            &ConstantVolSurface { volatility: 0.25 },
            1.0,
            100.0,
            vec![-0.4, 0.0, 0.4],
            0.25,
        )
        .expect("density row")
    }

    #[test]
    fn active_domain_is_connected_component_containing_forward() {
        let nodes = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6];
        let densities = [0.9, 0.1, 0.8, 0.7, 0.2, 1.0];
        let domain = active_density_domain(&nodes, &densities, 0.5).expect("domain");

        assert_eq!(domain.start_index(), 2);
        assert_eq!(domain.end_index(), 3);
        assert_eq!(domain.forward_index(), 2);
        assert_eq!(domain.len(), 2);
        assert!(!domain.contains_index(5));
        assert_eq!(domain.max_density(), 1.0);
    }

    #[test]
    fn forward_node_uses_lower_index_on_absolute_tie() {
        let nodes = [-0.1, 0.1, 0.3];
        let densities = [0.8, 0.8, 1.0];
        let domain = active_density_domain(&nodes, &densities, 0.75).expect("domain");

        assert_eq!(domain.forward_index(), 0);
    }

    #[test]
    fn active_domain_errors_when_forward_does_not_qualify() {
        let nodes = [-0.2, 0.0, 0.2];
        let densities = [1.0, 0.1, 1.0];

        assert!(matches!(
            active_density_domain(&nodes, &densities, 0.5),
            Err(RiskConfigError::ForwardNodeBelowDensityThreshold {
                forward_index: 1,
                ..
            })
        ));
    }

    #[test]
    fn active_domain_rejects_invalid_density_grid() {
        assert!(matches!(
            active_density_domain(&[0.0], &[1.0], 0.5),
            Err(RiskConfigError::TooFewVegaKtDomainNodes { count: 1 })
        ));
        assert!(matches!(
            active_density_domain(&[0.0, 0.1], &[1.0], 0.5),
            Err(RiskConfigError::VegaKtDensityLengthMismatch {
                node_count: 2,
                density_count: 1
            })
        ));
        assert!(matches!(
            active_density_domain(&[0.0, 0.1], &[1.0, -0.1], 0.5),
            Err(RiskConfigError::NegativeVegaKtDensity { index: 1, .. })
        ));
        assert!(matches!(
            active_density_domain(&[0.0, 0.1], &[0.0, 0.0], 0.5),
            Err(RiskConfigError::NoPositiveVegaKtDensity)
        ));
    }

    #[test]
    fn analytic_call_density_row_samples_surface_and_reports_excluded_mass() {
        let surface = ConstantVolSurface { volatility: 0.25 };
        let row = analytic_call_density_row_from_surface(
            &surface,
            1.0,
            100.0,
            vec![-1.0, -0.2, 0.0, 0.2, 1.0],
            0.5,
        )
        .expect("density row");

        assert_eq!(row.maturity().get(), 1.0);
        assert_eq!(row.forward().get(), 100.0);
        assert_eq!(row.call_densities().len(), 5);
        assert!(row.call_densities().iter().all(|density| *density >= 0.0));
        assert_eq!(row.active_domain().forward_index(), 2);
        assert!(row.active_domain().contains_index(2));
        assert!(row.excluded_probability_mass() > 0.0);
    }

    #[test]
    fn analytic_call_density_row_propagates_surface_errors() {
        let error = analytic_call_density_row_from_surface(
            &FailingSurface,
            1.0,
            100.0,
            vec![-0.2, 0.0, 0.2],
            0.5,
        )
        .expect_err("surface error");

        assert!(matches!(error, RiskConfigError::Market(_)));
    }

    #[test]
    fn analytic_call_density_rows_materialize_every_operator_maturity() {
        let surface = ConstantVolSurface { volatility: 0.25 };
        let rows = analytic_call_density_rows_from_surface(
            &surface,
            &[0.5, 1.0, 2.0],
            &[99.0, 100.0, 101.0],
            vec![-0.4, 0.0, 0.4],
            0.25,
        )
        .expect("density rows");

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].maturity().get(), 0.5);
        assert_eq!(rows[1].forward().get(), 100.0);
        assert_eq!(rows[2].log_moneyness_nodes(), &[-0.4, 0.0, 0.4]);
        assert!(rows.iter().all(|row| {
            row.active_domain()
                .contains_index(row.active_domain().forward_index())
        }));
    }

    #[test]
    fn analytic_call_density_rows_reject_forward_shape_mismatch() {
        assert!(matches!(
            analytic_call_density_rows_from_surface(
                &ConstantVolSurface { volatility: 0.25 },
                &[0.5, 1.0],
                &[100.0],
                vec![-0.4, 0.0, 0.4],
                0.25,
            ),
            Err(RiskConfigError::VegaKtForwardLengthMismatch {
                maturity_count: 2,
                forward_count: 1,
            })
        ));
    }

    #[test]
    fn mass_lumped_hat_areas_match_piecewise_linear_basis_integrals() {
        let areas = mass_lumped_hat_areas(&[-0.3, -0.1, 0.2, 0.6]).expect("areas");

        assert_close(areas[0], 0.1);
        assert_close(areas[1], 0.25);
        assert_close(areas[2], 0.35);
        assert_close(areas[3], 0.2);
        assert_close(areas.iter().sum::<f64>(), 0.9);
    }

    #[test]
    fn local_vega_density_divides_node_adjoints_by_hat_areas() {
        let density =
            local_vega_density_from_node_adjoints(&[0.2, 0.5, 0.7, 0.4], &[-0.3, -0.1, 0.2, 0.6])
                .expect("density");

        for value in density {
            assert_close(value, 2.0);
        }
    }

    #[test]
    fn equation_11_recovers_local_gamma_with_metadata() {
        let gamma =
            equation_11_local_gamma_from_strike_vega(0.24, 2.0, 0.3, 0.04, 0.5).expect("gamma");

        assert_close(gamma.value(), 2.0);
        assert_close(gamma.local_volatility().get(), 0.2);
        assert_eq!(gamma.policy_label(), EQUATION_11_FIRST_ORDER_LABEL);
        assert_eq!(gamma.truncation_order(), EQUATION_11_TRUNCATION_ORDER);
    }

    #[test]
    fn equation_11_converts_log_moneyness_density_to_strike_density() {
        let gamma = equation_11_local_gamma_from_log_moneyness_vega(0.48, 2.0, 0.3, 0.04, 0.5)
            .expect("gamma");

        assert_close(gamma.value(), 2.0);
    }

    #[test]
    fn equation_11_rejects_invalid_positive_inputs() {
        assert!(equation_11_local_gamma_from_strike_vega(1.0, 0.0, 0.3, 0.04, 0.5).is_err());
        assert!(equation_11_local_gamma_from_strike_vega(1.0, 2.0, 0.0, 0.04, 0.5).is_err());
        assert!(equation_11_local_gamma_from_strike_vega(1.0, 2.0, 0.3, 0.0, 0.5).is_err());
        assert!(equation_11_local_gamma_from_strike_vega(1.0, 2.0, 0.3, 0.04, 0.0).is_err());
    }

    #[test]
    fn equation_11_refinement_diagnostics_reports_first_order_behavior() {
        let diagnostics =
            equation_11_refinement_diagnostics(&[0.5, 0.25, 0.125], &[2.5, 2.25, 2.125], 2.0)
                .expect("diagnostics");

        assert_eq!(diagnostics.levels().len(), 3);
        assert_close(diagnostics.levels()[0].delta_t().get(), 0.5);
        assert_close(diagnostics.levels()[0].approximation(), 2.5);
        assert_close(diagnostics.levels()[0].absolute_error(), 0.5);
        assert_close(diagnostics.observed_orders()[0].expect("order"), 1.0);
        assert_close(diagnostics.observed_orders()[1].expect("order"), 1.0);
        assert_eq!(diagnostics.policy_label(), EQUATION_11_FIRST_ORDER_LABEL);
        assert_eq!(diagnostics.truncation_order(), EQUATION_11_TRUNCATION_ORDER);
    }

    #[test]
    fn equation_11_refinement_diagnostics_rejects_invalid_shapes_and_steps() {
        assert!(matches!(
            equation_11_refinement_diagnostics(&[0.5], &[2.5], 2.0),
            Err(RiskConfigError::TooFewVegaKtRefinementLevels { count: 1 })
        ));
        assert!(matches!(
            equation_11_refinement_diagnostics(&[0.5, 0.25], &[2.5], 2.0),
            Err(RiskConfigError::VegaKtGammaLengthMismatch {
                strike_count: 2,
                gamma_count: 1
            })
        ));
        assert!(matches!(
            equation_11_refinement_diagnostics(&[0.25, 0.5], &[2.25, 2.5], 2.0),
            Err(RiskConfigError::NonDecreasingVegaKtRefinementStep { left_index: 0 })
        ));
    }

    #[test]
    fn transition_cells_conserve_probability_and_match_lognormal_first_moment() {
        let cells = local_gamma_transition_cells(100.0, 0.04, &[90.0, 110.0]).expect("cells");

        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0].lower_strike(), 0.0);
        assert_eq!(cells[0].upper_strike(), 90.0);
        assert_eq!(cells[2].lower_strike(), 110.0);
        assert_eq!(cells[2].upper_strike(), f64::INFINITY);

        let probability = cells.iter().map(|cell| cell.probability()).sum::<f64>();
        let first_moment = cells.iter().map(|cell| cell.first_moment()).sum::<f64>();
        assert_close(probability, 1.0);
        assert_close(first_moment, 100.0 * (2.0_f64 * 0.04).exp());
    }

    #[test]
    fn transition_cells_reject_invalid_strikes() {
        assert!(matches!(
            local_gamma_transition_cells(100.0, 0.04, &[100.0]),
            Err(RiskConfigError::TooFewVegaKtTransitionStrikes { count: 1 })
        ));
        assert!(matches!(
            local_gamma_transition_cells(100.0, 0.04, &[100.0, 100.0]),
            Err(RiskConfigError::UnsortedVegaKtTransitionStrikes { left_index: 0 })
        ));
        assert!(local_gamma_transition_cells(100.0, 0.0, &[90.0, 110.0]).is_err());
        assert!(local_gamma_transition_cells(0.0, 0.04, &[90.0, 110.0]).is_err());
    }

    #[test]
    fn transition_operator_preserves_constant_gamma_with_kernel_normalization() {
        let value = integrate_piecewise_linear_local_gamma_transition(
            100.0,
            0.04,
            &[90.0, 110.0],
            &[3.0, 3.0],
        )
        .expect("transition");

        assert_close(value, 3.0 * 0.04_f64.exp());
    }

    #[test]
    fn transition_operator_matches_linear_gamma_reconstruction() {
        let spot = 100.0;
        let variance = 0.04;
        let strikes = [90.0, 110.0];
        let gamma = [1.0, 2.0];
        let value =
            integrate_piecewise_linear_local_gamma_transition(spot, variance, &strikes, &gamma)
                .expect("transition");
        let cells = local_gamma_transition_cells(spot, variance, &strikes).expect("cells");
        let expected = variance.exp()
            * (gamma[0] * cells[0].probability()
                + ((gamma[0] - 0.05 * strikes[0]) * cells[1].probability()
                    + 0.05 * cells[1].first_moment())
                + gamma[1] * cells[2].probability());

        assert_close(value, expected);
    }

    #[test]
    fn transition_operator_rejects_gamma_length_mismatch() {
        assert!(matches!(
            integrate_piecewise_linear_local_gamma_transition(100.0, 0.04, &[90.0, 110.0], &[1.0]),
            Err(RiskConfigError::VegaKtGammaLengthMismatch {
                strike_count: 2,
                gamma_count: 1
            })
        ));
    }

    #[test]
    fn reporting_iv_basis_interpolates_bilinearly_and_projects_transpose_weights() {
        let basis = ReportingIvBasis::new(
            vec![1.0, 2.0],
            vec![-0.2, 0.0, 0.3],
            vec![0.2, 0.22, 0.25, 0.3, 0.32, 0.35],
        )
        .expect("basis");
        let interpolation = basis.interpolate(1.5, 0.15).expect("interpolation");

        assert_close(interpolation.value(), 0.285);
        assert_eq!(interpolation.lower_time_index(), 0);
        assert_eq!(interpolation.lower_log_moneyness_index(), 1);
        assert_close(interpolation.time_weight(), 0.5);
        assert_close(interpolation.log_moneyness_weight(), 0.5);
        assert_eq!(interpolation.boundary(), ReportingIvBoundary::InRange);

        let mut buckets = vec![0.0; basis.bucket_count()];
        let mut stats = ReportingIvProjectionStats::default();
        basis
            .project_weight(1.5, 0.15, 4.0, &mut buckets, &mut stats)
            .expect("projection");

        assert_eq!(buckets, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(stats.total_edge_count(), 0);
    }

    #[test]
    fn reporting_iv_basis_samples_implied_volatility_surface_row_major() {
        let surface = ConstantVolSurface { volatility: 0.25 };
        let basis = ReportingIvBasis::from_surface(&surface, vec![1.0, 2.0], vec![-0.2, 0.0, 0.3])
            .expect("basis");

        assert_eq!(basis.implied_volatilities(), &[0.25; 6]);
        assert_close(
            basis.interpolate(1.5, 0.1).expect("interpolation").value(),
            0.25,
        );
    }

    #[test]
    fn reporting_iv_basis_propagates_surface_sampling_errors() {
        let error =
            ReportingIvBasis::from_surface(&FailingSurface, vec![1.0, 2.0], vec![-0.2, 0.0, 0.3])
                .expect_err("surface error");

        assert!(matches!(error, RiskConfigError::Market(_)));
    }

    #[test]
    fn reporting_iv_basis_assigns_out_of_range_log_moneyness_to_edges() {
        let basis = ReportingIvBasis::new(
            vec![1.0, 2.0],
            vec![-0.2, 0.0, 0.3],
            vec![0.2, 0.22, 0.25, 0.3, 0.32, 0.35],
        )
        .expect("basis");
        let mut buckets = vec![0.0; basis.bucket_count()];
        let mut stats = ReportingIvProjectionStats::default();

        basis
            .project_weight(1.25, -0.4, 2.0, &mut buckets, &mut stats)
            .expect("left edge");
        basis
            .project_weight(1.75, 0.5, -3.0, &mut buckets, &mut stats)
            .expect("right edge");

        assert_eq!(buckets, vec![1.5, 0.0, -0.75, 0.5, 0.0, -2.25]);
        assert_eq!(stats.left_edge_count(), 1);
        assert_eq!(stats.right_edge_count(), 1);
        assert_close(stats.left_edge_sensitivity(), 2.0);
        assert_close(stats.right_edge_sensitivity(), -3.0);
    }

    #[test]
    fn reporting_iv_basis_rejects_invalid_shapes_and_time_queries() {
        assert!(matches!(
            ReportingIvBasis::new(vec![1.0], vec![-0.1, 0.1], vec![0.2, 0.3]),
            Err(RiskConfigError::TooFewReportingIvMaturities { count: 1 })
        ));
        assert!(matches!(
            ReportingIvBasis::new(vec![1.0, 2.0], vec![0.0, 0.0], vec![0.2, 0.3, 0.4, 0.5]),
            Err(RiskConfigError::UnsortedReportingIvStrikes { left_index: 0 })
        ));
        assert!(matches!(
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2]),
            Err(RiskConfigError::ReportingIvValueLengthMismatch {
                expected: 4,
                actual: 1
            })
        ));
        assert!(matches!(
            reporting_iv_value_count(usize::MAX, 2),
            Err(RiskConfigError::ReportingIvValueLengthMismatch {
                expected: usize::MAX,
                actual: 0
            })
        ));
        let basis =
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2, 0.3, 0.4, 0.5])
                .expect("basis");
        assert!(matches!(
            basis.interpolate(0.5, 0.0),
            Err(RiskConfigError::ReportingIvQueryTimeOutOfRange { .. })
        ));
    }

    #[test]
    fn projection_keeps_outside_active_domain_in_residual() {
        let basis = ReportingIvBasis::new(
            vec![1.0, 2.0],
            vec![-0.2, 0.0, 0.2],
            vec![0.2, 0.22, 0.24, 0.3, 0.32, 0.34],
        )
        .expect("basis");
        let domain = active_density_domain(&[-0.3, -0.1, 0.1, 0.3], &[0.1, 1.0, 1.0, 0.1], 0.5)
            .expect("domain");
        let projection = project_local_vega_nodes_to_reporting_iv(
            &basis,
            1.5,
            &[-0.3, -0.1, 0.1, 0.3],
            &[1.0, 2.0, 3.0, 4.0],
            domain,
        )
        .expect("projection");

        assert_eq!(
            projection.raw_buckets(),
            &[0.5, 1.25, 0.75, 0.5, 1.25, 0.75]
        );
        assert_close(projection.scalar_vega(), 5.0);
        assert_close(projection.signed_residual(), 5.0);
        assert_close(projection.pre_projection(), 10.0);
        assert!(projection.reconciles());
        assert_eq!(projection.reporting_stats().total_edge_count(), 0);
    }

    #[test]
    fn projection_records_reporting_edge_assignments_for_active_nodes() {
        let basis = ReportingIvBasis::new(
            vec![1.0, 2.0],
            vec![-0.2, 0.0, 0.2],
            vec![0.2, 0.22, 0.24, 0.3, 0.32, 0.34],
        )
        .expect("basis");
        let domain =
            active_density_domain(&[-0.3, 0.0, 0.3], &[1.0, 1.0, 1.0], 0.5).expect("domain");
        let projection = project_local_vega_nodes_to_reporting_iv(
            &basis,
            1.5,
            &[-0.3, 0.0, 0.3],
            &[1.0, 2.0, 3.0],
            domain,
        )
        .expect("projection");

        assert_eq!(projection.reporting_stats().left_edge_count(), 1);
        assert_eq!(projection.reporting_stats().right_edge_count(), 1);
        assert_close(projection.reporting_stats().left_edge_sensitivity(), 1.0);
        assert_close(projection.reporting_stats().right_edge_sensitivity(), 3.0);
        assert!(projection.reconciles());
    }

    #[test]
    fn projection_exposes_market_scaled_buckets() {
        let projection = VegaKtProjection {
            raw_buckets: vec![100.0, -50.0].into_boxed_slice(),
            scalar_vega: 50.0,
            signed_residual: 0.0,
            pre_projection: 50.0,
            reporting_stats: ReportingIvProjectionStats::default(),
        };

        assert_eq!(projection.market_scaled_buckets(), vec![1.0, -0.5]);
    }

    #[test]
    fn report_preserves_coordinates_units_projection_and_optional_covariance() {
        let basis =
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2, 0.21, 0.3, 0.31])
                .expect("basis");
        let projection = VegaKtProjection {
            raw_buckets: vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(),
            scalar_vega: 10.0,
            signed_residual: -1.0,
            pre_projection: 9.0,
            reporting_stats: ReportingIvProjectionStats::default(),
        };
        let estimates =
            vega_kt_bucket_estimates(&[10.0, 11.0], &[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], 4)
                .expect("estimates");
        let covariance =
            vega_kt_full_bucket_covariance(&[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], 4)
                .expect("covariance");
        let density_row = test_density_row();
        let report = vega_kt_report(
            &basis,
            &density_row,
            projection,
            estimates,
            Some(covariance),
        )
        .expect("report");

        assert_eq!(report.coordinates().len(), 4);
        assert_close(report.coordinates()[0].maturity(), 1.0);
        assert_close(report.coordinates()[0].log_moneyness(), -0.1);
        assert_close(report.coordinates()[3].maturity(), 2.0);
        assert_close(report.coordinates()[3].implied_volatility(), 0.31);
        assert_eq!(
            report.covariance_layout(),
            VegaKtCovarianceLayout::FullBucketMatrixRowMajor
        );
        assert_eq!(
            report.full_bucket_covariance().expect("covariance").len(),
            16
        );
        assert_eq!(
            report.raw_unit(),
            VegaKtBucketUnit::CurrencyPerUnitAbsoluteVolatility
        );
        assert_eq!(
            report.market_scaled_unit(),
            VegaKtBucketUnit::CurrencyPerVolatilityPoint
        );
        assert_eq!(report.policy_label(), EQUATION_11_FIRST_ORDER_LABEL);
        assert_eq!(report.truncation_order(), EQUATION_11_TRUNCATION_ORDER);
        assert_close(report.projection().scalar_vega(), 10.0);
        assert_eq!(report.estimates().len(), 4);
        assert_eq!(
            report.residual_diagnostics().active_domain(),
            density_row.active_domain()
        );
        assert_close(
            report.residual_diagnostics().excluded_probability_mass(),
            density_row.excluded_probability_mass(),
        );
        assert_close(report.residual_diagnostics().signed_residual(), -1.0);
        assert_close(report.residual_diagnostics().pre_projection(), 9.0);
    }

    #[test]
    fn report_omits_full_covariance_unless_requested_and_validates_shapes() {
        let basis =
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2, 0.21, 0.3, 0.31])
                .expect("basis");
        let projection = VegaKtProjection {
            raw_buckets: vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(),
            scalar_vega: 10.0,
            signed_residual: 0.0,
            pre_projection: 10.0,
            reporting_stats: ReportingIvProjectionStats::default(),
        };
        let estimates = vec![
            VegaKtBucketEstimate {
                raw_mean: 1.0,
                market_scaled_mean: 0.01,
                sample_variance: None,
                price_covariance: None,
            };
            4
        ];

        let density_row = test_density_row();
        let report = vega_kt_report(
            &basis,
            &density_row,
            projection.clone(),
            estimates.clone(),
            None,
        )
        .expect("report");
        assert_eq!(
            report.covariance_layout(),
            VegaKtCovarianceLayout::PriceAndBucketVarianceOnly
        );
        assert!(report.full_bucket_covariance().is_none());

        assert!(matches!(
            vega_kt_report(
                &basis,
                &density_row,
                projection.clone(),
                estimates[..3].to_vec(),
                None
            ),
            Err(RiskConfigError::VegaKtBucketEstimateLengthMismatch {
                expected: 4,
                actual: 3
            })
        ));
        assert!(matches!(
            vega_kt_report(
                &basis,
                &density_row,
                projection,
                estimates,
                Some(vec![None; 15])
            ),
            Err(RiskConfigError::VegaKtFullCovarianceLengthMismatch {
                expected: 16,
                actual: 15
            })
        ));
    }

    #[test]
    fn report_rejects_projection_that_does_not_reconcile() {
        let basis =
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2, 0.21, 0.3, 0.31])
                .expect("basis");
        let projection = VegaKtProjection {
            raw_buckets: vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(),
            scalar_vega: 10.0,
            signed_residual: 0.0,
            pre_projection: 9.0,
            reporting_stats: ReportingIvProjectionStats::default(),
        };
        let estimates = vec![
            VegaKtBucketEstimate {
                raw_mean: 1.0,
                market_scaled_mean: 0.01,
                sample_variance: None,
                price_covariance: None,
            };
            4
        ];
        let density_row = test_density_row();

        assert!(!projection.reconciles());
        assert_close(projection.reconstructed_pre_projection(), 10.0);
        assert!(matches!(
            vega_kt_report(&basis, &density_row, projection, estimates, None),
            Err(RiskConfigError::VegaKtProjectionReconciliationFailure { .. })
        ));
    }

    #[test]
    fn report_from_samples_builds_estimates_and_respects_covariance_request() {
        let basis =
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2, 0.21, 0.3, 0.31])
                .expect("basis");
        let projection = VegaKtProjection {
            raw_buckets: vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(),
            scalar_vega: 10.0,
            signed_residual: 0.0,
            pre_projection: 10.0,
            reporting_stats: ReportingIvProjectionStats::default(),
        };
        let density_row = test_density_row();
        let price_samples = [10.0, 11.0, 13.0];
        let bucket_samples = [
            1.0, 2.0, 3.0, 4.0, //
            2.0, 3.0, 4.0, 5.0, //
            4.0, 5.0, 6.0, 7.0,
        ];

        let compact_report = vega_kt_report_from_samples(
            &basis,
            &density_row,
            projection.clone(),
            &price_samples,
            &bucket_samples,
            false,
        )
        .expect("compact report");
        assert_eq!(
            compact_report.covariance_layout(),
            VegaKtCovarianceLayout::PriceAndBucketVarianceOnly
        );
        assert!(compact_report.full_bucket_covariance().is_none());
        assert_eq!(compact_report.estimates().len(), 4);
        assert_close(compact_report.estimates()[0].raw_mean(), 7.0 / 3.0);

        let full_report = vega_kt_report_from_samples(
            &basis,
            &density_row,
            projection,
            &price_samples,
            &bucket_samples,
            true,
        )
        .expect("full report");
        assert_eq!(
            full_report.covariance_layout(),
            VegaKtCovarianceLayout::FullBucketMatrixRowMajor
        );
        assert_eq!(
            full_report
                .full_bucket_covariance()
                .expect("covariance")
                .len(),
            16
        );
    }

    #[test]
    fn report_from_samples_rejects_incomplete_ordered_sample_rows() {
        let basis =
            ReportingIvBasis::new(vec![1.0, 2.0], vec![-0.1, 0.1], vec![0.2, 0.21, 0.3, 0.31])
                .expect("basis");
        let projection = VegaKtProjection {
            raw_buckets: vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(),
            scalar_vega: 10.0,
            signed_residual: 0.0,
            pre_projection: 10.0,
            reporting_stats: ReportingIvProjectionStats::default(),
        };
        let density_row = test_density_row();

        assert!(matches!(
            vega_kt_report_from_samples(
                &basis,
                &density_row,
                projection,
                &[10.0, 11.0],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                true,
            ),
            Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
                expected_multiple: 4,
                actual: 5
            })
        ));
    }

    #[test]
    fn bucket_estimates_report_raw_market_scaled_variance_and_price_covariance() {
        let estimates =
            vega_kt_bucket_estimates(&[10.0, 14.0, 16.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)
                .expect("estimates");

        assert_eq!(estimates.len(), 2);
        assert_close(estimates[0].raw_mean(), 3.0);
        assert_close(estimates[0].market_scaled_mean(), 0.03);
        assert_close(estimates[0].sample_variance().expect("variance"), 4.0);
        assert_close(estimates[0].price_covariance().expect("covariance"), 6.0);
        assert_eq!(
            estimates[0].raw_unit(),
            VegaKtBucketUnit::CurrencyPerUnitAbsoluteVolatility
        );
        assert_eq!(
            estimates[0].market_scaled_unit(),
            VegaKtBucketUnit::CurrencyPerVolatilityPoint
        );
    }

    #[test]
    fn bucket_estimates_reject_empty_or_mismatched_samples() {
        assert!(matches!(
            vega_kt_bucket_estimates(&[], &[], 1),
            Err(RiskConfigError::EmptyVegaKtBucketSamples)
        ));
        assert!(matches!(
            vega_kt_bucket_estimates(&[1.0, 2.0], &[1.0, 2.0, 3.0], 2),
            Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
                expected_multiple: 2,
                actual: 3
            })
        ));
        assert!(matches!(
            vega_kt_bucket_estimates(&[1.0, 2.0], &[], usize::MAX),
            Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
                expected_multiple: usize::MAX,
                actual: 0
            })
        ));
    }

    #[test]
    fn full_bucket_covariance_is_row_major_and_symmetric() {
        let covariance =
            vega_kt_full_bucket_covariance(&[1.0, 2.0, 3.0, 4.0, 5.0, 8.0], 2).expect("covariance");

        assert_eq!(covariance.len(), 4);
        assert_close(covariance[0].expect("variance 0"), 4.0);
        assert_close(covariance[1].expect("cov 0 1"), 6.0);
        assert_close(covariance[2].expect("cov 1 0"), 6.0);
        assert_close(covariance[3].expect("variance 1"), 9.333_333_333_333_334);
    }

    #[test]
    fn full_bucket_covariance_requires_ordered_complete_rows() {
        assert!(matches!(
            vega_kt_full_bucket_covariance(&[], 2),
            Err(RiskConfigError::EmptyVegaKtBucketSamples)
        ));
        assert!(matches!(
            vega_kt_full_bucket_covariance(&[1.0, 2.0, 3.0], 2),
            Err(RiskConfigError::VegaKtBucketSampleLengthMismatch {
                expected_multiple: 2,
                actual: 3
            })
        ));
    }
}
