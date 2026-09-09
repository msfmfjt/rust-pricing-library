use std::num::NonZeroU64;

use pricing_core::{FiniteF64, NonNegativeF64, PositiveF64, SchemaVersion};
use pricing_risk::{VegaKtBucketUnit, VegaKtReport};

use crate::ResultBuildError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EstimatorKind {
    Analytical,
    PseudoMonteCarlo,
    RandomizedQuasiMonteCarlo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfidenceInterval {
    lower: FiniteF64,
    upper: FiniteF64,
}

impl ConfidenceInterval {
    #[must_use]
    pub const fn lower(self) -> FiniteF64 {
        self.lower
    }

    #[must_use]
    pub const fn upper(self) -> FiniteF64 {
        self.upper
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Estimate {
    value: FiniteF64,
    standard_error: NonNegativeF64,
    confidence_interval: ConfidenceInterval,
    estimator: EstimatorKind,
    effective_sampling_units: NonZeroU64,
}

impl Estimate {
    pub fn new(
        value: f64,
        standard_error: f64,
        confidence_lower: f64,
        confidence_upper: f64,
        estimator: EstimatorKind,
        effective_sampling_units: u64,
    ) -> Result<Self, ResultBuildError> {
        let value = FiniteF64::new(value, "estimate_value")?;
        let standard_error = NonNegativeF64::new(standard_error, "standard_error")?;
        let lower = FiniteF64::new(confidence_lower, "confidence_lower")?;
        let upper = FiniteF64::new(confidence_upper, "confidence_upper")?;
        if lower.get() > value.get() || value.get() > upper.get() {
            return Err(ResultBuildError::InvalidConfidenceInterval {
                lower_bits: lower.to_bits(),
                value_bits: value.to_bits(),
                upper_bits: upper.to_bits(),
            });
        }
        Ok(Self {
            value,
            standard_error,
            confidence_interval: ConfidenceInterval { lower, upper },
            estimator,
            effective_sampling_units: NonZeroU64::new(effective_sampling_units)
                .ok_or(ResultBuildError::ZeroEffectiveSamplingUnits)?,
        })
    }

    #[must_use]
    pub const fn value(self) -> FiniteF64 {
        self.value
    }

    #[must_use]
    pub const fn standard_error(self) -> NonNegativeF64 {
        self.standard_error
    }

    #[must_use]
    pub const fn confidence_interval(self) -> ConfidenceInterval {
        self.confidence_interval
    }

    #[must_use]
    pub const fn estimator(self) -> EstimatorKind {
        self.estimator
    }

    #[must_use]
    pub const fn effective_sampling_units(self) -> NonZeroU64 {
        self.effective_sampling_units
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RiskUnit {
    DeltaRaw,
    DeltaOnePercentSpot,
    GammaRaw,
    GammaOnePercentSpotSquared,
    VegaRaw,
    VegaOneVolPoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VegaKtResultUnit {
    CurrencyPerUnitAbsoluteVolatility,
    CurrencyPerVolatilityPoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VegaKtResultCovarianceLayout {
    PriceAndBucketVarianceOnly,
    FullBucketMatrixRowMajor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VegaKtResultCoordinate {
    maturity: PositiveF64,
    log_moneyness: FiniteF64,
    implied_volatility: PositiveF64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VegaKtResultBucketEstimate {
    raw_mean: FiniteF64,
    market_scaled_mean: FiniteF64,
    sample_variance: Option<NonNegativeF64>,
    price_covariance: Option<FiniteF64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VegaKtResultReportingStats {
    left_edge_count: u64,
    right_edge_count: u64,
    left_edge_sensitivity: FiniteF64,
    right_edge_sensitivity: FiniteF64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtResultProjection {
    scalar_vega: FiniteF64,
    signed_residual: FiniteF64,
    pre_projection: FiniteF64,
    reporting_stats: VegaKtResultReportingStats,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VegaKtResultResidualDiagnostics {
    active_domain_start_index: usize,
    active_domain_end_index: usize,
    active_domain_forward_index: usize,
    excluded_probability_mass: NonNegativeF64,
    signed_residual: FiniteF64,
    pre_projection: FiniteF64,
    reporting_stats: VegaKtResultReportingStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtResult {
    coordinates: Box<[VegaKtResultCoordinate]>,
    estimates: Box<[VegaKtResultBucketEstimate]>,
    raw_buckets: Box<[FiniteF64]>,
    full_bucket_covariance: Option<Box<[Option<FiniteF64>]>>,
    covariance_layout: VegaKtResultCovarianceLayout,
    projection: VegaKtResultProjection,
    residual_diagnostics: VegaKtResultResidualDiagnostics,
    raw_unit: VegaKtResultUnit,
    market_scaled_unit: VegaKtResultUnit,
    policy_label: String,
    truncation_order: String,
}

impl VegaKtResultCoordinate {
    pub fn new(
        maturity: f64,
        log_moneyness: f64,
        implied_volatility: f64,
    ) -> Result<Self, ResultBuildError> {
        Ok(Self {
            maturity: PositiveF64::new(maturity, "vega_kt_coordinate_maturity")?,
            log_moneyness: FiniteF64::new(log_moneyness, "vega_kt_coordinate_log_moneyness")?,
            implied_volatility: PositiveF64::new(
                implied_volatility,
                "vega_kt_coordinate_implied_volatility",
            )?,
        })
    }

    #[must_use]
    pub const fn maturity(self) -> PositiveF64 {
        self.maturity
    }

    #[must_use]
    pub const fn log_moneyness(self) -> FiniteF64 {
        self.log_moneyness
    }

    #[must_use]
    pub const fn implied_volatility(self) -> PositiveF64 {
        self.implied_volatility
    }
}

impl VegaKtResultBucketEstimate {
    pub fn new(
        raw_mean: f64,
        market_scaled_mean: f64,
        sample_variance: Option<f64>,
        price_covariance: Option<f64>,
    ) -> Result<Self, ResultBuildError> {
        Ok(Self {
            raw_mean: FiniteF64::new(raw_mean, "vega_kt_raw_mean")?,
            market_scaled_mean: FiniteF64::new(market_scaled_mean, "vega_kt_market_scaled_mean")?,
            sample_variance: sample_variance
                .map(|value| NonNegativeF64::new(value, "vega_kt_sample_variance"))
                .transpose()?,
            price_covariance: price_covariance
                .map(|value| FiniteF64::new(value, "vega_kt_price_covariance"))
                .transpose()?,
        })
    }

    #[must_use]
    pub const fn raw_mean(self) -> FiniteF64 {
        self.raw_mean
    }

    #[must_use]
    pub const fn market_scaled_mean(self) -> FiniteF64 {
        self.market_scaled_mean
    }

    #[must_use]
    pub const fn sample_variance(self) -> Option<NonNegativeF64> {
        self.sample_variance
    }

    #[must_use]
    pub const fn price_covariance(self) -> Option<FiniteF64> {
        self.price_covariance
    }
}

impl VegaKtResultReportingStats {
    pub fn new(
        left_edge_count: u64,
        right_edge_count: u64,
        left_edge_sensitivity: f64,
        right_edge_sensitivity: f64,
    ) -> Result<Self, ResultBuildError> {
        Ok(Self {
            left_edge_count,
            right_edge_count,
            left_edge_sensitivity: FiniteF64::new(
                left_edge_sensitivity,
                "vega_kt_left_edge_sensitivity",
            )?,
            right_edge_sensitivity: FiniteF64::new(
                right_edge_sensitivity,
                "vega_kt_right_edge_sensitivity",
            )?,
        })
    }

    #[must_use]
    pub const fn left_edge_count(self) -> u64 {
        self.left_edge_count
    }

    #[must_use]
    pub const fn right_edge_count(self) -> u64 {
        self.right_edge_count
    }

    #[must_use]
    pub const fn left_edge_sensitivity(self) -> FiniteF64 {
        self.left_edge_sensitivity
    }

    #[must_use]
    pub const fn right_edge_sensitivity(self) -> FiniteF64 {
        self.right_edge_sensitivity
    }
}

impl VegaKtResultProjection {
    pub fn new(
        scalar_vega: f64,
        signed_residual: f64,
        pre_projection: f64,
        reporting_stats: VegaKtResultReportingStats,
    ) -> Result<Self, ResultBuildError> {
        Ok(Self {
            scalar_vega: FiniteF64::new(scalar_vega, "vega_kt_scalar_vega")?,
            signed_residual: FiniteF64::new(signed_residual, "vega_kt_signed_residual")?,
            pre_projection: FiniteF64::new(pre_projection, "vega_kt_pre_projection")?,
            reporting_stats,
        })
    }

    #[must_use]
    pub const fn scalar_vega(self) -> FiniteF64 {
        self.scalar_vega
    }

    #[must_use]
    pub const fn signed_residual(self) -> FiniteF64 {
        self.signed_residual
    }

    #[must_use]
    pub const fn pre_projection(self) -> FiniteF64 {
        self.pre_projection
    }

    #[must_use]
    pub const fn reporting_stats(self) -> VegaKtResultReportingStats {
        self.reporting_stats
    }
}

impl VegaKtResultResidualDiagnostics {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        active_domain_start_index: usize,
        active_domain_end_index: usize,
        active_domain_forward_index: usize,
        excluded_probability_mass: f64,
        signed_residual: f64,
        pre_projection: f64,
        reporting_stats: VegaKtResultReportingStats,
    ) -> Result<Self, ResultBuildError> {
        Ok(Self {
            active_domain_start_index,
            active_domain_end_index,
            active_domain_forward_index,
            excluded_probability_mass: NonNegativeF64::new(
                excluded_probability_mass,
                "vega_kt_excluded_probability_mass",
            )?,
            signed_residual: FiniteF64::new(signed_residual, "vega_kt_signed_residual")?,
            pre_projection: FiniteF64::new(pre_projection, "vega_kt_pre_projection")?,
            reporting_stats,
        })
    }

    #[must_use]
    pub const fn active_domain_start_index(self) -> usize {
        self.active_domain_start_index
    }

    #[must_use]
    pub const fn active_domain_end_index(self) -> usize {
        self.active_domain_end_index
    }

    #[must_use]
    pub const fn active_domain_forward_index(self) -> usize {
        self.active_domain_forward_index
    }

    #[must_use]
    pub const fn excluded_probability_mass(self) -> NonNegativeF64 {
        self.excluded_probability_mass
    }

    #[must_use]
    pub const fn signed_residual(self) -> FiniteF64 {
        self.signed_residual
    }

    #[must_use]
    pub const fn pre_projection(self) -> FiniteF64 {
        self.pre_projection
    }

    #[must_use]
    pub const fn reporting_stats(self) -> VegaKtResultReportingStats {
        self.reporting_stats
    }
}

impl VegaKtResult {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coordinates: Vec<VegaKtResultCoordinate>,
        estimates: Vec<VegaKtResultBucketEstimate>,
        raw_buckets: Vec<f64>,
        full_bucket_covariance: Option<Vec<Option<f64>>>,
        covariance_layout: VegaKtResultCovarianceLayout,
        projection: VegaKtResultProjection,
        residual_diagnostics: VegaKtResultResidualDiagnostics,
        raw_unit: VegaKtResultUnit,
        market_scaled_unit: VegaKtResultUnit,
        policy_label: impl Into<String>,
        truncation_order: impl Into<String>,
    ) -> Result<Self, ResultBuildError> {
        let bucket_count = coordinates.len();
        if estimates.len() != bucket_count || raw_buckets.len() != bucket_count {
            return Err(ResultBuildError::VegaKtLengthMismatch {
                coordinates: bucket_count,
                estimates: estimates.len(),
                raw_buckets: raw_buckets.len(),
            });
        }
        let full_bucket_covariance = match full_bucket_covariance {
            Some(values) => {
                let expected = bucket_count
                    .checked_mul(bucket_count)
                    .expect("bucket matrix size fits usize");
                if values.len() != expected {
                    return Err(ResultBuildError::VegaKtFullCovarianceLengthMismatch {
                        expected,
                        actual: values.len(),
                    });
                }
                Some(
                    values
                        .into_iter()
                        .map(|value| {
                            value
                                .map(|value| FiniteF64::new(value, "vega_kt_full_covariance"))
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                )
            }
            None => None,
        };
        Ok(Self {
            coordinates: coordinates.into_boxed_slice(),
            estimates: estimates.into_boxed_slice(),
            raw_buckets: raw_buckets
                .into_iter()
                .map(|value| FiniteF64::new(value, "vega_kt_raw_bucket"))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
            full_bucket_covariance,
            covariance_layout,
            projection,
            residual_diagnostics,
            raw_unit,
            market_scaled_unit,
            policy_label: policy_label.into(),
            truncation_order: truncation_order.into(),
        })
    }

    #[must_use]
    pub fn coordinates(&self) -> &[VegaKtResultCoordinate] {
        &self.coordinates
    }

    #[must_use]
    pub fn estimates(&self) -> &[VegaKtResultBucketEstimate] {
        &self.estimates
    }

    #[must_use]
    pub fn raw_buckets(&self) -> &[FiniteF64] {
        &self.raw_buckets
    }

    #[must_use]
    pub fn full_bucket_covariance(&self) -> Option<&[Option<FiniteF64>]> {
        self.full_bucket_covariance.as_deref()
    }

    #[must_use]
    pub const fn covariance_layout(&self) -> VegaKtResultCovarianceLayout {
        self.covariance_layout
    }

    #[must_use]
    pub const fn projection(&self) -> VegaKtResultProjection {
        self.projection
    }

    #[must_use]
    pub const fn residual_diagnostics(&self) -> VegaKtResultResidualDiagnostics {
        self.residual_diagnostics
    }

    #[must_use]
    pub const fn raw_unit(&self) -> VegaKtResultUnit {
        self.raw_unit
    }

    #[must_use]
    pub const fn market_scaled_unit(&self) -> VegaKtResultUnit {
        self.market_scaled_unit
    }

    #[must_use]
    pub fn policy_label(&self) -> &str {
        &self.policy_label
    }

    #[must_use]
    pub fn truncation_order(&self) -> &str {
        &self.truncation_order
    }
}

impl From<VegaKtBucketUnit> for VegaKtResultUnit {
    fn from(value: VegaKtBucketUnit) -> Self {
        match value {
            VegaKtBucketUnit::CurrencyPerUnitAbsoluteVolatility => {
                Self::CurrencyPerUnitAbsoluteVolatility
            }
            VegaKtBucketUnit::CurrencyPerVolatilityPoint => Self::CurrencyPerVolatilityPoint,
        }
    }
}

impl TryFrom<&VegaKtReport> for VegaKtResult {
    type Error = ResultBuildError;

    fn try_from(value: &VegaKtReport) -> Result<Self, Self::Error> {
        let residual = value.residual_diagnostics();
        let active_domain = residual.active_domain();
        Self::new(
            value
                .coordinates()
                .iter()
                .map(|coordinate| {
                    VegaKtResultCoordinate::new(
                        coordinate.maturity(),
                        coordinate.log_moneyness(),
                        coordinate.implied_volatility(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            value
                .estimates()
                .iter()
                .map(|estimate| {
                    VegaKtResultBucketEstimate::new(
                        estimate.raw_mean(),
                        estimate.market_scaled_mean(),
                        estimate.sample_variance(),
                        estimate.price_covariance(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            value.projection().raw_buckets().to_vec(),
            value.full_bucket_covariance().map(<[_]>::to_vec),
            match value.covariance_layout() {
                pricing_risk::VegaKtCovarianceLayout::PriceAndBucketVarianceOnly => {
                    VegaKtResultCovarianceLayout::PriceAndBucketVarianceOnly
                }
                pricing_risk::VegaKtCovarianceLayout::FullBucketMatrixRowMajor => {
                    VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor
                }
            },
            VegaKtResultProjection::new(
                value.projection().scalar_vega(),
                value.projection().signed_residual(),
                value.projection().pre_projection(),
                VegaKtResultReportingStats::new(
                    value.projection().reporting_stats().left_edge_count(),
                    value.projection().reporting_stats().right_edge_count(),
                    value.projection().reporting_stats().left_edge_sensitivity(),
                    value
                        .projection()
                        .reporting_stats()
                        .right_edge_sensitivity(),
                )?,
            )?,
            VegaKtResultResidualDiagnostics::new(
                active_domain.start_index(),
                active_domain.end_index(),
                active_domain.forward_index(),
                residual.excluded_probability_mass(),
                residual.signed_residual(),
                residual.pre_projection(),
                VegaKtResultReportingStats::new(
                    residual.reporting_stats().left_edge_count(),
                    residual.reporting_stats().right_edge_count(),
                    residual.reporting_stats().left_edge_sensitivity(),
                    residual.reporting_stats().right_edge_sensitivity(),
                )?,
            )?,
            value.raw_unit().into(),
            value.market_scaled_unit().into(),
            value.policy_label(),
            value.truncation_order(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RiskEstimate {
    raw: Estimate,
    market_scaled: Estimate,
    raw_unit: RiskUnit,
    market_scaled_unit: RiskUnit,
}

impl RiskEstimate {
    #[must_use]
    pub const fn new(
        raw: Estimate,
        market_scaled: Estimate,
        raw_unit: RiskUnit,
        market_scaled_unit: RiskUnit,
    ) -> Self {
        Self {
            raw,
            market_scaled,
            raw_unit,
            market_scaled_unit,
        }
    }

    #[must_use]
    pub const fn raw(self) -> Estimate {
        self.raw
    }

    #[must_use]
    pub const fn market_scaled(self) -> Estimate {
        self.market_scaled
    }

    #[must_use]
    pub const fn raw_unit(self) -> RiskUnit {
        self.raw_unit
    }

    #[must_use]
    pub const fn market_scaled_unit(self) -> RiskUnit {
        self.market_scaled_unit
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RiskReport {
    pub delta: Option<RiskEstimate>,
    pub gamma: Option<RiskEstimate>,
    pub vega: Option<RiskEstimate>,
    pub vega_kt: Option<VegaKtResult>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PricingWarning {
    code: String,
    message: String,
}

impl PricingWarning {
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Diagnostics {
    warnings: Box<[PricingWarning]>,
}

impl Diagnostics {
    #[must_use]
    pub fn new(warnings: Vec<PricingWarning>) -> Self {
        Self {
            warnings: warnings.into_boxed_slice(),
        }
    }

    #[must_use]
    pub fn warnings(&self) -> &[PricingWarning] {
        &self.warnings
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayMetadata {
    schema_version: SchemaVersion,
    request_fingerprint: [u8; 32],
    library_version: String,
    platform: String,
}

impl ReplayMetadata {
    #[must_use]
    pub fn new(
        schema_version: SchemaVersion,
        request_fingerprint: [u8; 32],
        library_version: impl Into<String>,
        platform: impl Into<String>,
    ) -> Self {
        Self {
            schema_version,
            request_fingerprint,
            library_version: library_version.into(),
            platform: platform.into(),
        }
    }

    #[must_use]
    pub const fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }

    #[must_use]
    pub const fn request_fingerprint(&self) -> &[u8; 32] {
        &self.request_fingerprint
    }

    #[must_use]
    pub fn library_version(&self) -> &str {
        &self.library_version
    }

    #[must_use]
    pub fn platform(&self) -> &str {
        &self.platform
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PricingResult {
    pub value: Estimate,
    pub risks: RiskReport,
    pub diagnostics: Diagnostics,
    pub replay: ReplayMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_requires_finite_ordered_interval_and_samples() {
        let estimate = Estimate::new(10.0, 0.5, 9.0, 11.0, EstimatorKind::PseudoMonteCarlo, 100)
            .expect("valid estimate");
        assert_eq!(estimate.value().get(), 10.0);
        assert_eq!(estimate.effective_sampling_units().get(), 100);
        assert!(
            Estimate::new(10.0, 0.5, 11.0, 12.0, EstimatorKind::PseudoMonteCarlo, 100).is_err()
        );
        assert!(Estimate::new(10.0, 0.5, 9.0, 11.0, EstimatorKind::PseudoMonteCarlo, 0).is_err());
    }

    #[test]
    fn diagnostics_preserve_warning_order() {
        let diagnostics = Diagnostics::new(vec![
            PricingWarning::new("first", "first warning"),
            PricingWarning::new("second", "second warning"),
        ]);
        assert_eq!(diagnostics.warnings()[0].code(), "first");
        assert_eq!(diagnostics.warnings()[1].code(), "second");
    }
}
