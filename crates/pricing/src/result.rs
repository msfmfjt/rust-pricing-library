use std::num::NonZeroU64;

use pricing_core::{FiniteF64, NonNegativeF64, SchemaVersion};

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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RiskReport {
    pub delta: Option<RiskEstimate>,
    pub gamma: Option<RiskEstimate>,
    pub vega: Option<RiskEstimate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PricingWarning {
    code: &'static str,
    message: String,
}

impl PricingWarning {
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
        assert!(Estimate::new(10.0, 0.5, 11.0, 12.0, EstimatorKind::PseudoMonteCarlo, 100).is_err());
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
