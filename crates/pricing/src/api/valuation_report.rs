//! Common estimates with explicit units, differentiation and uncertainty scope.
//! Native diagnostics remain available on `PricingTaskResult::native_result`.
use crate::{EstimatorKind, Fingerprint, VegaKtResult, WireError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UncertaintyScope {
    PricingSampling,
    ConditionalOnCalibration,
    ConditionalOnExercisePolicy,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationTreatment {
    NotApplicable,
    Recalibrated,
    FixedCalibration,
    FixedExercisePolicy,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SensitivityUnit {
    CurrencyPerSpot,
    CurrencyPerSpotSquared,
    CurrencyPerUnitVolatility,
    CurrencyPerVariance,
    CurrencyPerLogDiscountFactor,
    CurrencyPerLogDensity,
    CurrencyPerUnitParameter,
    CurrencyPerOnePercentSpot,
    CurrencyPerOnePercentSpotSquared,
    CurrencyPerVolatilityPoint,
    CurrencyPerBasisPoint,
}

/// Differentiation coordinate supplied by the native risk producer.
/// Display labels are deliberately independent of this classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RiskParameterKind {
    Spot,
    Volatility,
    LocalVariance,
    LogDiscountFactor,
    ForwardLogDensity,
    CashMean,
    /// A unit change in the named native parameter (not market-IV Vega).
    ModelParameter,
}
impl RiskParameterKind {
    #[must_use]
    pub const fn unit(self) -> SensitivityUnit {
        match self {
            Self::Spot => SensitivityUnit::CurrencyPerSpot,
            Self::Volatility => SensitivityUnit::CurrencyPerUnitVolatility,
            Self::LocalVariance => SensitivityUnit::CurrencyPerVariance,
            Self::LogDiscountFactor => SensitivityUnit::CurrencyPerLogDiscountFactor,
            Self::ForwardLogDensity => SensitivityUnit::CurrencyPerLogDensity,
            Self::CashMean | Self::ModelParameter => SensitivityUnit::CurrencyPerUnitParameter,
        }
    }
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Spot => "spot",
            Self::Volatility => "volatility",
            Self::LocalVariance => "local_variance",
            Self::LogDiscountFactor => "log_discount_factor",
            Self::ForwardLogDensity => "forward_log_density",
            Self::CashMean => "cash_mean",
            Self::ModelParameter => "model_parameter",
        }
    }
}
impl SensitivityUnit {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CurrencyPerSpot => "currency_per_spot",
            Self::CurrencyPerSpotSquared => "currency_per_spot_squared",
            Self::CurrencyPerUnitVolatility => "currency_per_unit_volatility",
            Self::CurrencyPerVariance => "currency_per_variance",
            Self::CurrencyPerLogDiscountFactor => "currency_per_log_discount_factor",
            Self::CurrencyPerLogDensity => "currency_per_log_density",
            Self::CurrencyPerUnitParameter => "currency_per_unit_parameter",
            Self::CurrencyPerOnePercentSpot => "currency_per_one_percent_spot",
            Self::CurrencyPerOnePercentSpotSquared => "currency_per_one_percent_spot_squared",
            Self::CurrencyPerVolatilityPoint => "currency_per_volatility_point",
            Self::CurrencyPerBasisPoint => "currency_per_basis_point",
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct ReportAxis {
    pub name: String,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ReportedRisk {
    pub name: String,
    pub underlying_ids: Vec<u32>,
    /// Cartesian axes in row-major order; empty for a scalar.
    pub axes: Vec<ReportAxis>,
    pub values: Vec<f64>,
    /// None means unavailable, never an implicit zero sampling error.
    pub standard_errors: Option<Vec<f64>>,
    pub unit: SensitivityUnit,
    pub method: String,
    pub calibration: CalibrationTreatment,
    /// Coordinate and fixed-input conventions, alongside the exact saved task.
    pub convention: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ValuationReport {
    pub(crate) task_fingerprint: Fingerprint,
    pub(crate) plan_fingerprint: String,
    pub(crate) library_version: String,
    pub(crate) platform: String,
    pub(crate) currency_id: u16,
    pub(crate) value: f64,
    pub(crate) standard_error: f64,
    pub(crate) estimator: EstimatorKind,
    pub(crate) independent_sampling_units: u64,
    pub(crate) evaluated_paths: u128,
    pub(crate) uncertainty_scope: UncertaintyScope,
    pub(crate) calibration_seeds: Vec<u64>,
    pub(crate) risks: Vec<ReportedRisk>,
    /// Existing typed VegaKT contract, including its covariance layout and units.
    pub(crate) vega_kt: Option<VegaKtResult>,
}
impl ValuationReport {
    #[must_use]
    pub fn task_fingerprint(&self) -> Fingerprint {
        self.task_fingerprint
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> &str {
        &self.plan_fingerprint
    }
    #[must_use]
    pub fn library_version(&self) -> &str {
        &self.library_version
    }
    #[must_use]
    pub fn platform(&self) -> &str {
        &self.platform
    }
    #[must_use]
    pub fn currency_id(&self) -> u16 {
        self.currency_id
    }
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.standard_error
    }
    #[must_use]
    pub fn estimator(&self) -> EstimatorKind {
        self.estimator
    }
    #[must_use]
    pub fn independent_sampling_units(&self) -> u64 {
        self.independent_sampling_units
    }
    #[must_use]
    pub fn evaluated_paths(&self) -> u128 {
        self.evaluated_paths
    }
    #[must_use]
    pub fn uncertainty_scope(&self) -> UncertaintyScope {
        self.uncertainty_scope
    }
    #[must_use]
    pub fn calibration_seeds(&self) -> &[u64] {
        &self.calibration_seeds
    }
    #[must_use]
    pub fn risks(&self) -> &[ReportedRisk] {
        &self.risks
    }
    #[must_use]
    pub fn vega_kt(&self) -> Option<&VegaKtResult> {
        self.vega_kt.as_ref()
    }
    pub(crate) fn validate(&self) -> Result<(), WireError> {
        let invalid = |message: &str| WireError::DomainAt {
            pointer: "/report".into(),
            message: message.into(),
        };
        if !self.value.is_finite()
            || !self.standard_error.is_finite()
            || self.standard_error < 0.0
            || self.independent_sampling_units == 0
            || self.evaluated_paths == 0
        {
            return Err(invalid(
                "finite price, nonnegative error and positive sampling counts are required",
            ));
        }
        if self.library_version.is_empty()
            || self.platform.is_empty()
            || self
                .plan_fingerprint
                .strip_prefix("blake3-256:")
                .is_none_or(|hex| {
                    hex.len() != 64
                        || !hex
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
        {
            return Err(invalid("invalid replay metadata"));
        }
        let mut identities = std::collections::HashSet::new();
        for r in &self.risks {
            let count = r
                .axes
                .iter()
                .try_fold(1_usize, |n, a| n.checked_mul(a.values.len()));
            if r.name.is_empty()
                || r.method.is_empty()
                || r.convention.is_empty()
                || !identities.insert((&r.name, &r.underlying_ids))
                || count != Some(r.values.len())
                || r.values.is_empty()
                || r.values.iter().any(|v| !v.is_finite())
                || r.axes.iter().any(|a| {
                    a.name.is_empty()
                        || a.values.is_empty()
                        || a.values.iter().any(|v| !v.is_finite())
                })
                || r.standard_errors.as_ref().is_some_and(|s| {
                    s.len() != r.values.len() || s.iter().any(|v| !v.is_finite() || *v < 0.0)
                })
            {
                return Err(invalid(
                    "invalid or duplicate risk coordinate, shape, value or uncertainty",
                ));
            }
        }
        Ok(())
    }
}
