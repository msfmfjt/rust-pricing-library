//! Common estimates with explicit units, differentiation and uncertainty scope.
//! Native diagnostics remain available on `PricingTaskResult::native_result`.
use crate::{EstimatorKind, Fingerprint, MeasureKind, MeasureMethod, VegaKtResult, WireError};

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
    /// The requested measure this row answers. Set by a `Measures` task and
    /// `None` for the three legacy operations, whose rows are named only.
    pub measure: Option<MeasureKind>,
    /// The method the task requested for `measure`, including any bump.
    /// Present exactly when `measure` is.
    pub method_detail: Option<MeasureMethod>,
    /// The parameter a `model_parameters` row differentiates. Present exactly
    /// when `measure` is `ModelParameters`.
    pub model_parameter: Option<crate::ModelParameterId>,
    /// The correlation a `correlations` row differentiates. Present exactly
    /// when `measure` is `Correlations`.
    pub correlation: Option<crate::CorrelationId>,
}

/// One flat row of a report: the price or one coordinate of one risk row.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportRecord {
    /// `price`, a measure name, or the row's display name when it has no
    /// typed measure.
    pub measure: String,
    /// The risk row's display name; `price` for the price record.
    pub name: String,
    pub underlying_ids: Vec<u32>,
    /// Axis name and value for this coordinate; empty for a scalar.
    pub coordinate: Vec<(String, f64)>,
    pub value: f64,
    /// `None` means unavailable, never zero.
    pub standard_error: Option<f64>,
    /// `currency` for the price.
    pub unit: String,
    pub method: String,
    pub calibration: CalibrationTreatment,
    pub uncertainty_scope: UncertaintyScope,
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
    /// None for a historical report without diagnostics; Some([]) for a fresh
    /// report with no applicable calibration. These states are distinct.
    pub(crate) calibration_diagnostics: Option<Vec<crate::ReportedCalibration>>,
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
    pub fn calibration_diagnostics(&self) -> Option<&[crate::ReportedCalibration]> {
        self.calibration_diagnostics.as_deref()
    }
    #[must_use]
    pub fn risks(&self) -> &[ReportedRisk] {
        &self.risks
    }
    #[must_use]
    pub fn vega_kt(&self) -> Option<&VegaKtResult> {
        self.vega_kt.as_ref()
    }
    /// The first row answering `measure`, which is the one in raw units; a
    /// market-scaled row of the same measure follows it in `risks()`.
    /// Bucketed VegaKT keeps its typed result under `vega_kt()`.
    #[must_use]
    pub fn risk(&self, measure: MeasureKind) -> Option<&ReportedRisk> {
        self.risks.iter().find(|r| r.measure == Some(measure))
    }
    /// The `model_parameters` row of one parameter.
    #[must_use]
    pub fn model_parameter_risk(
        &self,
        parameter: crate::ModelParameterId,
    ) -> Option<&ReportedRisk> {
        self.risks
            .iter()
            .find(|r| r.model_parameter == Some(parameter))
    }
    /// The `correlations` row of one correlation.
    #[must_use]
    pub fn correlation_risk(&self, correlation: crate::CorrelationId) -> Option<&ReportedRisk> {
        self.risks
            .iter()
            .find(|r| r.correlation == Some(correlation))
    }
    /// The price followed by every coordinate of every risk row, flattened in
    /// row-major axis order. Bucketed VegaKT is not flattened here.
    #[must_use]
    pub fn to_records(&self) -> Vec<ReportRecord> {
        let estimator = match self.estimator {
            EstimatorKind::Analytical => "analytical",
            EstimatorKind::PseudoMonteCarlo => "pseudo_monte_carlo",
            EstimatorKind::RandomizedQuasiMonteCarlo => "randomized_quasi_monte_carlo",
        };
        let mut records = vec![ReportRecord {
            measure: MeasureKind::Price.as_str().into(),
            name: MeasureKind::Price.as_str().into(),
            underlying_ids: Vec::new(),
            coordinate: Vec::new(),
            value: self.value,
            standard_error: Some(self.standard_error),
            unit: "currency".into(),
            method: estimator.into(),
            calibration: if self.calibration_seeds.is_empty() {
                CalibrationTreatment::NotApplicable
            } else {
                CalibrationTreatment::Recalibrated
            },
            uncertainty_scope: self.uncertainty_scope,
        }];
        for risk in &self.risks {
            for (index, value) in risk.values.iter().enumerate() {
                // Row-major: the last axis varies fastest.
                let mut stride = risk.values.len();
                let coordinate = risk
                    .axes
                    .iter()
                    .map(|axis| {
                        stride /= axis.values.len();
                        (
                            axis.name.clone(),
                            axis.values[(index / stride) % axis.values.len()],
                        )
                    })
                    .collect();
                records.push(ReportRecord {
                    measure: risk
                        .measure
                        .map_or_else(|| risk.name.clone(), |m| m.as_str().into()),
                    name: risk.name.clone(),
                    underlying_ids: risk.underlying_ids.clone(),
                    coordinate,
                    value: *value,
                    standard_error: risk.standard_errors.as_ref().map(|e| e[index]),
                    unit: risk.unit.as_str().into(),
                    method: risk.method.clone(),
                    calibration: risk.calibration,
                    uncertainty_scope: self.uncertainty_scope,
                });
            }
        }
        records
    }
    /// Whether any row carries a typed measure, which report v3 introduced.
    pub(crate) fn has_measures(&self) -> bool {
        self.risks.iter().any(|r| r.measure.is_some())
    }
    pub(crate) fn validate(&self) -> Result<(), WireError> {
        let invalid = |pointer: &str, message: &str| WireError::DomainAt {
            pointer: pointer.into(),
            message: message.into(),
        };
        for (pointer, invalid_value) in [
            ("/value", !self.value.is_finite()),
            (
                "/standard_error",
                !self.standard_error.is_finite() || self.standard_error < 0.0,
            ),
            (
                "/independent_sampling_units",
                self.independent_sampling_units == 0,
            ),
            ("/evaluated_paths", self.evaluated_paths == 0),
        ] {
            if invalid_value {
                return Err(invalid(
                    pointer,
                    "finite price, nonnegative error and positive sampling counts are required",
                ));
            }
        }
        for (pointer, invalid_metadata) in [
            ("/library_version", self.library_version.is_empty()),
            ("/platform", self.platform.is_empty()),
            (
                "/plan_fingerprint",
                self.plan_fingerprint
                    .strip_prefix("blake3-256:")
                    .is_none_or(|hex| {
                        hex.len() != 64
                            || !hex
                                .bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    }),
            ),
        ] {
            if invalid_metadata {
                return Err(invalid(pointer, "invalid replay metadata"));
            }
        }
        let mut identities = std::collections::HashSet::new();
        for (i, r) in self.risks.iter().enumerate() {
            let count = r
                .axes
                .iter()
                .try_fold(1_usize, |n, a| n.checked_mul(a.values.len()));
            if r.name.is_empty()
                || r.method.is_empty()
                || r.convention.is_empty()
                || r.measure.is_some() != r.method_detail.is_some()
                || r.model_parameter.is_some() != (r.measure == Some(MeasureKind::ModelParameters))
                || r.correlation.is_some() != (r.measure == Some(MeasureKind::Correlations))
                || r.measure == Some(MeasureKind::Price)
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
                    &format!("/risks/{i}"),
                    "invalid or duplicate risk coordinate, shape, value or uncertainty",
                ));
            }
        }
        if let Some(blocks) = &self.calibration_diagnostics {
            super::calibration_report::validate(blocks)?;
        }
        Ok(())
    }
}
