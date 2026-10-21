//! A common report document has its own schema; legacy result v1-v3 are unchanged.
use super::*;
use crate::{
    CalibrationKind, CalibrationMetric, CalibrationTreatment, DiagnosticUnit, ReportAxis,
    ReportedCalibration, ReportedRisk, SensitivityUnit, UncertaintyScope, ValuationReport,
};
const KIND: &str = "valuation_report";
macro_rules! wire_enum {
    ($(#[$meta:meta])* $wire:ident,$domain:ident,[$($variant:ident),+])=>{
        $(#[$meta])*
        #[derive(Serialize,Deserialize)] #[serde(rename_all="snake_case")]
        enum $wire {$($variant),+}
        impl From<$domain> for $wire {fn from(v:$domain)->Self {match v {$($domain::$variant=>Self::$variant),+}}}
        impl From<$wire> for $domain {fn from(v:$wire)->Self {match v {$($wire::$variant=>Self::$variant),+}}}
    }
}
wire_enum!(
    DiagnosticUnitV2,
    DiagnosticUnit,
    [
        Dimensionless,
        Count,
        Indicator,
        Time,
        Variance,
        EquityCoordinate,
        BrownianCoordinate
    ]
);
wire_enum!(
    CalibrationKindV2,
    CalibrationKind,
    [
        ParticleLsv,
        HullWhiteLsv,
        LocalCorrelation,
        BassFixedPoint,
        BassSurfaceProjection
    ]
);
wire_enum!(
    ScopeV1,
    UncertaintyScope,
    [
        PricingSampling,
        ConditionalOnCalibration,
        ConditionalOnExercisePolicy
    ]
);
wire_enum!(
    CalibrationV1,
    CalibrationTreatment,
    [
        NotApplicable,
        Recalibrated,
        FixedCalibration,
        FixedExercisePolicy
    ]
);
wire_enum!(
    // Unit spellings deliberately match the public domain and persisted schema.
    #[allow(clippy::enum_variant_names)]
    UnitV1,
    SensitivityUnit,
    [
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
        CurrencyPerBasisPoint
    ]
);
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReportDocument {
    document_kind: String,
    schema_version: u32,
    task_fingerprint: String,
    plan_fingerprint: String,
    library_version: String,
    platform: String,
    currency_id: u16,
    value: f64,
    standard_error: f64,
    estimator: EstimatorV1,
    independent_sampling_units: u64,
    evaluated_paths: String,
    uncertainty_scope: ScopeV1,
    calibration_seeds: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    calibration_diagnostics: Option<Vec<CalibrationV2>>,
    risks: Vec<RiskV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vega_kt: Option<VegaKtReportV1>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AxisV1 {
    name: String,
    values: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskV1 {
    name: String,
    underlying_ids: Vec<u32>,
    axes: Vec<AxisV1>,
    values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    standard_errors: Option<Vec<f64>>,
    unit: UnitV1,
    method: String,
    calibration: CalibrationV1,
    convention: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationV2 {
    kind: CalibrationKindV2,
    underlying_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    axes: Vec<AxisV1>,
    metrics: Vec<MetricV2>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetricV2 {
    name: String,
    values: Vec<f64>,
    unit: DiagnosticUnitV2,
    convention: String,
}
impl From<&ValuationReport> for ReportDocument {
    fn from(r: &ValuationReport) -> Self {
        Self {
            document_kind: KIND.into(),
            schema_version: 2,
            task_fingerprint: r.task_fingerprint().to_string(),
            plan_fingerprint: r.plan_fingerprint().into(),
            library_version: r.library_version().into(),
            platform: r.platform().into(),
            currency_id: r.currency_id(),
            value: r.value(),
            standard_error: r.standard_error(),
            estimator: r.estimator().into(),
            independent_sampling_units: r.independent_sampling_units(),
            evaluated_paths: r.evaluated_paths().to_string(),
            uncertainty_scope: r.uncertainty_scope().into(),
            calibration_seeds: r.calibration_seeds().to_vec(),
            calibration_diagnostics: r.calibration_diagnostics().map(|blocks| {
                blocks
                    .iter()
                    .map(|b| CalibrationV2 {
                        kind: b.kind.into(),
                        underlying_ids: b.underlying_ids.clone(),
                        seed: b.seed,
                        axes: b
                            .axes
                            .iter()
                            .map(|a| AxisV1 {
                                name: a.name.clone(),
                                values: a.values.clone(),
                            })
                            .collect(),
                        metrics: b
                            .metrics
                            .iter()
                            .map(|m| MetricV2 {
                                name: m.name.clone(),
                                values: m.values.clone(),
                                unit: m.unit.into(),
                                convention: m.convention.clone(),
                            })
                            .collect(),
                    })
                    .collect()
            }),
            risks: r
                .risks()
                .iter()
                .map(|v| RiskV1 {
                    name: v.name.clone(),
                    underlying_ids: v.underlying_ids.clone(),
                    axes: v
                        .axes
                        .iter()
                        .map(|a| AxisV1 {
                            name: a.name.clone(),
                            values: a.values.clone(),
                        })
                        .collect(),
                    values: v.values.clone(),
                    standard_errors: v.standard_errors.clone(),
                    unit: v.unit.into(),
                    method: v.method.clone(),
                    calibration: v.calibration.into(),
                    convention: v.convention.clone(),
                })
                .collect(),
            vega_kt: r.vega_kt().map(Into::into),
        }
    }
}
pub fn valuation_report_to_json(r: &ValuationReport) -> Result<String, WireError> {
    r.validate()?;
    serialize(&ReportDocument::from(r), false)
}
pub fn valuation_report_to_pretty_json(r: &ValuationReport) -> Result<String, WireError> {
    r.validate()?;
    serialize(&ReportDocument::from(r), true)
}
pub fn parse_valuation_report_json(
    input: &[u8],
    limits: JsonLimits,
) -> Result<ValuationReport, WireError> {
    let text = validate_and_decode(input, limits)?;
    let header: Envelope = serde_json::from_str(text).map_err(json)?;
    if header.document_kind != KIND {
        return Err(WireError::WrongDocumentKind {
            expected: KIND,
            actual: header.document_kind,
        });
    }
    if !matches!(header.schema_version, 1 | 2) {
        return Err(WireError::UnsupportedSchemaVersion(header.schema_version));
    }
    let r: ReportDocument = serde_json::from_str(text).map_err(json)?;
    if header.schema_version == 1 && r.calibration_diagnostics.is_some() {
        return Err(WireError::DomainAt {
            pointer: "/calibration_diagnostics".into(),
            message: "calibration diagnostics require report schema version 2".into(),
        });
    }
    let r = ValuationReport {
        task_fingerprint: Fingerprint::from_bytes(parse_fingerprint_owned_at(
            &r.task_fingerprint,
            "/task_fingerprint",
        )?),
        plan_fingerprint: r.plan_fingerprint,
        library_version: r.library_version,
        platform: r.platform,
        currency_id: r.currency_id,
        value: r.value,
        standard_error: r.standard_error,
        estimator: r.estimator.into(),
        independent_sampling_units: r.independent_sampling_units,
        evaluated_paths: parse_u128_decimal_at(&r.evaluated_paths, "/evaluated_paths")?,
        uncertainty_scope: r.uncertainty_scope.into(),
        calibration_seeds: r.calibration_seeds,
        calibration_diagnostics: r.calibration_diagnostics.map(|blocks| {
            blocks
                .into_iter()
                .map(|b| ReportedCalibration {
                    kind: b.kind.into(),
                    underlying_ids: b.underlying_ids,
                    seed: b.seed,
                    axes: b
                        .axes
                        .into_iter()
                        .map(|a| ReportAxis {
                            name: a.name,
                            values: a.values,
                        })
                        .collect(),
                    metrics: b
                        .metrics
                        .into_iter()
                        .map(|m| CalibrationMetric {
                            name: m.name,
                            values: m.values,
                            unit: m.unit.into(),
                            convention: m.convention,
                        })
                        .collect(),
                })
                .collect()
        }),
        risks: r
            .risks
            .into_iter()
            .map(|v| ReportedRisk {
                name: v.name,
                underlying_ids: v.underlying_ids,
                axes: v
                    .axes
                    .into_iter()
                    .map(|a| ReportAxis {
                        name: a.name,
                        values: a.values,
                    })
                    .collect(),
                values: v.values,
                standard_errors: v.standard_errors,
                unit: v.unit.into(),
                method: v.method,
                calibration: v.calibration.into(),
                convention: v.convention,
            })
            .collect(),
        vega_kt: r
            .vega_kt
            .map(|v| vega_kt_from_wire(v, "/vega_kt"))
            .transpose()?,
    };
    r.validate()?;
    Ok(r)
}
pub fn current_valuation_report_schema() -> &'static str {
    include_str!("../../../../schemas/report-v2/valuation_report.schema.json")
}
