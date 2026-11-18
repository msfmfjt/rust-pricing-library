//! Versioned complete tasks. Domain constructors and native compilers remain authoritative.
use super::super::product_graph::GraphProductV1;
use super::*;
use crate::market::{CorrelationTermStructure, CorrelationToleranceConfig};
use crate::models::BuehlerDividendModel;
use crate::multi_asset::*;
use crate::{
    MultiAssetModelSpec, MultiAssetSpecification, PricingOperation, PricingSource, PricingTask,
    StochasticDividendSpecification,
};
const TASK_KIND: &str = "pricing_task";

// Reused source readers report paths relative to their own document. Preserve
// that detail when embedding them in a task, without reclassifying the error.
fn in_source(error: WireError, prefix: &str) -> WireError {
    match error {
        WireError::DomainAt { pointer, message } => WireError::DomainAt {
            pointer: if pointer.is_empty() || pointer == "/" {
                prefix.to_owned()
            } else {
                format!("{prefix}{pointer}")
            },
            message,
        },
        other => other,
    }
}

fn basket_source_error(error: SpecificationValidationError) -> WireError {
    match error {
        SpecificationValidationError::Invalid { field, message } => {
            // The domain's model configuration is flattened across models,
            // LSV, correlations and rates in the task document.
            let pointer = match field {
                "maximum_step" => "/source/numerics/maximum_step",
                "market" => "/source/markets",
                "product" => "/source/product",
                _ => "/source",
            };
            domain_at(pointer, message)
        }
        other => in_source(source_error(other), "/source"),
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskV1 {
    document_kind: String,
    schema_version: u32,
    operation: OperationV1,
    source: SourceV1,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum OperationV1 {
    Evaluate {},
    Aad {
        #[serde(skip_serializing_if = "Option::is_none")]
        gamma_relative_bump: Option<f64>,
    },
    LocalVarianceRisk {},
    /// Task v3 only.
    Measures {
        measures: Vec<MeasureRequestV3>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasureRequestV3 {
    measure: MeasureV3,
    method: MeasureMethodV3,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum MeasureV3 {
    Price {},
    SpotDelta {},
    SpotGamma {},
    ParallelMarketIvVega {},
    VegaKt { grid: VegaKtV1 },
    ModelParameters { parameters: Vec<ModelParameterV3> },
}
/// A model parameter, named by its role; a component parameter carries its
/// index.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
pub(in crate::wire) enum ModelParameterV3 {
    InitialVariance {},
    InitialState {},
    MeanReversion {},
    LongRunVariance {},
    VolOfVol {},
    ComponentVolOfVol { component: u16 },
    Correlation {},
    Hurst {},
    Quadratic {},
    Shift {},
    VarianceFloor {},
}
impl From<crate::ModelParameterId> for ModelParameterV3 {
    fn from(v: crate::ModelParameterId) -> Self {
        use crate::ModelParameterId as P;
        match v {
            P::InitialVariance => Self::InitialVariance {},
            P::InitialState => Self::InitialState {},
            P::MeanReversion => Self::MeanReversion {},
            P::LongRunVariance => Self::LongRunVariance {},
            P::VolOfVol => Self::VolOfVol {},
            P::ComponentVolOfVol { component } => Self::ComponentVolOfVol { component },
            P::Correlation => Self::Correlation {},
            P::Hurst => Self::Hurst {},
            P::Quadratic => Self::Quadratic {},
            P::Shift => Self::Shift {},
            P::VarianceFloor => Self::VarianceFloor {},
        }
    }
}
impl From<ModelParameterV3> for crate::ModelParameterId {
    fn from(v: ModelParameterV3) -> Self {
        use ModelParameterV3 as W;
        match v {
            W::InitialVariance {} => Self::InitialVariance,
            W::InitialState {} => Self::InitialState,
            W::MeanReversion {} => Self::MeanReversion,
            W::LongRunVariance {} => Self::LongRunVariance,
            W::VolOfVol {} => Self::VolOfVol,
            W::ComponentVolOfVol { component } => Self::ComponentVolOfVol { component },
            W::Correlation {} => Self::Correlation,
            W::Hurst {} => Self::Hurst,
            W::Quadratic {} => Self::Quadratic,
            W::Shift {} => Self::Shift,
            W::VarianceFloor {} => Self::VarianceFloor,
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum MeasureMethodV3 {
    Native {},
    NativeSpotBump {
        bump: SpotBumpV1,
    },
    NativeCalibrated {
        calibration: BumpCalibrationV3,
    },
    NativeSpotBumpCalibrated {
        bump: SpotBumpV1,
        calibration: BumpCalibrationV3,
    },
    BumpAndRevalue {
        size: SpotBumpV1,
        calibration: BumpCalibrationV3,
    },
}
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::wire) enum BumpCalibrationV3 {
    NotApplicable,
    Fixed,
    Recalibrated,
}
impl From<crate::BumpCalibration> for BumpCalibrationV3 {
    fn from(v: crate::BumpCalibration) -> Self {
        match v {
            crate::BumpCalibration::NotApplicable => Self::NotApplicable,
            crate::BumpCalibration::Fixed => Self::Fixed,
            crate::BumpCalibration::Recalibrated => Self::Recalibrated,
        }
    }
}
impl From<BumpCalibrationV3> for crate::BumpCalibration {
    fn from(v: BumpCalibrationV3) -> Self {
        match v {
            BumpCalibrationV3::NotApplicable => Self::NotApplicable,
            BumpCalibrationV3::Fixed => Self::Fixed,
            BumpCalibrationV3::Recalibrated => Self::Recalibrated,
        }
    }
}
pub(in crate::wire) fn bump_size_to_wire(size: crate::BumpSize) -> SpotBumpV1 {
    match size {
        crate::BumpSize::Absolute(h) => SpotBumpV1::Absolute(h.get()),
        crate::BumpSize::Relative(h) => SpotBumpV1::Relative(h.get()),
    }
}
pub(in crate::wire) fn bump_size_from_wire(
    size: SpotBumpV1,
) -> Result<crate::BumpSize, crate::core::CoreError> {
    match size {
        SpotBumpV1::Absolute(h) => crate::BumpSize::absolute(h),
        SpotBumpV1::Relative(h) => crate::BumpSize::relative(h),
    }
}
impl From<&crate::MeasureMethod> for MeasureMethodV3 {
    fn from(v: &crate::MeasureMethod) -> Self {
        match *v {
            crate::MeasureMethod::Native => Self::Native {},
            crate::MeasureMethod::NativeSpotBump { bump } => {
                Self::NativeSpotBump { bump: bump.into() }
            }
            crate::MeasureMethod::NativeCalibrated { calibration } => Self::NativeCalibrated {
                calibration: calibration.into(),
            },
            crate::MeasureMethod::NativeSpotBumpCalibrated { bump, calibration } => {
                Self::NativeSpotBumpCalibrated {
                    bump: bump.into(),
                    calibration: calibration.into(),
                }
            }
            crate::MeasureMethod::BumpAndRevalue { size, calibration } => Self::BumpAndRevalue {
                size: bump_size_to_wire(size),
                calibration: calibration.into(),
            },
        }
    }
}
impl From<&crate::MeasureRequest> for MeasureRequestV3 {
    fn from(v: &crate::MeasureRequest) -> Self {
        use crate::Measure as M;
        Self {
            measure: match &v.measure {
                M::Price => MeasureV3::Price {},
                M::SpotDelta => MeasureV3::SpotDelta {},
                M::SpotGamma => MeasureV3::SpotGamma {},
                M::ParallelMarketIvVega => MeasureV3::ParallelMarketIvVega {},
                M::VegaKt(grid) => MeasureV3::VegaKt { grid: grid.into() },
                M::ModelParameters(selection) => MeasureV3::ModelParameters {
                    parameters: selection.parameters().iter().map(|&p| p.into()).collect(),
                },
            },
            method: (&v.method).into(),
        }
    }
}
impl MeasureRequestV3 {
    fn into_domain(self, index: usize) -> Result<crate::MeasureRequest, WireError> {
        use crate::Measure as M;
        let at = |field: &str| format!("/operation/measures/{index}/{field}");
        let measure = match self.measure {
            MeasureV3::Price {} => M::Price,
            MeasureV3::SpotDelta {} => M::SpotDelta,
            MeasureV3::SpotGamma {} => M::SpotGamma,
            MeasureV3::ParallelMarketIvVega {} => M::ParallelMarketIvVega,
            MeasureV3::VegaKt { grid } => {
                M::VegaKt(vega_kt_from_wire_at(grid, &at("measure/grid"))?)
            }
            MeasureV3::ModelParameters { parameters } => M::ModelParameters(
                crate::ModelParameterSelection::new(
                    parameters.into_iter().map(Into::into).collect(),
                )
                .map_err(|e| domain_at(at("measure/parameters"), e))?,
            ),
        };
        let method = match self.method {
            MeasureMethodV3::Native {} => crate::MeasureMethod::Native,
            MeasureMethodV3::NativeSpotBump { bump } => crate::MeasureMethod::NativeSpotBump {
                bump: spot_bump_from_wire(bump).map_err(|e| domain_at(at("method/bump"), e))?,
            },
            MeasureMethodV3::NativeCalibrated { calibration } => {
                crate::MeasureMethod::NativeCalibrated {
                    calibration: calibration.into(),
                }
            }
            MeasureMethodV3::NativeSpotBumpCalibrated { bump, calibration } => {
                crate::MeasureMethod::NativeSpotBumpCalibrated {
                    bump: spot_bump_from_wire(bump).map_err(|e| domain_at(at("method/bump"), e))?,
                    calibration: calibration.into(),
                }
            }
            MeasureMethodV3::BumpAndRevalue { size, calibration } => {
                crate::MeasureMethod::BumpAndRevalue {
                    size: bump_size_from_wire(size).map_err(|e| domain_at(at("method/size"), e))?,
                    calibration: calibration.into(),
                }
            }
        };
        Ok(crate::MeasureRequest::new(measure, method))
    }
}
impl From<PricingOperation> for OperationV1 {
    fn from(v: PricingOperation) -> Self {
        match v {
            PricingOperation::Evaluate => Self::Evaluate {},
            PricingOperation::Aad {
                gamma_relative_bump,
            } => Self::Aad {
                gamma_relative_bump,
            },
            PricingOperation::LocalVarianceRisk => Self::LocalVarianceRisk {},
            PricingOperation::Measures(set) => Self::Measures {
                measures: set.requests().iter().map(Into::into).collect(),
            },
        }
    }
}
impl OperationV1 {
    fn into_domain(self) -> Result<PricingOperation, WireError> {
        Ok(match self {
            OperationV1::Evaluate {} => PricingOperation::Evaluate,
            OperationV1::Aad {
                gamma_relative_bump,
            } => PricingOperation::Aad {
                gamma_relative_bump,
            },
            OperationV1::LocalVarianceRisk {} => PricingOperation::LocalVarianceRisk,
            OperationV1::Measures { measures } => PricingOperation::Measures(
                crate::MeasureSet::new(
                    measures
                        .into_iter()
                        .enumerate()
                        .map(|(i, m)| m.into_domain(i))
                        .collect::<Result<_, _>>()?,
                )
                .map_err(|e| domain_at("/operation/measures", e))?,
            ),
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum SourceV1 {
    SingleAsset {
        specification: SpecificationV1,
    },
    MultiAsset {
        valuation_date: String,
        product: GraphProductV1,
        markets: Vec<MarketV1>,
        models: Vec<ModelV1>,
        correlation: CorrelationV1,
        lsv: Vec<SlotV1<MarginalV1>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hull_white: Option<BasketRatesV1>,
        #[serde(skip_serializing_if = "Option::is_none")]
        local_correlation: Option<LocalCorrelationV1>,
        engine: EngineV1,
        numerics: NumericsV1,
    },
    StochasticDividend {
        equity: SpecificationV1,
        dividends: DividendsV1,
        volatility_correlations: Vec<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rate_correlation: Option<f64>,
        maximum_step: f64,
    },
}
impl SourceV1 {
    fn requires_v2(&self) -> bool {
        match self {
            Self::SingleAsset { specification } => specification.schema_version >= 3,
            Self::StochasticDividend { equity, .. } => equity.schema_version >= 3,
            Self::MultiAsset { product, .. } => product.requires_v2(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DividendsV1 {
    mean_reversion: f64,
    equity_linkage: f64,
    dividend_volatility: f64,
    equity_dividend_correlation: f64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MarginalV1 {
    factor: FactorV1,
    particles: ParticlesV1,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BasketRatesV1 {
    mean_reversion: f64,
    volatility_times: Vec<f64>,
    volatilities: Vec<f64>,
    rate_correlations: Vec<f64>,
    lsv_targets: Vec<SlotV1<DensityV1>>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorrelationV1 {
    underlying_ids: Vec<u32>,
    entries: Vec<CorrelationEntryV1>,
    tolerances: TolerancesV1,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorrelationEntryV1 {
    date: String,
    matrix: Vec<Vec<f64>>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TolerancesV1 {
    symmetry_abs_tol: f64,
    diagonal_abs_tol: f64,
    psd_abs_tol: f64,
    psd_rel_tol: f64,
    zero_pivot_abs_tol: f64,
    zero_pivot_rel_tol: f64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalCorrelationV1 {
    basket_weights: Vec<f64>,
    target: LocalVarianceGridV1,
    second_correlation: CorrelationV1,
    particles: ParticlesV1,
    feasibility: FeasibilityV1,
    minimum_variance_span: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    second_driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hull_white_target: Option<DensityV1>,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeasibilityV1 {
    Reject,
    ProjectAndReport,
}

impl From<&CorrelationTermStructure> for CorrelationV1 {
    fn from(c: &CorrelationTermStructure) -> Self {
        let t = c.tolerances();
        let n = c.underlyings().len();
        Self {
            underlying_ids: c.underlyings().iter().map(|u| u.get()).collect(),
            entries: c
                .entries()
                .iter()
                .map(|(date, f)| CorrelationEntryV1 {
                    date: date.to_string(),
                    matrix: f.raw().chunks(n).map(<[f64]>::to_vec).collect(),
                })
                .collect(),
            tolerances: TolerancesV1 {
                symmetry_abs_tol: t.symmetry_abs_tol,
                diagonal_abs_tol: t.diagonal_abs_tol,
                psd_abs_tol: t.psd_abs_tol,
                psd_rel_tol: t.psd_rel_tol,
                zero_pivot_abs_tol: t.zero_pivot_abs_tol,
                zero_pivot_rel_tol: t.zero_pivot_rel_tol,
            },
        }
    }
}
impl CorrelationV1 {
    fn into_domain(self, pointer: &str) -> Result<CorrelationTermStructure, WireError> {
        let t = self.tolerances;
        CorrelationTermStructure::new(
            self.underlying_ids
                .into_iter()
                .map(UnderlyingId::new)
                .collect(),
            self.entries
                .into_iter()
                .enumerate()
                .map(|(i, e)| {
                    Ok((
                        parse_date_at(&e.date, "/correlation/entries/date").map_err(|e| {
                            rebase(
                                e,
                                "/correlation/entries/date",
                                &format!("{pointer}/entries/{i}/date"),
                            )
                        })?,
                        e.matrix,
                    ))
                })
                .collect::<Result<Vec<_>, WireError>>()?,
            CorrelationToleranceConfig {
                symmetry_abs_tol: t.symmetry_abs_tol,
                diagonal_abs_tol: t.diagonal_abs_tol,
                psd_abs_tol: t.psd_abs_tol,
                psd_rel_tol: t.psd_rel_tol,
                zero_pivot_abs_tol: t.zero_pivot_abs_tol,
                zero_pivot_rel_tol: t.zero_pivot_rel_tol,
            },
        )
        .map_err(|e| domain_at(pointer, e))
    }
}
impl From<&MultiAssetBergomiLsvConfig> for MarginalV1 {
    fn from(c: &MultiAssetBergomiLsvConfig) -> Self {
        let (f, p) = match c {
            MultiAssetBergomiLsvConfig::OneFactor(c) => {
                (VolatilityFactorSpec::Bergomi(c.factor), &c.particles)
            }
            MultiAssetBergomiLsvConfig::TwoFactor(c) => (
                VolatilityFactorSpec::BergomiTwoFactor(c.factor),
                &c.particles,
            ),
            MultiAssetBergomiLsvConfig::Rough(c) => {
                (VolatilityFactorSpec::RoughBergomi(c.factor), &c.particles)
            }
        };
        Self {
            factor: (&f).into(),
            particles: p.into(),
        }
    }
}
impl MarginalV1 {
    fn into_domain(self, pointer: &str) -> Result<MultiAssetBergomiLsvConfig, WireError> {
        let particles = self.particles.into_domain().map_err(|e| {
            rebase(
                e,
                "/model/equity/particles",
                &format!("{pointer}/particles"),
            )
        })?;
        Ok(
            match self
                .factor
                .into_domain()
                .map_err(|e| rebase(e, "/model/equity/factor", &format!("{pointer}/factor")))?
            {
                VolatilityFactorSpec::Bergomi(factor) => {
                    MultiAssetLsvConfig { factor, particles }.into()
                }
                VolatilityFactorSpec::BergomiTwoFactor(factor) => {
                    MultiAssetLsv2FactorConfig { factor, particles }.into()
                }
                VolatilityFactorSpec::RoughBergomi(factor) => {
                    MultiAssetRoughLsvConfig { factor, particles }.into()
                }
            },
        )
    }
}
impl From<&MultiAssetHullWhiteConfig> for BasketRatesV1 {
    fn from(c: &MultiAssetHullWhiteConfig) -> Self {
        Self {
            mean_reversion: c.rate_model.mean_reversion(),
            volatility_times: c.rate_model.volatility_times().to_vec(),
            volatilities: c.rate_model.volatilities().to_vec(),
            rate_correlations: c.rate_correlations.clone(),
            lsv_targets: c
                .lsv_targets
                .iter()
                .map(|t| SlotV1::from_option(t.as_ref().map(Into::into)))
                .collect(),
        }
    }
}
impl BasketRatesV1 {
    fn into_domain(self) -> Result<MultiAssetHullWhiteConfig, WireError> {
        Ok(MultiAssetHullWhiteConfig {
            rate_model: HullWhite1Factor::new(
                self.mean_reversion,
                self.volatility_times,
                self.volatilities,
            )
            .map_err(|e| domain_at("/source/hull_white", e))?,
            rate_correlations: self.rate_correlations,
            lsv_targets: self
                .lsv_targets
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    t.into_option()
                        .map(DensityV1::into_domain)
                        .transpose()
                        .map_err(|e| {
                            rebase(
                                e,
                                "/model/equity/forward_density",
                                &format!("/source/hull_white/lsv_targets/{i}/value"),
                            )
                        })
                })
                .collect::<Result<_, _>>()?,
        })
    }
}
impl From<&(LocalCorrelationConfig, LocalCorrelationExtensions)> for LocalCorrelationV1 {
    fn from((c, e): &(LocalCorrelationConfig, LocalCorrelationExtensions)) -> Self {
        Self {
            basket_weights: c.basket_weights.clone(),
            target: (&c.target).into(),
            second_correlation: (&c.second_correlation).into(),
            particles: (&c.particles).into(),
            feasibility: match c.feasibility {
                LocalCorrelationFeasibility::Reject => FeasibilityV1::Reject,
                LocalCorrelationFeasibility::ProjectAndReport => FeasibilityV1::ProjectAndReport,
            },
            minimum_variance_span: c.minimum_variance_span,
            second_driver_correlations: e.second_driver_correlations.clone(),
            hull_white_target: e.hull_white_target.as_ref().map(Into::into),
        }
    }
}
impl LocalCorrelationV1 {
    fn into_domain(
        self,
    ) -> Result<(LocalCorrelationConfig, LocalCorrelationExtensions), WireError> {
        let target = local_volatility_from_wire(
            self.target,
            None,
            "/source/local_correlation/target",
            "/source/local_correlation/reporting_iv",
        )?
        .local_variance_grid()
        .clone();
        Ok((
            LocalCorrelationConfig {
                basket_weights: self.basket_weights,
                target,
                second_correlation: self
                    .second_correlation
                    .into_domain("/source/local_correlation/second_correlation")?,
                particles: self.particles.into_domain().map_err(|e| {
                    rebase(
                        e,
                        "/model/equity/particles",
                        "/source/local_correlation/particles",
                    )
                })?,
                feasibility: match self.feasibility {
                    FeasibilityV1::Reject => LocalCorrelationFeasibility::Reject,
                    FeasibilityV1::ProjectAndReport => {
                        LocalCorrelationFeasibility::ProjectAndReport
                    }
                },
                minimum_variance_span: self.minimum_variance_span,
            },
            LocalCorrelationExtensions {
                second_driver_correlations: self.second_driver_correlations,
                hull_white_target: self
                    .hull_white_target
                    .map(DensityV1::into_domain)
                    .transpose()
                    .map_err(|e| {
                        rebase(
                            e,
                            "/model/equity/forward_density",
                            "/source/local_correlation/hull_white_target",
                        )
                    })?,
            },
        ))
    }
}
impl From<&PricingTask> for TaskV1 {
    fn from(t: &PricingTask) -> Self {
        let source = match t.source() {
            PricingSource::SingleAsset(s) => SourceV1::SingleAsset {
                specification: s.into(),
            },
            PricingSource::StochasticDividend(s) => {
                let d = s.dividends();
                SourceV1::StochasticDividend {
                    equity: s.equity().into(),
                    dividends: DividendsV1 {
                        mean_reversion: d.mean_reversion(),
                        equity_linkage: d.equity_linkage(),
                        dividend_volatility: d.dividend_volatility(),
                        equity_dividend_correlation: d.equity_dividend_correlation(),
                    },
                    volatility_correlations: s.volatility_correlations().to_vec(),
                    rate_correlation: s.rate_correlation(),
                    maximum_step: s.maximum_step(),
                }
            }
            PricingSource::MultiAsset(s) => {
                let m = s.model();
                SourceV1::MultiAsset {
                    valuation_date: s.valuation_date().to_string(),
                    product: s.product().into(),
                    markets: s
                        .markets()
                        .iter()
                        .map(|m| MarketV1::from(&MarketContext::Equity(m.clone())))
                        .collect(),
                    models: m.models.iter().map(Into::into).collect(),
                    correlation: (&m.correlation).into(),
                    lsv: m
                        .lsv
                        .iter()
                        .map(|x| SlotV1::from_option(x.as_ref().map(Into::into)))
                        .collect(),
                    driver_correlations: m.driver_correlations.clone(),
                    hull_white: m.hull_white.as_ref().map(Into::into),
                    local_correlation: m.local_correlation.as_ref().map(Into::into),
                    engine: s.engine().into(),
                    numerics: NumericsV1 {
                        worker_threads: s.execution().worker_threads().get(),
                        reduction_block_size: s.execution().reduction_block_size().get(),
                        maximum_step: Some(s.maximum_step()),
                    },
                }
            }
        };
        Self {
            document_kind: TASK_KIND.into(),
            // The oldest version that can express the task, so that a task
            // written before a version existed keeps its bytes and fingerprint.
            schema_version: if t.measures().is_some() {
                3
            } else if source.requires_v2() {
                2
            } else {
                1
            },
            operation: t.operation().into(),
            source,
        }
    }
}
impl TaskV1 {
    fn into_domain(self) -> Result<PricingTask, WireError> {
        let source = match self.source {
            SourceV1::SingleAsset { specification } => PricingSource::SingleAsset(
                specification_from_wire(specification)
                    .map_err(|e| in_source(e, "/source/specification"))?,
            ),
            SourceV1::StochasticDividend {
                equity,
                dividends: d,
                volatility_correlations,
                rate_correlation,
                maximum_step,
            } => PricingSource::StochasticDividend(
                StochasticDividendSpecification::new(
                    specification_from_wire(equity).map_err(|e| in_source(e, "/source/equity"))?,
                    BuehlerDividendModel::new(
                        d.mean_reversion,
                        d.equity_linkage,
                        d.dividend_volatility,
                        d.equity_dividend_correlation,
                    )
                    .map_err(|e| domain_at("/source/dividends", e))?,
                    volatility_correlations,
                    rate_correlation,
                    maximum_step,
                )
                .map_err(|e| in_source(source_error(e), "/source"))?,
            ),
            SourceV1::MultiAsset {
                valuation_date,
                product,
                markets,
                models,
                correlation,
                lsv,
                driver_correlations,
                hull_white,
                local_correlation,
                engine,
                numerics,
            } => {
                let mut m = MultiAssetModelSpec::new(
                    models
                        .into_iter()
                        .enumerate()
                        .map(|(i, m)| {
                            model_from_wire(m)
                                .map_err(|e| rebase(e, "/model", &format!("/source/models/{i}")))
                        })
                        .collect::<Result<_, _>>()?,
                    correlation.into_domain("/source/correlation")?,
                );
                m.lsv = lsv
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| {
                        x.into_option()
                            .map(|m| m.into_domain(&format!("/source/lsv/{i}/value")))
                            .transpose()
                    })
                    .collect::<Result<_, _>>()?;
                m.driver_correlations = driver_correlations;
                m.hull_white = hull_white.map(BasketRatesV1::into_domain).transpose()?;
                m.local_correlation = local_correlation
                    .map(LocalCorrelationV1::into_domain)
                    .transpose()?;
                PricingSource::MultiAsset(
                    MultiAssetSpecification::new(
                        parse_date_at(&valuation_date, "/source/valuation_date")?,
                        product.into_domain("/source/product")?,
                        markets
                            .into_iter()
                            .enumerate()
                            .map(|(i, m)| {
                                market_from_wire(m)
                                    .map(|m| m.equity().clone())
                                    .map_err(|e| {
                                        rebase(e, "/market", &format!("/source/markets/{i}"))
                                    })
                            })
                            .collect::<Result<_, _>>()?,
                        m,
                        engine_from_wire(engine).map_err(|e| in_source(e, "/source"))?,
                        ExecutionPolicy::new(
                            numerics.worker_threads,
                            Some(numerics.reduction_block_size),
                        )
                        .map_err(|e| domain_at("/source/numerics", e))?,
                        numerics.maximum_step.ok_or_else(|| {
                            domain_at("/source/numerics/maximum_step", "required basket step")
                        })?,
                    )
                    .map_err(basket_source_error)?,
                )
            }
        };
        PricingTask::new(source, self.operation.into_domain()?).map_err(source_error)
    }
}
pub fn parse_pricing_task_json(input: &[u8], limits: JsonLimits) -> Result<PricingTask, WireError> {
    let text = validate_and_decode(input, limits)?;
    let header: Envelope = serde_json::from_str(text).map_err(json)?;
    if header.document_kind != TASK_KIND {
        return Err(WireError::WrongDocumentKind {
            expected: TASK_KIND,
            actual: header.document_kind,
        });
    }
    if !matches!(header.schema_version, 1..=3) {
        return Err(WireError::UnsupportedSchemaVersion(header.schema_version));
    }
    let task: TaskV1 = serde_json::from_str(text).map_err(json)?;
    if task.schema_version == 1 && task.source.requires_v2() {
        return Err(WireError::UnsupportedSchemaFeature {
            feature: "contract operations require pricing_task v2",
            schema_version: 1,
        });
    }
    if task.schema_version < 3 && matches!(task.operation, OperationV1::Measures { .. }) {
        return Err(WireError::UnsupportedSchemaFeature {
            feature: "the measures operation requires pricing_task v3",
            schema_version: task.schema_version,
        });
    }
    task.into_domain()
}
pub fn pricing_task_to_json(task: &PricingTask) -> Result<String, WireError> {
    serialize(&TaskV1::from(task), false)
}
pub fn pricing_task_to_pretty_json(task: &PricingTask) -> Result<String, WireError> {
    serialize(&TaskV1::from(task), true)
}
pub fn fingerprint_pricing_task(task: &PricingTask) -> Result<Fingerprint, WireError> {
    let document = TaskV1::from(task);
    let mut bytes = match document.schema_version {
        1 => b"pricing/task/v1\0".to_vec(),
        2 => b"pricing/task/v2\0".to_vec(),
        _ => b"pricing/task/v3\0".to_vec(),
    };
    let value = serde_json::to_value(document).map_err(json)?;
    encode_value(&value, &mut bytes)?;
    Ok(Fingerprint::from_bytes(*blake3::hash(&bytes).as_bytes()))
}
pub fn current_pricing_task_schema() -> &'static str {
    include_str!("../../../../../schemas/task-v3/pricing_task.schema.json")
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum SlotV1<T> {
    Absent {},
    Present { value: T },
}
impl<T> SlotV1<T> {
    fn from_option(value: Option<T>) -> Self {
        match value {
            None => Self::Absent {},
            Some(value) => Self::Present { value },
        }
    }
    fn into_option(self) -> Option<T> {
        match self {
            Self::Absent {} => None,
            Self::Present { value } => Some(value),
        }
    }
}
