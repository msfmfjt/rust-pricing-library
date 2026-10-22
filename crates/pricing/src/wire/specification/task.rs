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
        }
    }
}
impl From<OperationV1> for PricingOperation {
    fn from(v: OperationV1) -> Self {
        match v {
            OperationV1::Evaluate {} => Self::Evaluate,
            OperationV1::Aad {
                gamma_relative_bump,
            } => Self::Aad {
                gamma_relative_bump,
            },
            OperationV1::LocalVarianceRisk {} => Self::LocalVarianceRisk,
        }
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
            schema_version: 1,
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
        PricingTask::new(source, self.operation.into()).map_err(source_error)
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
    if header.schema_version != 1 {
        return Err(WireError::UnsupportedSchemaVersion(header.schema_version));
    }
    let task: TaskV1 = serde_json::from_str(text).map_err(json)?;
    task.into_domain()
}
pub fn pricing_task_to_json(task: &PricingTask) -> Result<String, WireError> {
    serialize(&TaskV1::from(task), false)
}
pub fn pricing_task_to_pretty_json(task: &PricingTask) -> Result<String, WireError> {
    serialize(&TaskV1::from(task), true)
}
pub fn fingerprint_pricing_task(task: &PricingTask) -> Result<Fingerprint, WireError> {
    let value = serde_json::to_value(TaskV1::from(task)).map_err(json)?;
    let mut bytes = b"pricing/task/v1\0".to_vec();
    encode_value(&value, &mut bytes)?;
    Ok(Fingerprint::from_bytes(*blake3::hash(&bytes).as_bytes()))
}
pub fn current_pricing_task_schema() -> &'static str {
    include_str!("../../../../../schemas/task-v1/pricing_task.schema.json")
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
