//! Independent source-document version; legacy request/result schemas stay frozen.
use super::*;
use crate::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use crate::models::{Bergomi1Factor, Bergomi2Factor, HullWhite1Factor, RoughBergomi};
use crate::{
    EquityModelSpec, PricingModelSpec, PricingSpecification, RateModelSpec,
    SpecificationValidationError, VolatilityFactorSpec,
};

mod rough;
use rough::RoughModelV1;
const KIND: &str = "pricing_specification";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecificationV1 {
    document_kind: String,
    schema_version: u32,
    valuation_date: String,
    product: ProductV1,
    market: MarketV1,
    model: CompositionV1,
    engine: EngineV1,
    risk: RiskV2,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsm: Option<LsmV3>,
    numerics: NumericsV1,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericsV1 {
    worker_threads: u32,
    reduction_block_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    maximum_step: Option<f64>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompositionV1 {
    equity: EquityV1,
    rates: RatesV1,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum EquityV1 {
    Standard {
        model: ModelV1,
    },
    StochasticVolatility {
        initial_volatility: f64,
        factor: FactorV1,
    },
    LocalStochasticVolatility {
        target: LocalTargetV1,
        factor: FactorV1,
        particles: ParticlesV1,
        #[serde(skip_serializing_if = "Option::is_none")]
        forward_density: Option<DensityV1>,
    },
    RoughVolatility {
        model: RoughModelV1,
    },
    RoughLocalStochasticVolatility {
        target: LocalTargetV1,
        model: RoughModelV1,
        particles: ParticlesV1,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum FactorV1 {
    Bergomi {
        mean_reversion: f64,
        vol_of_vol: f64,
        correlation: f64,
    },
    BergomiTwoFactor {
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
    },
    RoughBergomi {
        hurst: f64,
        vol_of_vol: f64,
        correlation: f64,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RatesV1 {
    // An empty struct variant, not a unit variant: serde must reject extra fields.
    Deterministic {},
    HullWhite {
        mean_reversion: f64,
        volatility_times: Vec<f64>,
        volatilities: Vec<f64>,
        equity_rate_correlation: f64,
        vol_rate_correlations: Vec<f64>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalTargetV1 {
    local_variance_grid: LocalVarianceGridV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    reporting_iv_basis: Option<ReportingIvBasisV1>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParticlesV1 {
    particle_count: usize,
    seed: u64,
    log_bandwidth: f64,
    minimum_effective_samples: f64,
    retain_reverse_trace: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DensityV1 {
    local_variance_grid: LocalVarianceGridV1,
    log_densities: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    market_iv: Option<MarketIvV1>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MarketIvV1 {
    maturity_nodes: Vec<f64>,
    log_moneyness_nodes: Vec<f64>,
    implied_volatilities: Vec<f64>,
}

impl From<&VolatilityFactorSpec> for FactorV1 {
    fn from(f: &VolatilityFactorSpec) -> Self {
        match f {
            VolatilityFactorSpec::Bergomi(f) => Self::Bergomi {
                mean_reversion: f.mean_reversion(),
                vol_of_vol: f.vol_of_vol(),
                correlation: f.correlation(),
            },
            VolatilityFactorSpec::BergomiTwoFactor(f) => Self::BergomiTwoFactor {
                mean_reversions: f.mean_reversions(),
                vol_of_vol: f.vol_of_vol(),
                mixing_weight: f.mixing_weight(),
                spot_correlations: f.spot_correlations(),
                factor_correlation: f.factor_correlation(),
            },
            VolatilityFactorSpec::RoughBergomi(f) => Self::RoughBergomi {
                hurst: f.hurst(),
                vol_of_vol: f.vol_of_vol(),
                correlation: f.correlation(),
            },
        }
    }
}
impl FactorV1 {
    fn into_domain(self) -> Result<VolatilityFactorSpec, WireError> {
        let error = |e: &dyn fmt::Display| domain_at("/model/equity/factor", e);
        Ok(match self {
            Self::Bergomi {
                mean_reversion,
                vol_of_vol,
                correlation,
            } => VolatilityFactorSpec::Bergomi(
                Bergomi1Factor::new(mean_reversion, vol_of_vol, correlation)
                    .map_err(|e| error(&e))?,
            ),
            Self::BergomiTwoFactor {
                mean_reversions,
                vol_of_vol,
                mixing_weight,
                spot_correlations,
                factor_correlation,
            } => VolatilityFactorSpec::BergomiTwoFactor(
                Bergomi2Factor::new(
                    mean_reversions,
                    vol_of_vol,
                    mixing_weight,
                    spot_correlations,
                    factor_correlation,
                )
                .map_err(|e| error(&e))?,
            ),
            Self::RoughBergomi {
                hurst,
                vol_of_vol,
                correlation,
            } => VolatilityFactorSpec::RoughBergomi(
                RoughBergomi::new(hurst, vol_of_vol, correlation).map_err(|e| error(&e))?,
            ),
        })
    }
}
impl From<&LocalVolatilitySpec> for LocalTargetV1 {
    fn from(t: &LocalVolatilitySpec) -> Self {
        Self {
            local_variance_grid: t.local_variance_grid().into(),
            reporting_iv_basis: t.reporting_iv_basis().map(Into::into),
        }
    }
}
impl LocalTargetV1 {
    fn into_domain(self) -> Result<LocalVolatilitySpec, WireError> {
        local_volatility_from_wire(
            self.local_variance_grid,
            self.reporting_iv_basis,
            "/model/equity/target/local_variance_grid",
            "/model/equity/target/reporting_iv_basis",
        )
    }
}
impl From<&LsvParticleConfig> for ParticlesV1 {
    fn from(p: &LsvParticleConfig) -> Self {
        Self {
            particle_count: p.particle_count(),
            seed: p.seed(),
            log_bandwidth: p.log_bandwidth(),
            minimum_effective_samples: p.minimum_effective_samples(),
            retain_reverse_trace: p.retain_reverse_trace(),
        }
    }
}
impl ParticlesV1 {
    fn into_domain(self) -> Result<LsvParticleConfig, WireError> {
        LsvParticleConfig::new(
            self.particle_count,
            self.seed,
            self.log_bandwidth,
            self.minimum_effective_samples,
            self.retain_reverse_trace,
        )
        .map_err(|e| domain_at("/model/equity/particles", e))
    }
}
impl From<&HullWhiteLsvTarget> for DensityV1 {
    fn from(t: &HullWhiteLsvTarget) -> Self {
        Self {
            local_variance_grid: t.grid().into(),
            log_densities: t.log_densities().to_vec(),
            market_iv: t.market_iv_surface().map(|s| MarketIvV1 {
                maturity_nodes: s.maturity_nodes().to_vec(),
                log_moneyness_nodes: s.log_moneyness_nodes().to_vec(),
                implied_volatilities: s.implied_volatilities().to_vec(),
            }),
        }
    }
}
impl DensityV1 {
    fn into_domain(self) -> Result<HullWhiteLsvTarget, WireError> {
        let pointer = "/model/equity/forward_density";
        let target = local_volatility_from_wire(
            self.local_variance_grid,
            None,
            "/model/equity/forward_density/local_variance_grid",
            pointer,
        )?;
        let grid = target.local_variance_grid();
        if let Some(s) = self.market_iv {
            let surface = crate::market::MarketIvSurface::new(
                s.maturity_nodes,
                s.log_moneyness_nodes,
                s.implied_volatilities,
            )
            .map_err(|e| domain_at(pointer, e))?;
            let retained = HullWhiteLsvTarget::from_market_iv(
                surface,
                grid.time_nodes().to_vec(),
                grid.log_moneyness_nodes().to_vec(),
                grid.floor(),
                grid.cap(),
            )
            .map_err(|e| domain_at(pointer, e))?;
            if retained.grid() != grid || retained.log_densities() != self.log_densities {
                return Err(domain_at(
                    pointer,
                    "retained market-IV source must reproduce both target arrays",
                ));
            }
            Ok(retained)
        } else {
            HullWhiteLsvTarget::new(grid.clone(), self.log_densities)
                .map_err(|e| domain_at(pointer, e))
        }
    }
}

impl From<&PricingModelSpec> for CompositionV1 {
    fn from(m: &PricingModelSpec) -> Self {
        let equity = match &m.equity {
            EquityModelSpec::Standard(m) => EquityV1::Standard { model: m.into() },
            EquityModelSpec::StochasticVolatility {
                initial_volatility,
                factor,
            } => EquityV1::StochasticVolatility {
                initial_volatility: *initial_volatility,
                factor: factor.into(),
            },
            EquityModelSpec::LocalStochasticVolatility {
                target,
                factor,
                particles,
                forward_density,
            } => EquityV1::LocalStochasticVolatility {
                target: target.into(),
                factor: factor.into(),
                particles: particles.into(),
                forward_density: forward_density.as_ref().map(Into::into),
            },
            EquityModelSpec::RoughVolatility(model) => EquityV1::RoughVolatility {
                model: model.into(),
            },
            EquityModelSpec::RoughLocalStochasticVolatility {
                target,
                model,
                particles,
            } => EquityV1::RoughLocalStochasticVolatility {
                target: target.into(),
                model: model.into(),
                particles: particles.into(),
            },
        };
        let rates = match &m.rates {
            RateModelSpec::Deterministic => RatesV1::Deterministic {},
            RateModelSpec::HullWhite {
                model,
                equity_rate_correlation,
                vol_rate_correlations,
            } => RatesV1::HullWhite {
                mean_reversion: model.mean_reversion(),
                volatility_times: model.volatility_times().to_vec(),
                volatilities: model.volatilities().to_vec(),
                equity_rate_correlation: *equity_rate_correlation,
                vol_rate_correlations: vol_rate_correlations.clone(),
            },
        };
        Self { equity, rates }
    }
}
impl CompositionV1 {
    fn into_domain(self) -> Result<PricingModelSpec, WireError> {
        let equity = match self.equity {
            EquityV1::Standard { model } => EquityModelSpec::Standard(
                model_from_wire(model).map_err(|e| rebase(e, "/model", "/model/equity/model"))?,
            ),
            EquityV1::StochasticVolatility {
                initial_volatility,
                factor,
            } => {
                BlackScholesSpec::new(initial_volatility)
                    .map_err(|e| domain_at("/model/equity/initial_volatility", e))?;
                EquityModelSpec::StochasticVolatility {
                    initial_volatility,
                    factor: factor.into_domain()?,
                }
            }
            EquityV1::LocalStochasticVolatility {
                target,
                factor,
                particles,
                forward_density,
            } => EquityModelSpec::LocalStochasticVolatility {
                target: target.into_domain()?,
                factor: factor.into_domain()?,
                particles: particles.into_domain()?,
                forward_density: forward_density.map(DensityV1::into_domain).transpose()?,
            },
            EquityV1::RoughVolatility { model } => {
                EquityModelSpec::RoughVolatility(model.into_domain()?)
            }
            EquityV1::RoughLocalStochasticVolatility {
                target,
                model,
                particles,
            } => EquityModelSpec::RoughLocalStochasticVolatility {
                target: target.into_domain()?,
                model: model.into_domain()?,
                particles: particles.into_domain()?,
            },
        };
        let rates = match self.rates {
            RatesV1::Deterministic {} => RateModelSpec::Deterministic,
            RatesV1::HullWhite {
                mean_reversion,
                volatility_times,
                volatilities,
                equity_rate_correlation,
                vol_rate_correlations,
            } => RateModelSpec::HullWhite {
                model: HullWhite1Factor::new(mean_reversion, volatility_times, volatilities)
                    .map_err(|e| domain_at("/model/rates", e))?,
                equity_rate_correlation,
                vol_rate_correlations,
            },
        };
        Ok(PricingModelSpec::new(equity, rates))
    }
}
fn rebase(error: WireError, old: &str, new: &str) -> WireError {
    match error {
        WireError::DomainAt { pointer, message } => WireError::DomainAt {
            pointer: pointer.replacen(old, new, 1),
            message,
        },
        other => other,
    }
}
fn source_error(error: SpecificationValidationError) -> WireError {
    match error {
        SpecificationValidationError::Invalid { field, message } => {
            domain_at(format!("/{}", field.replace('.', "/")), message)
        }
        SpecificationValidationError::Request(e) => domain_at("/", e),
    }
}
impl From<&PricingSpecification> for SpecificationV1 {
    fn from(s: &PricingSpecification) -> Self {
        let legacy = RequestV3::from(s.carrier_request());
        let policy = s.numerics().execution_policy;
        Self {
            document_kind: KIND.into(),
            schema_version: 2,
            valuation_date: legacy.valuation_date,
            product: legacy.product,
            market: legacy.market,
            model: s.model().into(),
            engine: legacy.engine,
            risk: legacy.risk,
            lsm: legacy.lsm,
            numerics: NumericsV1 {
                worker_threads: policy.worker_threads().get(),
                reduction_block_size: policy.reduction_block_size().get(),
                maximum_step: s.numerics().maximum_step,
            },
        }
    }
}

pub fn parse_specification_json(
    input: &[u8],
    limits: JsonLimits,
) -> Result<PricingSpecification, WireError> {
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
    let value: SpecificationV1 = serde_json::from_str(text).map_err(json)?;
    if header.schema_version == 1 && matches!(&value.product, ProductV1::Graph { .. }) {
        return Err(WireError::UnsupportedSchemaFeature {
            feature: "graph products require pricing_specification v2",
            schema_version: 1,
        });
    }
    let model = value.model.into_domain()?;
    let policy = ExecutionPolicy::new(
        value.numerics.worker_threads,
        Some(value.numerics.reduction_block_size),
    )
    .map_err(|e| domain_at("/numerics", e))?;
    let carrier = match &model.equity {
        EquityModelSpec::Standard(m) => m.into(),
        EquityModelSpec::StochasticVolatility {
            initial_volatility, ..
        } => ModelV1::BlackScholes {
            volatility: *initial_volatility,
        },
        EquityModelSpec::RoughVolatility(_) => ModelV1::BlackScholes { volatility: 0.0 },
        EquityModelSpec::LocalStochasticVolatility { target, .. }
        | EquityModelSpec::RoughLocalStochasticVolatility { target, .. } => {
            ModelV1::from(&ModelSpec::LocalVolatility(target.clone()))
        }
    };
    let request = PricingRequest::try_from(RequestV3 {
        document_kind: DOCUMENT_REQUEST.into(),
        schema_version: 3,
        valuation_date: value.valuation_date,
        product: value.product,
        market: value.market,
        model: carrier,
        engine: value.engine,
        lsm: value.lsm,
        risk: value.risk,
    })?;
    PricingSpecification::from_request(request, policy)
        .with_model(model, value.numerics.maximum_step)
        .map_err(source_error)
}
pub fn specification_to_json(source: &PricingSpecification) -> Result<String, WireError> {
    serialize(&SpecificationV1::from(source), false)
}
pub fn specification_to_pretty_json(source: &PricingSpecification) -> Result<String, WireError> {
    serialize(&SpecificationV1::from(source), true)
}
pub fn fingerprint_specification(source: &PricingSpecification) -> Result<Fingerprint, WireError> {
    let value = serde_json::to_value(SpecificationV1::from(source)).map_err(json)?;
    let mut bytes = b"pricing/specification\0".to_vec();
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    encode_value(&value, &mut bytes)?;
    Ok(Fingerprint::from_bytes(*blake3::hash(&bytes).as_bytes()))
}
#[must_use]
pub fn current_specification_schema() -> &'static str {
    include_str!("../../../../schemas/specification-v2/pricing_specification.schema.json")
}
