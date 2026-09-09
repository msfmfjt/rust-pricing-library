use std::error::Error;
use std::fmt;
use std::sync::Arc;

use pricing_core::{CurrencyId, CurveId, Date, EventId, PositiveF64, SchemaVersion, UnderlyingId};
use pricing_market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
    MarketContext,
};
use pricing_mc::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing_models::{
    Black76Spec, BlackScholesSpec, LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec,
};
use pricing_product::{
    ArithmeticAsianSpec, AsianObservation, AsianObservationValue, DigitalPayout, DigitalSpec,
    EuropeanVanillaSpec, FixedLookbackSpec, OptionSide, ProductSpec,
};
use pricing_risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Diagnostics, Estimate, EstimatorKind, PricingRequest, PricingResult, PricingWarning,
    ReplayMetadata, RiskEstimate, RiskReport, RiskUnit, VegaKtResult, VegaKtResultBucketEstimate,
    VegaKtResultCoordinate, VegaKtResultCovarianceLayout, VegaKtResultProjection,
    VegaKtResultReportingStats, VegaKtResultResidualDiagnostics, VegaKtResultUnit,
};

const DOCUMENT_REQUEST: &str = "pricing_request";
const DOCUMENT_RESULT: &str = "pricing_result";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JsonLimits {
    pub max_input_bytes: usize,
    pub max_nesting_depth: usize,
    pub max_string_bytes: usize,
    pub max_number_token_bytes: usize,
    pub max_array_elements: usize,
    pub max_object_members: usize,
    pub max_total_values: usize,
}

impl JsonLimits {
    pub const DEFAULT: Self = Self {
        max_input_bytes: 4 * 1024 * 1024,
        max_nesting_depth: 64,
        max_string_bytes: 1024 * 1024,
        max_number_token_bytes: 128,
        max_array_elements: 100_000,
        max_object_members: 10_000,
        max_total_values: 250_000,
    };
    pub const HARD_CAP: Self = Self {
        max_input_bytes: 64 * 1024 * 1024,
        max_nesting_depth: 256,
        max_string_bytes: 16 * 1024 * 1024,
        max_number_token_bytes: 1024,
        max_array_elements: 2_000_000,
        max_object_members: 250_000,
        max_total_values: 4_000_000,
    };

    pub fn checked(self) -> Result<Self, WireError> {
        let hard = Self::HARD_CAP;
        let valid = self.max_input_bytes <= hard.max_input_bytes
            && self.max_nesting_depth <= hard.max_nesting_depth
            && self.max_string_bytes <= hard.max_string_bytes
            && self.max_number_token_bytes <= hard.max_number_token_bytes
            && self.max_array_elements <= hard.max_array_elements
            && self.max_object_members <= hard.max_object_members
            && self.max_total_values <= hard.max_total_values;
        if valid {
            Ok(self)
        } else {
            Err(WireError::LimitOverrideExceedsHardCap)
        }
    }
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireError {
    Utf8Bom,
    ResourceLimit {
        name: &'static str,
        observed: usize,
        limit: usize,
    },
    LimitOverrideExceedsHardCap,
    Json(String),
    WrongDocumentKind {
        expected: &'static str,
        actual: String,
    },
    UnsupportedSchemaVersion(u32),
    Domain(String),
    InvalidFingerprint(String),
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8Bom => write!(formatter, "UTF-8 BOM is not permitted"),
            Self::ResourceLimit {
                name,
                observed,
                limit,
            } => {
                write!(
                    formatter,
                    "JSON resource limit {name} exceeded: {observed} > {limit}"
                )
            }
            Self::LimitOverrideExceedsHardCap => {
                write!(formatter, "JSON limit override exceeds hard cap")
            }
            Self::Json(message) | Self::Domain(message) => message.fmt(formatter),
            Self::WrongDocumentKind { expected, actual } => {
                write!(
                    formatter,
                    "expected document_kind {expected:?}, received {actual:?}"
                )
            }
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported schema_version {version}")
            }
            Self::InvalidFingerprint(value) => {
                write!(formatter, "invalid BLAKE3-256 fingerprint {value:?}")
            }
        }
    }
}

impl Error for WireError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Fingerprint([u8; 32]);

impl Fingerprint {
    #[must_use]
    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "blake3-256:{}",
            self.0
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MigrationRegistry;

impl MigrationRegistry {
    #[must_use]
    pub const fn current_version(self) -> SchemaVersion {
        SchemaVersion::CURRENT
    }

    #[must_use]
    pub const fn accepted_source_versions(self) -> &'static [u32] {
        &[1]
    }

    pub fn validate_source(self, version: u32) -> Result<(), WireError> {
        if version == SchemaVersion::CURRENT.get() {
            Ok(())
        } else {
            Err(WireError::UnsupportedSchemaVersion(version))
        }
    }
}

#[derive(Deserialize)]
struct Envelope {
    document_kind: String,
    schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestV1 {
    document_kind: String,
    schema_version: u32,
    valuation_date: String,
    product: ProductV1,
    market: MarketV1,
    model: ModelV1,
    engine: EngineV1,
    risk: RiskV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ProductV1 {
    EuropeanVanilla {
        underlying_id: u32,
        currency_id: u16,
        expiry: String,
        strike: f64,
        notional: f64,
        side: SideV1,
    },
    Digital {
        underlying_id: u32,
        currency_id: u16,
        expiry: String,
        strike: f64,
        payout: f64,
        side: SideV1,
        payout_kind: DigitalPayoutV1,
    },
    ArithmeticAsian {
        underlying_id: u32,
        currency_id: u16,
        strike: f64,
        notional: f64,
        side: SideV1,
        observations: Vec<AsianObservationV1>,
        payment_date: String,
    },
    FixedLookback {
        underlying_id: u32,
        currency_id: u16,
        strike: f64,
        notional: f64,
        side: SideV1,
        monitoring_dates: Vec<String>,
        historical_extremum: Option<f64>,
        payment_date: String,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum SideV1 {
    Call,
    Put,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum DigitalPayoutV1 {
    Cash,
    Asset,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AsianObservationV1 {
    date: String,
    weight: f64,
    value: AsianObservationValueV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum AsianObservationValueV1 {
    Known { fixing: f64 },
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum MarketV1 {
    Equity {
        currency_id: u16,
        underlying_id: u32,
        spot: f64,
        discount_curve: CurveV1,
        dividend_curve: CurveV1,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        discrete_dividends: Vec<DividendEventV1>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DividendEventV1 {
    event_id: u32,
    ex_time: f64,
    quote: DividendQuoteV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum DividendQuoteV1 {
    FixedCash { amount: f64 },
    Proportional { beta: f64 },
    FixedCashAndProportional { fixed_cash: f64, beta: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurveV1 {
    curve_id: u32,
    times: Vec<f64>,
    discount_factors: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ModelV1 {
    BlackScholes {
        volatility: f64,
    },
    #[serde(rename = "black_76")]
    Black76 {
        volatility: f64,
    },
    LocalVolatility {
        local_variance_grid: LocalVarianceGridV1,
        #[serde(skip_serializing_if = "Option::is_none")]
        reporting_iv_basis: Option<ReportingIvBasisV1>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalVarianceGridV1 {
    time_nodes: Vec<f64>,
    log_forward_moneyness_nodes: Vec<f64>,
    shape: [usize; 2],
    values: Vec<f64>,
    floor: f64,
    cap: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReportingIvBasisV1 {
    maturity_nodes: Vec<f64>,
    log_forward_moneyness_nodes: Vec<f64>,
    shape: [usize; 2],
    implied_volatilities: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum EngineV1 {
    PseudoMonteCarlo {
        master_seed: u64,
        independent_sampling_units: u64,
        variance_reduction: VarianceReductionV1,
    },
    RandomizedQuasiMonteCarlo {
        points_per_scramble: u64,
        scramble_count: u32,
        master_scramble_seed: u64,
        variance_reduction: VarianceReductionV1,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VarianceReductionV1 {
    antithetic: bool,
    brownian_bridge: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskV1 {
    delta: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    gamma: Option<GammaV1>,
    vega: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    vega_kt: Option<VegaKtV1>,
    smile_dynamics: SmileDynamicsV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint_interval: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aad_tile_capacity: Option<u32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GammaV1 {
    bump: SpotBumpV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum SpotBumpV1 {
    Absolute(f64),
    Relative(f64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VegaKtV1 {
    maturity_nodes: Vec<String>,
    log_forward_moneyness_nodes: Vec<f64>,
    relative_density_threshold: f64,
    full_bucket_covariance: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum SmileDynamicsV1 {
    #[serde(rename = "sticky_log_moneyness")]
    LogMoneyness,
    #[serde(rename = "sticky_strike")]
    Strike,
    #[serde(rename = "sticky_delta")]
    Delta,
}

impl From<&PricingRequest> for RequestV1 {
    fn from(request: &PricingRequest) -> Self {
        Self {
            document_kind: DOCUMENT_REQUEST.to_owned(),
            schema_version: SchemaVersion::CURRENT.get(),
            valuation_date: request.valuation_date().to_string(),
            product: ProductV1::from(request.product()),
            market: MarketV1::from(request.market()),
            model: ModelV1::from(request.model()),
            engine: EngineV1::from(request.engine()),
            risk: RiskV1::from(request.risk()),
        }
    }
}

impl From<&ProductSpec> for ProductV1 {
    fn from(product: &ProductSpec) -> Self {
        match product {
            ProductSpec::EuropeanVanilla(spec) => Self::EuropeanVanilla {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                expiry: spec.expiry().to_string(),
                strike: spec.strike().get(),
                notional: spec.notional().get(),
                side: spec.side().into(),
            },
            ProductSpec::Digital(spec) => Self::Digital {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                expiry: spec.expiry().to_string(),
                strike: spec.strike().get(),
                payout: spec.payout().get(),
                side: spec.side().into(),
                payout_kind: spec.payout_kind().into(),
            },
            ProductSpec::ArithmeticAsian(spec) => Self::ArithmeticAsian {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                strike: spec.strike().get(),
                notional: spec.notional().get(),
                side: spec.side().into(),
                observations: spec
                    .observations()
                    .iter()
                    .map(|observation| AsianObservationV1 {
                        date: observation.date().to_string(),
                        weight: observation.weight().get(),
                        value: match observation.value() {
                            AsianObservationValue::Known(fixing) => {
                                AsianObservationValueV1::Known {
                                    fixing: fixing.get(),
                                }
                            }
                            AsianObservationValue::Unknown => AsianObservationValueV1::Unknown,
                        },
                    })
                    .collect(),
                payment_date: spec.payment_date().to_string(),
            },
            ProductSpec::FixedLookback(spec) => Self::FixedLookback {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                strike: spec.strike().get(),
                notional: spec.notional().get(),
                side: spec.side().into(),
                monitoring_dates: spec
                    .monitoring_dates()
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                historical_extremum: spec.historical_extremum().map(|value| value.get()),
                payment_date: spec.payment_date().to_string(),
            },
        }
    }
}

impl From<OptionSide> for SideV1 {
    fn from(side: OptionSide) -> Self {
        match side {
            OptionSide::Call => Self::Call,
            OptionSide::Put => Self::Put,
        }
    }
}

impl From<DigitalPayout> for DigitalPayoutV1 {
    fn from(value: DigitalPayout) -> Self {
        match value {
            DigitalPayout::Cash => Self::Cash,
            DigitalPayout::Asset => Self::Asset,
        }
    }
}

impl From<&MarketContext> for MarketV1 {
    fn from(market: &MarketContext) -> Self {
        let equity = market.equity();
        let forward = equity.forward();
        Self::Equity {
            currency_id: equity.currency().get(),
            underlying_id: forward.underlying().get(),
            spot: forward.spot().get(),
            discount_curve: CurveV1::from(forward.discount_curve()),
            dividend_curve: CurveV1::from(forward.dividend_curve()),
            discrete_dividends: forward
                .discrete_dividends()
                .map(|dividends| {
                    dividends
                        .events()
                        .iter()
                        .map(|event| DividendEventV1 {
                            event_id: event.event().get(),
                            ex_time: event.ex_time(),
                            quote: DividendQuoteV1::from(*event),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

impl From<pricing_market::CompiledDividendEvent> for DividendQuoteV1 {
    fn from(event: pricing_market::CompiledDividendEvent) -> Self {
        match (event.fixed_cash(), event.beta()) {
            (fixed_cash, 0.0) => Self::FixedCash { amount: fixed_cash },
            (0.0, beta) => Self::Proportional { beta },
            (fixed_cash, beta) => Self::FixedCashAndProportional { fixed_cash, beta },
        }
    }
}

impl From<&LogLinearDiscountCurve> for CurveV1 {
    fn from(curve: &LogLinearDiscountCurve) -> Self {
        Self {
            curve_id: curve.id().get(),
            times: curve.times().to_vec(),
            discount_factors: curve.discount_factors().to_vec(),
        }
    }
}

impl From<&ModelSpec> for ModelV1 {
    fn from(model: &ModelSpec) -> Self {
        match model {
            ModelSpec::BlackScholes(spec) => Self::BlackScholes {
                volatility: spec.volatility().get(),
            },
            ModelSpec::Black76(spec) => Self::Black76 {
                volatility: spec.volatility().get(),
            },
            ModelSpec::LocalVolatility(spec) => Self::LocalVolatility {
                local_variance_grid: LocalVarianceGridV1::from(spec.local_variance_grid()),
                reporting_iv_basis: spec.reporting_iv_basis().map(ReportingIvBasisV1::from),
            },
        }
    }
}

impl From<&pricing_market::LocalVarianceGrid> for LocalVarianceGridV1 {
    fn from(grid: &pricing_market::LocalVarianceGrid) -> Self {
        Self {
            time_nodes: grid.time_nodes().to_vec(),
            log_forward_moneyness_nodes: grid.log_moneyness_nodes().to_vec(),
            shape: [grid.time_nodes().len(), grid.log_moneyness_nodes().len()],
            values: grid.values().to_vec(),
            floor: grid.floor(),
            cap: grid.cap(),
        }
    }
}

impl From<&LocalVolatilityReportingBasis> for ReportingIvBasisV1 {
    fn from(basis: &LocalVolatilityReportingBasis) -> Self {
        Self {
            maturity_nodes: basis.maturity_nodes().to_vec(),
            log_forward_moneyness_nodes: basis.log_forward_moneyness_nodes().to_vec(),
            shape: [
                basis.maturity_nodes().len(),
                basis.log_forward_moneyness_nodes().len(),
            ],
            implied_volatilities: basis.implied_volatilities().to_vec(),
        }
    }
}

impl From<EngineConfig> for EngineV1 {
    fn from(engine: EngineConfig) -> Self {
        match engine {
            EngineConfig::PseudoMonteCarlo(value) => Self::PseudoMonteCarlo {
                master_seed: value.master_seed(),
                independent_sampling_units: value.independent_sampling_units().get(),
                variance_reduction: value.variance_reduction().into(),
            },
            EngineConfig::RandomizedQuasiMonteCarlo(value) => Self::RandomizedQuasiMonteCarlo {
                points_per_scramble: value.points_per_scramble().get(),
                scramble_count: value.scramble_count().get(),
                master_scramble_seed: value.master_scramble_seed(),
                variance_reduction: value.variance_reduction().into(),
            },
        }
    }
}

impl From<VarianceReduction> for VarianceReductionV1 {
    fn from(value: VarianceReduction) -> Self {
        Self {
            antithetic: value.antithetic(),
            brownian_bridge: value.brownian_bridge(),
        }
    }
}

impl From<&RiskRequest> for RiskV1 {
    fn from(risk: &RiskRequest) -> Self {
        Self {
            delta: risk.delta(),
            gamma: risk.gamma().map(|value| GammaV1 {
                bump: value.bump().into(),
            }),
            vega: risk.vega(),
            vega_kt: risk.vega_kt().map(VegaKtV1::from),
            smile_dynamics: risk.smile_dynamics().into(),
            checkpoint_interval: risk.checkpoint_interval().map(std::num::NonZeroU32::get),
            aad_tile_capacity: risk.aad_tile_capacity().map(std::num::NonZeroU32::get),
        }
    }
}

impl From<SpotBump> for SpotBumpV1 {
    fn from(value: SpotBump) -> Self {
        match value {
            SpotBump::Absolute(x) => Self::Absolute(x.get()),
            SpotBump::Relative(x) => Self::Relative(x.get()),
        }
    }
}

impl From<&VegaKtConfig> for VegaKtV1 {
    fn from(value: &VegaKtConfig) -> Self {
        Self {
            maturity_nodes: value
                .maturity_nodes()
                .iter()
                .map(ToString::to_string)
                .collect(),
            log_forward_moneyness_nodes: value
                .log_forward_moneyness_nodes()
                .iter()
                .map(|x| x.get())
                .collect(),
            relative_density_threshold: value.relative_density_threshold().get(),
            full_bucket_covariance: value.full_bucket_covariance(),
        }
    }
}

impl From<SmileDynamics> for SmileDynamicsV1 {
    fn from(value: SmileDynamics) -> Self {
        match value {
            SmileDynamics::StickyLogMoneyness => Self::LogMoneyness,
            SmileDynamics::StickyStrike => Self::Strike,
            SmileDynamics::StickyDelta => Self::Delta,
        }
    }
}

impl TryFrom<RequestV1> for PricingRequest {
    type Error = WireError;
    fn try_from(value: RequestV1) -> Result<Self, Self::Error> {
        check_header(&value.document_kind, value.schema_version, DOCUMENT_REQUEST)?;
        let valuation_date = parse_date(&value.valuation_date)?;
        let product = match value.product {
            ProductV1::EuropeanVanilla {
                underlying_id,
                currency_id,
                expiry,
                strike,
                notional,
                side,
            } => ProductSpec::EuropeanVanilla(
                EuropeanVanillaSpec::new(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    parse_date(&expiry)?,
                    strike,
                    notional,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                )
                .map_err(domain)?,
            ),
            ProductV1::Digital {
                underlying_id,
                currency_id,
                expiry,
                strike,
                payout,
                side,
                payout_kind,
            } => ProductSpec::Digital(
                DigitalSpec::new(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    parse_date(&expiry)?,
                    strike,
                    payout,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                    match payout_kind {
                        DigitalPayoutV1::Cash => DigitalPayout::Cash,
                        DigitalPayoutV1::Asset => DigitalPayout::Asset,
                    },
                )
                .map_err(domain)?,
            ),
            ProductV1::ArithmeticAsian {
                underlying_id,
                currency_id,
                strike,
                notional,
                side,
                observations,
                payment_date,
            } => ProductSpec::ArithmeticAsian(
                ArithmeticAsianSpec::new(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    strike,
                    notional,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                    observations
                        .into_iter()
                        .map(|observation| {
                            let date = parse_date(&observation.date)?;
                            match observation.value {
                                AsianObservationValueV1::Known { fixing } => {
                                    AsianObservation::known(date, observation.weight, fixing)
                                        .map_err(domain)
                                }
                                AsianObservationValueV1::Unknown => {
                                    AsianObservation::unknown(date, observation.weight)
                                        .map_err(domain)
                                }
                            }
                        })
                        .collect::<Result<Vec<_>, WireError>>()?,
                    parse_date(&payment_date)?,
                )
                .map_err(domain)?,
            ),
            ProductV1::FixedLookback {
                underlying_id,
                currency_id,
                strike,
                notional,
                side,
                monitoring_dates,
                historical_extremum,
                payment_date,
            } => ProductSpec::FixedLookback(
                FixedLookbackSpec::new(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    strike,
                    notional,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                    monitoring_dates
                        .into_iter()
                        .map(|date| parse_date(&date))
                        .collect::<Result<Vec<_>, _>>()?,
                    historical_extremum,
                    parse_date(&payment_date)?,
                )
                .map_err(domain)?,
            ),
        };
        let market = match value.market {
            MarketV1::Equity {
                currency_id,
                underlying_id,
                spot,
                discount_curve,
                dividend_curve,
                discrete_dividends,
            } => {
                let discount = Arc::new(curve_from_wire(discount_curve)?);
                let dividend = Arc::new(curve_from_wire(dividend_curve)?);
                let underlying = UnderlyingId::new(underlying_id);
                let spot = PositiveF64::new(spot, "spot").map_err(domain)?;
                let forward = if discrete_dividends.is_empty() {
                    EquityForward::new(underlying, spot, discount, dividend)
                } else {
                    EquityForward::with_discrete_dividends(
                        underlying,
                        spot,
                        discount,
                        dividend,
                        discrete_dividends
                            .into_iter()
                            .map(dividend_event_from_wire)
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .map_err(domain)?
                };
                MarketContext::Equity(EquityMarket::new(CurrencyId::new(currency_id), forward))
            }
        };
        let model = match value.model {
            ModelV1::BlackScholes { volatility } => {
                ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).map_err(domain)?)
            }
            ModelV1::Black76 { volatility } => {
                ModelSpec::Black76(Black76Spec::new(volatility).map_err(domain)?)
            }
            ModelV1::LocalVolatility {
                local_variance_grid,
                reporting_iv_basis,
            } => ModelSpec::LocalVolatility(local_volatility_from_wire(
                local_variance_grid,
                reporting_iv_basis,
            )?),
        };
        let engine = match value.engine {
            EngineV1::PseudoMonteCarlo {
                master_seed,
                independent_sampling_units,
                variance_reduction,
            } => EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(
                    master_seed,
                    independent_sampling_units,
                    variance_reduction.into(),
                )
                .map_err(domain)?,
            ),
            EngineV1::RandomizedQuasiMonteCarlo {
                points_per_scramble,
                scramble_count,
                master_scramble_seed,
                variance_reduction,
            } => EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(
                    points_per_scramble,
                    scramble_count,
                    master_scramble_seed,
                    variance_reduction.into(),
                )
                .map_err(domain)?,
            ),
        };
        let risk = risk_from_wire(value.risk)?;
        PricingRequest::new(valuation_date, product, market, model, engine, risk).map_err(domain)
    }
}

fn local_volatility_from_wire(
    value: LocalVarianceGridV1,
    reporting_iv_basis: Option<ReportingIvBasisV1>,
) -> Result<LocalVolatilitySpec, WireError> {
    let expected_shape = [
        value.time_nodes.len(),
        value.log_forward_moneyness_nodes.len(),
    ];
    if value.shape != expected_shape {
        return Err(WireError::Domain(format!(
            "local_variance_grid shape {:?} does not match node dimensions {:?}",
            value.shape, expected_shape
        )));
    }
    let mut spec = LocalVolatilitySpec::from_explicit_grid(
        value.time_nodes,
        value.log_forward_moneyness_nodes,
        value.values,
        value.floor,
        value.cap,
    )
    .map_err(domain)?;
    if let Some(basis) = reporting_iv_basis {
        spec = spec.with_reporting_iv_basis(reporting_iv_basis_from_wire(basis)?);
    }
    Ok(spec)
}

fn reporting_iv_basis_from_wire(
    value: ReportingIvBasisV1,
) -> Result<LocalVolatilityReportingBasis, WireError> {
    let expected_shape = [
        value.maturity_nodes.len(),
        value.log_forward_moneyness_nodes.len(),
    ];
    if value.shape != expected_shape {
        return Err(WireError::Domain(format!(
            "reporting_iv_basis shape {:?} does not match node dimensions {:?}",
            value.shape, expected_shape
        )));
    }
    LocalVolatilityReportingBasis::new(
        value.maturity_nodes,
        value.log_forward_moneyness_nodes,
        value.implied_volatilities,
    )
    .map_err(domain)
}

fn dividend_event_from_wire(value: DividendEventV1) -> Result<DividendEvent, WireError> {
    let event = EventId::new(value.event_id);
    let quote = match value.quote {
        DividendQuoteV1::FixedCash { amount } => {
            DividendQuote::fixed_cash(amount, event).map_err(domain)?
        }
        DividendQuoteV1::Proportional { beta } => {
            DividendQuote::proportional(beta, event).map_err(domain)?
        }
        DividendQuoteV1::FixedCashAndProportional { fixed_cash, beta } => {
            DividendQuote::fixed_cash_and_proportional(fixed_cash, beta, event).map_err(domain)?
        }
    };
    DividendEvent::new(event, value.ex_time, quote).map_err(domain)
}

impl From<VarianceReductionV1> for VarianceReduction {
    fn from(value: VarianceReductionV1) -> Self {
        Self::new(value.antithetic, value.brownian_bridge)
    }
}

fn curve_from_wire(value: CurveV1) -> Result<LogLinearDiscountCurve, WireError> {
    LogLinearDiscountCurve::new(
        CurveId::new(value.curve_id),
        value.times,
        value.discount_factors,
    )
    .map_err(domain)
}

fn risk_from_wire(value: RiskV1) -> Result<RiskRequest, WireError> {
    let gamma = value
        .gamma
        .map(|item| {
            let bump = match item.bump {
                SpotBumpV1::Absolute(x) => SpotBump::absolute(x),
                SpotBumpV1::Relative(x) => SpotBump::relative(x),
            }
            .map_err(domain)?;
            Ok::<_, WireError>(GammaConfig::new(bump))
        })
        .transpose()?;
    let vega_kt = value
        .vega_kt
        .map(|item| {
            let dates = item
                .maturity_nodes
                .iter()
                .map(|date| parse_date(date))
                .collect::<Result<Vec<_>, _>>()?;
            VegaKtConfig::new(
                dates,
                item.log_forward_moneyness_nodes,
                item.relative_density_threshold,
                item.full_bucket_covariance,
            )
            .map_err(domain)
        })
        .transpose()?;
    let smile = match value.smile_dynamics {
        SmileDynamicsV1::LogMoneyness => SmileDynamics::StickyLogMoneyness,
        SmileDynamicsV1::Strike => SmileDynamics::StickyStrike,
        SmileDynamicsV1::Delta => SmileDynamics::StickyDelta,
    };
    RiskRequest::new(
        value.delta,
        gamma,
        value.vega,
        vega_kt,
        smile,
        value.checkpoint_interval,
        value.aad_tile_capacity,
    )
    .map_err(domain)
}

fn parse_date(value: &str) -> Result<Date, WireError> {
    value.parse().map_err(domain)
}
fn domain(error: impl fmt::Display) -> WireError {
    WireError::Domain(error.to_string())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultV1 {
    document_kind: String,
    schema_version: u32,
    value: EstimateV1,
    risks: RiskReportV1,
    diagnostics: DiagnosticsV1,
    replay: ReplayV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimateV1 {
    value: f64,
    standard_error: f64,
    confidence_lower: f64,
    confidence_upper: f64,
    estimator: EstimatorV1,
    effective_sampling_units: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum EstimatorV1 {
    Analytical,
    PseudoMonteCarlo,
    RandomizedQuasiMonteCarlo,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskReportV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<RiskEstimateV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gamma: Option<RiskEstimateV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vega: Option<RiskEstimateV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vega_kt: Option<VegaKtReportV1>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskEstimateV1 {
    raw: EstimateV1,
    market_scaled: EstimateV1,
    raw_unit: RiskUnitV1,
    market_scaled_unit: RiskUnitV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RiskUnitV1 {
    DeltaRaw,
    DeltaOnePercentSpot,
    GammaRaw,
    GammaOnePercentSpotSquared,
    VegaRaw,
    VegaOneVolPoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VegaKtReportV1 {
    coordinates: Vec<VegaKtCoordinateV1>,
    estimates: Vec<VegaKtBucketEstimateV1>,
    raw_buckets: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_bucket_covariance: Option<Vec<VegaKtCovarianceEntryV1>>,
    covariance_layout: VegaKtCovarianceLayoutV1,
    projection: VegaKtProjectionV1,
    residual_diagnostics: VegaKtResidualDiagnosticsV1,
    raw_unit: VegaKtBucketUnitV1,
    market_scaled_unit: VegaKtBucketUnitV1,
    policy_label: String,
    truncation_order: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VegaKtCoordinateV1 {
    maturity: f64,
    log_moneyness: f64,
    implied_volatility: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VegaKtBucketEstimateV1 {
    raw_mean: f64,
    market_scaled_mean: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_variance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    price_covariance: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum VegaKtCovarianceLayoutV1 {
    PriceAndBucketVarianceOnly,
    FullBucketMatrixRowMajor,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum VegaKtCovarianceEntryV1 {
    Value { value: f64 },
    Unavailable,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VegaKtProjectionV1 {
    scalar_vega: f64,
    signed_residual: f64,
    pre_projection: f64,
    reporting_stats: ReportingIvProjectionStatsV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VegaKtResidualDiagnosticsV1 {
    active_domain_start_index: usize,
    active_domain_end_index: usize,
    active_domain_forward_index: usize,
    excluded_probability_mass: f64,
    signed_residual: f64,
    pre_projection: f64,
    reporting_stats: ReportingIvProjectionStatsV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReportingIvProjectionStatsV1 {
    left_edge_count: u64,
    right_edge_count: u64,
    left_edge_sensitivity: f64,
    right_edge_sensitivity: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum VegaKtBucketUnitV1 {
    CurrencyPerUnitAbsoluteVolatility,
    CurrencyPerVolatilityPoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticsV1 {
    warnings: Vec<WarningV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WarningV1 {
    code: String,
    message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayV1 {
    schema_version: u32,
    request_fingerprint: String,
    library_version: String,
    platform: String,
}

impl From<&PricingResult> for ResultV1 {
    fn from(result: &PricingResult) -> Self {
        Self {
            document_kind: DOCUMENT_RESULT.to_owned(),
            schema_version: SchemaVersion::CURRENT.get(),
            value: result.value.into(),
            risks: RiskReportV1::from(&result.risks),
            diagnostics: DiagnosticsV1 {
                warnings: result
                    .diagnostics
                    .warnings()
                    .iter()
                    .map(|w| WarningV1 {
                        code: w.code().to_owned(),
                        message: w.message().to_owned(),
                    })
                    .collect(),
            },
            replay: ReplayV1 {
                schema_version: result.replay.schema_version().get(),
                request_fingerprint: Fingerprint(*result.replay.request_fingerprint()).to_string(),
                library_version: result.replay.library_version().to_owned(),
                platform: result.replay.platform().to_owned(),
            },
        }
    }
}

impl From<Estimate> for EstimateV1 {
    fn from(value: Estimate) -> Self {
        Self {
            value: value.value().get(),
            standard_error: value.standard_error().get(),
            confidence_lower: value.confidence_interval().lower().get(),
            confidence_upper: value.confidence_interval().upper().get(),
            estimator: value.estimator().into(),
            effective_sampling_units: value.effective_sampling_units().get(),
        }
    }
}

impl From<EstimatorKind> for EstimatorV1 {
    fn from(value: EstimatorKind) -> Self {
        match value {
            EstimatorKind::Analytical => Self::Analytical,
            EstimatorKind::PseudoMonteCarlo => Self::PseudoMonteCarlo,
            EstimatorKind::RandomizedQuasiMonteCarlo => Self::RandomizedQuasiMonteCarlo,
        }
    }
}

impl From<&RiskReport> for RiskReportV1 {
    fn from(value: &RiskReport) -> Self {
        Self {
            delta: value.delta.map(Into::into),
            gamma: value.gamma.map(Into::into),
            vega: value.vega.map(Into::into),
            vega_kt: value.vega_kt.as_ref().map(VegaKtReportV1::from),
        }
    }
}
impl From<RiskEstimate> for RiskEstimateV1 {
    fn from(value: RiskEstimate) -> Self {
        Self {
            raw: value.raw().into(),
            market_scaled: value.market_scaled().into(),
            raw_unit: value.raw_unit().into(),
            market_scaled_unit: value.market_scaled_unit().into(),
        }
    }
}
impl From<RiskUnit> for RiskUnitV1 {
    fn from(value: RiskUnit) -> Self {
        match value {
            RiskUnit::DeltaRaw => Self::DeltaRaw,
            RiskUnit::DeltaOnePercentSpot => Self::DeltaOnePercentSpot,
            RiskUnit::GammaRaw => Self::GammaRaw,
            RiskUnit::GammaOnePercentSpotSquared => Self::GammaOnePercentSpotSquared,
            RiskUnit::VegaRaw => Self::VegaRaw,
            RiskUnit::VegaOneVolPoint => Self::VegaOneVolPoint,
        }
    }
}

impl From<&VegaKtResult> for VegaKtReportV1 {
    fn from(value: &VegaKtResult) -> Self {
        Self {
            coordinates: value
                .coordinates()
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
            estimates: value.estimates().iter().copied().map(Into::into).collect(),
            raw_buckets: value
                .raw_buckets()
                .iter()
                .map(|value| value.get())
                .collect(),
            full_bucket_covariance: value.full_bucket_covariance().map(|values| {
                values
                    .iter()
                    .map(|value| match value {
                        Some(value) => VegaKtCovarianceEntryV1::Value { value: value.get() },
                        None => VegaKtCovarianceEntryV1::Unavailable,
                    })
                    .collect()
            }),
            covariance_layout: value.covariance_layout().into(),
            projection: value.projection().into(),
            residual_diagnostics: value.residual_diagnostics().into(),
            raw_unit: value.raw_unit().into(),
            market_scaled_unit: value.market_scaled_unit().into(),
            policy_label: value.policy_label().to_owned(),
            truncation_order: value.truncation_order().to_owned(),
        }
    }
}

impl From<VegaKtResultCoordinate> for VegaKtCoordinateV1 {
    fn from(value: VegaKtResultCoordinate) -> Self {
        Self {
            maturity: value.maturity().get(),
            log_moneyness: value.log_moneyness().get(),
            implied_volatility: value.implied_volatility().get(),
        }
    }
}

impl From<VegaKtResultBucketEstimate> for VegaKtBucketEstimateV1 {
    fn from(value: VegaKtResultBucketEstimate) -> Self {
        Self {
            raw_mean: value.raw_mean().get(),
            market_scaled_mean: value.market_scaled_mean().get(),
            sample_variance: value.sample_variance().map(|value| value.get()),
            price_covariance: value.price_covariance().map(|value| value.get()),
        }
    }
}

impl From<VegaKtResultCovarianceLayout> for VegaKtCovarianceLayoutV1 {
    fn from(value: VegaKtResultCovarianceLayout) -> Self {
        match value {
            VegaKtResultCovarianceLayout::PriceAndBucketVarianceOnly => {
                Self::PriceAndBucketVarianceOnly
            }
            VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor => {
                Self::FullBucketMatrixRowMajor
            }
        }
    }
}

impl From<VegaKtResultProjection> for VegaKtProjectionV1 {
    fn from(value: VegaKtResultProjection) -> Self {
        Self {
            scalar_vega: value.scalar_vega().get(),
            signed_residual: value.signed_residual().get(),
            pre_projection: value.pre_projection().get(),
            reporting_stats: value.reporting_stats().into(),
        }
    }
}

impl From<VegaKtResultResidualDiagnostics> for VegaKtResidualDiagnosticsV1 {
    fn from(value: VegaKtResultResidualDiagnostics) -> Self {
        Self {
            active_domain_start_index: value.active_domain_start_index(),
            active_domain_end_index: value.active_domain_end_index(),
            active_domain_forward_index: value.active_domain_forward_index(),
            excluded_probability_mass: value.excluded_probability_mass().get(),
            signed_residual: value.signed_residual().get(),
            pre_projection: value.pre_projection().get(),
            reporting_stats: value.reporting_stats().into(),
        }
    }
}

impl From<VegaKtResultReportingStats> for ReportingIvProjectionStatsV1 {
    fn from(value: VegaKtResultReportingStats) -> Self {
        Self {
            left_edge_count: value.left_edge_count(),
            right_edge_count: value.right_edge_count(),
            left_edge_sensitivity: value.left_edge_sensitivity().get(),
            right_edge_sensitivity: value.right_edge_sensitivity().get(),
        }
    }
}

impl From<VegaKtResultUnit> for VegaKtBucketUnitV1 {
    fn from(value: VegaKtResultUnit) -> Self {
        match value {
            VegaKtResultUnit::CurrencyPerUnitAbsoluteVolatility => {
                Self::CurrencyPerUnitAbsoluteVolatility
            }
            VegaKtResultUnit::CurrencyPerVolatilityPoint => Self::CurrencyPerVolatilityPoint,
        }
    }
}

impl TryFrom<ResultV1> for PricingResult {
    type Error = WireError;
    fn try_from(value: ResultV1) -> Result<Self, Self::Error> {
        check_header(&value.document_kind, value.schema_version, DOCUMENT_RESULT)?;
        let replay_version = SchemaVersion::new(value.replay.schema_version).map_err(domain)?;
        Ok(Self {
            value: estimate_from_wire(value.value)?,
            risks: RiskReport {
                delta: value.risks.delta.map(risk_estimate_from_wire).transpose()?,
                gamma: value.risks.gamma.map(risk_estimate_from_wire).transpose()?,
                vega: value.risks.vega.map(risk_estimate_from_wire).transpose()?,
                vega_kt: value.risks.vega_kt.map(vega_kt_from_wire).transpose()?,
            },
            diagnostics: Diagnostics::new(
                value
                    .diagnostics
                    .warnings
                    .into_iter()
                    .map(|w| PricingWarning::new(w.code, w.message))
                    .collect(),
            ),
            replay: ReplayMetadata::new(
                replay_version,
                parse_fingerprint(&value.replay.request_fingerprint)?,
                value.replay.library_version,
                value.replay.platform,
            ),
        })
    }
}

fn estimate_from_wire(value: EstimateV1) -> Result<Estimate, WireError> {
    Estimate::new(
        value.value,
        value.standard_error,
        value.confidence_lower,
        value.confidence_upper,
        match value.estimator {
            EstimatorV1::Analytical => EstimatorKind::Analytical,
            EstimatorV1::PseudoMonteCarlo => EstimatorKind::PseudoMonteCarlo,
            EstimatorV1::RandomizedQuasiMonteCarlo => EstimatorKind::RandomizedQuasiMonteCarlo,
        },
        value.effective_sampling_units,
    )
    .map_err(domain)
}
fn risk_estimate_from_wire(value: RiskEstimateV1) -> Result<RiskEstimate, WireError> {
    Ok(RiskEstimate::new(
        estimate_from_wire(value.raw)?,
        estimate_from_wire(value.market_scaled)?,
        risk_unit_from_wire(value.raw_unit),
        risk_unit_from_wire(value.market_scaled_unit),
    ))
}
fn risk_unit_from_wire(value: RiskUnitV1) -> RiskUnit {
    match value {
        RiskUnitV1::DeltaRaw => RiskUnit::DeltaRaw,
        RiskUnitV1::DeltaOnePercentSpot => RiskUnit::DeltaOnePercentSpot,
        RiskUnitV1::GammaRaw => RiskUnit::GammaRaw,
        RiskUnitV1::GammaOnePercentSpotSquared => RiskUnit::GammaOnePercentSpotSquared,
        RiskUnitV1::VegaRaw => RiskUnit::VegaRaw,
        RiskUnitV1::VegaOneVolPoint => RiskUnit::VegaOneVolPoint,
    }
}

fn vega_kt_from_wire(value: VegaKtReportV1) -> Result<VegaKtResult, WireError> {
    VegaKtResult::new(
        value
            .coordinates
            .into_iter()
            .map(|coordinate| {
                VegaKtResultCoordinate::new(
                    coordinate.maturity,
                    coordinate.log_moneyness,
                    coordinate.implied_volatility,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(domain)?,
        value
            .estimates
            .into_iter()
            .map(|estimate| {
                VegaKtResultBucketEstimate::new(
                    estimate.raw_mean,
                    estimate.market_scaled_mean,
                    estimate.sample_variance,
                    estimate.price_covariance,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(domain)?,
        value.raw_buckets,
        value.full_bucket_covariance.map(|values| {
            values
                .into_iter()
                .map(|value| match value {
                    VegaKtCovarianceEntryV1::Value { value } => Some(value),
                    VegaKtCovarianceEntryV1::Unavailable => None,
                })
                .collect()
        }),
        vega_kt_covariance_layout_from_wire(value.covariance_layout),
        vega_kt_projection_from_wire(value.projection)?,
        vega_kt_residual_diagnostics_from_wire(value.residual_diagnostics)?,
        vega_kt_unit_from_wire(value.raw_unit),
        vega_kt_unit_from_wire(value.market_scaled_unit),
        value.policy_label,
        value.truncation_order,
    )
    .map_err(domain)
}

fn vega_kt_projection_from_wire(
    value: VegaKtProjectionV1,
) -> Result<VegaKtResultProjection, WireError> {
    VegaKtResultProjection::new(
        value.scalar_vega,
        value.signed_residual,
        value.pre_projection,
        reporting_stats_from_wire(value.reporting_stats)?,
    )
    .map_err(domain)
}

fn vega_kt_residual_diagnostics_from_wire(
    value: VegaKtResidualDiagnosticsV1,
) -> Result<VegaKtResultResidualDiagnostics, WireError> {
    VegaKtResultResidualDiagnostics::new(
        value.active_domain_start_index,
        value.active_domain_end_index,
        value.active_domain_forward_index,
        value.excluded_probability_mass,
        value.signed_residual,
        value.pre_projection,
        reporting_stats_from_wire(value.reporting_stats)?,
    )
    .map_err(domain)
}

fn reporting_stats_from_wire(
    value: ReportingIvProjectionStatsV1,
) -> Result<VegaKtResultReportingStats, WireError> {
    VegaKtResultReportingStats::new(
        value.left_edge_count,
        value.right_edge_count,
        value.left_edge_sensitivity,
        value.right_edge_sensitivity,
    )
    .map_err(domain)
}

fn vega_kt_covariance_layout_from_wire(
    value: VegaKtCovarianceLayoutV1,
) -> VegaKtResultCovarianceLayout {
    match value {
        VegaKtCovarianceLayoutV1::PriceAndBucketVarianceOnly => {
            VegaKtResultCovarianceLayout::PriceAndBucketVarianceOnly
        }
        VegaKtCovarianceLayoutV1::FullBucketMatrixRowMajor => {
            VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor
        }
    }
}

fn vega_kt_unit_from_wire(value: VegaKtBucketUnitV1) -> VegaKtResultUnit {
    match value {
        VegaKtBucketUnitV1::CurrencyPerUnitAbsoluteVolatility => {
            VegaKtResultUnit::CurrencyPerUnitAbsoluteVolatility
        }
        VegaKtBucketUnitV1::CurrencyPerVolatilityPoint => {
            VegaKtResultUnit::CurrencyPerVolatilityPoint
        }
    }
}

pub fn request_to_json(request: &PricingRequest) -> Result<String, WireError> {
    serialize(&RequestV1::from(request), false)
}
pub fn request_to_pretty_json(request: &PricingRequest) -> Result<String, WireError> {
    serialize(&RequestV1::from(request), true)
}
pub fn result_to_json(result: &PricingResult) -> Result<String, WireError> {
    serialize(&ResultV1::from(result), false)
}
pub fn result_to_pretty_json(result: &PricingResult) -> Result<String, WireError> {
    serialize(&ResultV1::from(result), true)
}

fn serialize(value: &impl Serialize, pretty: bool) -> Result<String, WireError> {
    let mut json = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|e| WireError::Json(e.to_string()))?;
    json.push('\n');
    Ok(json)
}

pub fn parse_request_json(input: &[u8], limits: JsonLimits) -> Result<PricingRequest, WireError> {
    let text = validate_and_decode(input, limits)?;
    validate_envelope(text, DOCUMENT_REQUEST)?;
    serde_json::from_str::<RequestV1>(text)
        .map_err(json)?
        .try_into()
}

pub fn parse_result_json(input: &[u8], limits: JsonLimits) -> Result<PricingResult, WireError> {
    let text = validate_and_decode(input, limits)?;
    validate_envelope(text, DOCUMENT_RESULT)?;
    serde_json::from_str::<ResultV1>(text)
        .map_err(json)?
        .try_into()
}

pub fn fingerprint_request(request: &PricingRequest) -> Result<Fingerprint, WireError> {
    let value = serde_json::to_value(RequestV1::from(request)).map_err(json)?;
    let mut bytes = b"pricing/request\0".to_vec();
    bytes.extend_from_slice(&SchemaVersion::CURRENT.get().to_be_bytes());
    encode_value(&value, &mut bytes)?;
    Ok(Fingerprint(*blake3::hash(&bytes).as_bytes()))
}

fn encode_value(value: &Value, output: &mut Vec<u8>) -> Result<(), WireError> {
    let mut payload = Vec::new();
    let tag = match value {
        Value::Null => 0,
        Value::Bool(value) => {
            payload.push(u8::from(*value));
            1
        }
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                payload.extend_from_slice(&value.to_be_bytes());
                2
            } else if let Some(value) = number.as_i64() {
                payload.extend_from_slice(&value.to_be_bytes());
                3
            } else {
                let value = number
                    .as_f64()
                    .ok_or_else(|| WireError::Json("non-finite JSON number".to_owned()))?;
                payload.extend_from_slice(&value.to_bits().to_be_bytes());
                4
            }
        }
        Value::String(value) => {
            payload.extend_from_slice(value.as_bytes());
            5
        }
        Value::Array(values) => {
            payload.extend_from_slice(&(values.len() as u64).to_be_bytes());
            for value in values {
                encode_value(value, &mut payload)?;
            }
            6
        }
        Value::Object(values) => {
            payload.extend_from_slice(&(values.len() as u64).to_be_bytes());
            for (key, value) in values {
                encode_value(&Value::String(key.clone()), &mut payload)?;
                encode_value(value, &mut payload)?;
            }
            7
        }
    };
    output.push(tag);
    output.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    output.extend_from_slice(&payload);
    Ok(())
}

fn validate_and_decode(input: &[u8], limits: JsonLimits) -> Result<&str, WireError> {
    let limits = limits.checked()?;
    if input.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(WireError::Utf8Bom);
    }
    enforce("input_bytes", input.len(), limits.max_input_bytes)?;
    let text = std::str::from_utf8(input).map_err(|e| WireError::Json(e.to_string()))?;
    lexical_limits(text, limits)?;
    let value: Value = serde_json::from_str(text).map_err(json)?;
    structural_limits(&value, limits)?;
    Ok(text)
}

fn lexical_limits(text: &str, limits: JsonLimits) -> Result<(), WireError> {
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut string_bytes = 0_usize;
    let mut number_bytes = 0_usize;
    for byte in text.bytes() {
        if in_string {
            if escaped {
                escaped = false;
                string_bytes += 1;
                continue;
            }
            if byte == b'\\' {
                escaped = true;
                continue;
            }
            if byte == b'"' {
                enforce("string_bytes", string_bytes, limits.max_string_bytes)?;
                in_string = false;
                string_bytes = 0;
            } else {
                string_bytes += 1;
            }
            continue;
        }
        if byte == b'"' {
            in_string = true;
            number_bytes = 0;
            continue;
        }
        if byte == b'{' || byte == b'[' {
            depth += 1;
            enforce("nesting_depth", depth, limits.max_nesting_depth)?;
        }
        if byte == b'}' || byte == b']' {
            depth = depth.saturating_sub(1);
        }
        if byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E') {
            number_bytes += 1;
            enforce(
                "number_token_bytes",
                number_bytes,
                limits.max_number_token_bytes,
            )?;
        } else {
            number_bytes = 0;
        }
    }
    Ok(())
}

fn structural_limits(root: &Value, limits: JsonLimits) -> Result<(), WireError> {
    let mut stack = vec![root];
    let mut total = 0_usize;
    while let Some(value) = stack.pop() {
        total += 1;
        enforce("total_values", total, limits.max_total_values)?;
        match value {
            Value::Null => {
                return Err(WireError::Json(
                    "JSON null is not permitted by schema v1".to_owned(),
                ));
            }
            Value::Array(values) => {
                enforce("array_elements", values.len(), limits.max_array_elements)?;
                stack.extend(values);
            }
            Value::Object(values) => {
                enforce("object_members", values.len(), limits.max_object_members)?;
                stack.extend(values.values());
            }
            _ => {}
        }
    }
    Ok(())
}

fn enforce(name: &'static str, observed: usize, limit: usize) -> Result<(), WireError> {
    if observed > limit {
        Err(WireError::ResourceLimit {
            name,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}
fn json(error: serde_json::Error) -> WireError {
    WireError::Json(error.to_string())
}

fn validate_envelope(text: &str, expected: &'static str) -> Result<(), WireError> {
    let envelope: Envelope = serde_json::from_str(text).map_err(json)?;
    check_header(&envelope.document_kind, envelope.schema_version, expected)
}

fn check_header(kind: &str, version: u32, expected: &'static str) -> Result<(), WireError> {
    if kind != expected {
        return Err(WireError::WrongDocumentKind {
            expected,
            actual: kind.to_owned(),
        });
    }
    MigrationRegistry.validate_source(version)
}

fn parse_fingerprint(value: &str) -> Result<[u8; 32], WireError> {
    let hex = value
        .strip_prefix("blake3-256:")
        .ok_or_else(|| WireError::InvalidFingerprint(value.to_owned()))?;
    if hex.len() != 64 {
        return Err(WireError::InvalidFingerprint(value.to_owned()));
    }
    let mut bytes = [0_u8; 32];
    for (index, chunk) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let text = std::str::from_utf8(chunk)
            .map_err(|_| WireError::InvalidFingerprint(value.to_owned()))?;
        bytes[index] = u8::from_str_radix(text, 16)
            .map_err(|_| WireError::InvalidFingerprint(value.to_owned()))?;
    }
    Ok(bytes)
}

#[must_use]
pub const fn current_request_schema() -> &'static str {
    include_str!("../../../schemas/v1/pricing_request.schema.json")
}
#[must_use]
pub const fn current_result_schema() -> &'static str {
    include_str!("../../../schemas/v1/pricing_result.schema.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                "2027-09-04".parse().expect("date"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let forward = EquityForward::new(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(10, 0.95),
            curve(11, 0.98),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("date"),
            product,
            MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn black_76_request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                "2027-09-04".parse().expect("date"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let forward = EquityForward::new(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(10, 0.95),
            curve(11, 0.95),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("date"),
            product,
            MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
            ModelSpec::Black76(Black76Spec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn digital_request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::Digital(
            DigitalSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                "2027-09-04".parse().expect("date"),
                100.0,
                10.0,
                OptionSide::Call,
                DigitalPayout::Cash,
            )
            .expect("product"),
        );
        let forward = EquityForward::new(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(10, 0.95),
            curve(11, 0.98),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("date"),
            product,
            MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn asian_request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown("2026-09-04".parse().expect("date"), 0.25)
                        .expect("first"),
                    AsianObservation::unknown("2027-09-04".parse().expect("date"), 0.75)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
            )
            .expect("product"),
        );
        let forward = EquityForward::new(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(10, 0.95),
            curve(11, 0.98),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("date"),
            product,
            MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn lookback_request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                100.0,
                1.0,
                OptionSide::Put,
                vec![
                    "2026-03-04".parse().expect("date"),
                    "2027-09-04".parse().expect("date"),
                ],
                Some(92.0),
                "2027-09-04".parse().expect("payment"),
            )
            .expect("product"),
        );
        let forward = EquityForward::new(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(10, 0.95),
            curve(11, 0.98),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("date"),
            product,
            MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn dividend_request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                "2027-09-04".parse().expect("date"),
                95.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let event = EventId::new(77);
        let forward = EquityForward::with_discrete_dividends(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(10, 0.95),
            curve(11, 0.98),
            vec![
                DividendEvent::new(
                    event,
                    0.25,
                    DividendQuote::fixed_cash_and_proportional(1.5, 0.02, event).expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("forward");
        PricingRequest::new(
            "2026-09-04".parse().expect("date"),
            product,
            MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn local_vol_request() -> PricingRequest {
        local_vol_request_with_model(
            LocalVolatilitySpec::from_explicit_grid(
                vec![0.25, 1.0],
                vec![-0.1, 0.0, 0.2],
                vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
                1.0e-8,
                4.0,
            )
            .expect("local vol"),
        )
    }

    fn local_vol_request_with_model(model: LocalVolatilitySpec) -> PricingRequest {
        let mut request = request();
        request = PricingRequest::new(
            request.valuation_date(),
            request.product().clone(),
            request.market().clone(),
            ModelSpec::LocalVolatility(model),
            request.engine(),
            request.risk().clone(),
        )
        .expect("request");
        request
    }

    fn local_vol_vega_kt_request() -> PricingRequest {
        let request = local_vol_request();
        PricingRequest::new(
            request.valuation_date(),
            request.product().clone(),
            request.market().clone(),
            request.model().clone(),
            request.engine(),
            RiskRequest::new(
                true,
                None,
                true,
                Some(
                    VegaKtConfig::new(
                        vec![
                            "2027-03-04".parse().expect("date"),
                            "2027-09-04".parse().expect("date"),
                        ],
                        vec![-0.2, 0.0, 0.2],
                        1.0e-8,
                        true,
                    )
                    .expect("vega kt"),
                ),
                SmileDynamics::StickyLogMoneyness,
                Some(16),
                Some(256),
            )
            .expect("risk"),
        )
        .expect("request")
    }

    #[test]
    fn request_round_trip_and_noncanonical_input_have_same_fingerprint() {
        let request = request();
        let compact = request_to_json(&request).expect("json");
        assert_eq!(
            compact,
            include_str!("../../../fixtures/v1/pricing_request.golden.json")
        );
        assert!(compact.ends_with('\n'));
        let parsed = parse_request_json(compact.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        let pretty = request_to_pretty_json(&parsed).expect("pretty");
        let reparsed =
            parse_request_json(pretty.as_bytes(), JsonLimits::DEFAULT).expect("parse pretty");
        assert_eq!(
            fingerprint_request(&parsed).expect("fingerprint"),
            fingerprint_request(&reparsed).expect("fingerprint")
        );
    }

    #[test]
    fn request_json_round_trips_black_76_model() {
        let request = black_76_request();

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"type\":\"black_76\""));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert!(matches!(parsed.model(), ModelSpec::Black76(_)));
        assert_eq!(request_to_json(&parsed).expect("json"), json);
    }

    #[test]
    fn request_json_round_trips_digital_product() {
        let request = digital_request();

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"type\":\"digital\""));
        assert!(json.contains("\"payout_kind\":{\"type\":\"cash\"}"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert!(matches!(parsed.product(), ProductSpec::Digital(_)));
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        assert_eq!(request_to_json(&parsed).expect("json"), json);
    }

    #[test]
    fn request_json_round_trips_arithmetic_asian_product() {
        let request = asian_request();

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"type\":\"arithmetic_asian\""));
        assert!(json.contains("\"observations\""));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert!(matches!(parsed.product(), ProductSpec::ArithmeticAsian(_)));
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        assert_eq!(request_to_json(&parsed).expect("json"), json);
    }

    #[test]
    fn request_json_round_trips_fixed_lookback_product() {
        let request = lookback_request();

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"type\":\"fixed_lookback\""));
        assert!(json.contains("\"historical_extremum\":92.0"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert!(matches!(parsed.product(), ProductSpec::FixedLookback(_)));
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        assert_eq!(request_to_json(&parsed).expect("json"), json);
    }

    #[test]
    fn request_json_round_trips_discrete_dividends() {
        let request = dividend_request();
        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"discrete_dividends\""));
        assert!(json.contains("\"fixed_cash_and_proportional\""));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        let dividends = parsed
            .market()
            .equity()
            .forward()
            .discrete_dividends()
            .expect("dividends");
        assert_eq!(dividends.events()[0].event(), EventId::new(77));
        assert_eq!(dividends.events()[0].fixed_cash(), 1.5);
        assert_eq!(dividends.events()[0].beta(), 0.02);
    }

    #[test]
    fn request_json_round_trips_local_volatility_grid_shape() {
        let request = local_vol_request();
        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"local_volatility\""));
        assert!(json.contains("\"shape\":[2,3]"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        let ModelSpec::LocalVolatility(model) = parsed.model() else {
            panic!("local vol model");
        };
        assert_eq!(model.local_variance_grid().values()[4], 0.045);
        assert!(model.reporting_iv_basis().is_none());
    }

    #[test]
    fn request_json_round_trips_local_volatility_reporting_iv_basis() {
        let basis = LocalVolatilityReportingBasis::new(
            vec![0.25, 1.0],
            vec![-0.2, 0.0, 0.2],
            vec![0.22, 0.20, 0.21, 0.24, 0.22, 0.23],
        )
        .expect("basis");
        let model = LocalVolatilitySpec::from_explicit_grid(
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
            1.0e-8,
            4.0,
        )
        .expect("local vol")
        .with_reporting_iv_basis(basis);
        let request = local_vol_request_with_model(model);
        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"reporting_iv_basis\""));
        assert!(json.contains("\"shape\":[2,3]"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        let ModelSpec::LocalVolatility(model) = parsed.model() else {
            panic!("local vol model");
        };
        let basis = model.reporting_iv_basis().expect("basis");
        assert_eq!(basis.maturity_nodes(), [0.25, 1.0]);
        assert_eq!(basis.log_forward_moneyness_nodes(), [-0.2, 0.0, 0.2]);
        assert_eq!(basis.implied_volatilities()[4], 0.22);
    }

    #[test]
    fn request_json_rejects_mismatched_reporting_iv_basis_shape() {
        let basis = LocalVolatilityReportingBasis::new(
            vec![0.25, 1.0],
            vec![-0.2, 0.0, 0.2],
            vec![0.22, 0.20, 0.21, 0.24, 0.22, 0.23],
        )
        .expect("basis");
        let model = LocalVolatilitySpec::from_explicit_grid(
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
            1.0e-8,
            4.0,
        )
        .expect("local vol")
        .with_reporting_iv_basis(basis);
        let json = request_to_json(&local_vol_request_with_model(model)).expect("json");
        let invalid = json.replacen(
            "\"reporting_iv_basis\":{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[2,3]",
            "\"reporting_iv_basis\":{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[3,2]",
            1,
        );
        assert!(matches!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::Domain(message)) if message.contains("reporting_iv_basis shape")
        ));
    }

    #[test]
    fn request_json_rejects_mismatched_local_volatility_grid_shape() {
        let json = request_to_json(&local_vol_request()).expect("json");
        let invalid = json.replacen("\"shape\":[2,3]", "\"shape\":[3,2]", 1);
        assert!(matches!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::Domain(message)) if message.contains("shape")
        ));
    }

    #[test]
    fn request_json_round_trips_local_volatility_vega_kt_request() {
        let request = local_vol_vega_kt_request();
        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"vega_kt\""));
        assert!(json.contains("\"full_bucket_covariance\":true"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        let vega_kt = parsed.risk().vega_kt().expect("vega kt");
        assert_eq!(vega_kt.maturity_nodes()[0].to_string(), "2027-03-04");
        assert!(vega_kt.full_bucket_covariance());
    }

    #[test]
    fn strict_reader_rejects_unknown_null_future_and_limits() {
        let json = request_to_json(&request()).expect("json");
        let unknown = json.replacen("\"valuation_date\"", "\"unknown\":1,\"valuation_date\"", 1);
        assert!(parse_request_json(unknown.as_bytes(), JsonLimits::DEFAULT).is_err());
        let null = json.replacen("\"risk\":{", "\"risk\":{\"aad_tile_capacity\":null,", 1);
        assert!(parse_request_json(null.as_bytes(), JsonLimits::DEFAULT).is_err());
        let future = json.replacen("\"schema_version\":1", "\"schema_version\":2", 1);
        assert!(matches!(
            parse_request_json(future.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::UnsupportedSchemaVersion(2))
        ));
        let limits = JsonLimits {
            max_input_bytes: 8,
            ..JsonLimits::DEFAULT
        };
        assert!(matches!(
            parse_request_json(json.as_bytes(), limits),
            Err(WireError::ResourceLimit {
                name: "input_bytes",
                ..
            })
        ));
    }

    #[test]
    fn bundled_schemas_are_draft_2020_12_json() {
        for schema in [current_request_schema(), current_result_schema()] {
            let value: Value = serde_json::from_str(schema).expect("schema JSON");
            assert_eq!(
                value["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
        }
    }

    #[test]
    fn result_compact_json_matches_golden_and_round_trips() {
        let value = Estimate::new(10.0, 0.5, 9.0, 11.0, EstimatorKind::PseudoMonteCarlo, 1024)
            .expect("estimate");
        let result = PricingResult {
            value,
            risks: RiskReport::default(),
            diagnostics: Diagnostics::new(vec![PricingWarning::new(
                "curve_extrapolation",
                "discount curve extrapolated",
            )]),
            replay: ReplayMetadata::new(
                SchemaVersion::CURRENT,
                [0; 32],
                "0.1.0",
                "acceptance-test",
            ),
        };
        let json = result_to_json(&result).expect("json");
        assert_eq!(
            json,
            include_str!("../../../fixtures/v1/pricing_result.golden.json")
        );
        assert_eq!(
            parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("round trip"),
            result
        );
    }

    #[test]
    fn result_json_round_trips_vega_kt_report_without_null_covariance_entries() {
        let estimate = Estimate::new(10.0, 0.5, 9.0, 11.0, EstimatorKind::PseudoMonteCarlo, 64)
            .expect("estimate");
        let reporting_stats = VegaKtResultReportingStats::new(1, 2, -0.5, 0.25).expect("stats");
        let vega_kt = VegaKtResult::new(
            vec![
                VegaKtResultCoordinate::new(0.5, -0.1, 0.2).expect("coordinate"),
                VegaKtResultCoordinate::new(0.5, 0.1, 0.21).expect("coordinate"),
            ],
            vec![
                VegaKtResultBucketEstimate::new(1.0, 0.01, Some(0.5), Some(-0.1)).expect("bucket"),
                VegaKtResultBucketEstimate::new(2.0, 0.02, None, Some(0.2)).expect("bucket"),
            ],
            vec![1.0, 2.0],
            Some(vec![Some(0.5), None, None, Some(0.75)]),
            VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor,
            VegaKtResultProjection::new(3.0, -0.25, 2.75, reporting_stats).expect("projection"),
            VegaKtResultResidualDiagnostics::new(1, 2, 1, 0.001, -0.25, 2.75, reporting_stats)
                .expect("residual"),
            VegaKtResultUnit::CurrencyPerUnitAbsoluteVolatility,
            VegaKtResultUnit::CurrencyPerVolatilityPoint,
            "equation_11_first_order_v1",
            "O(delta_t_k)",
        )
        .expect("vega kt");
        let result = PricingResult {
            value: estimate,
            risks: RiskReport {
                vega_kt: Some(vega_kt),
                ..RiskReport::default()
            },
            diagnostics: Diagnostics::default(),
            replay: ReplayMetadata::new(SchemaVersion::CURRENT, [7; 32], "0.1.0", "test-platform"),
        };
        let json = result_to_json(&result).expect("json");
        assert!(json.contains("\"vega_kt\""));
        assert!(json.contains("\"unavailable\""));
        assert!(!json.contains("null"));
        let parsed = parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        let parsed_vega_kt = parsed.risks.vega_kt.expect("vega kt");
        assert_eq!(parsed_vega_kt.coordinates().len(), 2);
        assert_eq!(parsed_vega_kt.estimates()[1].sample_variance(), None);
        assert_eq!(
            parsed_vega_kt.full_bucket_covariance().expect("covariance")[1],
            None
        );
        assert_eq!(parsed_vega_kt.projection().scalar_vega().get(), 3.0);
    }

    #[test]
    fn result_writer_preserves_negative_zero() {
        let estimate =
            Estimate::new(-0.0, 0.0, -0.0, 0.0, EstimatorKind::Analytical, 1).expect("estimate");
        let result = PricingResult {
            value: estimate,
            risks: RiskReport::default(),
            diagnostics: Diagnostics::default(),
            replay: ReplayMetadata::new(SchemaVersion::CURRENT, [1; 32], "0.1.0", "test"),
        };
        assert!(
            result_to_json(&result)
                .expect("json")
                .contains("\"value\":-0.0")
        );
    }
}
