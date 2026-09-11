use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use pricing_core::{CurrencyId, CurveId, Date, EventId, PositiveF64, SchemaVersion, UnderlyingId};
use pricing_market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
    MarketContext,
};
use pricing_mc::{
    ContinueAllReason, CpqrConfig, EngineConfig, ExerciseDecisionModel, ExercisePolicyFingerprint,
    ExerciseRegressionDiagnostics, FeatureScaling, LsmConfig, LsmStateVariable, LsmWarning,
    PolynomialBasisSpec, PolynomialRegressionModel, PseudoMcConfig, RandomDomain, RqmcConfig,
    VarianceReduction,
};
use pricing_models::{
    Black76Spec, BlackScholesSpec, LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec,
};
use pricing_product::{
    AmericanVanillaSpec, ArithmeticAsianSpec, AsianObservation, AsianObservationValue,
    BarrierDirection, BarrierMonitoring, BarrierSpec, BarrierStyle, DigitalPayout, DigitalSpec,
    EuropeanVanillaSpec, FixedLookbackSpec, OptionSide, ProductSpec,
};
use pricing_risk::{
    GammaConfig, PayoffSmoothing, PayoffSmoothingWidthLadder, RiskRequest, SmileDynamics, SpotBump,
    VegaKtConfig,
};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::{
    BarrierBridgeDiagnostics, BarrierHitIndicatorMode, Diagnostics, EarlyExerciseDiagnostics,
    Estimate, EstimatorKind, ExerciseStrategyRisk, MigrationProvenance, MonteCarloDiagnostics,
    MonteCarloPrice, PathStateDiagnostics, PayoffSmoothingDiagnostics, PayoffSmoothingKernel,
    PayoffSmoothingWidthUnit, PayoffValuationKind, PricingRequest, PricingResult, PricingWarning,
    ReplayMetadata, RiskDiagnostics, RiskEstimate, RiskMethod, RiskMethodMetadata, RiskReport,
    RiskUnit, RiskValidation, StoppingIndexRisk, VegaKtResult, VegaKtResultBucketEstimate,
    VegaKtResultCoordinate, VegaKtResultCovarianceLayout, VegaKtResultProjection,
    VegaKtResultReportingStats, VegaKtResultResidualDiagnostics, VegaKtResultUnit,
};

const DOCUMENT_REQUEST: &str = "pricing_request";
const DOCUMENT_RESULT: &str = "pricing_result";
const MIGRATION_REQUEST_V1_TO_V2: &str = "pricing_request/v1-to-v2";
const MIGRATION_RESULT_V1_TO_V2: &str = "pricing_result/v1-to-v2";
const MIGRATION_REQUEST_V2_TO_V3: &str = "pricing_request/v2-to-v3";
const MIGRATION_RESULT_V2_TO_V3: &str = "pricing_result/v2-to-v3";
const WIRE_MAX_BASIS_COLUMNS: usize = 100_000;
const WIRE_MAX_BASIS_EXPONENTS: usize = 1_000_000;

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
    DomainAt {
        pointer: String,
        message: String,
    },
    WrongDocumentKind {
        expected: &'static str,
        actual: String,
    },
    UnsupportedSchemaVersion(u32),
    Domain(String),
    InvalidFingerprint(String),
    UnsupportedSchemaFeature {
        feature: &'static str,
        schema_version: u32,
    },
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
            Self::DomainAt { message, .. } => message.fmt(formatter),
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
            Self::UnsupportedSchemaFeature {
                feature,
                schema_version,
            } => write!(
                formatter,
                "{feature} is not available in wire schema version {schema_version}"
            ),
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
        &[1, 2, 3]
    }

    pub fn validate_source(self, version: u32) -> Result<(), WireError> {
        if self.accepted_source_versions().contains(&version) {
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
struct RequestV2 {
    document_kind: String,
    schema_version: u32,
    valuation_date: String,
    product: ProductV1,
    market: MarketV1,
    model: ModelV1,
    engine: EngineV1,
    risk: RiskV2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestV3 {
    document_kind: String,
    schema_version: u32,
    valuation_date: String,
    product: ProductV1,
    market: MarketV1,
    model: ModelV1,
    engine: EngineV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsm: Option<LsmV3>,
    risk: RiskV2,
}

#[derive(Clone, Debug, Deserialize)]
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
    AmericanVanilla {
        underlying_id: u32,
        currency_id: u16,
        expiry: String,
        strike: f64,
        notional: f64,
        side: SideV1,
        exercise_dates: Vec<String>,
    },
    Digital {
        underlying_id: u32,
        currency_id: u16,
        expiry: String,
        strike: f64,
        payout: f64,
        side: SideV1,
        payout_kind: DigitalPayoutV1,
        payment_date: Option<String>,
    },
    Barrier {
        underlying_id: u32,
        currency_id: u16,
        expiry: String,
        strike: f64,
        barrier: f64,
        notional: f64,
        side: SideV1,
        direction: BarrierDirectionV1,
        style: BarrierStyleV1,
        monitoring: BarrierMonitoringV1,
        monitoring_dates: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rebate: Option<f64>,
        payment_date: String,
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
        #[serde(skip_serializing_if = "Option::is_none")]
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum BarrierDirectionV1 {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum BarrierStyleV1 {
    KnockIn,
    KnockOut,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum BarrierMonitoringV1 {
    Discrete,
    Continuous,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LsmV3 {
    training_engine: EngineV1,
    state_variables: Vec<LsmStateVariableV3>,
    basis: PolynomialBasisV3,
    itm_abs_tolerance: f64,
    cpqr: CpqrV3,
    max_matrix_elements: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum LsmStateVariableV3 {
    Spot,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolynomialBasisV3 {
    feature_count: u32,
    max_degree: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpqrV3 {
    abs_rank_tolerance: f64,
    rel_rank_tolerance: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VarianceReductionV1 {
    antithetic: bool,
    brownian_bridge: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskV2 {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    payoff_smoothing: Option<PayoffSmoothingV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payoff_smoothing_width_ladder: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskV1 {
    delta: bool,
    gamma: Option<GammaV1>,
    vega: bool,
    vega_kt: Option<VegaKtV1>,
    smile_dynamics: SmileDynamicsV1,
    checkpoint_interval: Option<u32>,
    aad_tile_capacity: Option<u32>,
    payoff_smoothing: Option<PayoffSmoothingV1>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PayoffSmoothingV1 {
    CompactC2 { half_width: f64 },
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

impl From<&PricingRequest> for RequestV2 {
    fn from(request: &PricingRequest) -> Self {
        Self {
            document_kind: DOCUMENT_REQUEST.to_owned(),
            schema_version: SchemaVersion::CURRENT.get(),
            valuation_date: request.valuation_date().to_string(),
            product: ProductV1::from(request.product()),
            market: MarketV1::from(request.market()),
            model: ModelV1::from(request.model()),
            engine: EngineV1::from(request.engine()),
            risk: RiskV2::from(request.risk()),
        }
    }
}

impl From<&PricingRequest> for RequestV3 {
    fn from(request: &PricingRequest) -> Self {
        Self {
            document_kind: DOCUMENT_REQUEST.to_owned(),
            schema_version: 3,
            valuation_date: request.valuation_date().to_string(),
            product: ProductV1::from(request.product()),
            market: MarketV1::from(request.market()),
            model: ModelV1::from(request.model()),
            engine: EngineV1::from(request.engine()),
            lsm: request.lsm().map(LsmV3::from),
            risk: RiskV2::from(request.risk()),
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
            ProductSpec::AmericanVanilla(spec) => Self::AmericanVanilla {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                expiry: spec.expiry().to_string(),
                strike: spec.strike().get(),
                notional: spec.notional().get(),
                side: spec.side().into(),
                exercise_dates: spec
                    .exercise_dates()
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            },
            ProductSpec::Digital(spec) => Self::Digital {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                expiry: spec.expiry().to_string(),
                strike: spec.strike().get(),
                payout: spec.payout().get(),
                side: spec.side().into(),
                payout_kind: spec.payout_kind().into(),
                payment_date: Some(spec.payment_date().to_string()),
            },
            ProductSpec::Barrier(spec) => Self::Barrier {
                underlying_id: spec.underlying().get(),
                currency_id: spec.currency().get(),
                expiry: spec.expiry().to_string(),
                strike: spec.strike().get(),
                barrier: spec.barrier().get(),
                notional: spec.notional().get(),
                side: spec.side().into(),
                direction: spec.direction().into(),
                style: spec.style().into(),
                monitoring: spec.monitoring().into(),
                monitoring_dates: spec
                    .monitoring_dates()
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                rebate: spec.rebate().map(|value| value.get()),
                payment_date: spec.payment_date().to_string(),
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

impl From<BarrierDirection> for BarrierDirectionV1 {
    fn from(value: BarrierDirection) -> Self {
        match value {
            BarrierDirection::Up => Self::Up,
            BarrierDirection::Down => Self::Down,
        }
    }
}

impl From<BarrierStyle> for BarrierStyleV1 {
    fn from(value: BarrierStyle) -> Self {
        match value {
            BarrierStyle::KnockIn => Self::KnockIn,
            BarrierStyle::KnockOut => Self::KnockOut,
        }
    }
}

impl From<BarrierMonitoring> for BarrierMonitoringV1 {
    fn from(value: BarrierMonitoring) -> Self {
        match value {
            BarrierMonitoring::Discrete => Self::Discrete,
            BarrierMonitoring::Continuous => Self::Continuous,
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

impl From<&LsmConfig> for LsmV3 {
    fn from(value: &LsmConfig) -> Self {
        Self {
            training_engine: value.training_engine().into(),
            state_variables: value
                .state_variables()
                .iter()
                .map(|state| match state {
                    LsmStateVariable::Spot => LsmStateVariableV3::Spot,
                })
                .collect(),
            basis: PolynomialBasisV3 {
                feature_count: value.basis().feature_count(),
                max_degree: value.basis().max_degree(),
            },
            itm_abs_tolerance: value.itm_abs_tolerance(),
            cpqr: CpqrV3 {
                abs_rank_tolerance: value.cpqr_config().abs_rank_tolerance(),
                rel_rank_tolerance: value.cpqr_config().rel_rank_tolerance(),
            },
            max_matrix_elements: value.max_matrix_elements(),
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

impl From<&RiskRequest> for RiskV2 {
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
            payoff_smoothing: risk.payoff_smoothing().map(Into::into),
            payoff_smoothing_width_ladder: risk.payoff_smoothing_width_ladder().map(|ladder| {
                ladder
                    .half_widths()
                    .iter()
                    .map(|value| value.get())
                    .collect()
            }),
        }
    }
}

impl From<RequestV1> for RequestV2 {
    fn from(value: RequestV1) -> Self {
        debug_assert_eq!(value.schema_version, 1);
        Self {
            document_kind: value.document_kind,
            schema_version: 2,
            valuation_date: value.valuation_date,
            product: value.product,
            market: value.market,
            model: value.model,
            engine: value.engine,
            risk: value.risk.into(),
        }
    }
}

impl From<RequestV2> for RequestV3 {
    fn from(value: RequestV2) -> Self {
        debug_assert_eq!(value.schema_version, 2);
        Self {
            document_kind: value.document_kind,
            schema_version: 3,
            valuation_date: value.valuation_date,
            product: value.product,
            market: value.market,
            model: value.model,
            engine: value.engine,
            lsm: None,
            risk: value.risk,
        }
    }
}

impl From<RiskV1> for RiskV2 {
    fn from(value: RiskV1) -> Self {
        Self {
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            vega_kt: value.vega_kt,
            smile_dynamics: value.smile_dynamics,
            checkpoint_interval: value.checkpoint_interval,
            aad_tile_capacity: value.aad_tile_capacity,
            payoff_smoothing: value.payoff_smoothing,
            payoff_smoothing_width_ladder: None,
        }
    }
}

impl From<PayoffSmoothing> for PayoffSmoothingV1 {
    fn from(value: PayoffSmoothing) -> Self {
        match value {
            PayoffSmoothing::CompactC2 { half_width } => Self::CompactC2 {
                half_width: half_width.get(),
            },
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

impl TryFrom<RequestV3> for PricingRequest {
    type Error = WireError;
    fn try_from(value: RequestV3) -> Result<Self, Self::Error> {
        check_header(&value.document_kind, value.schema_version, DOCUMENT_REQUEST)?;
        let valuation_date = parse_date_at(&value.valuation_date, "/valuation_date")?;
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
                    parse_date_at(&expiry, "/product/expiry")?,
                    strike,
                    notional,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                )
                .map_err(|error| domain_at("/product", error))?,
            ),
            ProductV1::AmericanVanilla {
                underlying_id,
                currency_id,
                expiry,
                strike,
                notional,
                side,
                exercise_dates,
            } => ProductSpec::AmericanVanilla(
                AmericanVanillaSpec::new(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    parse_date_at(&expiry, "/product/expiry")?,
                    strike,
                    notional,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                    exercise_dates
                        .into_iter()
                        .enumerate()
                        .map(|(index, date)| {
                            parse_date_owned_at(&date, format!("/product/exercise_dates/{index}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
                .map_err(|error| domain_at("/product", error))?,
            ),
            ProductV1::Digital {
                underlying_id,
                currency_id,
                expiry,
                strike,
                payout,
                side,
                payout_kind,
                payment_date,
            } => ProductSpec::Digital(
                DigitalSpec::with_payment_date(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    parse_date_at(&expiry, "/product/expiry")?,
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
                    match payment_date {
                        Some(payment_date) => {
                            parse_date_at(&payment_date, "/product/payment_date")?
                        }
                        None => parse_date_at(&expiry, "/product/expiry")?,
                    },
                )
                .map_err(|error| domain_at("/product", error))?,
            ),
            ProductV1::Barrier {
                underlying_id,
                currency_id,
                expiry,
                strike,
                barrier,
                notional,
                side,
                direction,
                style,
                monitoring,
                monitoring_dates,
                rebate,
                payment_date,
            } => ProductSpec::Barrier(
                BarrierSpec::new(
                    UnderlyingId::new(underlying_id),
                    CurrencyId::new(currency_id),
                    parse_date_at(&expiry, "/product/expiry")?,
                    strike,
                    barrier,
                    notional,
                    match side {
                        SideV1::Call => OptionSide::Call,
                        SideV1::Put => OptionSide::Put,
                    },
                    match direction {
                        BarrierDirectionV1::Up => BarrierDirection::Up,
                        BarrierDirectionV1::Down => BarrierDirection::Down,
                    },
                    match style {
                        BarrierStyleV1::KnockIn => BarrierStyle::KnockIn,
                        BarrierStyleV1::KnockOut => BarrierStyle::KnockOut,
                    },
                    match monitoring {
                        BarrierMonitoringV1::Discrete => BarrierMonitoring::Discrete,
                        BarrierMonitoringV1::Continuous => BarrierMonitoring::Continuous,
                    },
                    monitoring_dates
                        .into_iter()
                        .enumerate()
                        .map(|(index, date)| {
                            parse_date_owned_at(&date, format!("/product/monitoring_dates/{index}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    rebate,
                    parse_date_at(&payment_date, "/product/payment_date")?,
                )
                .map_err(|error| domain_at("/product", error))?,
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
                        .enumerate()
                        .map(|(index, observation)| {
                            let date = parse_date_owned_at(
                                &observation.date,
                                format!("/product/observations/{index}/date"),
                            )?;
                            match observation.value {
                                AsianObservationValueV1::Known { fixing } => {
                                    AsianObservation::known(date, observation.weight, fixing)
                                        .map_err(|error| {
                                            domain_at(
                                                format!("/product/observations/{index}"),
                                                error,
                                            )
                                        })
                                }
                                AsianObservationValueV1::Unknown => AsianObservation::unknown(
                                    date,
                                    observation.weight,
                                )
                                .map_err(|error| {
                                    domain_at(format!("/product/observations/{index}"), error)
                                }),
                            }
                        })
                        .collect::<Result<Vec<_>, WireError>>()?,
                    parse_date_at(&payment_date, "/product/payment_date")?,
                )
                .map_err(|error| domain_at("/product", error))?,
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
                        .enumerate()
                        .map(|(index, date)| {
                            parse_date_owned_at(&date, format!("/product/monitoring_dates/{index}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    historical_extremum,
                    parse_date_at(&payment_date, "/product/payment_date")?,
                )
                .map_err(|error| domain_at("/product", error))?,
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
                let discount = Arc::new(curve_from_wire(discount_curve, "/market/discount_curve")?);
                let dividend = Arc::new(curve_from_wire(dividend_curve, "/market/dividend_curve")?);
                let underlying = UnderlyingId::new(underlying_id);
                let spot = PositiveF64::new(spot, "spot")
                    .map_err(|error| domain_at("/market/spot", error))?;
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
                            .enumerate()
                            .map(|(index, event)| {
                                dividend_event_from_wire(
                                    event,
                                    format!("/market/discrete_dividends/{index}"),
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .map_err(|error| domain_at("/market/discrete_dividends", error))?
                };
                MarketContext::Equity(EquityMarket::new(CurrencyId::new(currency_id), forward))
            }
        };
        let model = match value.model {
            ModelV1::BlackScholes { volatility } => ModelSpec::BlackScholes(
                BlackScholesSpec::new(volatility)
                    .map_err(|error| domain_at("/model/volatility", error))?,
            ),
            ModelV1::Black76 { volatility } => ModelSpec::Black76(
                Black76Spec::new(volatility)
                    .map_err(|error| domain_at("/model/volatility", error))?,
            ),
            ModelV1::LocalVolatility {
                local_variance_grid,
                reporting_iv_basis,
            } => ModelSpec::LocalVolatility(local_volatility_from_wire(
                local_variance_grid,
                reporting_iv_basis,
                "/model/local_variance_grid",
                "/model/reporting_iv_basis",
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
                .map_err(|error| domain_at("/engine", error))?,
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
                .map_err(|error| domain_at("/engine", error))?,
            ),
        };
        let lsm = value.lsm.map(lsm_from_wire).transpose()?;
        let risk = risk_from_wire(value.risk)?;
        PricingRequest::new_with_lsm(valuation_date, product, market, model, engine, risk, lsm)
            .map_err(|error| domain_at("", error))
    }
}

fn lsm_from_wire(value: LsmV3) -> Result<LsmConfig, WireError> {
    let training_engine = match value.training_engine {
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
            .map_err(|error| domain_at("/lsm/training_engine", error))?,
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
            .map_err(|error| domain_at("/lsm/training_engine", error))?,
        ),
    };
    let state_variables = value
        .state_variables
        .into_iter()
        .map(|state| match state {
            LsmStateVariableV3::Spot => LsmStateVariable::Spot,
        })
        .collect::<Vec<_>>();
    let max_total_exponents =
        polynomial_basis_resource_limits(value.basis.feature_count, value.basis.max_degree)?;
    let basis = PolynomialBasisSpec::new(
        value.basis.feature_count,
        value.basis.max_degree,
        max_total_exponents.0,
        max_total_exponents.1,
    )
    .map_err(|error| domain_at("/lsm/basis", error))?;
    let cpqr = CpqrConfig::new(value.cpqr.abs_rank_tolerance, value.cpqr.rel_rank_tolerance)
        .map_err(|error| domain_at("/lsm/cpqr", error))?;
    LsmConfig::new(
        training_engine,
        state_variables,
        basis,
        value.itm_abs_tolerance,
        cpqr,
        value.max_matrix_elements,
    )
    .map_err(|error| domain_at("/lsm", error))
}

fn polynomial_basis_resource_limits(
    feature_count: u32,
    max_degree: u32,
) -> Result<(usize, usize), WireError> {
    let n = usize::try_from(feature_count)
        .map_err(|_| domain_at("/lsm/basis/feature_count", "feature_count exceeds usize"))?;
    let degree = usize::try_from(max_degree)
        .map_err(|_| domain_at("/lsm/basis/max_degree", "max_degree exceeds usize"))?;
    if n == 0 {
        return Err(domain_at(
            "/lsm/basis/feature_count",
            "feature_count must be positive",
        ));
    }
    if degree >= WIRE_MAX_BASIS_COLUMNS {
        return Err(domain_at(
            "/lsm/basis/max_degree",
            "polynomial basis exceeds the wire column resource limit",
        ));
    }
    let mut columns = 1_usize;
    for index in 1..=degree {
        let factor = n
            .checked_add(index)
            .ok_or_else(|| domain_at("/lsm/basis", "polynomial basis resource limit overflow"))?;
        columns = columns
            .checked_mul(factor)
            .ok_or_else(|| domain_at("/lsm/basis", "polynomial basis resource limit overflow"))?
            / index;
        if columns > WIRE_MAX_BASIS_COLUMNS {
            return Err(domain_at(
                "/lsm/basis",
                "polynomial basis exceeds the wire column resource limit",
            ));
        }
    }
    let exponents = columns
        .checked_mul(n)
        .ok_or_else(|| domain_at("/lsm/basis", "polynomial basis resource limit overflow"))?;
    if exponents > WIRE_MAX_BASIS_EXPONENTS {
        return Err(domain_at(
            "/lsm/basis",
            "polynomial basis exceeds the wire exponent resource limit",
        ));
    }
    Ok((columns, exponents))
}

fn local_volatility_from_wire(
    value: LocalVarianceGridV1,
    reporting_iv_basis: Option<ReportingIvBasisV1>,
    grid_pointer: &'static str,
    basis_pointer: &'static str,
) -> Result<LocalVolatilitySpec, WireError> {
    let expected_shape = [
        value.time_nodes.len(),
        value.log_forward_moneyness_nodes.len(),
    ];
    if value.shape != expected_shape {
        return Err(domain_at(
            grid_pointer,
            format!(
                "local_variance_grid shape {:?} does not match node dimensions {:?}",
                value.shape, expected_shape
            ),
        ));
    }
    let mut spec = LocalVolatilitySpec::from_explicit_grid(
        value.time_nodes,
        value.log_forward_moneyness_nodes,
        value.values,
        value.floor,
        value.cap,
    )
    .map_err(|error| domain_at(grid_pointer, error))?;
    if let Some(basis) = reporting_iv_basis {
        spec = spec.with_reporting_iv_basis(reporting_iv_basis_from_wire(basis, basis_pointer)?);
    }
    Ok(spec)
}

fn reporting_iv_basis_from_wire(
    value: ReportingIvBasisV1,
    pointer: &'static str,
) -> Result<LocalVolatilityReportingBasis, WireError> {
    let expected_shape = [
        value.maturity_nodes.len(),
        value.log_forward_moneyness_nodes.len(),
    ];
    if value.shape != expected_shape {
        return Err(domain_at(
            pointer,
            format!(
                "reporting_iv_basis shape {:?} does not match node dimensions {:?}",
                value.shape, expected_shape
            ),
        ));
    }
    LocalVolatilityReportingBasis::new(
        value.maturity_nodes,
        value.log_forward_moneyness_nodes,
        value.implied_volatilities,
    )
    .map_err(|error| domain_at(pointer, error))
}

fn dividend_event_from_wire(
    value: DividendEventV1,
    pointer: String,
) -> Result<DividendEvent, WireError> {
    let event = EventId::new(value.event_id);
    let quote = match value.quote {
        DividendQuoteV1::FixedCash { amount } => DividendQuote::fixed_cash(amount, event)
            .map_err(|error| domain_at(format!("{pointer}/quote/amount"), error))?,
        DividendQuoteV1::Proportional { beta } => DividendQuote::proportional(beta, event)
            .map_err(|error| domain_at(format!("{pointer}/quote/beta"), error))?,
        DividendQuoteV1::FixedCashAndProportional { fixed_cash, beta } => {
            DividendQuote::fixed_cash_and_proportional(fixed_cash, beta, event)
                .map_err(|error| domain_at(format!("{pointer}/quote"), error))?
        }
    };
    DividendEvent::new(event, value.ex_time, quote).map_err(|error| domain_at(pointer, error))
}

impl From<VarianceReductionV1> for VarianceReduction {
    fn from(value: VarianceReductionV1) -> Self {
        Self::new(value.antithetic, value.brownian_bridge)
    }
}

fn curve_from_wire(
    value: CurveV1,
    pointer: &'static str,
) -> Result<LogLinearDiscountCurve, WireError> {
    LogLinearDiscountCurve::new(
        CurveId::new(value.curve_id),
        value.times,
        value.discount_factors,
    )
    .map_err(|error| domain_at(pointer, error))
}

fn risk_from_wire(value: RiskV2) -> Result<RiskRequest, WireError> {
    let payoff_smoothing_width_ladder = value
        .payoff_smoothing_width_ladder
        .map(PayoffSmoothingWidthLadder::new)
        .transpose()
        .map_err(|error| domain_at("/risk/payoff_smoothing_width_ladder", error))?;
    let payoff_smoothing = value
        .payoff_smoothing
        .map(|smoothing| match smoothing {
            PayoffSmoothingV1::CompactC2 { half_width } => PayoffSmoothing::compact_c2(half_width),
        })
        .transpose()
        .map_err(|error| domain_at("/risk/payoff_smoothing/half_width", error))?;
    let gamma = value
        .gamma
        .map(|item| {
            let bump = match item.bump {
                SpotBumpV1::Absolute(x) => SpotBump::absolute(x),
                SpotBumpV1::Relative(x) => SpotBump::relative(x),
            }
            .map_err(|error| domain_at("/risk/gamma/bump", error))?;
            Ok::<_, WireError>(GammaConfig::new(bump))
        })
        .transpose()?;
    let vega_kt = value
        .vega_kt
        .map(|item| {
            let dates = item
                .maturity_nodes
                .iter()
                .enumerate()
                .map(|(index, date)| {
                    parse_date_owned_at(date, format!("/risk/vega_kt/maturity_nodes/{index}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            VegaKtConfig::new(
                dates,
                item.log_forward_moneyness_nodes,
                item.relative_density_threshold,
                item.full_bucket_covariance,
            )
            .map_err(|error| domain_at("/risk/vega_kt", error))
        })
        .transpose()?;
    let smile = match value.smile_dynamics {
        SmileDynamicsV1::LogMoneyness => SmileDynamics::StickyLogMoneyness,
        SmileDynamicsV1::Strike => SmileDynamics::StickyStrike,
        SmileDynamicsV1::Delta => SmileDynamics::StickyDelta,
    };
    let request = RiskRequest::new(
        value.delta,
        gamma,
        value.vega,
        vega_kt,
        smile,
        value.checkpoint_interval,
        value.aad_tile_capacity,
    )
    .map_err(|error| domain_at("/risk", error))?;
    let request = match payoff_smoothing {
        Some(smoothing) => request.with_payoff_smoothing(smoothing),
        None => request,
    };
    Ok(match payoff_smoothing_width_ladder {
        Some(ladder) => request.with_payoff_smoothing_width_ladder(ladder),
        None => request,
    })
}

fn parse_date_at(value: &str, pointer: &'static str) -> Result<Date, WireError> {
    value
        .parse::<Date>()
        .map_err(|error| domain_at(pointer, error))
}

fn parse_date_owned_at(value: &str, pointer: String) -> Result<Date, WireError> {
    value.parse::<Date>().map_err(|error| WireError::DomainAt {
        pointer,
        message: error.to_string(),
    })
}

fn domain_at(pointer: impl Into<String>, error: impl fmt::Display) -> WireError {
    WireError::DomainAt {
        pointer: pointer.into(),
        message: error.to_string(),
    }
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultV2 {
    document_kind: String,
    schema_version: u32,
    value: EstimateV1,
    risks: RiskReportV1,
    diagnostics: DiagnosticsV1,
    replay: ReplayV2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultV3 {
    document_kind: String,
    schema_version: u32,
    value: EstimateV1,
    risks: RiskReportV1,
    diagnostics: DiagnosticsV1,
    replay: ReplayV2,
    #[serde(skip_serializing_if = "Option::is_none")]
    monte_carlo: Option<MonteCarloResultV3>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MonteCarloResultV3 {
    sampling_variance: f64,
    estimator_variance: f64,
    independent_sampling_units: u64,
    evaluated_paths: String,
    diagnostics: MonteCarloDiagnosticsV3,
    risk_diagnostics: RiskDiagnosticsV3,
    #[serde(skip_serializing_if = "Option::is_none")]
    early_exercise: Option<EarlyExerciseDiagnosticsV3>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MonteCarloDiagnosticsV3 {
    master_seed: u64,
    estimator: EstimatorV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    scramble_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    direction_checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scramble_checksum: Option<String>,
    policy_version: u32,
    worker_threads: u32,
    reduction_block_size: u64,
    aad_tile_policy_version: u32,
    aad_tile_capacity: u32,
    checkpoint_policy_version: u32,
    checkpoint_interval: u32,
    antithetic: bool,
    discount_region: CurveRegionV3,
    dividend_region: CurveRegionV3,
    payoff_fingerprint: String,
    valuation_kind: PayoffValuationKindV3,
    #[serde(skip_serializing_if = "Option::is_none")]
    payoff_smoothing: Option<PayoffSmoothingDiagnosticsV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_state: Option<PathStateDiagnosticsV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    barrier_bridge: Option<BarrierBridgeDiagnosticsV3>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum CurveRegionV3 {
    Pillar,
    Interpolated,
    RightExtrapolated,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PayoffValuationKindV3 {
    ExactContractual,
    SmoothedSurrogate,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayoffSmoothingDiagnosticsV3 {
    kernel: PayoffSmoothingKernelV3,
    policy_version: u32,
    half_width: f64,
    full_transition_width: f64,
    width_unit: PayoffSmoothingWidthUnitV3,
    price_and_greeks_share_payoff: bool,
    endpoint_count: u32,
    dividend_jump_count: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PayoffSmoothingKernelV3 {
    CompactC2,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PayoffSmoothingWidthUnitV3 {
    Spot,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PathStateDiagnosticsV3 {
    ArithmeticAsian {
        known_observation_count: u32,
        unknown_observation_count: u32,
        known_weight_sum: f64,
        unknown_weight_sum: f64,
        weighted_known_fixing_sum: f64,
    },
    FixedLookback {
        past_monitoring_count: u32,
        future_monitoring_count: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        historical_extremum: Option<f64>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BarrierBridgeDiagnosticsV3 {
    abi: BarrierBridgeAbiV3,
    policy_version: u32,
    indicator_mode: BarrierHitIndicatorModeV3,
    endpoint_hit_fraction: f64,
    dividend_jump_hit_fraction: f64,
    mean_conditional_bridge_hit_weight: f64,
    mean_interval_count: f64,
    mean_finite_correction_count: f64,
    mean_zero_variance_count: f64,
    mean_survival_underflow_count: f64,
    mean_certain_survival_count: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum BarrierBridgeAbiV3 {
    ContinuousBarrierBridgeLogSurvivalV1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum BarrierHitIndicatorModeV3 {
    Exact,
    CompactC2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskDiagnosticsV3 {
    methods: RiskMethodMetadataV3,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta_validation: Option<RiskValidationV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gamma_validation: Option<RiskValidationV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vega_validation: Option<RiskValidationV3>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskMethodMetadataV3 {
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<RiskMethodV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gamma: Option<RiskMethodV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vega: Option<RiskMethodV3>,
    smile_dynamics: SmileDynamicsV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    gamma_spot_bump: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_spot_bump: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_volatility_bump: Option<f64>,
    bump_policy_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    exercise_strategy: Option<ExerciseStrategyRiskV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stopping_indices: Option<StoppingIndexRiskV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exercise_policy_fingerprint: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RiskMethodV3 {
    AadReverse,
    CentralBump,
    CentralBumpOfAadDelta,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ExerciseStrategyRiskV3 {
    FixedExerciseStrategy,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum StoppingIndexRiskV3 {
    FrozenStoppingIndices,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RiskValidationV3 {
    bump_and_revalue: EstimateV1,
    bump_minus_primary: EstimateV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EarlyExerciseDiagnosticsV3 {
    policy_fingerprint: String,
    training_random_domain: RandomDomainV3,
    valuation_random_domain: RandomDomainV3,
    #[serde(skip_serializing_if = "Option::is_none")]
    training_direction_checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    training_scramble_checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    valuation_direction_checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    valuation_scramble_checksum: Option<String>,
    training_sampling_units: u64,
    training_trajectories: u64,
    valuation_sampling_units: u64,
    valuation_trajectories: u64,
    in_sample_value: f64,
    exercise_dates: Vec<String>,
    exercise_counts: Vec<usize>,
    exercise_probabilities: Vec<f64>,
    stopping_indices: Vec<usize>,
    dividend_collisions: Vec<bool>,
    regression_diagnostics: Vec<ExerciseRegressionDiagnosticsV3>,
    policy_basis: PolynomialBasisReplayV3,
    itm_abs_tolerance: f64,
    cpqr: CpqrV3,
    max_matrix_elements: usize,
    decision_models: Vec<ExerciseDecisionModelV3>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RandomDomainV3 {
    Valuation,
    LsmTrain,
    RqmcScramble,
    Diagnostics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolynomialBasisReplayV3 {
    feature_count: u32,
    max_degree: u32,
    exponents: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExerciseRegressionDiagnosticsV3 {
    candidate_rows: usize,
    itm_rows: usize,
    feature_count: usize,
    warnings: Vec<LsmWarningV3>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum LsmWarningV3 {
    ZeroItmTrainingPaths,
    InactiveFeature { feature: usize },
    RankExcludedBasisColumn { column: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ExerciseDecisionModelV3 {
    Regression {
        model: Box<PolynomialRegressionModelV3>,
    },
    ContinueAll {
        reason: ContinueAllReasonV3,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ContinueAllReasonV3 {
    ZeroItmTrainingPaths,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolynomialRegressionModelV3 {
    basis: PolynomialBasisReplayV3,
    feature_scalings: Vec<FeatureScalingV3>,
    active_basis_columns: Vec<usize>,
    pre_excluded_basis_columns: Vec<usize>,
    pivot_order: Vec<usize>,
    diagonal_abs: Vec<f64>,
    rank_threshold: f64,
    rank: usize,
    rank_excluded_basis_columns: Vec<usize>,
    coefficients: Vec<f64>,
    residual_sum_squares: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureScalingV3 {
    mean: f64,
    population_variance: f64,
    scale: f64,
    zero_scale_threshold: f64,
    inactive: bool,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayV2 {
    schema_version: u32,
    request_fingerprint: String,
    library_version: String,
    platform: String,
    migration: MigrationProvenanceV2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MigrationProvenanceV2 {
    original_schema_version: u32,
    current_schema_version: u32,
    migration_ids: Vec<String>,
    pre_migration_fingerprint: String,
    post_migration_fingerprint: String,
}

impl From<&PricingResult> for ResultV2 {
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
            replay: ReplayV2 {
                schema_version: result.replay.schema_version().get(),
                request_fingerprint: Fingerprint(*result.replay.request_fingerprint()).to_string(),
                library_version: result.replay.library_version().to_owned(),
                platform: result.replay.platform().to_owned(),
                migration: MigrationProvenanceV2::from(result.replay.migration()),
            },
        }
    }
}

impl From<&PricingResult> for ResultV3 {
    fn from(result: &PricingResult) -> Self {
        Self {
            document_kind: DOCUMENT_RESULT.to_owned(),
            schema_version: 3,
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
            replay: ReplayV2 {
                schema_version: result.replay.schema_version().get(),
                request_fingerprint: Fingerprint(*result.replay.request_fingerprint()).to_string(),
                library_version: result.replay.library_version().to_owned(),
                platform: result.replay.platform().to_owned(),
                migration: MigrationProvenanceV2::from(result.replay.migration()),
            },
            monte_carlo: None,
        }
    }
}

impl From<&MonteCarloPrice> for ResultV3 {
    fn from(value: &MonteCarloPrice) -> Self {
        let mut result = Self::from(&value.pricing_result);
        result.monte_carlo = Some(MonteCarloResultV3::from(value));
        result
    }
}

impl From<&MonteCarloPrice> for MonteCarloResultV3 {
    fn from(value: &MonteCarloPrice) -> Self {
        Self {
            sampling_variance: value.sampling_variance,
            estimator_variance: value.estimator_variance,
            independent_sampling_units: value.independent_sampling_units,
            evaluated_paths: value.evaluated_paths.to_string(),
            diagnostics: value.diagnostics.into(),
            risk_diagnostics: RiskDiagnosticsV3::from(&value.risk_diagnostics),
            early_exercise: value
                .early_exercise_diagnostics
                .as_ref()
                .map(EarlyExerciseDiagnosticsV3::from),
        }
    }
}

impl From<MonteCarloDiagnostics> for MonteCarloDiagnosticsV3 {
    fn from(value: MonteCarloDiagnostics) -> Self {
        Self {
            master_seed: value.master_seed,
            estimator: value.estimator.into(),
            scramble_count: value.scramble_count,
            direction_checksum: value.direction_checksum.map(format_checksum),
            scramble_checksum: value.scramble_checksum.map(format_checksum),
            policy_version: value.policy_version,
            worker_threads: value.worker_threads,
            reduction_block_size: value.reduction_block_size,
            aad_tile_policy_version: value.aad_tile_policy_version,
            aad_tile_capacity: value.aad_tile_capacity,
            checkpoint_policy_version: value.checkpoint_policy_version,
            checkpoint_interval: value.checkpoint_interval,
            antithetic: value.antithetic,
            discount_region: value.discount_region.into(),
            dividend_region: value.dividend_region.into(),
            payoff_fingerprint: Fingerprint(*value.payoff_fingerprint.as_bytes()).to_string(),
            valuation_kind: value.valuation_kind.into(),
            payoff_smoothing: value.payoff_smoothing.map(Into::into),
            path_state: value.path_state.map(Into::into),
            barrier_bridge: value.barrier_bridge.map(Into::into),
        }
    }
}

impl From<pricing_market::CurveRegion> for CurveRegionV3 {
    fn from(value: pricing_market::CurveRegion) -> Self {
        match value {
            pricing_market::CurveRegion::Pillar => Self::Pillar,
            pricing_market::CurveRegion::Interpolated => Self::Interpolated,
            pricing_market::CurveRegion::RightExtrapolated => Self::RightExtrapolated,
        }
    }
}

impl From<PayoffValuationKind> for PayoffValuationKindV3 {
    fn from(value: PayoffValuationKind) -> Self {
        match value {
            PayoffValuationKind::ExactContractual => Self::ExactContractual,
            PayoffValuationKind::SmoothedSurrogate => Self::SmoothedSurrogate,
        }
    }
}

impl From<PayoffSmoothingDiagnostics> for PayoffSmoothingDiagnosticsV3 {
    fn from(value: PayoffSmoothingDiagnostics) -> Self {
        Self {
            kernel: match value.kernel {
                PayoffSmoothingKernel::CompactC2 => PayoffSmoothingKernelV3::CompactC2,
            },
            policy_version: value.policy_version,
            half_width: value.half_width.get(),
            full_transition_width: value.full_transition_width.get(),
            width_unit: match value.width_unit {
                PayoffSmoothingWidthUnit::Spot => PayoffSmoothingWidthUnitV3::Spot,
            },
            price_and_greeks_share_payoff: value.price_and_greeks_share_payoff,
            endpoint_count: value.endpoint_count,
            dividend_jump_count: value.dividend_jump_count,
        }
    }
}

impl From<PathStateDiagnostics> for PathStateDiagnosticsV3 {
    fn from(value: PathStateDiagnostics) -> Self {
        match value {
            PathStateDiagnostics::ArithmeticAsian {
                known_observation_count,
                unknown_observation_count,
                known_weight_sum,
                unknown_weight_sum,
                weighted_known_fixing_sum,
            } => Self::ArithmeticAsian {
                known_observation_count,
                unknown_observation_count,
                known_weight_sum,
                unknown_weight_sum,
                weighted_known_fixing_sum,
            },
            PathStateDiagnostics::FixedLookback {
                past_monitoring_count,
                future_monitoring_count,
                historical_extremum,
            } => Self::FixedLookback {
                past_monitoring_count,
                future_monitoring_count,
                historical_extremum,
            },
        }
    }
}

impl From<BarrierBridgeDiagnostics> for BarrierBridgeDiagnosticsV3 {
    fn from(value: BarrierBridgeDiagnostics) -> Self {
        debug_assert_eq!(value.abi, pricing_mc::BARRIER_BRIDGE_ABI);
        Self {
            abi: BarrierBridgeAbiV3::ContinuousBarrierBridgeLogSurvivalV1,
            policy_version: value.policy_version,
            indicator_mode: match value.indicator_mode {
                BarrierHitIndicatorMode::Exact => BarrierHitIndicatorModeV3::Exact,
                BarrierHitIndicatorMode::CompactC2 => BarrierHitIndicatorModeV3::CompactC2,
            },
            endpoint_hit_fraction: value.endpoint_hit_fraction,
            dividend_jump_hit_fraction: value.dividend_jump_hit_fraction,
            mean_conditional_bridge_hit_weight: value.mean_conditional_bridge_hit_weight,
            mean_interval_count: value.mean_interval_count,
            mean_finite_correction_count: value.mean_finite_correction_count,
            mean_zero_variance_count: value.mean_zero_variance_count,
            mean_survival_underflow_count: value.mean_survival_underflow_count,
            mean_certain_survival_count: value.mean_certain_survival_count,
        }
    }
}

impl From<&RiskDiagnostics> for RiskDiagnosticsV3 {
    fn from(value: &RiskDiagnostics) -> Self {
        Self {
            methods: value.methods.into(),
            delta_validation: value.delta_validation.map(Into::into),
            gamma_validation: value.gamma_validation.map(Into::into),
            vega_validation: value.vega_validation.map(Into::into),
        }
    }
}

impl From<RiskMethodMetadata> for RiskMethodMetadataV3 {
    fn from(value: RiskMethodMetadata) -> Self {
        Self {
            delta: value.delta.map(Into::into),
            gamma: value.gamma.map(Into::into),
            vega: value.vega.map(Into::into),
            smile_dynamics: value.smile_dynamics.into(),
            gamma_spot_bump: value.gamma_spot_bump,
            validation_spot_bump: value.validation_spot_bump,
            validation_volatility_bump: value.validation_volatility_bump,
            bump_policy_version: value.bump_policy_version,
            exercise_strategy: value
                .exercise_strategy
                .map(|_| ExerciseStrategyRiskV3::FixedExerciseStrategy),
            stopping_indices: value
                .stopping_indices
                .map(|_| StoppingIndexRiskV3::FrozenStoppingIndices),
            exercise_policy_fingerprint: value
                .exercise_policy_fingerprint
                .map(|fingerprint| Fingerprint(*fingerprint.as_bytes()).to_string()),
        }
    }
}

impl From<RiskMethod> for RiskMethodV3 {
    fn from(value: RiskMethod) -> Self {
        match value {
            RiskMethod::AadReverse => Self::AadReverse,
            RiskMethod::CentralBump => Self::CentralBump,
            RiskMethod::CentralBumpOfAadDelta => Self::CentralBumpOfAadDelta,
        }
    }
}

impl From<RiskValidation> for RiskValidationV3 {
    fn from(value: RiskValidation) -> Self {
        Self {
            bump_and_revalue: value.bump_and_revalue.into(),
            bump_minus_primary: value.bump_minus_primary.into(),
        }
    }
}

impl From<&EarlyExerciseDiagnostics> for EarlyExerciseDiagnosticsV3 {
    fn from(value: &EarlyExerciseDiagnostics) -> Self {
        Self {
            policy_fingerprint: Fingerprint(*value.policy_fingerprint.as_bytes()).to_string(),
            training_random_domain: value.training_random_domain.into(),
            valuation_random_domain: value.valuation_random_domain.into(),
            training_direction_checksum: value.training_direction_checksum.map(format_checksum),
            training_scramble_checksum: value.training_scramble_checksum.map(format_checksum),
            valuation_direction_checksum: value.valuation_direction_checksum.map(format_checksum),
            valuation_scramble_checksum: value.valuation_scramble_checksum.map(format_checksum),
            training_sampling_units: value.training_sampling_units,
            training_trajectories: value.training_trajectories,
            valuation_sampling_units: value.valuation_sampling_units,
            valuation_trajectories: value.valuation_trajectories,
            in_sample_value: value.in_sample_value,
            exercise_dates: value
                .exercise_dates
                .iter()
                .map(ToString::to_string)
                .collect(),
            exercise_counts: value.exercise_counts.to_vec(),
            exercise_probabilities: value.exercise_probabilities.to_vec(),
            stopping_indices: value.stopping_indices.to_vec(),
            dividend_collisions: value.dividend_collisions.to_vec(),
            regression_diagnostics: value
                .regression_diagnostics
                .iter()
                .map(ExerciseRegressionDiagnosticsV3::from)
                .collect(),
            policy_basis: PolynomialBasisReplayV3::from(&value.policy_basis),
            itm_abs_tolerance: value.itm_abs_tolerance,
            cpqr: CpqrV3 {
                abs_rank_tolerance: value.cpqr_config.abs_rank_tolerance(),
                rel_rank_tolerance: value.cpqr_config.rel_rank_tolerance(),
            },
            max_matrix_elements: value.max_matrix_elements,
            decision_models: value
                .decision_models
                .iter()
                .map(ExerciseDecisionModelV3::from)
                .collect(),
        }
    }
}

impl From<RandomDomain> for RandomDomainV3 {
    fn from(value: RandomDomain) -> Self {
        match value {
            RandomDomain::Valuation => Self::Valuation,
            RandomDomain::LsmTrain => Self::LsmTrain,
            RandomDomain::RqmcScramble => Self::RqmcScramble,
            RandomDomain::Diagnostics => Self::Diagnostics,
        }
    }
}

impl From<&PolynomialBasisSpec> for PolynomialBasisReplayV3 {
    fn from(value: &PolynomialBasisSpec) -> Self {
        Self {
            feature_count: value.feature_count(),
            max_degree: value.max_degree(),
            exponents: value.exponents().iter().map(|row| row.to_vec()).collect(),
        }
    }
}

impl From<&ExerciseRegressionDiagnostics> for ExerciseRegressionDiagnosticsV3 {
    fn from(value: &ExerciseRegressionDiagnostics) -> Self {
        Self {
            candidate_rows: value.candidate_rows(),
            itm_rows: value.itm_rows(),
            feature_count: value.feature_count(),
            warnings: value.warnings().iter().copied().map(Into::into).collect(),
        }
    }
}

impl From<LsmWarning> for LsmWarningV3 {
    fn from(value: LsmWarning) -> Self {
        match value {
            LsmWarning::ZeroItmTrainingPaths => Self::ZeroItmTrainingPaths,
            LsmWarning::InactiveFeature { feature } => Self::InactiveFeature { feature },
            LsmWarning::RankExcludedBasisColumn { column } => {
                Self::RankExcludedBasisColumn { column }
            }
        }
    }
}

impl From<&ExerciseDecisionModel> for ExerciseDecisionModelV3 {
    fn from(value: &ExerciseDecisionModel) -> Self {
        match value {
            ExerciseDecisionModel::Regression(model) => Self::Regression {
                model: Box::new(PolynomialRegressionModelV3::from(model)),
            },
            ExerciseDecisionModel::ContinueAll {
                reason: ContinueAllReason::ZeroItmTrainingPaths,
            } => Self::ContinueAll {
                reason: ContinueAllReasonV3::ZeroItmTrainingPaths,
            },
        }
    }
}

impl From<&PolynomialRegressionModel> for PolynomialRegressionModelV3 {
    fn from(value: &PolynomialRegressionModel) -> Self {
        Self {
            basis: PolynomialBasisReplayV3::from(value.basis()),
            feature_scalings: value
                .feature_scalings()
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
            active_basis_columns: value.active_basis_columns().to_vec(),
            pre_excluded_basis_columns: value.pre_excluded_basis_columns().to_vec(),
            pivot_order: value.pivot_order().to_vec(),
            diagonal_abs: value.diagonal_abs().to_vec(),
            rank_threshold: value.rank_threshold(),
            rank: value.rank(),
            rank_excluded_basis_columns: value.rank_excluded_basis_columns().to_vec(),
            coefficients: value.coefficients().to_vec(),
            residual_sum_squares: value.residual_sum_squares(),
        }
    }
}

impl From<FeatureScaling> for FeatureScalingV3 {
    fn from(value: FeatureScaling) -> Self {
        Self {
            mean: value.mean(),
            population_variance: value.population_variance(),
            scale: value.scale(),
            zero_scale_threshold: value.zero_scale_threshold(),
            inactive: value.inactive(),
        }
    }
}

fn format_checksum(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl From<&MigrationProvenance> for MigrationProvenanceV2 {
    fn from(value: &MigrationProvenance) -> Self {
        Self {
            original_schema_version: value.original_schema_version().get(),
            current_schema_version: value.current_schema_version().get(),
            migration_ids: value.migration_ids().to_vec(),
            pre_migration_fingerprint: Fingerprint(*value.pre_migration_fingerprint()).to_string(),
            post_migration_fingerprint: Fingerprint(*value.post_migration_fingerprint())
                .to_string(),
        }
    }
}

impl From<ResultV1> for ResultV2 {
    fn from(value: ResultV1) -> Self {
        let request_fingerprint = value.replay.request_fingerprint;
        Self {
            document_kind: value.document_kind,
            schema_version: 2,
            value: value.value,
            risks: value.risks,
            diagnostics: value.diagnostics,
            replay: ReplayV2 {
                schema_version: 2,
                migration: MigrationProvenanceV2 {
                    original_schema_version: 1,
                    current_schema_version: 2,
                    migration_ids: vec![MIGRATION_RESULT_V1_TO_V2.to_owned()],
                    pre_migration_fingerprint: request_fingerprint.clone(),
                    post_migration_fingerprint: request_fingerprint.clone(),
                },
                request_fingerprint,
                library_version: value.replay.library_version,
                platform: value.replay.platform,
            },
        }
    }
}

impl From<ResultV2> for ResultV3 {
    fn from(value: ResultV2) -> Self {
        debug_assert_eq!(value.schema_version, 2);
        let mut migration = value.replay.migration;
        migration.current_schema_version = 3;
        migration
            .migration_ids
            .push(MIGRATION_RESULT_V2_TO_V3.to_owned());
        Self {
            document_kind: value.document_kind,
            schema_version: 3,
            value: value.value,
            risks: value.risks,
            diagnostics: value.diagnostics,
            replay: ReplayV2 {
                schema_version: 3,
                request_fingerprint: value.replay.request_fingerprint,
                library_version: value.replay.library_version,
                platform: value.replay.platform,
                migration,
            },
            monte_carlo: None,
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

impl TryFrom<ResultV3> for PricingResult {
    type Error = WireError;
    fn try_from(value: ResultV3) -> Result<Self, Self::Error> {
        check_header(&value.document_kind, value.schema_version, DOCUMENT_RESULT)?;
        let monte_carlo = value.monte_carlo;
        let replay_version = SchemaVersion::new(value.replay.schema_version)
            .map_err(|error| domain_at("/replay/schema_version", error))?;
        if replay_version != SchemaVersion::CURRENT {
            return Err(domain_at(
                "/replay/schema_version",
                format!(
                    "unsupported replay schema_version {}; expected {}",
                    replay_version.get(),
                    SchemaVersion::CURRENT.get()
                ),
            ));
        }
        let request_fingerprint = parse_fingerprint_at(
            &value.replay.request_fingerprint,
            "/replay/request_fingerprint",
        )?;
        let migration =
            migration_provenance_from_wire(value.replay.migration, request_fingerprint)?;
        let result = Self {
            value: estimate_from_wire(value.value, "/value")?,
            risks: RiskReport {
                delta: value
                    .risks
                    .delta
                    .map(|risk| risk_estimate_from_wire(risk, "/risks/delta"))
                    .transpose()?,
                gamma: value
                    .risks
                    .gamma
                    .map(|risk| risk_estimate_from_wire(risk, "/risks/gamma"))
                    .transpose()?,
                vega: value
                    .risks
                    .vega
                    .map(|risk| risk_estimate_from_wire(risk, "/risks/vega"))
                    .transpose()?,
                vega_kt: value
                    .risks
                    .vega_kt
                    .map(|risk| vega_kt_from_wire(risk, "/risks/vega_kt"))
                    .transpose()?,
            },
            diagnostics: Diagnostics::new(
                value
                    .diagnostics
                    .warnings
                    .into_iter()
                    .map(|w| PricingWarning::new(w.code, w.message))
                    .collect(),
            ),
            replay: ReplayMetadata::with_migration(
                replay_version,
                request_fingerprint,
                non_empty_string_at(value.replay.library_version, "/replay/library_version")?,
                non_empty_string_at(value.replay.platform, "/replay/platform")?,
                migration,
            ),
        };
        if let Some(monte_carlo) = monte_carlo {
            monte_carlo_price_from_wire(monte_carlo, result.clone())?;
        }
        Ok(result)
    }
}

fn migration_provenance_from_wire(
    value: MigrationProvenanceV2,
    request_fingerprint: [u8; 32],
) -> Result<MigrationProvenance, WireError> {
    let original_schema_version = SchemaVersion::new(value.original_schema_version)
        .map_err(|error| domain_at("/replay/migration/original_schema_version", error))?;
    let current_schema_version = SchemaVersion::new(value.current_schema_version)
        .map_err(|error| domain_at("/replay/migration/current_schema_version", error))?;
    if current_schema_version != SchemaVersion::CURRENT {
        return Err(domain_at(
            "/replay/migration/current_schema_version",
            format!(
                "unsupported current_schema_version {}; expected {}",
                current_schema_version.get(),
                SchemaVersion::CURRENT.get()
            ),
        ));
    }
    if !MigrationRegistry
        .accepted_source_versions()
        .contains(&original_schema_version.get())
    {
        return Err(domain_at(
            "/replay/migration/original_schema_version",
            "original_schema_version is not accepted by the migration registry",
        ));
    }
    let identifiers_valid = match original_schema_version.get() {
        1 => {
            value.migration_ids
                == [MIGRATION_REQUEST_V1_TO_V2, MIGRATION_REQUEST_V2_TO_V3].map(str::to_owned)
                || value.migration_ids
                    == [MIGRATION_RESULT_V1_TO_V2, MIGRATION_RESULT_V2_TO_V3].map(str::to_owned)
        }
        2 => {
            value.migration_ids == [MIGRATION_REQUEST_V2_TO_V3.to_owned()]
                || value.migration_ids == [MIGRATION_RESULT_V2_TO_V3.to_owned()]
        }
        3 => value.migration_ids.is_empty(),
        _ => unreachable!("accepted versions are checked above"),
    };
    if !identifiers_valid {
        return Err(domain_at(
            "/replay/migration/migration_ids",
            "migration identifiers do not match the registered schema path",
        ));
    }
    let pre_migration_fingerprint = parse_fingerprint_at(
        &value.pre_migration_fingerprint,
        "/replay/migration/pre_migration_fingerprint",
    )?;
    let post_migration_fingerprint = parse_fingerprint_at(
        &value.post_migration_fingerprint,
        "/replay/migration/post_migration_fingerprint",
    )?;
    if post_migration_fingerprint != request_fingerprint {
        return Err(domain_at(
            "/replay/migration/post_migration_fingerprint",
            "post_migration_fingerprint must equal replay.request_fingerprint",
        ));
    }
    if original_schema_version == current_schema_version
        && pre_migration_fingerprint != post_migration_fingerprint
    {
        return Err(domain_at(
            "/replay/migration/pre_migration_fingerprint",
            "an unmigrated result must have identical pre/post fingerprints",
        ));
    }
    Ok(MigrationProvenance::new(
        original_schema_version,
        current_schema_version,
        value.migration_ids,
        pre_migration_fingerprint,
        post_migration_fingerprint,
    ))
}

fn estimate_from_wire(
    value: EstimateV1,
    pointer: impl Into<String>,
) -> Result<Estimate, WireError> {
    let pointer = pointer.into();
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
    .map_err(|error| domain_at(pointer, error))
}
fn risk_estimate_from_wire(
    value: RiskEstimateV1,
    pointer: &'static str,
) -> Result<RiskEstimate, WireError> {
    Ok(RiskEstimate::new(
        estimate_from_wire(value.raw, format!("{pointer}/raw"))?,
        estimate_from_wire(value.market_scaled, format!("{pointer}/market_scaled"))?,
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

fn vega_kt_from_wire(
    value: VegaKtReportV1,
    pointer: &'static str,
) -> Result<VegaKtResult, WireError> {
    VegaKtResult::new(
        value
            .coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinate)| {
                VegaKtResultCoordinate::new(
                    coordinate.maturity,
                    coordinate.log_moneyness,
                    coordinate.implied_volatility,
                )
                .map_err(|error| domain_at(format!("{pointer}/coordinates/{index}"), error))
            })
            .collect::<Result<Vec<_>, _>>()?,
        value
            .estimates
            .into_iter()
            .enumerate()
            .map(|(index, estimate)| {
                VegaKtResultBucketEstimate::new(
                    estimate.raw_mean,
                    estimate.market_scaled_mean,
                    estimate.sample_variance,
                    estimate.price_covariance,
                )
                .map_err(|error| domain_at(format!("{pointer}/estimates/{index}"), error))
            })
            .collect::<Result<Vec<_>, _>>()?,
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
        vega_kt_projection_from_wire(value.projection, format!("{pointer}/projection"))?,
        vega_kt_residual_diagnostics_from_wire(
            value.residual_diagnostics,
            format!("{pointer}/residual_diagnostics"),
        )?,
        vega_kt_unit_from_wire(value.raw_unit),
        vega_kt_unit_from_wire(value.market_scaled_unit),
        value.policy_label,
        value.truncation_order,
    )
    .map_err(|error| domain_at(pointer, error))
}

fn vega_kt_projection_from_wire(
    value: VegaKtProjectionV1,
    pointer: String,
) -> Result<VegaKtResultProjection, WireError> {
    let stats =
        reporting_stats_from_wire(value.reporting_stats, format!("{pointer}/reporting_stats"))?;
    VegaKtResultProjection::new(
        value.scalar_vega,
        value.signed_residual,
        value.pre_projection,
        stats,
    )
    .map_err(|error| domain_at(pointer, error))
}

fn vega_kt_residual_diagnostics_from_wire(
    value: VegaKtResidualDiagnosticsV1,
    pointer: String,
) -> Result<VegaKtResultResidualDiagnostics, WireError> {
    let stats =
        reporting_stats_from_wire(value.reporting_stats, format!("{pointer}/reporting_stats"))?;
    VegaKtResultResidualDiagnostics::new(
        value.active_domain_start_index,
        value.active_domain_end_index,
        value.active_domain_forward_index,
        value.excluded_probability_mass,
        value.signed_residual,
        value.pre_projection,
        stats,
    )
    .map_err(|error| domain_at(pointer, error))
}

fn reporting_stats_from_wire(
    value: ReportingIvProjectionStatsV1,
    pointer: String,
) -> Result<VegaKtResultReportingStats, WireError> {
    VegaKtResultReportingStats::new(
        value.left_edge_count,
        value.right_edge_count,
        value.left_edge_sensitivity,
        value.right_edge_sensitivity,
    )
    .map_err(|error| domain_at(pointer, error))
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

fn monte_carlo_price_from_wire(
    value: MonteCarloResultV3,
    pricing_result: PricingResult,
) -> Result<MonteCarloPrice, WireError> {
    let sampling_variance =
        non_negative_finite_at(value.sampling_variance, "/monte_carlo/sampling_variance")?;
    let estimator_variance =
        non_negative_finite_at(value.estimator_variance, "/monte_carlo/estimator_variance")?;
    if value.independent_sampling_units == 0 {
        return Err(domain_at(
            "/monte_carlo/independent_sampling_units",
            "independent sampling units must be positive",
        ));
    }
    if value.independent_sampling_units != pricing_result.value.effective_sampling_units().get() {
        return Err(domain_at(
            "/monte_carlo/independent_sampling_units",
            "must equal value.effective_sampling_units",
        ));
    }
    let evaluated_paths =
        parse_u128_decimal_at(&value.evaluated_paths, "/monte_carlo/evaluated_paths")?;
    let diagnostics = monte_carlo_diagnostics_from_wire(value.diagnostics)?;
    if diagnostics.estimator != pricing_result.value.estimator() {
        return Err(domain_at(
            "/monte_carlo/diagnostics/estimator",
            "must equal value.estimator",
        ));
    }
    let risk_diagnostics = risk_diagnostics_from_wire(value.risk_diagnostics)?;
    let early_exercise_diagnostics = value
        .early_exercise
        .map(early_exercise_diagnostics_from_wire)
        .transpose()?;
    if let Some(early) = &early_exercise_diagnostics {
        if diagnostics.direction_checksum != early.valuation_direction_checksum
            || diagnostics.scramble_checksum != early.valuation_scramble_checksum
        {
            return Err(domain_at(
                "/monte_carlo/early_exercise",
                "valuation checksums must match Monte Carlo diagnostics",
            ));
        }
        if let Some(risk_fingerprint) = risk_diagnostics.methods.exercise_policy_fingerprint
            && risk_fingerprint != early.policy_fingerprint
        {
            return Err(domain_at(
                "/monte_carlo/risk_diagnostics/methods/exercise_policy_fingerprint",
                "must match early-exercise policy fingerprint",
            ));
        }
    }
    Ok(MonteCarloPrice {
        pricing_result,
        sampling_variance,
        estimator_variance,
        risk_diagnostics,
        independent_sampling_units: value.independent_sampling_units,
        evaluated_paths,
        diagnostics,
        early_exercise_diagnostics,
    })
}

fn monte_carlo_diagnostics_from_wire(
    value: MonteCarloDiagnosticsV3,
) -> Result<MonteCarloDiagnostics, WireError> {
    if value.worker_threads == 0
        || value.reduction_block_size == 0
        || value.aad_tile_capacity == 0
        || value.checkpoint_interval == 0
    {
        return Err(domain_at(
            "/monte_carlo/diagnostics",
            "worker, reduction, tile, and checkpoint counts must be positive",
        ));
    }
    Ok(MonteCarloDiagnostics {
        master_seed: value.master_seed,
        estimator: value.estimator.into(),
        scramble_count: value.scramble_count,
        direction_checksum: value
            .direction_checksum
            .map(|item| parse_checksum_at(&item, "/monte_carlo/diagnostics/direction_checksum"))
            .transpose()?,
        scramble_checksum: value
            .scramble_checksum
            .map(|item| parse_checksum_at(&item, "/monte_carlo/diagnostics/scramble_checksum"))
            .transpose()?,
        policy_version: value.policy_version,
        worker_threads: value.worker_threads,
        reduction_block_size: value.reduction_block_size,
        aad_tile_policy_version: value.aad_tile_policy_version,
        aad_tile_capacity: value.aad_tile_capacity,
        checkpoint_policy_version: value.checkpoint_policy_version,
        checkpoint_interval: value.checkpoint_interval,
        antithetic: value.antithetic,
        discount_region: value.discount_region.into(),
        dividend_region: value.dividend_region.into(),
        payoff_fingerprint: pricing_product::GraphFingerprint::from_bytes(
            parse_fingerprint_owned_at(
                &value.payoff_fingerprint,
                "/monte_carlo/diagnostics/payoff_fingerprint",
            )?,
        ),
        valuation_kind: value.valuation_kind.into(),
        payoff_smoothing: value
            .payoff_smoothing
            .map(payoff_smoothing_diagnostics_from_wire)
            .transpose()?,
        path_state: value
            .path_state
            .map(path_state_diagnostics_from_wire)
            .transpose()?,
        barrier_bridge: value
            .barrier_bridge
            .map(barrier_bridge_diagnostics_from_wire)
            .transpose()?,
    })
}

impl From<EstimatorV1> for EstimatorKind {
    fn from(value: EstimatorV1) -> Self {
        match value {
            EstimatorV1::Analytical => Self::Analytical,
            EstimatorV1::PseudoMonteCarlo => Self::PseudoMonteCarlo,
            EstimatorV1::RandomizedQuasiMonteCarlo => Self::RandomizedQuasiMonteCarlo,
        }
    }
}

impl From<CurveRegionV3> for pricing_market::CurveRegion {
    fn from(value: CurveRegionV3) -> Self {
        match value {
            CurveRegionV3::Pillar => Self::Pillar,
            CurveRegionV3::Interpolated => Self::Interpolated,
            CurveRegionV3::RightExtrapolated => Self::RightExtrapolated,
        }
    }
}

impl From<PayoffValuationKindV3> for PayoffValuationKind {
    fn from(value: PayoffValuationKindV3) -> Self {
        match value {
            PayoffValuationKindV3::ExactContractual => Self::ExactContractual,
            PayoffValuationKindV3::SmoothedSurrogate => Self::SmoothedSurrogate,
        }
    }
}

fn payoff_smoothing_diagnostics_from_wire(
    value: PayoffSmoothingDiagnosticsV3,
) -> Result<PayoffSmoothingDiagnostics, WireError> {
    let half_width = PositiveF64::new(value.half_width, "half_width").map_err(|error| {
        domain_at(
            "/monte_carlo/diagnostics/payoff_smoothing/half_width",
            error,
        )
    })?;
    let full_transition_width =
        PositiveF64::new(value.full_transition_width, "full_transition_width").map_err(
            |error| {
                domain_at(
                    "/monte_carlo/diagnostics/payoff_smoothing/full_transition_width",
                    error,
                )
            },
        )?;
    if full_transition_width.get() != half_width.get() * 2.0 {
        return Err(domain_at(
            "/monte_carlo/diagnostics/payoff_smoothing/full_transition_width",
            "must equal twice half_width",
        ));
    }
    Ok(PayoffSmoothingDiagnostics {
        kernel: match value.kernel {
            PayoffSmoothingKernelV3::CompactC2 => PayoffSmoothingKernel::CompactC2,
        },
        policy_version: value.policy_version,
        half_width,
        full_transition_width,
        width_unit: match value.width_unit {
            PayoffSmoothingWidthUnitV3::Spot => PayoffSmoothingWidthUnit::Spot,
        },
        price_and_greeks_share_payoff: value.price_and_greeks_share_payoff,
        endpoint_count: value.endpoint_count,
        dividend_jump_count: value.dividend_jump_count,
    })
}

fn path_state_diagnostics_from_wire(
    value: PathStateDiagnosticsV3,
) -> Result<PathStateDiagnostics, WireError> {
    match value {
        PathStateDiagnosticsV3::ArithmeticAsian {
            known_observation_count,
            unknown_observation_count,
            known_weight_sum,
            unknown_weight_sum,
            weighted_known_fixing_sum,
        } => Ok(PathStateDiagnostics::ArithmeticAsian {
            known_observation_count,
            unknown_observation_count,
            known_weight_sum: non_negative_finite_at(
                known_weight_sum,
                "/monte_carlo/diagnostics/path_state/known_weight_sum",
            )?,
            unknown_weight_sum: non_negative_finite_at(
                unknown_weight_sum,
                "/monte_carlo/diagnostics/path_state/unknown_weight_sum",
            )?,
            weighted_known_fixing_sum: finite_at(
                weighted_known_fixing_sum,
                "/monte_carlo/diagnostics/path_state/weighted_known_fixing_sum",
            )?,
        }),
        PathStateDiagnosticsV3::FixedLookback {
            past_monitoring_count,
            future_monitoring_count,
            historical_extremum,
        } => Ok(PathStateDiagnostics::FixedLookback {
            past_monitoring_count,
            future_monitoring_count,
            historical_extremum: historical_extremum
                .map(|item| {
                    PositiveF64::new(item, "historical_extremum")
                        .map(PositiveF64::get)
                        .map_err(|error| {
                            domain_at(
                                "/monte_carlo/diagnostics/path_state/historical_extremum",
                                error,
                            )
                        })
                })
                .transpose()?,
        }),
    }
}

fn barrier_bridge_diagnostics_from_wire(
    value: BarrierBridgeDiagnosticsV3,
) -> Result<BarrierBridgeDiagnostics, WireError> {
    let fraction = |item, pointer| {
        let item = non_negative_finite_at(item, pointer)?;
        if item > 1.0 {
            return Err(domain_at(pointer, "fraction must not exceed one"));
        }
        Ok(item)
    };
    Ok(BarrierBridgeDiagnostics {
        abi: match value.abi {
            BarrierBridgeAbiV3::ContinuousBarrierBridgeLogSurvivalV1 => {
                pricing_mc::BARRIER_BRIDGE_ABI
            }
        },
        policy_version: value.policy_version,
        indicator_mode: match value.indicator_mode {
            BarrierHitIndicatorModeV3::Exact => BarrierHitIndicatorMode::Exact,
            BarrierHitIndicatorModeV3::CompactC2 => BarrierHitIndicatorMode::CompactC2,
        },
        endpoint_hit_fraction: fraction(
            value.endpoint_hit_fraction,
            "/monte_carlo/diagnostics/barrier_bridge/endpoint_hit_fraction",
        )?,
        dividend_jump_hit_fraction: fraction(
            value.dividend_jump_hit_fraction,
            "/monte_carlo/diagnostics/barrier_bridge/dividend_jump_hit_fraction",
        )?,
        mean_conditional_bridge_hit_weight: fraction(
            value.mean_conditional_bridge_hit_weight,
            "/monte_carlo/diagnostics/barrier_bridge/mean_conditional_bridge_hit_weight",
        )?,
        mean_interval_count: non_negative_finite_at(
            value.mean_interval_count,
            "/monte_carlo/diagnostics/barrier_bridge/mean_interval_count",
        )?,
        mean_finite_correction_count: non_negative_finite_at(
            value.mean_finite_correction_count,
            "/monte_carlo/diagnostics/barrier_bridge/mean_finite_correction_count",
        )?,
        mean_zero_variance_count: non_negative_finite_at(
            value.mean_zero_variance_count,
            "/monte_carlo/diagnostics/barrier_bridge/mean_zero_variance_count",
        )?,
        mean_survival_underflow_count: non_negative_finite_at(
            value.mean_survival_underflow_count,
            "/monte_carlo/diagnostics/barrier_bridge/mean_survival_underflow_count",
        )?,
        mean_certain_survival_count: non_negative_finite_at(
            value.mean_certain_survival_count,
            "/monte_carlo/diagnostics/barrier_bridge/mean_certain_survival_count",
        )?,
    })
}

fn risk_diagnostics_from_wire(value: RiskDiagnosticsV3) -> Result<RiskDiagnostics, WireError> {
    let methods = value.methods;
    for (item, pointer) in [
        (
            methods.gamma_spot_bump,
            "/monte_carlo/risk_diagnostics/methods/gamma_spot_bump",
        ),
        (
            methods.validation_spot_bump,
            "/monte_carlo/risk_diagnostics/methods/validation_spot_bump",
        ),
        (
            methods.validation_volatility_bump,
            "/monte_carlo/risk_diagnostics/methods/validation_volatility_bump",
        ),
    ] {
        if let Some(item) = item {
            PositiveF64::new(item, "risk bump").map_err(|error| domain_at(pointer, error))?;
        }
    }
    Ok(RiskDiagnostics {
        methods: RiskMethodMetadata {
            delta: methods.delta.map(Into::into),
            gamma: methods.gamma.map(Into::into),
            vega: methods.vega.map(Into::into),
            smile_dynamics: match methods.smile_dynamics {
                SmileDynamicsV1::LogMoneyness => SmileDynamics::StickyLogMoneyness,
                SmileDynamicsV1::Strike => SmileDynamics::StickyStrike,
                SmileDynamicsV1::Delta => SmileDynamics::StickyDelta,
            },
            gamma_spot_bump: methods.gamma_spot_bump,
            validation_spot_bump: methods.validation_spot_bump,
            validation_volatility_bump: methods.validation_volatility_bump,
            bump_policy_version: methods.bump_policy_version,
            exercise_strategy: methods
                .exercise_strategy
                .map(|_| ExerciseStrategyRisk::FixedExerciseStrategy),
            stopping_indices: methods
                .stopping_indices
                .map(|_| StoppingIndexRisk::FrozenStoppingIndices),
            exercise_policy_fingerprint: methods
                .exercise_policy_fingerprint
                .map(|item| {
                    parse_fingerprint_owned_at(
                        &item,
                        "/monte_carlo/risk_diagnostics/methods/exercise_policy_fingerprint",
                    )
                    .map(ExercisePolicyFingerprint::from_bytes)
                })
                .transpose()?,
        },
        delta_validation: value
            .delta_validation
            .map(|item| {
                risk_validation_from_wire(item, "/monte_carlo/risk_diagnostics/delta_validation")
            })
            .transpose()?,
        gamma_validation: value
            .gamma_validation
            .map(|item| {
                risk_validation_from_wire(item, "/monte_carlo/risk_diagnostics/gamma_validation")
            })
            .transpose()?,
        vega_validation: value
            .vega_validation
            .map(|item| {
                risk_validation_from_wire(item, "/monte_carlo/risk_diagnostics/vega_validation")
            })
            .transpose()?,
    })
}

impl From<RiskMethodV3> for RiskMethod {
    fn from(value: RiskMethodV3) -> Self {
        match value {
            RiskMethodV3::AadReverse => Self::AadReverse,
            RiskMethodV3::CentralBump => Self::CentralBump,
            RiskMethodV3::CentralBumpOfAadDelta => Self::CentralBumpOfAadDelta,
        }
    }
}

fn risk_validation_from_wire(
    value: RiskValidationV3,
    pointer: &'static str,
) -> Result<RiskValidation, WireError> {
    Ok(RiskValidation {
        bump_and_revalue: estimate_from_wire(value.bump_and_revalue, pointer)?,
        bump_minus_primary: estimate_from_wire(value.bump_minus_primary, pointer)?,
    })
}

fn early_exercise_diagnostics_from_wire(
    value: EarlyExerciseDiagnosticsV3,
) -> Result<EarlyExerciseDiagnostics, WireError> {
    let date_count = value.exercise_dates.len();
    let decision_count = date_count.saturating_sub(1);
    if date_count == 0
        || value.exercise_counts.len() != date_count
        || value.exercise_probabilities.len() != date_count
        || value.dividend_collisions.len() != date_count
        || value.regression_diagnostics.len() != decision_count
        || value.decision_models.len() != decision_count
        || value.training_sampling_units == 0
        || value.training_trajectories == 0
        || value.valuation_sampling_units == 0
        || value.valuation_trajectories == 0
        || value.stopping_indices.len() as u128 != u128::from(value.valuation_trajectories)
        || value.max_matrix_elements == 0
    {
        return Err(domain_at(
            "/monte_carlo/early_exercise",
            "inconsistent or empty LSM count arrays",
        ));
    }
    let exercise_dates = value
        .exercise_dates
        .iter()
        .enumerate()
        .map(|(index, item)| {
            parse_date_owned_at(
                item,
                format!("/monte_carlo/early_exercise/exercise_dates/{index}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if exercise_dates.windows(2).any(|dates| dates[0] >= dates[1]) {
        return Err(domain_at(
            "/monte_carlo/early_exercise/exercise_dates",
            "exercise dates must be strictly increasing",
        ));
    }
    if value
        .stopping_indices
        .iter()
        .any(|&index| index >= date_count)
    {
        return Err(domain_at(
            "/monte_carlo/early_exercise/stopping_indices",
            "stopping index is outside the exercise schedule",
        ));
    }
    let exercise_total = value
        .exercise_counts
        .iter()
        .try_fold(0_u128, |sum, &count| sum.checked_add(count as u128));
    if exercise_total != Some(u128::from(value.valuation_trajectories)) {
        return Err(domain_at(
            "/monte_carlo/early_exercise/exercise_counts",
            "exercise counts must sum to valuation trajectories",
        ));
    }
    for (index, (&count, &probability)) in value
        .exercise_counts
        .iter()
        .zip(&value.exercise_probabilities)
        .enumerate()
    {
        let expected = count as f64 / value.valuation_trajectories as f64;
        if !probability.is_finite()
            || probability < 0.0
            || probability > 1.0
            || probability != expected
        {
            return Err(domain_at(
                format!("/monte_carlo/early_exercise/exercise_probabilities/{index}"),
                "probability must equal count divided by valuation trajectories",
            ));
        }
    }
    let policy_basis = polynomial_basis_from_wire(
        value.policy_basis,
        "/monte_carlo/early_exercise/policy_basis",
    )?;
    let feature_count = policy_basis.feature_count() as usize;
    let regression_diagnostics = value
        .regression_diagnostics
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            if item.feature_count != feature_count {
                return Err(domain_at(
                    format!(
                        "/monte_carlo/early_exercise/regression_diagnostics/{index}/feature_count"
                    ),
                    "must equal policy basis feature count",
                ));
            }
            ExerciseRegressionDiagnostics::from_replay_parts(
                item.candidate_rows,
                item.itm_rows,
                item.feature_count,
                item.warnings.into_iter().map(Into::into).collect(),
            )
            .map_err(|error| {
                domain_at(
                    format!("/monte_carlo/early_exercise/regression_diagnostics/{index}"),
                    error,
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decision_models = value
        .decision_models
        .into_iter()
        .enumerate()
        .map(|(index, item)| exercise_decision_model_from_wire(item, index, &policy_basis))
        .collect::<Result<Vec<_>, _>>()?;
    let cpqr_config = CpqrConfig::new(value.cpqr.abs_rank_tolerance, value.cpqr.rel_rank_tolerance)
        .map_err(|error| domain_at("/monte_carlo/early_exercise/cpqr", error))?;
    let itm_abs_tolerance = non_negative_finite_at(
        value.itm_abs_tolerance,
        "/monte_carlo/early_exercise/itm_abs_tolerance",
    )?;
    Ok(EarlyExerciseDiagnostics {
        policy_fingerprint: ExercisePolicyFingerprint::from_bytes(parse_fingerprint_owned_at(
            &value.policy_fingerprint,
            "/monte_carlo/early_exercise/policy_fingerprint",
        )?),
        training_random_domain: value.training_random_domain.into(),
        valuation_random_domain: value.valuation_random_domain.into(),
        training_direction_checksum: optional_checksum_from_wire(
            value.training_direction_checksum,
            "/monte_carlo/early_exercise/training_direction_checksum",
        )?,
        training_scramble_checksum: optional_checksum_from_wire(
            value.training_scramble_checksum,
            "/monte_carlo/early_exercise/training_scramble_checksum",
        )?,
        valuation_direction_checksum: optional_checksum_from_wire(
            value.valuation_direction_checksum,
            "/monte_carlo/early_exercise/valuation_direction_checksum",
        )?,
        valuation_scramble_checksum: optional_checksum_from_wire(
            value.valuation_scramble_checksum,
            "/monte_carlo/early_exercise/valuation_scramble_checksum",
        )?,
        training_sampling_units: value.training_sampling_units,
        training_trajectories: value.training_trajectories,
        valuation_sampling_units: value.valuation_sampling_units,
        valuation_trajectories: value.valuation_trajectories,
        in_sample_value: finite_at(
            value.in_sample_value,
            "/monte_carlo/early_exercise/in_sample_value",
        )?,
        exercise_dates: exercise_dates.into_boxed_slice(),
        exercise_counts: value.exercise_counts.into_boxed_slice(),
        exercise_probabilities: value.exercise_probabilities.into_boxed_slice(),
        stopping_indices: value.stopping_indices.into_boxed_slice(),
        dividend_collisions: value.dividend_collisions.into_boxed_slice(),
        regression_diagnostics: regression_diagnostics.into_boxed_slice(),
        policy_basis,
        itm_abs_tolerance,
        cpqr_config,
        max_matrix_elements: value.max_matrix_elements,
        decision_models: decision_models.into_boxed_slice(),
    })
}

impl From<RandomDomainV3> for RandomDomain {
    fn from(value: RandomDomainV3) -> Self {
        match value {
            RandomDomainV3::Valuation => Self::Valuation,
            RandomDomainV3::LsmTrain => Self::LsmTrain,
            RandomDomainV3::RqmcScramble => Self::RqmcScramble,
            RandomDomainV3::Diagnostics => Self::Diagnostics,
        }
    }
}

impl From<LsmWarningV3> for LsmWarning {
    fn from(value: LsmWarningV3) -> Self {
        match value {
            LsmWarningV3::ZeroItmTrainingPaths => Self::ZeroItmTrainingPaths,
            LsmWarningV3::InactiveFeature { feature } => Self::InactiveFeature { feature },
            LsmWarningV3::RankExcludedBasisColumn { column } => {
                Self::RankExcludedBasisColumn { column }
            }
        }
    }
}

fn exercise_decision_model_from_wire(
    value: ExerciseDecisionModelV3,
    index: usize,
    policy_basis: &PolynomialBasisSpec,
) -> Result<ExerciseDecisionModel, WireError> {
    match value {
        ExerciseDecisionModelV3::ContinueAll {
            reason: ContinueAllReasonV3::ZeroItmTrainingPaths,
        } => Ok(ExerciseDecisionModel::ContinueAll {
            reason: ContinueAllReason::ZeroItmTrainingPaths,
        }),
        ExerciseDecisionModelV3::Regression { model } => {
            let model = *model;
            let basis = polynomial_basis_from_wire(
                model.basis,
                "/monte_carlo/early_exercise/decision_models/basis",
            )?;
            if &basis != policy_basis {
                return Err(domain_at(
                    format!("/monte_carlo/early_exercise/decision_models/{index}/model/basis"),
                    "must equal policy_basis",
                ));
            }
            let feature_scalings = model
                .feature_scalings
                .into_iter()
                .map(|item| {
                    FeatureScaling::from_replay_parts(
                        item.mean,
                        item.population_variance,
                        item.scale,
                        item.zero_scale_threshold,
                        item.inactive,
                    )
                    .map_err(|error| {
                        domain_at(
                            format!(
                                "/monte_carlo/early_exercise/decision_models/{index}/model/feature_scalings"
                            ),
                            error,
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            PolynomialRegressionModel::from_replay_parts(
                basis,
                feature_scalings,
                model.active_basis_columns,
                model.pre_excluded_basis_columns,
                model.pivot_order,
                model.diagonal_abs,
                model.rank_threshold,
                model.rank,
                model.rank_excluded_basis_columns,
                model.coefficients,
                model.residual_sum_squares,
            )
            .map(ExerciseDecisionModel::Regression)
            .map_err(|error| {
                domain_at(
                    format!("/monte_carlo/early_exercise/decision_models/{index}/model"),
                    error,
                )
            })
        }
    }
}

fn polynomial_basis_from_wire(
    value: PolynomialBasisReplayV3,
    pointer: &'static str,
) -> Result<PolynomialBasisSpec, WireError> {
    let basis = PolynomialBasisSpec::new(
        value.feature_count,
        value.max_degree,
        WIRE_MAX_BASIS_COLUMNS,
        WIRE_MAX_BASIS_EXPONENTS,
    )
    .map_err(|error| domain_at(pointer, error))?;
    if basis.exponents().len() != value.exponents.len()
        || basis
            .exponents()
            .iter()
            .zip(&value.exponents)
            .any(|(expected, actual)| expected.as_ref() != actual.as_slice())
    {
        return Err(domain_at(
            pointer,
            "basis exponents do not match canonical enumeration",
        ));
    }
    Ok(basis)
}

fn optional_checksum_from_wire(
    value: Option<String>,
    pointer: &'static str,
) -> Result<Option<[u8; 32]>, WireError> {
    value
        .map(|item| parse_checksum_at(&item, pointer))
        .transpose()
}

fn parse_checksum_at(value: &str, pointer: &'static str) -> Result<[u8; 32], WireError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(domain_at(
            pointer,
            "checksum must be 64 lowercase hexadecimal digits",
        ));
    }
    let prefixed = format!("blake3-256:{value}");
    parse_fingerprint(&prefixed).map_err(|error| domain_at(pointer, error))
}

fn parse_fingerprint_owned_at(value: &str, pointer: &str) -> Result<[u8; 32], WireError> {
    parse_fingerprint(value).map_err(|error| domain_at(pointer, error))
}

fn parse_u128_decimal_at(value: &str, pointer: &'static str) -> Result<u128, WireError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(domain_at(
            pointer,
            "must be a canonical unsigned decimal string",
        ));
    }
    value
        .parse::<u128>()
        .map_err(|error| domain_at(pointer, error))
}

fn finite_at(value: f64, pointer: impl Into<String>) -> Result<f64, WireError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(domain_at(pointer, "value must be finite"))
    }
}

fn non_negative_finite_at(value: f64, pointer: impl Into<String>) -> Result<f64, WireError> {
    let pointer = pointer.into();
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(domain_at(pointer, "value must be finite and non-negative"))
    }
}

pub fn request_to_json(request: &PricingRequest) -> Result<String, WireError> {
    serialize(&RequestV3::from(request), false)
}
pub fn request_to_pretty_json(request: &PricingRequest) -> Result<String, WireError> {
    serialize(&RequestV3::from(request), true)
}
pub fn result_to_json(result: &PricingResult) -> Result<String, WireError> {
    serialize(&ResultV3::from(result), false)
}
pub fn result_to_pretty_json(result: &PricingResult) -> Result<String, WireError> {
    serialize(&ResultV3::from(result), true)
}
pub fn monte_carlo_result_to_json(result: &MonteCarloPrice) -> Result<String, WireError> {
    serialize(&ResultV3::from(result), false)
}
pub fn monte_carlo_result_to_pretty_json(result: &MonteCarloPrice) -> Result<String, WireError> {
    serialize(&ResultV3::from(result), true)
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
    let version = validate_envelope(text, DOCUMENT_REQUEST)?;
    let source_fingerprint = (version != SchemaVersion::CURRENT.get())
        .then(|| {
            serde_json::from_str::<Value>(text)
                .map_err(json)
                .and_then(|value| fingerprint_request_value(&value, version))
        })
        .transpose()?;
    let current = match version {
        1 => RequestV3::from(RequestV2::from(
            serde_json::from_str::<RequestV1>(text).map_err(json)?,
        )),
        2 => RequestV3::from(serde_json::from_str::<RequestV2>(text).map_err(json)?),
        3 => serde_json::from_str::<RequestV3>(text).map_err(json)?,
        _ => return Err(WireError::UnsupportedSchemaVersion(version)),
    };
    let request = PricingRequest::try_from(current)?;
    let current_fingerprint = *fingerprint_request(&request)?.as_bytes();
    let original_schema_version =
        SchemaVersion::new(version).map_err(|error| WireError::Domain(error.to_string()))?;
    let migration_ids = match version {
        1 => vec![
            MIGRATION_REQUEST_V1_TO_V2.to_owned(),
            MIGRATION_REQUEST_V2_TO_V3.to_owned(),
        ],
        2 => vec![MIGRATION_REQUEST_V2_TO_V3.to_owned()],
        3 => Vec::new(),
        _ => unreachable!("accepted versions are checked above"),
    };
    Ok(request.with_wire_migration(MigrationProvenance::new(
        original_schema_version,
        SchemaVersion::CURRENT,
        migration_ids,
        source_fingerprint.map_or(current_fingerprint, |fingerprint| *fingerprint.as_bytes()),
        current_fingerprint,
    )))
}

pub fn parse_result_json(input: &[u8], limits: JsonLimits) -> Result<PricingResult, WireError> {
    let text = validate_and_decode(input, limits)?;
    let version = validate_envelope(text, DOCUMENT_RESULT)?;
    let current = match version {
        1 => ResultV3::from(ResultV2::from(
            serde_json::from_str::<ResultV1>(text).map_err(json)?,
        )),
        2 => ResultV3::from(serde_json::from_str::<ResultV2>(text).map_err(json)?),
        3 => serde_json::from_str::<ResultV3>(text).map_err(json)?,
        _ => return Err(WireError::UnsupportedSchemaVersion(version)),
    };
    current.try_into()
}

pub fn parse_monte_carlo_result_json(
    input: &[u8],
    limits: JsonLimits,
) -> Result<MonteCarloPrice, WireError> {
    let text = validate_and_decode(input, limits)?;
    let version = validate_envelope(text, DOCUMENT_RESULT)?;
    let mut current = match version {
        1 => ResultV3::from(ResultV2::from(
            serde_json::from_str::<ResultV1>(text).map_err(json)?,
        )),
        2 => ResultV3::from(serde_json::from_str::<ResultV2>(text).map_err(json)?),
        3 => serde_json::from_str::<ResultV3>(text).map_err(json)?,
        _ => return Err(WireError::UnsupportedSchemaVersion(version)),
    };
    let monte_carlo = current.monte_carlo.take().ok_or_else(|| {
        domain_at(
            "/monte_carlo",
            "Monte Carlo result metadata is required for this operation",
        )
    })?;
    let pricing_result = PricingResult::try_from(current)?;
    monte_carlo_price_from_wire(monte_carlo, pricing_result)
}

pub fn fingerprint_request(request: &PricingRequest) -> Result<Fingerprint, WireError> {
    let value = serde_json::to_value(RequestV3::from(request)).map_err(json)?;
    fingerprint_request_value(&value, SchemaVersion::CURRENT.get())
}

fn fingerprint_request_value(value: &Value, schema_version: u32) -> Result<Fingerprint, WireError> {
    let mut bytes = b"pricing/request\0".to_vec();
    bytes.extend_from_slice(&schema_version.to_be_bytes());
    encode_value(value, &mut bytes)?;
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
    reject_duplicate_members(text)?;
    let value: Value = serde_json::from_str(text).map_err(json)?;
    structural_limits(&value, limits)?;
    Ok(text)
}

fn reject_duplicate_members(text: &str) -> Result<(), WireError> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    DuplicateRejectingValue
        .deserialize(&mut deserializer)
        .map_err(json)?;
    deserializer.end().map_err(json)?;
    Ok(())
}

struct DuplicateRejectingValue;

impl<'de> DeserializeSeed<'de> for DuplicateRejectingValue {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateRejectingVisitor)
    }
}

struct DuplicateRejectingVisitor;

impl<'de> Visitor<'de> for DuplicateRejectingVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<(), E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        DuplicateRejectingValue.deserialize(deserializer)
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element_seed(DuplicateRejectingValue)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate object member {key:?}"
                )));
            }
            map.next_value_seed(DuplicateRejectingValue)?;
        }
        Ok(())
    }
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
                    "JSON null is not permitted by the pricing schema".to_owned(),
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

fn validate_envelope(text: &str, expected: &'static str) -> Result<u32, WireError> {
    let envelope: Envelope = serde_json::from_str(text).map_err(json)?;
    check_header(&envelope.document_kind, envelope.schema_version, expected)?;
    Ok(envelope.schema_version)
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
    if !hex
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
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

fn parse_fingerprint_at(value: &str, pointer: &'static str) -> Result<[u8; 32], WireError> {
    parse_fingerprint(value).map_err(|error| domain_at(pointer, error))
}

fn non_empty_string_at(value: String, pointer: &'static str) -> Result<String, WireError> {
    if value.is_empty() {
        return Err(domain_at(pointer, "string must not be empty"));
    }
    Ok(value)
}

#[must_use]
pub const fn current_request_schema() -> &'static str {
    include_str!("../../../schemas/v3/pricing_request.schema.json")
}
#[must_use]
pub const fn current_result_schema() -> &'static str {
    include_str!("../../../schemas/v3/pricing_result.schema.json")
}

#[cfg(test)]
mod tests {
    use pricing_mc::ExecutionPolicy;

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

    fn american_request() -> PricingRequest {
        let base = request();
        let product = ProductSpec::AmericanVanilla(
            AmericanVanillaSpec::new(
                base.product().underlying(),
                base.product().currency(),
                base.product().expiry(),
                100.0,
                1.0,
                OptionSide::Put,
                vec![
                    "2027-03-04".parse().expect("exercise date"),
                    base.product().expiry(),
                ],
            )
            .expect("American product"),
        );
        let training_engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(19, 512, VarianceReduction::new(true, false))
                .expect("training engine"),
        );
        let lsm = LsmConfig::new(
            training_engine,
            vec![LsmStateVariable::Spot],
            PolynomialBasisSpec::new(1, 2, 3, 3).expect("basis"),
            1.0e-12,
            CpqrConfig::new(1.0e-12, 1.0e-10).expect("cpqr"),
            4096,
        )
        .expect("lsm");
        PricingRequest::new_with_lsm(
            base.valuation_date(),
            product,
            base.market().clone(),
            base.model().clone(),
            base.engine(),
            base.risk().clone(),
            Some(lsm),
        )
        .expect("American request")
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

    fn barrier_request() -> PricingRequest {
        let curve = |id, discount| {
            Arc::new(
                LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                    .expect("curve"),
            )
        };
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                "2027-09-04".parse().expect("expiry"),
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Continuous,
                vec![
                    "2027-03-04".parse().expect("monitoring"),
                    "2027-09-04".parse().expect("expiry"),
                ],
                Some(3.0),
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
            include_str!("../../../fixtures/v3/pricing_request.golden.json")
        );
        assert_json_text_contract(&compact);
        let parsed = parse_request_json(compact.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(request, parsed);
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        let pretty = request_to_pretty_json(&parsed).expect("pretty");
        assert_json_text_contract(&pretty);
        let reparsed =
            parse_request_json(pretty.as_bytes(), JsonLimits::DEFAULT).expect("parse pretty");
        assert_eq!(
            fingerprint_request(&parsed).expect("fingerprint"),
            fingerprint_request(&reparsed).expect("fingerprint")
        );
    }

    #[test]
    fn schema_v1_documents_migrate_forward_and_remain_strict() {
        assert_eq!(MigrationRegistry.accepted_source_versions(), &[1, 2, 3]);
        let request_v1 = include_str!("../../../fixtures/v1/pricing_request.golden.json");
        let request = parse_request_json(request_v1.as_bytes(), JsonLimits::DEFAULT)
            .expect("migrate request");
        let request_migration = request.wire_migration().expect("request migration");
        assert_eq!(request_migration.original_schema_version().get(), 1);
        assert_eq!(request_migration.current_schema_version().get(), 3);
        assert_eq!(
            request_migration.migration_ids(),
            &[
                MIGRATION_REQUEST_V1_TO_V2.to_owned(),
                MIGRATION_REQUEST_V2_TO_V3.to_owned(),
            ]
        );
        assert_ne!(
            request_migration.pre_migration_fingerprint(),
            request_migration.post_migration_fingerprint()
        );
        let migrated_request_result = crate::SimulationPlan::compile(
            &request,
            ExecutionPolicy::new(1, Some(256)).expect("policy"),
        )
        .expect("plan")
        .execute()
        .expect("result");
        assert_eq!(
            migrated_request_result.pricing_result.replay.migration(),
            request_migration
        );
        assert_eq!(
            request_to_json(&request).expect("current request"),
            include_str!("../../../fixtures/v3/pricing_request.golden.json")
        );
        let invalid_v1 = request_v1.replacen(
            "\"smile_dynamics\"",
            "\"payoff_smoothing_width_ladder\":[2.0,1.0],\"smile_dynamics\"",
            1,
        );
        assert!(parse_request_json(invalid_v1.as_bytes(), JsonLimits::DEFAULT).is_err());

        let result_v1 = include_str!("../../../fixtures/v1/pricing_result.golden.json");
        let result =
            parse_result_json(result_v1.as_bytes(), JsonLimits::DEFAULT).expect("migrate result");
        assert_eq!(
            result_to_json(&result).expect("current result"),
            include_str!("../../../fixtures/v3/pricing_result_v1_migrated.golden.json")
        );
        assert_eq!(result.replay.schema_version(), SchemaVersion::CURRENT);
        assert_eq!(
            result.replay.migration().migration_ids(),
            &[
                MIGRATION_RESULT_V1_TO_V2.to_owned(),
                MIGRATION_RESULT_V2_TO_V3.to_owned(),
            ]
        );
    }

    #[test]
    fn schema_v2_documents_migrate_to_v3_with_adjacent_provenance() {
        let request_v2 = include_str!("../../../fixtures/v2/pricing_request.golden.json");
        let request =
            parse_request_json(request_v2.as_bytes(), JsonLimits::DEFAULT).expect("request");
        let migration = request.wire_migration().expect("migration");
        assert_eq!(migration.original_schema_version().get(), 2);
        assert_eq!(migration.current_schema_version().get(), 3);
        assert_eq!(
            migration.migration_ids(),
            &[MIGRATION_REQUEST_V2_TO_V3.to_owned()]
        );
        assert_ne!(
            migration.pre_migration_fingerprint(),
            migration.post_migration_fingerprint()
        );
        assert_eq!(
            request_to_json(&request).expect("current request"),
            include_str!("../../../fixtures/v3/pricing_request.golden.json")
        );

        let result_v2 = include_str!("../../../fixtures/v2/pricing_result.golden.json");
        let result = parse_result_json(result_v2.as_bytes(), JsonLimits::DEFAULT).expect("result");
        assert_eq!(result.replay.migration().original_schema_version().get(), 2);
        assert_eq!(result.replay.migration().current_schema_version().get(), 3);
        assert_eq!(
            result.replay.migration().migration_ids(),
            &[MIGRATION_RESULT_V2_TO_V3.to_owned()]
        );
        assert_eq!(
            result_to_json(&result).expect("current result"),
            include_str!("../../../fixtures/v3/pricing_result_v2_migrated.golden.json")
        );
    }

    #[test]
    fn request_json_round_trips_american_product_and_lsm_configuration() {
        let request = american_request();
        let expected_lsm_fingerprint = request.lsm().expect("lsm").fingerprint();

        let json = request_to_json(&request).expect("json");
        assert_eq!(
            json,
            include_str!("../../../fixtures/v3/pricing_request_american.golden.json")
        );
        assert!(json.contains("\"type\":\"american_vanilla\""));
        assert!(json.contains("\"exercise_dates\":[\"2027-03-04\",\"2027-09-04\"]"));
        assert!(json.contains("\"lsm\":{"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");

        assert_eq!(request, parsed);
        assert_eq!(
            parsed.lsm().expect("parsed lsm").fingerprint(),
            expected_lsm_fingerprint
        );
        assert_eq!(request_to_json(&parsed).expect("json"), json);
        assert_eq!(
            fingerprint_request(&parsed).expect("fingerprint"),
            fingerprint_request(&request).expect("fingerprint")
        );

        let mut legacy_value: Value = serde_json::from_str(&json).expect("request value");
        legacy_value["schema_version"] = 2.into();
        legacy_value
            .as_object_mut()
            .expect("request object")
            .remove("lsm");
        assert!(
            parse_request_json(
                serde_json::to_string(&legacy_value)
                    .expect("legacy-shaped JSON")
                    .as_bytes(),
                JsonLimits::DEFAULT,
            )
            .is_err()
        );

        let oversized_basis = json.replacen("\"max_degree\":2", "\"max_degree\":4294967295", 1);
        assert!(matches!(
            parse_request_json(oversized_basis.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/lsm/basis/max_degree" && message.contains("resource limit")
        ));
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
        assert!(json.contains("\"payment_date\":\"2027-09-04\""));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert!(matches!(parsed.product(), ProductSpec::Digital(_)));
        assert_eq!(
            fingerprint_request(&request).expect("fingerprint"),
            fingerprint_request(&parsed).expect("fingerprint")
        );
        assert_eq!(request_to_json(&parsed).expect("json"), json);
    }

    #[test]
    fn request_json_round_trips_digital_payoff_smoothing() {
        let base = digital_request();
        let risk = RiskRequest::new(
            true,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(2.0).expect("smoothing"));
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            base.model().clone(),
            base.engine(),
            risk,
        )
        .expect("request");

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"payoff_smoothing\":{\"type\":\"compact_c2\",\"half_width\":2.0}"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            parsed.risk().payoff_smoothing(),
            request.risk().payoff_smoothing()
        );
        assert_eq!(fingerprint_request(&parsed), fingerprint_request(&request));
    }

    #[test]
    fn request_json_round_trips_payoff_smoothing_width_ladder() {
        let base = digital_request();
        let risk = RiskRequest::new(
            true,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(3.0).expect("primary"))
        .with_payoff_smoothing_width_ladder(
            PayoffSmoothingWidthLadder::new(vec![4.0, 2.0, 1.0]).expect("ladder"),
        );
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            base.model().clone(),
            base.engine(),
            risk,
        )
        .expect("request");

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"payoff_smoothing_width_ladder\":[4.0,2.0,1.0]"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert_eq!(
            parsed
                .risk()
                .payoff_smoothing_width_ladder()
                .expect("ladder")
                .half_widths(),
            request
                .risk()
                .payoff_smoothing_width_ladder()
                .expect("ladder")
                .half_widths()
        );
        assert_eq!(request_to_json(&parsed).expect("json"), json);
        assert_eq!(fingerprint_request(&parsed), fingerprint_request(&request));
    }

    #[test]
    fn digital_request_json_defaults_missing_payment_date_to_expiry() {
        let json = request_to_json(&digital_request()).expect("json");
        let legacy_json = json.replace(",\"payment_date\":\"2027-09-04\"", "");

        let parsed =
            parse_request_json(legacy_json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        let ProductSpec::Digital(product) = parsed.product() else {
            panic!("digital product");
        };
        assert_eq!(product.payment_date(), product.expiry());
        assert_eq!(request_to_json(&parsed).expect("json"), json);
    }

    #[test]
    fn request_json_round_trips_barrier_product() {
        let request = barrier_request();

        let json = request_to_json(&request).expect("json");
        assert!(json.contains("\"type\":\"barrier\""));
        assert!(json.contains("\"direction\":{\"type\":\"up\"}"));
        assert!(json.contains("\"style\":{\"type\":\"knock_out\"}"));
        assert!(json.contains("\"monitoring\":{\"type\":\"continuous\"}"));
        assert!(json.contains("\"rebate\":3.0"));
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        assert!(matches!(
            parsed.product(),
            ProductSpec::Barrier(product)
                if product.monitoring() == BarrierMonitoring::Continuous
        ));
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
    fn absent_optional_product_values_are_omitted_instead_of_serialized_as_null() {
        let barrier = ProductV1::Barrier {
            underlying_id: 1,
            currency_id: 2,
            expiry: "2027-09-04".to_owned(),
            strike: 100.0,
            barrier: 120.0,
            notional: 1.0,
            side: SideV1::Call,
            direction: BarrierDirectionV1::Up,
            style: BarrierStyleV1::KnockOut,
            monitoring: BarrierMonitoringV1::Discrete,
            monitoring_dates: vec!["2027-09-04".to_owned()],
            rebate: None,
            payment_date: "2027-09-04".to_owned(),
        };
        let lookback = ProductV1::FixedLookback {
            underlying_id: 1,
            currency_id: 2,
            strike: 100.0,
            notional: 1.0,
            side: SideV1::Put,
            monitoring_dates: vec!["2027-09-04".to_owned()],
            historical_extremum: None,
            payment_date: "2027-09-04".to_owned(),
        };
        let barrier_json = serde_json::to_value(barrier).expect("Barrier JSON");
        let lookback_json = serde_json::to_value(lookback).expect("Lookback JSON");
        assert!(barrier_json.get("rebate").is_none());
        assert!(lookback_json.get("historical_extremum").is_none());
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
    fn request_json_rejects_coincident_discrete_dividend_events() {
        let json = request_to_json(&dividend_request()).expect("json");
        let mut value: Value = serde_json::from_str(&json).expect("json value");
        let dividends = value
            .pointer_mut("/market/discrete_dividends")
            .and_then(Value::as_array_mut)
            .expect("dividends");
        let mut duplicate_time_event = dividends[0].clone();
        duplicate_time_event["event_id"] = Value::from(78);
        dividends.push(duplicate_time_event);
        let invalid = serde_json::to_string(&value).expect("invalid json");

        assert!(matches!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/market/discrete_dividends"
                    && message.contains("not strictly increasing")
        ));
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
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/model/reporting_iv_basis" && message.contains("reporting_iv_basis shape")
        ));

        for value in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = json.replacen(
                "\"reporting_iv_basis\":{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[2,3]",
                &format!("\"reporting_iv_basis\":{{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[{value},3]"),
                1,
            );
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted reporting_iv_basis shape value {value}"
            );
        }
    }

    #[test]
    fn request_json_rejects_mismatched_local_volatility_grid_shape() {
        let json = request_to_json(&local_vol_request()).expect("json");
        let invalid = json.replacen("\"shape\":[2,3]", "\"shape\":[3,2]", 1);
        assert!(matches!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/model/local_variance_grid" && message.contains("shape")
        ));

        for value in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = json.replacen("\"shape\":[2,3]", &format!("\"shape\":[{value},3]"), 1);
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted local_variance_grid shape value {value}"
            );
        }
    }

    #[test]
    fn request_json_domain_date_errors_include_instance_path() {
        let json = request_to_json(&request()).expect("json");
        let invalid = json.replacen(
            "\"valuation_date\":\"2026-09-04\"",
            "\"valuation_date\":\"2026-02-31\"",
            1,
        );
        assert!(matches!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/valuation_date" && message.contains("invalid date")
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
        let future = json.replacen("\"schema_version\":3", "\"schema_version\":4", 1);
        assert!(matches!(
            parse_request_json(future.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::UnsupportedSchemaVersion(4))
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
        for constant in ["NaN", "Infinity", "-Infinity"] {
            let invalid = json.replacen("100.0", constant, 1);
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted {constant}"
            );
        }
        for schema_version in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = json.replacen(
                "\"schema_version\":3",
                &format!("\"schema_version\":{schema_version}"),
                1,
            );
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted schema_version {schema_version}"
            );
        }
        for (field, original) in [
            ("underlying_id", "\"underlying_id\":1"),
            ("curve_id", "\"curve_id\":10"),
            ("master_seed", "\"master_seed\":7"),
            (
                "independent_sampling_units",
                "\"independent_sampling_units\":1024",
            ),
        ] {
            for value in ["1.0", "1e0", "-0", "\"1\""] {
                let invalid = json.replacen(original, &format!("\"{field}\":{value}"), 1);
                assert!(
                    parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                    "request JSON accepted {field} {value}"
                );
            }
        }

        let rqmc_request = PricingRequest::new(
            request().valuation_date(),
            request().product().clone(),
            request().market().clone(),
            request().model().clone(),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(256, 4, 11, VarianceReduction::new(true, false))
                    .expect("rqmc engine"),
            ),
            request().risk().clone(),
        )
        .expect("rqmc request");
        let rqmc_json = request_to_json(&rqmc_request).expect("rqmc json");
        for (field, original) in [
            ("points_per_scramble", "\"points_per_scramble\":256"),
            ("scramble_count", "\"scramble_count\":4"),
            ("master_scramble_seed", "\"master_scramble_seed\":11"),
        ] {
            for value in ["1.0", "1e0", "-0", "\"1\""] {
                let invalid = rqmc_json.replacen(original, &format!("\"{field}\":{value}"), 1);
                assert!(
                    parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                    "request JSON accepted {field} {value}"
                );
            }
        }

        let risk_json = request_to_json(&local_vol_vega_kt_request()).expect("risk json");
        for (field, original) in [
            ("checkpoint_interval", "\"checkpoint_interval\":16"),
            ("aad_tile_capacity", "\"aad_tile_capacity\":256"),
        ] {
            for value in ["1.0", "1e0", "-0", "\"1\""] {
                let invalid = risk_json.replacen(original, &format!("\"{field}\":{value}"), 1);
                assert!(
                    parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                    "request JSON accepted {field} {value}"
                );
            }
        }
    }

    #[test]
    fn strict_reader_rejects_malformed_tagged_union_types() {
        let request_json = request_to_json(&request()).expect("request json");
        for (name, invalid) in [
            (
                "numeric document kind",
                request_json.replacen(
                    "\"document_kind\":\"pricing_request\"",
                    "\"document_kind\":1",
                    1,
                ),
            ),
            (
                "unknown document kind",
                request_json.replacen(
                    "\"document_kind\":\"pricing_request\"",
                    "\"document_kind\":\"PricingRequest\"",
                    1,
                ),
            ),
            (
                "missing product type",
                request_json.replacen("\"type\":\"european_vanilla\",", "", 1),
            ),
            (
                "numeric product type",
                request_json.replacen("\"type\":\"european_vanilla\"", "\"type\":1", 1),
            ),
            (
                "unknown product type",
                request_json.replacen(
                    "\"type\":\"european_vanilla\"",
                    "\"type\":\"EuropeanVanilla\"",
                    1,
                ),
            ),
            (
                "bare side variant",
                request_json.replacen("\"side\":{\"type\":\"call\"}", "\"side\":\"call\"", 1),
            ),
        ] {
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted {name}"
            );
        }

        let result_json = include_str!("../../../fixtures/v1/pricing_result.golden.json");
        for (name, invalid) in [
            (
                "numeric document kind",
                result_json.replacen(
                    "\"document_kind\":\"pricing_result\"",
                    "\"document_kind\":1",
                    1,
                ),
            ),
            (
                "unknown document kind",
                result_json.replacen(
                    "\"document_kind\":\"pricing_result\"",
                    "\"document_kind\":\"PricingResult\"",
                    1,
                ),
            ),
            (
                "missing estimator type",
                result_json.replacen(
                    "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                    "\"estimator\":{}",
                    1,
                ),
            ),
            (
                "numeric estimator type",
                result_json.replacen(
                    "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                    "\"estimator\":{\"type\":1}",
                    1,
                ),
            ),
            (
                "unknown estimator type",
                result_json.replacen(
                    "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                    "\"estimator\":{\"type\":\"PseudoMonteCarlo\"}",
                    1,
                ),
            ),
            (
                "bare estimator variant",
                result_json.replacen(
                    "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                    "\"estimator\":\"pseudo_monte_carlo\"",
                    1,
                ),
            ),
        ] {
            assert!(
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "result JSON accepted {name}"
            );
        }
    }

    #[test]
    fn strict_reader_rejects_duplicate_object_members_recursively() {
        let request_json = request_to_json(&request()).expect("request json");
        let duplicate_request_root = request_json.replacen(
            "\"valuation_date\"",
            "\"schema_version\":3,\"valuation_date\"",
            1,
        );
        let duplicate_request_nested =
            request_json.replacen("\"spot\":100.0", "\"spot\":100.0,\"spot\":101.0", 1);
        for invalid in [duplicate_request_root, duplicate_request_nested] {
            assert!(
                matches!(
                    parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
                    Err(WireError::Json(message)) if message.contains("duplicate object member")
                ),
                "request JSON accepted a duplicate object member"
            );
        }

        let result_json = include_str!("../../../fixtures/v1/pricing_result.golden.json");
        let duplicate_result_root =
            result_json.replacen("\"value\"", "\"schema_version\":1,\"value\"", 1);
        let duplicate_result_nested = result_json.replacen(
            "\"standard_error\":0.5",
            "\"standard_error\":0.5,\"standard_error\":0.6",
            1,
        );
        for invalid in [duplicate_result_root, duplicate_result_nested] {
            assert!(
                matches!(
                    parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT),
                    Err(WireError::Json(message)) if message.contains("duplicate object member")
                ),
                "result JSON accepted a duplicate object member"
            );
        }
    }

    #[test]
    fn strict_reader_rejects_bom_comments_and_trailing_tokens() {
        let request_json = request_to_json(&request()).expect("request json");
        let result_json = include_str!("../../../fixtures/v1/pricing_result.golden.json");
        let cases = [
            ("\u{feff}".to_owned() + &request_json, "request BOM"),
            (
                request_json.replacen("{", "{// comment\n", 1),
                "request comment",
            ),
            (request_json.clone() + "{}", "request trailing token"),
            ("\u{feff}".to_owned() + result_json, "result BOM"),
            (
                result_json.replacen("{", "{// comment\n", 1),
                "result comment",
            ),
            (result_json.to_owned() + "{}", "result trailing token"),
        ];

        assert!(parse_request_json(&[0xff], JsonLimits::DEFAULT).is_err());
        assert!(parse_result_json(&[0xff], JsonLimits::DEFAULT).is_err());

        for (invalid, name) in cases {
            let rejected = if name.starts_with("request") {
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err()
            } else {
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err()
            };
            assert!(rejected, "{name} was accepted");
        }
    }

    #[test]
    fn json_limits_report_stable_resource_names() {
        let cases = [
            (
                br#"{"document_kind":"pricing_request"}"#.as_slice(),
                JsonLimits {
                    max_string_bytes: 8,
                    ..JsonLimits::DEFAULT
                },
                "string_bytes",
                13,
                8,
            ),
            (
                b"1234".as_slice(),
                JsonLimits {
                    max_number_token_bytes: 3,
                    ..JsonLimits::DEFAULT
                },
                "number_token_bytes",
                4,
                3,
            ),
            (
                b"[[[]]]".as_slice(),
                JsonLimits {
                    max_nesting_depth: 2,
                    ..JsonLimits::DEFAULT
                },
                "nesting_depth",
                3,
                2,
            ),
            (
                b"[1,2]".as_slice(),
                JsonLimits {
                    max_array_elements: 1,
                    ..JsonLimits::DEFAULT
                },
                "array_elements",
                2,
                1,
            ),
            (
                br#"{"a":1,"b":2}"#.as_slice(),
                JsonLimits {
                    max_object_members: 1,
                    ..JsonLimits::DEFAULT
                },
                "object_members",
                2,
                1,
            ),
            (
                b"[1]".as_slice(),
                JsonLimits {
                    max_total_values: 1,
                    ..JsonLimits::DEFAULT
                },
                "total_values",
                2,
                1,
            ),
        ];

        for (input, limits, expected_name, expected_observed, expected_limit) in cases {
            assert!(
                matches!(
                    parse_request_json(input, limits),
                    Err(WireError::ResourceLimit { name, observed, limit })
                        if name == expected_name
                            && observed == expected_observed
                            && limit == expected_limit
                ),
                "expected request resource limit {expected_name}"
            );
            assert!(
                matches!(
                    parse_result_json(input, limits),
                    Err(WireError::ResourceLimit { name, observed, limit })
                        if name == expected_name
                            && observed == expected_observed
                            && limit == expected_limit
                ),
                "expected result resource limit {expected_name}"
            );
        }
    }

    #[test]
    fn json_limits_have_stable_default_and_hard_cap_values() {
        assert_eq!(
            JsonLimits::DEFAULT,
            JsonLimits {
                max_input_bytes: 4 * 1024 * 1024,
                max_nesting_depth: 64,
                max_string_bytes: 1024 * 1024,
                max_number_token_bytes: 128,
                max_array_elements: 100_000,
                max_object_members: 10_000,
                max_total_values: 250_000,
            }
        );
        assert_eq!(
            JsonLimits::HARD_CAP,
            JsonLimits {
                max_input_bytes: 64 * 1024 * 1024,
                max_nesting_depth: 256,
                max_string_bytes: 16 * 1024 * 1024,
                max_number_token_bytes: 1024,
                max_array_elements: 2_000_000,
                max_object_members: 250_000,
                max_total_values: 4_000_000,
            }
        );
        assert_eq!(JsonLimits::default(), JsonLimits::DEFAULT);
    }

    #[test]
    fn json_limit_overrides_reject_each_hard_cap_excess() {
        let hard = JsonLimits::HARD_CAP;
        assert_eq!(hard.checked(), Ok(hard));

        let cases = [
            JsonLimits {
                max_input_bytes: hard.max_input_bytes + 1,
                ..hard
            },
            JsonLimits {
                max_nesting_depth: hard.max_nesting_depth + 1,
                ..hard
            },
            JsonLimits {
                max_string_bytes: hard.max_string_bytes + 1,
                ..hard
            },
            JsonLimits {
                max_number_token_bytes: hard.max_number_token_bytes + 1,
                ..hard
            },
            JsonLimits {
                max_array_elements: hard.max_array_elements + 1,
                ..hard
            },
            JsonLimits {
                max_object_members: hard.max_object_members + 1,
                ..hard
            },
            JsonLimits {
                max_total_values: hard.max_total_values + 1,
                ..hard
            },
        ];

        for limits in cases {
            assert_eq!(
                limits.checked(),
                Err(WireError::LimitOverrideExceedsHardCap)
            );
        }
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
            include_str!("../../../fixtures/v3/pricing_result.golden.json")
        );
        assert_json_text_contract(&json);
        assert_json_text_contract(&result_to_pretty_json(&result).expect("pretty json"));
        assert_eq!(
            parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("round trip"),
            result
        );
    }

    #[test]
    fn monte_carlo_result_round_trips_complete_american_lsm_replay_state() {
        let result = crate::price_monte_carlo(
            &american_request(),
            ExecutionPolicy::new(2, Some(256)).expect("execution policy"),
        )
        .expect("American result");
        let json = monte_carlo_result_to_json(&result).expect("JSON");
        if std::env::var_os("UPDATE_AMERICAN_RESULT_GOLDEN").is_some() {
            std::fs::write(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../fixtures/v3/pricing_result_american.golden.json"
                ),
                &json,
            )
            .expect("write American result golden");
        }
        assert_eq!(
            json,
            include_str!("../../../fixtures/v3/pricing_result_american.golden.json")
        );
        assert_json_text_contract(&json);
        assert!(json.contains("\"early_exercise\""));
        assert!(json.contains("\"decision_models\""));
        assert_eq!(
            parse_monte_carlo_result_json(json.as_bytes(), JsonLimits::DEFAULT)
                .expect("round trip"),
            result
        );
        assert_eq!(
            parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("core result"),
            result.pricing_result
        );

        let mut invalid: Value = serde_json::from_str(&json).expect("JSON value");
        invalid["monte_carlo"]["early_exercise"]["stopping_indices"][0] = serde_json::json!(999);
        assert!(matches!(
            parse_monte_carlo_result_json(
                serde_json::to_string(&invalid).expect("JSON").as_bytes(),
                JsonLimits::DEFAULT,
            ),
            Err(WireError::DomainAt { pointer, .. })
                if pointer == "/monte_carlo/early_exercise/stopping_indices"
        ));
        assert!(
            parse_result_json(
                serde_json::to_string(&invalid).expect("JSON").as_bytes(),
                JsonLimits::DEFAULT,
            )
            .is_err()
        );

        let mut unknown: Value = serde_json::from_str(&json).expect("JSON value");
        unknown["monte_carlo"]["early_exercise"]["unknown"] = serde_json::json!(true);
        assert!(
            parse_monte_carlo_result_json(
                serde_json::to_string(&unknown).expect("JSON").as_bytes(),
                JsonLimits::DEFAULT,
            )
            .is_err()
        );
    }

    fn assert_json_text_contract(json: &str) {
        let without_final_lf = json.strip_suffix('\n').expect("final newline");
        assert!(!without_final_lf.ends_with('\n'));
        assert!(!json.as_bytes().starts_with(&[0xef, 0xbb, 0xbf]));
        assert!(!json.contains("\r\n"));
    }

    #[test]
    fn result_json_domain_errors_include_instance_paths() {
        let json = include_str!("../../../fixtures/v3/pricing_result.golden.json");
        let invalid_estimate =
            json.replacen("\"standard_error\":0.5", "\"standard_error\":-0.5", 1);
        assert!(matches!(
            parse_result_json(invalid_estimate.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/value" && message.contains("standard_error")
        ));

        let invalid_fingerprint = json.replacen(
            "\"request_fingerprint\":\"blake3-256:0000000000000000000000000000000000000000000000000000000000000000\"",
            "\"request_fingerprint\":\"not-a-fingerprint\"",
            1,
        );
        assert!(matches!(
            parse_result_json(invalid_fingerprint.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/replay/request_fingerprint" && message.contains("fingerprint")
        ));
        let uppercase_fingerprint = json.replacen(
            "\"request_fingerprint\":\"blake3-256:0000000000000000000000000000000000000000000000000000000000000000\"",
            "\"request_fingerprint\":\"blake3-256:ABCDEF0000000000000000000000000000000000000000000000000000000000\"",
            1,
        );
        assert!(matches!(
            parse_result_json(uppercase_fingerprint.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/replay/request_fingerprint" && message.contains("fingerprint")
        ));

        let future_replay_schema = json.replace(
            "\"replay\":{\"schema_version\":3",
            "\"replay\":{\"schema_version\":4",
        );
        assert!(matches!(
            parse_result_json(future_replay_schema.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/replay/schema_version"
                    && message.contains("unsupported replay schema_version 4")
        ));

        for schema_version in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid_top_level = json.replacen(
                "\"schema_version\":3",
                &format!("\"schema_version\":{schema_version}"),
                1,
            );
            assert!(
                parse_result_json(invalid_top_level.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "result JSON accepted schema_version {schema_version}"
            );
            let invalid_replay = json.replacen(
                "\"replay\":{\"schema_version\":3",
                &format!("\"replay\":{{\"schema_version\":{schema_version}"),
                1,
            );
            assert!(
                parse_result_json(invalid_replay.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "result JSON accepted replay schema_version {schema_version}"
            );
        }

        for effective_sampling_units in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = json.replacen(
                "\"effective_sampling_units\":1024",
                &format!("\"effective_sampling_units\":{effective_sampling_units}"),
                1,
            );
            assert!(
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "result JSON accepted effective_sampling_units {effective_sampling_units}"
            );
        }

        for (field, original, pointer) in [
            ("library_version", "0.1.0", "/replay/library_version"),
            ("platform", "acceptance-test", "/replay/platform"),
        ] {
            let invalid = json.replacen(
                &format!("\"{field}\":\"{original}\""),
                &format!("\"{field}\":\"\""),
                1,
            );
            assert!(matches!(
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT),
                Err(WireError::DomainAt { pointer: actual, message })
                    if actual == pointer && message.contains("empty")
            ));
        }

        for constant in ["NaN", "Infinity", "-Infinity"] {
            let invalid = json.replacen("10.0", constant, 1);
            assert!(
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "result JSON accepted {constant}"
            );
        }
    }

    #[test]
    fn result_json_rejects_inconsistent_migration_provenance() {
        let json = include_str!("../../../fixtures/v3/pricing_result.golden.json");

        let mut wrong_ids: serde_json::Value = serde_json::from_str(json).expect("fixture");
        wrong_ids["replay"]["migration"]["original_schema_version"] = 1.into();
        assert!(matches!(
            parse_result_json(
                serde_json::to_string(&wrong_ids).expect("json").as_bytes(),
                JsonLimits::DEFAULT
            ),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/replay/migration/migration_ids"
                    && message.contains("schema path")
        ));

        let mut wrong_post: serde_json::Value = serde_json::from_str(json).expect("fixture");
        wrong_post["replay"]["migration"]["post_migration_fingerprint"] =
            "blake3-256:1111111111111111111111111111111111111111111111111111111111111111".into();
        assert!(matches!(
            parse_result_json(
                serde_json::to_string(&wrong_post).expect("json").as_bytes(),
                JsonLimits::DEFAULT
            ),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/replay/migration/post_migration_fingerprint"
                    && message.contains("request_fingerprint")
        ));

        let mut wrong_pre: serde_json::Value = serde_json::from_str(json).expect("fixture");
        wrong_pre["replay"]["migration"]["pre_migration_fingerprint"] =
            "blake3-256:1111111111111111111111111111111111111111111111111111111111111111".into();
        assert!(matches!(
            parse_result_json(
                serde_json::to_string(&wrong_pre).expect("json").as_bytes(),
                JsonLimits::DEFAULT
            ),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/replay/migration/pre_migration_fingerprint"
                    && message.contains("identical")
        ));
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

        for (field, original) in [
            ("active_domain_start_index", "1"),
            ("active_domain_end_index", "2"),
            ("active_domain_forward_index", "1"),
            ("left_edge_count", "1"),
            ("right_edge_count", "2"),
        ] {
            for value in ["1.0", "1e0", "-0", "\"1\""] {
                let invalid = json.replacen(
                    &format!("\"{field}\":{original}"),
                    &format!("\"{field}\":{value}"),
                    1,
                );
                assert!(
                    parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                    "result JSON accepted {field} {value}"
                );
            }
        }

        let parsed = parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        let parsed_vega_kt = parsed.risks.vega_kt.expect("vega kt");
        assert_eq!(parsed_vega_kt.coordinates().len(), 2);
        assert_eq!(parsed_vega_kt.estimates()[1].sample_variance(), None);
        assert_eq!(
            parsed_vega_kt.full_bucket_covariance().expect("covariance")[1],
            None
        );
        assert_eq!(parsed_vega_kt.projection().scalar_vega().get(), 3.0);

        let mut missing_full_covariance: Value = serde_json::from_str(&json).expect("result JSON");
        missing_full_covariance["risks"]["vega_kt"]
            .as_object_mut()
            .expect("vega kt")
            .remove("full_bucket_covariance");
        let missing_full_covariance = serde_json::to_vec(&missing_full_covariance).expect("JSON");
        assert!(matches!(
            parse_result_json(&missing_full_covariance, JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/risks/vega_kt"
                    && message.contains("requires full_bucket_covariance")
        ));

        let mut unexpected_full_covariance: Value =
            serde_json::from_str(&json).expect("result JSON");
        unexpected_full_covariance["risks"]["vega_kt"]["covariance_layout"] =
            serde_json::json!({"type": "price_and_bucket_variance_only"});
        let unexpected_full_covariance =
            serde_json::to_vec(&unexpected_full_covariance).expect("JSON");
        assert!(matches!(
            parse_result_json(&unexpected_full_covariance, JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer, message })
                if pointer == "/risks/vega_kt"
                    && message.contains("must not include full_bucket_covariance")
        ));
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
