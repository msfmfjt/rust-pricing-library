use std::error::Error;
use std::fmt;

use pricing_aad::AadConfigError;
use pricing_core::{CoreError, CurrencyId, Date, UnderlyingId};
use pricing_market::MarketError;
use pricing_mc::{
    ExecutionError, ExecutorBuildError, LocalVolError, RqmcPlanError, TryExecutionError,
};
use pricing_product::GraphError;
use pricing_risk::RiskConfigError;

use crate::WireError;

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RequestValidationError {
    CurrencyMismatch {
        product: CurrencyId,
        market: CurrencyId,
    },
    UnderlyingMismatch {
        product: UnderlyingId,
        market: UnderlyingId,
    },
    ExpiryBeforeValuation {
        valuation_date: Date,
        expiry: Date,
    },
    AsianPastObservationRequiresKnownFixing {
        observation_date: Date,
        valuation_date: Date,
    },
    AsianFutureObservationCannotCarryFixing {
        observation_date: Date,
        valuation_date: Date,
    },
    VegaKtUnsupportedForConstantVolatility,
    RiskUnsupportedForDiscontinuousProduct,
}

impl fmt::Display for RequestValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrencyMismatch { product, market } => write!(
                formatter,
                "product currency {product} does not match market currency {market}"
            ),
            Self::UnderlyingMismatch { product, market } => write!(
                formatter,
                "product underlying {product} does not match market underlying {market}"
            ),
            Self::ExpiryBeforeValuation {
                valuation_date,
                expiry,
            } => write!(
                formatter,
                "expiry {expiry} is before valuation date {valuation_date}"
            ),
            Self::AsianPastObservationRequiresKnownFixing {
                observation_date,
                valuation_date,
            } => write!(
                formatter,
                "Asian observation {observation_date} before valuation date {valuation_date} requires a known fixing"
            ),
            Self::AsianFutureObservationCannotCarryFixing {
                observation_date,
                valuation_date,
            } => write!(
                formatter,
                "Asian observation {observation_date} after valuation date {valuation_date} cannot carry a known fixing"
            ),
            Self::VegaKtUnsupportedForConstantVolatility => {
                write!(formatter, "VegaKT requires a Local Volatility model")
            }
            Self::RiskUnsupportedForDiscontinuousProduct => write!(
                formatter,
                "Delta, Gamma, Vega, and VegaKT require a smooth product payoff"
            ),
        }
    }
}

impl Error for RequestValidationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResultBuildError {
    Core(CoreError),
    InvalidConfidenceInterval {
        lower_bits: u64,
        value_bits: u64,
        upper_bits: u64,
    },
    VegaKtLengthMismatch {
        coordinates: usize,
        estimates: usize,
        raw_buckets: usize,
    },
    VegaKtFullCovarianceLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ZeroEffectiveSamplingUnits,
}

impl From<CoreError> for ResultBuildError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl fmt::Display for ResultBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => error.fmt(formatter),
            Self::InvalidConfidenceInterval {
                lower_bits,
                value_bits,
                upper_bits,
            } => write!(
                formatter,
                "confidence interval must satisfy lower <= value <= upper; received 0x{lower_bits:016x}, 0x{value_bits:016x}, 0x{upper_bits:016x}"
            ),
            Self::VegaKtLengthMismatch {
                coordinates,
                estimates,
                raw_buckets,
            } => write!(
                formatter,
                "VegaKT result lengths must match; coordinates={coordinates}, estimates={estimates}, raw_buckets={raw_buckets}"
            ),
            Self::VegaKtFullCovarianceLengthMismatch { expected, actual } => write!(
                formatter,
                "VegaKT full covariance length must be {expected}; received {actual}"
            ),
            Self::ZeroEffectiveSamplingUnits => {
                write!(formatter, "effective sampling-unit count must be positive")
            }
        }
    }
}

impl Error for ResultBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum MonteCarloError {
    UnsupportedEngine,
    UnsupportedModel {
        model: &'static str,
    },
    UnsupportedRiskForModel {
        model: &'static str,
    },
    InvalidLocalVolatilityTimeGrid {
        expiry_bits: u64,
        first_bits: u64,
        last_bits: u64,
    },
    MissingLocalVolatilityReportingBasis,
    MismatchedLocalVolatilityReportingBasis,
    InvalidGammaBump {
        spot_bits: u64,
        bump_bits: u64,
    },
    InsufficientSamplingUnits {
        count: u64,
    },
    NonFiniteTotalVariance {
        bits: u64,
    },
    UnsupportedObservationUnderlying {
        product: UnderlyingId,
        market: UnderlyingId,
    },
    Market(MarketError),
    LocalVol(LocalVolError),
    Graph(GraphError),
    ExecutorBuild(ExecutorBuildError),
    Execution(ExecutionError),
    ResultBuild(ResultBuildError),
    Wire(WireError),
    AadConfig(AadConfigError),
    RqmcPlan(RqmcPlanError),
    RiskConfig(RiskConfigError),
}

impl From<MarketError> for MonteCarloError {
    fn from(error: MarketError) -> Self {
        Self::Market(error)
    }
}

impl From<LocalVolError> for MonteCarloError {
    fn from(error: LocalVolError) -> Self {
        Self::LocalVol(error)
    }
}

impl From<GraphError> for MonteCarloError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<ExecutorBuildError> for MonteCarloError {
    fn from(error: ExecutorBuildError) -> Self {
        Self::ExecutorBuild(error)
    }
}

impl From<ResultBuildError> for MonteCarloError {
    fn from(error: ResultBuildError) -> Self {
        Self::ResultBuild(error)
    }
}

impl From<WireError> for MonteCarloError {
    fn from(error: WireError) -> Self {
        Self::Wire(error)
    }
}

impl From<AadConfigError> for MonteCarloError {
    fn from(error: AadConfigError) -> Self {
        Self::AadConfig(error)
    }
}

impl From<RqmcPlanError> for MonteCarloError {
    fn from(error: RqmcPlanError) -> Self {
        Self::RqmcPlan(error)
    }
}

impl From<RiskConfigError> for MonteCarloError {
    fn from(error: RiskConfigError) -> Self {
        Self::RiskConfig(error)
    }
}

impl From<TryExecutionError<GraphError>> for MonteCarloError {
    fn from(error: TryExecutionError<GraphError>) -> Self {
        match error {
            TryExecutionError::Execution(error) => Self::Execution(error),
            TryExecutionError::Evaluation { source, .. } => Self::Graph(source),
        }
    }
}

impl From<TryExecutionError<MonteCarloError>> for MonteCarloError {
    fn from(error: TryExecutionError<MonteCarloError>) -> Self {
        match error {
            TryExecutionError::Execution(error) => Self::Execution(error),
            TryExecutionError::Evaluation { source, .. } => source,
        }
    }
}

impl fmt::Display for MonteCarloError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEngine => {
                write!(
                    formatter,
                    "the selected pricing entry point does not support this engine"
                )
            }
            Self::UnsupportedModel { model } => {
                write!(
                    formatter,
                    "the selected pricing entry point does not support the {model} model yet"
                )
            }
            Self::UnsupportedRiskForModel { model } => {
                write!(
                    formatter,
                    "the selected pricing entry point does not support risk requests for the {model} model yet"
                )
            }
            Self::InvalidLocalVolatilityTimeGrid {
                expiry_bits,
                first_bits,
                last_bits,
            } => write!(
                formatter,
                "Local Volatility Price-only evaluation requires explicit time nodes from 0.0 through expiry; expiry=0x{expiry_bits:016x}, first=0x{first_bits:016x}, last=0x{last_bits:016x}"
            ),
            Self::MissingLocalVolatilityReportingBasis => write!(
                formatter,
                "Local Volatility VegaKT evaluation requires a reporting-IV basis on the model"
            ),
            Self::MismatchedLocalVolatilityReportingBasis => write!(
                formatter,
                "Local Volatility VegaKT request nodes must match the model reporting-IV basis"
            ),
            Self::InvalidGammaBump {
                spot_bits,
                bump_bits,
            } => write!(
                formatter,
                "gamma bump must leave a positive down-bumped spot; spot=0x{spot_bits:016x}, bump=0x{bump_bits:016x}"
            ),
            Self::InsufficientSamplingUnits { count } => write!(
                formatter,
                "at least two independent sampling units are required for stochastic error estimation; received {count}"
            ),
            Self::NonFiniteTotalVariance { bits } => {
                write!(
                    formatter,
                    "Black-Scholes total variance is non-finite: 0x{bits:016x}"
                )
            }
            Self::UnsupportedObservationUnderlying { product, market } => write!(
                formatter,
                "product observation underlying {product} does not match market underlying {market}"
            ),
            Self::Market(error) => error.fmt(formatter),
            Self::LocalVol(error) => error.fmt(formatter),
            Self::Graph(error) => error.fmt(formatter),
            Self::ExecutorBuild(error) => error.fmt(formatter),
            Self::Execution(error) => error.fmt(formatter),
            Self::ResultBuild(error) => error.fmt(formatter),
            Self::Wire(error) => error.fmt(formatter),
            Self::AadConfig(error) => error.fmt(formatter),
            Self::RqmcPlan(error) => error.fmt(formatter),
            Self::RiskConfig(error) => error.fmt(formatter),
        }
    }
}

impl Error for MonteCarloError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Market(error) => Some(error),
            Self::LocalVol(error) => Some(error),
            Self::Graph(error) => Some(error),
            Self::ExecutorBuild(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::ResultBuild(error) => Some(error),
            Self::Wire(error) => Some(error),
            Self::AadConfig(error) => Some(error),
            Self::RqmcPlan(error) => Some(error),
            Self::RiskConfig(error) => Some(error),
            _ => None,
        }
    }
}
