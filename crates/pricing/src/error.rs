use std::error::Error;
use std::fmt;

use pricing_core::{CoreError, CurrencyId, Date, UnderlyingId};
use pricing_aad::AadConfigError;
use pricing_market::MarketError;
use pricing_mc::{ExecutionError, ExecutorBuildError, TryExecutionError};
use pricing_product::GraphError;

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
    VegaKtUnsupportedForBlackScholes,
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
            Self::VegaKtUnsupportedForBlackScholes => {
                write!(formatter, "VegaKT requires a Local Volatility model")
            }
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
    InvalidGammaBump {
        spot_bits: u64,
        bump_bits: u64,
    },
    InsufficientSamplingUnits { count: u64 },
    NonFiniteTotalVariance { bits: u64 },
    Market(MarketError),
    Graph(GraphError),
    ExecutorBuild(ExecutorBuildError),
    Execution(ExecutionError),
    ResultBuild(ResultBuildError),
    Wire(WireError),
    AadConfig(AadConfigError),
}

impl From<MarketError> for MonteCarloError {
    fn from(error: MarketError) -> Self {
        Self::Market(error)
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

impl From<TryExecutionError<GraphError>> for MonteCarloError {
    fn from(error: TryExecutionError<GraphError>) -> Self {
        match error {
            TryExecutionError::Execution(error) => Self::Execution(error),
            TryExecutionError::Evaluation { source, .. } => Self::Graph(source),
        }
    }
}

impl fmt::Display for MonteCarloError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEngine => {
                write!(formatter, "the European BS simulation path supports pseudo-MC only")
            }
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
            Self::Market(error) => error.fmt(formatter),
            Self::Graph(error) => error.fmt(formatter),
            Self::ExecutorBuild(error) => error.fmt(formatter),
            Self::Execution(error) => error.fmt(formatter),
            Self::ResultBuild(error) => error.fmt(formatter),
            Self::Wire(error) => error.fmt(formatter),
            Self::AadConfig(error) => error.fmt(formatter),
        }
    }
}

impl Error for MonteCarloError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Market(error) => Some(error),
            Self::Graph(error) => Some(error),
            Self::ExecutorBuild(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::ResultBuild(error) => Some(error),
            Self::Wire(error) => Some(error),
            Self::AadConfig(error) => Some(error),
            _ => None,
        }
    }
}
