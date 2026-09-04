use std::error::Error;
use std::fmt;

use pricing_core::CoreError;

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RiskConfigError {
    Core(CoreError),
    TooFewVegaKtMaturities { count: usize },
    TooFewVegaKtStrikes { count: usize },
    UnsortedVegaKtMaturities { left_index: usize },
    UnsortedVegaKtStrikes { left_index: usize },
    InvalidDensityThreshold { bits: u64 },
    ZeroCheckpointInterval,
    ZeroAadTileCapacity,
}

impl From<CoreError> for RiskConfigError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl fmt::Display for RiskConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => error.fmt(formatter),
            Self::TooFewVegaKtMaturities { count } => write!(
                formatter,
                "VegaKT requires at least two maturity nodes; received {count}"
            ),
            Self::TooFewVegaKtStrikes { count } => write!(
                formatter,
                "VegaKT requires at least two log-moneyness nodes; received {count}"
            ),
            Self::UnsortedVegaKtMaturities { left_index } => write!(
                formatter,
                "VegaKT maturity nodes are not strictly increasing at {left_index}"
            ),
            Self::UnsortedVegaKtStrikes { left_index } => write!(
                formatter,
                "VegaKT log-moneyness nodes are not strictly increasing at {left_index}"
            ),
            Self::InvalidDensityThreshold { bits } => write!(
                formatter,
                "VegaKT relative density threshold must be in (0, 1]; received 0x{bits:016x}"
            ),
            Self::ZeroCheckpointInterval => {
                write!(formatter, "AAD checkpoint interval must be positive")
            }
            Self::ZeroAadTileCapacity => write!(formatter, "AAD tile capacity must be positive"),
        }
    }
}

impl Error for RiskConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}
