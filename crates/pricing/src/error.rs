use std::error::Error;
use std::fmt;

use pricing_core::{CoreError, CurrencyId, Date, UnderlyingId};

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
