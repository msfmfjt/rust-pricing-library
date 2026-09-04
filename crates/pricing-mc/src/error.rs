use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EngineConfigError {
    ZeroSamplingUnits,
    InvalidSobolPointCount { value: u64 },
    SobolPointCountOverflow { requested: u64 },
    ZeroScrambleCount,
}

impl fmt::Display for EngineConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSamplingUnits => write!(formatter, "sampling-unit count must be positive"),
            Self::InvalidSobolPointCount { value } => write!(
                formatter,
                "Sobol points per scramble must be a positive power of two; received {value}"
            ),
            Self::SobolPointCountOverflow { requested } => write!(
                formatter,
                "no representable power-of-two Sobol point count covers {requested}"
            ),
            Self::ZeroScrambleCount => write!(formatter, "RQMC scramble count must be positive"),
        }
    }
}

impl Error for EngineConfigError {}
