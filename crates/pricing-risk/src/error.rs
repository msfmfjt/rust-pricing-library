use std::error::Error;
use std::fmt;

use pricing_core::CoreError;
use pricing_market::MarketError;

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RiskConfigError {
    Core(CoreError),
    Market(MarketError),
    TooFewVegaKtMaturities {
        count: usize,
    },
    TooFewVegaKtStrikes {
        count: usize,
    },
    UnsortedVegaKtMaturities {
        left_index: usize,
    },
    UnsortedVegaKtStrikes {
        left_index: usize,
    },
    TooFewVegaKtDomainNodes {
        count: usize,
    },
    VegaKtDensityLengthMismatch {
        node_count: usize,
        density_count: usize,
    },
    VegaKtForwardLengthMismatch {
        maturity_count: usize,
        forward_count: usize,
    },
    UnsortedVegaKtDomainNodes {
        left_index: usize,
    },
    NegativeVegaKtDensity {
        index: usize,
        bits: u64,
    },
    NoPositiveVegaKtDensity,
    ForwardNodeBelowDensityThreshold {
        forward_index: usize,
        density_bits: u64,
        max_density_bits: u64,
        threshold_bits: u64,
    },
    VegaKtActiveDomainOutOfRange {
        end_index: usize,
        node_count: usize,
    },
    InvalidVegaKtEquation11Denominator {
        bits: u64,
    },
    InvalidVegaKtEquation11Gamma {
        bits: u64,
    },
    TooFewVegaKtRefinementLevels {
        count: usize,
    },
    NonDecreasingVegaKtRefinementStep {
        left_index: usize,
    },
    TooFewVegaKtTransitionStrikes {
        count: usize,
    },
    UnsortedVegaKtTransitionStrikes {
        left_index: usize,
    },
    InvalidVegaKtTransitionProbability {
        cell_index: usize,
        bits: u64,
    },
    VegaKtTransitionMassDefect {
        bits: u64,
    },
    VegaKtGammaLengthMismatch {
        strike_count: usize,
        gamma_count: usize,
    },
    InvalidVegaKtTransitionValue {
        bits: u64,
    },
    TooFewReportingIvMaturities {
        count: usize,
    },
    TooFewReportingIvStrikes {
        count: usize,
    },
    UnsortedReportingIvMaturities {
        left_index: usize,
    },
    UnsortedReportingIvStrikes {
        left_index: usize,
    },
    ReportingIvValueLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ReportingIvQueryTimeOutOfRange {
        bits: u64,
    },
    ReportingIvBucketLengthMismatch {
        expected: usize,
        actual: usize,
    },
    VegaKtBucketEstimateLengthMismatch {
        expected: usize,
        actual: usize,
    },
    VegaKtFullCovarianceLengthMismatch {
        expected: usize,
        actual: usize,
    },
    VegaKtProjectionReconciliationFailure {
        reconstructed_bits: u64,
        pre_projection_bits: u64,
    },
    EmptyVegaKtBucketSamples,
    VegaKtBucketSampleLengthMismatch {
        expected_multiple: usize,
        actual: usize,
    },
    InvalidDensityThreshold {
        bits: u64,
    },
    EmptyPayoffSmoothingWidthLadder,
    NonMonotonePayoffSmoothingWidthLadder {
        left_index: usize,
    },
    PayoffSmoothingWidthTooLarge {
        bits: u64,
    },
    ZeroCheckpointInterval,
    ZeroAadTileCapacity,
}

impl From<CoreError> for RiskConfigError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl From<MarketError> for RiskConfigError {
    fn from(error: MarketError) -> Self {
        Self::Market(error)
    }
}

impl fmt::Display for RiskConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => error.fmt(formatter),
            Self::Market(error) => error.fmt(formatter),
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
            Self::TooFewVegaKtDomainNodes { count } => write!(
                formatter,
                "VegaKT active domain requires at least two log-moneyness nodes; received {count}"
            ),
            Self::VegaKtDensityLengthMismatch {
                node_count,
                density_count,
            } => write!(
                formatter,
                "VegaKT density count {density_count} does not match node count {node_count}"
            ),
            Self::VegaKtForwardLengthMismatch {
                maturity_count,
                forward_count,
            } => write!(
                formatter,
                "VegaKT forward count {forward_count} does not match operator maturity count {maturity_count}"
            ),
            Self::UnsortedVegaKtDomainNodes { left_index } => write!(
                formatter,
                "VegaKT active-domain log-moneyness nodes are not strictly increasing at {left_index}"
            ),
            Self::NegativeVegaKtDensity { index, bits } => write!(
                formatter,
                "VegaKT call density at index {index} must be non-negative; received 0x{bits:016x}"
            ),
            Self::NoPositiveVegaKtDensity => {
                write!(
                    formatter,
                    "VegaKT active domain requires a positive maximum call density"
                )
            }
            Self::ForwardNodeBelowDensityThreshold {
                forward_index,
                density_bits,
                max_density_bits,
                threshold_bits,
            } => write!(
                formatter,
                "VegaKT forward node {forward_index} is below the active density threshold; density=0x{density_bits:016x}, max_density=0x{max_density_bits:016x}, threshold=0x{threshold_bits:016x}"
            ),
            Self::VegaKtActiveDomainOutOfRange {
                end_index,
                node_count,
            } => write!(
                formatter,
                "VegaKT active domain end index {end_index} is outside node count {node_count}"
            ),
            Self::InvalidVegaKtEquation11Denominator { bits } => write!(
                formatter,
                "VegaKT equation (11) denominator must be finite and positive; received 0x{bits:016x}"
            ),
            Self::InvalidVegaKtEquation11Gamma { bits } => write!(
                formatter,
                "VegaKT equation (11) Local Gamma must be finite; received 0x{bits:016x}"
            ),
            Self::TooFewVegaKtRefinementLevels { count } => write!(
                formatter,
                "VegaKT equation (11) refinement diagnostics require at least two levels; received {count}"
            ),
            Self::NonDecreasingVegaKtRefinementStep { left_index } => write!(
                formatter,
                "VegaKT equation (11) refinement steps must be strictly decreasing at {left_index}"
            ),
            Self::TooFewVegaKtTransitionStrikes { count } => write!(
                formatter,
                "VegaKT transition operator requires at least two strike nodes; received {count}"
            ),
            Self::UnsortedVegaKtTransitionStrikes { left_index } => write!(
                formatter,
                "VegaKT transition strike nodes are not strictly increasing at {left_index}"
            ),
            Self::InvalidVegaKtTransitionProbability { cell_index, bits } => write!(
                formatter,
                "VegaKT transition cell {cell_index} probability must be finite before clipping; received 0x{bits:016x}"
            ),
            Self::VegaKtTransitionMassDefect { bits } => write!(
                formatter,
                "VegaKT transition cell probabilities do not conserve mass; defect bits 0x{bits:016x}"
            ),
            Self::VegaKtGammaLengthMismatch {
                strike_count,
                gamma_count,
            } => write!(
                formatter,
                "VegaKT Local Gamma count {gamma_count} does not match strike count {strike_count}"
            ),
            Self::InvalidVegaKtTransitionValue { bits } => write!(
                formatter,
                "VegaKT transitioned Local Gamma value must be finite; received 0x{bits:016x}"
            ),
            Self::TooFewReportingIvMaturities { count } => write!(
                formatter,
                "VegaKT reporting-IV basis requires at least two maturity nodes; received {count}"
            ),
            Self::TooFewReportingIvStrikes { count } => write!(
                formatter,
                "VegaKT reporting-IV basis requires at least two log-moneyness nodes; received {count}"
            ),
            Self::UnsortedReportingIvMaturities { left_index } => write!(
                formatter,
                "VegaKT reporting-IV maturity nodes are not strictly increasing at {left_index}"
            ),
            Self::UnsortedReportingIvStrikes { left_index } => write!(
                formatter,
                "VegaKT reporting-IV log-moneyness nodes are not strictly increasing at {left_index}"
            ),
            Self::ReportingIvValueLengthMismatch { expected, actual } => write!(
                formatter,
                "VegaKT reporting-IV value count {actual} does not match expected row-major count {expected}"
            ),
            Self::ReportingIvQueryTimeOutOfRange { bits } => write!(
                formatter,
                "VegaKT reporting-IV query time is outside the reporting maturity range; received 0x{bits:016x}"
            ),
            Self::ReportingIvBucketLengthMismatch { expected, actual } => write!(
                formatter,
                "VegaKT reporting-IV bucket count {actual} does not match expected row-major count {expected}"
            ),
            Self::VegaKtBucketEstimateLengthMismatch { expected, actual } => write!(
                formatter,
                "VegaKT bucket estimate count {actual} does not match expected row-major count {expected}"
            ),
            Self::VegaKtFullCovarianceLengthMismatch { expected, actual } => write!(
                formatter,
                "VegaKT full covariance count {actual} does not match expected row-major matrix count {expected}"
            ),
            Self::VegaKtProjectionReconciliationFailure {
                reconstructed_bits,
                pre_projection_bits,
            } => write!(
                formatter,
                "VegaKT bucket sum plus residual does not reconcile with pre-projection sensitivity; reconstructed=0x{reconstructed_bits:016x}, pre_projection=0x{pre_projection_bits:016x}"
            ),
            Self::EmptyVegaKtBucketSamples => write!(
                formatter,
                "VegaKT bucket statistics require at least one ordered sample"
            ),
            Self::VegaKtBucketSampleLengthMismatch {
                expected_multiple,
                actual,
            } => write!(
                formatter,
                "VegaKT bucket sample count {actual} is not a multiple of bucket count {expected_multiple}"
            ),
            Self::InvalidDensityThreshold { bits } => write!(
                formatter,
                "VegaKT relative density threshold must be in (0, 1]; received 0x{bits:016x}"
            ),
            Self::EmptyPayoffSmoothingWidthLadder => {
                write!(formatter, "payoff smoothing width ladder must not be empty")
            }
            Self::NonMonotonePayoffSmoothingWidthLadder { left_index } => write!(
                formatter,
                "payoff smoothing width ladder must be strictly monotone; direction changes or repeats at {left_index}"
            ),
            Self::PayoffSmoothingWidthTooLarge { bits } => write!(
                formatter,
                "payoff smoothing half-width must have a finite doubled transition width; received 0x{bits:016x}"
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
            Self::Market(error) => Some(error),
            _ => None,
        }
    }
}
