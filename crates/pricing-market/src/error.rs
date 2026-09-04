use std::error::Error;
use std::fmt;

use pricing_core::{CurveId, UnderlyingId};

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MarketError {
    InvalidPillarCount {
        curve: CurveId,
        count: usize,
    },
    PillarLengthMismatch {
        curve: CurveId,
        times: usize,
        discount_factors: usize,
    },
    NonFinitePillarTime {
        curve: CurveId,
        index: usize,
        bits: u64,
    },
    NegativePillarTime {
        curve: CurveId,
        index: usize,
        bits: u64,
    },
    NonPositiveDiscountFactor {
        curve: CurveId,
        index: usize,
        bits: u64,
    },
    NonFiniteDiscountFactor {
        curve: CurveId,
        index: usize,
        bits: u64,
    },
    UnsortedPillars {
        curve: CurveId,
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    MissingValuationAnchor {
        curve: CurveId,
        first_time_bits: u64,
    },
    InvalidValuationDiscount {
        curve: CurveId,
        discount_bits: u64,
    },
    InvalidQueryTime {
        curve: CurveId,
        bits: u64,
    },
    NegativeQueryTime {
        curve: CurveId,
        bits: u64,
    },
    NonFiniteCurveValue {
        curve: CurveId,
        time_bits: u64,
        log_discount_bits: u64,
    },
    DiagnosticCountOverflow {
        curve: CurveId,
    },
    NonFiniteForward {
        underlying: UnderlyingId,
        time_bits: u64,
        forward_bits: u64,
    },
}

impl fmt::Display for MarketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPillarCount { curve, count } => {
                write!(formatter, "curve {curve} requires at least two pillars; received {count}")
            }
            Self::PillarLengthMismatch {
                curve,
                times,
                discount_factors,
            } => write!(
                formatter,
                "curve {curve} has {times} times and {discount_factors} discount factors"
            ),
            Self::NonFinitePillarTime { curve, index, bits } => write!(
                formatter,
                "curve {curve} pillar {index} time is non-finite: 0x{bits:016x}"
            ),
            Self::NegativePillarTime { curve, index, bits } => write!(
                formatter,
                "curve {curve} pillar {index} time is negative: 0x{bits:016x}"
            ),
            Self::NonPositiveDiscountFactor { curve, index, bits } => write!(
                formatter,
                "curve {curve} pillar {index} discount factor is not positive: 0x{bits:016x}"
            ),
            Self::NonFiniteDiscountFactor { curve, index, bits } => write!(
                formatter,
                "curve {curve} pillar {index} discount factor is non-finite: 0x{bits:016x}"
            ),
            Self::UnsortedPillars {
                curve,
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "curve {curve} times are not strictly increasing at {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::MissingValuationAnchor {
                curve,
                first_time_bits,
            } => write!(
                formatter,
                "curve {curve} first time must be zero; received 0x{first_time_bits:016x}"
            ),
            Self::InvalidValuationDiscount {
                curve,
                discount_bits,
            } => write!(
                formatter,
                "curve {curve} discount factor at time zero must be one; received 0x{discount_bits:016x}"
            ),
            Self::InvalidQueryTime { curve, bits } => {
                write!(formatter, "curve {curve} query time is non-finite: 0x{bits:016x}")
            }
            Self::NegativeQueryTime { curve, bits } => {
                write!(formatter, "curve {curve} query time is negative: 0x{bits:016x}")
            }
            Self::NonFiniteCurveValue {
                curve,
                time_bits,
                log_discount_bits,
            } => write!(
                formatter,
                "curve {curve} produced a non-finite value at 0x{time_bits:016x}: 0x{log_discount_bits:016x}"
            ),
            Self::DiagnosticCountOverflow { curve } => {
                write!(formatter, "curve {curve} extrapolation counter overflowed")
            }
            Self::NonFiniteForward {
                underlying,
                time_bits,
                forward_bits,
            } => write!(
                formatter,
                "underlying {underlying} produced an invalid forward at 0x{time_bits:016x}: 0x{forward_bits:016x}"
            ),
        }
    }
}

impl Error for MarketError {}
