use std::error::Error;
use std::fmt;

use pricing_core::{CurveId, EventId, PathIndex, UnderlyingId};

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
    InvalidDividendTime {
        event: EventId,
        index: usize,
        bits: u64,
    },
    UnsortedDividendEvents {
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    InvalidDividendCash {
        event: EventId,
        bits: u64,
    },
    InvalidDividendProportion {
        event: EventId,
        bits: u64,
    },
    NonFiniteDividendTransform {
        event: EventId,
        field: &'static str,
        bits: u64,
    },
    NonPositivePostDividendSpot {
        underlying: UnderlyingId,
        event: EventId,
        path: PathIndex,
        pre_spot_bits: u64,
        alpha_bits: u64,
        beta_bits: u64,
        fixed_cash_bits: u64,
        post_spot_bits: u64,
    },
    DividendMatchingConditionViolation {
        event: EventId,
        expected_bits: u64,
        actual_bits: u64,
        abs_error_bits: u64,
        abs_tol_bits: u64,
        rel_tol_bits: u64,
    },
    InvalidSurfaceKnotCount {
        count: usize,
    },
    SurfaceKnotLengthMismatch {
        times: usize,
        values: usize,
    },
    InvalidSurfaceKnotTime {
        index: usize,
        bits: u64,
    },
    UnsortedSurfaceKnots {
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    InvalidTheta {
        index: usize,
        bits: u64,
    },
    DecreasingTheta {
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    InvalidSurfaceParameter {
        parameter: &'static str,
        bits: u64,
    },
    SsviAdmissibilityViolation {
        condition: &'static str,
        left_bits: u64,
        right_bits: u64,
    },
    InvalidSurfaceQuery {
        coordinate: &'static str,
        bits: u64,
    },
    NonFiniteSurfaceValue {
        field: &'static str,
        time_bits: u64,
        log_moneyness_bits: u64,
        value_bits: u64,
    },
    NonPositiveSurfaceValue {
        field: &'static str,
        time_bits: u64,
        log_moneyness_bits: u64,
        value_bits: u64,
    },
    InvalidEssviSliceCount {
        count: usize,
    },
    InvalidEssviSlice {
        index: usize,
        field: &'static str,
        bits: u64,
    },
    UnsortedEssviSlices {
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    EssviSliceAdmissibilityViolation {
        index: usize,
        condition: &'static str,
        left_bits: u64,
        right_bits: u64,
    },
    InconsistentEssviSlices {
        left_index: usize,
        condition: &'static str,
        left_bits: u64,
        right_bits: u64,
    },
    InvalidLocalVarianceNodeCount {
        coordinate: &'static str,
        count: usize,
    },
    InvalidLocalVarianceNode {
        coordinate: &'static str,
        index: usize,
        bits: u64,
    },
    UnsortedLocalVarianceNodes {
        coordinate: &'static str,
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    LocalVarianceValueLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidLocalVarianceValue {
        index: usize,
        bits: u64,
    },
    LocalVarianceBoundaryCountOverflow {
        boundary: &'static str,
    },
    SurfaceQuantileNotBracketed {
        side: &'static str,
        time_bits: u64,
        probability_bits: u64,
    },
}

impl fmt::Display for MarketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPillarCount { curve, count } => {
                write!(
                    formatter,
                    "curve {curve} requires at least two pillars; received {count}"
                )
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
                write!(
                    formatter,
                    "curve {curve} query time is non-finite: 0x{bits:016x}"
                )
            }
            Self::NegativeQueryTime { curve, bits } => {
                write!(
                    formatter,
                    "curve {curve} query time is negative: 0x{bits:016x}"
                )
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
            Self::InvalidDividendTime { event, index, bits } => write!(
                formatter,
                "dividend event {event} at index {index} has an invalid ex-time: 0x{bits:016x}"
            ),
            Self::UnsortedDividendEvents {
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "dividend event times are not strictly increasing at {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::InvalidDividendCash { event, bits } => write!(
                formatter,
                "dividend event {event} fixed cash amount must be finite and non-negative: 0x{bits:016x}"
            ),
            Self::InvalidDividendProportion { event, bits } => write!(
                formatter,
                "dividend event {event} proportional amount must be finite with 0 <= beta < 1: 0x{bits:016x}"
            ),
            Self::NonFiniteDividendTransform { event, field, bits } => write!(
                formatter,
                "dividend event {event} produced a non-finite affine transform field {field}: 0x{bits:016x}"
            ),
            Self::NonPositivePostDividendSpot {
                underlying,
                event,
                path,
                pre_spot_bits,
                alpha_bits,
                beta_bits,
                fixed_cash_bits,
                post_spot_bits,
            } => write!(
                formatter,
                "underlying {underlying} path {path} has non-positive post-dividend spot at event {event}: pre=0x{pre_spot_bits:016x}, alpha=0x{alpha_bits:016x}, beta=0x{beta_bits:016x}, fixed_cash=0x{fixed_cash_bits:016x}, post=0x{post_spot_bits:016x}"
            ),
            Self::DividendMatchingConditionViolation {
                event,
                expected_bits,
                actual_bits,
                abs_error_bits,
                abs_tol_bits,
                rel_tol_bits,
            } => write!(
                formatter,
                "dividend event {event} violates the affine call-price matching condition: expected=0x{expected_bits:016x}, actual=0x{actual_bits:016x}, abs_error=0x{abs_error_bits:016x}, abs_tol=0x{abs_tol_bits:016x}, rel_tol=0x{rel_tol_bits:016x}"
            ),
            Self::InvalidSurfaceKnotCount { count } => write!(
                formatter,
                "an implied surface theta curve requires at least two knots; received {count}"
            ),
            Self::SurfaceKnotLengthMismatch { times, values } => write!(
                formatter,
                "an implied surface theta curve has {times} times and {values} values"
            ),
            Self::InvalidSurfaceKnotTime { index, bits } => write!(
                formatter,
                "implied surface knot {index} time must be finite and positive: 0x{bits:016x}"
            ),
            Self::UnsortedSurfaceKnots {
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "implied surface times are not strictly increasing at {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::InvalidTheta { index, bits } => write!(
                formatter,
                "implied surface theta {index} must be finite and positive: 0x{bits:016x}"
            ),
            Self::DecreasingTheta {
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "implied surface theta decreases at {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::InvalidSurfaceParameter { parameter, bits } => write!(
                formatter,
                "implied surface parameter {parameter} is invalid: 0x{bits:016x}"
            ),
            Self::SsviAdmissibilityViolation {
                condition,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "SSVI admissibility condition {condition} failed: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::InvalidSurfaceQuery { coordinate, bits } => write!(
                formatter,
                "implied surface query {coordinate} is invalid: 0x{bits:016x}"
            ),
            Self::NonFiniteSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            } => write!(
                formatter,
                "implied surface {field} is non-finite at time 0x{time_bits:016x}, log-moneyness 0x{log_moneyness_bits:016x}: 0x{value_bits:016x}"
            ),
            Self::NonPositiveSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            } => write!(
                formatter,
                "implied surface {field} is not positive at time 0x{time_bits:016x}, log-moneyness 0x{log_moneyness_bits:016x}: 0x{value_bits:016x}"
            ),
            Self::InvalidEssviSliceCount { count } => write!(
                formatter,
                "an eSSVI surface requires at least two slices; received {count}"
            ),
            Self::InvalidEssviSlice { index, field, bits } => write!(
                formatter,
                "eSSVI slice {index} field {field} is invalid: 0x{bits:016x}"
            ),
            Self::UnsortedEssviSlices {
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "eSSVI slice times are not strictly increasing at {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::EssviSliceAdmissibilityViolation {
                index,
                condition,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "eSSVI slice {index} admissibility condition {condition} failed: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::InconsistentEssviSlices {
                left_index,
                condition,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "eSSVI slices at {left_index} violate {condition}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::InvalidLocalVarianceNodeCount { coordinate, count } => write!(
                formatter,
                "Local variance {coordinate} grid requires at least two nodes; received {count}"
            ),
            Self::InvalidLocalVarianceNode {
                coordinate,
                index,
                bits,
            } => write!(
                formatter,
                "Local variance {coordinate} node {index} is invalid: 0x{bits:016x}"
            ),
            Self::UnsortedLocalVarianceNodes {
                coordinate,
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "Local variance {coordinate} nodes are not strictly increasing at {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::LocalVarianceValueLengthMismatch { expected, actual } => write!(
                formatter,
                "Local variance grid expected {expected} row-major values; received {actual}"
            ),
            Self::InvalidLocalVarianceValue { index, bits } => write!(
                formatter,
                "Local variance grid value {index} is invalid: 0x{bits:016x}"
            ),
            Self::LocalVarianceBoundaryCountOverflow { boundary } => write!(
                formatter,
                "Local variance {boundary} boundary counter overflowed"
            ),
            Self::SurfaceQuantileNotBracketed {
                side,
                time_bits,
                probability_bits,
            } => write!(
                formatter,
                "implied surface {side} quantile was not bracketed at time 0x{time_bits:016x}, probability 0x{probability_bits:016x}"
            ),
        }
    }
}

impl Error for MarketError {}
