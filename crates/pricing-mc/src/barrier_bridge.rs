use std::error::Error;
use std::fmt;

use pricing_market::AffineDividendCoordinate;

pub const BARRIER_BRIDGE_ABI: &str = "continuous-barrier-bridge-log-survival-v1";

const LOG_MAX_EXP_ARGUMENT: f64 = 6.564_958_884_017_094;
const LOG_MIN_POSITIVE: f64 = -744.440_071_921_381_2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierBridgeDirection {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierBridgeStatus {
    FiniteCorrection,
    TouchedEndpoint,
    ZeroVariance,
    SurvivalUnderflow,
    CertainSurvival,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BarrierBridgeError {
    InvalidPositiveInput { field: &'static str, bits: u64 },
    InvalidAffineScale { bits: u64 },
    InvalidTransformedBarrier { bits: u64 },
    InvalidLocalVariance { endpoint: &'static str, bits: u64 },
    InvalidIntervalLength { bits: u64 },
    NonFiniteIntegratedVariance { bits: u64 },
}

impl fmt::Display for BarrierBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPositiveInput { field, bits } => write!(
                formatter,
                "Barrier bridge {field} must be finite and positive: 0x{bits:016x}"
            ),
            Self::InvalidAffineScale { bits } => write!(
                formatter,
                "Barrier bridge affine scale B must be finite and positive: 0x{bits:016x}"
            ),
            Self::InvalidTransformedBarrier { bits } => write!(
                formatter,
                "Barrier bridge transformed F barrier must be finite and positive: 0x{bits:016x}"
            ),
            Self::InvalidLocalVariance { endpoint, bits } => write!(
                formatter,
                "Barrier bridge {endpoint} Local variance must be finite and non-negative: 0x{bits:016x}"
            ),
            Self::InvalidIntervalLength { bits } => write!(
                formatter,
                "Barrier bridge interval length must be finite and positive: 0x{bits:016x}"
            ),
            Self::NonFiniteIntegratedVariance { bits } => write!(
                formatter,
                "Barrier bridge trapezoidal integrated variance is non-finite: 0x{bits:016x}"
            ),
        }
    }
}

impl Error for BarrierBridgeError {}

pub fn transformed_barrier(
    spot_barrier: f64,
    initial_spot: f64,
    coordinate: AffineDividendCoordinate,
) -> Result<f64, BarrierBridgeError> {
    validate_positive("spot barrier", spot_barrier)?;
    validate_positive("initial spot", initial_spot)?;
    if !coordinate.b().is_finite() || coordinate.b() <= 0.0 {
        return Err(BarrierBridgeError::InvalidAffineScale {
            bits: coordinate.b().to_bits(),
        });
    }
    let numerator = spot_barrier - coordinate.a() * initial_spot;
    let barrier = numerator / coordinate.b();
    if !barrier.is_finite() || barrier <= 0.0 {
        return Err(BarrierBridgeError::InvalidTransformedBarrier {
            bits: barrier.to_bits(),
        });
    }
    Ok(barrier)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarrierBridgeIntervalInput {
    pub direction: BarrierBridgeDirection,
    pub left_state: f64,
    pub right_state: f64,
    pub left_barrier: f64,
    pub right_barrier: f64,
    pub left_local_variance: f64,
    pub right_local_variance: f64,
    pub dt: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BarrierBridgeIntervalAdjoints {
    pub left_state: f64,
    pub right_state: f64,
    pub left_barrier: f64,
    pub right_barrier: f64,
    pub left_local_variance: f64,
    pub right_local_variance: f64,
    pub dt: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarrierBridgeInterval {
    status: BarrierBridgeStatus,
    log_survival: f64,
    log_survival_derivatives: BarrierBridgeIntervalAdjoints,
}

impl BarrierBridgeInterval {
    pub fn evaluate(input: BarrierBridgeIntervalInput) -> Result<Self, BarrierBridgeError> {
        validate_positive("left state", input.left_state)?;
        validate_positive("right state", input.right_state)?;
        validate_positive("left transformed barrier", input.left_barrier)?;
        validate_positive("right transformed barrier", input.right_barrier)?;
        validate_variance("left", input.left_local_variance)?;
        validate_variance("right", input.right_local_variance)?;
        if !input.dt.is_finite() || input.dt <= 0.0 {
            return Err(BarrierBridgeError::InvalidIntervalLength {
                bits: input.dt.to_bits(),
            });
        }

        let left_distance =
            signed_log_distance(input.direction, input.left_state, input.left_barrier);
        let right_distance =
            signed_log_distance(input.direction, input.right_state, input.right_barrier);
        if left_distance <= 0.0 || right_distance <= 0.0 {
            return Ok(Self::deterministic(
                BarrierBridgeStatus::TouchedEndpoint,
                f64::NEG_INFINITY,
            ));
        }

        let average_variance = 0.5 * input.left_local_variance + 0.5 * input.right_local_variance;
        let integrated_variance = average_variance * input.dt;
        if !integrated_variance.is_finite() {
            return Err(BarrierBridgeError::NonFiniteIntegratedVariance {
                bits: integrated_variance.to_bits(),
            });
        }
        if integrated_variance == 0.0 {
            return Ok(Self::deterministic(BarrierBridgeStatus::ZeroVariance, 0.0));
        }

        let log_exponent = std::f64::consts::LN_2 + left_distance.ln() + right_distance.ln()
            - integrated_variance.ln();
        if log_exponent >= LOG_MAX_EXP_ARGUMENT {
            return Ok(Self::deterministic(
                BarrierBridgeStatus::CertainSurvival,
                0.0,
            ));
        }

        let (status, log_survival, log_exponent_derivative) = if log_exponent <= LOG_MIN_POSITIVE {
            (BarrierBridgeStatus::SurvivalUnderflow, log_exponent, 1.0)
        } else {
            let exponent = log_exponent.exp();
            let survival = -(-exponent).exp_m1();
            (
                BarrierBridgeStatus::FiniteCorrection,
                survival.ln(),
                exponent / exponent.exp_m1(),
            )
        };
        let left_distance_derivative = log_exponent_derivative / left_distance;
        let right_distance_derivative = log_exponent_derivative / right_distance;
        let integrated_variance_derivative = -log_exponent_derivative / integrated_variance;
        let direction_sign = match input.direction {
            BarrierBridgeDirection::Up => 1.0,
            BarrierBridgeDirection::Down => -1.0,
        };
        let average_variance_derivative = integrated_variance_derivative * input.dt;
        Ok(Self {
            status,
            log_survival,
            log_survival_derivatives: BarrierBridgeIntervalAdjoints {
                left_state: -direction_sign * left_distance_derivative / input.left_state,
                right_state: -direction_sign * right_distance_derivative / input.right_state,
                left_barrier: direction_sign * left_distance_derivative / input.left_barrier,
                right_barrier: direction_sign * right_distance_derivative / input.right_barrier,
                left_local_variance: 0.5 * average_variance_derivative,
                right_local_variance: 0.5 * average_variance_derivative,
                dt: integrated_variance_derivative * average_variance,
            },
        })
    }

    const fn deterministic(status: BarrierBridgeStatus, log_survival: f64) -> Self {
        Self {
            status,
            log_survival,
            log_survival_derivatives: BarrierBridgeIntervalAdjoints {
                left_state: 0.0,
                right_state: 0.0,
                left_barrier: 0.0,
                right_barrier: 0.0,
                left_local_variance: 0.0,
                right_local_variance: 0.0,
                dt: 0.0,
            },
        }
    }

    #[must_use]
    pub const fn status(self) -> BarrierBridgeStatus {
        self.status
    }

    #[must_use]
    pub const fn log_survival(self) -> f64 {
        self.log_survival
    }

    #[must_use]
    pub fn survival(self) -> f64 {
        self.log_survival.exp()
    }

    #[must_use]
    pub fn reverse(self, log_survival_adjoint: f64) -> BarrierBridgeIntervalAdjoints {
        let derivatives = self.log_survival_derivatives;
        BarrierBridgeIntervalAdjoints {
            left_state: log_survival_adjoint * derivatives.left_state,
            right_state: log_survival_adjoint * derivatives.right_state,
            left_barrier: log_survival_adjoint * derivatives.left_barrier,
            right_barrier: log_survival_adjoint * derivatives.right_barrier,
            left_local_variance: log_survival_adjoint * derivatives.left_local_variance,
            right_local_variance: log_survival_adjoint * derivatives.right_local_variance,
            dt: log_survival_adjoint * derivatives.dt,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BarrierBridgePathDiagnostics {
    pub interval_count: u32,
    pub finite_correction_count: u32,
    pub touched_endpoint_count: u32,
    pub zero_variance_count: u32,
    pub survival_underflow_count: u32,
    pub certain_survival_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarrierBridgePath {
    intervals: Box<[BarrierBridgeInterval]>,
    log_survival: f64,
    diagnostics: BarrierBridgePathDiagnostics,
}

impl BarrierBridgePath {
    pub fn evaluate(inputs: &[BarrierBridgeIntervalInput]) -> Result<Self, BarrierBridgeError> {
        let mut intervals = Vec::with_capacity(inputs.len());
        let mut log_survival = 0.0;
        let mut diagnostics = BarrierBridgePathDiagnostics {
            interval_count: u32::try_from(inputs.len()).unwrap_or(u32::MAX),
            ..BarrierBridgePathDiagnostics::default()
        };
        for input in inputs.iter().copied() {
            let interval = BarrierBridgeInterval::evaluate(input)?;
            match interval.status() {
                BarrierBridgeStatus::FiniteCorrection => {
                    diagnostics.finite_correction_count =
                        diagnostics.finite_correction_count.saturating_add(1);
                }
                BarrierBridgeStatus::TouchedEndpoint => {
                    diagnostics.touched_endpoint_count =
                        diagnostics.touched_endpoint_count.saturating_add(1);
                }
                BarrierBridgeStatus::ZeroVariance => {
                    diagnostics.zero_variance_count =
                        diagnostics.zero_variance_count.saturating_add(1);
                }
                BarrierBridgeStatus::SurvivalUnderflow => {
                    diagnostics.survival_underflow_count =
                        diagnostics.survival_underflow_count.saturating_add(1);
                }
                BarrierBridgeStatus::CertainSurvival => {
                    diagnostics.certain_survival_count =
                        diagnostics.certain_survival_count.saturating_add(1);
                }
            }
            log_survival += interval.log_survival();
            intervals.push(interval);
        }
        Ok(Self {
            intervals: intervals.into_boxed_slice(),
            log_survival,
            diagnostics,
        })
    }

    #[must_use]
    pub const fn log_survival(&self) -> f64 {
        self.log_survival
    }

    #[must_use]
    pub fn survival(&self) -> f64 {
        self.log_survival.exp()
    }

    #[must_use]
    pub const fn diagnostics(&self) -> BarrierBridgePathDiagnostics {
        self.diagnostics
    }

    #[must_use]
    pub fn reverse(&self, log_survival_adjoint: f64) -> Vec<BarrierBridgeIntervalAdjoints> {
        if self.diagnostics.touched_endpoint_count != 0 {
            return vec![BarrierBridgeIntervalAdjoints::default(); self.intervals.len()];
        }
        self.intervals
            .iter()
            .map(|interval| interval.reverse(log_survival_adjoint))
            .collect()
    }
}

fn signed_log_distance(direction: BarrierBridgeDirection, state: f64, barrier: f64) -> f64 {
    match direction {
        BarrierBridgeDirection::Up => barrier.ln() - state.ln(),
        BarrierBridgeDirection::Down => state.ln() - barrier.ln(),
    }
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), BarrierBridgeError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(BarrierBridgeError::InvalidPositiveInput {
            field,
            bits: value.to_bits(),
        })
    }
}

fn validate_variance(endpoint: &'static str, value: f64) -> Result<(), BarrierBridgeError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(BarrierBridgeError::InvalidLocalVariance {
            endpoint,
            bits: value.to_bits(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pricing_core::{EventId, PositiveF64, UnderlyingId};
    use pricing_market::{AffineDividendTransform, DividendEvent, DividendQuote};

    fn input(direction: BarrierBridgeDirection) -> BarrierBridgeIntervalInput {
        BarrierBridgeIntervalInput {
            direction,
            left_state: 90.0,
            right_state: 92.0,
            left_barrier: 100.0,
            right_barrier: 100.0,
            left_local_variance: 0.04,
            right_local_variance: 0.06,
            dt: 0.25,
        }
    }

    #[test]
    fn interval_matches_log_brownian_bridge_formula() {
        let input = input(BarrierBridgeDirection::Up);
        let interval = BarrierBridgeInterval::evaluate(input).expect("interval");
        let left = (input.left_barrier / input.left_state).ln();
        let right = (input.right_barrier / input.right_state).ln();
        let variance = 0.5 * (input.left_local_variance + input.right_local_variance) * input.dt;
        let expected = 1.0 - (-2.0 * left * right / variance).exp();
        assert_eq!(interval.status(), BarrierBridgeStatus::FiniteCorrection);
        assert!((interval.survival() - expected).abs() < 5.0e-15);
        assert!((interval.log_survival() - expected.ln()).abs() < 5.0e-15);
    }

    #[test]
    fn up_and_down_are_symmetric_under_reciprocal_coordinates() {
        let up_input = input(BarrierBridgeDirection::Up);
        let down_input = BarrierBridgeIntervalInput {
            direction: BarrierBridgeDirection::Down,
            left_state: 1.0 / up_input.left_state,
            right_state: 1.0 / up_input.right_state,
            left_barrier: 1.0 / up_input.left_barrier,
            right_barrier: 1.0 / up_input.right_barrier,
            ..up_input
        };
        let up = BarrierBridgeInterval::evaluate(up_input).expect("up");
        let down = BarrierBridgeInterval::evaluate(down_input).expect("down");
        assert!((up.log_survival() - down.log_survival()).abs() < 1.0e-14);
    }

    #[test]
    fn reverse_matches_central_differences() {
        let base = input(BarrierBridgeDirection::Up);
        let evaluation = BarrierBridgeInterval::evaluate(base).expect("base");
        let reverse = evaluation.reverse(1.0);
        let bump = 1.0e-5;
        let finite_difference = |down: BarrierBridgeIntervalInput,
                                 up: BarrierBridgeIntervalInput| {
            (BarrierBridgeInterval::evaluate(up)
                .expect("up")
                .log_survival()
                - BarrierBridgeInterval::evaluate(down)
                    .expect("down")
                    .log_survival())
                / (2.0 * bump)
        };
        let cases = [
            (
                BarrierBridgeIntervalInput {
                    left_state: base.left_state - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    left_state: base.left_state + bump,
                    ..base
                },
                reverse.left_state,
            ),
            (
                BarrierBridgeIntervalInput {
                    right_state: base.right_state - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    right_state: base.right_state + bump,
                    ..base
                },
                reverse.right_state,
            ),
            (
                BarrierBridgeIntervalInput {
                    left_barrier: base.left_barrier - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    left_barrier: base.left_barrier + bump,
                    ..base
                },
                reverse.left_barrier,
            ),
            (
                BarrierBridgeIntervalInput {
                    right_barrier: base.right_barrier - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    right_barrier: base.right_barrier + bump,
                    ..base
                },
                reverse.right_barrier,
            ),
            (
                BarrierBridgeIntervalInput {
                    left_local_variance: base.left_local_variance - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    left_local_variance: base.left_local_variance + bump,
                    ..base
                },
                reverse.left_local_variance,
            ),
            (
                BarrierBridgeIntervalInput {
                    right_local_variance: base.right_local_variance - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    right_local_variance: base.right_local_variance + bump,
                    ..base
                },
                reverse.right_local_variance,
            ),
            (
                BarrierBridgeIntervalInput {
                    dt: base.dt - bump,
                    ..base
                },
                BarrierBridgeIntervalInput {
                    dt: base.dt + bump,
                    ..base
                },
                reverse.dt,
            ),
        ];
        for (down, up, analytic) in cases {
            let finite_difference = finite_difference(down, up);
            assert!(
                (analytic - finite_difference).abs() < 2.0e-8,
                "analytic={analytic}, finite_difference={finite_difference}"
            );
        }
    }

    #[test]
    fn endpoint_touch_and_zero_variance_are_deterministic() {
        let touched = BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
            left_state: 100.0,
            ..input(BarrierBridgeDirection::Up)
        })
        .expect("touched");
        assert_eq!(touched.status(), BarrierBridgeStatus::TouchedEndpoint);
        assert_eq!(touched.survival(), 0.0);
        assert_eq!(
            touched.reverse(1.0),
            BarrierBridgeIntervalAdjoints::default()
        );

        let zero = BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
            left_local_variance: 0.0,
            right_local_variance: 0.0,
            ..input(BarrierBridgeDirection::Up)
        })
        .expect("zero variance");
        assert_eq!(zero.status(), BarrierBridgeStatus::ZeroVariance);
        assert_eq!(zero.survival(), 1.0);
        assert_eq!(zero.reverse(1.0), BarrierBridgeIntervalAdjoints::default());
    }

    #[test]
    fn extreme_exponents_have_typed_deterministic_outcomes() {
        let underflow = BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
            left_state: 1.0 - f64::EPSILON,
            right_state: 1.0 - f64::EPSILON,
            left_barrier: 1.0,
            right_barrier: 1.0,
            left_local_variance: f64::MAX,
            right_local_variance: f64::MAX,
            dt: 1.0,
            direction: BarrierBridgeDirection::Up,
        })
        .expect("underflow");
        assert_eq!(underflow.status(), BarrierBridgeStatus::SurvivalUnderflow);
        assert_eq!(underflow.survival(), 0.0);

        let certain = BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
            left_state: f64::MIN_POSITIVE,
            right_state: f64::MIN_POSITIVE,
            left_barrier: f64::MAX,
            right_barrier: f64::MAX,
            left_local_variance: f64::MIN_POSITIVE,
            right_local_variance: f64::MIN_POSITIVE,
            dt: 1.0,
            direction: BarrierBridgeDirection::Up,
        })
        .expect("certain");
        assert_eq!(certain.status(), BarrierBridgeStatus::CertainSurvival);
        assert_eq!(certain.survival(), 1.0);

        let overflow = BarrierBridgeInterval::evaluate(BarrierBridgeIntervalInput {
            left_local_variance: f64::MAX,
            right_local_variance: f64::MAX,
            dt: 2.0,
            ..input(BarrierBridgeDirection::Up)
        });
        assert!(matches!(
            overflow,
            Err(BarrierBridgeError::NonFiniteIntegratedVariance { .. })
        ));
    }

    #[test]
    fn transformed_barrier_uses_affine_f_coordinate() {
        let identity = AffineDividendCoordinate::identity();
        assert_eq!(transformed_barrier(120.0, 100.0, identity), Ok(120.0));

        let event = EventId::new(1);
        let transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            PositiveF64::new(100.0, "spot").expect("spot"),
            vec![
                DividendEvent::new(
                    event,
                    0.5,
                    DividendQuote::fixed_cash_and_proportional(5.0, 0.2, event).expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("transform");
        let coordinate = transform.coordinate_after_time(0.5).expect("coordinate");
        let f_barrier = transformed_barrier(120.0, 100.0, coordinate).expect("barrier");
        assert!((coordinate.reconstruct_spot(transform.spot(), f_barrier) - 120.0).abs() < 1.0e-14);

        assert!(matches!(
            transformed_barrier(f64::INFINITY, 100.0, identity),
            Err(BarrierBridgeError::InvalidPositiveInput {
                field: "spot barrier",
                ..
            })
        ));

        let overflow_event = EventId::new(2);
        let overflow_transform = AffineDividendTransform::new(
            UnderlyingId::new(7),
            PositiveF64::new(1.0, "spot").expect("spot"),
            vec![
                DividendEvent::new(
                    overflow_event,
                    0.5,
                    DividendQuote::fixed_cash(f64::MAX, overflow_event).expect("quote"),
                )
                .expect("event"),
            ],
        )
        .expect("transform");
        let overflow_coordinate = overflow_transform
            .coordinate_after_time(0.5)
            .expect("coordinate");
        assert!(matches!(
            transformed_barrier(120.0, 100.0, overflow_coordinate),
            Err(BarrierBridgeError::InvalidTransformedBarrier { .. })
        ));
    }

    #[test]
    fn path_accumulates_in_log_space_and_reverses_every_interval() {
        let first = input(BarrierBridgeDirection::Up);
        let second = BarrierBridgeIntervalInput {
            left_state: first.right_state,
            right_state: 93.0,
            ..first
        };
        let path = BarrierBridgePath::evaluate(&[first, second]).expect("path");
        let first_evaluation = BarrierBridgeInterval::evaluate(first).expect("first");
        let second_evaluation = BarrierBridgeInterval::evaluate(second).expect("second");
        assert!(
            (path.log_survival()
                - first_evaluation.log_survival()
                - second_evaluation.log_survival())
            .abs()
                < 1.0e-15
        );
        assert_eq!(path.diagnostics().interval_count, 2);
        assert_eq!(path.diagnostics().finite_correction_count, 2);
        assert_eq!(path.reverse(3.0).len(), 2);
    }
}
