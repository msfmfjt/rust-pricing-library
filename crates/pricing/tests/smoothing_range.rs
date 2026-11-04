//! Compact smoothing must recover exact exterior payoffs at every finite scale.
use pricing::core::{Date, UnderlyingId};
use pricing::product::{
    CompactC2Smoothing, GraphLimitPolicy, SourceGraphBuilder, SourceOpcode as Op,
};

#[test]
fn exterior_extrema_select_the_original_operand_without_cancellation() {
    let smoothing = CompactC2Smoothing::new(2.0).unwrap();
    for (a, b) in [
        (101.0, 1e16),
        (101.0, -1e16),
        (f64::MIN_POSITIVE, 1.0),
        (-f64::MIN_POSITIVE, -1.0),
        (f64::MAX, -f64::MAX),
        (2.0, 0.0),
        (2.0_f64.next_up(), 0.0),
    ] {
        // Only use exterior points for this oracle.
        let width = if (a - b).abs() < 2.0 { 0.25 } else { 2.0 };
        let smoothing = if width == 2.0 {
            smoothing
        } else {
            CompactC2Smoothing::new(width).unwrap()
        };
        for (left, right) in [(a, b), (b, a)] {
            let maximum = smoothing.maximum(left, right);
            let minimum = smoothing.minimum(left, right);
            assert_eq!(
                maximum.value.to_bits(),
                left.max(right).to_bits(),
                "max({left}, {right})"
            );
            assert_eq!(
                minimum.value.to_bits(),
                left.min(right).to_bits(),
                "min({left}, {right})"
            );
            assert_eq!(maximum.left_first, f64::from(left > right));
            assert_eq!(minimum.left_first, f64::from(left < right));
            assert_eq!(maximum.right_first, 1.0 - maximum.left_first);
            assert_eq!(minimum.right_first, 1.0 - minimum.left_first);
        }
    }
}

#[test]
fn wide_transition_values_and_derivatives_match_exact_dyadic_reference_points() {
    for width in [2.0_f64.powi(1023), f64::MAX] {
        let smoothing = CompactC2Smoothing::new(width).unwrap();
        for (fraction, indicator, positive, first_coefficient) in [
            (-0.5, 0.103_515_625, 0.014_160_156_25, 0.527_343_75),
            (0.0, 0.5, 0.156_25, 0.937_5),
            (0.5, 0.896_484_375, 0.514_160_156_25, 0.527_343_75),
        ] {
            let actual = smoothing.indicator(fraction * width);
            assert_eq!(actual.value, indicator);
            assert_eq!(actual.first, first_coefficient / width);
            assert_eq!(actual.second, 0.0); // True curvature rounds below binary64.
            let actual = smoothing.positive_part(fraction * width);
            assert_eq!(actual.value, positive * width);
            assert_eq!(actual.first, indicator);
            assert_eq!(actual.second, first_coefficient / width);
        }
    }
}

#[test]
fn indicator_curvature_survives_overflowing_width_squares() {
    // h=2^e, x=-h/2 gives curvature 45/(32*h^2), exactly representable
    // here even though the width square overflows. x=h/2 changes its sign.
    for exponent in [512, 520, 530] {
        let width = 2.0_f64.powi(exponent);
        let expected = (45.0 / 32.0) * f64::from_bits(1 << (1074 - 2 * exponent));
        let smoothing = CompactC2Smoothing::new(width).unwrap();
        assert_eq!(smoothing.indicator(-0.5 * width).second, expected);
        assert_eq!(smoothing.indicator(0.5 * width).second, -expected);
        assert_eq!(smoothing.indicator(0.0).second, 0.0);
    }
}

#[test]
fn zero_curvature_is_not_nan_when_the_width_square_underflows() {
    for width in [1e-200, 1e-308, f64::from_bits(1)] {
        let midpoint = CompactC2Smoothing::new(width).unwrap().indicator(0.0);
        assert_eq!(midpoint.value, 0.5);
        assert_eq!(midpoint.second, 0.0);
        assert_eq!(midpoint.first, 0.9375 / width);
    }
}

#[test]
fn genuine_output_and_derivative_overflow_remain_nonfinite() {
    let wide = CompactC2Smoothing::new(f64::MAX).unwrap();
    assert_eq!(wide.maximum(f64::MAX, f64::MAX).value, f64::INFINITY);
    assert_eq!(wide.minimum(-f64::MAX, -f64::MAX).value, f64::NEG_INFINITY);
    let narrow = CompactC2Smoothing::new(2.0_f64.powi(-600)).unwrap();
    assert_eq!(
        narrow.indicator(-0.5 * narrow.half_width().get()).second,
        f64::INFINITY
    );
}

#[test]
fn subnormal_positive_parts_are_rounded_once_at_the_final_scale() {
    for bits in 1..=256 {
        let width = f64::from_bits(bits);
        // At zero the exact polynomial is 5*h/32. Integer ties-to-even is
        // independent of the implementation's floating-point products.
        let numerator = 5 * bits;
        let quotient = numerator / 32;
        let remainder = numerator % 32;
        let rounded =
            quotient + u64::from(remainder > 16 || (remainder == 16 && quotient % 2 == 1));
        let actual = CompactC2Smoothing::new(width).unwrap().positive_part(0.0);
        assert_eq!(actual.value.to_bits(), rounded, "width bits={bits}");
        assert_eq!(actual.first, 0.5);
    }
}

#[test]
fn ordinary_interior_arithmetic_keeps_all_existing_bits() {
    for width in [1e-100, 0.5, 2.0, 16.0, 1e50] {
        let smoothing = CompactC2Smoothing::new(width).unwrap();
        for fraction in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let input = fraction * width;
            let u = (input + width) / (2.0 * width);
            let u2 = u * u;
            let u3 = u2 * u;
            let u4 = u2 * u2;
            let value = u3 * (10.0 + u * (-15.0 + 6.0 * u));
            let first = 15.0 * u2 * (1.0 - u) * (1.0 - u) / width;
            let second = (120.0 * u3 - 180.0 * u2 + 60.0 * u) / (4.0 * width * width);
            let positive = 2.0 * width * u4 * (2.5 + u * (-3.0 + u));
            let actual = smoothing.indicator(input);
            assert_eq!(actual.value.to_bits(), value.to_bits());
            assert_eq!(actual.first.to_bits(), first.to_bits());
            assert_eq!(actual.second.to_bits(), second.to_bits());
            let actual = smoothing.positive_part(input);
            assert_eq!(actual.value.to_bits(), positive.to_bits());
            assert_eq!(actual.first.to_bits(), value.to_bits());
            assert_eq!(actual.second.to_bits(), first.to_bits());
            let maximum = smoothing.maximum(input, 0.0);
            let minimum = smoothing.minimum(input, 0.0);
            assert_eq!(maximum.value.to_bits(), positive.to_bits());
            assert_eq!(minimum.value.to_bits(), (input - positive).to_bits());
        }
    }
}

#[test]
fn graph_constant_folding_forward_and_both_adjoint_inputs_use_the_same_extrema() {
    let date: Date = "2027-01-01".parse().unwrap();
    let smoothing = CompactC2Smoothing::new(2.0).unwrap();
    for (a, b) in [(101.0, 1e16), (101.0, -1e16), (f64::MAX, -f64::MAX)] {
        for (left_value, right_value) in [(a, b), (b, a)] {
            for minimum in [false, true] {
                for constant in [false, true] {
                    let mut builder = SourceGraphBuilder::new();
                    let left = if constant {
                        builder.literal(left_value).unwrap()
                    } else {
                        builder
                            .push(Op::TerminalSpot {
                                underlying: UnderlyingId::new(1),
                                observation_date: date,
                            })
                            .unwrap()
                    };
                    let right = if constant {
                        builder.literal(right_value).unwrap()
                    } else {
                        builder
                            .push(Op::PreDividendSpot {
                                underlying: UnderlyingId::new(1),
                                observation_date: date,
                            })
                            .unwrap()
                    };
                    let output = builder
                        .push(if minimum {
                            Op::SmoothMinimum {
                                left,
                                right,
                                smoothing,
                            }
                        } else {
                            Op::SmoothMaximum {
                                left,
                                right,
                                smoothing,
                            }
                        })
                        .unwrap();
                    let compiled = builder
                        .finish(vec![output])
                        .compile(GraphLimitPolicy::DEFAULT)
                        .unwrap();
                    let expected = if minimum {
                        left_value.min(right_value)
                    } else {
                        left_value.max(right_value)
                    };
                    let forward = compiled
                        .evaluate_with_pre_dividend_spots(
                            |_, _| Some(left_value),
                            |_, _| Some(right_value),
                        )
                        .unwrap();
                    assert_eq!(forward, [expected]);
                    let reverse = compiled
                        .evaluate_single_with_observation_adjoints(
                            |_, _| Some(left_value),
                            |_, _| Some(right_value),
                        )
                        .unwrap();
                    assert_eq!(reverse.value, expected);
                    if !constant {
                        let left_selected = if minimum {
                            left_value < right_value
                        } else {
                            left_value > right_value
                        };
                        assert_eq!(reverse.terminal_adjoints[0].value, f64::from(left_selected));
                        assert_eq!(
                            reverse.pre_dividend_adjoints[0].value,
                            f64::from(!left_selected)
                        );
                    }
                }
            }
        }
    }
}
