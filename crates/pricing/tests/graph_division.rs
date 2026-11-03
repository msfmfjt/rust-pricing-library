//! Division pullbacks must not depend on an arbitrary finite payoff scale.
use pricing::core::{Date, UnderlyingId};
use pricing::product::{GraphError, GraphLimitPolicy, SourceGraphBuilder, SourceOpcode as Op};

fn weighted_ratio(
    x: f64,
    y: f64,
    weight: f64,
) -> Result<pricing::product::PayoffEvaluation, GraphError> {
    let date: Date = "2027-01-01".parse().unwrap();
    let mut b = SourceGraphBuilder::new();
    let numerator = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: date,
        })
        .unwrap();
    let denominator = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(2),
            observation_date: date,
        })
        .unwrap();
    let ratio = b
        .push(Op::Divide {
            numerator,
            denominator,
        })
        .unwrap();
    let weight = b.literal(weight).unwrap();
    let output = b
        .push(Op::Multiply {
            left: ratio,
            right: weight,
        })
        .unwrap();
    b.finish(vec![output])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap()
        .evaluate_single_with_terminal_adjoint(|u, _| Some(if u.get() == 1 { x } else { y }))
}

fn power(exponent: i32) -> f64 {
    if exponent >= -1022 {
        2.0_f64.powi(exponent)
    } else if exponent >= -1074 {
        f64::from_bits(1 << (exponent + 1074))
    } else {
        0.0
    }
}

#[test]
fn division_seed_product_range_does_not_destroy_representable_adjoints() {
    // (numerator exponent, denominator exponent, seed exponent). Expected
    // derivatives follow d(w*x/y)/dx = w/y and d(w*x/y)/dy = -w*x/y^2.
    for (x, y, w) in [
        (600, 500, 500),
        (-600, -500, -500),
        (-1074, -500, -500),
        (500, 1000, 500),
        (500, 800, 26),
        (500, 800, 25),
        (500, 800, 24),
        (500, 800, 23),
        (500, 600, 700),
        (500, -1, 521),
        (500, -1, 520),
    ] {
        for (sx, sy, sw) in [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)] {
            let pow = power;
            let out = weighted_ratio(sx * pow(x), sy * pow(y), sw * pow(w)).unwrap();
            assert_eq!(out.value, sx * sy * sw * pow(x - y + w));
            let adjoint = |id| {
                out.terminal_adjoints
                    .iter()
                    .find(|a| a.underlying.get() == id)
                    .unwrap()
                    .value
            };
            assert_eq!(adjoint(1), sy * sw * pow(w - y));
            assert_eq!(adjoint(2), -sx * sw * pow(w + x - 2 * y));
        }
    }
}

#[test]
fn zero_division_pullbacks_remain_zero_at_extreme_scales() {
    for y in [2.0_f64.powi(-1000), 2.0_f64.powi(1000)] {
        for sign in [-1.0, 1.0] {
            let out = weighted_ratio(sign * y, y, 0.0).unwrap();
            assert_eq!(out.value, 0.0);
            assert!(out.terminal_adjoints.iter().all(|a| a.value == 0.0));
            let out = weighted_ratio(0.0, y, sign).unwrap();
            assert_eq!(
                out.terminal_adjoints
                    .iter()
                    .find(|a| a.underlying.get() == 2)
                    .unwrap()
                    .value,
                0.0
            );
        }
    }
}

#[test]
fn ordinary_division_pullbacks_retain_the_existing_bits() {
    for (x, y, w) in [(1.25, 3.7, -0.3), (-91.1, 103.2, 2.5), (0.031, -0.75, 1.0)] {
        for exponent in [-500, -200, 0, 200, 500] {
            let x = x * power(exponent);
            let y = y * power(exponent);
            let out = weighted_ratio(x, y, w).unwrap();
            let derivative = out
                .terminal_adjoints
                .iter()
                .find(|a| a.underlying.get() == 2)
                .unwrap()
                .value;
            assert_eq!(derivative.to_bits(), (-w * x / (y * y)).to_bits());
        }
    }
}

#[test]
fn genuine_division_domain_and_adjoint_overflow_remain_errors() {
    assert!(matches!(
        weighted_ratio(1.0, 0.0, 1.0),
        Err(GraphError::NonFiniteRuntimeValue { .. })
    ));
    // Forward price and numerator adjoint are finite, denominator adjoint is not.
    assert!(matches!(
        weighted_ratio(2.0_f64.powi(500), 0.5, 2.0_f64.powi(522)),
        Err(GraphError::NonFiniteRuntimeAdjoint { .. })
    ));
    for seed in [2.0_f64.powi(26), 2.0_f64.powi(50)] {
        assert!(matches!(
            weighted_ratio(2.0_f64.powi(-200), 1.9 * 2.0_f64.powi(-600), seed),
            Err(GraphError::NonFiniteRuntimeAdjoint { .. })
        ));
    }
}

#[test]
fn division_pullbacks_match_independent_decimal_references() {
    // Reference values: Decimal.from_float for each input, 2500 decimal digits,
    // -w*x/(y*y), then one conversion to binary64. Bound these nonbinary
    // references to two ULPs; none is small enough for a false zero to pass.
    for (x, y, w, expected) in [
        (1.3e200, 7.7e200, -0.3, 6.577_837_746_668_916e-203_f64),
        (1.3e-200, -7.7e-200, 0.3, -6.577_837_746_668_916e197),
        (1.3e200, 7.7e150, 1e150, -2.192_612_582_222_972e48),
        (1.3e-200, 7.7e-150, 1e-150, -2.192_612_582_222_972e-52),
        (
            f64::from_bits(3),
            1.7e-150,
            1.3e-150,
            -6.667_321_864_293_638e-174,
        ),
        (1.3e200, 7.7e200, 1.4e100, -3.069_657_615_112_160_4e-102),
        (1.1e308, 1.3e308, 0.75, -4.881_656_804_733_725e-309),
        (1.3e-100, 1.7e100, 1.9e-10, -8.546_712_802_768e-311),
        (
            2.0_f64.powi(-200),
            1.5 * 2.0_f64.powi(-600),
            2.0_f64.powi(24),
            -7.989_747_266_054_737e307,
        ),
        (
            2.0_f64.powi(-200),
            1.9 * 2.0_f64.powi(-600),
            2.0_f64.powi(25),
            -9.959_518_752_699_812e307,
        ),
    ] {
        let out = weighted_ratio(x, y, w).unwrap();
        let actual = out
            .terminal_adjoints
            .iter()
            .find(|a| a.underlying.get() == 2)
            .unwrap()
            .value;
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 2,
            "x={x}, y={y}, w={w}: {actual:e} != {expected:e}"
        );
    }
}

#[test]
fn ratio_graph_preserves_both_spot_adjoints_across_scales() {
    let date: Date = "2027-01-01".parse().unwrap();
    for exponent in [-1000, -600, -520, -500, 0, 500, 520, 600, 1000] {
        for sign in [-1.0, 1.0] {
            let mut b = SourceGraphBuilder::new();
            let x = b
                .push(Op::TerminalSpot {
                    underlying: UnderlyingId::new(1),
                    observation_date: date,
                })
                .unwrap();
            let y = b
                .push(Op::PreDividendSpot {
                    underlying: UnderlyingId::new(2),
                    observation_date: date,
                })
                .unwrap();
            let scale = b.literal(sign * 2.0_f64.powi(exponent)).unwrap();
            let numerator = b
                .push(Op::Multiply {
                    left: x,
                    right: scale,
                })
                .unwrap();
            let denominator = b
                .push(Op::Multiply {
                    left: y,
                    right: scale,
                })
                .unwrap();
            let ratio = b
                .push(Op::Divide {
                    numerator,
                    denominator,
                })
                .unwrap();
            let compiled = b
                .finish(vec![ratio])
                .compile(GraphLimitPolicy::DEFAULT)
                .unwrap();
            let out = compiled
                .evaluate_single_with_observation_adjoints(|_, _| Some(4.0), |_, _| Some(2.0))
                .unwrap_or_else(|e| panic!("scale 2^{exponent}, sign {sign}: {e}"));
            // Powers of two make the analytical price and both derivatives exact.
            assert_eq!(out.value, 2.0);
            assert_eq!(out.terminal_adjoints[0].value, 0.5, "{exponent}, {sign}");
            assert_eq!(
                out.pre_dividend_adjoints[0].value, -1.0,
                "{exponent}, {sign}"
            );
        }
    }
}
