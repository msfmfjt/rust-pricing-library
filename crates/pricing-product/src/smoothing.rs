use pricing_core::{CoreError, PositiveF64};

/// Versioned compact C2 smoothing policy for payoff graph operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactC2Smoothing {
    half_width: PositiveF64,
}

impl CompactC2Smoothing {
    pub const POLICY_VERSION: u32 = 1;

    pub fn new(half_width: f64) -> Result<Self, CoreError> {
        Ok(Self {
            half_width: PositiveF64::new(half_width, "smoothing_half_width")?,
        })
    }

    #[must_use]
    pub const fn from_positive(half_width: PositiveF64) -> Self {
        Self { half_width }
    }

    #[must_use]
    pub const fn half_width(self) -> PositiveF64 {
        self.half_width
    }

    #[must_use]
    pub fn indicator(self, input: f64) -> SmoothingDerivatives {
        let half_width = self.half_width.get();
        if input <= -half_width {
            return SmoothingDerivatives::ZERO;
        }
        if input >= half_width {
            return SmoothingDerivatives {
                value: 1.0,
                first: 0.0,
                second: 0.0,
            };
        }

        let u = (input + half_width) / (2.0 * half_width);
        let u2 = u * u;
        let u3 = u2 * u;
        let value = u3 * (10.0 + u * (-15.0 + 6.0 * u));
        let one_minus_u = 1.0 - u;
        let first = 15.0 * u2 * one_minus_u * one_minus_u / half_width;
        let second = (120.0 * u3 - 180.0 * u2 + 60.0 * u) / (4.0 * half_width * half_width);
        SmoothingDerivatives {
            value,
            first,
            second,
        }
    }

    #[must_use]
    pub fn positive_part(self, input: f64) -> SmoothingDerivatives {
        let half_width = self.half_width.get();
        if input <= -half_width {
            return SmoothingDerivatives::ZERO;
        }
        if input >= half_width {
            return SmoothingDerivatives {
                value: input,
                first: 1.0,
                second: 0.0,
            };
        }

        let u = (input + half_width) / (2.0 * half_width);
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let indicator = u3 * (10.0 + u * (-15.0 + 6.0 * u));
        let indicator_first = 15.0 * u2 * (1.0 - u) * (1.0 - u) / half_width;
        let value = 2.0 * half_width * u4 * (2.5 + u * (-3.0 + u));
        SmoothingDerivatives {
            value,
            first: indicator,
            second: indicator_first,
        }
    }

    #[must_use]
    pub fn maximum(self, left: f64, right: f64) -> SmoothingBinaryDerivatives {
        let positive_part = self.positive_part(left - right);
        SmoothingBinaryDerivatives {
            value: right + positive_part.value,
            left_first: positive_part.first,
            right_first: 1.0 - positive_part.first,
        }
    }

    #[must_use]
    pub fn minimum(self, left: f64, right: f64) -> SmoothingBinaryDerivatives {
        let positive_part = self.positive_part(left - right);
        SmoothingBinaryDerivatives {
            value: left - positive_part.value,
            left_first: 1.0 - positive_part.first,
            right_first: positive_part.first,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothingDerivatives {
    pub value: f64,
    pub first: f64,
    pub second: f64,
}

impl SmoothingDerivatives {
    const ZERO: Self = Self {
        value: 0.0,
        first: 0.0,
        second: 0.0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothingBinaryDerivatives {
    pub value: f64,
    pub left_first: f64,
    pub right_first: f64,
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2.0e-15,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn indicator_and_positive_part_match_p0_reference_points() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/path-dependence/reference-cases-v0.1.json"
        ))
        .expect("fixture");
        let cases = fixture["smoothing_cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 7);
        for case in cases {
            let parse = |field: &str| {
                case[field]
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64")
            };
            let expected = &case["expected"];
            let expected_value = |field: &str| {
                expected[field]
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64")
            };
            let smoothing = CompactC2Smoothing::new(parse("half_width")).expect("smoothing");
            let indicator = smoothing.indicator(parse("input"));
            assert_close(indicator.value, expected_value("indicator"));
            assert_close(indicator.first, expected_value("indicator_first"));
            assert_close(indicator.second, expected_value("indicator_second"));
            let positive = smoothing.positive_part(parse("input"));
            assert_close(positive.value, expected_value("positive_part"));
            assert_close(positive.first, expected_value("positive_part_first"));
            assert_close(positive.second, expected_value("positive_part_second"));
        }
    }

    #[test]
    fn indicator_and_positive_part_obey_exterior_branches() {
        let smoothing = CompactC2Smoothing::new(2.0).expect("smoothing");
        let cases = [
            (-3.0, 0.0, 0.0, 0.0, 0.0),
            (-2.0, 0.0, 0.0, 0.0, 0.0),
            (
                -1.0,
                0.103_515_625,
                0.263_671_875,
                0.351_562_5,
                0.028_320_312_5,
            ),
            (0.0, 0.5, 0.468_75, 0.0, 0.312_5),
            (
                1.0,
                0.896_484_375,
                0.263_671_875,
                -0.351_562_5,
                1.028_320_312_5,
            ),
            (2.0, 1.0, 0.0, 0.0, 2.0),
            (3.0, 1.0, 0.0, 0.0, 3.0),
        ];
        for (input, value, first, second, positive_part) in cases {
            let indicator = smoothing.indicator(input);
            assert_close(indicator.value, value);
            assert_close(indicator.first, first);
            assert_close(indicator.second, second);
            let positive = smoothing.positive_part(input);
            assert_close(positive.value, positive_part);
            assert_close(positive.first, value);
            assert_close(positive.second, first);
        }
    }

    #[test]
    fn extrema_are_symmetric_translation_equivariant_and_partition_the_sum() {
        let smoothing = CompactC2Smoothing::new(4.0).expect("smoothing");
        for (left, right) in [(101.0, 99.0), (100.0, 100.0), (97.0, 101.0)] {
            let maximum = smoothing.maximum(left, right);
            let reversed = smoothing.maximum(right, left);
            let minimum = smoothing.minimum(left, right);
            assert_close(maximum.value, reversed.value);
            assert_close(minimum.value + maximum.value, left + right);
            assert_close(maximum.left_first + maximum.right_first, 1.0);
            assert_close(minimum.left_first + minimum.right_first, 1.0);
            assert_close(
                smoothing.maximum(left + 7.0, right + 7.0).value,
                maximum.value + 7.0,
            );
        }
    }

    #[test]
    fn analytic_derivatives_match_central_differences_inside_the_band() {
        let smoothing = CompactC2Smoothing::new(2.0).expect("smoothing");
        let first_step = 1.0e-5;
        let second_step = 3.0e-4;
        for input in [-1.75, -1.0, -0.25, 0.0, 0.25, 1.0, 1.75] {
            let indicator = smoothing.indicator(input);
            let numerical_first = (smoothing.indicator(input + first_step).value
                - smoothing.indicator(input - first_step).value)
                / (2.0 * first_step);
            let numerical_second = (smoothing.indicator(input + second_step).value
                - 2.0 * indicator.value
                + smoothing.indicator(input - second_step).value)
                / (second_step * second_step);
            assert!((indicator.first - numerical_first).abs() <= 2.0e-10);
            assert!((indicator.second - numerical_second).abs() <= 2.0e-8);

            let positive = smoothing.positive_part(input);
            let positive_first = (smoothing.positive_part(input + first_step).value
                - smoothing.positive_part(input - first_step).value)
                / (2.0 * first_step);
            assert!((positive.first - positive_first).abs() <= 2.0e-10);
            assert!((positive.second - numerical_first).abs() <= 2.0e-10);
        }
    }

    #[test]
    fn invalid_half_widths_are_rejected() {
        for half_width in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(CompactC2Smoothing::new(half_width).is_err());
        }
    }
}
