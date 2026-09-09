use crate::MarketError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThetaRegion {
    ShortExtrapolated,
    Knot,
    Interpolated,
    LongExtrapolated,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThetaEvaluation {
    pub theta: f64,
    pub derivative: f64,
    pub region: ThetaRegion,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThetaPchip {
    times: Box<[f64]>,
    values: Box<[f64]>,
    derivatives: Box<[f64]>,
    terminal_slope: f64,
}

impl ThetaPchip {
    pub fn new(
        times: Vec<f64>,
        values: Vec<f64>,
        terminal_slope: f64,
    ) -> Result<Self, MarketError> {
        if times.len() != values.len() {
            return Err(MarketError::SurfaceKnotLengthMismatch {
                times: times.len(),
                values: values.len(),
            });
        }
        if times.len() < 2 {
            return Err(MarketError::InvalidSurfaceKnotCount { count: times.len() });
        }
        for (index, time) in times.iter().copied().enumerate() {
            if !time.is_finite() || time <= 0.0 {
                return Err(MarketError::InvalidSurfaceKnotTime {
                    index,
                    bits: time.to_bits(),
                });
            }
        }
        for (left_index, pair) in times.windows(2).enumerate() {
            if pair[1] <= pair[0] {
                return Err(MarketError::UnsortedSurfaceKnots {
                    left_index,
                    left_bits: pair[0].to_bits(),
                    right_bits: pair[1].to_bits(),
                });
            }
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(MarketError::InvalidTheta {
                    index,
                    bits: value.to_bits(),
                });
            }
        }
        for (left_index, pair) in values.windows(2).enumerate() {
            if pair[1] < pair[0] {
                return Err(MarketError::DecreasingTheta {
                    left_index,
                    left_bits: pair[0].to_bits(),
                    right_bits: pair[1].to_bits(),
                });
            }
        }
        if !terminal_slope.is_finite() || terminal_slope < 0.0 {
            return Err(MarketError::InvalidSurfaceParameter {
                parameter: "terminal_slope",
                bits: terminal_slope.to_bits(),
            });
        }

        let derivatives = pchip_derivatives(&times, &values).into_boxed_slice();
        Ok(Self {
            times: times.into_boxed_slice(),
            values: values.into_boxed_slice(),
            derivatives,
            terminal_slope,
        })
    }

    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub fn derivatives(&self) -> &[f64] {
        &self.derivatives
    }

    #[must_use]
    pub const fn terminal_slope(&self) -> f64 {
        self.terminal_slope
    }

    pub fn evaluate(&self, time: f64) -> Result<ThetaEvaluation, MarketError> {
        if !time.is_finite() || time < 0.0 {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "time",
                bits: time.to_bits(),
            });
        }
        let time = if time == 0.0 { 0.0 } else { time };
        if time < self.times[0] {
            let derivative = self.values[0] / self.times[0];
            return Ok(ThetaEvaluation {
                theta: time * derivative,
                derivative,
                region: ThetaRegion::ShortExtrapolated,
            });
        }

        match self.times.binary_search_by(|value| value.total_cmp(&time)) {
            Ok(index) => Ok(ThetaEvaluation {
                theta: self.values[index],
                derivative: self.derivatives[index],
                region: ThetaRegion::Knot,
            }),
            Err(right) if right < self.times.len() => {
                let left = right - 1;
                let width = self.times[right] - self.times[left];
                let scaled = (time - self.times[left]) / width;
                let scaled_squared = scaled * scaled;
                let scaled_cubed = scaled_squared * scaled;
                let h00 = 2.0 * scaled_cubed - 3.0 * scaled_squared + 1.0;
                let h10 = scaled_cubed - 2.0 * scaled_squared + scaled;
                let h01 = -2.0 * scaled_cubed + 3.0 * scaled_squared;
                let h11 = scaled_cubed - scaled_squared;
                let theta = h00 * self.values[left]
                    + h10 * width * self.derivatives[left]
                    + h01 * self.values[right]
                    + h11 * width * self.derivatives[right];

                let h00_derivative = (6.0 * scaled_squared - 6.0 * scaled) / width;
                let h10_derivative = 3.0 * scaled_squared - 4.0 * scaled + 1.0;
                let h01_derivative = (-6.0 * scaled_squared + 6.0 * scaled) / width;
                let h11_derivative = 3.0 * scaled_squared - 2.0 * scaled;
                let derivative = h00_derivative * self.values[left]
                    + h10_derivative * self.derivatives[left]
                    + h01_derivative * self.values[right]
                    + h11_derivative * self.derivatives[right];
                Ok(ThetaEvaluation {
                    theta,
                    derivative,
                    region: ThetaRegion::Interpolated,
                })
            }
            Err(_) => {
                let last = self.times.len() - 1;
                Ok(ThetaEvaluation {
                    theta: self.values[last]
                        + self.terminal_slope * (time - self.times[last]),
                    derivative: self.terminal_slope,
                    region: ThetaRegion::LongExtrapolated,
                })
            }
        }
    }
}

fn pchip_derivatives(times: &[f64], values: &[f64]) -> Vec<f64> {
    let intervals = times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let secants = values
        .windows(2)
        .zip(&intervals)
        .map(|(pair, width)| (pair[1] - pair[0]) / width)
        .collect::<Vec<_>>();

    if values.len() == 2 {
        return vec![secants[0], secants[0]];
    }

    let mut derivatives = vec![0.0; values.len()];
    derivatives[0] = endpoint_derivative(intervals[0], intervals[1], secants[0], secants[1]);
    for (index, derivative) in derivatives
        .iter_mut()
        .enumerate()
        .take(values.len() - 1)
        .skip(1)
    {
        let left = secants[index - 1];
        let right = secants[index];
        *derivative = if left == 0.0 || right == 0.0 || left.signum() != right.signum() {
            0.0
        } else {
            let left_width = intervals[index - 1];
            let right_width = intervals[index];
            let weight_one = 2.0 * right_width + left_width;
            let weight_two = right_width + 2.0 * left_width;
            (weight_one + weight_two) / (weight_one / left + weight_two / right)
        };
    }
    let last_interval = intervals.len() - 1;
    derivatives[values.len() - 1] = endpoint_derivative(
        intervals[last_interval],
        intervals[last_interval - 1],
        secants[last_interval],
        secants[last_interval - 1],
    );
    derivatives
}

fn endpoint_derivative(
    adjacent_width: f64,
    next_width: f64,
    adjacent_secant: f64,
    next_secant: f64,
) -> f64 {
    let mut derivative = ((2.0 * adjacent_width + next_width) * adjacent_secant
        - adjacent_width * next_secant)
        / (adjacent_width + next_width);
    if derivative.signum() != adjacent_secant.signum() {
        derivative = 0.0;
    } else if adjacent_secant.signum() != next_secant.signum()
        && derivative.abs() > 3.0 * adjacent_secant.abs()
    {
        derivative = 3.0 * adjacent_secant;
    }
    derivative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pchip_preserves_knots_and_monotonicity() {
        let curve = ThetaPchip::new(
            vec![0.25, 1.0, 2.0, 5.0],
            vec![0.01, 0.04, 0.06, 0.09],
            0.005,
        )
        .expect("valid theta curve");
        for (time, theta) in curve.times().iter().zip(curve.values()) {
            assert_eq!(curve.evaluate(*time).expect("knot").theta, *theta);
        }
        let mut previous = curve.evaluate(0.0).expect("origin").theta;
        for index in 1..=600 {
            let time = f64::from(index) / 100.0;
            let value = curve.evaluate(time).expect("curve value");
            assert!(value.theta >= previous);
            assert!(value.derivative >= -1.0e-15);
            previous = value.theta;
        }
    }

    #[test]
    fn pchip_derivative_matches_central_difference() {
        let curve = ThetaPchip::new(
            vec![0.25, 1.0, 2.0, 5.0],
            vec![0.01, 0.04, 0.06, 0.09],
            0.005,
        )
        .expect("valid theta curve");
        for time in [0.5, 1.5, 3.0] {
            let bump = 1.0e-6;
            let finite_difference = (curve.evaluate(time + bump).expect("up").theta
                - curve.evaluate(time - bump).expect("down").theta)
                / (2.0 * bump);
            let analytic = curve.evaluate(time).expect("value").derivative;
            assert!((analytic - finite_difference).abs() < 1.0e-10);
        }
    }

    #[test]
    fn extrapolation_rules_are_explicit() {
        let curve = ThetaPchip::new(vec![0.5, 1.0], vec![0.02, 0.03], 0.004)
            .expect("valid theta curve");
        assert_eq!(
            curve.evaluate(0.25).expect("short"),
            ThetaEvaluation {
                theta: 0.01,
                derivative: 0.04,
                region: ThetaRegion::ShortExtrapolated,
            }
        );
        assert_eq!(
            curve.evaluate(2.0).expect("long"),
            ThetaEvaluation {
                theta: 0.034,
                derivative: 0.004,
                region: ThetaRegion::LongExtrapolated,
            }
        );
    }

    #[test]
    fn invalid_theta_curves_are_rejected() {
        assert!(ThetaPchip::new(vec![1.0], vec![0.04], 0.01).is_err());
        assert!(ThetaPchip::new(vec![0.0, 1.0], vec![0.01, 0.04], 0.01).is_err());
        assert!(ThetaPchip::new(vec![1.0, 1.0], vec![0.01, 0.04], 0.01).is_err());
        assert!(ThetaPchip::new(vec![0.5, 1.0], vec![0.04, 0.03], 0.01).is_err());
        assert!(ThetaPchip::new(vec![0.5, 1.0], vec![0.02, 0.03], -0.01).is_err());
    }

    #[test]
    fn plateau_has_zero_adjacent_derivatives() {
        let curve = ThetaPchip::new(
            vec![0.25, 0.5, 1.0, 2.0],
            vec![0.01, 0.02, 0.02, 0.04],
            0.01,
        )
        .expect("valid theta curve");
        assert_eq!(curve.derivatives()[1], 0.0);
        assert_eq!(curve.derivatives()[2], 0.0);
    }
}
