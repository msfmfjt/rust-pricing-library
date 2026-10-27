use crate::market::{
    ImpliedVarianceSurface, MarketError, SurfaceValidationTolerance, ThetaRegion,
    TotalVarianceDerivatives,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EssviSlice {
    time: f64,
    theta: f64,
    psi: f64,
    rho_psi: f64,
}

impl EssviSlice {
    pub fn new(time: f64, theta: f64, psi: f64, rho_psi: f64) -> Result<Self, MarketError> {
        for (field, value) in [
            ("time", time),
            ("theta", theta),
            ("psi", psi),
            ("rho_psi", rho_psi),
        ] {
            if !value.is_finite() {
                return Err(MarketError::InvalidEssviSlice {
                    index: 0,
                    field,
                    bits: value.to_bits(),
                });
            }
        }
        if time <= 0.0 {
            return Err(MarketError::InvalidEssviSlice {
                index: 0,
                field: "time",
                bits: time.to_bits(),
            });
        }
        if theta <= 0.0 {
            return Err(MarketError::InvalidEssviSlice {
                index: 0,
                field: "theta",
                bits: theta.to_bits(),
            });
        }
        if psi <= 0.0 {
            return Err(MarketError::InvalidEssviSlice {
                index: 0,
                field: "psi",
                bits: psi.to_bits(),
            });
        }
        let rho = rho_psi / psi;
        if !rho.is_finite() || rho.abs() >= 1.0 {
            return Err(MarketError::InvalidEssviSlice {
                index: 0,
                field: "rho",
                bits: rho.to_bits(),
            });
        }
        Ok(Self {
            time,
            theta,
            psi,
            rho_psi,
        })
    }

    #[must_use]
    pub const fn time(self) -> f64 {
        self.time
    }

    #[must_use]
    pub const fn theta(self) -> f64 {
        self.theta
    }

    #[must_use]
    pub const fn psi(self) -> f64 {
        self.psi
    }

    #[must_use]
    pub const fn rho_psi(self) -> f64 {
        self.rho_psi
    }

    #[must_use]
    pub fn rho(self) -> f64 {
        self.rho_psi / self.psi
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EssviParameters {
    pub theta: f64,
    pub psi: f64,
    pub rho_psi: f64,
    pub theta_derivative: f64,
    pub psi_derivative: f64,
    pub rho_psi_derivative: f64,
    pub region: ThetaRegion,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EssviSurface {
    slices: Box<[EssviSlice]>,
    terminal_theta_slope: f64,
    tolerance: SurfaceValidationTolerance,
}

impl EssviSurface {
    pub fn new(
        slices: Vec<EssviSlice>,
        terminal_theta_slope: f64,
        tolerance: SurfaceValidationTolerance,
    ) -> Result<Self, MarketError> {
        if slices.len() < 2 {
            return Err(MarketError::InvalidEssviSliceCount {
                count: slices.len(),
            });
        }
        if !terminal_theta_slope.is_finite() || terminal_theta_slope < 0.0 {
            return Err(MarketError::InvalidSurfaceParameter {
                parameter: "terminal_theta_slope",
                bits: terminal_theta_slope.to_bits(),
            });
        }

        for (index, slice) in slices.iter().copied().enumerate() {
            validate_slice(index, slice, tolerance)?;
        }
        for (left_index, pair) in slices.windows(2).enumerate() {
            let left = pair[0];
            let right = pair[1];
            if right.time <= left.time {
                return Err(MarketError::UnsortedEssviSlices {
                    left_index,
                    left_bits: left.time.to_bits(),
                    right_bits: right.time.to_bits(),
                });
            }
            if right.theta < left.theta {
                return Err(MarketError::InconsistentEssviSlices {
                    left_index,
                    condition: "theta_non_decreasing",
                    left_bits: left.theta.to_bits(),
                    right_bits: right.theta.to_bits(),
                });
            }
            if right.psi < left.psi {
                return Err(MarketError::InconsistentEssviSlices {
                    left_index,
                    condition: "psi_non_decreasing",
                    left_bits: left.psi.to_bits(),
                    right_bits: right.psi.to_bits(),
                });
            }
            let rho_psi_change = (right.rho_psi - left.rho_psi).abs();
            let psi_change = right.psi - left.psi;
            if !tolerance.permits_non_strict(rho_psi_change, psi_change) {
                return Err(MarketError::InconsistentEssviSlices {
                    left_index,
                    condition: "rho_psi_change_bound",
                    left_bits: rho_psi_change.to_bits(),
                    right_bits: psi_change.to_bits(),
                });
            }
        }

        Ok(Self {
            slices: slices.into_boxed_slice(),
            terminal_theta_slope,
            tolerance,
        })
    }

    #[must_use]
    pub fn slices(&self) -> &[EssviSlice] {
        &self.slices
    }

    #[must_use]
    pub const fn terminal_theta_slope(&self) -> f64 {
        self.terminal_theta_slope
    }

    #[must_use]
    pub const fn tolerance(&self) -> SurfaceValidationTolerance {
        self.tolerance
    }

    pub fn parameters(&self, time: f64) -> Result<EssviParameters, MarketError> {
        if !time.is_finite() || time <= 0.0 {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "time",
                bits: time.to_bits(),
            });
        }
        let first = self.slices[0];
        if time < first.time {
            let inverse_time = 1.0 / first.time;
            let scale = time * inverse_time;
            return Ok(EssviParameters {
                theta: scale * first.theta,
                psi: scale * first.psi,
                rho_psi: scale * first.rho_psi,
                theta_derivative: first.theta * inverse_time,
                psi_derivative: first.psi * inverse_time,
                rho_psi_derivative: first.rho_psi * inverse_time,
                region: ThetaRegion::ShortExtrapolated,
            });
        }

        match self
            .slices
            .binary_search_by(|slice| slice.time.total_cmp(&time))
        {
            Ok(index) if index + 1 < self.slices.len() => Ok(interpolate(
                self.slices[index],
                self.slices[index + 1],
                time,
                ThetaRegion::Knot,
            )),
            Ok(index) => Ok(long_parameters(
                self.slices[index],
                time,
                self.terminal_theta_slope,
                ThetaRegion::Knot,
            )),
            Err(right) if right < self.slices.len() => Ok(interpolate(
                self.slices[right - 1],
                self.slices[right],
                time,
                ThetaRegion::Interpolated,
            )),
            Err(_) => Ok(long_parameters(
                self.slices[self.slices.len() - 1],
                time,
                self.terminal_theta_slope,
                ThetaRegion::LongExtrapolated,
            )),
        }
    }
}

impl ImpliedVarianceSurface for EssviSurface {
    fn total_variance_derivatives(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        if !log_moneyness.is_finite() {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "log_moneyness",
                bits: log_moneyness.to_bits(),
            });
        }
        let parameters = self.parameters(time)?;
        evaluate_total_variance(time, log_moneyness, parameters)
    }
}

fn validate_slice(
    index: usize,
    slice: EssviSlice,
    tolerance: SurfaceValidationTolerance,
) -> Result<(), MarketError> {
    let rho = slice.rho();
    let wing = slice.psi * (1.0 + rho.abs());
    if !wing.is_finite() || !tolerance.permits_strict(wing, 4.0) {
        return Err(MarketError::EssviSliceAdmissibilityViolation {
            index,
            condition: "strict_wing_slope",
            left_bits: wing.to_bits(),
            right_bits: 4.0_f64.to_bits(),
        });
    }
    let curvature = slice.psi * slice.psi * (1.0 + rho.abs());
    let limit = 4.0 * slice.theta;
    if !curvature.is_finite()
        || !limit.is_finite()
        || !tolerance.permits_non_strict(curvature, limit)
    {
        return Err(MarketError::EssviSliceAdmissibilityViolation {
            index,
            condition: "curvature_bound",
            left_bits: curvature.to_bits(),
            right_bits: limit.to_bits(),
        });
    }
    Ok(())
}

fn interpolate(
    left: EssviSlice,
    right: EssviSlice,
    time: f64,
    region: ThetaRegion,
) -> EssviParameters {
    let width = right.time - left.time;
    let weight = (time - left.time) / width;
    let theta_derivative = (right.theta - left.theta) / width;
    let psi_derivative = (right.psi - left.psi) / width;
    let rho_psi_derivative = (right.rho_psi - left.rho_psi) / width;
    EssviParameters {
        theta: left.theta + weight * (right.theta - left.theta),
        psi: left.psi + weight * (right.psi - left.psi),
        rho_psi: left.rho_psi + weight * (right.rho_psi - left.rho_psi),
        theta_derivative,
        psi_derivative,
        rho_psi_derivative,
        region,
    }
}

fn long_parameters(
    last: EssviSlice,
    time: f64,
    terminal_theta_slope: f64,
    region: ThetaRegion,
) -> EssviParameters {
    EssviParameters {
        theta: last.theta + terminal_theta_slope * (time - last.time),
        psi: last.psi,
        rho_psi: last.rho_psi,
        theta_derivative: terminal_theta_slope,
        psi_derivative: 0.0,
        rho_psi_derivative: 0.0,
        region,
    }
}

fn evaluate_total_variance(
    time: f64,
    log_moneyness: f64,
    parameters: EssviParameters,
) -> Result<TotalVarianceDerivatives, MarketError> {
    let phi = parameters.psi / parameters.theta;
    let rho = parameters.rho_psi / parameters.psi;
    let rho_derivative = (parameters.rho_psi_derivative * parameters.psi
        - parameters.rho_psi * parameters.psi_derivative)
        / (parameters.psi * parameters.psi);
    let phi_derivative = (parameters.psi_derivative * parameters.theta
        - parameters.psi * parameters.theta_derivative)
        / (parameters.theta * parameters.theta);
    let y = phi * log_moneyness;
    let y_derivative = log_moneyness * phi_derivative;
    let a = y + rho;
    let q = (a * a + 1.0 - rho * rho).sqrt();
    let q_derivative = (a * (y_derivative + rho_derivative) - rho * rho_derivative) / q;
    let shape = 1.0 + rho * y + q;
    let shape_derivative = rho_derivative * y + rho * y_derivative + q_derivative;
    let total_variance = parameters.theta * shape / 2.0;
    let log_moneyness_derivative = parameters.theta * phi * (rho + a / q) / 2.0;
    let log_moneyness_second_derivative =
        parameters.theta * phi * phi * (1.0 - rho * rho) / (2.0 * q * q * q);
    let time_derivative =
        parameters.theta_derivative * shape / 2.0 + parameters.theta * shape_derivative / 2.0;

    let result = TotalVarianceDerivatives {
        total_variance,
        log_moneyness_derivative,
        log_moneyness_second_derivative,
        time_derivative,
        theta: parameters.theta,
        theta_derivative: parameters.theta_derivative,
        theta_region: parameters.region,
    };
    for (field, value) in [
        ("total_variance", result.total_variance),
        ("log_moneyness_derivative", result.log_moneyness_derivative),
        (
            "log_moneyness_second_derivative",
            result.log_moneyness_second_derivative,
        ),
        ("time_derivative", result.time_derivative),
    ] {
        if !value.is_finite() {
            return Err(MarketError::NonFiniteSurfaceValue {
                field,
                time_bits: time.to_bits(),
                log_moneyness_bits: log_moneyness.to_bits(),
                value_bits: value.to_bits(),
            });
        }
    }
    if result.total_variance <= 0.0 {
        return Err(MarketError::NonPositiveSurfaceValue {
            field: "total_variance",
            time_bits: time.to_bits(),
            log_moneyness_bits: log_moneyness.to_bits(),
            value_bits: result.total_variance.to_bits(),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_surface() -> EssviSurface {
        EssviSurface::new(
            vec![
                EssviSlice::new(1.0, 0.04, 0.2, -0.08).expect("first slice"),
                EssviSlice::new(2.0, 0.09, 0.25, -0.05).expect("second slice"),
            ],
            0.02,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("valid eSSVI surface")
    }

    fn parameter_policy_surface(time_scale: f64) -> EssviSurface {
        EssviSurface::new(
            vec![
                EssviSlice::new(2.0 * time_scale, 0.25, 0.25, -0.125).expect("first"),
                EssviSlice::new(4.0 * time_scale, 0.75, 0.5, 0.0).expect("second"),
                EssviSlice::new(8.0 * time_scale, 1.0, 1.0, -0.25).expect("third"),
            ],
            0.03125 / time_scale,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("admissible unequal-interval surface")
    }

    fn parameter_components(parameters: EssviParameters) -> [f64; 6] {
        [
            parameters.theta,
            parameters.psi,
            parameters.rho_psi,
            parameters.theta_derivative,
            parameters.psi_derivative,
            parameters.rho_psi_derivative,
        ]
    }

    fn surface_with_candidate(
        candidate: EssviSlice,
        tolerance: SurfaceValidationTolerance,
    ) -> Result<EssviSurface, MarketError> {
        EssviSurface::new(
            vec![
                EssviSlice::new(1.0, 0.0625, 0.125, 0.0).expect("first slice"),
                candidate,
            ],
            0.125,
            tolerance,
        )
    }

    #[test]
    fn slice_accessors_preserve_valid_inputs_and_derived_correlation() {
        for (time, theta, psi, rho_psi, rho) in [
            (2.0, 0.5, 0.25, -0.125, -0.5),
            (4.0, 2.0, 0.5, 0.125, 0.25),
            (8.0, 4.0, 2.0, -1.0, -0.5),
            (
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                f64::MIN_POSITIVE,
                -0.0,
                -0.0,
            ),
            (f64::MAX, f64::MAX, f64::MAX, 0.0, 0.0),
        ] {
            let slice =
                EssviSlice::new(time, theta, psi, rho_psi).expect("syntactically valid slice");
            for (actual, expected) in [
                (slice.time(), time),
                (slice.theta(), theta),
                (slice.psi(), psi),
                (slice.rho_psi(), rho_psi),
                (slice.rho(), rho),
            ] {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn slice_errors_preserve_field_input_bits_and_validation_order() {
        let fields = ["time", "theta", "psi", "rho_psi"];
        let nan = f64::from_bits(0x7ff8_0000_0000_0042);
        for (index, field) in fields.into_iter().enumerate() {
            for value in [f64::INFINITY, f64::NEG_INFINITY, nan] {
                let mut inputs = [2.0, 0.5, 0.25, -0.125];
                inputs[index] = value;
                assert_eq!(
                    EssviSlice::new(inputs[0], inputs[1], inputs[2], inputs[3]),
                    Err(MarketError::InvalidEssviSlice {
                        index: 0,
                        field,
                        bits: value.to_bits()
                    })
                );
            }
        }
        for (index, field) in fields[..3].iter().copied().enumerate() {
            for value in [0.0, -0.0, -f64::from_bits(1), -1.0] {
                let mut inputs = [2.0, 0.5, 0.25, -0.125];
                inputs[index] = value;
                assert_eq!(
                    EssviSlice::new(inputs[0], inputs[1], inputs[2], inputs[3]),
                    Err(MarketError::InvalidEssviSlice {
                        index: 0,
                        field,
                        bits: value.to_bits()
                    })
                );
            }
        }
        // All inputs are checked for finiteness before positivity/correlation.
        assert_eq!(
            EssviSlice::new(-1.0, f64::INFINITY, 0.0, nan),
            Err(MarketError::InvalidEssviSlice {
                index: 0,
                field: "theta",
                bits: f64::INFINITY.to_bits()
            })
        );
        assert_eq!(
            EssviSlice::new(nan, f64::INFINITY, 0.0, 0.0),
            Err(MarketError::InvalidEssviSlice {
                index: 0,
                field: "time",
                bits: nan.to_bits()
            })
        );
    }

    #[test]
    fn slice_correlation_is_open_and_rejects_division_overflow() {
        for sign in [-1.0, 1.0] {
            let interior = sign * 1.0_f64.next_down();
            let slice = EssviSlice::new(1.0, 1.0, 1.0, interior).expect("interior correlation");
            assert_eq!(slice.rho(), interior);
            for rho in [sign, sign * 1.0_f64.next_up()] {
                assert_eq!(
                    EssviSlice::new(1.0, 1.0, 1.0, rho),
                    Err(MarketError::InvalidEssviSlice {
                        index: 0,
                        field: "rho",
                        bits: rho.to_bits()
                    })
                );
            }
            assert_eq!(
                EssviSlice::new(1.0, 1.0, f64::MIN_POSITIVE, sign * f64::MAX),
                Err(MarketError::InvalidEssviSlice {
                    index: 0,
                    field: "rho",
                    bits: (sign * f64::INFINITY).to_bits()
                })
            );
        }
    }

    #[test]
    fn surface_accessors_preserve_slices_slope_and_custom_tolerance() {
        let slices = vec![
            EssviSlice::new(2.0, 0.5, 0.25, -0.125).expect("first"),
            EssviSlice::new(4.0, 1.0, 0.5, 0.0).expect("second"),
            EssviSlice::new(8.0, 2.0, 1.0, 0.25).expect("third"),
        ];
        let tolerance = SurfaceValidationTolerance::new(0.03125, 0.0625).expect("custom tolerance");
        for slope in [-0.0, f64::MIN_POSITIVE, 0.125, f64::MAX] {
            let surface =
                EssviSurface::new(slices.clone(), slope, tolerance).expect("valid surface");
            assert_eq!(surface.slices(), slices);
            assert_eq!(surface.terminal_theta_slope().to_bits(), slope.to_bits());
            assert_eq!(surface.tolerance(), tolerance);
        }
    }

    #[test]
    fn surface_structure_errors_preserve_counts_slopes_and_pair_index() {
        let tolerance = SurfaceValidationTolerance::local_vol_vegakt_v1();
        let first = EssviSlice::new(1.0, 1.0, 0.5, 0.0).expect("first");
        for slices in [vec![], vec![first]] {
            assert_eq!(
                EssviSurface::new(slices.clone(), f64::NAN, tolerance),
                Err(MarketError::InvalidEssviSliceCount {
                    count: slices.len()
                })
            );
        }
        let second = EssviSlice::new(3.0, 2.0, 1.0, 0.0).expect("second");
        for slope in [
            -f64::from_bits(1),
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            assert_eq!(
                EssviSurface::new(vec![first, second], slope, tolerance),
                Err(MarketError::InvalidSurfaceParameter {
                    parameter: "terminal_theta_slope",
                    bits: slope.to_bits()
                })
            );
        }
        for time in [2.0_f64, 3.0] {
            let third = EssviSlice::new(time, 3.0, 1.5, 0.0).expect("third");
            assert_eq!(
                EssviSurface::new(vec![first, second, third], 0.0, tolerance),
                Err(MarketError::UnsortedEssviSlices {
                    left_index: 1,
                    left_bits: 3.0_f64.to_bits(),
                    right_bits: time.to_bits()
                })
            );
        }
    }

    #[test]
    fn surface_monotonicity_is_exact_despite_positive_tolerance() {
        let tolerance = SurfaceValidationTolerance::new(0.25, 0.0).expect("positive tolerance");
        let first = EssviSlice::new(1.0, 1.0, 0.5, 0.0).expect("first");
        let second = EssviSlice::new(2.0, 2.0, 1.0, 0.0).expect("second");
        for (theta, psi, condition, left, right) in [
            (
                2.0_f64.next_down(),
                1.0,
                "theta_non_decreasing",
                2.0_f64,
                2.0_f64.next_down(),
            ),
            (
                2.0,
                1.0_f64.next_down(),
                "psi_non_decreasing",
                1.0,
                1.0_f64.next_down(),
            ),
            (
                2.0_f64.next_down(),
                1.0_f64.next_down(),
                "theta_non_decreasing",
                2.0,
                2.0_f64.next_down(),
            ),
        ] {
            let third = EssviSlice::new(3.0, theta, psi, 0.0).expect("valid individual slice");
            assert_eq!(
                EssviSurface::new(vec![first, second, third], 0.0, tolerance),
                Err(MarketError::InconsistentEssviSlices {
                    left_index: 1,
                    condition,
                    left_bits: left.to_bits(),
                    right_bits: right.to_bits()
                })
            );
        }
    }

    #[test]
    fn surface_correlation_change_includes_the_tolerance_boundary() {
        for (absolute, relative, boundary, rejected_change) in [
            (0.0, 0.0, 0.5_f64, 0.2500000000000001_f64),
            (0.0625, 0.0, 0.5625, 0.3125000000000001),
            (0.0, 0.0625, 0.5625, 0.3125000000000001),
        ] {
            let tolerance = SurfaceValidationTolerance::new(absolute, relative).expect("tolerance");
            for sign in [-1.0, 1.0] {
                let left = EssviSlice::new(1.0, 4.0, 1.0, sign * 0.25).expect("left");
                for (index, magnitude) in [boundary.next_down(), boundary, boundary.next_up()]
                    .into_iter()
                    .enumerate()
                {
                    let right = EssviSlice::new(2.0, 4.0, 1.25, sign * magnitude).expect("right");
                    let actual = EssviSurface::new(vec![left, right], 0.0, tolerance);
                    if index < 2 {
                        assert!(actual.is_ok(), "at/below boundary: {actual:?}");
                    } else {
                        assert_eq!(
                            actual,
                            Err(MarketError::InconsistentEssviSlices {
                                left_index: 0,
                                condition: "rho_psi_change_bound",
                                left_bits: rejected_change.to_bits(),
                                right_bits: 0.25_f64.to_bits(),
                            })
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn slice_wing_bound_excludes_the_tolerance_boundary() {
        for (absolute, relative, boundary) in [
            (0.0, 0.0, 4.0_f64),
            (0.125, 0.0, 3.875),
            (0.0, 0.03125, 3.875),
        ] {
            let tolerance = SurfaceValidationTolerance::new(absolute, relative).expect("tolerance");
            for (psi, accepted) in [
                (boundary.next_down(), true),
                (boundary, false),
                (boundary.next_up(), false),
            ] {
                let candidate = EssviSlice::new(2.0, 4.0, psi, 0.0).expect("candidate");
                let actual = surface_with_candidate(candidate, tolerance);
                if accepted {
                    assert!(actual.is_ok(), "below wing boundary: {actual:?}");
                } else {
                    assert_eq!(
                        actual,
                        Err(MarketError::EssviSliceAdmissibilityViolation {
                            index: 1,
                            condition: "strict_wing_slope",
                            left_bits: psi.to_bits(),
                            right_bits: 4.0_f64.to_bits(),
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn slice_curvature_bound_includes_the_tolerance_boundary() {
        // psi=1 and |rho|=1/2 give exact curvature 3/2. The relative case
        // scales its tolerance by 3/2, giving a boundary theta of 45/128.
        for (absolute, relative, boundary) in [
            (0.0, 0.0, 0.375_f64),
            (0.125, 0.0, 0.34375),
            (0.0, 0.0625, 0.3515625),
        ] {
            let tolerance = SurfaceValidationTolerance::new(absolute, relative).expect("tolerance");
            for sign in [-1.0, 1.0] {
                for (theta, accepted) in [
                    (boundary.next_down(), false),
                    (boundary, true),
                    (boundary.next_up(), true),
                ] {
                    let candidate =
                        EssviSlice::new(2.0, theta, 1.0, sign * 0.5).expect("candidate");
                    let actual = surface_with_candidate(candidate, tolerance);
                    if accepted {
                        assert!(actual.is_ok(), "at/above theta boundary: {actual:?}");
                    } else {
                        assert_eq!(
                            actual,
                            Err(MarketError::EssviSliceAdmissibilityViolation {
                                index: 1,
                                condition: "curvature_bound",
                                left_bits: 1.5_f64.to_bits(),
                                right_bits: (4.0 * theta).to_bits(),
                            })
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn slice_admissibility_reports_exact_residuals_for_both_signs() {
        let tolerance = SurfaceValidationTolerance::new(0.0, 0.0).expect("zero tolerance");
        for sign in [-1.0, 1.0] {
            for (theta, psi, rho_psi, condition, left, right) in [
                (2.0, 3.0, 1.5, "strict_wing_slope", 4.5_f64, 4.0_f64),
                (0.25, 1.5, 0.75, "curvature_bound", 3.375, 1.0),
                (0.0625, 0.5, 0.125, "curvature_bound", 0.3125, 0.25),
                (0.0625, 4.0, 2.0, "strict_wing_slope", 6.0, 4.0),
            ] {
                let candidate =
                    EssviSlice::new(2.0, theta, psi, sign * rho_psi).expect("candidate");
                assert_eq!(
                    surface_with_candidate(candidate, tolerance),
                    Err(MarketError::EssviSliceAdmissibilityViolation {
                        index: 1,
                        condition,
                        left_bits: left.to_bits(),
                        right_bits: right.to_bits(),
                    })
                );
            }
        }
    }

    #[test]
    fn slice_admissibility_rejects_nonfinite_wing_and_curvature_limit() {
        for tolerance in [
            SurfaceValidationTolerance::new(0.0, 0.0).expect("zero tolerance"),
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        ] {
            let wing =
                EssviSlice::new(2.0, 1.0, f64::MAX, f64::MAX / 2.0).expect("finite slice inputs");
            assert_eq!(
                surface_with_candidate(wing, tolerance),
                Err(MarketError::EssviSliceAdmissibilityViolation {
                    index: 1,
                    condition: "strict_wing_slope",
                    left_bits: f64::INFINITY.to_bits(),
                    right_bits: 4.0_f64.to_bits(),
                })
            );
            let limit = EssviSlice::new(2.0, f64::MAX, 0.5, 0.25).expect("finite slice inputs");
            assert_eq!(
                surface_with_candidate(limit, tolerance),
                Err(MarketError::EssviSliceAdmissibilityViolation {
                    index: 1,
                    condition: "curvature_bound",
                    left_bits: 0.375_f64.to_bits(),
                    right_bits: f64::INFINITY.to_bits(),
                })
            );
        }
    }

    #[test]
    fn all_slice_admissibility_checks_precede_pair_checks() {
        let slices = vec![
            EssviSlice::new(2.0, 0.5, 0.5, 0.0).expect("first"),
            EssviSlice::new(1.0, 0.5, 0.5, 0.0).expect("unsorted second"),
            EssviSlice::new(3.0, 2.0, 3.0, 1.5).expect("inadmissible third"),
        ];
        assert_eq!(
            EssviSurface::new(
                slices,
                0.0,
                SurfaceValidationTolerance::local_vol_vegakt_v1()
            ),
            Err(MarketError::EssviSliceAdmissibilityViolation {
                index: 2,
                condition: "strict_wing_slope",
                left_bits: 4.5_f64.to_bits(),
                right_bits: 4.0_f64.to_bits(),
            })
        );
    }

    fn variance_components(value: TotalVarianceDerivatives) -> [f64; 4] {
        [
            value.total_variance,
            value.log_moneyness_derivative,
            value.log_moneyness_second_derivative,
            value.time_derivative,
        ]
    }

    fn assert_variance_close(actual: f64, expected: f64) {
        assert!(actual.is_finite(), "nonfinite result: {actual}");
        assert!(
            (actual - expected).abs() <= 5.0e-13 * expected.abs(),
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn variance_matches_independently_differenced_scalar_references() {
        let surface = parameter_policy_surface(1.0);
        // scripts/generate_essvi_variance_references.py evaluates only the
        // scalar formula in theta/psi/rho_psi. Derivatives are high-precision
        // centered differences along the specified local affine parameter path.
        // At knots that path follows the right-hand derivative, not both sides.
        let cases = [
            (
                0.5,
                -2.0,
                [
                    0.14517972847076846,
                    -0.04515347445384588,
                    0.0012655060480219662,
                    0.2903594569415369,
                ],
                0.0625,
                0.125,
                ThetaRegion::ShortExtrapolated,
            ),
            (
                2.0,
                0.5,
                [
                    0.20200317547305482,
                    -0.0625,
                    0.14433756729740643,
                    0.28496256729740643,
                ],
                0.25,
                0.25,
                ThetaRegion::Knot,
            ),
            (
                2.5,
                -0.5,
                [
                    0.434653454113693,
                    -0.14072657099403707,
                    0.06975785515064446,
                    0.22505251773683638,
                ],
                0.375,
                0.25,
                ThetaRegion::Interpolated,
            ),
            (
                3.5,
                1.0,
                [
                    0.6653070663297374,
                    0.101000977830996,
                    0.09296135414095905,
                    0.32058660512549436,
                ],
                0.625,
                0.25,
                ThetaRegion::Interpolated,
            ),
            (
                4.0,
                -1.5,
                [
                    0.9053300858899106,
                    -0.1767766952966369,
                    0.05892556509887896,
                    0.19965897801643787,
                ],
                0.75,
                0.0625,
                ThetaRegion::Knot,
            ),
            (
                5.0,
                0.25,
                [
                    0.8043869080239556,
                    -0.002382486540518712,
                    0.2385098112052254,
                    0.049297980807451985,
                ],
                0.8125,
                0.0625,
                ThetaRegion::Interpolated,
            ),
            (
                7.0,
                -0.75,
                [
                    1.1662072575121711,
                    -0.39272403710077436,
                    0.16267500272883148,
                    0.12359167261306908,
                ],
                0.9375,
                0.0625,
                ThetaRegion::Interpolated,
            ),
            (
                8.0,
                2.0,
                [1.25, 0.3125, 0.05859375, 0.01953125],
                1.0,
                0.03125,
                ThetaRegion::Knot,
            ),
            (
                10.0,
                -0.25,
                [
                    1.1378735199850616,
                    -0.349039246719847,
                    0.34726383642818864,
                    0.03090040318544411,
                ],
                1.0625,
                0.03125,
                ThetaRegion::LongExtrapolated,
            ),
        ];
        for (time, k, expected, theta, theta_t, region) in cases {
            let actual = surface
                .total_variance_derivatives(time, k)
                .expect("reference query");
            for (value, reference) in variance_components(actual).into_iter().zip(expected) {
                assert_variance_close(value, reference);
            }
            assert_eq!(actual.theta, theta);
            assert_eq!(actual.theta_derivative, theta_t);
            assert_eq!(actual.theta_region, region);
        }
    }

    #[test]
    fn variance_reflects_moneyness_with_all_slice_correlations() {
        let original = parameter_policy_surface(1.0);
        let reflected = EssviSurface::new(
            original
                .slices()
                .iter()
                .map(|slice| {
                    EssviSlice::new(slice.time(), slice.theta(), slice.psi(), -slice.rho_psi())
                        .expect("reflected slice")
                })
                .collect(),
            original.terminal_theta_slope(),
            original.tolerance(),
        )
        .expect("reflected surface");
        for time in [0.5, 2.0, 2.5, 3.5, 4.0, 5.0, 7.0, 8.0, 10.0] {
            for k in [-2.0, -0.25, 0.0, 0.5, 2.0] {
                let expected = original
                    .total_variance_derivatives(time, k)
                    .expect("original");
                let actual = reflected
                    .total_variance_derivatives(time, -k)
                    .expect("reflected");
                for (value, reference) in variance_components(actual).into_iter().zip([
                    expected.total_variance,
                    -expected.log_moneyness_derivative,
                    expected.log_moneyness_second_derivative,
                    expected.time_derivative,
                ]) {
                    assert_variance_close(value, reference);
                }
                assert_eq!(actual.theta, expected.theta);
                assert_eq!(actual.theta_derivative, expected.theta_derivative);
                assert_eq!(actual.theta_region, expected.theta_region);
            }
        }
    }

    #[test]
    fn variance_time_derivative_transforms_with_the_time_unit() {
        let original = parameter_policy_surface(1.0);
        for scale in [0.25, 0.5, 2.0, 4.0] {
            let scaled = parameter_policy_surface(scale);
            for time in [0.5, 2.0, 2.5, 3.5, 4.0, 5.0, 7.0, 8.0, 10.0] {
                for k in [-2.0, -0.25, 0.5, 2.0] {
                    let expected = original
                        .total_variance_derivatives(time, k)
                        .expect("original");
                    let actual = scaled
                        .total_variance_derivatives(time * scale, k)
                        .expect("scaled");
                    for (value, reference) in variance_components(actual).into_iter().zip([
                        expected.total_variance,
                        expected.log_moneyness_derivative,
                        expected.log_moneyness_second_derivative,
                        expected.time_derivative / scale,
                    ]) {
                        assert_variance_close(value, reference);
                    }
                    assert_eq!(actual.theta, expected.theta);
                    assert_eq!(actual.theta_derivative * scale, expected.theta_derivative);
                    assert_eq!(actual.theta_region, expected.theta_region);
                }
            }
        }
    }

    #[test]
    fn variance_query_errors_preserve_coordinate_bits_and_order() {
        let surface = parameter_policy_surface(1.0);
        let nan = f64::from_bits(0x7ff8_0000_0000_0042);
        for k in [f64::INFINITY, f64::NEG_INFINITY, nan] {
            for time in [0.0, -1.0, f64::INFINITY, nan, 2.0] {
                assert_eq!(
                    surface.total_variance_derivatives(time, k),
                    Err(MarketError::InvalidSurfaceQuery {
                        coordinate: "log_moneyness",
                        bits: k.to_bits(),
                    })
                );
            }
        }
        for time in [
            0.0,
            -0.0,
            -f64::from_bits(1),
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            nan,
        ] {
            assert_eq!(
                surface.total_variance_derivatives(time, 0.25),
                Err(MarketError::InvalidSurfaceQuery {
                    coordinate: "time",
                    bits: time.to_bits(),
                })
            );
        }
    }

    #[test]
    fn variance_rejects_nonfinite_arithmetic_with_query_context() {
        let surface = parameter_policy_surface(1.0);
        for k in [-f64::MAX, f64::MAX] {
            assert_eq!(
                surface.total_variance_derivatives(2.0, k),
                Err(MarketError::NonFiniteSurfaceValue {
                    field: "total_variance",
                    time_bits: 2.0_f64.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: f64::INFINITY.to_bits(),
                })
            );
        }

        let tiny = EssviSurface::new(
            [1.0, 2.0]
                .into_iter()
                .map(|time| {
                    EssviSlice::new(time, 1.0e-200, 1.0e-200, 0.0).expect("tiny flat slice")
                })
                .collect(),
            0.0,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("admissible tiny surface");
        // The quotient-rule denominators theta^2 and psi^2 underflow to zero.
        // Keep the finite-result rejection; do not freeze a computed NaN payload
        // or claim accurate evaluation over the whole finite input domain.
        match tiny.total_variance_derivatives(3.0, 0.5) {
            Err(MarketError::NonFiniteSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            }) => {
                assert_eq!(field, "time_derivative");
                assert_eq!(time_bits, 3.0_f64.to_bits());
                assert_eq!(log_moneyness_bits, 0.5_f64.to_bits());
                assert!(f64::from_bits(value_bits).is_nan());
            }
            other => panic!("expected a nonfinite time derivative: {other:?}"),
        }
    }

    #[test]
    fn flat_smile_has_zero_time_derivative() {
        let surface = EssviSurface::new(
            [2.0, 4.0, 8.0]
                .into_iter()
                .map(|time| EssviSlice::new(time, 0.25, 0.25, -0.125).expect("flat slice"))
                .collect(),
            0.0,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("flat surface");
        for k in [-2.0, -0.25, 0.0, 0.5, 2.0] {
            let first = surface
                .total_variance_derivatives(2.0, k)
                .expect("first slice");
            for time in [2.0, 3.0, 4.0, 5.0, 8.0, 10.0] {
                let actual = surface
                    .total_variance_derivatives(time, k)
                    .expect("flat smile");
                assert_eq!(actual.time_derivative, 0.0);
                assert_eq!(actual.theta_derivative, 0.0);
                assert_eq!(variance_components(actual), variance_components(first));
                assert!(actual.total_variance > 0.0);
                assert!(actual.log_moneyness_second_derivative > 0.0);
            }
        }
    }

    #[test]
    fn parameters_match_exact_piecewise_affine_references() {
        let surface = parameter_policy_surface(1.0);
        // Independently tabulated affine values and slopes, all exact dyadic
        // rationals. The slopes change at every knot; the last knot uses the
        // explicit terminal slope, not the preceding interpolation interval.
        let cases = [
            (
                0.5,
                [0.0625, 0.0625, -0.03125, 0.125, 0.125, -0.0625],
                ThetaRegion::ShortExtrapolated,
            ),
            (
                2.0,
                [0.25, 0.25, -0.125, 0.25, 0.125, 0.0625],
                ThetaRegion::Knot,
            ),
            (
                2.5,
                [0.375, 0.3125, -0.09375, 0.25, 0.125, 0.0625],
                ThetaRegion::Interpolated,
            ),
            (
                3.5,
                [0.625, 0.4375, -0.03125, 0.25, 0.125, 0.0625],
                ThetaRegion::Interpolated,
            ),
            (
                4.0,
                [0.75, 0.5, 0.0, 0.0625, 0.125, -0.0625],
                ThetaRegion::Knot,
            ),
            (
                5.0,
                [0.8125, 0.625, -0.0625, 0.0625, 0.125, -0.0625],
                ThetaRegion::Interpolated,
            ),
            (
                7.0,
                [0.9375, 0.875, -0.1875, 0.0625, 0.125, -0.0625],
                ThetaRegion::Interpolated,
            ),
            (8.0, [1.0, 1.0, -0.25, 0.03125, 0.0, 0.0], ThetaRegion::Knot),
            (
                10.0,
                [1.0625, 1.0, -0.25, 0.03125, 0.0, 0.0],
                ThetaRegion::LongExtrapolated,
            ),
        ];
        for (time, expected, region) in cases {
            let actual = surface.parameters(time).expect("valid query");
            assert_eq!(parameter_components(actual), expected, "time={time}");
            assert_eq!(actual.region, region, "time={time}");
        }
    }

    #[test]
    fn knot_neighbors_select_right_hand_parameter_slopes() {
        let surface = parameter_policy_surface(1.0);
        for (time, left_region, left, right_region, right) in [
            (
                2.0_f64,
                ThetaRegion::ShortExtrapolated,
                [0.125, 0.125, -0.0625],
                ThetaRegion::Interpolated,
                [0.25, 0.125, 0.0625],
            ),
            (
                4.0,
                ThetaRegion::Interpolated,
                [0.25, 0.125, 0.0625],
                ThetaRegion::Interpolated,
                [0.0625, 0.125, -0.0625],
            ),
            (
                8.0,
                ThetaRegion::Interpolated,
                [0.0625, 0.125, -0.0625],
                ThetaRegion::LongExtrapolated,
                [0.03125, 0.0, 0.0],
            ),
        ] {
            for (query, region, slopes) in [
                (time.next_down(), left_region, left),
                (time, ThetaRegion::Knot, right),
                (time.next_up(), right_region, right),
            ] {
                let actual = surface.parameters(query).expect("knot or neighbor");
                assert_eq!(actual.region, region, "query={query:?}");
                assert_eq!(parameter_components(actual)[3..], slopes, "query={query:?}");
            }
        }
    }

    #[test]
    fn parameters_transform_with_the_time_unit() {
        let original = parameter_policy_surface(1.0);
        for time_scale in [0.25, 0.5, 2.0, 4.0] {
            let scaled = parameter_policy_surface(time_scale);
            for time in [0.5, 2.0, 2.5, 3.5, 4.0, 5.0, 7.0, 8.0, 10.0] {
                let expected = original.parameters(time).expect("original");
                let actual = scaled.parameters(time * time_scale).expect("scaled");
                assert_eq!(actual.region, expected.region);
                let expected = parameter_components(expected);
                let actual = parameter_components(actual);
                assert_eq!(actual[..3], expected[..3]);
                for index in 3..6 {
                    assert_eq!(actual[index] * time_scale, expected[index]);
                }
            }
        }
    }

    #[test]
    fn flat_slices_and_zero_terminal_slope_preserve_parameters() {
        let surface = EssviSurface::new(
            [2.0, 4.0, 8.0]
                .into_iter()
                .map(|time| EssviSlice::new(time, 0.25, 0.25, -0.125).expect("flat slice"))
                .collect(),
            0.0,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("flat surface");
        for (time, region) in [
            (2.0, ThetaRegion::Knot),
            (3.0, ThetaRegion::Interpolated),
            (4.0, ThetaRegion::Knot),
            (7.0, ThetaRegion::Interpolated),
            (8.0, ThetaRegion::Knot),
            (16.0, ThetaRegion::LongExtrapolated),
        ] {
            let actual = surface.parameters(time).expect("flat query");
            assert_eq!(
                parameter_components(actual),
                [0.25, 0.25, -0.125, 0.0, 0.0, 0.0]
            );
            assert_eq!(actual.region, region);
        }
    }

    #[test]
    fn parameter_query_errors_preserve_time_bits() {
        let surface = parameter_policy_surface(1.0);
        for time in [
            0.0,
            -0.0,
            -f64::from_bits(1),
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            assert_eq!(
                surface.parameters(time),
                Err(MarketError::InvalidSurfaceQuery {
                    coordinate: "time",
                    bits: time.to_bits(),
                })
            );
        }
    }

    #[test]
    fn short_end_scales_the_smile_at_constant_rho() {
        let surface = parameter_policy_surface(1.0);
        for time in [0.125, 0.5, 1.0] {
            let parameters = surface.parameters(time).expect("short parameters");
            assert_eq!(parameters.rho_psi / parameters.psi, -0.5);
            assert_eq!(parameters.psi / parameters.theta, 1.0);
            for log_moneyness in [-2.0, -0.25, 0.0, 0.5, 2.0] {
                let first = surface
                    .total_variance_derivatives(2.0, log_moneyness)
                    .expect("first slice");
                let short = surface
                    .total_variance_derivatives(time, log_moneyness)
                    .expect("short smile");
                for (actual, expected) in [
                    (short.total_variance, first.total_variance * time / 2.0),
                    (
                        short.log_moneyness_derivative,
                        first.log_moneyness_derivative * time / 2.0,
                    ),
                    (
                        short.log_moneyness_second_derivative,
                        first.log_moneyness_second_derivative * time / 2.0,
                    ),
                    (short.time_derivative, first.total_variance / 2.0),
                ] {
                    assert!(actual.is_finite());
                    assert!(
                        (actual - expected).abs() <= 2.0e-14 * expected.abs(),
                        "actual={actual}, expected={expected}"
                    );
                }
                assert_eq!(short.theta_region, ThetaRegion::ShortExtrapolated);
            }
        }
    }

    #[test]
    fn atm_variance_uses_selected_theta_and_one_sided_derivative() {
        let surface = parameter_policy_surface(1.0);
        for (time, theta, derivative, region) in [
            (0.5, 0.0625, 0.125, ThetaRegion::ShortExtrapolated),
            (2.0, 0.25, 0.25, ThetaRegion::Knot),
            (2.5, 0.375, 0.25, ThetaRegion::Interpolated),
            (4.0, 0.75, 0.0625, ThetaRegion::Knot),
            (5.0, 0.8125, 0.0625, ThetaRegion::Interpolated),
            (8.0, 1.0, 0.03125, ThetaRegion::Knot),
            (10.0, 1.0625, 0.03125, ThetaRegion::LongExtrapolated),
        ] {
            let actual = surface
                .total_variance_derivatives(time, 0.0)
                .expect("ATM query");
            assert_eq!(actual.theta, theta);
            assert_eq!(actual.theta_derivative, derivative);
            assert_eq!(actual.theta_region, region);
            assert!((actual.total_variance - theta).abs() <= 2.0e-14 * theta);
            assert!((actual.time_derivative - derivative).abs() <= 2.0e-14 * derivative);
        }
    }

    #[test]
    fn interpolation_and_extrapolation_follow_v1_policy() {
        let surface = valid_surface();
        let short = surface.parameters(0.5).expect("short end");
        assert_eq!(short.theta, 0.02);
        assert_eq!(short.psi, 0.1);
        assert_eq!(short.rho_psi, -0.04);
        assert_eq!(short.region, ThetaRegion::ShortExtrapolated);

        let middle = surface.parameters(1.5).expect("middle");
        assert_eq!(middle.theta, 0.065);
        assert_eq!(middle.psi, 0.225);
        assert_eq!(middle.rho_psi, -0.065);
        assert_eq!(middle.region, ThetaRegion::Interpolated);

        let long = surface.parameters(3.0).expect("long end");
        assert_eq!(long.theta, 0.11);
        assert_eq!(long.psi, 0.25);
        assert_eq!(long.rho_psi, -0.05);
        assert_eq!(long.region, ThetaRegion::LongExtrapolated);
    }

    #[test]
    fn invalid_slices_and_inconsistent_pairs_are_rejected() {
        let tolerance = SurfaceValidationTolerance::local_vol_vegakt_v1();
        assert!(EssviSlice::new(0.0, 0.04, 0.2, -0.08).is_err());
        assert!(EssviSlice::new(1.0, 0.04, 0.2, 0.2).is_err());
        let first = EssviSlice::new(1.0, 0.04, 0.2, -0.08).expect("first");
        let decreasing_theta = EssviSlice::new(2.0, 0.03, 0.21, -0.07).expect("second");
        assert!(EssviSurface::new(vec![first, decreasing_theta], 0.01, tolerance).is_err());
        let excessive_change = EssviSlice::new(2.0, 0.09, 0.21, 0.0).expect("second");
        assert!(EssviSurface::new(vec![first, excessive_change], 0.01, tolerance).is_err());
    }

    #[test]
    fn analytic_time_derivative_matches_central_difference() {
        let surface = valid_surface();
        let time = 1.5;
        let log_moneyness = -0.1;
        let bump = 1.0e-6;
        let value = surface
            .total_variance_derivatives(time, log_moneyness)
            .expect("value");
        let up = surface
            .total_variance_derivatives(time + bump, log_moneyness)
            .expect("up")
            .total_variance;
        let down = surface
            .total_variance_derivatives(time - bump, log_moneyness)
            .expect("down")
            .total_variance;
        let finite_difference = (up - down) / (2.0 * bump);
        assert!((value.time_derivative - finite_difference).abs() < 1.0e-10);
    }
}
