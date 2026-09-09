use crate::{
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
