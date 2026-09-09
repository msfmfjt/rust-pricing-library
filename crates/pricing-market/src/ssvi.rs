use crate::{MarketError, ThetaPchip, ThetaRegion};

const SMALL_HESTON_Z: f64 = 1.0e-4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceValidationTolerance {
    absolute: f64,
    relative: f64,
}

impl SurfaceValidationTolerance {
    pub fn new(absolute: f64, relative: f64) -> Result<Self, MarketError> {
        for (parameter, value) in [
            ("absolute_tolerance", absolute),
            ("relative_tolerance", relative),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(MarketError::InvalidSurfaceParameter {
                    parameter,
                    bits: value.to_bits(),
                });
            }
        }
        Ok(Self { absolute, relative })
    }

    #[must_use]
    pub const fn local_vol_vegakt_v1() -> Self {
        Self {
            absolute: 1.0e-14,
            relative: 1.0e-12,
        }
    }

    #[must_use]
    pub const fn absolute(self) -> f64 {
        self.absolute
    }

    #[must_use]
    pub const fn relative(self) -> f64 {
        self.relative
    }

    pub(crate) fn effective(self, left: f64, right: f64) -> f64 {
        self.absolute
            .max(self.relative * 1.0_f64.max(left.abs()).max(right.abs()))
    }

    pub(crate) fn permits_non_strict(self, left: f64, right: f64) -> bool {
        left - right <= self.effective(left, right)
    }

    pub(crate) fn permits_strict(self, left: f64, right: f64) -> bool {
        right - left > self.effective(left, right)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhiSpec {
    PowerLaw { eta: f64, gamma: f64 },
    HestonLike { lambda: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhiEvaluation {
    pub value: f64,
    pub theta_derivative: f64,
}

impl PhiSpec {
    fn validate_syntax(self) -> Result<(), MarketError> {
        match self {
            Self::PowerLaw { eta, gamma } => {
                require_positive_parameter("eta", eta)?;
                require_open_unit_parameter("gamma", gamma)?;
            }
            Self::HestonLike { lambda } => {
                require_positive_parameter("lambda", lambda)?;
            }
        }
        Ok(())
    }

    fn validate(self, rho: f64, tolerance: SurfaceValidationTolerance) -> Result<(), MarketError> {
        self.validate_syntax()?;
        match self {
            Self::PowerLaw { eta, gamma } => {
                if !tolerance.permits_non_strict(gamma, 0.5) {
                    return Err(MarketError::SsviAdmissibilityViolation {
                        condition: "power_law_gamma_at_most_one_half",
                        left_bits: gamma.to_bits(),
                        right_bits: 0.5_f64.to_bits(),
                    });
                }
                let wing_bound = eta * (1.0 + rho.abs());
                if !wing_bound.is_finite() || !tolerance.permits_non_strict(wing_bound, 2.0) {
                    return Err(MarketError::SsviAdmissibilityViolation {
                        condition: "power_law_eta_wing_bound",
                        left_bits: wing_bound.to_bits(),
                        right_bits: 2.0_f64.to_bits(),
                    });
                }
            }
            Self::HestonLike { lambda } => {
                let lower_bound = (1.0 + rho.abs()) / 4.0;
                if !tolerance.permits_non_strict(lower_bound, lambda) {
                    return Err(MarketError::SsviAdmissibilityViolation {
                        condition: "heston_like_lambda_lower_bound",
                        left_bits: lower_bound.to_bits(),
                        right_bits: lambda.to_bits(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn evaluate(self, theta: f64) -> Result<PhiEvaluation, MarketError> {
        self.validate_syntax()?;
        if !theta.is_finite() || theta <= 0.0 {
            return Err(MarketError::InvalidSurfaceParameter {
                parameter: "theta",
                bits: theta.to_bits(),
            });
        }
        let evaluation = match self {
            Self::PowerLaw { eta, gamma } => {
                let value = eta / (theta.powf(gamma) * (1.0 + theta).powf(1.0 - gamma));
                let theta_derivative = -value * (gamma / theta + (1.0 - gamma) / (1.0 + theta));
                PhiEvaluation {
                    value,
                    theta_derivative,
                }
            }
            Self::HestonLike { lambda } => {
                let z = lambda * theta;
                if z.abs() <= SMALL_HESTON_Z {
                    let value = 0.5
                        + z * (-1.0 / 6.0
                            + z * (1.0 / 24.0
                                + z * (-1.0 / 120.0
                                    + z * (1.0 / 720.0 + z * (-1.0 / 5040.0 + z / 40320.0)))));
                    let derivative_by_z = -1.0 / 6.0
                        + z * (1.0 / 12.0
                            + z * (-1.0 / 40.0
                                + z * (1.0 / 180.0 + z * (-1.0 / 1008.0 + z / 6720.0))));
                    PhiEvaluation {
                        value,
                        theta_derivative: lambda * derivative_by_z,
                    }
                } else {
                    let exp_minus_z = (-z).exp();
                    PhiEvaluation {
                        value: (z - 1.0 + exp_minus_z) / (z * z),
                        theta_derivative: lambda * (2.0 - z - (z + 2.0) * exp_minus_z)
                            / (z * z * z),
                    }
                }
            }
        };
        if !evaluation.value.is_finite()
            || evaluation.value <= 0.0
            || !evaluation.theta_derivative.is_finite()
        {
            return Err(MarketError::NonFiniteSurfaceValue {
                field: "phi",
                time_bits: 0,
                log_moneyness_bits: 0,
                value_bits: evaluation.value.to_bits(),
            });
        }
        Ok(evaluation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TotalVarianceDerivatives {
    pub total_variance: f64,
    pub log_moneyness_derivative: f64,
    pub log_moneyness_second_derivative: f64,
    pub time_derivative: f64,
    pub theta: f64,
    pub theta_derivative: f64,
    pub theta_region: ThetaRegion,
}

pub trait ImpliedVarianceSurface: Send + Sync {
    fn total_variance_derivatives(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandardSsvi {
    theta_curve: ThetaPchip,
    rho: f64,
    phi: PhiSpec,
    tolerance: SurfaceValidationTolerance,
}

impl StandardSsvi {
    pub fn new(
        theta_curve: ThetaPchip,
        rho: f64,
        phi: PhiSpec,
        tolerance: SurfaceValidationTolerance,
    ) -> Result<Self, MarketError> {
        if !rho.is_finite() || rho.abs() >= 1.0 {
            return Err(MarketError::InvalidSurfaceParameter {
                parameter: "rho",
                bits: rho.to_bits(),
            });
        }
        phi.validate(rho, tolerance)?;
        for theta in theta_curve.values() {
            let evaluated = phi.evaluate(*theta)?;
            validate_butterfly_bounds(*theta, rho, evaluated.value, tolerance)?;
            if evaluated.theta_derivative >= 0.0 {
                return Err(MarketError::SsviAdmissibilityViolation {
                    condition: "phi_strictly_decreasing",
                    left_bits: evaluated.theta_derivative.to_bits(),
                    right_bits: 0.0_f64.to_bits(),
                });
            }
            let skew_derivative = evaluated.value + theta * evaluated.theta_derivative;
            if skew_derivative < -tolerance.effective(skew_derivative, 0.0) {
                return Err(MarketError::SsviAdmissibilityViolation {
                    condition: "theta_phi_non_decreasing",
                    left_bits: skew_derivative.to_bits(),
                    right_bits: 0.0_f64.to_bits(),
                });
            }
        }
        Ok(Self {
            theta_curve,
            rho,
            phi,
            tolerance,
        })
    }

    #[must_use]
    pub fn theta_curve(&self) -> &ThetaPchip {
        &self.theta_curve
    }

    #[must_use]
    pub const fn rho(&self) -> f64 {
        self.rho
    }

    #[must_use]
    pub const fn phi_spec(&self) -> PhiSpec {
        self.phi
    }

    #[must_use]
    pub const fn tolerance(&self) -> SurfaceValidationTolerance {
        self.tolerance
    }
}

impl ImpliedVarianceSurface for StandardSsvi {
    fn total_variance_derivatives(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        if !time.is_finite() || time <= 0.0 {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "time",
                bits: time.to_bits(),
            });
        }
        if !log_moneyness.is_finite() {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "log_moneyness",
                bits: log_moneyness.to_bits(),
            });
        }

        let theta = self.theta_curve.evaluate(time)?;
        let phi = self.phi.evaluate(theta.theta)?;
        let y = phi.value * log_moneyness;
        let a = y + self.rho;
        let q = (a * a + 1.0 - self.rho * self.rho).sqrt();
        let shape = 1.0 + self.rho * y + q;
        let total_variance = theta.theta * shape / 2.0;
        let common = self.rho + a / q;
        let log_moneyness_derivative = theta.theta * phi.value * common / 2.0;
        let log_moneyness_second_derivative =
            theta.theta * phi.value * phi.value * (1.0 - self.rho * self.rho) / (2.0 * q * q * q);
        let theta_partial =
            shape / 2.0 + theta.theta * log_moneyness * phi.theta_derivative * common / 2.0;
        let time_derivative = theta_partial * theta.derivative;
        let result = TotalVarianceDerivatives {
            total_variance,
            log_moneyness_derivative,
            log_moneyness_second_derivative,
            time_derivative,
            theta: theta.theta,
            theta_derivative: theta.derivative,
            theta_region: theta.region,
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
}

fn require_positive_parameter(parameter: &'static str, value: f64) -> Result<(), MarketError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

fn require_open_unit_parameter(parameter: &'static str, value: f64) -> Result<(), MarketError> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(MarketError::InvalidSurfaceParameter {
            parameter,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

fn validate_butterfly_bounds(
    theta: f64,
    rho: f64,
    phi: f64,
    tolerance: SurfaceValidationTolerance,
) -> Result<(), MarketError> {
    let correlation_factor = 1.0 + rho.abs();
    let wing = theta * phi * correlation_factor;
    if !wing.is_finite() || !tolerance.permits_strict(wing, 4.0) {
        return Err(MarketError::SsviAdmissibilityViolation {
            condition: "strict_wing_slope",
            left_bits: wing.to_bits(),
            right_bits: 4.0_f64.to_bits(),
        });
    }
    let curvature = theta * phi * phi * correlation_factor;
    if !curvature.is_finite() || !tolerance.permits_non_strict(curvature, 4.0) {
        return Err(MarketError::SsviAdmissibilityViolation {
            condition: "curvature_bound",
            left_bits: curvature.to_bits(),
            right_bits: 4.0_f64.to_bits(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theta_curve() -> ThetaPchip {
        ThetaPchip::new(vec![0.5, 1.0, 2.0], vec![0.02, 0.04, 0.07], 0.02)
            .expect("valid theta curve")
    }

    #[test]
    fn heston_small_theta_branch_has_stable_limit() {
        let phi = PhiSpec::HestonLike { lambda: 2.0 }
            .evaluate(1.0e-12)
            .expect("small theta");
        assert!((phi.value - 0.5).abs() < 1.0e-12);
        assert!((phi.theta_derivative + 1.0 / 3.0).abs() < 1.0e-11);
    }

    #[test]
    fn invalid_family_parameters_are_rejected() {
        let tolerance = SurfaceValidationTolerance::local_vol_vegakt_v1();
        assert!(
            StandardSsvi::new(
                theta_curve(),
                -0.3,
                PhiSpec::PowerLaw {
                    eta: 0.5,
                    gamma: 0.6,
                },
                tolerance,
            )
            .is_err()
        );
        assert!(
            StandardSsvi::new(
                theta_curve(),
                -0.3,
                PhiSpec::HestonLike { lambda: 0.1 },
                tolerance,
            )
            .is_err()
        );
        assert!(
            StandardSsvi::new(
                theta_curve(),
                1.0,
                PhiSpec::HestonLike { lambda: 2.0 },
                tolerance,
            )
            .is_err()
        );
        assert!(
            PhiSpec::PowerLaw {
                eta: -0.5,
                gamma: 0.5,
            }
            .evaluate(0.04)
            .is_err()
        );
        assert!(PhiSpec::HestonLike { lambda: 0.0 }.evaluate(0.04).is_err());
    }

    #[test]
    fn analytic_derivatives_match_finite_differences() {
        let surface = StandardSsvi::new(
            theta_curve(),
            -0.3,
            PhiSpec::PowerLaw {
                eta: 0.5,
                gamma: 0.5,
            },
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("valid surface");
        let time = 1.5;
        let k = -0.1;
        let bump = 1.0e-5;
        let value = surface
            .total_variance_derivatives(time, k)
            .expect("surface value");
        let up_k = surface
            .total_variance_derivatives(time, k + bump)
            .expect("up k")
            .total_variance;
        let down_k = surface
            .total_variance_derivatives(time, k - bump)
            .expect("down k")
            .total_variance;
        let finite_k = (up_k - down_k) / (2.0 * bump);
        let finite_kk = (up_k - 2.0 * value.total_variance + down_k) / (bump * bump);
        let up_t = surface
            .total_variance_derivatives(time + bump, k)
            .expect("up time")
            .total_variance;
        let down_t = surface
            .total_variance_derivatives(time - bump, k)
            .expect("down time")
            .total_variance;
        let finite_t = (up_t - down_t) / (2.0 * bump);
        assert!((value.log_moneyness_derivative - finite_k).abs() < 1.0e-10);
        assert!((value.log_moneyness_second_derivative - finite_kk).abs() < 2.0e-6);
        assert!((value.time_derivative - finite_t).abs() < 1.0e-10);
    }

    #[test]
    fn invalid_queries_are_rejected() {
        let surface = StandardSsvi::new(
            theta_curve(),
            -0.3,
            PhiSpec::HestonLike { lambda: 2.0 },
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("valid surface");
        assert!(surface.total_variance_derivatives(0.0, 0.0).is_err());
        assert!(surface.total_variance_derivatives(1.0, f64::NAN).is_err());
    }
}
