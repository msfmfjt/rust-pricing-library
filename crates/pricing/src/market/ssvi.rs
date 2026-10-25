use crate::market::{MarketError, ThetaPchip, ThetaRegion};
use pricing_numerics::{standard_normal_cdf, standard_normal_pdf};

const SMALL_HESTON_Z: f64 = 0.5;

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
                    let (value, derivative_by_z) = heston_phi_series(z);
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

fn heston_phi_series(z: f64) -> (f64, f64) {
    // Degree 14 of sum_n (-z)^n/(n+2)!, with its analytic derivative.
    // For 0 <= z <= 0.5, alternating-series truncation is below 8.6e-20
    // in value and 2.6e-18 in derivative, before floating-point rounding.
    let mut value = 1.0;
    let mut derivative = 0.0;
    for divisor in (3..=16).rev().map(f64::from) {
        derivative = -(value + z * derivative) / divisor;
        value = 1.0 - z * value / divisor;
    }
    (0.5 * value, 0.5 * derivative)
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForwardCallEvaluation {
    pub forward: f64,
    pub strike: f64,
    pub undiscounted_price: f64,
    pub density_factor: f64,
    pub call_density: f64,
    pub d1: f64,
    pub d2: f64,
    pub variance: TotalVarianceDerivatives,
}

pub trait ImpliedVarianceSurface: Send + Sync {
    fn total_variance_derivatives(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError>;

    fn forward_call_evaluation(
        &self,
        time: f64,
        log_moneyness: f64,
        forward: f64,
    ) -> Result<ForwardCallEvaluation, MarketError> {
        if !forward.is_finite() || forward <= 0.0 {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "forward",
                bits: forward.to_bits(),
            });
        }
        let variance = self.total_variance_derivatives(time, log_moneyness)?;
        evaluate_forward_call(time, log_moneyness, forward, variance)
    }
}

pub fn durrleman_density_factor(
    time: f64,
    log_moneyness: f64,
    variance: TotalVarianceDerivatives,
) -> Result<f64, MarketError> {
    let total_variance = variance.total_variance;
    if !total_variance.is_finite() || total_variance <= 0.0 {
        return Err(MarketError::NonPositiveSurfaceValue {
            field: "total_variance",
            time_bits: time.to_bits(),
            log_moneyness_bits: log_moneyness.to_bits(),
            value_bits: total_variance.to_bits(),
        });
    }
    for (field, value) in [
        (
            "log_moneyness_derivative",
            variance.log_moneyness_derivative,
        ),
        (
            "log_moneyness_second_derivative",
            variance.log_moneyness_second_derivative,
        ),
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
    let u = 1.0 - log_moneyness * variance.log_moneyness_derivative / (2.0 * total_variance);
    let density_factor = u * u
        - (variance.log_moneyness_derivative * variance.log_moneyness_derivative / 4.0)
            * (1.0 / total_variance + 1.0 / 4.0)
        + variance.log_moneyness_second_derivative / 2.0;
    if !density_factor.is_finite() || density_factor <= 0.0 {
        return Err(MarketError::NonPositiveSurfaceValue {
            field: "density_factor",
            time_bits: time.to_bits(),
            log_moneyness_bits: log_moneyness.to_bits(),
            value_bits: density_factor.to_bits(),
        });
    }
    Ok(density_factor)
}

fn evaluate_forward_call(
    time: f64,
    log_moneyness: f64,
    forward: f64,
    variance: TotalVarianceDerivatives,
) -> Result<ForwardCallEvaluation, MarketError> {
    let strike = forward * log_moneyness.exp();
    if !strike.is_finite() || strike <= 0.0 {
        return Err(MarketError::NonPositiveSurfaceValue {
            field: "strike",
            time_bits: time.to_bits(),
            log_moneyness_bits: log_moneyness.to_bits(),
            value_bits: strike.to_bits(),
        });
    }

    let total_variance = variance.total_variance;
    let root_variance = total_variance.sqrt();
    let d2 = -log_moneyness / root_variance - root_variance / 2.0;
    let d1 = d2 + root_variance;
    let undiscounted_price = forward * standard_normal_cdf(d1) - strike * standard_normal_cdf(d2);
    let density_factor = durrleman_density_factor(time, log_moneyness, variance)?;
    let call_density = standard_normal_pdf(d2) / (strike * root_variance) * density_factor;
    let result = ForwardCallEvaluation {
        forward,
        strike,
        undiscounted_price,
        density_factor,
        call_density,
        d1,
        d2,
        variance,
    };
    for (field, value) in [
        ("undiscounted_call_price", result.undiscounted_price),
        ("call_density", result.call_density),
        ("d1", result.d1),
        ("d2", result.d2),
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
    if result.undiscounted_price < 0.0 || result.call_density < 0.0 {
        let (field, value) = if result.undiscounted_price < 0.0 {
            ("undiscounted_call_price", result.undiscounted_price)
        } else {
            ("call_density", result.call_density)
        };
        return Err(MarketError::NonPositiveSurfaceValue {
            field,
            time_bits: time.to_bits(),
            log_moneyness_bits: log_moneyness.to_bits(),
            value_bits: value.to_bits(),
        });
    }
    Ok(result)
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

    struct FixedVarianceSurface(TotalVarianceDerivatives);

    impl ImpliedVarianceSurface for FixedVarianceSurface {
        fn total_variance_derivatives(
            &self,
            _time: f64,
            _log_moneyness: f64,
        ) -> Result<TotalVarianceDerivatives, MarketError> {
            Ok(self.0)
        }
    }

    fn assert_forward_reference(field: &str, actual: f64, expected: f64) {
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5.0e-13,
            "{field}: actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.3e}"
        );
    }

    #[test]
    fn forward_call_matches_independent_lognormal_references() {
        // 100-digit Decimal Gaussian integrals; see the optional generator.
        for (w, slope, curvature, k, forward, expected) in [
            (
                0.25,
                0.125,
                0.0625,
                0.5,
                128.0,
                (
                    211.0363226496164,
                    6.712361380821717,
                    0.7802734375,
                    0.001350632231678362,
                    -0.75,
                    -1.25,
                ),
            ),
            (
                0.25,
                0.125,
                0.0625,
                -0.5,
                128.0,
                (
                    77.63592444321708,
                    54.43532853332232,
                    1.2802734375,
                    0.009931955037303309,
                    1.25,
                    0.75,
                ),
            ),
            (
                1.0,
                -0.25,
                0.5,
                0.25,
                1.0,
                (
                    1.2840254166877414,
                    0.3077110451143759,
                    1.2939453125,
                    0.30346390631436787,
                    0.25,
                    -0.75,
                ),
            ),
            (
                4.0,
                0.5,
                -0.25,
                1.0,
                0.125,
                (
                    0.33978522855738064,
                    0.06373270750683377,
                    0.72265625,
                    0.13772920675556027,
                    0.5,
                    -1.5,
                ),
            ),
            (
                0.0625,
                -0.0625,
                0.125,
                -0.25,
                4.0,
                (
                    3.1152031322856195,
                    0.9579589122925151,
                    0.812255859375,
                    0.2837417107283103,
                    1.125,
                    0.875,
                ),
            ),
            (
                0.5,
                0.0,
                0.5,
                2.0,
                16.0,
                (
                    118.2248975828904,
                    0.020161870919953355,
                    1.25,
                    3.775811593011915e-05,
                    -2.4748737341529163,
                    -3.181980515339464,
                ),
            ),
            (
                0.25,
                0.0,
                0.0,
                0.0,
                100.0,
                (
                    100.0,
                    19.741265136584744,
                    1.0,
                    0.007733362336056984,
                    0.25,
                    -0.25,
                ),
            ),
            (
                1.0,
                0.0,
                0.0,
                -2.0,
                1.0,
                (
                    0.1353352832366127,
                    0.8674964229435774,
                    1.0,
                    0.9570127801738912,
                    2.5,
                    1.5,
                ),
            ),
        ] {
            let variance = density_variance(w, slope, curvature);
            let call = FixedVarianceSurface(variance)
                .forward_call_evaluation(1.25, k, forward)
                .expect("regular forward call");
            let (strike, price, g, density, d1, d2) = expected;
            assert_eq!(call.forward, forward);
            assert_eq!(call.variance, variance);
            assert_eq!(call.density_factor, g);
            for (field, actual, reference) in [
                ("strike", call.strike, strike),
                ("undiscounted_price", call.undiscounted_price, price),
                ("call_density", call.call_density, density),
                ("d1", call.d1, d1),
                ("d2", call.d2, d2),
            ] {
                assert_forward_reference(field, actual, reference);
            }
        }
    }

    #[test]
    fn forward_call_is_homogeneous_in_forward() {
        let surface = FixedVarianceSurface(density_variance(0.25, 0.125, 0.0625));
        for k in [-0.5, 0.0, 0.5] {
            let base = surface.forward_call_evaluation(1.25, k, 1.0).unwrap();
            for scale in [0.125, 0.5, 2.0, 128.0] {
                let scaled = surface.forward_call_evaluation(1.25, k, scale).unwrap();
                assert_eq!(scaled.forward, scale);
                assert_eq!(scaled.strike, scale * base.strike);
                assert_eq!(scaled.undiscounted_price, scale * base.undiscounted_price);
                assert_eq!(scaled.call_density, base.call_density / scale);
                assert_eq!((scaled.d1, scaled.d2), (base.d1, base.d2));
                assert_eq!(scaled.density_factor, base.density_factor);
                assert_eq!(scaled.variance, base.variance);
            }
        }
    }

    #[test]
    fn forward_call_rejects_invalid_forward_before_surface_queries() {
        let surface = flat_phi_surface(
            PhiSpec::HestonLike { lambda: 2.0 },
            0.0,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .unwrap();
        for forward in [
            0.0_f64,
            -0.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            assert_eq!(
                surface.forward_call_evaluation(f64::NAN, f64::NAN, forward),
                Err(MarketError::InvalidSurfaceQuery {
                    coordinate: "forward",
                    bits: forward.to_bits(),
                })
            );
        }
    }

    #[test]
    fn forward_call_preserves_queries_and_surface_errors() {
        struct RejectingSurface;
        impl ImpliedVarianceSurface for RejectingSurface {
            fn total_variance_derivatives(
                &self,
                time: f64,
                log_moneyness: f64,
            ) -> Result<TotalVarianceDerivatives, MarketError> {
                assert_eq!(time.to_bits(), 1.25_f64.to_bits());
                assert_eq!(log_moneyness.to_bits(), (-0.375_f64).to_bits());
                Err(MarketError::InvalidSurfaceQuery {
                    coordinate: "custom_surface",
                    bits: 0x7ff8_0000_0000_0042,
                })
            }
        }
        assert_eq!(
            RejectingSurface.forward_call_evaluation(1.25, -0.375, 128.0),
            Err(MarketError::InvalidSurfaceQuery {
                coordinate: "custom_surface",
                bits: 0x7ff8_0000_0000_0042,
            })
        );
    }

    #[test]
    fn forward_call_rejects_unrepresentable_strikes() {
        let time = 1.25_f64;
        for (k, forward, strike) in [
            (f64::MAX, 1.0, f64::INFINITY),
            (1.0, f64::MAX, f64::INFINITY),
            (-f64::MAX, 1.0, 0.0),
            (-1.0, f64::from_bits(1), 0.0),
        ] {
            // Strike validation precedes validation of the supplied variance.
            let surface = FixedVarianceSurface(density_variance(-1.0, 0.0, 0.0));
            assert_eq!(
                surface.forward_call_evaluation(time, k, forward),
                Err(MarketError::NonPositiveSurfaceValue {
                    field: "strike",
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: strike.to_bits(),
                })
            );
        }
    }

    #[test]
    fn forward_call_preserves_variance_and_density_errors() {
        let time = 1.25_f64;
        let k = -0.375_f64;
        for w in [0.0_f64, -0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(
                FixedVarianceSurface(density_variance(w, 0.0, 0.0))
                    .forward_call_evaluation(time, k, 1.0),
                Err(MarketError::NonPositiveSurfaceValue {
                    field: "total_variance",
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: w.to_bits(),
                })
            );
        }
        for (slope, curvature, field) in [
            (f64::INFINITY, 0.0, "log_moneyness_derivative"),
            (0.0, f64::INFINITY, "log_moneyness_second_derivative"),
        ] {
            assert_eq!(
                FixedVarianceSurface(density_variance(0.25, slope, curvature))
                    .forward_call_evaluation(time, k, 1.0),
                Err(MarketError::NonFiniteSurfaceValue {
                    field,
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: f64::INFINITY.to_bits(),
                })
            );
        }
        for (curvature, g) in [(-2.0, 0.0_f64), (-3.0, -0.5)] {
            assert_eq!(
                FixedVarianceSurface(density_variance(0.25, 0.0, curvature))
                    .forward_call_evaluation(time, k, 1.0),
                Err(MarketError::NonPositiveSurfaceValue {
                    field: "density_factor",
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: g.to_bits(),
                })
            );
        }
    }

    #[test]
    fn forward_call_accepts_zero_price_and_zero_density() {
        let surface = FixedVarianceSurface(density_variance(1.0, 0.0, 0.0));
        for (k, expected_price, d1, d2) in [(40.0, 0.0, -39.5, -40.5), (-40.0, 1.0, 40.5, 39.5)] {
            let call = surface.forward_call_evaluation(1.25, k, 1.0).unwrap();
            assert_eq!(call.undiscounted_price, expected_price);
            assert_eq!(call.call_density, 0.0);
            assert_eq!(call.density_factor, 1.0);
            assert_eq!((call.d1, call.d2), (d1, d2));
            assert!(call.strike.is_finite() && call.strike > 0.0);
        }
    }

    #[test]
    fn forward_call_rejects_nonfinite_density_from_finite_inputs() {
        let time = 1.25_f64;
        let k = -0.0_f64;
        // The positive strike times sqrt(w) underflows to zero.
        assert_eq!(
            FixedVarianceSurface(density_variance(0.25, 0.0, 0.0)).forward_call_evaluation(
                time,
                k,
                f64::from_bits(1)
            ),
            Err(MarketError::NonFiniteSurfaceValue {
                field: "call_density",
                time_bits: time.to_bits(),
                log_moneyness_bits: k.to_bits(),
                value_bits: f64::INFINITY.to_bits(),
            })
        );
    }

    #[test]
    fn forward_call_rejects_negative_roundoff_with_price_context() {
        let time = 1.25_f64;
        let k = 1.12e-16_f64;
        // At this tiny variance d1 and d2 round to the same float, while K > F.
        // The existing failure contract rejects a negative computed price.
        let error = FixedVarianceSurface(density_variance(1.0e-32, 0.0, 0.0))
            .forward_call_evaluation(time, k, 1.0)
            .expect_err("negative roundoff must not be returned as a valid price");
        match error {
            MarketError::NonPositiveSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            } => {
                assert_eq!(field, "undiscounted_call_price");
                assert_eq!(time_bits, time.to_bits());
                assert_eq!(log_moneyness_bits, k.to_bits());
                let value = f64::from_bits(value_bits);
                assert!(value.is_finite() && value < 0.0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn density_variance(w: f64, slope: f64, curvature: f64) -> TotalVarianceDerivatives {
        TotalVarianceDerivatives {
            total_variance: w,
            log_moneyness_derivative: slope,
            log_moneyness_second_derivative: curvature,
            time_derivative: 0.25,
            theta: 0.25,
            theta_derivative: 0.125,
            theta_region: ThetaRegion::Interpolated,
        }
    }

    #[test]
    fn density_factor_matches_exact_quadratic_variance_references() {
        // Local quadratic variance jets; exact rational evaluation of the
        // Durrleman expression gives dyadic results without rounding ambiguity.
        for (w, slope, curvature, k, expected) in [
            (0.25, 0.125, 0.0625, 0.5, 799.0 / 1024.0),
            (0.25, 0.125, 0.0625, -0.5, 1311.0 / 1024.0),
            (1.0, -0.25, 0.5, 0.25, 1325.0 / 1024.0),
            (4.0, 0.5, -0.25, 1.0, 185.0 / 256.0),
            (0.0625, -0.0625, 0.125, -0.25, 3327.0 / 4096.0),
            (0.5, 0.0, 0.5, 2.0, 1.25),
        ] {
            assert_eq!(
                durrleman_density_factor(1.25, k, density_variance(w, slope, curvature)),
                Ok(expected),
                "w={w}, w_k={slope}, w_kk={curvature}, k={k}"
            );
        }
        for w in [0.0625, 0.25, 1.0, 4.0] {
            for k in [-2.0, 0.0, 2.0] {
                assert_eq!(
                    durrleman_density_factor(1.25, k, density_variance(w, 0.0, 0.0)),
                    Ok(1.0),
                    "flat total variance has unit density factor"
                );
            }
        }
    }

    #[test]
    fn density_factor_rejects_invalid_variance_with_query_context() {
        let time = 1.25_f64;
        let k = -0.375_f64;
        for w in [
            0.0_f64,
            -0.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            assert_eq!(
                durrleman_density_factor(time, k, density_variance(w, 0.125, 0.0625)),
                Err(MarketError::NonPositiveSurfaceValue {
                    field: "total_variance",
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: w.to_bits(),
                })
            );
        }
    }

    #[test]
    fn density_factor_checks_both_derivatives_before_arithmetic() {
        let time = 1.25_f64;
        let k = -0.375_f64;
        for bad in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            for (slope, curvature, field) in [
                (bad, 0.0625, "log_moneyness_derivative"),
                (0.125, bad, "log_moneyness_second_derivative"),
                (bad, bad, "log_moneyness_derivative"),
            ] {
                assert_eq!(
                    durrleman_density_factor(time, k, density_variance(0.25, slope, curvature)),
                    Err(MarketError::NonFiniteSurfaceValue {
                        field,
                        time_bits: time.to_bits(),
                        log_moneyness_bits: k.to_bits(),
                        value_bits: bad.to_bits(),
                    })
                );
            }
        }
    }

    #[test]
    fn density_factor_distinguishes_zero_and_positive_boundary() {
        let time = 1.25_f64;
        let k = -0.375_f64;
        for (curvature, expected) in [((-2.0_f64).next_down(), -f64::EPSILON), (-2.0, 0.0)] {
            assert_eq!(
                durrleman_density_factor(time, k, density_variance(0.25, 0.0, curvature)),
                Err(MarketError::NonPositiveSurfaceValue {
                    field: "density_factor",
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: expected.to_bits(),
                })
            );
        }
        assert_eq!(
            durrleman_density_factor(time, k, density_variance(0.25, 0.0, (-2.0_f64).next_up())),
            Ok(f64::EPSILON / 2.0)
        );
    }

    #[test]
    fn density_factor_rejects_nonfinite_computed_results() {
        let time = 1.25_f64;
        for (k, slope, expected) in [
            (f64::MAX, 1.0, f64::INFINITY),
            (0.0, f64::MAX, f64::NEG_INFINITY),
        ] {
            assert_eq!(
                durrleman_density_factor(time, k, density_variance(0.25, slope, 0.0)),
                Err(MarketError::NonPositiveSurfaceValue {
                    field: "density_factor",
                    time_bits: time.to_bits(),
                    log_moneyness_bits: k.to_bits(),
                    value_bits: expected.to_bits(),
                })
            );
        }
        let error = durrleman_density_factor(time, 1.0, density_variance(0.25, f64::MAX, 0.0))
            .expect_err("overflowed positive and negative terms must not return NaN success");
        match error {
            MarketError::NonPositiveSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            } => {
                assert_eq!(field, "density_factor");
                assert_eq!(time_bits, time.to_bits());
                assert_eq!(log_moneyness_bits, 1.0_f64.to_bits());
                assert!(f64::from_bits(value_bits).is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn assert_phi_reference(actual: f64, expected: f64) {
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            actual.is_finite() && relative_error <= 5.0e-13,
            "actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.3e}"
        );
    }

    fn flat_phi_surface(
        phi: PhiSpec,
        rho: f64,
        tolerance: SurfaceValidationTolerance,
    ) -> Result<StandardSsvi, MarketError> {
        StandardSsvi::new(
            ThetaPchip::new(vec![1.0, 2.0], vec![1.0, 1.0], 0.0).expect("flat theta"),
            rho,
            phi,
            tolerance,
        )
    }

    #[test]
    fn power_law_phi_matches_independent_logarithmic_references() {
        // 100-digit references use log/exp and the independently simplified
        // derivative -eta*(gamma+theta)/(theta^(gamma+1)*(1+theta)^(2-gamma)).
        let cases: [(f64, f64, f64, f64, f64); 8] = [
            (0.75, 0.25, 0.04, 1.6284381233316532, -11.352092686687005),
            (0.375, 0.6, 0.125, 1.2457309396155172, -6.4224350664622225),
            (1.25, 0.125, 2.0, 0.4383289606286017, -0.15524150688929644),
            (0.5, 0.5, 0.25, 0.8944271909999159, -2.1466252583997982),
            (0.5, 0.5, 4.0, 0.11180339887498948, -0.025155764746872634),
            (
                0.625,
                0.875,
                16.0,
                0.03876760013710127,
                -0.0024051590158587644,
            ),
            (1.5, 0.375, 1e-08, 1499.999990625, -56250000585.937485),
            (
                0.125,
                0.25,
                32.0,
                0.003817130989241902,
                -0.00011657431288167741,
            ),
        ];
        for (eta, gamma, theta, value, derivative) in cases {
            let evaluated = PhiSpec::PowerLaw { eta, gamma }
                .evaluate(theta)
                .expect("standalone phi");
            assert_phi_reference(evaluated.value, value);
            assert_phi_reference(evaluated.theta_derivative, derivative);
        }
    }

    #[test]
    fn heston_phi_matches_independent_integral_references() {
        // Integrate (1-u)*exp(-z*u) and -u*(1-u)*exp(-z*u) on [0,1]
        // at 100 digits; includes neighbors of both old and new cutoffs.
        let cases: [(f64, f64, f64); 14] = [
            (1e-08, 0.49999999833333336, -0.16666666583333334),
            (
                9.999999999999999e-05,
                0.49998333374999165,
                -0.16665833358332777,
            ),
            (0.0001, 0.49998333374999165, -0.16665833358332777),
            (
                0.00010000000000000002,
                0.49998333374999165,
                -0.16665833358332777,
            ),
            (0.0002, 0.4999666683332667, -0.16665000099995556),
            (0.001, 0.4998333749916681, -0.16658335832777876),
            (0.01, 0.49833749168053576, -0.16583582778768355),
            (0.1, 0.4837418035959573, -0.15857787551510363),
            (
                0.49999999999999994,
                0.4261226388505337,
                -0.13061319425266849,
            ),
            (0.5, 0.4261226388505337, -0.13061319425266849),
            (0.5000000000000001, 0.4261226388505337, -0.13061319425266846),
            (0.75, 0.39531831598402617, -0.11616715860809587),
            (2.0, 0.28383382080915315, -0.06766764161830635),
            (16.0, 0.058593750439590524, -0.003417969244539342),
        ];
        for lambda in [0.25, 2.0, 8.0] {
            for (z, value, derivative_by_z) in cases {
                let evaluated = PhiSpec::HestonLike { lambda }
                    .evaluate(z / lambda)
                    .expect("Heston-like reference node");
                assert_phi_reference(evaluated.theta_derivative, lambda * derivative_by_z);
                assert_phi_reference(evaluated.value, value);
            }
        }
    }

    #[test]
    fn phi_family_syntax_rejects_all_invalid_parameters() {
        for bad in [
            0.0_f64,
            -0.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            for (phi, parameter) in [
                (
                    PhiSpec::PowerLaw {
                        eta: bad,
                        gamma: 0.5,
                    },
                    "eta",
                ),
                (
                    PhiSpec::PowerLaw {
                        eta: 0.5,
                        gamma: bad,
                    },
                    "gamma",
                ),
                (PhiSpec::HestonLike { lambda: bad }, "lambda"),
            ] {
                assert_eq!(
                    phi.evaluate(0.25),
                    Err(MarketError::InvalidSurfaceParameter {
                        parameter,
                        bits: bad.to_bits(),
                    })
                );
            }
        }
        for gamma in [1.0_f64, 1.0_f64.next_up(), 2.0] {
            assert_eq!(
                PhiSpec::PowerLaw { eta: 0.5, gamma }.evaluate(0.25),
                Err(MarketError::InvalidSurfaceParameter {
                    parameter: "gamma",
                    bits: gamma.to_bits()
                })
            );
        }
        for gamma in [f64::from_bits(1), 1.0_f64.next_down()] {
            let evaluated = PhiSpec::PowerLaw { eta: 0.5, gamma }
                .evaluate(0.25)
                .expect("open gamma boundaries");
            assert!(evaluated.value > 0.0 && evaluated.theta_derivative < 0.0);
        }
    }

    #[test]
    fn phi_query_domain_returns_theta_error_bits() {
        for phi in [
            PhiSpec::PowerLaw {
                eta: 0.5,
                gamma: 0.25,
            },
            PhiSpec::HestonLike { lambda: 2.0 },
        ] {
            for theta in [
                0.0_f64,
                -0.0,
                -f64::from_bits(1),
                -1.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::from_bits(0x7ff8_0000_0000_0042),
            ] {
                assert_eq!(
                    phi.evaluate(theta),
                    Err(MarketError::InvalidSurfaceParameter {
                        parameter: "theta",
                        bits: theta.to_bits(),
                    })
                );
            }
        }
    }

    #[test]
    fn phi_syntax_validation_precedes_query_validation() {
        for (phi, parameter) in [
            (
                PhiSpec::PowerLaw {
                    eta: -1.0,
                    gamma: -1.0,
                },
                "eta",
            ),
            (
                PhiSpec::PowerLaw {
                    eta: 0.5,
                    gamma: -1.0,
                },
                "gamma",
            ),
            (PhiSpec::HestonLike { lambda: -1.0 }, "lambda"),
        ] {
            assert_eq!(
                phi.evaluate(f64::NAN),
                Err(MarketError::InvalidSurfaceParameter {
                    parameter,
                    bits: (-1.0_f64).to_bits(),
                })
            );
        }
    }

    #[test]
    fn standalone_phi_does_not_apply_surface_admissibility() {
        for phi in [
            PhiSpec::PowerLaw {
                eta: 0.5,
                gamma: 0.75,
            },
            PhiSpec::PowerLaw {
                eta: 3.0,
                gamma: 0.25,
            },
            PhiSpec::HestonLike { lambda: 0.125 },
        ] {
            let evaluated = phi.evaluate(0.25).expect("valid family syntax");
            assert!(evaluated.value > 0.0 && evaluated.theta_derivative < 0.0);
            assert!(matches!(
                flat_phi_surface(
                    phi,
                    0.5,
                    SurfaceValidationTolerance::new(0.0, 0.0).expect("zero tolerance")
                ),
                Err(MarketError::SsviAdmissibilityViolation { .. })
            ));
        }
    }

    #[test]
    fn power_law_admissibility_is_symmetric_and_inclusive() {
        let exact = SurfaceValidationTolerance::new(0.0, 0.0).expect("zero tolerance");
        for rho in [-0.5, 0.5] {
            let eta = 4.0_f64 / 3.0;
            flat_phi_surface(PhiSpec::PowerLaw { eta, gamma: 0.5 }, rho, exact)
                .expect("inclusive bounds");
            let gamma = 0.5_f64.next_up();
            assert_eq!(
                flat_phi_surface(PhiSpec::PowerLaw { eta: 0.5, gamma }, rho, exact),
                Err(MarketError::SsviAdmissibilityViolation {
                    condition: "power_law_gamma_at_most_one_half",
                    left_bits: gamma.to_bits(),
                    right_bits: 0.5_f64.to_bits(),
                })
            );
            // The first successor still rounds eta*1.5 to exactly 2.0.
            flat_phi_surface(
                PhiSpec::PowerLaw {
                    eta: eta.next_up(),
                    gamma: 0.5,
                },
                rho,
                exact,
            )
            .expect("rounded wing is still exactly on the boundary");
            for eta in [eta.next_up().next_up(), f64::MAX] {
                assert_eq!(
                    flat_phi_surface(PhiSpec::PowerLaw { eta, gamma: 0.5 }, rho, exact),
                    Err(MarketError::SsviAdmissibilityViolation {
                        condition: "power_law_eta_wing_bound",
                        left_bits: (eta * 1.5).to_bits(),
                        right_bits: 2.0_f64.to_bits(),
                    })
                );
            }
            let tolerant = SurfaceValidationTolerance::new(0.125, 0.0).expect("absolute tolerance");
            flat_phi_surface(
                PhiSpec::PowerLaw {
                    eta: 0.5,
                    gamma: 0.625,
                },
                rho,
                tolerant,
            )
            .expect("inclusive gamma tolerance");
            assert!(matches!(
                flat_phi_surface(
                    PhiSpec::PowerLaw {
                        eta: 0.5,
                        gamma: 0.625_f64.next_up()
                    },
                    rho,
                    tolerant
                ),
                Err(MarketError::SsviAdmissibilityViolation {
                    condition: "power_law_gamma_at_most_one_half",
                    ..
                })
            ));
            flat_phi_surface(
                PhiSpec::PowerLaw {
                    eta: 17.0 / 12.0,
                    gamma: 0.5,
                },
                rho,
                tolerant,
            )
            .expect("inclusive wing tolerance");
            assert!(matches!(
                flat_phi_surface(
                    PhiSpec::PowerLaw {
                        eta: (17.0_f64 / 12.0).next_up(),
                        gamma: 0.5
                    },
                    rho,
                    tolerant
                ),
                Err(MarketError::SsviAdmissibilityViolation {
                    condition: "power_law_eta_wing_bound",
                    ..
                })
            ));
        }
    }

    #[test]
    fn heston_admissibility_lower_bound_is_symmetric_and_inclusive() {
        for rho in [-0.5, 0.5] {
            for (absolute, minimum) in [(0.0, 0.375_f64), (0.125, 0.25)] {
                let tolerance = SurfaceValidationTolerance::new(absolute, 0.0).expect("tolerance");
                for lambda in [minimum, minimum.next_up()] {
                    flat_phi_surface(PhiSpec::HestonLike { lambda }, rho, tolerance)
                        .expect("inclusive lambda boundary");
                }
                let lambda = minimum.next_down();
                assert_eq!(
                    flat_phi_surface(PhiSpec::HestonLike { lambda }, rho, tolerance),
                    Err(MarketError::SsviAdmissibilityViolation {
                        condition: "heston_like_lambda_lower_bound",
                        left_bits: 0.375_f64.to_bits(),
                        right_bits: lambda.to_bits(),
                    })
                );
            }
        }
    }

    #[test]
    fn phi_extreme_arithmetic_returns_typed_failures() {
        for (eta, theta, expected_value) in [
            (f64::MAX, f64::MIN_POSITIVE, f64::INFINITY),
            (f64::from_bits(1), f64::MAX, 0.0),
        ] {
            assert_eq!(
                PhiSpec::PowerLaw { eta, gamma: 0.5 }.evaluate(theta),
                Err(MarketError::NonFiniteSurfaceValue {
                    field: "phi",
                    time_bits: 0,
                    log_moneyness_bits: 0,
                    value_bits: expected_value.to_bits(),
                })
            );
        }
        let error = PhiSpec::PowerLaw {
            eta: 1.0,
            gamma: 0.5,
        }
        .evaluate(f64::MIN_POSITIVE)
        .expect_err("finite phi but infinite derivative");
        match error {
            MarketError::NonFiniteSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            } => {
                assert_eq!((field, time_bits, log_moneyness_bits), ("phi", 0, 0));
                assert!(f64::from_bits(value_bits).is_finite());
                assert!(f64::from_bits(value_bits) > 0.0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        let error = PhiSpec::HestonLike { lambda: f64::MAX }
            .evaluate(f64::MAX)
            .expect_err("unrepresentable scaled theta");
        match error {
            MarketError::NonFiniteSurfaceValue {
                field,
                time_bits,
                log_moneyness_bits,
                value_bits,
            } => {
                assert_eq!((field, time_bits, log_moneyness_bits), ("phi", 0, 0));
                assert!(f64::from_bits(value_bits).is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn heston_phi_handles_underflowed_scaled_theta() {
        let lambda = f64::MIN_POSITIVE;
        let evaluated = PhiSpec::HestonLike { lambda }
            .evaluate(f64::MIN_POSITIVE)
            .expect("underflowed z retains the analytic limit");
        assert_eq!(evaluated.value, 0.5);
        assert!((evaluated.theta_derivative + lambda / 6.0).abs() <= f64::from_bits(1));
        assert!(evaluated.theta_derivative < 0.0);
        let evaluated = PhiSpec::HestonLike { lambda: 2.0 }
            .evaluate(f64::from_bits(1))
            .expect("smallest positive theta");
        assert_eq!(evaluated.value, 0.5);
        assert_phi_reference(evaluated.theta_derivative, -1.0 / 3.0);
    }

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
        assert!(surface.forward_call_evaluation(1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn non_positive_density_factor_is_rejected() {
        struct InvalidDensitySurface;

        impl ImpliedVarianceSurface for InvalidDensitySurface {
            fn total_variance_derivatives(
                &self,
                _time: f64,
                _log_moneyness: f64,
            ) -> Result<TotalVarianceDerivatives, MarketError> {
                Ok(TotalVarianceDerivatives {
                    total_variance: 0.04,
                    log_moneyness_derivative: 0.0,
                    log_moneyness_second_derivative: -3.0,
                    time_derivative: 0.04,
                    theta: 0.04,
                    theta_derivative: 0.04,
                    theta_region: ThetaRegion::Interpolated,
                })
            }
        }

        let error = InvalidDensitySurface
            .forward_call_evaluation(1.0, 0.0, 100.0)
            .expect_err("negative density factor must fail");
        assert!(matches!(
            error,
            MarketError::NonPositiveSurfaceValue {
                field: "density_factor",
                ..
            }
        ));
    }
}
