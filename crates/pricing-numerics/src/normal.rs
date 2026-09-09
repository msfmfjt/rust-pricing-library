use std::f64::consts::{PI, SQRT_2};

#[must_use]
pub fn standard_normal_pdf(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * PI).sqrt()
}

// Hart-style rational approximation in the central region and an asymptotic
// continued fraction in the tails. The operation order is fixed for replay.
#[must_use]
pub fn standard_normal_cdf(value: f64) -> f64 {
    let magnitude = value.abs();
    let tail = if magnitude > 37.0 {
        0.0
    } else if magnitude < 7.071_067_811_865_475 {
        let numerator = horner(
            magnitude,
            &[
                0.035_262_496_599_891_1,
                0.700_383_064_443_688,
                6.373_962_203_531_65,
                33.912_866_078_383,
                112.079_291_497_871,
                221.213_596_169_931,
                220.206_867_912_376,
            ],
        );
        let denominator = horner(
            magnitude,
            &[
                0.088_388_347_648_318_4,
                1.755_667_163_182_64,
                16.064_177_579_207,
                86.780_732_202_946_1,
                296.564_248_779_674,
                637.333_633_378_831,
                793.826_512_519_948,
                440.413_735_824_752,
            ],
        );
        (-0.5 * magnitude * magnitude).exp() * numerator / denominator
    } else {
        let continued_fraction = magnitude
            + 1.0 / (magnitude + 2.0 / (magnitude + 3.0 / (magnitude + 4.0 / (magnitude + 0.65))));
        (-0.5 * magnitude * magnitude).exp() / (continued_fraction * SQRT_2 * PI.sqrt())
    };
    if value > 0.0 { 1.0 - tail } else { tail }
}

fn horner(value: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .copied()
        .reduce(|accumulator, coefficient| accumulator * value + coefficient)
        .expect("CDF coefficient tables are non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_matches_reference_values_and_symmetry() {
        for (value, expected) in [
            (-8.0, 6.220_960_574_271_74e-16),
            (-3.0, 0.001_349_898_031_630_095_7),
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_542_9),
            (5.0, 0.999_999_713_348_428_1),
        ] {
            assert!((standard_normal_cdf(value) - expected).abs() < 2.0e-15);
            assert!(
                (standard_normal_cdf(value) + standard_normal_cdf(-value) - 1.0).abs() < 2.0e-15
            );
        }
    }

    #[test]
    fn pdf_is_symmetric_and_has_known_atm_value() {
        let at_the_money = 1.0 / (2.0 * PI).sqrt();
        assert_eq!(standard_normal_pdf(0.0), at_the_money);
        assert_eq!(standard_normal_pdf(-2.5), standard_normal_pdf(2.5));
    }

    #[test]
    fn infinities_have_distribution_limits() {
        assert_eq!(standard_normal_cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(standard_normal_cdf(f64::INFINITY), 1.0);
        assert_eq!(standard_normal_pdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(standard_normal_pdf(f64::INFINITY), 0.0);
    }
}
