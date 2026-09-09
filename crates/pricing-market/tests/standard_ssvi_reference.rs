use pricing_market::{
    ImpliedVarianceSurface, PhiSpec, StandardSsvi, SurfaceValidationTolerance, ThetaPchip,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/reference-cases-v0.1.json");

fn decimal(object: &Value, field: &str) -> f64 {
    object[field]
        .as_str()
        .unwrap_or_else(|| panic!("{field} must be a decimal string"))
        .parse()
        .unwrap_or_else(|error| panic!("invalid decimal string for {field}: {error}"))
}

fn assert_close(field: &str, actual: f64, expected: f64) {
    let tolerance = 5.0e-14_f64.max(5.0e-13 * expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{field}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn standard_ssvi_matches_v1_reference_fixture() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("valid reference fixture");
    let cases = fixture["standard_ssvi"]
        .as_array()
        .expect("standard_ssvi cases");

    for case in cases {
        let id = case["id"].as_str().expect("case id");
        let inputs = &case["inputs"];
        let expected = &case["expected"];
        let theta = decimal(inputs, "theta");
        let theta_t = decimal(inputs, "theta_t");
        let rho = decimal(inputs, "rho");
        let log_moneyness = decimal(inputs, "k");
        let phi = match id {
            "power_regular" => PhiSpec::PowerLaw {
                eta: decimal(inputs, "eta"),
                gamma: decimal(inputs, "gamma"),
            },
            "heston_like_regular" | "heston_like_small_theta" => PhiSpec::HestonLike {
                lambda: decimal(inputs, "lambda"),
            },
            _ => panic!("unknown Standard SSVI fixture: {id}"),
        };

        let phi_value = phi.evaluate(theta).expect("valid phi evaluation");
        assert_close("phi", phi_value.value, decimal(expected, "phi"));
        assert_close(
            "phi_theta",
            phi_value.theta_derivative,
            decimal(expected, "phi_theta"),
        );

        let theta_curve = ThetaPchip::new(vec![1.0, 2.0], vec![theta, theta + theta_t], theta_t)
            .expect("valid linear theta curve");
        let surface = StandardSsvi::new(
            theta_curve,
            rho,
            phi,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .expect("valid Standard SSVI surface");
        let result = surface
            .total_variance_derivatives(1.0, log_moneyness)
            .expect("valid surface evaluation");

        assert_close("w", result.total_variance, decimal(expected, "w"));
        assert_close(
            "w_k",
            result.log_moneyness_derivative,
            decimal(expected, "w_k"),
        );
        assert_close(
            "w_kk",
            result.log_moneyness_second_derivative,
            decimal(expected, "w_kk"),
        );
        assert_close("w_t", result.time_derivative, decimal(expected, "w_t"));
    }
}
