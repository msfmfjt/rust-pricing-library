use pricing_market::{
    EssviSlice, EssviSurface, ImpliedVarianceSurface, SurfaceValidationTolerance,
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
    let tolerance = 1.0e-13_f64.max(5.0e-13 * expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{field}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn essvi_matches_v1_interpolation_fixture() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("valid reference fixture");
    let case = &fixture["essvi_interpolation"];
    let inputs = &case["inputs"];
    let expected = &case["expected"];
    let first = EssviSlice::new(
        decimal(inputs, "t0"),
        decimal(inputs, "theta0"),
        decimal(inputs, "psi0"),
        decimal(inputs, "rho_psi0"),
    )
    .expect("valid first slice");
    let second = EssviSlice::new(
        decimal(inputs, "t1"),
        decimal(inputs, "theta1"),
        decimal(inputs, "psi1"),
        decimal(inputs, "rho_psi1"),
    )
    .expect("valid second slice");
    let surface = EssviSurface::new(
        vec![first, second],
        0.02,
        SurfaceValidationTolerance::local_vol_vegakt_v1(),
    )
    .expect("valid eSSVI surface");
    let time = decimal(inputs, "t");
    let parameters = surface.parameters(time).expect("interpolated parameters");
    let rho = parameters.rho_psi / parameters.psi;
    let rho_derivative = (parameters.rho_psi_derivative * parameters.psi
        - parameters.rho_psi * parameters.psi_derivative)
        / (parameters.psi * parameters.psi);
    assert_close("theta", parameters.theta, decimal(expected, "theta"));
    assert_close("psi", parameters.psi, decimal(expected, "psi"));
    assert_close("rho_psi", parameters.rho_psi, decimal(expected, "rho_psi"));
    assert_close("rho", rho, decimal(expected, "rho"));
    assert_close("rho_t", rho_derivative, decimal(expected, "rho_t"));

    let result = surface
        .total_variance_derivatives(time, decimal(inputs, "k"))
        .expect("valid total variance");
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
