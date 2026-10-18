use pricing::risk::{
    integrate_piecewise_linear_local_gamma_transition, local_gamma_transition_cells,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/gamma-numerics-v0.1.json");

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn close(label: &str, actual: f64, expected: f64, relative: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= relative * expected.abs(),
        "{label}: actual={actual:.17e}, expected={expected:.17e}, relative_error={:.4e}",
        (actual - expected).abs() / expected.abs()
    );
}

#[test]
fn gamma_transition_cells_match_independent_high_precision_references() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture");
    let tolerance = fixture["relative_tolerance"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["id"].as_str().unwrap();
        let cells = local_gamma_transition_cells(
            case["spot"].as_f64().unwrap(),
            case["variance"].as_f64().unwrap(),
            &numbers(&case["strikes"]),
        )
        .expect("cells");
        let expected = &case["expected"];
        for (i, cell) in cells.iter().enumerate() {
            close(
                &format!("{label} probability {i}"),
                cell.probability(),
                expected["probabilities"][i].as_f64().unwrap(),
                tolerance,
            );
            close(
                &format!("{label} first moment {i}"),
                cell.first_moment(),
                expected["first_moments"][i].as_f64().unwrap(),
                tolerance,
            );
        }
    }
}

#[test]
fn gamma_transition_operator_matches_independent_high_precision_references() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture");
    for case in fixture["cases"].as_array().unwrap() {
        let value = integrate_piecewise_linear_local_gamma_transition(
            case["spot"].as_f64().unwrap(),
            case["variance"].as_f64().unwrap(),
            &numbers(&case["strikes"]),
            &numbers(&case["gamma"]),
        )
        .expect("transition");
        close(
            case["id"].as_str().unwrap(),
            value,
            case["expected"]["gamma_integral"].as_f64().unwrap(),
            fixture["relative_tolerance"].as_f64().unwrap(),
        );
    }
}

#[test]
fn gamma_transition_rejects_nonfinite_moments() {
    assert!(local_gamma_transition_cells(f64::MAX / 2., 1., &[90., 110.]).is_err());
}

#[test]
fn gamma_transition_preserves_signed_linearity_and_positive_kernel() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture");
    for case in fixture["cases"].as_array().unwrap() {
        let spot = case["spot"].as_f64().unwrap();
        let variance = case["variance"].as_f64().unwrap();
        let strikes = numbers(&case["strikes"]);
        let gamma = numbers(&case["gamma"]);
        let negative = gamma.iter().map(|x| -2.0 * x).collect::<Vec<_>>();
        let value =
            integrate_piecewise_linear_local_gamma_transition(spot, variance, &strikes, &gamma)
                .unwrap();
        assert!(value >= 0.0);
        close(
            "signed Gamma",
            integrate_piecewise_linear_local_gamma_transition(spot, variance, &strikes, &negative)
                .unwrap(),
            -2.0 * value,
            1e-12,
        );
        let constant = vec![1.; strikes.len()];
        close(
            "constant Gamma",
            integrate_piecewise_linear_local_gamma_transition(spot, variance, &strikes, &constant)
                .unwrap(),
            variance.exp(),
            1e-12,
        );
    }
}

#[test]
fn gamma_transition_is_invariant_under_monetary_unit_rescaling() {
    for strikes in [[90., 110.], [640., 790.]] {
        let base =
            integrate_piecewise_linear_local_gamma_transition(100., 0.04, &strikes, &[0., 1.])
                .unwrap();
        for scale in [1e-100, 1e-20, 1., 1e20, 1e100] {
            let scaled = strikes.map(|k| k * scale);
            close(
                "currency units",
                integrate_piecewise_linear_local_gamma_transition(
                    100. * scale,
                    0.04,
                    &scaled,
                    &[0., 1.],
                )
                .unwrap(),
                base,
                1e-11,
            );
        }
    }
}
