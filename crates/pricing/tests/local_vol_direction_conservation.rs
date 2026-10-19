use pricing::risk::{VegaKtConvolutionPlan, VegaKtGammaRecoveryPolicy};
use serde_json::Value;

fn nodes() -> Vec<f64> {
    (0..17).map(|i| -0.6 + 1.2 * i as f64 / 16.).collect()
}
fn plan(lambda: f64) -> VegaKtConvolutionPlan {
    VegaKtConvolutionPlan::new(
        100.,
        0.02,
        0.01,
        0.05,
        &nodes(),
        nodes().iter().map(|x| 100. * x.exp()).collect(),
        lambda,
    )
    .unwrap()
}
fn directions() -> Vec<Vec<f64>> {
    vec![nodes(), nodes().iter().map(|x| x * x).collect()]
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[test]
fn simultaneous_inverse_matches_independent_primal_kkt_including_gain_and_trace() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/direction-convolution-gamma-v0.1.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let op = plan(case["relative_regularization"].as_f64().unwrap())
            .conserve_local_vega_directions(directions(), true)
            .unwrap();
        let expected: Vec<Vec<f64>> = serde_json::from_value(case["inverse"].clone()).unwrap();
        for (actual, expected) in op
            .recovery_matrix()
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
        {
            assert!((actual - expected).abs() < 2e-7 * expected.abs().max(1.));
        }
        assert!(
            (op.inverse_noise_gain_bound() - case["inverse_noise_gain_bound"].as_f64().unwrap())
                .abs()
                < 2e-6
        );
        assert!((op.resolution_trace() - case["resolution_trace"].as_f64().unwrap()).abs() < 2e-8);
        assert!(op.preserves_local_vega_total());
        assert_eq!(op.conservation_directions(), directions());
    }
}

#[test]
fn every_input_basis_signed_linearity_and_spatial_residual_are_preserved() {
    let op = plan(0.01)
        .conserve_local_vega_directions(directions(), true)
        .unwrap();
    for col in 0..17 {
        let mut input = vec![0.; 17];
        input[col] = 1.;
        let reconstructed = op.reconstruct(&op.recover(&input).unwrap()).unwrap();
        for d in directions().iter().chain([vec![1.; 17]].iter()) {
            assert!((dot(d, &reconstructed) - dot(d, &input)).abs() < 2e-10);
        }
    }
    let b: Vec<_> = (0..17).map(|i| (i as f64).sin()).collect();
    let q = op.recover(&b).unwrap();
    let negative = op
        .recover(&b.iter().map(|b| -2.5 * b).collect::<Vec<_>>())
        .unwrap();
    for (a, b) in q.iter().zip(negative) {
        assert!((2.5 * a + b).abs() < 1e-9);
    }
    assert!(q.iter().any(|&q| q < 0.));
    let reconstructed = op.reconstruct(&q).unwrap();
    assert!(
        reconstructed
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            > 0.1
    );
    assert_eq!(op.recover(&[0.; 17]).unwrap(), vec![0.; 17]);
}

#[test]
fn direction_scaling_sign_and_order_do_not_change_the_constrained_solution() {
    let a = plan(0.01)
        .conserve_local_vega_directions(directions(), true)
        .unwrap();
    let mut d = directions();
    d[0].iter_mut().for_each(|x| *x *= -1e20);
    d[1].iter_mut().for_each(|x| *x *= 1e-20);
    d.reverse();
    let b = plan(0.01).conserve_local_vega_directions(d, true).unwrap();
    for (a, b) in a
        .recovery_matrix()
        .iter()
        .flatten()
        .zip(b.recovery_matrix().iter().flatten())
    {
        assert!((a - b).abs() < 2e-10);
    }
}

#[test]
fn rank_shape_nonfinite_and_sequential_constraint_composition_fail_closed() {
    for d in [
        vec![],
        vec![vec![0.; 17]],
        vec![vec![f64::NAN; 17]],
        vec![vec![f64::INFINITY; 17]],
        vec![vec![1.; 16]],
        vec![nodes(), nodes()],
        vec![vec![1.; 17]],
        vec![nodes(); 9],
    ] {
        assert!(plan(0.01).conserve_local_vega_directions(d, true).is_err());
    }
    assert!(
        plan(0.01)
            .conserve_local_vega_total()
            .unwrap()
            .conserve_local_vega_directions(directions(), true)
            .is_err()
    );
    let p = plan(0.01)
        .conserve_local_vega_directions(directions(), false)
        .unwrap();
    assert!(!p.preserves_local_vega_total());
    assert!(p.clone().conserve_local_vega_total().is_err());
    assert!(
        p.conserve_local_vega_directions(directions(), false)
            .is_err()
    );
}

#[test]
fn policy_tensor_limits_and_shapes_are_checked_before_sampling() {
    let p = VegaKtGammaRecoveryPolicy::new(vec![0.01], 128, 0.03, 0.25, 2.).unwrap();
    assert!(p.local_vega_conservation_directions().is_empty());
    for d in [
        vec![],
        vec![vec![]],
        vec![vec![vec![0.; 17]]],
        vec![directions(), vec![nodes()]],
        vec![vec![vec![1.; 1]]],
        vec![vec![vec![f64::NAN; 17]]],
        vec![directions(); 65],
        vec![vec![vec![1.; 129]; 8]; 64],
    ] {
        assert!(p.clone().with_local_vega_direction_conservation(d).is_err());
    }
    let expected = vec![directions(), directions()];
    assert_eq!(
        p.with_local_vega_direction_conservation(expected.clone())
            .unwrap()
            .local_vega_conservation_directions(),
        expected
    );
}
