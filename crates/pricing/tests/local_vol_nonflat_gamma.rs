use pricing::risk::VegaKtConvolutionPlan;
use serde_json::Value;

fn nodes() -> Vec<f64> {
    (0..17).map(|i| -0.6 + 1.2 * i as f64 / 16.).collect()
}
fn strikes() -> Vec<f64> {
    nodes().iter().map(|x| 100. * x.exp()).collect()
}

#[test]
fn variance_operator_flat_limit_matches_analytic_volatility_operator() {
    let flat =
        VegaKtConvolutionPlan::new(100., 0.02, 0.01, 0.05, &nodes(), strikes(), 0.003).unwrap();
    let (variance, accuracy) = VegaKtConvolutionPlan::from_gaussian_variance_source(
        100.,
        0.02,
        0.25,
        &nodes(),
        &[0.04; 17],
        strikes(),
        0.003,
    )
    .unwrap();
    for (a, b) in flat
        .forward_matrix()
        .iter()
        .flatten()
        .zip(variance.forward_matrix().iter().flatten())
    {
        assert!((a / 0.4 - b).abs() < 1e-10);
    }
    assert!(accuracy.refinement_maximum_difference < accuracy.entry_tolerance);
    assert!(accuracy.omitted_tail_entry_bound < accuracy.entry_tolerance);
    let b: Vec<_> = (0..17).map(|i| (i as f64).sin()).collect();
    let q = flat.recover(&b).unwrap();
    let other = variance
        .recover(&b.iter().map(|x| x / 0.4).collect::<Vec<_>>())
        .unwrap();
    for (a, b) in q.iter().zip(other) {
        assert!((a - b).abs() < 1e-8);
    }
}

#[test]
fn nonflat_operator_and_signed_recovery_match_independent_quadrature_and_kkt() {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/nonflat-variance-gamma-v0.1.json"
    ))
    .unwrap();
    let array = |key: &str| {
        f[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let (op, _) = VegaKtConvolutionPlan::from_gaussian_variance_source(
        100.,
        0.02,
        0.25,
        &array("source_log_nodes"),
        &array("local_variances"),
        array("gamma_strikes"),
        0.003,
    )
    .unwrap();
    for (a, b) in op.forward_matrix().iter().flatten().zip(
        f["forward_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap()),
    ) {
        assert!((a - b.as_f64().unwrap()).abs() < 1e-10);
    }
    let dirs = f["directions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let op = op.conserve_local_vega_directions(dirs, false).unwrap();
    let q = op.recover(&array("variance_sensitivities")).unwrap();
    for (a, b) in q.iter().zip(array("recovered_masses")) {
        assert!((a - b).abs() < 2e-8);
    }
    assert!(q[10] < 0.);
}

#[test]
fn invalid_variance_models_and_unresolved_quadrature_are_rejected() {
    let build = |v0, dt, v: Vec<f64>, lambda| {
        VegaKtConvolutionPlan::from_gaussian_variance_source(
            100.,
            v0,
            dt,
            &nodes(),
            &v,
            strikes(),
            lambda,
        )
    };
    assert!(build(0., 0.25, vec![0.04; 17], 0.01).is_err());
    assert!(build(0.02, 0., vec![0.04; 17], 0.01).is_err());
    assert!(build(0.02, 0.25, vec![0.; 17], 0.01).is_err());
    assert!(build(0.02, 0.25, vec![f64::NAN; 17], 0.01).is_err());
    assert!(build(0.02, 0.25, vec![0.04; 16], 0.01).is_err());
    assert!(build(0.02, 0.25, vec![0.04; 17], 0.).is_err());
    // Near-deterministic transition creates a peak the fixed rule cannot resolve.
    assert!(build(0.02, 0.25, vec![1e-12; 17], 0.01).is_err());
    // Twelve source SDs no longer certify a sufficiently small omitted tail.
    assert!(build(100., 0.25, vec![0.04; 17], 0.01).is_err());
}
