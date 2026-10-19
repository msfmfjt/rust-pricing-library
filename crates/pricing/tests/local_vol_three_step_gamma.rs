use pricing::risk::VegaKtConvolutionPlan;
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/three-step-gamma-v0.1.json"
    ))
    .unwrap()
}
fn array(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
#[test]
fn actual_two_transition_source_matches_independent_operator_and_signed_kkt() {
    let f = fixture();
    let g = &f["request"]["model"]["local_variance_grid"];
    let nodes = array(&g["log_forward_moneyness_nodes"]);
    let times = array(&g["time_nodes"]);
    let v = array(&g["values"]);
    let m = nodes.len();
    let strikes = nodes.iter().map(|x| 100. * x.exp()).collect();
    let (op, a) = VegaKtConvolutionPlan::from_two_transition_variance_source(
        100.,
        v[0] * times[1],
        [times[2] - times[1], times[3] - times[2]],
        &nodes,
        [&v[m..2 * m], &v[2 * m..3 * m]],
        strikes,
        0.003,
    )
    .unwrap();
    assert!((a.source_probability - 1.).abs() < 1e-10);
    assert!((a.source_first_spot_moment - 100.).abs() < 1e-8);
    assert!(a.source_spot_tail_bound > 0. && a.source_spot_tail_bound < 1e-20);
    for (native, reference) in op
        .forward_matrix()
        .iter()
        .zip(f["forward_matrix"].as_array().unwrap())
    {
        for (a, b) in native.iter().zip(array(reference)) {
            assert!((a - b).abs() < 1e-10);
        }
    }
    let dirs = f["directions"]
        .as_array()
        .unwrap()
        .iter()
        .map(array)
        .collect();
    let op = op.conserve_local_vega_directions(dirs, false).unwrap();
    let recovered = op
        .recover(&array(&f["input_variance_sensitivities"]))
        .unwrap();
    assert!(recovered.iter().any(|q| *q < 0.));
    for (a, b) in recovered.iter().zip(array(&f["recovered_signed_masses"])) {
        assert!((a - b).abs() < 2e-8);
    }
}
#[test]
fn constant_second_transition_reduces_to_exact_gaussian_source() {
    let nodes: Vec<_> = (0..17).map(|i| -0.8 + 0.1 * i as f64).collect();
    let strikes: Vec<_> = nodes.iter().map(|x| 100. * x.exp()).collect();
    let third: Vec<_> = nodes
        .iter()
        .map(|x| 0.04 + 0.01 * x + 0.01 * x * x)
        .collect();
    let (actual, _) = VegaKtConvolutionPlan::from_two_transition_variance_source(
        100.,
        0.01,
        [0.25, 0.2],
        &nodes,
        [&[0.04; 17], &third],
        strikes.clone(),
        0.003,
    )
    .unwrap();
    let (gaussian, _) = VegaKtConvolutionPlan::from_gaussian_variance_source(
        100., 0.02, 0.2, &nodes, &third, strikes, 0.003,
    )
    .unwrap();
    for (a, b) in actual
        .forward_matrix()
        .iter()
        .flatten()
        .zip(gaussian.forward_matrix().iter().flatten())
    {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
fn invalid_models_and_unresolved_narrow_transition_fail_explicitly() {
    let nodes: Vec<_> = (0..17).map(|i| -0.8 + 0.1 * i as f64).collect();
    let build = |first, second: Vec<f64>| {
        VegaKtConvolutionPlan::from_two_transition_variance_source(
            100.,
            first,
            [0.25, 0.2],
            &nodes,
            [&second, &[0.04; 17]],
            vec![80., 100., 120.],
            0.003,
        )
    };
    assert!(build(0., vec![0.04; 17]).is_err());
    assert!(build(0.01, vec![f64::NAN; 17]).is_err());
    assert!(build(0.01, vec![0.04; 16]).is_err());
    assert!(build(1e4, vec![0.04; 17]).is_err());
    assert!(build(0.01, vec![1e-8; 17]).is_err());
}
