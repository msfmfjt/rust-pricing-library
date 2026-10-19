use pricing::risk::{lognormal_spot_squared_hat_masses, refine_vega_kt_log_nodes};
use serde_json::Value;

#[test]
fn integrated_hats_match_independent_high_precision_moments_including_narrow_tails() {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/density-mass-v0.1.json"
    ))
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let nodes: Vec<_> = c["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let m = lognormal_spot_squared_hat_masses(100., c["variance"].as_f64().unwrap(), &nodes)
            .unwrap();
        for (&actual, expected) in m.iter().zip(c["expected"].as_array().unwrap()) {
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= 5e-11 * expected.abs(),
                "{}: {actual:.17e} vs {expected:.17e}",
                c["id"]
            );
        }
        let total = 10000. * c["variance"].as_f64().unwrap().exp();
        assert!((m.iter().sum::<f64>() - total).abs() < 2e-13 * total);
    }
}

#[test]
fn refinement_preserves_nodes_limits_spacing_and_rejects_unbounded_work() {
    let source = [-0.8, -0.1, 0., 0.17, 0.8];
    let out = refine_vega_kt_log_nodes(&source, 0.014).unwrap();
    assert!(source.iter().all(|x| out.contains(x)));
    assert!(
        out.windows(2)
            .all(|p| p[1] > p[0] && p[1] - p[0] <= 0.014 * (1. + 1e-13))
    );
    for h in [0., -0.1, f64::NAN, f64::INFINITY, 1e-30] {
        assert!(refine_vega_kt_log_nodes(&source, h).is_err());
    }
    assert!(refine_vega_kt_log_nodes(&[0., 0.], 0.1).is_err());
    assert!(refine_vega_kt_log_nodes(&[0.], 0.1).is_err());
}

#[test]
fn invalid_or_unrepresentable_density_inputs_fail_instead_of_dividing_by_zero() {
    for (s, v) in [
        (0., 0.04),
        (-1., 0.04),
        (100., 0.),
        (100., -0.1),
        (100., f64::NAN),
        (f64::MAX, 0.04),
    ] {
        assert!(lognormal_spot_squared_hat_masses(s, v, &[-0.1, 0., 0.1]).is_err());
    }
    assert!(lognormal_spot_squared_hat_masses(100., 0.04, &[-0.1, -0.1, 0.1]).is_err());
}
