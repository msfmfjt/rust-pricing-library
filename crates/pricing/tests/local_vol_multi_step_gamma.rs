use pricing::risk::VegaKtConvolutionPlan;
use serde_json::Value;
fn array(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
#[test]
fn four_and_six_transition_operators_match_independent_quadrature() {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/multi-step-gamma-v0.1.json"
    ))
    .unwrap();
    for case in f["models"].as_array().unwrap() {
        let g = &case["request"]["model"]["local_variance_grid"];
        let nodes = array(&g["log_forward_moneyness_nodes"]);
        let times = array(&g["time_nodes"]);
        let rows: Vec<Vec<_>> = array(&g["values"])
            .chunks_exact(nodes.len())
            .take(times.len() - 1)
            .map(|r| r.to_vec())
            .collect();
        let (op, a) = VegaKtConvolutionPlan::from_multi_transition_variance_source(
            100.,
            &times,
            &nodes,
            &rows,
            nodes.iter().map(|x| 100. * x.exp()).collect(),
            0.003,
        )
        .unwrap();
        assert_eq!(a.steps.len(), times.len() - 2);
        for stage in a.steps {
            assert!((stage.probability - 1.).abs() < 1e-10);
            assert!((stage.first_spot_moment - 100.).abs() < 1e-8);
        }
        for (native, reference) in op
            .forward_matrix()
            .iter()
            .zip(case["forward_matrix"].as_array().unwrap())
        {
            for (a, b) in native.iter().zip(array(reference)) {
                assert!((a - b).abs() < 1e-10);
            }
        }
        let directions: Vec<Vec<f64>> = case["directions"]
            .as_array()
            .unwrap()
            .iter()
            .map(array)
            .collect();
        let constrained = op
            .conserve_local_vega_directions(directions.clone(), false)
            .unwrap();
        let input: Vec<_> = (0..nodes.len())
            .map(|i| if i % 2 == 0 { 1. } else { -1. })
            .collect();
        let q = constrained.recover(&input).unwrap();
        assert!(q.iter().any(|v| *v < 0.));
        let back = constrained.reconstruct(&q).unwrap();
        for d in directions {
            assert!(
                d.iter()
                    .zip(&input)
                    .zip(&back)
                    .map(|((d, x), y)| d * (x - y))
                    .sum::<f64>()
                    .abs()
                    < 1e-7
            );
        }
    }
}
#[test]
fn constant_earlier_variance_reduces_to_the_exact_gaussian_marginal() {
    let nodes: Vec<_> = (0..17).map(|i| -0.8 + i as f64 * 0.1).collect();
    let strikes: Vec<_> = nodes.iter().map(|x| 100. * x.exp()).collect();
    for count in [2, 4, 6, 16] {
        let times: Vec<_> = (0..=count).map(|i| i as f64 * 0.2).collect();
        let rows = vec![vec![0.04; 17]; count];
        let (op, _) = VegaKtConvolutionPlan::from_multi_transition_variance_source(
            100.,
            &times,
            &nodes,
            &rows,
            strikes.clone(),
            0.003,
        )
        .unwrap();
        let dt = times[count] - times[count - 1];
        let expected = VegaKtConvolutionPlan::new(
            100.,
            0.04 * times[count - 1],
            0.04 * dt,
            dt / 2.,
            &nodes,
            strikes.clone(),
            0.003,
        )
        .unwrap();
        for (a, b) in op
            .forward_matrix()
            .iter()
            .flatten()
            .zip(expected.forward_matrix().iter().flatten())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
#[test]
fn invalid_or_unresolved_transitions_are_rejected_without_renormalization() {
    let nodes: Vec<_> = (0..17).map(|i| -0.8 + i as f64 * 0.1).collect();
    let times = vec![0., 0.2, 0.4, 0.6, 0.8];
    let rows = vec![vec![0.04; 17]; 4];
    let build = |t: &[f64], r: &[Vec<f64>]| {
        VegaKtConvolutionPlan::from_multi_transition_variance_source(
            100.,
            t,
            &nodes,
            r,
            vec![80., 100., 120.],
            0.003,
        )
    };
    assert!(build(&[0., 0.2], &rows[..1]).is_err());
    assert!(
        build(
            &(0..=17).map(|i| i as f64 * 0.2).collect::<Vec<_>>(),
            &vec![vec![0.04; 17]; 17]
        )
        .is_err()
    );
    for (row, value) in [(0, 1e-32), (1, 1e-16), (2, f64::NAN)] {
        let mut r = rows.clone();
        r[row].fill(value);
        assert!(build(&times, &r).is_err());
    }
}
