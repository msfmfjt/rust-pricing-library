use pricing::mc::ExecutionPolicy;
use pricing::risk::{VegaKtGammaRecoveryPolicy, VegaKtJointGammaDirections};
use pricing::{JsonLimits, PricingPlan, parse_request_json};
use serde_json::Value;

fn context() -> (PricingPlan, Vec<f64>, VegaKtJointGammaDirections) {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/initial-smile-gamma-v0.1.json"
    ))
    .unwrap();
    let mut p = f["cases"][1]["request"].clone();
    p["engine"]["independent_sampling_units"] = 4096.into();
    let nodes: Vec<_> = p["model"]["local_variance_grid"]["log_forward_moneyness_nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let plan = PricingPlan::compile(
        &parse_request_json(p.to_string().as_bytes(), JsonLimits::default()).unwrap(),
        ExecutionPolicy::new(1, Some(256)).unwrap(),
    )
    .unwrap();
    let mut first = vec![-0.3; nodes.len() + 1];
    first[0] = 0.4;
    let mut second = vec![1.; nodes.len() + 1];
    second[0] = 0.;
    let sum = first.iter().zip(&second).map(|(a, b)| a + b).collect();
    let d = VegaKtJointGammaDirections {
        conservation_basis: vec![vec![1.; nodes.len()]],
        risk_directions: vec![first, second, sum],
    };
    (plan, nodes.iter().map(|x| 100. * x.exp()).collect(), d)
}
fn policy() -> VegaKtGammaRecoveryPolicy {
    VegaKtGammaRecoveryPolicy::new(vec![0.01, 0.003, 0.001, 0.0003, 0.0001], 4096, 0.5, 0.7, 1.)
        .unwrap()
}
#[test]
fn joint_covariance_propagates_dependent_direction_without_losing_cross_terms() {
    let (plan, strikes, directions) = context();
    let d = plan
        .evaluate_vega_kt_joint_gamma_audit(
            strikes,
            vec![80., 100., 120.],
            &policy(),
            directions,
            32768,
            1e-8,
        )
        .unwrap();
    let s = d.joint_validation.unwrap();
    assert!(s.passed);
    let cov = |i: usize, j: usize| s.joint_sample_covariance[i * 9 + j];
    for offset in [0, 3, 6] {
        assert!(
            (cov(offset + 2, offset + 2)
                - cov(offset, offset)
                - cov(offset + 1, offset + 1)
                - 2. * cov(offset, offset + 1))
            .abs()
                < 1e-9
        );
    }
    for (i, m) in s
        .native_directions
        .iter()
        .chain(&s.recovered_directions)
        .chain(&s.differences)
        .enumerate()
    {
        assert!((cov(i, i) - m.sample_variance().unwrap()).abs() < 1e-9);
    }
    assert!(cov(3, 4).abs() > 1.);
}
#[test]
fn failed_component_withholds_joint_statistics() {
    let (plan, strikes, directions) = context();
    let d = plan
        .evaluate_vega_kt_joint_gamma_audit(
            strikes,
            vec![80., 100., 120.],
            &policy(),
            directions,
            64,
            1e-8,
        )
        .unwrap();
    assert!(d.initial.estimate.is_none());
    assert!(d.joint_validation.is_none());
}
#[test]
fn nonfinite_or_wrong_shape_directions_are_rejected() {
    let (plan, strikes, mut directions) = context();
    directions.risk_directions[0][0] = f64::NAN;
    assert!(
        plan.evaluate_vega_kt_joint_gamma_audit(
            strikes.clone(),
            vec![100.],
            &policy(),
            directions.clone(),
            32768,
            1e-8
        )
        .is_err()
    );
    directions.risk_directions = vec![vec![1.]];
    assert!(
        plan.evaluate_vega_kt_joint_gamma_audit(
            strikes,
            vec![100.],
            &policy(),
            directions,
            32768,
            1e-8
        )
        .is_err()
    );
}
