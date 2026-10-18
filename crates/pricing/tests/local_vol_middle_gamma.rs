use pricing::mc::ExecutionPolicy;
use pricing::risk::VegaKtGammaRecoveryPolicy;
use pricing::{JsonLimits, PricingPlan, parse_request_json};
use serde_json::Value;
fn request() -> Value {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/three-step-gamma-v0.1.json"
    ))
    .unwrap();
    let mut p = f["request"].clone();
    p["engine"]["independent_sampling_units"] = 4096.into();
    p
}
fn compile(p: &Value) -> PricingPlan {
    PricingPlan::compile(
        &parse_request_json(p.to_string().as_bytes(), JsonLimits::default()).unwrap(),
        ExecutionPolicy::new(1, Some(256)).unwrap(),
    )
    .unwrap()
}
#[test]
fn middle_scope_and_work_budgets_fail_before_recovery() {
    let mut p = request();
    let plan = compile(&p);
    for (bins, n, tol) in [
        (vec![], 512, 1e-8),
        (vec![100., 100.], 512, 1e-8),
        (vec![100.], 63, 1e-8),
        (vec![100.], 4097, 1e-8),
        (vec![100.], 512, f64::NAN),
    ] {
        assert!(
            plan.evaluate_vega_kt_middle_smile_gamma_audit(bins, n, tol)
                .is_err()
        );
    }
    p["model"]["local_variance_grid"]["values"][0] = 0.041.into();
    assert!(
        compile(&p)
            .evaluate_vega_kt_middle_smile_gamma_audit(vec![100.], 64, 1e-8)
            .is_err()
    );
}
#[test]
fn terminal_dependent_readouts_keep_signed_and_zero_directions() {
    let p = request();
    let plan = compile(&p);
    let nodes: Vec<_> = p["model"]["local_variance_grid"]["log_forward_moneyness_nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let strikes: Vec<_> = nodes.iter().map(|x| 100. * x.exp()).collect();
    let m = nodes.len();
    let policy = VegaKtGammaRecoveryPolicy::new(
        vec![0.01, 0.003, 0.001, 0.0003, 0.0001],
        4096,
        0.5,
        0.7,
        1.,
    )
    .unwrap();
    let basis = vec![vec![1.; m]];
    let d = plan
        .evaluate_vega_kt_three_step_terminal_readout_audit(
            strikes.clone(),
            &policy,
            basis.clone(),
            vec![vec![1.; m], vec![-2.; m], vec![0.; m]],
        )
        .unwrap();
    let s = d.validation.unwrap();
    let risks = s.recovered_direction_risks;
    assert!((risks[1].mean() + 2. * risks[0].mean()).abs() < 1e-10);
    assert!(
        (risks[1].sample_variance().unwrap() - 4. * risks[0].sample_variance().unwrap()).abs()
            < 1e-8
    );
    assert_eq!(risks[2].mean(), 0.);
    assert_eq!(risks[2].sample_variance().unwrap(), 0.);
    let mut unconserved = vec![0.; m];
    unconserved[m / 2] = 1.;
    assert!(
        plan.evaluate_vega_kt_three_step_terminal_readout_audit(
            strikes,
            &policy,
            basis,
            vec![unconserved]
        )
        .is_err()
    );
}
