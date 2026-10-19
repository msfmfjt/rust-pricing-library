use pricing::mc::ExecutionPolicy;
use pricing::{JsonLimits, PricingPlan, parse_request_json};
use serde_json::Value;
fn model() -> Value {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/three-step-gamma-v0.1.json"
    ))
    .unwrap();
    let mut p = f["request"].clone();
    p["engine"]["independent_sampling_units"] = 2.into();
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
fn initial_three_scope_and_precision_budgets_reject_invalid_requests() {
    let p = model();
    let plan = compile(&p);
    for (bins, n, tol) in [
        (vec![], 512, 1e-8),
        (vec![100., 100.], 512, 1e-8),
        (vec![100.], 63, 1e-8),
        (vec![100.], 4097, 1e-8),
        (vec![100.], 512, f64::NAN),
    ] {
        assert!(
            plan.evaluate_vega_kt_three_step_initial_gamma_audit(bins, n, tol)
                .is_err()
        );
    }
    let mut q = p.clone();
    q["model"]["local_variance_grid"]["values"][0] = 0.041.into();
    assert!(
        compile(&q)
            .evaluate_vega_kt_three_step_initial_gamma_audit(vec![100.], 64, 1e-8)
            .is_err()
    );
    let mut q = p;
    q["engine"]["independent_sampling_units"] = 10000001.into();
    assert!(
        compile(&q)
            .evaluate_vega_kt_three_step_initial_gamma_audit(vec![100.], 64, 1e-8)
            .is_err()
    );
}
#[test]
fn collapsed_conditional_transition_is_not_renormalized_or_accepted_as_zero() {
    let mut p = model();
    p["model"]["local_variance_grid"]["floor"] = 1e-20.into();
    for i in 33..66 {
        p["model"]["local_variance_grid"]["values"][i] = 1e-16.into();
    }
    let error = compile(&p)
        .evaluate_vega_kt_three_step_initial_gamma_audit(vec![100.], 64, 1e-8)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("source probability or martingale")
    );
    let mut p = model();
    p["model"]["local_variance_grid"]["floor"] = 1e-40.into();
    for i in 0..33 {
        p["model"]["local_variance_grid"]["values"][i] = 1e-32.into();
    }
    let error = compile(&p)
        .evaluate_vega_kt_three_step_initial_gamma_audit(vec![100.], 64, 1e-8)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("below floating-point resolution")
    );
}
