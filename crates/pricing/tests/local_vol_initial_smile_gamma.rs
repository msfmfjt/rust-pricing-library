use pricing::mc::ExecutionPolicy;
use pricing::{JsonLimits, PricingPlan, parse_request_json};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/initial-smile-gamma-v0.1.json"
    ))
    .unwrap()
}
fn compile(p: &Value) -> PricingPlan {
    PricingPlan::compile(
        &parse_request_json(p.to_string().as_bytes(), JsonLimits::default()).unwrap(),
        ExecutionPolicy::new(1, Some(256)).unwrap(),
    )
    .unwrap()
}
fn array(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
#[test]
fn nonflat_initial_gamma_matches_independent_slopes_and_integration_by_parts() {
    let f = fixture();
    for c in f["cases"].as_array().unwrap() {
        let plan = compile(&c["request"]);
        let d = plan
            .evaluate_vega_kt_initial_smile_gamma_audit(
                array(&f["gamma_bin_boundaries"]),
                32768,
                1e-8,
            )
            .unwrap();
        assert!(d.recovery_passed, "{}", c["product"]);
        assert!(d.validation_passed);
        assert_eq!(
            d.fine.interpolation_nodes,
            2 * d.coarse.interpolation_nodes - 1
        );
        assert!(d.estimate.is_some());
        assert!(
            (d.fine.recovered_variance_sensitivity
                - c["reference"]["recovered_variance_sensitivity"]
                    .as_f64()
                    .unwrap())
            .abs()
                < d.functional_tolerance
        );
        for (a, b) in d
            .fine
            .gamma_bin_masses
            .iter()
            .zip(array(&c["reference"]["gamma_bin_masses"]))
        {
            assert!((a - b).abs() < d.mass_tolerance);
        }
        assert!(d.variance_tail_bound < 1e-20);
    }
}
#[test]
fn coarse_reconstruction_is_retained_as_failure_not_refitted_or_zeroed() {
    let f = fixture();
    let plan = compile(&f["cases"][1]["request"]);
    let d = plan
        .evaluate_vega_kt_initial_smile_gamma_audit(array(&f["gamma_bin_boundaries"]), 64, 1e-8)
        .unwrap();
    assert!(!d.recovery_passed);
    assert!(d.estimate.is_none());
    assert!(d.variance_refinement_difference > d.functional_tolerance);
    assert_eq!(d.coarse.subdivisions, 64);
    assert_eq!(d.fine.subdivisions, 128);
    assert!(d.fine.recovered_variance_sensitivity > 1.);
}
#[test]
fn input_scope_and_invalid_precision_budgets_are_rejected() {
    let f = fixture();
    let p = &f["cases"][0]["request"];
    let plan = compile(p);
    for (bounds, n, tol) in [
        (vec![], 32768, 1e-8),
        (vec![100., 100.], 32768, 1e-8),
        (vec![100.], 63, 1e-8),
        (vec![100.], 32769, 1e-8),
        (vec![100.], 32768, f64::NAN),
        (vec![100.], 32768, 0.),
    ] {
        assert!(
            plan.evaluate_vega_kt_initial_smile_gamma_audit(bounds, n, tol)
                .is_err()
        );
    }
    let mut nonflat_first = p.clone();
    nonflat_first["model"]["local_variance_grid"]["values"][0] = serde_json::json!(0.041);
    assert!(
        compile(&nonflat_first)
            .evaluate_vega_kt_initial_smile_gamma_audit(vec![100.], 64, 1e-8)
            .is_err()
    );
}
