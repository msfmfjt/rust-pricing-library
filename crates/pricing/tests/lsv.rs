use pricing::lsv::BergomiLsvPricingPlan;
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::Bergomi1Factor;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn payload(qmc: bool, bump: f64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":[0.0,0.3,0.7,1.0],"log_forward_moneyness_nodes":[-0.8,-0.2,0.12,0.45,0.9],"shape":[4,5],
        "values":(0..20).map(|i|0.04+0.002*(i%5) as f64+bump*(0.7*i as f64).cos()).collect::<Vec<_>>(),"floor":1e-8,"cap":4.0}});
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":1.5}}]);
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,"scramble_count":4,"master_scramble_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    v
}
fn request(qmc: bool, bump: f64) -> PricingRequest {
    parse_request_json(
        &serde_json::to_vec(&payload(qmc, bump)).unwrap(),
        JsonLimits::DEFAULT,
    )
    .unwrap()
}
fn plan(qmc: bool, bump: f64, workers: u32) -> BergomiLsvPricingPlan {
    BergomiLsvPricingPlan::compile(
        &request(qmc, bump),
        Bergomi1Factor::new(2.0, 0.6, -0.5).unwrap(),
        LsvParticleConfig::new(256, 429, 0.35, 5.0, true).unwrap(),
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}

#[test]
fn facade_reuses_dividend_grid_and_replays_across_worker_counts() {
    for qmc in [false, true] {
        let one = plan(qmc, 0.0, 1);
        let four = plan(qmc, 0.0, 4);
        assert!(one.calibration().surface().times().contains(&0.25));
        let a = one.evaluate_local_variance_risk().unwrap();
        let b = four.evaluate_local_variance_risk().unwrap();
        assert_eq!(a.price.value.to_bits(), b.price.value.to_bits());
        assert_eq!(
            a.price.standard_error.to_bits(),
            b.price.standard_error.to_bits()
        );
        assert_eq!(a.node_adjoints, b.node_adjoints);
        assert_eq!(a.standard_errors, b.standard_errors);
        assert_eq!(
            a.price.value.to_bits(),
            one.evaluate().unwrap().value.to_bits()
        );
        assert_eq!(a.time_nodes.as_ref(), &[0.0, 0.3, 0.7, 1.0]);
        assert_eq!(a.node_adjoints.len(), 20);
        assert_eq!(a.standard_errors.is_some(), qmc);
        assert_eq!(one.plan_fingerprint(), plan(qmc, 0.0, 1).plan_fingerprint());
        assert_ne!(one.plan_fingerprint(), four.plan_fingerprint());
        if qmc {
            assert_eq!(a.price.independent_sampling_units, 4);
            assert_eq!(a.price.evaluated_paths, 1024);
        }
        let expected = (plan(qmc, 1e-7, 1).evaluate().unwrap().value
            - plan(qmc, -1e-7, 1).evaluate().unwrap().value)
            / 2e-7;
        let actual = a
            .node_adjoints
            .iter()
            .enumerate()
            .map(|(i, a)| a * (0.7 * i as f64).cos())
            .sum::<f64>();
        assert!(
            (actual - expected).abs() < 2e-5 * (1.0 + expected.abs()),
            "{actual} vs {expected}"
        );
    }
}

#[test]
fn unsupported_quote_risk_is_rejected_before_valuation() {
    let mut v = payload(false, 0.0);
    v["risk"]["vega"] = json!(true);
    let request =
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    assert!(
        BergomiLsvPricingPlan::compile(
            &request,
            Bergomi1Factor::new(2.0, 0.6, -0.5).unwrap(),
            LsvParticleConfig::new(256, 429, 0.35, 5.0, true).unwrap(),
            ExecutionPolicy::new(1, Some(64)).unwrap()
        )
        .is_err()
    );
}
