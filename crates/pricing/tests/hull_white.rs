use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::market::LocalVarianceGrid;
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::hull_white::{b, black_value};
use pricing::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use pricing_core::DayCountConvention;
use pricing_numerics::standard_normal_pdf;
use serde_json::{Value, json};

fn payload(qmc: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":2048,"scramble_count":8,"master_scramble_seed":612,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn rates(sigma: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0, 0.45], vec![sigma, sigma * 1.5]).unwrap()
}
fn policy(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(128)).unwrap()
}

#[test]
fn bs_hw_matches_gaussian_closed_form_including_payment_lag_and_correlation() {
    for (rho, lag, beta) in [
        (-0.6, false, 0.0),
        (0.0, false, 0.0),
        (0.6, false, 0.0),
        (0.6, true, 0.0),
        (0.6, false, 0.1),
    ] {
        let mut v = payload(true);
        if lag {
            v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":100.0,"notional":1.0,"side":{"type":"call"},"observations":[{"date":"2027-09-04","weight":1.0,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
        }
        if beta != 0.0 {
            v["market"]["discrete_dividends"] =
                json!([{"event_id":1,"ex_time":0.5,"quote":{"type":"proportional","beta":beta}}]);
        }
        let request = request(v);
        let model = rates(0.016);
        let t = 1.0;
        let u = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), request.product().payment_date());
        let c = model
            .transition(0.0, t, 0.0, HybridCorrelation::new(0.0, rho, 0.0).unwrap())
            .unwrap()
            .covariance;
        let variance = 0.04 * t + c[3][3] + 0.4 * c[0][3];
        let forward =
            (1.0 - beta) * 100.0 * 0.98 / 0.95 * (-b(0.2, u - t) * (c[3][2] + 0.2 * c[0][2])).exp();
        let expected = black_value(forward, 100.0, variance, 0.95_f64.powf(u), true).unwrap();
        let result = HullWhiteEquityPricingPlan::compile_bs(&request, model, rho, 1.0, policy(2))
            .unwrap()
            .evaluate()
            .unwrap();
        assert!(
            (result.value - expected).abs() < 6.0 * result.standard_error + 0.0002,
            "rho={rho}, lag={lag}: {} vs {expected}, se={}",
            result.value,
            result.standard_error
        );
        assert_eq!(result.independent_sampling_units, 8);
        assert_eq!(result.evaluated_paths, 32768);
        assert!(result.calibration_method.is_none());
    }
}

#[test]
fn replay_and_zero_rate_volatility_match_bs_limit() {
    for qmc in [false, true] {
        let request = request(payload(qmc));
        let a = HullWhiteEquityPricingPlan::compile_bs(&request, rates(0.01), -0.3, 0.2, policy(1))
            .unwrap();
        let b = HullWhiteEquityPricingPlan::compile_bs(&request, rates(0.01), -0.3, 0.2, policy(4))
            .unwrap();
        let pa = a.evaluate().unwrap();
        let pb = b.evaluate().unwrap();
        assert_eq!(pa.value.to_bits(), pb.value.to_bits());
        assert_eq!(pa.standard_error.to_bits(), pb.standard_error.to_bits());
        assert_eq!(
            a.plan_fingerprint(),
            HullWhiteEquityPricingPlan::compile_bs(&request, rates(0.01), -0.3, 0.2, policy(1))
                .unwrap()
                .plan_fingerprint()
        );
        assert_ne!(
            a.plan_fingerprint(),
            HullWhiteEquityPricingPlan::compile_bs(&request, rates(0.012), -0.3, 0.2, policy(1))
                .unwrap()
                .plan_fingerprint()
        );
    }
    let mut v = payload(true);
    v["model"]["volatility"] = json!(0.0);
    let p = HullWhiteEquityPricingPlan::compile_bs(&request(v), rates(0.0), 1.0, 1.0, policy(1))
        .unwrap()
        .evaluate()
        .unwrap();
    assert!((p.value - 3.0).abs() < 1e-12);
    assert_eq!(p.standard_error, 0.0);
}

fn flat_target() -> HullWhiteLsvTarget {
    let times = (0..=32).map(|i| i as f64 / 32.0).collect::<Vec<_>>();
    let xs = (0..=30)
        .map(|i| -0.75 + i as f64 * 0.05)
        .collect::<Vec<_>>();
    let densities = times
        .iter()
        .flat_map(|&t| {
            xs.iter().map(move |&k| {
                if t == 0.0 {
                    0.0
                } else {
                    let root = (0.04 * t).sqrt();
                    standard_normal_pdf(-k / root - 0.5 * root) / root
                }
            })
        })
        .collect();
    HullWhiteLsvTarget::new(
        LocalVarianceGrid::new(times, xs, vec![0.04; 33 * 31], 1e-8, 4.0).unwrap(),
        densities,
    )
    .unwrap()
}
fn lsv_request(target: &HullWhiteLsvTarget) -> PricingRequest {
    let mut v = payload(true);
    let g = target.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),"shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    request(v)
}

#[test]
fn discounted_particle_calibration_reprices_an_independent_vanilla() {
    let target = flat_target();
    let request = lsv_request(&target);
    let factor = Bergomi1Factor::new(2.0, 0.4, -0.5).unwrap();
    let plan = HullWhiteEquityPricingPlan::compile_lsv(
        &request,
        &target,
        factor,
        rates(0.012),
        HybridCorrelation::new(-0.5, 0.25, -0.1).unwrap(),
        LsvParticleConfig::new(8192, 712, 0.14, 20.0, false).unwrap(),
        policy(2),
    )
    .unwrap();
    let calibration = plan.calibration().unwrap();
    assert!(calibration.rate_corrections.iter().any(|v| v.abs() > 1e-4));
    assert!(
        calibration
            .conditional_second_moments
            .iter()
            .any(|v| (v - 1.0).abs() > 1e-3)
    );
    for row in &calibration.diagnostics {
        assert!((row.mean_relative_discount - 1.0).abs() < 0.002);
        assert!((row.mean_discounted_normalized_equity - 100.0).abs() < 1.5);
    }
    let actual = plan.evaluate().unwrap();
    let expected = black_value(100.0 * 0.98 / 0.95, 100.0, 0.04, 0.95, true).unwrap();
    assert!(
        (actual.value - expected).abs() < 6.0 * actual.standard_error + 0.08,
        "{} vs {expected}, se={}",
        actual.value,
        actual.standard_error
    );
    assert_eq!(actual.calibration_seed, Some(712));
}

#[test]
fn calibration_zero_factor_limit_and_unsupported_contracts_are_explicit() {
    let target = flat_target();
    let factor = Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap();
    let corr = HybridCorrelation::new(-0.5, 0.0, 0.0).unwrap();
    let particles = LsvParticleConfig::new(128, 71, 0.14, 5.0, false).unwrap();
    let plan = HullWhiteEquityPricingPlan::compile_lsv(
        &lsv_request(&target),
        &target,
        factor,
        rates(0.0),
        corr,
        particles.clone(),
        policy(1),
    )
    .unwrap();
    assert_eq!(
        plan.calibration().unwrap().surface.squared_leverage(),
        target.grid().values()
    );
    let mut v = payload(false);
    v["risk"]["vega"] = json!(true);
    assert!(
        HullWhiteEquityPricingPlan::compile_bs(&request(v), rates(0.01), 0.0, 0.2, policy(1))
            .is_err()
    );
    let mut v = payload(false);
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":1.5}}]);
    assert!(
        HullWhiteEquityPricingPlan::compile_bs(&request(v), rates(0.01), 0.0, 0.2, policy(1))
            .is_err()
    );
    let mut densities = target.log_densities().to_vec();
    densities[40] = -1.0;
    assert!(HullWhiteLsvTarget::new(target.grid().clone(), densities).is_err());
    assert!(
        HullWhiteEquityPricingPlan::compile_lsv(
            &lsv_request(&target),
            &target,
            factor,
            rates(0.01),
            corr,
            LsvParticleConfig::new(128, 71, 0.14, 5.0, true).unwrap(),
            policy(1)
        )
        .is_err()
    );
}

#[test]
fn proportional_dividend_barrier_observes_pre_and_post_jump() {
    for style in ["knock_in", "knock_out"] {
        let mut v = payload(true);
        v["model"]["volatility"] = json!(0.0);
        v["market"]["discrete_dividends"] =
            json!([{"event_id":1,"ex_time":1.0,"quote":{"type":"proportional","beta":0.2}}]);
        v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,
            "expiry":"2027-09-04","strike":70.0,"barrier":102.0,"notional":1.0,
            "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":style},
            "monitoring_dates":["2027-09-04"],"payment_date":"2027-09-04"});
        // Pre-dividend Spot = 103.1579 crosses 102, post-dividend = 82.5263.
        let result =
            HullWhiteEquityPricingPlan::compile_bs(&request(v), rates(0.0), 0.0, 1.0, policy(1))
                .unwrap()
                .evaluate()
                .unwrap();
        let expected = if style == "knock_in" { 11.9 } else { 0.0 };
        assert!((result.value - expected).abs() < 1e-12);
        assert_eq!(result.standard_error, 0.0);
    }
}

#[test]
fn target_mismatch_missing_times_and_unsupported_density_fail_explicitly() {
    let target = flat_target();
    let factor = Bergomi1Factor::new(2.0, 0.2, -0.5).unwrap();
    let corr = HybridCorrelation::new(-0.5, 0.2, 0.0).unwrap();
    let particles = LsvParticleConfig::new(128, 71, 0.2, 5.0, false).unwrap();
    let altered = HullWhiteLsvTarget::flat(
        0.21,
        target.grid().time_nodes().to_vec(),
        target.grid().log_moneyness_nodes().to_vec(),
        1e-8,
        4.0,
    )
    .unwrap();
    let err = HullWhiteEquityPricingPlan::compile_lsv(
        &lsv_request(&target),
        &altered,
        factor,
        rates(0.01),
        corr,
        particles.clone(),
        policy(1),
    )
    .unwrap_err();
    assert!(err.to_string().contains("density_target_model_mismatch"));
    // Reuse the same LV request, but observe halfway through a calibration step.
    let json = pricing::request_to_json(&lsv_request(&target)).unwrap();
    let mut v: Value = serde_json::from_str(&json).unwrap();
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":100.0,"notional":1.0,"side":{"type":"call"},"observations":[{"date":"2027-08-04","weight":0.5,"value":{"type":"unknown"}},{"date":"2027-09-04","weight":0.5,"value":{"type":"unknown"}}],"payment_date":"2027-09-04"});
    let err = HullWhiteEquityPricingPlan::compile_lsv(
        &request(v),
        &target,
        factor,
        rates(0.01),
        corr,
        particles.clone(),
        policy(1),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("target_missing_contractual_time"),
        "{err}"
    );
    let zero_density = HullWhiteLsvTarget::new(
        target.grid().clone(),
        vec![0.0; target.grid().values().len()],
    )
    .unwrap();
    let err = HullWhiteEquityPricingPlan::compile_lsv(
        &lsv_request(&zero_density),
        &zero_density,
        factor,
        rates(0.01),
        corr,
        particles,
        policy(1),
    )
    .unwrap_err();
    assert!(err.to_string().contains("no_supported_calibration_node"));
}
