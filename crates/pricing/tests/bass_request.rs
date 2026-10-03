use pricing::bass_lv::*;
use pricing::core::{Date, DayCountConvention};
use pricing::market::DiscountCurve;
use pricing::mc::{ExecutionPolicy, Philox4x32, RandomCoordinate, RandomDomain};
use pricing::models::{ModelSpec, hull_white::black_value};
use pricing::*;
use pricing_numerics::{standard_normal_cdf as cdf, standard_normal_pdf as pdf};
use serde_json::{Value, json};

fn time(date: &str) -> f64 {
    DayCountConvention::Act365F
        .year_fraction("2026-09-04".parse::<Date>().unwrap(), date.parse().unwrap())
}
fn payload() -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    v["model"] = json!({"type":"bass_local_volatility","parameters":{
        "maturity_nodes":[time("2027-03-04"),1.0],"log_forward_moneyness_nodes":[-2.0,0.0,2.0],"implied_volatilities":vec![0.2;6],
        "projection_nodes":(0..=800).map(|i|-2.0+4.0*i as f64/800.0).collect::<Vec<_>>(),
        "grid_points":401,"grid_width":8.0,"max_iterations":2000,"cdf_tolerance":1e-7,"tail_tolerance":1e-7,
        "projection_tail_tolerance":1e-7,"projection_mean_tolerance":1e-4,"iv_bump":1e-4}});
    v["engine"]["independent_sampling_units"] = json!(512);
    v
}
fn request(v: &Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn plan(v: &Value, workers: u32) -> PricingPlan {
    compile(
        &request(v),
        ExecutionPolicy::new(workers, Some(128)).unwrap(),
    )
    .unwrap()
}
fn price(v: &Value) -> MonteCarloPrice {
    plan(v, 1).evaluate().unwrap()
}
fn risk(v: &mut Value) {
    v["risk"] = json!({"delta":true,"gamma":{"bump":{"type":"relative","value":0.002}},"vega":true,"vega_kt":{"maturity_nodes":["2027-03-04","2027-09-04"],"log_forward_moneyness_nodes":[-2.0,0.0,2.0],"relative_density_threshold":1e-5,"full_bucket_covariance":true},"smile_dynamics":{"type":"sticky_log_moneyness"}});
}
fn no_risk(v: &mut Value) {
    v["risk"] =
        json!({"delta":false,"vega":false,"smile_dynamics":{"type":"sticky_log_moneyness"}});
}
fn rqmc(v: &mut Value) {
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":2048,"scramble_count":8,"master_scramble_seed":41,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
}

#[test]
fn request_wire_plan_fingerprints_result_roundtrip_and_legacy_rejection() {
    let v = payload();
    let r = request(&v);
    let encoded = request_to_json(&r).unwrap();
    let decoded = parse_request_json(encoded.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(r, decoded);
    assert_eq!(
        fingerprint_request(&r).unwrap(),
        fingerprint_request(&decoded).unwrap()
    );
    let p = plan(&v, 2);
    assert_eq!(p.bass_calibration_diagnostics().unwrap().len(), 2);
    assert_eq!(p.bass_projection_diagnostics().unwrap().len(), 2);
    let result = p.evaluate().unwrap();
    let encoded = monte_carlo_result_to_json(&result).unwrap();
    assert_eq!(
        result,
        parse_monte_carlo_result_json(encoded.as_bytes(), JsonLimits::DEFAULT).unwrap()
    );
    for version in [1, 2] {
        let mut old = v.clone();
        old["schema_version"] = json!(version);
        assert!(matches!(
            parse_request_json(&serde_json::to_vec(&old).unwrap(), JsonLimits::DEFAULT),
            Err(WireError::UnsupportedSchemaFeature { .. })
        ));
    }
    let mut changed = v.clone();
    changed["model"]["parameters"]["iv_bump"] = json!(2e-4);
    assert_ne!(
        p.request_fingerprint(),
        plan(&changed, 2).request_fingerprint()
    );
    assert_ne!(p.plan_fingerprint(), plan(&v, 1).plan_fingerprint());
}

#[test]
fn carry_cash_proportional_and_future_dividends_match_shifted_black_and_risk() {
    let mut v = payload();
    rqmc(&mut v);
    risk(&mut v);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":6.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"proportional","beta":0.1}},
        {"event_id":3,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    let r = request(&v);
    let f = r.market().equity().forward().evaluate(1.0).unwrap();
    let forward = f.affine_coordinate.b() * f.forward;
    let strike = 100.0 - f.affine_coordinate.a() * 100.0;
    let expected = black_value(forward, strike, 0.04, 0.95, true).unwrap();
    let a = price(&v);
    let value = a.pricing_result.value;
    assert!(
        (value.value().get() - expected).abs() < 6.0 * value.standard_error().get() + 0.005,
        "price {} vs {expected}",
        value.value().get()
    );
    let d1 = (forward / strike).ln() / 0.2 + 0.1;
    let derivative = f.affine_coordinate.spot_scale() * f.forward / 100.0;
    for (estimate, reference, allowance) in [
        (
            a.pricing_result.risks.delta.unwrap().raw(),
            0.95 * cdf(d1) * derivative,
            0.002,
        ),
        (
            a.pricing_result.risks.gamma.unwrap().raw(),
            0.95 * pdf(d1) / (forward * 0.2) * derivative.powi(2),
            0.003,
        ),
        (
            a.pricing_result.risks.vega.unwrap().raw(),
            0.95 * forward * pdf(d1),
            0.05,
        ),
    ] {
        assert!(
            (estimate.value().get() - reference).abs()
                < 6.0 * estimate.standard_error().get() + allowance,
            "{} vs {reference}, se {}",
            estimate.value().get(),
            estimate.standard_error().get()
        );
    }
    assert_eq!(
        a.risk_diagnostics.methods.delta,
        Some(RiskMethod::CentralBump)
    );
    assert_eq!(
        a.risk_diagnostics.methods.gamma,
        Some(RiskMethod::CentralBump)
    );
    assert_eq!(a.independent_sampling_units, 8);
    assert_eq!(a.evaluated_paths, 32768);
    assert!(a.diagnostics.direction_checksum.is_some());
    let b = plan(&v, 3).evaluate().unwrap();
    assert_eq!(a.pricing_result, b.pricing_result);
    assert_eq!(a.risk_diagnostics, b.risk_diagnostics);
    let kt = a.pricing_result.risks.vega_kt.as_ref().unwrap();
    assert_eq!(kt.policy_label(), BASS_VEGA_KT_METHOD);
    assert_eq!(kt.full_bucket_covariance().unwrap().len(), 36);
    let encoded = monte_carlo_result_to_json(&a).unwrap();
    assert_eq!(
        a,
        parse_monte_carlo_result_json(encoded.as_bytes(), JsonLimits::DEFAULT).unwrap()
    );
}

#[test]
fn paired_spot_and_quote_risks_match_independent_request_recompiles() {
    let mut v = payload();
    risk(&mut v);
    let a = price(&v);
    assert_eq!(
        plan(&v, 1).bass_vega_scenario_diagnostics().unwrap().len(),
        14
    );
    let mut plain = v.clone();
    no_risk(&mut plain);
    assert_eq!(a.pricing_result.value, price(&plain).pricing_result.value);
    let mut up = plain.clone();
    let mut dn = plain.clone();
    up["market"]["spot"] = json!(100.2);
    dn["market"]["spot"] = json!(99.8);
    let pu = price(&up).pricing_result.value.value().get();
    let pd = price(&dn).pricing_result.value.value().get();
    let p = a.pricing_result.value.value().get();
    assert!(
        (a.pricing_result.risks.delta.unwrap().raw().value().get() - (pu - pd) / 0.4).abs() < 1e-10
    );
    assert!(
        (a.pricing_result.risks.gamma.unwrap().raw().value().get() - (pu - 2.0 * p + pd) / 0.04)
            .abs()
            < 1e-9
    );
    for j in [1, 4] {
        let mut up = plain.clone();
        let mut dn = plain.clone();
        up["model"]["parameters"]["implied_volatilities"][j] = json!(0.2001);
        dn["model"]["parameters"]["implied_volatilities"][j] = json!(0.1999);
        let fd = (price(&up).pricing_result.value.value().get()
            - price(&dn).pricing_result.value.value().get())
            / 2e-4;
        let got = a
            .pricing_result
            .risks
            .vega_kt
            .as_ref()
            .unwrap()
            .raw_buckets()[j]
            .get();
        assert!((got - fd).abs() < 1e-8, "j={j} {got} vs {fd}");
    }
}

#[test]
fn weighted_known_fixings_delayed_payment_and_fully_fixed_contract() {
    let mut v = payload();
    rqmc(&mut v);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":100.0,"notional":3.0,"side":{"type":"call"},"observations":[{"date":"2026-08-04","weight":0.4,"value":{"type":"known","fixing":90.0}},{"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    let r = request(&v);
    let df = r
        .market()
        .equity()
        .forward()
        .discount_curve()
        .evaluate(time("2027-12-04"))
        .unwrap()
        .discount;
    let expected = 3.0 * black_value(0.6 * 100.0 * 0.98 / 0.95, 64.0, 0.04, df, true).unwrap();
    let a = price(&v).pricing_result.value;
    assert!((a.value().get() - expected).abs() < 6.0 * a.standard_error().get() + 0.004);
    v["product"]["observations"][1] =
        json!({"date":"2026-08-20","weight":0.6,"value":{"type":"known","fixing":120.0}});
    risk(&mut v);
    v["engine"]["points_per_scramble"] = json!(16);
    let a = price(&v);
    assert!((a.pricing_result.value.value().get() - 24.0 * df).abs() < 1e-12);
    assert_eq!(a.pricing_result.value.standard_error().get(), 0.0);
    assert_eq!(
        a.pricing_result.risks.delta.unwrap().raw().value().get(),
        0.0
    );
    assert_eq!(
        a.pricing_result.risks.vega.unwrap().raw().value().get(),
        0.0
    );
    assert!(
        a.pricing_result
            .risks
            .vega_kt
            .unwrap()
            .raw_buckets()
            .iter()
            .all(|v| v.get() == 0.0)
    );
}

#[test]
fn discrete_barrier_evaluates_both_sides_of_cash_dividend_jump() {
    let mut v = payload();
    v["engine"]["variance_reduction"]["antithetic"] = json!(false);
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":10.0}}]);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,"side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},"monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],"payment_date":"2027-09-04"});
    let r = request(&v);
    let ModelSpec::BassLocalVolatility(s) = r.model() else {
        panic!()
    };
    let m = BassMarketIvModel::calibrate(
        1.0,
        s.surface().clone(),
        s.projection_nodes().to_vec(),
        s.config(),
        s.projection_config(),
    )
    .unwrap();
    let simulation = m.model().compile_simulation(vec![1.0]).unwrap();
    let f = r.market().equity().forward();
    let end = f.evaluate(1.0).unwrap();
    let events = f.discrete_dividends().unwrap().event_timeline().unwrap();
    let rng = Philox4x32::from_seed(7);
    let mut total = 0.0;
    let mut post_only = 0.0;
    for p in 0..512 {
        let z: Vec<_> = (0..simulation.normal_count())
            .map(|d| {
                rng.standard_normal(RandomCoordinate::new(p, d as u32, RandomDomain::Valuation))
            })
            .collect();
        let x = simulation.path_from_normals(&z).unwrap()[0];
        let pre = events[0]
            .before()
            .reconstruct_spot(f.spot(), end.forward * x);
        let post = end
            .affine_coordinate
            .reconstruct_spot(f.spot(), end.forward * x);
        let payoff = 0.95 * (post - 80.0).max(0.0);
        if pre >= 100.0 || post >= 100.0 {
            total += payoff;
        }
        if post >= 100.0 {
            post_only += payoff;
        }
    }
    let actual = price(&v).pricing_result.value.value().get();
    assert!((actual - total / 512.0).abs() < 1e-12);
    assert!(total - post_only > 1.0);
}

#[test]
fn incompatible_features_and_axes_fail_explicitly() {
    let mut v = payload();
    risk(&mut v);
    v["risk"]["smile_dynamics"] = json!({"type":"sticky_strike"});
    assert!(compile(&request(&v), ExecutionPolicy::new(1, None).unwrap()).is_err());
    v["risk"]["smile_dynamics"] = json!({"type":"sticky_log_moneyness"});
    v["risk"]["vega_kt"]["log_forward_moneyness_nodes"] = json!([-1.0, 0.0, 1.0]);
    assert!(compile(&request(&v), ExecutionPolicy::new(1, None).unwrap()).is_err());
    let mut v = payload();
    v["product"]["expiry"] = json!("2028-09-04");
    assert!(compile(&request(&v), ExecutionPolicy::new(1, None).unwrap()).is_err());
    let mut v = payload();
    v["model"]["parameters"]["grid_points"] = json!(100);
    assert!(parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).is_err());
}
