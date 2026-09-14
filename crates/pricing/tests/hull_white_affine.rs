use pricing::hull_white::HullWhiteEquityPricingPlan as Plan;
use pricing::market::{DiscountCurve, MarketIvSurface};
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::hull_white::black_value;
use pricing::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation, RoughBergomi};
use pricing::{JsonLimits, PricingRequest, SimulationPlan, parse_request_json};
use serde_json::{Value, json};

fn payload() -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["discount_curve"] =
        json!({"curve_id":10,"times":[0.0,0.4,1.0,2.0],"discount_factors":[1.0,0.985,0.95,0.88]});
    v["market"]["dividend_curve"] =
        json!({"curve_id":11,"times":[0.0,0.7,2.0],"discount_factors":[1.0,0.99,0.96]});
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.0,"quote":{"type":"fixed_cash","amount":1.0}},
        {"event_id":2,"ex_time":0.5,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":4.0,"beta":0.03}},
        {"event_id":3,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":2.0}}
    ]);
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn policy(n: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(n, Some(128)).unwrap()
}
fn rates(sigma: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0, 0.45], vec![sigma, 1.4 * sigma]).unwrap()
}
fn bs(v: Value, workers: u32) -> Plan {
    Plan::compile_bs(&request(v), rates(0.016), 0.35, 0.25, policy(workers)).unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 3e-6 + 3e-5 * b.abs(), "{a} != {b}");
}
fn shock(v: &Value, name: &str, index: usize, amount: f64) -> Value {
    let mut v = v.clone();
    if name == "spot" || name == "volatility" {
        let root = if name == "spot" { "market" } else { "model" };
        v[root][name] = json!(v[root][name].as_f64().unwrap() + amount);
    } else {
        let old = v["market"][name]["discount_factors"][index].as_f64().unwrap();
        v["market"][name]["discount_factors"][index] = json!(old * amount.exp());
    }
    v
}

#[test]
fn deterministic_limit_carries_each_cashflow_and_matches_standard_engine() {
    let mut v = payload();
    v["model"]["volatility"] = json!(0.0);
    v["product"]["strike"] = json!(60.0);
    let req = request(v);
    let market = req.market().equity().forward();
    let growth = |t| market.dividend_curve().discount(t).unwrap()
        / market.discount_curve().discount(t).unwrap();
    let expected_forward = (99.0 * growth(0.5) * 0.97 - 4.0)
        * growth(1.0) / growth(0.5) - 2.0;
    assert!((market.evaluate(1.0).unwrap().spot_contract_forward - expected_forward).abs() < 1e-12);
    let expected = (expected_forward - 60.0) * 0.95;
    let actual = Plan::compile_bs(&req, rates(0.0), 0.0, 0.25, policy(1))
        .unwrap().evaluate().unwrap();
    let standard = SimulationPlan::compile(&req, policy(1)).unwrap().execute().unwrap();
    assert!((actual.value - expected).abs() < 1e-12);
    assert!((standard.pricing_result.value.value().get() - expected).abs() < 1e-12);
    assert_eq!(actual.cash_dividend_model, Some("affine-paid-cash-realized-carry-v1"));
}

#[test]
fn deterministic_rate_limit_with_equity_vol_matches_shifted_black_reference() {
    let mut v = payload();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":2048,"scramble_count":8,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let req = request(v);
    let market = req.market().equity().forward();
    let growth = |t| market.dividend_curve().discount(t).unwrap()
        / market.discount_curve().discount(t).unwrap();
    let offset = -0.97 * growth(1.0) - 4.0 * growth(1.0) / growth(0.5) - 2.0;
    let expected = black_value(0.97 * 100.0 * growth(1.0), 100.0 - offset, 0.04, 0.95, true).unwrap();
    let actual = Plan::compile_bs(&req, rates(0.0), 0.35, 0.25, policy(1))
        .unwrap().evaluate().unwrap();
    assert!((actual.value - expected).abs() < 6.0 * actual.standard_error + 0.001,
        "{} != {expected}; se={}", actual.value, actual.standard_error);
}

#[test]
fn affine_bs_aad_matches_spot_vol_and_every_curve_pillar_with_payment_lag() {
    let mut v = payload();
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":90.0,"notional":1.0,"side":{"type":"call"},"observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},{"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    let plan = bs(v.clone(), 1);
    let risk = plan.evaluate_aad().unwrap();
    assert_eq!(risk.price.value.to_bits(), plan.evaluate().unwrap().value.to_bits());
    assert_eq!(plan.risky_spot(), 100.0);
    let mut checks = vec![("spot", 0, risk.delta(), 1e-5), ("volatility", 0, risk.vega().unwrap(), 1e-6)];
    for (name, bars) in [("discount_curve", risk.discount_log_df_adjoints()),
        ("dividend_curve", risk.dividend_log_df_adjoints())] {
        assert_eq!(bars[0], 0.0);
        checks.extend(bars.iter().enumerate().skip(1).map(|(i, &b)| (name, i, b, 1e-6)));
    }
    for (name, i, bar, h) in checks {
        let up = bs(shock(&v, name, i, h), 1).evaluate().unwrap().value;
        let down = bs(shock(&v, name, i, -h), 1).evaluate().unwrap().value;
        close(bar, (up - down) / (2.0 * h));
    }
    let replay = bs(v, 4).evaluate_aad().unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.price.value.to_bits(), replay.price.value.to_bits());
}

#[test]
fn future_cash_has_no_reserve_effect_and_rqmc_risks_replay() {
    let mut v = payload();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":256,"scramble_count":4,"master_scramble_seed":712,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let a = bs(v.clone(), 1);
    v["market"]["discrete_dividends"].as_array_mut().unwrap().push(
        json!({"event_id":4,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":10000.0}}));
    let b = bs(v.clone(), 1);
    assert_eq!(a.time_nodes(), b.time_nodes());
    assert_eq!(a.random_factor_count(), b.random_factor_count());
    let aa = a.evaluate_aad().unwrap();
    let ab = b.evaluate_aad().unwrap();
    assert_eq!(aa.price.value.to_bits(), ab.price.value.to_bits());
    assert_eq!(aa.derivatives, ab.derivatives);
    assert_eq!(aa.standard_errors, ab.standard_errors);
    assert!(ab.standard_errors.is_some());
    let replay = bs(v, 3).evaluate_aad().unwrap();
    assert_eq!(ab.derivatives, replay.derivatives);
    assert_eq!(ab.standard_errors, replay.standard_errors);
}

fn quotes() -> Vec<f64> {
    vec![0.23, 0.224, 0.22, 0.216, 0.22, 0.236, 0.23, 0.226, 0.222, 0.226]
}
fn target(q: Vec<f64>) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.4, 1.0], vec![-1.0, -0.3, 0.0, 0.4, 1.0], q).unwrap(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        1e-8, 4.0,
    ).unwrap()
}
fn lsv(t: &HullWhiteLsvTarget, rough: bool, trace: bool, workers: u32) -> Plan {
    let mut v = payload();
    let g = t.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),"shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    let req = request(v);
    let corr = HybridCorrelation::new(-0.5, 0.3, -0.1).unwrap();
    let particles = LsvParticleConfig::new(1024, 711, 0.32, 10.0, trace).unwrap();
    if rough {
        Plan::compile_rough_lsv(&req, t, RoughBergomi::new(0.1, 0.7, -0.5).unwrap(),
            rates(0.01), corr, particles, policy(workers)).unwrap()
    } else {
        Plan::compile_lsv(&req, t, Bergomi1Factor::new(2.0, 0.4, -0.5).unwrap(),
            rates(0.01), corr, particles, policy(workers)).unwrap()
    }
}

#[test]
fn affine_lsv_and_rough_lsv_vegakt_reverse_every_recalibrated_quote() {
    let q = quotes();
    let t = target(q.clone());
    for rough in [false, true] {
        let plan = lsv(&t, rough, true, 1);
        let risk = plan.evaluate_aad().unwrap();
        assert_eq!(risk.price.value.to_bits(), plan.evaluate().unwrap().value.to_bits());
        assert_eq!(risk.price.cash_dividend_model, Some("affine-paid-cash-realized-carry-v1"));
        assert!(plan.calibration().unwrap().conditional_rate_variances.iter().all(|v| *v == 0.0));
        let bars = risk.vega_kt_raw().unwrap();
        assert_eq!(bars.len(), q.len());
        for (i, &bar) in bars.iter().enumerate() {
            let h = 1e-7;
            let mut up = q.clone();
            let mut down = q.clone();
            up[i] += h;
            down[i] -= h;
            let a = lsv(&target(up), rough, false, 1).evaluate().unwrap().value;
            let b = lsv(&target(down), rough, false, 1).evaluate().unwrap().value;
            close(bar, (a - b) / (2.0 * h));
        }
        close(risk.vega().unwrap(), bars.iter().sum());
        assert!(lsv(&t, rough, false, 1).evaluate_aad().is_err());
        assert_eq!(risk.derivatives, lsv(&t, rough, true, 3).evaluate_aad().unwrap().derivatives);
    }
}

#[test]
fn affine_rough_bergomi_spot_and_sigma_adjoints_match_finite_differences() {
    let compile = |v: Value| Plan::compile_rough_bergomi(
        &request(v), RoughBergomi::new(0.1, 0.7, -0.5).unwrap(), rates(0.01),
        HybridCorrelation::new(-0.5, 0.3, -0.1).unwrap(), 0.25, policy(1),
    ).unwrap();
    let v = payload();
    let plan = compile(v.clone());
    let risk = plan.evaluate_aad().unwrap();
    assert_eq!(risk.price.value.to_bits(), plan.evaluate().unwrap().value.to_bits());
    for (name, bar, h) in [("spot", risk.delta(), 1e-5), ("volatility", risk.vega().unwrap(), 1e-6)] {
        let a = compile(shock(&v, name, 0, h)).evaluate().unwrap().value;
        let b = compile(shock(&v, name, 0, -h)).evaluate().unwrap().value;
        close(bar, (a - b) / (2.0 * h));
    }
}
