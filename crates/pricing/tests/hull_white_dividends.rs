use pricing::hull_white::HullWhiteEquityPricingPlan as Plan;
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::hull_white::black_value;
use pricing::models::hull_white_dividends::HullWhiteDividendPlan;
use pricing::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn payload() -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":2048,"scramble_count":8,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn policy(n: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(n, Some(128)).unwrap()
}
fn rates(sigma: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0], vec![sigma]).unwrap()
}
fn cash_request(future: bool) -> Value {
    let mut v = payload();
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":6.0}},
        {"event_id":2,"ex_time":if future {1.5} else {1.0},"quote":{"type":"fixed_cash_and_proportional","fixed_cash":4.0,"beta":0.1}}]);
    v
}

#[test]
fn stochastic_reserve_has_correct_jump_and_rate_loading() {
    let req = request(cash_request(true));
    let hw = rates(0.025);
    let times = [0.0, 0.5 - 1e-8, 0.5, 1.0, 1.5];
    let d = HullWhiteDividendPlan::new(&hw, req.market().equity().forward(), &times).unwrap();
    let expected0 = 6.0 * (0.95_f64 / 0.98).powf(0.5) + 4.0 / 0.9 * (0.95_f64 / 0.98).powf(1.5);
    assert!((d.risky_spot() - (100.0 - expected0)).abs() < 1e-12);
    for x in [-0.04, 0.0, 0.06] {
        let left = d.nodes()[1].spots(102.0, x).unwrap().0;
        let (post, pre) = d.nodes()[2].spots(102.0, x).unwrap();
        assert!((left - pre.unwrap()).abs() < 1e-6);
        assert!((left - post - 6.0).abs() < 1e-6);
        let (a, loading) = d.nodes()[3].reserve(x).unwrap();
        assert!(a > 0.0 && loading < 0.0);
        let h = 1e-5;
        let fd = (d.nodes()[3].reserve(x + h).unwrap().0 - d.nodes()[3].reserve(x - h).unwrap().0)
            / (2.0 * h);
        assert!((fd * 0.025 - loading).abs() < 1e-10);
        assert_eq!(d.nodes()[4].reserve(x).unwrap(), (0.0, 0.0));
    }
    assert!(d.nodes()[3].target_state(0.001, 100.0).is_err());
}

#[test]
fn bs_escrowed_dividends_match_gaussian_reference_and_worker_replay() {
    let req = request(cash_request(false));
    let hw = rates(0.02);
    let a = Plan::compile_bs_with_cash_dividends(&req, hw.clone(), 0.4, 0.25, policy(1)).unwrap();
    let b = Plan::compile_bs_with_cash_dividends(&req, hw.clone(), 0.4, 0.25, policy(4)).unwrap();
    let pa = a.evaluate().unwrap();
    let pb = b.evaluate().unwrap();
    assert_eq!(pa.value.to_bits(), pb.value.to_bits());
    assert_eq!(pa.standard_error.to_bits(), pb.standard_error.to_bits());
    let c = hw
        .transition(
            0.0,
            1.0,
            0.0,
            HybridCorrelation::new(0.0, 0.4, 0.0).unwrap(),
        )
        .unwrap()
        .covariance;
    let variance = 0.04 + c[3][3] + 0.4 * c[0][3];
    let expected = black_value(
        0.9 * a.risky_spot() * 0.98 / 0.95,
        100.0,
        variance,
        0.95,
        true,
    )
    .unwrap();
    assert!(
        (pa.value - expected).abs() < 6.0 * pa.standard_error + 0.0005,
        "{} vs {expected}, se={}",
        pa.value,
        pa.standard_error
    );
    assert_eq!(pa.cash_dividend_model, Some("escrowed-hw-bonds-v1"));
    assert!(Plan::compile_bs(&req, hw, 0.4, 0.25, policy(1)).is_err());
}

#[test]
fn cash_barrier_jump_and_immediate_ex_date_are_observed() {
    let mut v = payload();
    v["model"]["volatility"] = json!(0.0);
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":10.0}}]);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,"side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},"monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],"payment_date":"2027-09-04"});
    // Pre-dividend Spot is 103.1579; post-dividend is 93.1579.
    let actual = Plan::compile_bs_with_cash_dividends(&request(v), rates(0.0), 0.0, 1.0, policy(1))
        .unwrap()
        .evaluate()
        .unwrap();
    assert!((actual.value - 12.5).abs() < 1e-12);
    let mut v = payload();
    v["model"]["volatility"] = json!(0.0);
    v["product"]["strike"] = json!(80.0);
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":0.0,"quote":{"type":"fixed_cash","amount":10.0}}]);
    let actual = Plan::compile_bs_with_cash_dividends(&request(v), rates(0.0), 0.0, 1.0, policy(1))
        .unwrap()
        .evaluate()
        .unwrap();
    assert!((actual.value - (90.0 * 0.98 - 80.0 * 0.95)).abs() < 1e-12);
}

fn target_request(target: &HullWhiteLsvTarget) -> PricingRequest {
    let mut v = cash_request(true);
    let g = target.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),"shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    request(v)
}
#[test]
fn cash_lsv_reprices_the_shifted_market_target_with_future_dividends() {
    let target = HullWhiteLsvTarget::flat(
        0.2,
        (0..=32).map(|i| i as f64 / 32.0).collect(),
        (0..=30).map(|i| -0.75 + i as f64 * 0.05).collect(),
        1e-8,
        4.0,
    )
    .unwrap();
    let req = target_request(&target);
    let hw = rates(0.012);
    let plan = Plan::compile_lsv_with_cash_dividends(
        &req,
        &target,
        Bergomi1Factor::new(2.0, 0.4, -0.5).unwrap(),
        hw.clone(),
        HybridCorrelation::new(-0.5, 0.25, -0.1).unwrap(),
        LsvParticleConfig::new(8192, 712, 0.14, 20.0, false).unwrap(),
        policy(2),
    )
    .unwrap();
    let d = HullWhiteDividendPlan::new(
        &hw,
        req.market().equity().forward(),
        target.grid().time_nodes(),
    )
    .unwrap();
    let last = d.nodes().last().unwrap();
    let k = (100.0 - last.deterministic_reserve()) / last.scale();
    let expected = last.scale() * black_value(100.0, k, 0.04, 0.95, true).unwrap();
    let actual = plan.evaluate().unwrap();
    assert!(
        (actual.value - expected).abs() < 6.0 * actual.standard_error + 0.08,
        "{} vs {expected}, se={}",
        actual.value,
        actual.standard_error
    );
    let c = plan.calibration().unwrap();
    assert!(c.conditional_cross_moments.iter().any(|v| v.abs() > 1e-5));
    assert!(c.conditional_rate_variances.iter().any(|v| *v > 1e-8));
    assert!(c.rate_corrections.iter().any(|v| v.abs() > 1e-5));
    assert_eq!(
        actual.calibration_method,
        Some("lsv-hw-escrowed-quadratic-quartic-v1")
    );
}
#[test]
fn zero_rate_and_vol_factor_limits_and_bad_cash_inputs() {
    let target =
        HullWhiteLsvTarget::flat(0.2, vec![0.0, 0.5, 1.0], vec![-0.5, 0.0, 0.5], 1e-8, 4.0)
            .unwrap();
    let plan = Plan::compile_lsv_with_cash_dividends(
        &target_request(&target),
        &target,
        Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap(),
        rates(0.0),
        HybridCorrelation::new(-0.5, 0.25, -0.1).unwrap(),
        LsvParticleConfig::new(128, 712, 0.35, 5.0, false).unwrap(),
        policy(1),
    )
    .unwrap();
    assert_eq!(
        plan.calibration().unwrap().surface.squared_leverage(),
        target.grid().values()
    );
    // Retain the request compiler's exact expiry-grid contract. Dividends may
    // extend beyond expiry, but the effective variance grid must end at expiry.
    let longer_target = HullWhiteLsvTarget::flat(
        0.2,
        vec![0.0, 0.5, 1.0, 1.5, 2.0],
        vec![-0.5, 0.0, 0.5],
        1e-8,
        4.0,
    )
    .unwrap();
    let longer_plan = Plan::compile_lsv_with_cash_dividends(
        &target_request(&longer_target),
        &longer_target,
        Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap(),
        rates(0.0),
        HybridCorrelation::new(-0.5, 0.25, -0.1).unwrap(),
        LsvParticleConfig::new(128, 712, 0.35, 5.0, false).unwrap(),
        policy(1),
    );
    assert!(longer_plan.is_err());
    let mut v = cash_request(false);
    v["market"]["discrete_dividends"][0]["quote"]["amount"] = json!(200.0);
    assert!(
        Plan::compile_bs_with_cash_dividends(&request(v), rates(0.01), 0.2, 1.0, policy(1))
            .is_err()
    );
    let req = request(payload());
    let old = Plan::compile_bs(&req, rates(0.01), 0.2, 0.25, policy(1))
        .unwrap()
        .evaluate()
        .unwrap();
    let cash = Plan::compile_bs_with_cash_dividends(&req, rates(0.01), 0.2, 0.25, policy(1))
        .unwrap()
        .evaluate()
        .unwrap();
    assert!((old.value - cash.value).abs() < 1e-12);
}

#[test]
fn cash_leverage_rejects_incompatible_bond_variance() {
    let target =
        HullWhiteLsvTarget::flat(0.001, vec![0.0, 0.5, 1.0], vec![-0.5, 0.0, 0.5], 1e-8, 4.0)
            .unwrap();
    let err = Plan::compile_lsv_with_cash_dividends(
        &target_request(&target),
        &target,
        Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap(),
        rates(1.0),
        HybridCorrelation::new(-0.5, 0.25, -0.1).unwrap(),
        LsvParticleConfig::new(128, 712, 0.35, 5.0, false).unwrap(),
        policy(1),
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("cash_dividend_leverage_discriminant"),
        "{err}"
    );
}
