use pricing::market::MarketIvSurface;
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation, RoughBergomi};
use pricing::{
    JsonLimits, PricingRequest, hull_white::HullWhiteEquityPricingPlan as Plan, parse_request_json,
};
use serde_json::{Value, json};

fn payload(cash: bool, rqmc: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    if cash {
        v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.35,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":6.0,"beta":0.05}},
            {"event_id":2,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":3.0}}
        ]);
    } else {
        v["market"]["discrete_dividends"] =
            json!([{ "event_id":1,"ex_time":0.22,"quote":{"type":"proportional","beta":0.03} }]);
    }
    if rqmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":256,"scramble_count":4,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn rates() -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0, 0.2, 0.6], vec![0.005, 0.008, 0.006]).unwrap()
}
fn policy(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(128)).unwrap()
}
fn corr() -> HybridCorrelation {
    HybridCorrelation::new(-0.5, 0.3, -0.1).unwrap()
}
fn factor(h: f64, eta: f64) -> RoughBergomi {
    RoughBergomi::new(h, eta, -0.5).unwrap()
}
fn pure(v: Value, cash: bool, workers: u32) -> Plan {
    let compile = if cash {
        Plan::compile_rough_bergomi_with_cash_dividends
    } else {
        Plan::compile_rough_bergomi
    };
    compile(
        &request(v),
        factor(0.1, 0.7),
        rates(),
        corr(),
        0.125,
        policy(workers),
    )
    .unwrap()
}
fn quotes() -> Vec<f64> {
    vec![
        0.23, 0.224, 0.22, 0.216, 0.22, 0.236, 0.23, 0.226, 0.222, 0.226,
    ]
}
fn target(q: Vec<f64>) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.4, 1.0], vec![-1.0, -0.3, 0.0, 0.4, 1.0], q).unwrap(),
        vec![0.0, 0.1, 0.35, 0.7, 1.0],
        vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        1e-8,
        4.0,
    )
    .unwrap()
}
fn lsv_request(mut v: Value, t: &HullWhiteLsvTarget) -> PricingRequest {
    let g = t.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),"shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    request(v)
}
fn lsv(v: Value, t: &HullWhiteLsvTarget, cash: bool, workers: u32, trace: bool) -> Plan {
    let compile = if cash {
        Plan::compile_rough_lsv_with_cash_dividends
    } else {
        Plan::compile_rough_lsv
    };
    compile(
        &lsv_request(v, t),
        t,
        factor(0.1, 0.7),
        rates(),
        corr(),
        LsvParticleConfig::new(2048, 711, 0.32, 20.0, trace).unwrap(),
        policy(workers),
    )
    .unwrap()
}
fn close(label: &str, a: f64, b: f64) {
    assert!((a - b).abs() < 3e-6 + 3e-5 * b.abs(), "{label}: {a} != {b}");
}

#[test]
fn pure_rough_aad_matches_crn_spot_volatility_curves_and_worker_replay() {
    for cash in [false, true] {
        for rqmc in [false, true] {
            let v = payload(cash, rqmc);
            let plan = pure(v.clone(), cash, 1);
            let risk = plan.evaluate_aad().unwrap();
            assert_eq!(plan.random_factor_count(), 5);
            assert_eq!(risk.price.value, plan.evaluate().unwrap().value);
            assert_eq!(
                risk.price.scheme,
                "rough-bergomi-hw-hybrid-kappa1-log-euler-v1"
            );
            assert_eq!(risk.parameter_labels[1], "initial_volatility");
            assert!(risk.price.calibration_method.is_none());
            assert!(risk.vega_kt_raw().is_none());
            assert_eq!(risk.standard_errors.is_some(), rqmc);
            for (name, bar, h) in [
                ("spot", risk.delta(), 1e-5),
                ("volatility", risk.vega().unwrap(), 1e-6),
                ("discount_curve", risk.discount_log_df_adjoints()[1], 1e-7),
                ("dividend_curve", risk.dividend_log_df_adjoints()[1], 1e-7),
            ] {
                let price = |bump: f64| {
                    let mut x = v.clone();
                    if name == "spot" {
                        x["market"][name] = json!(100.0 + bump);
                    } else if name == "volatility" {
                        x["model"][name] = json!(0.2 + bump);
                    } else {
                        let old = x["market"][name]["discount_factors"][1].as_f64().unwrap();
                        x["market"][name]["discount_factors"][1] = json!(old * bump.exp());
                    }
                    pure(x, cash, 1).evaluate().unwrap().value
                };
                close(name, bar, (price(h) - price(-h)) / (2.0 * h));
            }
            let replay = pure(v, cash, 4).evaluate_aad().unwrap();
            assert_eq!(risk.derivatives, replay.derivatives);
            assert_eq!(risk.standard_errors, replay.standard_errors);
        }
    }
}

#[test]
fn rough_lsv_recalibrated_aad_vegakt_and_trace_contract() {
    let q = quotes();
    let t = target(q.clone());
    for cash in [false, true] {
        let v = payload(cash, true);
        let plan = lsv(v.clone(), &t, cash, 1, true);
        let risk = plan.evaluate_aad().unwrap();
        assert_eq!(risk.price.value, plan.evaluate().unwrap().value);
        assert!(
            risk.price
                .calibration_method
                .unwrap()
                .starts_with("rough-lsv-hw")
        );
        assert_eq!(risk.vega_kt_raw().unwrap().len(), 10);
        assert_eq!(risk.vega_kt_standard_errors().unwrap().len(), 10);
        let h = 1e-7;
        for (i, &bar) in risk.vega_kt_raw().unwrap().iter().enumerate() {
            let price = |bump: f64| {
                let mut x = q.clone();
                x[i] += bump;
                lsv(v.clone(), &target(x), cash, 1, false)
                    .evaluate()
                    .unwrap()
                    .value
            };
            close(
                &format!("rough IV {i} cash={cash}"),
                bar,
                (price(h) - price(-h)) / (2.0 * h),
            );
        }
        for (name, bar) in [
            ("spot", risk.delta()),
            ("discount_curve", risk.discount_log_df_adjoints()[1]),
            ("dividend_curve", risk.dividend_log_df_adjoints()[1]),
        ] {
            let price = |bump: f64| {
                let mut x = v.clone();
                if name == "spot" {
                    x["market"][name] = json!(100.0 + bump);
                } else {
                    let old = x["market"][name]["discount_factors"][1].as_f64().unwrap();
                    x["market"][name]["discount_factors"][1] = json!(old * bump.exp());
                }
                lsv(x, &t, cash, 1, false).evaluate().unwrap().value
            };
            close(name, bar, (price(h) - price(-h)) / (2.0 * h));
        }
        let replay = lsv(v.clone(), &t, cash, 4, true).evaluate_aad().unwrap();
        assert_eq!(risk.derivatives, replay.derivatives);
        assert_eq!(risk.standard_errors, replay.standard_errors);
        let no_trace = lsv(v, &t, cash, 1, false);
        assert!(no_trace.evaluate_aad().is_err());
        assert_eq!(no_trace.evaluate().unwrap().value, risk.price.value);
    }
}

#[test]
fn rough_zero_eta_matches_bs_and_brownian_lsv_matches_markovian_limit() {
    let mut v = payload(false, true);
    v["market"]["discrete_dividends"] = json!([]);
    let req = request(v.clone());
    let correlation = HybridCorrelation::new(0.0, 0.3, 0.0).unwrap();
    for h in [0.05, 0.5] {
        let rough = Plan::compile_rough_bergomi(
            &req,
            RoughBergomi::new(h, 0.0, 0.0).unwrap(),
            rates(),
            correlation,
            0.125,
            policy(1),
        )
        .unwrap();
        let bs = Plan::compile_bs(&req, rates(), 0.3, 0.125, policy(1)).unwrap();
        assert_eq!(
            rough.evaluate().unwrap().value,
            bs.evaluate().unwrap().value
        );
        assert_eq!(
            rough.evaluate_aad().unwrap().derivatives,
            bs.evaluate_aad().unwrap().derivatives
        );
        assert_ne!(rough.plan_fingerprint(), bs.plan_fingerprint());
    }
    let t = target(vec![0.23; 10]);
    let req = lsv_request(v, &t);
    let particles = LsvParticleConfig::new(2048, 711, 0.32, 20.0, true).unwrap();
    let rough = Plan::compile_rough_lsv(
        &req,
        &t,
        factor(0.5, 0.7),
        rates(),
        corr(),
        particles.clone(),
        policy(1),
    )
    .unwrap();
    let markov = Plan::compile_lsv(
        &req,
        &t,
        Bergomi1Factor::new(0.0, 0.35, -0.5).unwrap(),
        rates(),
        corr(),
        particles,
        policy(1),
    )
    .unwrap();
    assert!((rough.evaluate().unwrap().value - markov.evaluate().unwrap().value).abs() < 1e-11);
    for (&a, &b) in rough
        .evaluate_aad()
        .unwrap()
        .derivatives
        .iter()
        .zip(&markov.evaluate_aad().unwrap().derivatives)
    {
        assert!((a - b).abs() < 1e-8);
    }
    assert_ne!(rough.plan_fingerprint(), markov.plan_fingerprint());
}
