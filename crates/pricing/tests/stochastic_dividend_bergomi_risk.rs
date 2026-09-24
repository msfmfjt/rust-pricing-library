//! Model-parameter reverse versus separately recompiled prices; correlations fixed.
use pricing::mc::ExecutionPolicy;
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::stochastic_dividends::{BuehlerDividendModel, StochasticDividendPricingPlan as Plan};
use pricing::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

fn payload(qmc: bool, seed: u64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["schema_version"] = json!(2);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":4.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":8.0}}]);
    v["market"]["discount_curve"]["times"] = json!([0.0, 0.4, 1.1]);
    v["market"]["discount_curve"]["discount_factors"] = json!([1.0, 0.982, 0.951]);
    v["market"]["dividend_curve"]["times"] = json!([0.0, 0.7, 1.2]);
    v["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 0.992, 0.983]);
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo", "points_per_scramble":128,
        "scramble_count":4,"master_scramble_seed":seed,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":512,"master_seed":seed,
        "variance_reduction":{"antithetic":true,"brownian_bridge":false}})
    };
    v
}

fn compile(v: &Value, family: usize, b: [f64; 4], workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
    let p = ExecutionPolicy::new(workers, Some(64)).unwrap();
    if family == 1 {
        Plan::compile_bergomi(
            &r,
            d,
            Bergomi1Factor::new(b[0], b[2], -0.4).unwrap(),
            0.15,
            0.125,
            p,
        )
    } else {
        Plan::compile_bergomi_two_factor(
            &r,
            d,
            Bergomi2Factor::new([b[0], b[1]], b[2], b[3], [-0.4, -0.2], 0.3).unwrap(),
            [0.15, -0.1],
            0.125,
            p,
        )
    }
    .unwrap()
}
fn check(v: &Value, family: usize, b: [f64; 4]) {
    let plan = compile(v, family, b, 1);
    let basic = plan.evaluate_aad().unwrap();
    let risk = plan.evaluate_bergomi_aad().unwrap();
    assert_eq!(risk.price, plan.evaluate().unwrap());
    assert_eq!(risk.price, basic.price);
    let prefix = basic.derivatives.len();
    assert_eq!(&risk.parameter_labels[..prefix], &*basic.parameter_labels);
    assert_eq!(&risk.derivatives[..prefix], &*basic.derivatives);
    assert_eq!(&risk.standard_errors[..prefix], &*basic.standard_errors);
    assert_eq!(risk.cash_mean_adjoints(), basic.cash_mean_adjoints());
    assert_eq!(risk.discount_node_dv01(), basic.discount_node_dv01());
    assert_eq!(risk.repo_spread_node_dv01(), basic.repo_spread_node_dv01());
    assert_eq!(
        risk.method,
        "buehler-bergomi-parameter-reverse-fixed-correlation-v1"
    );
    assert!(
        risk.standard_errors
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    );
    let active = if family == 1 {
        &[0, 2][..]
    } else {
        &[0, 1, 2, 3][..]
    };
    assert_eq!(risk.derivatives.len(), prefix + active.len());
    for (j, &p) in active.iter().enumerate() {
        let aad = risk.derivatives[prefix + j];
        for h in [1e-5, 1e-6] {
            let mut plus = b;
            plus[p] += h;
            let mut minus = b;
            minus[p] -= h;
            let fd = (compile(v, family, plus, 1).evaluate().unwrap().value
                - compile(v, family, minus, 1).evaluate().unwrap().value)
                / (2.0 * h);
            let budget = 3e-5 + 2e-5 * aad.abs().max(fd.abs());
            assert!(
                (aad - fd).abs() <= budget,
                "family={family}, p={p}, h={h}, AAD={aad}, FD={fd}, budget={budget}"
            );
        }
    }
    let replay = compile(v, family, b, 3).evaluate_bergomi_aad().unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
    assert_eq!(risk.price.value, replay.price.value);
}
#[test]
fn bergomi_parameters_match_recompiled_prices_and_preserve_basic_risk_bitwise() {
    for family in [1, 2] {
        for qmc in [false, true] {
            for seed in [731, 912] {
                check(&payload(qmc, seed), family, [0.8, 2.1, 0.3, 0.35]);
            }
        }
    }
}
#[test]
fn path_dependent_and_smoothed_cash_jump_payoffs_include_model_risk() {
    let mut v = payload(true, 344);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":90.0,"notional":1.0,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
        {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    for family in [1, 2] {
        check(&v, family, [0.8, 2.1, 0.3, 0.35]);
    }
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},"monitoring":{"type":"discrete"},
        "monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
    for family in [1, 2] {
        assert!(
            compile(&v, family, [0.8, 2.1, 0.3, 0.35], 1)
                .evaluate_bergomi_aad()
                .is_err()
        );
    }
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.0});
    for family in [1, 2] {
        check(&v, family, [0.8, 2.1, 0.3, 0.35]);
    }
}
#[test]
fn parameter_boundaries_use_inward_derivatives_and_zero_sigma_is_finite() {
    let mut v = payload(true, 992);
    v["product"]["strike"] = json!(60.0);
    for family in [1, 2] {
        for b in [
            [0.0, 0.0, 0.3, 0.35],
            [0.8, 2.1, 0.0, 0.35],
            [0.8, 2.1, 0.3, 0.0],
            [0.8, 2.1, 0.3, 1.0],
        ] {
            let plan = compile(&v, family, b, 1);
            let risk = plan.evaluate_bergomi_aad().unwrap();
            let prefix = plan.evaluate_aad().unwrap().derivatives.len();
            let active = if family == 1 {
                &[0, 2][..]
            } else {
                &[0, 1, 2, 3][..]
            };
            for (j, &p) in active.iter().enumerate() {
                let h = if p == 3 && b[p] == 1.0 { -1e-7 } else { 1e-7 };
                let mut plus = b;
                plus[p] += h;
                let fd =
                    (compile(&v, family, plus, 1).evaluate().unwrap().value - risk.price.value) / h;
                assert!(
                    (fd - risk.derivatives[prefix + j]).abs() < 3e-4,
                    "family={family}, b={b:?}, p={p}, fd={fd}, aad={}",
                    risk.derivatives[prefix + j]
                );
            }
        }
    }
    v["model"]["volatility"] = json!(0.0);
    for family in [1, 2] {
        let p = compile(&v, family, [0.8, 2.1, 0.3, 0.35], 1);
        let n = p.evaluate_aad().unwrap().derivatives.len();
        let r = p.evaluate_bergomi_aad().unwrap();
        assert!(r.derivatives[n..].iter().all(|x| *x == 0.0));
        assert!(r.standard_errors[n..].iter().all(|x| *x == 0.0));
    }
}
#[test]
fn singular_and_near_singular_parameter_risk_reject_without_disabling_basic_risk() {
    let r = parse_request_json(
        &serde_json::to_vec(&payload(false, 731)).unwrap(),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    let policy = ExecutionPolicy::new(1, Some(64)).unwrap();
    for sd in [1.0, 1.0 - 1e-12] {
        let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, sd).unwrap();
        let p = Plan::compile_bergomi(
            &r,
            d,
            Bergomi1Factor::new(0.8, 0.3, 0.0).unwrap(),
            0.0,
            0.125,
            policy,
        )
        .unwrap();
        assert!(p.evaluate().is_ok());
        assert!(p.evaluate_aad().is_ok());
        let message = p.evaluate_bergomi_aad().unwrap_err().to_string();
        assert!(message.contains("pivots"), "{message}");
    }
    let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
    let bs = Plan::compile_bs(&r, d, 0.125, policy).unwrap();
    assert!(bs.evaluate_aad().is_ok());
    assert!(
        bs.evaluate_bergomi_aad()
            .unwrap_err()
            .to_string()
            .contains("requires a 1F or 2F")
    );
}
