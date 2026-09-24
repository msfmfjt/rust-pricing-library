//! Full-recompile finite differences use identical random streams, never an AAD
//! reference path. These are finite-algorithm derivative tests, not price accuracy.
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
fn compile(v: &Value, family: usize, d: [f64; 3], workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let d = BuehlerDividendModel::new(d[0], d[1], d[2], -0.25).unwrap();
    let p = ExecutionPolicy::new(workers, Some(64)).unwrap();
    match family {
        0 => Plan::compile_bs(&r, d, 0.125, p),
        1 => Plan::compile_bergomi(
            &r,
            d,
            Bergomi1Factor::new(0.8, 0.3, -0.4).unwrap(),
            0.15,
            0.125,
            p,
        ),
        _ => Plan::compile_bergomi_two_factor(
            &r,
            d,
            Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [-0.4, -0.2], 0.3).unwrap(),
            [0.15, -0.1],
            0.125,
            p,
        ),
    }
    .unwrap()
}
fn bumped(v: &Value, family: usize, d: [f64; 3], label: &str, h: f64) -> f64 {
    let mut v = v.clone();
    let mut d = d;
    match label {
        "spot" => v["market"]["spot"] = json!(v["market"]["spot"].as_f64().unwrap() + h),
        "initial_volatility" => {
            v["model"]["volatility"] = json!(v["model"]["volatility"].as_f64().unwrap() + h)
        }
        "dividend_mean_reversion" => d[0] += h,
        "equity_linkage" => d[1] += h,
        "dividend_volatility" => d[2] += h,
        _ => {
            let i = label
                .split('[')
                .nth(1)
                .unwrap()
                .trim_end_matches(']')
                .parse::<usize>()
                .unwrap();
            if label.starts_with("cash_mean") {
                let amount = &mut v["market"]["discrete_dividends"][i - 1]["quote"]["amount"];
                *amount = json!(amount.as_f64().unwrap() + h);
            } else {
                let curve = if label.starts_with("discount") {
                    "discount_curve"
                } else {
                    "dividend_curve"
                };
                let df = &mut v["market"][curve]["discount_factors"][i];
                *df = json!(df.as_f64().unwrap() * h.exp());
            }
        }
    }
    compile(&v, family, d, 1).evaluate().unwrap().value
}
fn check(v: &Value, family: usize, d: [f64; 3]) {
    let plan = compile(v, family, d, 1);
    let risk = plan.evaluate_aad().unwrap();
    assert_eq!(risk.price, plan.evaluate().unwrap());
    assert_eq!(risk.parameter_labels.len(), risk.derivatives.len());
    assert_eq!(risk.derivatives.len(), risk.standard_errors.len());
    assert!(
        risk.standard_errors
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    );
    for (label, &aad) in risk.parameter_labels.iter().zip(&risk.derivatives) {
        if label.ends_with("log_df[0]") {
            assert_eq!(aad, 0.0);
            continue;
        }
        for h in [1e-5, 1e-6] {
            let fd = (bumped(v, family, d, label, h) - bumped(v, family, d, label, -h)) / (2.0 * h);
            let budget = 3e-5 + 2e-5 * aad.abs().max(fd.abs());
            assert!(
                (aad - fd).abs() <= budget,
                "family={family}, {label}, h={h}, AAD={aad}, FD={fd}, budget={budget}"
            );
        }
    }
    let replay = compile(v, family, d, 3).evaluate_aad().unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
    assert_eq!(risk.price.value, replay.price.value);
    assert_eq!(
        risk.initial_volatility_vega_per_vol_point(),
        0.01 * risk.derivatives[1]
    );
    assert_eq!(
        risk.dividend_volatility_vega_per_vol_point(),
        0.01 * risk.derivatives[4]
    );
    assert_eq!(risk.cash_mean_adjoints(), &risk.derivatives[5..8]);
    for (i, v) in risk.discount_node_dv01().iter().enumerate() {
        assert_eq!(*v, -1e-4 * risk.discount_times[i] * risk.derivatives[8 + i]);
    }
    for (i, v) in risk.repo_spread_node_dv01().iter().enumerate() {
        assert_eq!(
            *v,
            -1e-4 * risk.repo_spread_times[i] * risk.derivatives[11 + i]
        );
    }
}
#[test]
fn all_market_and_dividend_derivatives_match_recompiled_fd_and_worker_replay() {
    for family in 0..3 {
        for qmc in [false, true] {
            for seed in [731, 912] {
                check(&payload(qmc, seed), family, [0.7, 0.6, 0.35]);
            }
        }
    }
}
#[test]
fn asian_observations_and_delayed_payment_reverse_curves_and_cash() {
    let mut v = payload(true, 344);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":90.0,"notional":1.0,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
        {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    for family in 0..3 {
        check(&v, family, [0.7, 0.6, 0.35]);
    }
}
#[test]
fn smoothed_barrier_reverse_includes_both_sides_of_dividend_jump() {
    let mut v = payload(false, 734);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},"monitoring":{"type":"discrete"},
        "monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
    for family in 0..3 {
        assert!(
            compile(&v, family, [0.7, 0.6, 0.35], 1)
                .evaluate_aad()
                .is_err()
        );
    }
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.0});
    for family in 0..3 {
        check(&v, family, [0.7, 0.6, 0.35]);
    }
}
#[test]
fn zero_parameter_boundaries_use_one_sided_derivatives_without_division() {
    let mut v = payload(true, 992);
    v["model"]["volatility"] = json!(0.0);
    v["product"]["strike"] = json!(60.0);
    let d = [0.0, 0.0, 0.0];
    for family in 0..3 {
        let p = compile(&v, family, d, 1);
        let r = p.evaluate_aad().unwrap();
        for j in 1..5 {
            let h = 1e-7;
            let fd = (bumped(&v, family, d, &r.parameter_labels[j], h) - r.price.value) / h;
            assert!(
                (fd - r.derivatives[j]).abs() < 3e-4,
                "family={family}, j={j}, {fd} != {}",
                r.derivatives[j]
            );
        }
    }
}
