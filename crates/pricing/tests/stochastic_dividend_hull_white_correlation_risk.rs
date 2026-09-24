//! Full-recompile correlation bumps use the original fixed random coordinates.
//! They verify the finite algorithm, not continuous-time risk convergence.
use pricing::mc::ExecutionPolicy;
use pricing::models::HullWhite1Factor;
use pricing::stochastic_dividends::{
    BuehlerDividendModel, StochasticDividendHullWhitePricingPlan as Plan,
};
use pricing::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

fn payload(qmc: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["schema_version"] = json!(2);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":4.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":8.0}}]);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,
        "strike":20.,"notional":1.,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
        {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,
            "scramble_count":4,"master_scramble_seed":1859,
            "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":256,"master_seed":3407,
            "variance_reduction":{"antithetic":true,"brownian_bridge":false}})
    };
    v
}

fn compile(v: &Value, a: f64, d: [f64; 3], rho: [f64; 3], workers: u32) -> Plan {
    let request = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    Plan::compile_bs(
        &request,
        BuehlerDividendModel::new(d[0], d[1], d[2], rho[0]).unwrap(),
        HullWhite1Factor::new(a, vec![0.0, 0.8, 1.15], vec![0.04, 0.06, 0.09]).unwrap(),
        rho[1],
        rho[2],
        0.125,
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}

fn check(v: &Value, a: f64, d: [f64; 3], rho: [f64; 3]) {
    let plan = compile(v, a, d, rho, 1);
    let risk = plan.evaluate_correlation_aad().unwrap();
    let rate_risk = plan.evaluate_hull_white_aad().unwrap();
    let width = rate_risk.derivatives.len();
    assert_eq!(risk.price, plan.evaluate().unwrap());
    assert_eq!(risk.price, rate_risk.price);
    assert_eq!(
        &risk.parameter_labels[..width],
        &*rate_risk.parameter_labels
    );
    assert_eq!(&risk.derivatives[..width], &*rate_risk.derivatives);
    assert_eq!(&risk.standard_errors[..width], &*rate_risk.standard_errors);
    assert_eq!(risk.cash_times, rate_risk.cash_times);
    assert_eq!(risk.discount_times, rate_risk.discount_times);
    assert_eq!(risk.repo_spread_times, rate_risk.repo_spread_times);
    assert_eq!(risk.parameter_labels.len(), width + 3);
    assert_eq!(
        risk.parameter_labels[width..]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "equity_dividend_correlation",
            "equity_rate_correlation",
            "dividend_rate_correlation"
        ]
    );
    assert_eq!(
        risk.method,
        "buehler-bs-hw-cash-payoff-forward-correlation-adjoint-v1"
    );
    assert!(
        risk.standard_errors
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    );
    for p in 0..3 {
        let derivative = risk.derivatives[width + p];
        for h in [1e-5, 1e-6] {
            let mut up = rho;
            let mut down = rho;
            up[p] += h;
            down[p] -= h;
            let finite_difference = (compile(v, a, d, up, 1).evaluate().unwrap().value
                - compile(v, a, d, down, 1).evaluate().unwrap().value)
                / (2.0 * h);
            let budget = 2e-5 + 2e-5 * derivative.abs().max(finite_difference.abs());
            assert!(
                (derivative - finite_difference).abs() <= budget,
                "rho={rho:?}, p={p}, h={h}, AAD={derivative}, FD={finite_difference}, budget={budget}"
            );
        }
    }
    let replay = compile(v, a, d, rho, 3).evaluate_correlation_aad().unwrap();
    assert_eq!(risk.price.value, replay.price.value);
    assert_eq!(risk.price.standard_error, replay.price.standard_error);
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
}

#[test]
fn hw_correlation_aad_recompiles_delayed_asian_with_future_cash_and_rate_knots() {
    for qmc in [false, true] {
        check(&payload(qmc), 0.4, [0.7, 0.6, 0.35], [-0.25, 0.25, -0.2]);
    }
}

#[test]
fn hw_correlation_aad_recompiles_smoothed_barrier_with_pre_and_post_cash_seeds() {
    let mut v = payload(true);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,
        "expiry":"2027-09-04","strike":20.,"barrier":100.,"notional":1.,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},
        "monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],
        "payment_date":"2027-12-04"});
    let unsmoothed = compile(&v, 0.4, [0.7, 0.6, 0.35], [-0.25, 0.25, -0.2], 1);
    assert!(unsmoothed.evaluate().is_ok());
    assert!(unsmoothed.evaluate_correlation_aad().is_err());
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.});
    check(&v, 0.4, [0.7, 0.6, 0.35], [-0.25, 0.25, -0.2]);
}

#[test]
fn hw_correlation_aad_supports_zero_correlations_ho_lee_and_zero_equity_dividend_diffusion() {
    let mut v = payload(false);
    check(&v, 0.0, [0.7, 0.6, 0.35], [0.0; 3]);
    v["model"]["volatility"] = json!(0.0);
    v["market"]["discrete_dividends"][2]["quote"]["amount"] = json!(0.0);
    check(&v, 0.4, [0.0, 0.0, 0.0], [-0.25, 0.25, -0.2]);
}

#[test]
fn hw_correlation_aad_rejects_instantaneous_and_simulated_covariance_boundaries() {
    let v = payload(false);
    // W_rate = W_equity at the driver level. Pricing and fixed-correlation
    // risk remain available on this boundary.
    let request =
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let plan = Plan::compile_bs(
        &request,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, 0.0).unwrap(),
        HullWhite1Factor::new(0.4, vec![0.0, 0.1, 0.3, 0.7], vec![0.02, 0.06, 0.03, 0.08]).unwrap(),
        1.0,
        0.0,
        1.0,
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert!(plan.evaluate().is_ok());
    assert!(plan.evaluate_aad().is_ok());
    assert!(plan.evaluate_correlation_aad().is_err());

    // Full rank must also hold on each simulated interval; deterministic
    // rates still permit pricing/basic risk at an interior driver correlation.
    let zero_rates = Plan::compile_bs(
        &request,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
        HullWhite1Factor::new(0.4, vec![0.0], vec![0.0]).unwrap(),
        0.25,
        -0.2,
        0.125,
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert!(zero_rates.evaluate().is_ok());
    assert!(zero_rates.evaluate_aad().is_ok());
    assert!(zero_rates.evaluate_correlation_aad().is_err());
}
