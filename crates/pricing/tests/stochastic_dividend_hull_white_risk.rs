//! Full-recompile finite differences use identical random streams, never an AAD
//! reference path. These are finite-algorithm derivative tests, not price accuracy.
use pricing::market::DiscountCurve;
use pricing::mc::{
    BrownianBridgePlan, EngineConfig, ExecutionPolicy, LocalVolTimeGrid, Philox4x32,
    RandomCoordinate, RandomDomain, RqmcPlan, inverse_standard_normal,
};
use pricing::models::HullWhite1Factor;
use pricing::stochastic_dividends::StochasticDividendHullWhitePathPlan;
use pricing::stochastic_dividends::{
    BuehlerDividendModel, StochasticDividendHullWhitePricingPlan as Plan,
};
use pricing::{JsonLimits, parse_request_json};
use pricing_numerics::NeumaierSum;
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
fn rates(family: usize) -> HullWhite1Factor {
    match family {
        0 => HullWhite1Factor::new(0.4, vec![0.0, 0.8, 1.15], vec![0.04, 0.06, 0.09]),
        1 => HullWhite1Factor::new(0.0, vec![0.0, 0.8, 1.15], vec![0.04, 0.06, 0.09]),
        _ => HullWhite1Factor::new(0.4, vec![0.0], vec![0.0]),
    }
    .unwrap()
}
fn compile(v: &Value, family: usize, d: [f64; 3], workers: u32) -> Plan {
    compile_with_rates(v, d, rates(family), workers)
}
fn compile_with_rates(v: &Value, d: [f64; 3], rates: HullWhite1Factor, workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    Plan::compile_bs(
        &r,
        BuehlerDividendModel::new(d[0], d[1], d[2], -0.25).unwrap(),
        rates,
        0.25,
        -0.2,
        0.125,
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}
fn bumped_rate_parameter(
    v: &Value,
    d: [f64; 3],
    mean_reversion: f64,
    volatility_times: &[f64],
    volatilities: &[f64],
    parameter: usize,
    bump: f64,
) -> f64 {
    let mut a = mean_reversion;
    let mut vols = volatilities.to_vec();
    if parameter == 0 {
        a += bump;
    } else {
        vols[parameter - 1] += bump;
    }
    let model = HullWhite1Factor::new(a, volatility_times.to_vec(), vols).unwrap();
    compile_with_rates(v, d, model, 1).evaluate().unwrap().value
}
fn shifted_inputs(v: &Value, d: [f64; 3], label: &str, h: f64) -> (Value, [f64; 3]) {
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
    (v, d)
}
fn bumped(v: &Value, family: usize, d: [f64; 3], label: &str, h: f64) -> f64 {
    let (v, d) = shifted_inputs(v, d, label, h);
    compile(&v, family, d, 1).evaluate().unwrap().value
}

// Test-only independent payoff stencil. This reconstructs terminal *primal*
// stock with public paths and original random coordinates, never AAD values.
// The unsmoothed sample mean need not be differentiable across a finite bump.
fn terminal_intrinsics(v: &Value, family: usize, d: [f64; 3]) -> Vec<f64> {
    assert_eq!(v["product"]["type"], "european_vanilla");
    assert_eq!(v["product"]["side"]["type"], "call");
    assert_eq!(v["product"]["expiry"], "2027-09-04");
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let plan = compile(v, family, d, 1);
    let grid = LocalVolTimeGrid::compile(plan.time_nodes().to_vec(), 1.0).unwrap();
    assert_eq!(grid.nodes(), plan.time_nodes());
    let market = r.market().equity().forward();
    let path = StochasticDividendHullWhitePathPlan::compile_bs(
        market,
        BuehlerDividendModel::new(d[0], d[1], d[2], -0.25).unwrap(),
        v["model"]["volatility"].as_f64().unwrap(),
        &rates(family),
        0.25,
        -0.2,
        &grid,
    )
    .unwrap();
    let vr = match r.engine() {
        EngineConfig::PseudoMonteCarlo(c) => c.variance_reduction(),
        EngineConfig::RandomizedQuasiMonteCarlo(c) => c.variance_reduction(),
    };
    let bridge = vr
        .brownian_bridge()
        .then(|| BrownianBridgePlan::compile(grid.nodes().to_vec(), 1).unwrap());
    let dimension = path.random_dimension();
    let strike = v["product"]["strike"].as_f64().unwrap();
    let scale =
        market.discount_curve().discount(1.0).unwrap() * v["product"]["notional"].as_f64().unwrap();
    let mut values = Vec::new();
    let mut append = |mut z: Vec<f64>| {
        if let Some(b) = &bridge {
            for factor in 0..4 {
                let input = z
                    .iter()
                    .skip(factor)
                    .step_by(4)
                    .copied()
                    .collect::<Vec<_>>();
                let output = b.apply_one_factor(&input).unwrap();
                for (i, value) in output.into_iter().enumerate() {
                    z[4 * i + factor] = value;
                }
            }
        }
        for &sign in if vr.antithetic() {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|x| sign * x).collect::<Vec<_>>();
            let states = path.evolve_path(&shocks).unwrap();
            let terminal = *states.last().unwrap();
            let stock = path.spots(states.len() - 1, terminal).unwrap().0;
            let relative = rates(family)
                .relative_discount(1.0, terminal.integrated_rate_factor())
                .unwrap();
            values.push(scale * relative * (stock - strike));
        }
    };
    match r.engine() {
        EngineConfig::PseudoMonteCarlo(c) => {
            let rng = Philox4x32::from_seed(c.master_seed());
            for i in 0..c.independent_sampling_units().get() {
                append(
                    (0..dimension)
                        .map(|j| {
                            rng.standard_normal(RandomCoordinate::new(
                                i,
                                j,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect(),
                );
            }
        }
        EngineConfig::RandomizedQuasiMonteCarlo(c) => {
            let q = RqmcPlan::compile(c, dimension).unwrap();
            for k in 0..c.scramble_count().get() {
                for i in 0..c.points_per_scramble().get() {
                    append(
                        (0..dimension)
                            .map(|j| inverse_standard_normal(q.uniform(k, i, j).unwrap()).unwrap())
                            .collect(),
                    );
                }
            }
        }
    }
    values
}
fn mean_positive(values: &[f64]) -> f64 {
    let mut sum = NeumaierSum::new();
    for &x in values {
        sum.add(x.max(0.0));
    }
    sum.total() / values.len() as f64
}
fn kink_remainder(base: &[f64], plus: &[f64], minus: &[f64], h: f64) -> (f64, usize) {
    assert_eq!(base.len(), plus.len());
    assert_eq!(base.len(), minus.len());
    let mut sum = NeumaierSum::new();
    let mut crossings = 0;
    for ((&x0, &xp), &xm) in base.iter().zip(plus).zip(minus) {
        let itm = x0 > 0.0;
        if (xp > 0.0) != itm || (xm > 0.0) != itm {
            crossings += 1;
            let slope = if itm { 1.0 } else { 0.0 };
            // Exact hinge remainder relative to the unshifted payoff branch.
            // It is not inferred from AAD or from the observed test error.
            sum.add((xp.max(0.0) - slope * xp - xm.max(0.0) + slope * xm) / (2.0 * h));
        }
    }
    (sum.total() / base.len() as f64, crossings)
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
    let baseline_intrinsics =
        (v["product"]["type"] == "european_vanilla").then(|| terminal_intrinsics(v, family, d));
    if let Some(x0) = &baseline_intrinsics {
        assert!((mean_positive(x0) - risk.price.value).abs() < 2e-12);
    }
    for (label, &aad) in risk.parameter_labels.iter().zip(&risk.derivatives) {
        if label.ends_with("log_df[0]") {
            assert_eq!(aad, 0.0);
            continue;
        }
        for h in [1e-5, 1e-6] {
            let up = bumped(v, family, d, label, h);
            let down = bumped(v, family, d, label, -h);
            let raw_fd = (up - down) / (2.0 * h);
            let correction = if let Some(x0) = &baseline_intrinsics {
                let (vp, dp) = shifted_inputs(v, d, label, h);
                let (vm, dm) = shifted_inputs(v, d, label, -h);
                let xp = terminal_intrinsics(&vp, family, dp);
                let xm = terminal_intrinsics(&vm, family, dm);
                assert!((mean_positive(&xp) - up).abs() < 2e-12);
                assert!((mean_positive(&xm) - down).abs() < 2e-12);
                let (remainder, crossings) = kink_remainder(x0, &xp, &xm, h);
                if crossings > 0 {
                    eprintln!(
                        "HW call-stencil {label} h={h}: rawFD={raw_fd} remainder={remainder} crossings={crossings}"
                    );
                }
                remainder
            } else {
                0.0
            };
            let fd = raw_fd - correction;
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
fn hw_all_basic_inputs_include_conditional_funding_and_worker_replay() {
    for family in 0..3 {
        for qmc in [false, true] {
            for seed in [1973, 1051] {
                check(&payload(qmc, seed), family, [0.7, 0.6, 0.35]);
            }
        }
    }
}

#[test]
fn hw_rate_parameter_adjoint_includes_claims_funding_and_delayed_payment() {
    let mut v = payload(true, 2207);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":20.,"notional":1.,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
        {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    let d = [0.7, 0.6, 0.35];
    let mean_reversion = 0.4;
    let volatility_times = [0.0, 0.8, 1.15];
    let volatilities = [0.04, 0.06, 0.09];
    let make_rates = |a, vols: &[f64]| {
        HullWhite1Factor::new(a, volatility_times.to_vec(), vols.to_vec()).unwrap()
    };
    let plan = compile_with_rates(&v, d, make_rates(mean_reversion, &volatilities), 1);
    let risk = plan.evaluate_hull_white_aad().unwrap();
    assert_eq!(risk.price, plan.evaluate().unwrap());
    assert_eq!(
        risk.parameter_labels[risk.parameter_labels.len() - 4..]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "rate_mean_reversion",
            "rate_volatility[0]",
            "rate_volatility[1]",
            "rate_volatility[2]"
        ]
    );
    assert_eq!(
        risk.method,
        "buehler-bs-hw-cash-payoff-forward-rate-parameter-adjoint-v1"
    );
    for parameter in 0..4 {
        let h = if parameter == 0 { 1e-5 } else { 1e-6 };
        let up = bumped_rate_parameter(
            &v,
            d,
            mean_reversion,
            &volatility_times,
            &volatilities,
            parameter,
            h,
        );
        let down = bumped_rate_parameter(
            &v,
            d,
            mean_reversion,
            &volatility_times,
            &volatilities,
            parameter,
            -h,
        );
        let finite_difference = (up - down) / (2.0 * h);
        let derivative = risk.derivatives[risk.derivatives.len() - 4 + parameter];
        let budget = 2e-5 + 2e-5 * derivative.abs().max(finite_difference.abs());
        assert!(
            (derivative - finite_difference).abs() <= budget,
            "parameter={parameter}, AAD={derivative}, FD={finite_difference}, budget={budget}"
        );
    }
    let replay = compile_with_rates(&v, d, make_rates(mean_reversion, &volatilities), 3)
        .evaluate_hull_white_aad()
        .unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);

    let singular = compile(&v, 2, d, 1);
    assert!(singular.evaluate_aad().is_ok());
    assert!(singular.evaluate_hull_white_aad().is_err());
}
#[test]
fn hw_asian_delayed_payment_reverses_curves_and_cash() {
    let mut v = payload(true, 344);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":20.,"notional":1.,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
        {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    for family in 0..3 {
        check(&v, family, [0.7, 0.6, 0.35]);
    }
}
#[test]
fn hw_pre_post_cash_barrier_seeds_and_explicit_discontinuity_rejection() {
    let mut v = payload(false, 734);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04","strike":20.,"barrier":100.,"notional":1.,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},"monitoring":{"type":"discrete"},
        "monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
    let plan = compile(&v, 0, [0.7, 0.6, 0.35], 1);
    let price = plan.evaluate().unwrap();
    assert!(plan.evaluate_aad().is_err());
    assert_eq!(plan.evaluate().unwrap(), price);
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.});
    for family in 0..3 {
        check(&v, family, [0.7, 0.6, 0.35]);
    }
}
#[test]
fn hw_zero_parameters_zero_cash_and_alpha_endpoints_use_inward_derivatives() {
    let mut v = payload(true, 992);
    v["product"]["strike"] = json!(20.);
    for sigma in [0., 0.2] {
        for alpha in [0., 1.] {
            v["model"]["volatility"] = json!(sigma);
            let d = [0., alpha, 0.];
            let plan = compile(&v, 0, d, 1);
            let r = plan.evaluate_aad().unwrap();
            for j in 1..5 {
                let h = if j == 3 && alpha == 1. { -1e-7 } else { 1e-7 };
                let fd = (bumped(&v, 0, d, &r.parameter_labels[j], h) - r.price.value) / h;
                assert!(
                    (fd - r.derivatives[j]).abs() < 3e-4,
                    "j={j}, {fd} != {}",
                    r.derivatives[j]
                );
            }
        }
    }
    v["market"]["discrete_dividends"][2]["quote"]["amount"] = json!(0.);
    let d = [0.7, 0.6, 0.35];
    let r = compile(&v, 0, d, 1).evaluate_aad().unwrap();
    let fd = (bumped(&v, 0, d, "cash_mean[3]", 1e-6) - r.price.value) / 1e-6;
    assert!((fd - r.cash_mean_adjoints()[2]).abs() < 3e-5);
    // Fixed singular Brownian law is valid: no correlation/rate factor derivative.
    let req = parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let p = Plan::compile_bs(
        &req,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, 1.).unwrap(),
        rates(0),
        0.2,
        0.2,
        0.125,
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert_eq!(p.evaluate_aad().unwrap().price, p.evaluate().unwrap());
}
#[test]
fn hw_zero_rate_one_step_risk_reduces_to_existing_fixed_rate_reverse() {
    let mut v = payload(false, 428);
    v["market"]["discrete_dividends"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    let r = parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let model = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
    let policy = ExecutionPolicy::new(1, Some(64)).unwrap();
    let a = Plan::compile_bs(&r, model, rates(2), 0.25, -0.2, 1., policy)
        .unwrap()
        .evaluate_aad()
        .unwrap();
    let b = pricing::stochastic_dividends::StochasticDividendPricingPlan::compile_bs(
        &r, model, 1., policy,
    )
    .unwrap()
    .evaluate_aad()
    .unwrap();
    assert_eq!(a.parameter_labels, b.parameter_labels);
    assert!((a.price.value - b.price.value).abs() < 2e-12);
    for (&a, &b) in a.derivatives.iter().zip(&b.derivatives) {
        assert!((a - b).abs() < 2e-11, "{a} != {b}");
    }
}
