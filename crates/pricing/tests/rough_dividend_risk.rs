//! Full-recompile finite differences use identical random streams, never an AAD
//! reference path. These are finite-algorithm derivative tests, not price accuracy.
use pricing::market::DiscountCurve;
use pricing::mc::{
    BrownianBridgePlan, EngineConfig, ExecutionPolicy, LocalVolTimeGrid, Philox4x32,
    RandomCoordinate, RandomDomain, RqmcPlan, inverse_standard_normal,
};
use pricing::models::RoughBergomi;
use pricing::risk::{GammaConfig, SpotBump};
use pricing::stochastic_dividends::StochasticDividendPathPlan;
use pricing::stochastic_dividends::{BuehlerDividendModel, StochasticDividendPricingPlan as Plan};
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
fn compile(v: &Value, family: usize, d: [f64; 3], workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let d = BuehlerDividendModel::new(d[0], d[1], d[2], -0.25).unwrap();
    let p = ExecutionPolicy::new(workers, Some(64)).unwrap();
    Plan::compile_rough_bergomi(
        &r,
        d,
        RoughBergomi::new([0.1, 0.49, 0.5][family], 0.6, -0.4).unwrap(),
        0.15,
        0.125,
        p,
    )
    .unwrap()
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
    let path = StochasticDividendPathPlan::compile_rough_bergomi(
        market,
        BuehlerDividendModel::new(d[0], d[1], d[2], -0.25).unwrap(),
        v["model"]["volatility"].as_f64().unwrap(),
        RoughBergomi::new([0.1, 0.49, 0.5][family], 0.6, -0.4).unwrap(),
        0.15,
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
            let stock = path
                .nodes()
                .last()
                .unwrap()
                .spots(*states.last().unwrap())
                .unwrap()
                .0;
            values.push(scale * (stock - strike));
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
                kink_remainder(x0, &xp, &xm, h).0
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
fn rough_basic_aad_matches_full_recompile_all_inputs_and_worker_replay() {
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
fn rough_zero_sigma_and_dividend_boundaries_use_inward_differences() {
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

fn rough_plan(v: &Value, h: f64, eta: f64, workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    Plan::compile_rough_bergomi(
        &r,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
        RoughBergomi::new(h, eta, -0.4).unwrap(),
        0.15,
        0.125,
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}
#[test]
fn rough_parameters_reverse_full_history_and_preserve_basic_prefix() {
    for qmc in [false, true] {
        for seed in [791, 433] {
            for h in [0.1, 0.49] {
                let v = payload(qmc, seed);
                let p = rough_plan(&v, h, 0.6, 1);
                let basic = p.evaluate_aad().unwrap();
                let risk = p.evaluate_rough_aad().unwrap();
                let n = basic.derivatives.len();
                assert_eq!(risk.price, basic.price);
                assert_eq!(&risk.parameter_labels[..n], &*basic.parameter_labels);
                assert_eq!(&risk.derivatives[..n], &*basic.derivatives);
                assert_eq!(&risk.standard_errors[..n], &*basic.standard_errors);
                assert_eq!(
                    &risk.parameter_labels[n..],
                    &["rough_hurst", "rough_vol_of_vol"]
                );
                for (j, &aad) in risk.derivatives[n..].iter().enumerate() {
                    for eps in [1e-5, 1e-6] {
                        let (hp, hm, ep, em) = if j == 0 {
                            (h + eps, h - eps, 0.6, 0.6)
                        } else {
                            (h, h, 0.6 + eps, 0.6 - eps)
                        };
                        let fd = (rough_plan(&v, hp, ep, 1).evaluate().unwrap().value
                            - rough_plan(&v, hm, em, 1).evaluate().unwrap().value)
                            / (2.0 * eps);
                        assert!(
                            (aad - fd).abs() <= 3e-5 + 2e-5 * aad.abs().max(fd.abs()),
                            "H={h}, param={j}, eps={eps}, AAD={aad}, FD={fd}"
                        );
                    }
                }
                let replay = rough_plan(&v, h, 0.6, 3).evaluate_rough_aad().unwrap();
                assert_eq!(risk.derivatives, replay.derivatives);
                assert_eq!(risk.standard_errors, replay.standard_errors);
            }
        }
    }
}
#[test]
fn rough_endpoint_parameter_derivatives_and_unsupported_scopes_are_explicit() {
    let mut v = payload(true, 882);
    v["product"]["strike"] = json!(60.0);
    for (h, eta) in [(0.5, 0.6), (0.1, 0.0), (0.5, 0.0)] {
        let p = rough_plan(&v, h, eta, 1);
        let risk = p.evaluate_rough_aad().unwrap();
        let n = risk.derivatives.len() - 2;
        for j in 0..2 {
            let eps = if j == 0 && h == 0.5 { -1e-7 } else { 1e-7 };
            let shifted = rough_plan(
                &v,
                if j == 0 { h + eps } else { h },
                if j == 1 { eta + eps } else { eta },
                1,
            );
            let fd = (shifted.evaluate().unwrap().value - risk.price.value) / eps;
            assert!(
                (fd - risk.derivatives[n + j]).abs() < 3e-4,
                "H={h}, eta={eta}, param={j}, FD={fd}, AAD={}",
                risk.derivatives[n + j]
            );
        }
        if eta == 0.0 {
            assert_eq!(risk.derivatives[n], 0.0);
            assert_eq!(risk.standard_errors[n], 0.0);
        }
        assert!(p.evaluate_bergomi_aad().is_err());
        assert!(p.evaluate_correlation_aad().is_err());
        assert_eq!(risk.price, p.evaluate().unwrap());
    }
    v["model"]["volatility"] = json!(0.0);
    let r = rough_plan(&v, 0.1, 0.6, 1).evaluate_rough_aad().unwrap();
    assert!(
        r.derivatives[r.derivatives.len() - 2..]
            .iter()
            .all(|x| *x == 0.0)
    );
    assert!(
        r.standard_errors[r.standard_errors.len() - 2..]
            .iter()
            .all(|x| *x == 0.0)
    );
    let req = parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let bs = Plan::compile_bs(
        &req,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
        0.125,
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert!(
        bs.evaluate_rough_aad()
            .unwrap_err()
            .to_string()
            .contains("requires a rough")
    );
}
#[test]
fn rough_gamma_matches_recompiled_delta_ladder_with_paired_worker_replay() {
    for h in [0.1, 0.5] {
        for qmc in [false, true] {
            let v = payload(qmc, 449);
            let p = rough_plan(&v, h, 0.6, 1);
            let config = GammaConfig::new(SpotBump::absolute(1.0).unwrap());
            let g = p.evaluate_gamma(config).unwrap();
            let basic = p.evaluate_aad().unwrap();
            assert_eq!(g.price, basic.price);
            assert_eq!(g.delta, basic.delta());
            for (&bump, &gamma) in g.spot_bumps.iter().zip(&g.gamma_estimates) {
                let mut plus = v.clone();
                plus["market"]["spot"] = json!(100.0 + bump);
                let mut minus = v.clone();
                minus["market"]["spot"] = json!(100.0 - bump);
                let fd = (rough_plan(&plus, h, 0.6, 1).evaluate_aad().unwrap().delta()
                    - rough_plan(&minus, h, 0.6, 1)
                        .evaluate_aad()
                        .unwrap()
                        .delta())
                    / (2.0 * bump);
                assert!((gamma - fd).abs() < 2e-12);
            }
            let replay = rough_plan(&v, h, 0.6, 3).evaluate_gamma(config).unwrap();
            assert_eq!(g.gamma_estimates, replay.gamma_estimates);
            assert_eq!(g.gamma_standard_errors, replay.gamma_standard_errors);
            assert_eq!(g.bump_differences, replay.bump_differences);
            assert_eq!(
                g.bump_difference_standard_errors,
                replay.bump_difference_standard_errors
            );
            assert!(
                g.gamma_standard_errors
                    .iter()
                    .all(|x| x.is_finite() && *x >= 0.0)
            );
        }
    }
}

#[test]
fn finite_call_stencil_crossing_is_measured_without_changing_seed_or_bump() {
    let v = payload(false, 912);
    let d = [0.7, 0.6, 0.35];
    let label = "discount_log_df[2]";
    let r = compile(&v, 0, d, 1).evaluate_aad().unwrap();
    let j = r.parameter_labels.iter().position(|x| x == label).unwrap();
    let x0 = terminal_intrinsics(&v, 0, d);
    for (h, expected_crossings) in [(1e-5, 1), (1e-6, 0)] {
        let (vp, dp) = shifted_inputs(&v, d, label, h);
        let (vm, dm) = shifted_inputs(&v, d, label, -h);
        let xp = terminal_intrinsics(&vp, 0, dp);
        let xm = terminal_intrinsics(&vm, 0, dm);
        let (correction, crossings) = kink_remainder(&x0, &xp, &xm, h);
        assert_eq!(crossings, expected_crossings);
        let raw_fd = (bumped(&v, 0, d, label, h) - bumped(&v, 0, d, label, -h)) / (2.0 * h);
        assert!((raw_fd - correction - r.derivatives[j]).abs() < 3e-5);
        if expected_crossings == 0 {
            assert_eq!(correction, 0.0);
        } else {
            assert!(correction.abs() > 0.02);
        }
        eprintln!(
            "call stencil seed=912 h={h}: raw_FD={raw_fd}, AAD={}, hinge_remainder={correction}, crossings={crossings}",
            r.derivatives[j]
        );
    }
}
