//! All three rough/dividend correlation partials: full-recompile finite
//! differences, elementary call-hinge diagnostic, exact prefixes and domains.
use pricing::mc::{
    BrownianBridgePlan, EngineConfig, ExecutionPolicy, LocalVolTimeGrid, Philox4x32,
    RandomCoordinate, RandomDomain, RqmcPlan, inverse_standard_normal,
};
use pricing::models::RoughBergomi;
use pricing::risk::{GammaConfig, SpotBump};
use pricing::stochastic_dividends::{
    BuehlerDividendModel, StochasticDividendPathPlan, StochasticDividendPricingPlan as Plan,
};
use pricing::{JsonLimits, parse_request_json};
use pricing_numerics::NeumaierSum;
use serde_json::{Value, json};

const RHO: [f64; 3] = [-0.25, -0.4, 0.15];
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

fn compile(v: &Value, rho: [f64; 3], h: f64, eta: f64, workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    Plan::compile_rough_bergomi(
        &r,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, rho[0]).unwrap(),
        RoughBergomi::new(h, eta, rho[1]).unwrap(),
        rho[2],
        0.125,
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}
fn terminal_intrinsics(v: &Value, rho: [f64; 3], h: f64, eta: f64) -> Vec<f64> {
    assert_eq!(v["product"]["type"], "european_vanilla");
    assert_eq!(v["product"]["side"]["type"], "call");
    assert_eq!(v["product"]["expiry"], "2027-09-04");
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let plan = compile(v, rho, h, eta, 1);
    let grid = LocalVolTimeGrid::compile(plan.time_nodes().to_vec(), 1.0).unwrap();
    assert_eq!(grid.nodes(), plan.time_nodes());
    let market = r.market().equity().forward();
    let path = StochasticDividendPathPlan::compile_rough_bergomi(
        market,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, rho[0]).unwrap(),
        v["model"]["volatility"].as_f64().unwrap(),
        RoughBergomi::new(h, eta, rho[1]).unwrap(),
        rho[2],
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

fn check(v: &Value, rho: [f64; 3], h: f64, eta: f64) {
    let p = compile(v, rho, h, eta, 1);
    let prefix = p.evaluate_rough_aad().unwrap();
    let risk = p.evaluate_correlation_aad().unwrap();
    let n = prefix.derivatives.len();
    assert_eq!(risk.price, p.evaluate().unwrap());
    assert_eq!(risk.price, prefix.price);
    assert_eq!(&risk.parameter_labels[..n], &*prefix.parameter_labels);
    assert_eq!(&risk.derivatives[..n], &*prefix.derivatives);
    assert_eq!(&risk.standard_errors[..n], &*prefix.standard_errors);
    assert_eq!(risk.cash_mean_adjoints(), prefix.cash_mean_adjoints());
    assert_eq!(risk.discount_node_dv01(), prefix.discount_node_dv01());
    assert_eq!(risk.repo_spread_node_dv01(), prefix.repo_spread_node_dv01());
    assert_eq!(risk.method, "buehler-joint-correlation-reverse-v1");
    assert_eq!(
        &risk.parameter_labels[n..],
        &[
            "equity_dividend_correlation",
            "spot_volatility_correlation[0]",
            "dividend_volatility_correlation[0]"
        ]
    );
    assert!(
        risk.standard_errors
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    );
    let baseline =
        (v["product"]["type"] == "european_vanilla").then(|| terminal_intrinsics(v, rho, h, eta));
    if let Some(x) = &baseline {
        assert!((mean_positive(x) - risk.price.value).abs() < 2e-12);
    }
    for (j, &aad) in risk.derivatives[n..].iter().enumerate() {
        for eps in [1e-5, 1e-6] {
            let mut up = rho;
            let mut down = rho;
            up[j] += eps;
            down[j] -= eps;
            let pp = compile(v, up, h, eta, 1).evaluate().unwrap().value;
            let pm = compile(v, down, h, eta, 1).evaluate().unwrap().value;
            let raw = (pp - pm) / (2.0 * eps);
            // Predeclared finite-stencil distinction for the unsmoothed call:
            // reconstruct its hinge contribution independently of all adjoints.
            let remainder = if let Some(x) = &baseline {
                let xp = terminal_intrinsics(v, up, h, eta);
                let xm = terminal_intrinsics(v, down, h, eta);
                assert!((mean_positive(&xp) - pp).abs() < 2e-12);
                assert!((mean_positive(&xm) - pm).abs() < 2e-12);
                let (rem, crossings) = kink_remainder(x, &xp, &xm, eps);
                if crossings > 0 {
                    eprintln!("H={h} p={j} eps={eps} raw={raw} hinge={rem} crossings={crossings}");
                }
                rem
            } else {
                0.0
            };
            let fd = raw - remainder;
            assert!(
                (aad - fd).abs() <= 3e-5 + 2e-5 * aad.abs().max(fd.abs()),
                "H={h} eta={eta} rho={rho:?} j={j} eps={eps} AAD={aad} FD={fd} raw={raw} hinge={remainder}"
            );
        }
    }
    let replay = compile(v, rho, h, eta, 3)
        .evaluate_correlation_aad()
        .unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
    assert_eq!(risk.price.value, replay.price.value);
    assert_eq!(risk.price.standard_error, replay.price.standard_error);
    assert_ne!(risk.price.plan_fingerprint, replay.price.plan_fingerprint);
    assert_eq!(prefix, p.evaluate_rough_aad().unwrap());
}
#[test]
fn all_entries_full_recompile_and_rough_prefix_for_mc_and_rqmc() {
    for qmc in [false, true] {
        for seed in [791, 433] {
            for h in [0.1, 0.49, 0.5] {
                check(&payload(qmc, seed), RHO, h, 0.6);
            }
        }
    }
}
#[test]
fn irregular_asian_and_smoothed_cash_barrier_include_correlation_history() {
    let mut v = payload(true, 344);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":90.0,"notional":1.0,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
        {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    for h in [0.1, 0.5] {
        check(&v, RHO, h, 0.6);
    }
    let mut v = payload(false, 734);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},"monitoring":{"type":"discrete"},
        "monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
    let p = compile(&v, RHO, 0.1, 0.6, 1);
    let price = p.evaluate().unwrap();
    assert!(p.evaluate_correlation_aad().is_err());
    assert_eq!(price, p.evaluate().unwrap());
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.0});
    for h in [0.1, 0.5] {
        check(&v, RHO, h, 0.6);
    }
}
#[test]
fn zero_correlations_eta_and_sigma_limits_preserve_direct_dividend_risk() {
    let v = payload(true, 882);
    for h in [0.1, 0.5] {
        check(&v, [0.0; 3], h, 0.6);
        check(&v, RHO, h, 0.0);
        let r = compile(&v, RHO, h, 0.0, 1)
            .evaluate_correlation_aad()
            .unwrap();
        let n = r.derivatives.len();
        assert_eq!(&r.derivatives[n - 2..], &[0.0, 0.0]);
        assert_eq!(&r.standard_errors[n - 2..], &[0.0, 0.0]);
    }
    let mut v = v;
    v["model"]["volatility"] = json!(0.0);
    v["product"]["strike"] = json!(60.0);
    check(&v, RHO, 0.1, 0.6);
    let r = compile(&v, RHO, 0.1, 0.6, 1)
        .evaluate_correlation_aad()
        .unwrap();
    let n = r.derivatives.len();
    assert_eq!(&r.derivatives[n - 2..], &[0.0, 0.0]);
    assert_eq!(&r.standard_errors[n - 2..], &[0.0, 0.0]);
}
#[test]
fn singular_domain_rejects_only_correlation_scope_even_at_zero_loading() {
    let v = payload(false, 221);
    for rho in [[1.0, 1.0, 1.0], [0.2, 1.0, 0.2], [0.0, 1.0 - 1e-12, 0.0]] {
        for h in [0.1, 0.5] {
            for eta in [0.0, 0.6] {
                let p = compile(&v, rho, h, eta, 1);
                let before = p.evaluate_rough_aad().unwrap();
                let gamma = p
                    .evaluate_gamma(GammaConfig::new(SpotBump::absolute(1.0).unwrap()))
                    .unwrap();
                let error = p.evaluate_correlation_aad().unwrap_err().to_string();
                assert!(
                    error.contains("correlation") && error.contains("1e-10"),
                    "{error}"
                );
                assert_eq!(before, p.evaluate_rough_aad().unwrap());
                assert_eq!(before.price, p.evaluate().unwrap());
                assert_eq!(
                    gamma,
                    p.evaluate_gamma(GammaConfig::new(SpotBump::absolute(1.0).unwrap()))
                        .unwrap()
                );
            }
        }
    }
}
