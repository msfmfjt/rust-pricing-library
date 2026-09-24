//! Independent reserve/PDE tests live with the compiled kernel. Public tests
//! cover projection, Gaussian prices, funding, settlement and reproducibility.
use pricing::market::DiscountCurve;
use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use pricing::models::HullWhite1Factor;
use pricing::stochastic_dividends::{
    BuehlerDividendModel as Model, StochasticDividendHullWhitePathPlan as Path,
    StochasticDividendHullWhitePricingPlan as Plan, StochasticDividendHullWhiteState as State,
    StochasticDividendPathPlan as DeterministicPath,
};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};
fn request(qmc: bool, cash: &[(f64, f64)]) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["schema_version"] = json!(2);
    v["market"]["discrete_dividends"]=json!(cash.iter().enumerate().map(|(i,(t,d))|json!({"event_id":i+1,"ex_time":t,"quote":{"type":"fixed_cash","amount":d}})).collect::<Vec<_>>());
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":4096,"scramble_count":8,"master_scramble_seed":973,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn model(k: f64) -> Model {
    Model::new(k, 0.6, 0.35, -0.25).unwrap()
}
fn rates(sigma: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.4, vec![0.0], vec![sigma]).unwrap()
}
fn policy(n: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(n, Some(64)).unwrap()
}
fn close(a: f64, b: f64, e: f64) {
    assert!((a - b).abs() <= e, "{a} != {b}; tolerance {e}");
}

#[test]
fn deterministic_rates_project_to_existing_paths_and_preserve_cash_event_order() {
    let r = request(false, &[(0.5, 4.0), (1.0, 3.0), (1.4, 8.0)]);
    let market = r.market().equity().forward();
    let grid = LocalVolTimeGrid::compile(vec![0.0, 0.5, 1.0], 0.125).unwrap();
    let hw = Path::compile_bs(market, model(0.7), 0.2, &rates(0.0), 0.25, -0.2, &grid).unwrap();
    let fixed = DeterministicPath::compile(market, model(0.7), 0.2, &grid).unwrap();
    close(hw.risky_spot(), fixed.risky_spot(), 3e-14);
    let z: Vec<f64> = (0..hw.random_dimension())
        .map(|i| (i as f64 * 0.71).sin())
        .collect();
    let projected: Vec<f64> = z.chunks_exact(4).flat_map(|z| [z[0], z[1]]).collect();
    let a = hw.evolve_path(&z).unwrap();
    let b = fixed.evolve_path(&projected).unwrap();
    for (i, (h, d)) in a.iter().zip(&b).enumerate() {
        assert_eq!(h.factors(), *d);
        let (post, pre) = hw.spots(i, *h).unwrap();
        let (post0, pre0) = fixed.nodes()[i].spots(*d).unwrap();
        close(post, post0, 2e-12);
        assert_eq!(pre.is_some(), pre0.is_some());
        if let Some(pre) = pre {
            close(pre, pre0.unwrap(), 2e-12);
        }
    }
    let last = *a.last().unwrap();
    close(
        hw.dividend_claim_value(8, 1, last).unwrap(),
        3.0 * last.factors().dividend(),
        1e-14,
    );
    assert_eq!(hw.dividend_claim_value(8, 0, last).unwrap(), 0.0);
    assert!(hw.dividend_claim_value(9, 0, last).is_err());
    assert!(hw.dividend_claim_value(0, 5, last).is_err());
}

#[test]
fn independent_conditional_gaussian_one_step_option_prices() {
    let r = request(true, &[(1.0, 3.0), (1.4, 8.0)]);
    // Independent three-dimensional Gaussian quadrature conditions on W_D,x,I
    // and integrates W_f with Black. No production simulation/reserve helpers.
    for (k, expected) in [(0.0, 7.029077680799348), (0.7, 7.266840212859318)] {
        let plan = Plan::compile_bs(&r, model(k), rates(0.04), 0.25, -0.2, 1.0, policy(2)).unwrap();
        let p = plan.evaluate().unwrap();
        eprintln!(
            "kappa={k}, price={}, SE={}, reference={expected}",
            p.value, p.standard_error
        );
        close(p.value, expected, 6.0 * p.standard_error + 0.002);
        assert_eq!(p.plan_fingerprint, plan.plan_fingerprint());
        assert_eq!(plan.random_factor_count(), 4);
        assert_eq!(p.independent_sampling_units, 8);
        assert_eq!(p.evaluated_paths, 65536);
    }
}

#[test]
fn cash_mean_and_collateral_forward_are_distinct_with_correlated_rates() {
    let r = request(false, &[(1.0, 3.0), (1.4, 8.0)]);
    let m = model(0.0);
    let p = Plan::compile_bs(&r, m, rates(0.04), 0.25, -0.2, 0.5, policy(1)).unwrap();
    let mut reserve = 0.0;
    for (i, (t, mean)) in [(1.0, 3.0), (1.4, 8.0)].iter().enumerate() {
        let a = 0.4;
        let j = (t - (-(-a * t).exp_m1()) / a) / a;
        let forward = mean * (-0.35_f64 * (-0.2) * 0.04 * j).exp();
        close(p.initial_dividend_forwards()[i], forward, 2e-13);
        close(
            p.initial_dividend_claim_values()[i],
            forward * 0.95_f64.powf(*t),
            2e-13,
        );
        reserve += forward * (0.95_f64 / 0.98).powf(*t);
        assert!((forward - mean).abs() > 1e-4);
    }
    close(p.risky_spot(), 100.0 - reserve, 2e-13);
    let fixed = Model::new(0.7, 0.0, 0.0, -0.25).unwrap();
    let p = Plan::compile_bs(&r, fixed, rates(0.04), 0.25, -0.2, 0.5, policy(1)).unwrap();
    close(p.initial_dividend_forwards()[0], 3.0, 2e-11);
    close(p.initial_dividend_forwards()[1], 8.0, 2e-11);
}

#[test]
fn discounted_carry_adjusted_stock_and_paid_cash_have_initial_value_in_expectation() {
    let r = request(false, &[(0.5, 4.0), (1.0, 3.0), (1.4, 8.0)]);
    let market = r.market().equity().forward();
    let hw = rates(0.04);
    let grid = LocalVolTimeGrid::compile(vec![0.5, 1.0], 1.0 / 32.0).unwrap();
    let p = Path::compile_bs(market, model(0.7), 0.2, &hw, 0.25, -0.2, &grid).unwrap();
    let rng = Philox4x32::from_seed(1087);
    let count = 8192u64;
    let mut values = Vec::new();
    let mut discounts = Vec::new();
    for path in 0..count {
        let z: Vec<f64> = (0..p.random_dimension())
            .map(|d| rng.standard_normal(RandomCoordinate::new(path, d, RandomDomain::Valuation)))
            .collect();
        let mut total = 0.0;
        let mut dfmean = 0.0;
        for sign in [1.0, -1.0] {
            let z: Vec<f64> = z.iter().map(|v| sign * v).collect();
            let states = p.evolve_path(&z).unwrap();
            let mut value = 0.0;
            for (i, &t) in p.times().iter().enumerate() {
                let df = market.discount_curve().discount(t).unwrap()
                    * hw.relative_discount(t, states[i].integrated_rate_factor())
                        .unwrap();
                let (post, pre) = p.spots(i, states[i]).unwrap();
                let paid = pre.map_or(0.0, |pre| pre - post);
                value += df / market.dividend_curve().discount(t).unwrap() * paid;
                if i + 1 == states.len() {
                    value += df / market.dividend_curve().discount(t).unwrap() * post;
                    dfmean += 0.5 * df;
                }
            }
            total += 0.5 * value;
        }
        values.push(total);
        discounts.push(dfmean);
    }
    for (v, target, discretization) in [(values, 100.0, 0.02), (discounts, 0.95, 1e-12)] {
        let mean = v.iter().sum::<f64>() / count as f64;
        let se = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((count - 1) * count) as f64)
            .sqrt();
        eprintln!("martingale target={target}, mean={mean}, SE={se}");
        close(mean, target, 6.0 * se + discretization);
    }
}

#[test]
fn worker_replay_fingerprints_and_invalid_domains_are_explicit() {
    let r = request(false, &[(0.5, 4.0), (1.4, 8.0)]);
    let compile =
        |workers, rf, rd, hw| Plan::compile_bs(&r, model(0.7), hw, rf, rd, 0.125, policy(workers));
    let a = compile(1, 0.25, -0.2, rates(0.04)).unwrap();
    let b = compile(3, 0.25, -0.2, rates(0.04)).unwrap();
    let x = a.evaluate().unwrap();
    let y = b.evaluate().unwrap();
    assert_eq!(x.value, y.value);
    assert_eq!(x.standard_error, y.standard_error);
    assert_ne!(a.plan_fingerprint(), b.plan_fingerprint());
    for (rf, rd, vol) in [(0.3, -0.2, 0.04), (0.25, -0.3, 0.04), (0.25, -0.2, 0.05)] {
        assert_ne!(
            a.plan_fingerprint(),
            compile(1, rf, rd, rates(vol)).unwrap().plan_fingerprint()
        );
    }
    // Each pair can be valid while the full triple is not PSD; no silent repair.
    assert!(compile(1, 0.99, 0.99, rates(0.0)).is_err());
    assert!(compile(1, f64::NAN, 0.0, rates(0.0)).is_err());
    let grid = LocalVolTimeGrid::compile(vec![0.5, 1.0], 0.5).unwrap();
    let path = Path::compile_bs(
        r.market().equity().forward(),
        model(0.7),
        0.2,
        &rates(0.04),
        0.25,
        -0.2,
        &grid,
    )
    .unwrap();
    assert!(path.evolve_path(&[]).is_err());
    assert!(
        path.evolve_path(&vec![f64::NAN; path.random_dimension() as usize])
            .is_err()
    );
    assert!(path.spots(99, State::initial()).is_err());
    // A PSD boundary remains priceable, with all four random coordinates.
    let perfect = Model::new(0.7, 0.6, 0.35, 1.0).unwrap();
    assert!(
        Plan::compile_bs(&r, perfect, rates(0.04), 0.2, 0.2, 0.125, policy(1))
            .unwrap()
            .evaluate()
            .is_ok()
    );
}
