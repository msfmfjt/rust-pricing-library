//! Public joint-model contracts and independently integrated two-step prices.
use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid};
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::stochastic_dividends::{
    BuehlerDividendModel as Dividend, StochasticDividendPathPlan as Path,
    StochasticDividendPricingPlan as Plan,
};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};
fn request(points: u64) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":points,
        "scramble_count":8,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":25.0}}]);
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn dividend() -> Dividend {
    Dividend::new(0.7, 0.6, 0.35, -0.25).unwrap()
}
fn one(nu: f64) -> Bergomi1Factor {
    Bergomi1Factor::new(0.8, nu, -0.4).unwrap()
}
fn two(nu: f64, theta: f64) -> Bergomi2Factor {
    Bergomi2Factor::new([0.8, 2.1], nu, theta, [-0.4, -0.2], 0.3).unwrap()
}
fn policy(n: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(n, Some(64)).unwrap()
}

#[test]
fn zero_vol_of_vol_projects_to_unchanged_bs_dividend_paths() {
    let r = request(16);
    let market = r.market().equity().forward();
    let grid = LocalVolTimeGrid::compile(vec![0.13, 0.5, 1.0], 1.0).unwrap();
    let bs = Path::compile(market, dividend(), 0.2, &grid).unwrap();
    let p1 = Path::compile_bergomi(market, dividend(), 0.2, one(0.0), 0.15, &grid).unwrap();
    let p2 = Path::compile_bergomi_two_factor(
        market,
        dividend(),
        0.2,
        two(0.0, 0.35),
        [0.15, -0.1],
        &grid,
    )
    .unwrap();
    let z = [0.2, -0.6, 1.1, 0.4, -0.8, 0.3];
    let expanded = |n: usize| {
        z.as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pair| (0..n).map(move |i| if i < 2 { pair[i] } else { 0.7 }))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bs.evolve_path(&z).unwrap(),
        p1.evolve_path(&expanded(3)).unwrap()
    );
    assert_eq!(
        bs.evolve_path(&z).unwrap(),
        p2.evolve_path(&expanded(4)).unwrap()
    );
    assert_eq!(p1.random_dimension(), 9);
    assert_eq!(p2.random_dimension(), 12);
    assert!(p1.evolve_path(&[0.0; 8]).is_err());
    assert!(p1.evolve_path(&[f64::NAN; 9]).is_err());
}

#[test]
fn zero_second_weight_retains_coordinates_and_matches_one_factor() {
    let r = request(16);
    let market = r.market().equity().forward();
    let grid = LocalVolTimeGrid::compile(vec![0.13, 0.5, 1.0], 1.0).unwrap();
    let a = Path::compile_bergomi(market, dividend(), 0.2, one(0.3), 0.15, &grid).unwrap();
    let b = Path::compile_bergomi_two_factor(
        market,
        dividend(),
        0.2,
        two(0.3, 0.0),
        [0.15, -0.1],
        &grid,
    )
    .unwrap();
    let z3 = vec![0.4, -0.3, 1.2, -0.7, 0.1, -0.2, 0.9, -0.4, 0.1];
    let z4: Vec<_> = z3
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2], 2.4])
        .collect();
    let x = a.evolve_path(&z3).unwrap();
    let y = b.evolve_path(&z4).unwrap();
    for (x, y) in x.iter().zip(y) {
        assert!((x.equity() - y.equity()).abs() < 2e-15);
        assert!((x.dividend() - y.dividend()).abs() < 2e-15);
    }
    assert_eq!(b.random_factor_count(), 4);
}

#[test]
fn instantaneous_psd_is_required_even_when_volatility_loading_is_zero() {
    let r = request(16);
    let d = Dividend::new(0.7, 0.6, 0.0, 0.9).unwrap();
    let v = Bergomi1Factor::new(20.0, 0.0, 0.9).unwrap();
    assert!(Plan::compile_bergomi(&r, d, v, -0.9, 0.5, policy(1)).is_err());
    for bad in [f64::NAN, f64::INFINITY, 1.01] {
        assert!(Plan::compile_bergomi(&r, dividend(), one(0.3), bad, 0.5, policy(1)).is_err());
        assert!(
            Plan::compile_bergomi_two_factor(
                &r,
                dividend(),
                two(0.3, 0.35),
                [0.15, bad],
                0.5,
                policy(1)
            )
            .is_err()
        );
    }
    // Perfectly identical Brownian drivers still have different OU kernels.
    let d = Dividend::new(0.7, 0.6, 0.35, 1.0).unwrap();
    let v = Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [1.0, 1.0], 1.0).unwrap();
    assert!(
        Plan::compile_bergomi_two_factor(&r, d, v, [1.0, 1.0], 0.5, policy(1))
            .unwrap()
            .evaluate()
            .is_ok()
    );
}

#[test]
fn independent_two_step_conditional_black_prices_with_nonzero_mean_reversion() {
    let r = request(8192);
    // Values independently integrated by tests/python/bergomi_dividend_reference.py,
    // Gaussian order 20. Order 16 differences are < 4.1e-7, not MC-derived prices.
    for (two_factor, expected) in [(false, 7.772111247866506), (true, 7.775600673296135)] {
        let p = if two_factor {
            Plan::compile_bergomi_two_factor(
                &r,
                dividend(),
                two(0.3, 0.35),
                [0.15, -0.1],
                0.5,
                policy(2),
            )
        } else {
            Plan::compile_bergomi(&r, dividend(), one(0.3), 0.15, 0.5, policy(2))
        }
        .unwrap();
        let result = p.evaluate().unwrap();
        println!(
            "two={two_factor}, price={}, se={}, reference={expected}",
            result.value, result.standard_error
        );
        assert!((result.value - expected).abs() < 6.0 * result.standard_error + 0.002);
        assert_eq!(result.independent_sampling_units, 8);
        assert_eq!(result.evaluated_paths, 131072);
        assert_eq!(result.scheme, p.scheme());
    }
}

#[test]
fn workers_and_fingerprints_include_all_joint_model_inputs() {
    let r = request(128);
    for is_two in [false, true] {
        let make = |workers| {
            if is_two {
                Plan::compile_bergomi_two_factor(
                    &r,
                    dividend(),
                    two(0.3, 0.35),
                    [0.15, -0.1],
                    0.125,
                    policy(workers),
                )
                .unwrap()
            } else {
                Plan::compile_bergomi(&r, dividend(), one(0.3), 0.15, 0.125, policy(workers))
                    .unwrap()
            }
        };
        let a = make(1);
        let b = make(3);
        let x = a.evaluate().unwrap();
        let y = b.evaluate().unwrap();
        assert_eq!(x.value, y.value);
        assert_eq!(x.standard_error, y.standard_error);
        assert_eq!(x, a.evaluate().unwrap());
        let bs = Plan::compile_bs(&r, dividend(), 0.125, policy(1)).unwrap();
        assert_ne!(a.plan_fingerprint(), bs.plan_fingerprint());
    }
    let a = Plan::compile_bergomi_two_factor(
        &r,
        dividend(),
        two(0.3, 0.0),
        [0.15, -0.1],
        0.5,
        policy(1),
    )
    .unwrap();
    let b = Plan::compile_bergomi_two_factor(
        &r,
        dividend(),
        two(0.3, 0.0),
        [0.15, 0.1],
        0.5,
        policy(1),
    )
    .unwrap();
    assert_ne!(a.plan_fingerprint(), b.plan_fingerprint()); // even unused factor correlation
}
