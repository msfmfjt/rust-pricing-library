use pricing::hull_white::HullWhiteEquityPricingPlan as HybridPlan;
use pricing::mc::hull_white::{HullWhiteEquityPlan, HybridEquityVolatility};
use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid};
use pricing::models::{
    Bergomi1Factor, Bergomi2Factor, HullWhite1Factor, HybridCorrelation, RoughBergomi,
};
use pricing::stochastic_volatility::StochasticVolatilityPricingPlan as PurePlan;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn rates(sigma: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.0, vec![0.0], vec![sigma]).unwrap()
}
fn policy(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(64)).unwrap()
}
fn payload(qmc: bool, cash: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,"scramble_count":4,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    if cash {
        v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.35,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":6.0,"beta":0.05}},
            {"event_id":2,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":3.0}}
        ]);
    }
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn factor2(nu: f64, theta: f64) -> Bergomi2Factor {
    Bergomi2Factor::new([0.7, 2.1], nu, theta, [-0.5, -0.3], 0.25).unwrap()
}
fn compile(v: Value, two: bool, rate_sigma: f64, workers: u32) -> HybridPlan {
    let r = request(v);
    if two {
        HybridPlan::compile_bergomi_two_factor(
            &r,
            factor2(0.6, 0.35),
            rates(rate_sigma),
            0.2,
            [-0.1, 0.1],
            0.125,
            policy(workers),
        )
        .unwrap()
    } else {
        HybridPlan::compile_bergomi(
            &r,
            Bergomi1Factor::new(0.7, 0.6, -0.5).unwrap(),
            rates(rate_sigma),
            HybridCorrelation::new(-0.5, 0.2, -0.1).unwrap(),
            0.125,
            policy(workers),
        )
        .unwrap()
    }
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b}, tolerance {tol}");
}

#[test]
fn pure_ou_centering_and_external_innovations_follow_analytic_covariance() {
    // Compute raw OU evolution and its covariance independently. This checks
    // every nonuniform node, including centering after the first step.
    let grid = LocalVolTimeGrid::compile(vec![0.13, 0.5, 0.9, 1.0], 1.0).unwrap();
    let nu = 0.6;
    let corr = HybridCorrelation::new(-0.5, 0.0, 0.0).unwrap();
    let b = |k: f64, t: f64| if k == 0.0 { t } else { -(-k * t).exp_m1() / k };
    for two in [false, true] {
        for k in [0.0, 1e-12, 0.7] {
            let f2 = Bergomi2Factor::new([k, 2.1], nu, 0.35, [-0.5, -0.3], 0.25).unwrap();
            let volatility = if two {
                HybridEquityVolatility::Bergomi2Factor {
                    factor: f2,
                    second_vol_rate_correlation: 0.0,
                    initial_volatility: 0.2,
                }
            } else {
                HybridEquityVolatility::Bergomi {
                    factor: Bergomi1Factor::new(k, nu, -0.5).unwrap(),
                    initial_volatility: 0.2,
                }
            };
            let p = HullWhiteEquityPlan::new(rates(0.0), volatility, corr, &grid).unwrap();
            let n = grid.nodes().len() - 1;
            let zero = p
                .evolve_path(100.0, &vec![0.0; n * p.random_factor_count()])
                .unwrap();
            let innovations: Vec<_> = (0..n)
                .map(|i| [0.03 * (i as f64 - 1.0), 0.07, 0.0, 0.0, -0.04])
                .collect();
            let states = p.evolve_with_innovations(100.0, &innovations).unwrap();
            let weights = if two {
                f2.normalized_weights()
            } else {
                [1.0, 0.0]
            };
            let mut x = [0.0; 2];
            let mut expected_equity = 100.0;
            let mut last_centered = 0.0;
            for (i, pair) in grid.nodes().windows(2).enumerate() {
                let dt = pair[1] - pair[0];
                let v = 0.04 * (2.0 * nu * last_centered).exp();
                expected_equity *= (-0.5 * v * dt + v.sqrt() * innovations[i][0]).exp();
                x[0] = (-k * dt).exp() * x[0] + innovations[i][1];
                x[1] = (-2.1 * dt).exp() * x[1] + innovations[i][4];
                let t = pair[1];
                let variance = weights[0].powi(2) * b(2.0 * k, t)
                    + weights[1].powi(2) * b(4.2, t)
                    + 2.0 * weights[0] * weights[1] * 0.25 * b(k + 2.1, t);
                last_centered = weights[0] * x[0] + weights[1] * x[1] - nu * variance;
                close(zero[i + 1].volatility_factor, -nu * variance, 2e-14);
                close(states[i + 1].volatility_factor, last_centered, 2e-14);
                close(states[i + 1].normalized_equity, expected_equity, 2e-12);
            }
        }
    }
}

#[test]
fn black_scholes_one_factor_and_rough_limits_share_paths() {
    let grid = LocalVolTimeGrid::compile(vec![0.13, 0.5, 0.9, 1.0], 1.0).unwrap();
    let n = grid.nodes().len() - 1;
    let shocks: Vec<_> = (0..5 * n).map(|i| (i as f64 * 1.7).sin()).collect();
    let corr = HybridCorrelation::new(-0.5, 0.2, -0.1).unwrap();
    for nu in [0.0, 0.6] {
        let one = HullWhiteEquityPlan::new(
            rates(0.01),
            HybridEquityVolatility::Bergomi {
                factor: Bergomi1Factor::new(0.0, nu, -0.5).unwrap(),
                initial_volatility: 0.2,
            },
            corr,
            &grid,
        )
        .unwrap();
        let two = HullWhiteEquityPlan::new(
            rates(0.01),
            HybridEquityVolatility::Bergomi2Factor {
                factor: Bergomi2Factor::new([0.0, 2.1], nu, 0.0, [-0.5, -0.3], 0.25).unwrap(),
                second_vol_rate_correlation: 0.1,
                initial_volatility: 0.2,
            },
            corr,
            &grid,
        )
        .unwrap();
        let rough = HullWhiteEquityPlan::new(
            rates(0.01),
            HybridEquityVolatility::RoughBergomi {
                factor: RoughBergomi::new(0.5, 2.0 * nu, -0.5).unwrap(),
                initial_volatility: 0.2,
            },
            corr,
            &grid,
        )
        .unwrap();
        let a = one.evolve_path(100.0, &shocks[..4 * n]).unwrap();
        for b in [
            two.evolve_path(100.0, &shocks).unwrap(),
            rough.evolve_path(100.0, &shocks).unwrap(),
        ] {
            for (x, y) in a.iter().zip(b) {
                close(x.normalized_equity, y.normalized_equity, 3e-12);
                close(x.volatility_factor, y.volatility_factor, 2e-14);
            }
        }
        if nu == 0.0 {
            let bs = HullWhiteEquityPlan::new(
                rates(0.01),
                HybridEquityVolatility::BlackScholes(0.2),
                corr,
                &grid,
            )
            .unwrap();
            for (x, y) in a
                .iter()
                .zip(bs.evolve_path(100.0, &shocks[..4 * n]).unwrap())
            {
                assert_eq!(x.normalized_equity, y.normalized_equity);
            }
        }
    }
}

#[test]
fn public_price_aad_replay_and_escrowed_dividend_contract() {
    for two in [false, true] {
        for qmc in [false, true] {
            for cash in [false, true] {
                let v = payload(qmc, cash);
                let p = compile(v.clone(), two, 0.007, 1);
                let risk = p.evaluate_aad().unwrap();
                assert_eq!(risk.price.value, p.evaluate().unwrap().value);
                assert_eq!(p.random_factor_count(), if two { 5 } else { 4 });
                assert!(p.calibration().is_none());
                assert!(risk.price.calibration_method.is_none());
                assert!(risk.price.calibration_seed.is_none());
                assert!(risk.vega_kt_raw().is_none());
                assert_eq!(&risk.parameter_labels[1], "initial_volatility");
                if cash {
                    assert!(p.risky_spot() < 91.0);
                }
                for (field, value, bar, h) in [
                    ("spot", 100.0, risk.delta(), 1e-5),
                    ("volatility", 0.2, risk.vega().unwrap(), 1e-6),
                ] {
                    let mut up = v.clone();
                    let mut down = v.clone();
                    let container = if field == "spot" { "market" } else { "model" };
                    up[container][field] = json!(value + h);
                    down[container][field] = json!(value - h);
                    let fd = (compile(up, two, 0.007, 1).evaluate().unwrap().value
                        - compile(down, two, 0.007, 1).evaluate().unwrap().value)
                        / (2.0 * h);
                    close(bar, fd, 3e-5);
                }
                let replay = compile(v, two, 0.007, 3).evaluate_aad().unwrap();
                assert_eq!(risk.derivatives, replay.derivatives);
                assert_eq!(risk.standard_errors, replay.standard_errors);
            }
        }
    }
}

#[test]
fn deterministic_facade_matches_zero_rate_volatility_and_separates_fingerprints() {
    let r = request(payload(true, true));
    let f = Bergomi1Factor::new(0.7, 0.6, -0.5).unwrap();
    let p = PurePlan::compile_bergomi(&r, f, 0.125, policy(1)).unwrap();
    let h = HybridPlan::compile_bergomi(
        &r,
        f,
        rates(0.0),
        HybridCorrelation::new(-0.5, 0.0, 0.0).unwrap(),
        0.125,
        policy(1),
    )
    .unwrap();
    assert_eq!(p.plan_fingerprint(), h.plan_fingerprint());
    assert_eq!(
        p.evaluate_aad().unwrap().derivatives,
        h.evaluate_aad().unwrap().derivatives
    );
    let two =
        PurePlan::compile_bergomi_two_factor(&r, factor2(0.6, 0.35), 0.125, policy(1)).unwrap();
    let rough = PurePlan::compile_rough_bergomi(
        &r,
        RoughBergomi::new(0.1, 1.2, -0.5).unwrap(),
        0.125,
        policy(1),
    )
    .unwrap();
    assert_ne!(p.plan_fingerprint(), two.plan_fingerprint());
    assert_ne!(p.plan_fingerprint(), rough.plan_fingerprint());
    let changed = PurePlan::compile_bergomi(
        &r,
        Bergomi1Factor::new(0.7, 0.61, -0.5).unwrap(),
        0.125,
        policy(1),
    )
    .unwrap();
    assert_ne!(p.plan_fingerprint(), changed.plan_fingerprint());
}

#[test]
fn invalid_targets_steps_and_joint_correlations_are_rejected() {
    let r = request(payload(false, false));
    let f = Bergomi1Factor::new(0.7, 0.6, -0.5).unwrap();
    for step in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(PurePlan::compile_bergomi(&r, f, step, policy(1)).is_err());
    }
    let mut v = payload(false, false);
    v["model"] = json!({"type":"black_76","volatility":0.2});
    assert!(PurePlan::compile_bergomi(&request(v), f, 0.1, policy(1)).is_err());
    assert!(
        HybridPlan::compile_bergomi(
            &r,
            f,
            rates(0.01),
            HybridCorrelation::new(0.0, 0.0, 0.0).unwrap(),
            0.1,
            policy(1)
        )
        .is_err()
    );
    assert!(
        HybridPlan::compile_bergomi_two_factor(
            &r,
            factor2(0.6, 0.35),
            rates(0.01),
            0.9,
            [0.9, 0.9],
            0.1,
            policy(1)
        )
        .is_err()
    );
}
