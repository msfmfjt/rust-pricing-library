//! Independent finite-bump references, and estimator covariance from raw units.
use pricing::mc::{
    EngineConfig, ExecutionPolicy, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan,
    inverse_standard_normal,
};
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::risk::{GammaConfig, SpotBump};
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
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":4.}},
        {"event_id":2,"ex_time":1.,"quote":{"type":"fixed_cash","amount":3.}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":8.}}]);
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,
        "scramble_count":4,"master_scramble_seed":seed,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":256,"master_seed":seed,
        "variance_reduction":{"antithetic":true,"brownian_bridge":false}})
    };
    v
}
fn config(h: f64) -> GammaConfig {
    GammaConfig::new(SpotBump::absolute(h).unwrap())
}
fn compile(v: &Value, family: usize, workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
    let policy = ExecutionPolicy::new(workers, Some(64)).unwrap();
    match family {
        0 => Plan::compile_bs(&r, d, 0.125, policy),
        1 => Plan::compile_bergomi(
            &r,
            d,
            Bergomi1Factor::new(0.8, 0.3, -0.4).unwrap(),
            0.15,
            0.125,
            policy,
        ),
        _ => Plan::compile_bergomi_two_factor(
            &r,
            d,
            Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [-0.4, -0.2], 0.3).unwrap(),
            [0.15, -0.1],
            0.125,
            policy,
        ),
    }
    .unwrap()
}
#[test]
fn ladder_matches_full_recompile_delta_for_all_families_and_payoff_events() {
    for qmc in [false, true] {
        for seed in [217, 813] {
            for family in 0..3 {
                for product in 0..3 {
                    let mut v = payload(qmc, seed);
                    if product == 1 {
                        v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,
                "strike":100.,"notional":1.,"side":{"type":"call"},
                "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
                {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
                    } else if product == 2 {
                        v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04",
                "strike":80.,"barrier":100.,"notional":1.,"side":{"type":"call"},"direction":{"type":"up"},
                "style":{"type":"knock_in"},"monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
                        v["risk"]["payoff_smoothing"] =
                            json!({"type":"compact_c2","half_width":8.});
                    }
                    let p = compile(&v, family, 1);
                    let r = p.evaluate_gamma(config(1.)).unwrap();
                    let basic = p.evaluate_aad().unwrap();
                    assert_eq!(r.price, p.evaluate().unwrap());
                    assert_eq!(r.delta, basic.delta());
                    assert_eq!(r.delta_standard_error, basic.standard_errors[0]);
                    let mut replay = compile(&v, family, 3).evaluate_gamma(config(1.)).unwrap();
                    // Execution-policy provenance changes, while every numeric result replays.
                    assert_ne!(r.price.plan_fingerprint, replay.price.plan_fingerprint);
                    assert_ne!(r.risk_fingerprint, replay.risk_fingerprint);
                    replay.price.plan_fingerprint = r.price.plan_fingerprint;
                    replay.risk_fingerprint = r.risk_fingerprint;
                    assert_eq!(r, replay);
                    assert_eq!(r.payoff_evaluations, 7 * r.price.evaluated_paths);
                    for (j, &h) in r.spot_bumps.iter().enumerate() {
                        let mut plus = v.clone();
                        let mut minus = v.clone();
                        plus["market"]["spot"] = json!(100. + h);
                        minus["market"]["spot"] = json!(100. - h);
                        let fd = (compile(&plus, family, 1).evaluate_aad().unwrap().delta()
                            - compile(&minus, family, 1).evaluate_aad().unwrap().delta())
                            / (2. * h);
                        assert!(
                            (r.gamma_estimates[j] - fd).abs() < 2e-12,
                            "family={family}, product={product}, qmc={qmc}, {fd} != {}",
                            r.gamma_estimates[j]
                        );
                    }
                    for j in 0..2 {
                        assert!(
                            (r.bump_differences[j]
                                - (r.gamma_estimates[j] - r.gamma_estimates[j + 1]))
                                .abs()
                                < 2e-12
                        );
                    }
                    let relative = p
                        .evaluate_gamma(GammaConfig::new(SpotBump::relative(0.01).unwrap()))
                        .unwrap();
                    assert_eq!(r.gamma_estimates, relative.gamma_estimates);
                    assert_eq!(r.gamma_standard_errors, relative.gamma_standard_errors);
                    assert_ne!(r.risk_fingerprint, relative.risk_fingerprint);
                    assert_ne!(
                        r.risk_fingerprint,
                        p.evaluate_gamma(config(0.5)).unwrap().risk_fingerprint
                    );
                }
            }
        }
    }
}
fn moments(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    (
        mean,
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt(),
    )
}
#[test]
fn paired_sampling_errors_match_independent_mc_units_and_rqmc_scramble_means() {
    // kappa=0, one time step: an independent lognormal terminal construction.
    for qmc in [false, true] {
        for antithetic in [false, true] {
            let mut v = payload(qmc, 119);
            v["engine"]["variance_reduction"]["brownian_bridge"] = json!(false);
            v["engine"]["variance_reduction"]["antithetic"] = json!(antithetic);
            v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":1.,"quote":{"type":"fixed_cash","amount":3.}},
            {"event_id":2,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":25.}}]);
            let r =
                parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
            let p = Plan::compile_bs(
                &r,
                BuehlerDividendModel::new(0., 0.6, 0.45, -0.35).unwrap(),
                1.,
                ExecutionPolicy::new(1, Some(64)).unwrap(),
            )
            .unwrap();
            let out = p.evaluate_gamma(config(1.)).unwrap();
            let growth: f64 = 0.98 / 0.95;
            let funded = 3. / growth + 25. / growth.powf(1.4);
            let b = 25. * growth.powf(-0.4);
            let single = |z: [f64; 2]| {
                let mut gammas = [0.; 3];
                for sign in if antithetic {
                    &[1., -1.][..]
                } else {
                    &[1.][..]
                } {
                    let f = (-0.5 * 0.2 * 0.2 + 0.2 * sign * z[0]).exp();
                    let zd = -0.35 * sign * z[0] + (1. - 0.35_f64.powi(2)).sqrt() * sign * z[1];
                    let y = (-0.5 * 0.45 * 0.45 + 0.45 * zd).exp();
                    let delta = |s: f64| {
                        if growth * (s - funded) * f + b * y > 100. {
                            0.95 * growth * f
                        } else {
                            0.
                        }
                    };
                    for (j, h) in [0.5, 1., 2.].iter().enumerate() {
                        gammas[j] += (delta(100. + h) - delta(100. - h)) / (2. * h);
                    }
                }
                if antithetic {
                    for g in &mut gammas {
                        *g *= 0.5;
                    }
                }
                [
                    gammas[0],
                    gammas[1],
                    gammas[2],
                    gammas[0] - gammas[1],
                    gammas[1] - gammas[2],
                ]
            };
            let samples = match r.engine() {
                EngineConfig::PseudoMonteCarlo(c) => {
                    let rng = Philox4x32::from_seed(c.master_seed());
                    (0..c.independent_sampling_units().get())
                        .map(|i| {
                            single([
                                rng.standard_normal(RandomCoordinate::new(
                                    i,
                                    0,
                                    RandomDomain::Valuation,
                                )),
                                rng.standard_normal(RandomCoordinate::new(
                                    i,
                                    1,
                                    RandomDomain::Valuation,
                                )),
                            ])
                        })
                        .collect::<Vec<_>>()
                }
                EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                    let q = RqmcPlan::compile(c, 2).unwrap();
                    (0..c.scramble_count().get())
                        .map(|s| {
                            let mut means = [0.; 5];
                            for i in 0..c.points_per_scramble().get() {
                                let a = single([
                                    inverse_standard_normal(q.uniform(s, i, 0).unwrap()).unwrap(),
                                    inverse_standard_normal(q.uniform(s, i, 1).unwrap()).unwrap(),
                                ]);
                                for (m, x) in means.iter_mut().zip(a) {
                                    *m += x;
                                }
                            }
                            for m in &mut means {
                                *m /= c.points_per_scramble().get() as f64;
                            }
                            means
                        })
                        .collect()
                }
            };
            for j in 0..5 {
                let (mean, se) = moments(&samples.iter().map(|s| s[j]).collect::<Vec<_>>());
                let (actual, err) = if j < 3 {
                    (out.gamma_estimates[j], out.gamma_standard_errors[j])
                } else {
                    (
                        out.bump_differences[j - 3],
                        out.bump_difference_standard_errors[j - 3],
                    )
                };
                assert!((mean - actual).abs() < 2e-12);
                assert!(
                    (se - err).abs() < 2e-12,
                    "qmc={qmc},anti={antithetic},j={j}: {se} != {err}"
                );
            }
            assert_eq!(out.price.independent_sampling_units, samples.len() as u64);
        }
    }
}
fn cdf(x: f64) -> f64 {
    pricing_numerics::standard_normal_cdf(x)
}
#[test]
fn fixed_cash_gamma_matches_independent_black_delta_bumps() {
    let mut v = payload(true, 772);
    v["engine"]["points_per_scramble"] = json!(32768);
    v["engine"]["scramble_count"] = json!(8);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":1.,"quote":{"type":"fixed_cash","amount":3.}},
        {"event_id":2,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":25.}}]);
    let request =
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let p = Plan::compile_bs(
        &request,
        BuehlerDividendModel::new(0., 0., 0., -0.35).unwrap(),
        1.,
        ExecutionPolicy::new(2, Some(64)).unwrap(),
    )
    .unwrap();
    let r = p.evaluate_gamma(config(1.)).unwrap();
    let growth: f64 = 0.98 / 0.95;
    let reserve = 3. / growth + 25. / growth.powf(1.4);
    let b = 25. * growth.powf(-0.4);
    let delta = |s: f64| {
        let a = growth * (s - reserve);
        0.95 * growth * cdf((a / (100. - b)).ln() / 0.2 + 0.1)
    };
    for (j, h) in r.spot_bumps.iter().enumerate() {
        let expected = (delta(100. + h) - delta(100. - h)) / (2. * h);
        assert!(
            (r.gamma_estimates[j] - expected).abs() < 6. * r.gamma_standard_errors[j] + 3e-5,
            "j={j}, {} vs {expected}",
            r.gamma_estimates[j]
        );
    }
}
#[test]
fn bump_domain_and_unsmoothed_discontinuities_reject_without_changing_basic_risk() {
    let p = compile(&payload(false, 45), 0, 1);
    for h in [1e-300, 60., f64::MAX] {
        assert!(p.evaluate_gamma(config(h)).is_err());
    }
    assert!(SpotBump::absolute(0.).is_err());
    assert!(SpotBump::relative(f64::NAN).is_err());
    let before = p.evaluate_aad().unwrap();
    let _ = p.evaluate_gamma(config(60.));
    assert_eq!(before, p.evaluate_aad().unwrap());
    let mut v = payload(false, 45);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04",
        "strike":80.,"barrier":100.,"notional":1.,"side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},
        "monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
    assert!(compile(&v, 0, 1).evaluate_gamma(config(1.)).is_err());
    // Largest downward ladder bump remains positive but exceeds funded equity.
    let mut v = payload(false, 46);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":1.,"quote":{"type":"fixed_cash","amount":90.}}]);
    assert!(compile(&v, 0, 1).evaluate_gamma(config(8.)).is_err());
}
#[test]
fn zero_volatility_itm_gamma_is_zero_and_perfect_correlation_keeps_gamma_available() {
    let mut v = payload(true, 53);
    v["model"]["volatility"] = json!(0.);
    v["product"]["strike"] = json!(40.);
    let req = parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    for rho in [-1., 1.] {
        let p = Plan::compile_bs(
            &req,
            BuehlerDividendModel::new(0., 0., 0., rho).unwrap(),
            0.125,
            ExecutionPolicy::new(1, Some(64)).unwrap(),
        )
        .unwrap();
        let r = p.evaluate_gamma(config(1.)).unwrap();
        assert!(r.gamma_estimates.iter().all(|x| x.abs() < 1e-12));
        assert!(r.gamma_standard_errors.iter().all(|x| x.abs() < 1e-12));
        assert!(p.evaluate_correlation_aad().is_err());
    }
}
