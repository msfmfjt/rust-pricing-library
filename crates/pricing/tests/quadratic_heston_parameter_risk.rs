//! Fixed-driver quadratic history derivatives, including optional Hurst.
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn model(p: [f64; 7]) -> RoughVolatilityModel {
    QuadraticRoughHeston::new(p[6], p[0], p[1], p[2], p[3], p[4], p[5])
        .unwrap()
        .into()
}
const PARAMETERS: [f64; 7] = [0.1, 0.7, 0.4, 0.5, 0.2, 0.03, 0.1];
fn shifted(mut p: [f64; 7], j: usize, e: f64) -> [f64; 7] {
    p[j] += e;
    p
}
fn difference(f: impl Fn(f64) -> f64, e: f64, left: bool) -> f64 {
    if left {
        (3.0 * f(0.0) - 4.0 * f(-e) + f(-2.0 * e)) / (2.0 * e)
    } else {
        (f(e) - f(-e)) / (2.0 * e)
    }
}
fn grid() -> LocalVarianceGrid {
    LocalVarianceGrid::new(
        vec![0.0, 0.15, 0.4, 0.7, 1.0],
        vec![-0.6, -0.2, 0.13, 0.45, 0.8],
        (0..25)
            .map(|i| 0.04 + 0.002 * (i % 5) as f64 + 0.001 * (i / 5) as f64)
            .collect(),
        1e-8,
        4.0,
    )
    .unwrap()
}
fn particles(trace: bool) -> LsvParticleConfig {
    LsvParticleConfig::new(128, 429, 0.5, 3.0, trace).unwrap()
}
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() <= t * (1.0 + b.abs()),
        "{a:.15e} vs {b:.15e} diff {:.6e}",
        (a - b).abs()
    );
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn payload(qmc: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(100.0);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    let g = grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),
        "shape":[5,5],"values":g.values(),"floor":1e-8,"cap":4.0}});
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":4,
        "master_scramble_seed":819,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":128,"master_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    };
    v
}
fn request(qmc: bool) -> PricingRequest {
    parse_request_json(
        &serde_json::to_vec(&payload(qmc)).unwrap(),
        JsonLimits::DEFAULT,
    )
    .unwrap()
}

fn lsv(req: &PricingRequest, p: [f64; 7], workers: u32, trace: bool) -> RoughFamilyLsvPricingPlan {
    RoughFamilyLsvPricingPlan::compile(
        req,
        model(p),
        particles(trace),
        ExecutionPolicy::new(workers, None).unwrap(),
    )
    .unwrap()
}
fn pure(req: &PricingRequest, p: [f64; 7], workers: u32) -> RoughVolatilityPricingPlan {
    RoughVolatilityPricingPlan::compile(
        req,
        model(p),
        0.25,
        ExecutionPolicy::new(workers, None).unwrap(),
    )
    .unwrap()
}
fn pure_request(qmc: bool) -> PricingRequest {
    let mut v = payload(qmc);
    v["model"] = json!({"type":"black_scholes","volatility":0.2});
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

#[test]
fn all_path_cotangents_and_hurst_match_two_widths() {
    let t = vec![0., 0.07, 0.21, 0.63, 1.];
    let fs = vec![0.1, -0.2, 0.3, 0.2, 0.7];
    let vs = vec![0.2, 0.4, -0.2, 0.6, 0.1];
    let mut count = 0;
    let mut gap = 0.0_f64;
    for h in [0.03, 0.1, 0.3, 0.5] {
        let mut p = PARAMETERS;
        p[6] = h;
        let path = RoughVolatilityPathPlan::compile(model(p), t.clone()).unwrap();
        let risk = QuadraticHestonMcRiskPlan::compile(&path, true).unwrap();
        for i in 0..16 {
            let z = path.pseudo_shocks(91, i, RandomDomain::Valuation);
            let record = risk.evolve_path(100., &z).unwrap();
            assert_eq!(record.path(), &path.evolve_path(100., &z).unwrap());
            let a = record.reverse(&fs, &vs).unwrap();
            let no_h = QuadraticHestonMcRiskPlan::compile(&path, false).unwrap();
            assert_eq!(
                &*no_h
                    .evolve_path(100., &z)
                    .unwrap()
                    .reverse(&fs, &vs)
                    .unwrap()
                    .parameters,
                &a.parameters[..6]
            );
            for (j, &adj) in a.parameters.iter().enumerate() {
                for e in [2e-7, 1e-7] {
                    let f = |d| {
                        let q =
                            RoughVolatilityPathPlan::compile(model(shifted(p, j, d)), t.clone())
                                .unwrap()
                                .evolve_path(100., &z)
                                .unwrap();
                        dot(&fs, &q.forwards) + dot(&vs, &q.variances)
                    };
                    let fd = difference(f, e, j == 6 && h == 0.5);
                    close(adj, fd, 3e-5);
                    gap = gap.max((adj - fd).abs());
                    count += 1;
                }
            }
            close(
                a.initial_forward,
                dot(&fs, &record.path().forwards) / 100.,
                1e-12,
            );
        }
    }
    println!("quadratic_path comparisons={count} max_gap={gap:.15e}");
}
#[test]
fn particle_transpose_and_rebuilt_prices_include_recalibration() {
    let seeds = (0..25).map(|i| (i as f64 * 0.4).cos()).collect::<Vec<_>>();
    let mut cal_gap = 0.0_f64;
    let mut price_gap = 0.0_f64;
    let mut nc = 0;
    let mut np = 0;
    for h in [0.1, 0.3, 0.5] {
        let mut p = PARAMETERS;
        p[6] = h;
        let build =
            |pp| calibrate_rough_family_lsv(&grid(), model(pp), 100., particles(true)).unwrap();
        let c = build(p);
        let a = c.reverse_quadratic_heston_parameters(&seeds, true).unwrap();
        let donors = c
            .conditional_moments()
            .iter()
            .map(|m| m.source_node)
            .collect::<Vec<_>>();
        assert!(c.conditional_moments().iter().any(|m| m.extrapolated));
        for (j, &adj) in a.iter().enumerate() {
            for e in [2e-7, 1e-7] {
                let f = |d| {
                    let x = build(shifted(p, j, d));
                    assert_eq!(
                        donors,
                        x.conditional_moments()
                            .iter()
                            .map(|m| m.source_node)
                            .collect::<Vec<_>>()
                    );
                    dot(&seeds, x.surface().squared_leverage())
                };
                let fd = difference(f, e, j == 6 && h == 0.5);
                close(adj, fd, 3e-5);
                cal_gap = cal_gap.max((adj - fd).abs());
                nc += 1;
            }
        }
        for qmc in [false, true] {
            let req = request(qmc);
            let pr = pure_request(qmc);
            let l = lsv(&req, p, 1, true);
            let v = pure(&pr, p, 1);
            let a = l.evaluate_quadratic_heston_parameter_risk(true).unwrap();
            let b = v.evaluate_quadratic_heston_parameter_risk(true).unwrap();
            assert_eq!(a.price, l.evaluate().unwrap());
            assert_eq!(b.price, v.evaluate().unwrap());
            assert_eq!(a.standard_errors.is_some(), qmc);
            assert_eq!(
                &*a.parameter_names,
                &[
                    "initial_state",
                    "mean_reversion",
                    "vol_of_vol",
                    "quadratic",
                    "shift",
                    "variance_floor",
                    "hurst"
                ]
            );
            let old = l.evaluate_quadratic_heston_parameter_risk(false).unwrap();
            assert_eq!(&*old.parameter_adjoints, &a.parameter_adjoints[..6]);
            assert_ne!(old.risk_fingerprint, a.risk_fingerprint);
            for j in 0..7 {
                close(
                    a.parameter_adjoints[j],
                    a.direct_adjoints[j] + a.calibration_adjoints[j],
                    1e-12,
                );
                for e in [1e-6, 5e-7] {
                    let lf = |d| {
                        lsv(&req, shifted(p, j, d), 1, false)
                            .evaluate()
                            .unwrap()
                            .value
                    };
                    let vf = |d| pure(&pr, shifted(p, j, d), 1).evaluate().unwrap().value;
                    for (adj, fd) in [
                        (
                            a.parameter_adjoints[j],
                            difference(lf, e, j == 6 && h == 0.5),
                        ),
                        (
                            b.parameter_adjoints[j],
                            difference(vf, e, j == 6 && h == 0.5),
                        ),
                    ] {
                        close(adj, fd, 3e-5);
                        np += 1;
                        price_gap = price_gap.max((adj - fd).abs());
                    }
                }
            }
            let workers = lsv(&req, p, 3, true)
                .evaluate_quadratic_heston_parameter_risk(true)
                .unwrap();
            assert_eq!(a.parameter_adjoints, workers.parameter_adjoints);
            assert_eq!(a.standard_errors, workers.standard_errors);
            let pw = pure(&pr, p, 3)
                .evaluate_quadratic_heston_parameter_risk(true)
                .unwrap();
            assert_eq!(b.parameter_adjoints, pw.parameter_adjoints);
            assert_eq!(b.standard_errors, pw.standard_errors);
        }
    }
    println!("quadratic_calibration comparisons={nc} max_gap={cal_gap:.15e}");
    println!("quadratic_prices comparisons={np} max_gap={price_gap:.15e}");
}
#[test]
fn boundaries_and_zero_variance_remain_explicit() {
    let t = vec![0., 0.3, 1.];
    let wrong = RoughVolatilityPathPlan::compile(
        RoughHeston::new(0.1, 0.04, 0.7, 0.04, 0.2, -0.6)
            .unwrap()
            .into(),
        t.clone(),
    )
    .unwrap();
    assert!(QuadraticHestonMcRiskPlan::compile(&wrong, true).is_err());
    for j in [1, 2, 3, 5] {
        let mut p = PARAMETERS;
        p[j] = 0.;
        let path = RoughVolatilityPathPlan::compile(model(p), t.clone()).unwrap();
        let r = QuadraticHestonMcRiskPlan::compile(&path, true).unwrap();
        let z = vec![0.1, -0.2, 0.05, -0.07];
        let a = r
            .evolve_path(100., &z)
            .unwrap()
            .reverse(&[0., 0., 1.], &[0.; 3])
            .unwrap();
        let f = |e| {
            RoughVolatilityPathPlan::compile(model(shifted(p, j, e)), t.clone())
                .unwrap()
                .evolve_path(100., &z)
                .unwrap()
                .forwards[2]
        };
        for e in [2e-7, 1e-7] {
            let fd = (-3. * f(0.) + 4. * f(e) - f(2. * e)) / (2. * e);
            close(a.parameters[j], fd, 3e-5);
        }
        if j == 3 {
            assert!(a.parameters[3].abs() > 0.01);
        }
    }
    let mut p = PARAMETERS;
    p[3] = 0.;
    p[5] = 0.;
    let path = RoughVolatilityPathPlan::compile(model(p), t.clone()).unwrap();
    let r = QuadraticHestonMcRiskPlan::compile(&path, true).unwrap();
    assert!(path.evolve_path(100., &[0.; 4]).is_ok());
    assert!(r.evolve_path(100., &[0.; 4]).is_err());
    let p = RoughVolatilityPathPlan::compile(model(PARAMETERS), t).unwrap();
    let r = QuadraticHestonMcRiskPlan::compile(&p, true).unwrap();
    assert!(r.evolve_path(100., &[0.]).is_err());
    let q = r.evolve_path(100., &[0.; 4]).unwrap();
    assert!(q.reverse(&[0.], &[0.]).is_err());
    assert!(q.reverse(&[0., f64::NAN, 0.], &[0.; 3]).is_err());
    assert!(
        lsv(&request(false), PARAMETERS, 1, false)
            .evaluate_quadratic_heston_parameter_risk(true)
            .is_err()
    );
}
#[test]
fn deterministic_fixed_target_risks_cancel_except_vol_of_vol() {
    let mut maximum = 0.0_f64;
    for h in [0.1, 0.5] {
        let mut p = PARAMETERS;
        p[2] = 0.;
        p[6] = h;
        let r = lsv(&request(true), p, 1, true)
            .evaluate_quadratic_heston_parameter_risk(true)
            .unwrap();
        assert!(r.direct_adjoints[5].abs() > 1.);
        for j in [0, 1, 3, 4, 5, 6] {
            maximum = maximum.max(r.parameter_adjoints[j].abs());
            close(r.parameter_adjoints[j], 0., 1e-9);
            close(r.standard_errors.as_ref().unwrap()[j], 0., 1e-9);
        }
    }
    println!("quadratic_deterministic_cancellation max_gap={maximum:.15e}");
}

#[test]
#[ignore = "independent nondegenerate two-step QRH expectation panel"]
fn independent_two_step_parameter_expectations() {
    let data: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/quadratic-parameter.json"
    ))
    .unwrap();
    for row in data["rows"].as_array().unwrap() {
        let p: [f64; 7] = row["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        for seed in [91, 1973] {
            let mut v = payload(true);
            v["market"]["discrete_dividends"] = json!([]);
            v["market"]["discount_curve"]["discount_factors"] = json!([1., 1.]);
            v["market"]["dividend_curve"]["discount_factors"] = json!([1., 1.]);
            v["model"] = json!({"type":"black_scholes","volatility":0.2});
            v["engine"]["master_scramble_seed"] = json!(seed);
            v["engine"]["points_per_scramble"] = json!(4096);
            v["engine"]["scramble_count"] = json!(8);
            let req =
                parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
            let plan = RoughVolatilityPricingPlan::compile(
                &req,
                model(p),
                0.5,
                ExecutionPolicy::new(1, None).unwrap(),
            )
            .unwrap();
            assert_eq!(plan.time_nodes(), &[0., 0.5, 1.]);
            let risk = plan.evaluate_quadratic_heston_parameter_risk(true).unwrap();
            let slack = data["mc_slack"].as_f64().unwrap();
            let cap = data["mc_se_cap"].as_f64().unwrap();
            assert!(
                (risk.price.value - row["price"].as_f64().unwrap()).abs()
                    <= 5. * risk.price.standard_error + slack
            );
            for (j, name) in risk.parameter_names.iter().enumerate() {
                let reference = row["adjoints"][j].as_f64().unwrap();
                let got = risk.parameter_adjoints[j];
                let se = risk.standard_errors[j];
                println!(
                    "quadratic_expectation H={} seed={seed} parameter={name} got={got:.15e} reference={reference:.15e} se={se:.15e}",
                    p[6]
                );
                assert!(se > 0. && se <= cap, "{name}: SE={se}");
                assert!(
                    (got - reference).abs() <= 5. * se + slack,
                    "{name}: {got} vs {reference}, SE={se}"
                );
            }
        }
    }
}

#[test]
fn brownian_endpoint_retains_hurst_residual_derivative() {
    let mut p = PARAMETERS;
    p[6] = 0.5;
    let dt = 0.5_f64;
    let path = RoughVolatilityPathPlan::compile(model(p), vec![0., dt]).unwrap();
    let risk = QuadraticHestonMcRiskPlan::compile(&path, true).unwrap();
    let z_plus = [0., 1.];
    let z_minus = [0., -1.];
    let plus = risk.evolve_path(100., &z_plus).unwrap();
    let minus = risk.evolve_path(100., &z_minus).unwrap();
    assert_eq!(plus.path(), minus.path());
    let a = plus.reverse(&[0.; 2], &[0., 1.]).unwrap();
    let b = minus.reverse(&[0.; 2], &[0., 1.]).unwrap();
    let expected = -4.
        * p[3]
        * (plus.path().latent_states[1] - p[4])
        * p[1]
        * p[2]
        * plus.path().variances[0].sqrt()
        * dt.sqrt();
    assert!(expected.abs() > 1e-4);
    close(a.parameters[6] - b.parameters[6], expected, 1e-13);
}
