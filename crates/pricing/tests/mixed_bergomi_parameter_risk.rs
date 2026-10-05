//! Finite-scheme eta/rho derivatives; targets, weights, H and xi are fixed.
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn model(h: f64, p: [f64; 3]) -> RoughVolatilityModel {
    MixedRoughBergomi::new(
        h,
        p[2],
        vec![0.35, 0.65],
        p[..2].to_vec(),
        ForwardVarianceCurve::piecewise_linear(vec![0., 0.4, 1.], vec![0.04, 0.05, 0.045]).unwrap(),
    )
    .unwrap()
    .into()
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

fn lsv(
    req: &PricingRequest,
    h: f64,
    p: [f64; 3],
    workers: u32,
    trace: bool,
) -> RoughFamilyLsvPricingPlan {
    RoughFamilyLsvPricingPlan::compile(
        req,
        model(h, p),
        particles(trace),
        ExecutionPolicy::new(workers, None).unwrap(),
    )
    .unwrap()
}
fn pure(req: &PricingRequest, h: f64, p: [f64; 3], workers: u32) -> RoughVolatilityPricingPlan {
    RoughVolatilityPricingPlan::compile(
        req,
        model(h, p),
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
fn path_all_eta_and_rho_cotangents_match_two_bumps() {
    let t = vec![0., 0.07, 0.21, 0.63, 1.];
    let p = [0.3, 0.8, -0.6];
    let fs = vec![0.1, -0.2, 0.3, 0.2, 0.7];
    let vs = vec![0.2, 0.4, -0.2, 0.6, 0.1];
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for h in [0.03, 0.1, 0.3, 0.5] {
        let plan = RoughVolatilityPathPlan::compile(model(h, p), t.clone()).unwrap();
        for i in 0..16 {
            let z = plan.pseudo_shocks(91, i, RandomDomain::Valuation);
            let record = plan.evolve_mixed_bergomi_parameter_path(100., &z).unwrap();
            assert_eq!(record.path(), &plan.evolve_path(100., &z).unwrap());
            let a = record.reverse(&fs, &vs).unwrap();
            for (j, &adj) in a.parameters.iter().enumerate() {
                for e in [2e-7, 1e-7] {
                    let f = |shift: f64| {
                        let mut pp = p;
                        pp[j] += shift;
                        let r = RoughVolatilityPathPlan::compile(model(h, pp), t.clone())
                            .unwrap()
                            .evolve_path(100., &z)
                            .unwrap();
                        dot(&fs, &r.forwards) + dot(&vs, &r.variances)
                    };
                    let fd = (f(e) - f(-e)) / (2. * e);
                    close(adj, fd, 2e-5);
                    count += 1;
                    maximum = maximum.max((adj - fd).abs());
                }
            }
            close(
                a.initial_forward,
                dot(&fs, &record.path().forwards) / 100.,
                1e-12,
            );
        }
    }
    println!("mixed_path comparisons={count} max_gap={maximum:.15e}");
}
#[test]
fn calibration_eta_and_rho_include_donors_and_particle_positions() {
    let g = grid();
    let p = [0.3, 0.8, -0.6];
    let seeds = (0..25).map(|i| (i as f64 * 0.4).cos()).collect::<Vec<_>>();
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for h in [0.1, 0.3, 0.5] {
        let build =
            |pp| calibrate_rough_family_lsv(&g, model(h, pp), 100., particles(true)).unwrap();
        let c = build(p);
        let a = c.reverse_mixed_bergomi_parameters(&seeds).unwrap();
        for (j, &adj) in a.iter().enumerate() {
            for e in [2e-7, 1e-7] {
                let mut up = p;
                up[j] += e;
                let mut dn = p;
                dn[j] -= e;
                let u = build(up);
                let d = build(dn);
                for b in [&u, &d] {
                    assert_eq!(
                        c.conditional_moments()
                            .iter()
                            .map(|m| m.source_node)
                            .collect::<Vec<_>>(),
                        b.conditional_moments()
                            .iter()
                            .map(|m| m.source_node)
                            .collect::<Vec<_>>()
                    );
                }
                let fd = (dot(&seeds, u.surface().squared_leverage())
                    - dot(&seeds, d.surface().squared_leverage()))
                    / (2. * e);
                close(adj, fd, 3e-5);
                count += 1;
                maximum = maximum.max((adj - fd).abs());
            }
        }
    }
    println!("mixed_calibration comparisons={count} max_gap={maximum:.15e}");
}
#[test]
fn pure_and_recalibrated_lsv_prices_match_two_bumps() {
    let p = [0.3, 0.8, -0.6];
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for h in [0.1, 0.3, 0.5] {
        for qmc in [false, true] {
            let req = request(qmc);
            let pr = pure_request(qmc);
            let b = lsv(&req, h, p, 1, true);
            let r = b.evaluate_mixed_bergomi_parameter_risk().unwrap();
            let v = pure(&pr, h, p, 1);
            let vr = v.evaluate_mixed_bergomi_parameter_risk().unwrap();
            assert_eq!(r.price, b.evaluate().unwrap());
            assert_eq!(vr.price, v.evaluate().unwrap());
            assert_eq!(
                &*r.parameter_names,
                &["vol_of_vol[0]", "vol_of_vol[1]", "correlation"]
            );
            assert_eq!(r.standard_errors.is_some(), qmc);
            for (j, &a) in r.parameter_adjoints.iter().enumerate() {
                close(a, r.direct_adjoints[j] + r.calibration_adjoints[j], 1e-12);
                for e in [1e-6, 5e-7] {
                    let mut up = p;
                    up[j] += e;
                    let mut dn = p;
                    dn[j] -= e;
                    let fd = (lsv(&req, h, up, 1, false).evaluate().unwrap().value
                        - lsv(&req, h, dn, 1, false).evaluate().unwrap().value)
                        / (2. * e);
                    let pfd = (pure(&pr, h, up, 1).evaluate().unwrap().value
                        - pure(&pr, h, dn, 1).evaluate().unwrap().value)
                        / (2. * e);
                    close(a, fd, 3e-5);
                    close(vr.parameter_adjoints[j], pfd, 3e-5);
                    count += 2;
                    maximum = maximum
                        .max((a - fd).abs())
                        .max((vr.parameter_adjoints[j] - pfd).abs());
                }
            }
            let rr = lsv(&req, h, p, 3, true)
                .evaluate_mixed_bergomi_parameter_risk()
                .unwrap();
            assert_eq!(r.parameter_adjoints, rr.parameter_adjoints);
            assert_eq!(r.standard_errors, rr.standard_errors);
            let rr = pure(&pr, h, p, 3)
                .evaluate_mixed_bergomi_parameter_risk()
                .unwrap();
            assert_eq!(vr.parameter_adjoints, rr.parameter_adjoints);
            assert_eq!(vr.standard_errors, rr.standard_errors);
        }
    }
    println!("mixed_prices comparisons={count} max_gap={maximum:.15e}");
}
#[test]
fn explicit_boundaries_and_zero_weight_eta() {
    let t = vec![0., 0.3, 1.];
    let p = [0.3, 0.8, -0.6];
    let plan = RoughVolatilityPathPlan::compile(model(0.2, p), t.clone()).unwrap();
    let z = plan.pseudo_shocks(91, 0, RandomDomain::Valuation);
    let path = plan.evolve_mixed_bergomi_parameter_path(100., &z).unwrap();
    assert!(path.reverse(&[0.], &[0.]).is_err());
    assert!(path.reverse(&[0., f64::NAN, 0.], &[0.; 3]).is_err());
    let single = MixedRoughBergomi::new(
        0.2,
        -0.6,
        vec![1., 0.],
        vec![0.3, 2.],
        ForwardVarianceCurve::constant(0.04).unwrap(),
    )
    .unwrap();
    let s = RoughVolatilityPathPlan::compile(single.into(), t.clone()).unwrap();
    let a = s
        .evolve_mixed_bergomi_parameter_path(100., &z)
        .unwrap()
        .reverse(&[0., 0., 1.], &[0.; 3])
        .unwrap();
    assert_eq!(a.parameters[1], 0.);
    let zero = MixedRoughBergomi::new(
        0.2,
        -0.6,
        vec![1.],
        vec![0.3],
        ForwardVarianceCurve::constant(0.).unwrap(),
    )
    .unwrap();
    let s = RoughVolatilityPathPlan::compile(zero.into(), t.clone()).unwrap();
    assert_eq!(
        &*s.evolve_mixed_bergomi_parameter_path(100., &z)
            .unwrap()
            .reverse(&[0., 0., 1.], &[1.; 3])
            .unwrap()
            .parameters,
        &[0., 0.]
    );
    for rho in [-1., 1.] {
        let s = RoughVolatilityPathPlan::compile(model(0.2, [0.3, 0.8, rho]), t.clone()).unwrap();
        assert!(s.evolve_path(100., &z).is_ok());
        assert!(s.evolve_mixed_bergomi_parameter_path(100., &z).is_err());
    }
    assert!(
        lsv(&request(false), 0.2, p, 1, false)
            .evaluate_mixed_bergomi_parameter_risk()
            .is_err()
    );
    let wrong = RoughHeston::new(0.2, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
    let wrong = RoughVolatilityPathPlan::compile(wrong.into(), t).unwrap();
    assert!(wrong.evolve_mixed_bergomi_parameter_path(100., &z).is_err());
}
#[test]
fn identical_components_split_eta_risk_by_fixed_weights() {
    let weights = [0.35, 0.65];
    let p = [0.5, 0.5, -0.6];
    let r = pure(&pure_request(true), 0.2, p, 1)
        .evaluate_mixed_bergomi_parameter_risk()
        .unwrap();
    let l = lsv(&request(true), 0.2, p, 1, true)
        .evaluate_mixed_bergomi_parameter_risk()
        .unwrap();
    for a in [
        &*r.parameter_adjoints,
        &*l.parameter_adjoints,
        &*l.direct_adjoints,
        &*l.calibration_adjoints,
    ] {
        close(a[0] / weights[0], a[1] / weights[1], 1e-11);
    }
}

#[test]
#[ignore = "extended independent two-step conditional-Black reference"]
fn independent_two_step_parameter_expectations() {
    let data: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/mixed-parameter.json"
    ))
    .unwrap();
    for row in data["rows"].as_array().unwrap() {
        let h = row["hurst"].as_f64().unwrap();
        for seed in [91, 1973] {
            let mut v = payload(true);
            v["market"]["discount_curve"]["discount_factors"] = json!([1., 1.]);
            v["market"]["dividend_curve"]["discount_factors"] = json!([1., 1.]);
            v["market"]
                .as_object_mut()
                .unwrap()
                .remove("discrete_dividends");
            v["model"] = json!({"type":"black_scholes","volatility":0.2});
            v["engine"]["points_per_scramble"] = json!(4096);
            v["engine"]["scramble_count"] = json!(8);
            v["engine"]["master_scramble_seed"] = json!(seed);
            let req =
                parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
            let m = MixedRoughBergomi::new(
                h,
                -0.6,
                vec![0.35, 0.65],
                vec![0.3, 0.8],
                ForwardVarianceCurve::piecewise_linear(vec![0., 0.5, 1.], vec![0.04, 0.05, 0.045])
                    .unwrap(),
            )
            .unwrap();
            let p = RoughVolatilityPricingPlan::compile(
                &req,
                m.into(),
                0.5,
                ExecutionPolicy::new(1, None).unwrap(),
            )
            .unwrap();
            assert_eq!(p.time_nodes(), &[0., 0.5, 1.]);
            let r = p.evaluate_mixed_bergomi_parameter_risk().unwrap();
            assert!(
                (r.price.value - row["price"].as_f64().unwrap()).abs()
                    <= 5. * r.price.standard_error + 0.002
            );
            for (j, &a) in r.parameter_adjoints.iter().enumerate() {
                let reference = row["parameter_adjoints"][j].as_f64().unwrap();
                let se = r.standard_errors[j];
                assert!(se > 0. && se <= 0.05);
                assert!(
                    (a - reference).abs() <= 5. * se + 0.002,
                    "H={h} p={j} {a} ref={reference} se={se}"
                );
                println!(
                    "mixed_oracle H={h} seed={seed} parameter={j} estimate={a:.15e} reference={reference:.15e} SE={se:.15e}"
                );
            }
        }
    }
}
