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
fn derivative(h: f64, e: f64, f: impl Fn(f64) -> f64) -> f64 {
    if h == 0.5 {
        (3.0 * f(h) - 4.0 * f(h - e) + f(h - 2.0 * e)) / (2.0 * e)
    } else {
        (f(h + e) - f(h - e)) / (2.0 * e)
    }
}
fn exact_prefix(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len() + 1);
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
#[test]
fn mixed_hurst_paths_match_two_widths_and_preserve_scalar_prefix() {
    let t = vec![0., 0.07, 0.21, 0.63, 1.];
    let params = [0.3, 0.8, -0.6];
    let fs = vec![0.1, -0.2, 0.3, 0.2, 0.7];
    let vs = vec![0.2, 0.4, -0.2, 0.6, 0.1];
    let mut count = 0;
    let mut max_gap = 0.0_f64;
    for h in [0.001, 0.03, 0.1, 0.3, 0.5] {
        let base = RoughVolatilityPathPlan::compile(model(h, params), t.clone()).unwrap();
        let plan = MixedBergomiMcHurstPlan::compile(&base).unwrap();
        for i in 0..16 {
            let z = base.pseudo_shocks(91, i, RandomDomain::Valuation);
            let old = base.evolve_mixed_bergomi_parameter_path(100., &z).unwrap();
            let old_risk = old.reverse(&fs, &vs).unwrap();
            let record = plan.evolve_path(100., &z).unwrap();
            assert_eq!(old.path(), record.path());
            let risk = record.reverse(&fs, &vs).unwrap();
            assert_eq!(
                risk.initial_forward.to_bits(),
                old_risk.initial_forward.to_bits()
            );
            exact_prefix(&risk.parameters, &old_risk.parameters);
            for e in [2e-7, 1e-7] {
                let fd = derivative(h, e, |hh| {
                    let p = RoughVolatilityPathPlan::compile(model(hh, params), t.clone())
                        .unwrap()
                        .evolve_path(100., &z)
                        .unwrap();
                    dot(&fs, &p.forwards) + dot(&vs, &p.variances)
                });
                close(risk.parameters[3], fd, 3e-5);
                max_gap = max_gap.max((risk.parameters[3] - fd).abs());
                count += 1;
            }
        }
    }
    println!("mixed_hurst_path comparisons={count} max_gap={max_gap:.15e}");
}
#[test]
fn mixed_hurst_calibration_differentiates_earlier_particles_and_centering() {
    let params = [0.3, 0.8, -0.6];
    let seeds = (0..25).map(|i| (i as f64 * 0.4).cos()).collect::<Vec<_>>();
    let mut max_gap = 0.0_f64;
    for h in [0.1, 0.3, 0.5] {
        let g = grid();
        let build =
            |hh| calibrate_rough_family_lsv(&g, model(hh, params), 100., particles(true)).unwrap();
        let c = build(h);
        assert!(c.conditional_moments().iter().any(|m| m.extrapolated));
        let old = c.reverse_mixed_bergomi_parameters(&seeds).unwrap();
        let a = c
            .reverse_mixed_bergomi_parameters_with_hurst(&seeds)
            .unwrap();
        exact_prefix(&a, &old);
        for e in [2e-7, 1e-7] {
            let fd = derivative(h, e, |hh| {
                let b = build(hh);
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
                dot(&seeds, b.surface().squared_leverage())
            });
            close(a[3], fd, 3e-5);
            max_gap = max_gap.max((a[3] - fd).abs());
        }
    }
    println!("mixed_hurst_calibration comparisons=6 max_gap={max_gap:.15e}");
}
#[test]
fn mixed_hurst_prices_full_recalibration_workers_and_old_prefix() {
    let params = [0.3, 0.8, -0.6];
    let mut max_gap = 0.0_f64;
    for h in [0.1, 0.3, 0.5] {
        for qmc in [false, true] {
            let req = request(qmc);
            let preq = pure_request(qmc);
            let p = lsv(&req, h, params, 1, true);
            let pp = pure(&preq, h, params, 1);
            let r = p
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap();
            let pr = pp
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap();
            let old = p.evaluate_mixed_bergomi_parameter_risk().unwrap();
            let pold = pp.evaluate_mixed_bergomi_parameter_risk().unwrap();
            assert_eq!(r.parameter_names.last().unwrap(), "hurst");
            assert_eq!(r.price, old.price);
            assert_eq!(pr.price, pold.price);
            assert_ne!(r.risk_fingerprint, old.risk_fingerprint);
            assert_ne!(pr.risk_fingerprint, pold.risk_fingerprint);
            assert_eq!(r.standard_errors.is_some(), qmc);
            exact_prefix(&r.parameter_adjoints, &old.parameter_adjoints);
            exact_prefix(&r.direct_adjoints, &old.direct_adjoints);
            exact_prefix(&r.calibration_adjoints, &old.calibration_adjoints);
            exact_prefix(&pr.parameter_adjoints, &pold.parameter_adjoints);
            exact_prefix(&pr.standard_errors, &pold.standard_errors);
            if let (Some(a), Some(b)) = (&r.standard_errors, &old.standard_errors) {
                exact_prefix(a, b);
            }
            close(
                r.parameter_adjoints[3],
                r.direct_adjoints[3] + r.calibration_adjoints[3],
                1e-12,
            );
            for e in [1e-6, 5e-7] {
                let fd = derivative(h, e, |hh| {
                    lsv(&req, hh, params, 1, false).evaluate().unwrap().value
                });
                let pfd = derivative(h, e, |hh| {
                    pure(&preq, hh, params, 1).evaluate().unwrap().value
                });
                close(r.parameter_adjoints[3], fd, 3e-5);
                close(pr.parameter_adjoints[3], pfd, 3e-5);
                max_gap = max_gap
                    .max((r.parameter_adjoints[3] - fd).abs())
                    .max((pr.parameter_adjoints[3] - pfd).abs());
            }
            let other = lsv(&req, h, params, 3, true)
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap();
            assert_eq!(other.parameter_adjoints, r.parameter_adjoints);
            assert_eq!(other.standard_errors, r.standard_errors);
            let other = pure(&preq, h, params, 3)
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap();
            assert_eq!(other.parameter_adjoints, pr.parameter_adjoints);
            assert_eq!(other.standard_errors, pr.standard_errors);
        }
    }
    println!("mixed_hurst_prices comparisons=24 max_gap={max_gap:.15e}");
}
#[test]
fn mixed_hurst_brownian_residual_and_domain_controls() {
    let m = MixedRoughBergomi::new(
        0.5,
        -0.6,
        vec![1.],
        vec![0.7],
        ForwardVarianceCurve::constant(0.04).unwrap(),
    )
    .unwrap();
    let base = RoughVolatilityPathPlan::compile(m.into(), vec![0., 1.]).unwrap();
    let plan = MixedBergomiMcHurstPlan::compile(&base).unwrap();
    let z = [0., 0., 1.];
    let r = plan.evolve_path(100., &z).unwrap();
    let g = r.reverse(&[0., 0.], &[0., 1.]).unwrap();
    let expected = -0.7 * 0.04 * (-0.5 * 0.7_f64.powi(2)).exp();
    close(g.parameters[2], expected, 1e-13);
    assert!(
        g.parameters[2].abs() > 0.01,
        "detect dropped residual left derivative"
    );
    assert!(r.reverse(&[0.], &[0.]).is_err());
    assert!(r.reverse(&[0., f64::NAN], &[0., 0.]).is_err());
    assert!(plan.evolve_path(100., &[0.]).is_err());
    let req = pure_request(true);
    for h in [0.001, 0.1, 0.5] {
        let p = pure(&req, h, [0., 0., -0.6], 1);
        assert_eq!(
            p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap()
                .parameter_adjoints[3],
            0.
        );
        let p = lsv(&request(true), h, [0., 0., -0.6], 1, true);
        assert_eq!(
            p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap()
                .parameter_adjoints[3],
            0.
        );
    }
    for rho in [-1., 1.] {
        let p =
            RoughVolatilityPathPlan::compile(model(0.2, [0.3, 0.8, rho]), vec![0., 1.]).unwrap();
        assert!(MixedBergomiMcHurstPlan::compile(&p).is_err());
    }
    let wrong = RoughHeston::new(0.2, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
    let p = RoughVolatilityPathPlan::compile(wrong.into(), vec![0., 1.]).unwrap();
    assert!(MixedBergomiMcHurstPlan::compile(&p).is_err());
    assert!(
        lsv(&request(true), 0.2, [0.3, 0.8, -0.6], 1, false)
            .evaluate_mixed_bergomi_parameter_risk_with_hurst()
            .is_err()
    );
}
#[test]
#[ignore = "extended independent two-step conditional-Black reference"]
fn independent_two_step_hurst_expectations() {
    let data: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/mixed-hurst.json"
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
            let r = p
                .evaluate_mixed_bergomi_parameter_risk_with_hurst()
                .unwrap();
            assert!(
                (r.price.value - row["price"].as_f64().unwrap()).abs()
                    <= 5. * r.price.standard_error + 0.002
            );
            let a = r.parameter_adjoints[3];
            let reference = row["hurst_sensitivity"].as_f64().unwrap();
            let se = r.standard_errors[3];
            assert!(se > 0. && se <= 0.1);
            assert!(
                (a - reference).abs() <= 5. * se + 0.002,
                "H={h}: {a} ref={reference} SE={se}"
            );
            println!(
                "mixed_hurst_oracle H={h} seed={seed} estimate={a:.15e} reference={reference:.15e} SE={se:.15e}"
            );
        }
    }
}
