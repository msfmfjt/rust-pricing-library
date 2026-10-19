//! Simplex weight transfers and ORIGINAL forward-variance inputs; fixed H/eta/rho.
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};
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

fn pure_request(qmc: bool) -> PricingRequest {
    let mut v = payload(qmc);
    v["model"] = json!({"type":"black_scholes","volatility":0.2});
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn inputs(exponential: bool) -> Vec<f64> {
    if exponential {
        vec![0.2, 0.35, 0.04, 0.2]
    } else {
        vec![0.2, 0.35, 0.04, 0.05, 0.045, 0.055]
    }
}
fn model(h: f64, p: &[f64], exponential: bool) -> RoughVolatilityModel {
    let xi = if exponential {
        ForwardVarianceCurve::exponential(p[2], p[3]).unwrap()
    } else {
        ForwardVarianceCurve::piecewise_linear(vec![0., 0.4, 0.9, 1.4], p[2..].to_vec()).unwrap()
    };
    MixedRoughBergomi::new(
        h,
        -0.6,
        vec![p[0], p[1], 1. - p[0] - p[1]],
        vec![0.3, 0.8, 1.1],
        xi,
    )
    .unwrap()
    .into()
}
fn lsv(
    req: &PricingRequest,
    m: RoughVolatilityModel,
    workers: u32,
    trace: bool,
) -> RoughFamilyLsvPricingPlan {
    RoughFamilyLsvPricingPlan::compile(
        req,
        m,
        particles(trace),
        ExecutionPolicy::new(workers, None).unwrap(),
    )
    .unwrap()
}
fn pure(req: &PricingRequest, m: RoughVolatilityModel, workers: u32) -> RoughVolatilityPricingPlan {
    RoughVolatilityPricingPlan::compile(req, m, 0.25, ExecutionPolicy::new(workers, None).unwrap())
        .unwrap()
}
fn bump(p: &[f64], j: usize, e: f64) -> Vec<f64> {
    let mut v = p.to_vec();
    v[j] += e;
    v
}

#[test]
fn shape_paths_original_curve_coordinates_two_widths() {
    let t = vec![0., 0.07, 0.21, 0.63, 1., 1.6];
    let fs = vec![0.1, -0.2, 0.3, 0.2, 0.7, 0.1];
    let vs = vec![0.2, 0.4, -0.2, 0.6, 0.1, 0.2];
    let mut count = 0;
    let mut gap = 0.0_f64;
    for exponential in [false, true] {
        for h in [0.03, 0.1, 0.5] {
            let p = inputs(exponential);
            let b = RoughVolatilityPathPlan::compile(model(h, &p, exponential), t.clone()).unwrap();
            for i in 0..16 {
                let z = b.pseudo_shocks(91, i, RandomDomain::Valuation);
                let record = b.evolve_mixed_bergomi_shape_path(100., &z).unwrap();
                assert_eq!(record.path(), &b.evolve_path(100., &z).unwrap());
                let a = record.reverse(&fs, &vs).unwrap();
                assert_eq!(a.parameters.len(), p.len());
                for j in 0..p.len() {
                    for e in [2e-7, 1e-7] {
                        let f = |e| {
                            let v = RoughVolatilityPathPlan::compile(
                                model(h, &bump(&p, j, e), exponential),
                                t.clone(),
                            )
                            .unwrap()
                            .evolve_path(100., &z)
                            .unwrap();
                            dot(&fs, &v.forwards) + dot(&vs, &v.variances)
                        };
                        let fd = (f(e) - f(-e)) / (2. * e);
                        close(a.parameters[j], fd, 3e-5);
                        gap = gap.max((a.parameters[j] - fd).abs());
                        count += 1;
                    }
                }
            }
        }
    }
    println!("mixed_shape_path comparisons={count} max_gap={gap:.15e}");
}

#[test]
fn shape_leverage_transpose_and_stochastic_xi_scale_cancellation() {
    let seeds = (0..25).map(|i| (i as f64 * 0.4).cos()).collect::<Vec<_>>();
    let mut count = 0;
    let mut gap = 0.0_f64;
    for exponential in [false, true] {
        for h in [0.1, 0.3, 0.5] {
            let p = inputs(exponential);
            let g = grid();
            let build = |q: &[f64]| {
                calibrate_rough_family_lsv(&g, model(h, q, exponential), 100., particles(true))
                    .unwrap()
            };
            let c = build(&p);
            assert!(c.conditional_moments().iter().any(|m| m.extrapolated));
            let a = c.reverse_mixed_bergomi_shape(&seeds).unwrap();
            for (j, &adjoint) in a.iter().enumerate() {
                for e in [2e-7, 1e-7] {
                    let f = |e| {
                        let b = build(&bump(&p, j, e));
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
                    };
                    let fd = (f(e) - f(-e)) / (2. * e);
                    close(adjoint, fd, 3e-5);
                    gap = gap.max((adjoint - fd).abs());
                    count += 1;
                }
            }
        }
    }
    println!("mixed_shape_calibration comparisons={count} max_gap={gap:.15e}");
}

#[test]
fn shape_price_full_recalibration_covariance_metadata_and_workers() {
    let mut count = 0;
    let mut gap = 0.0_f64;
    let mut cancellation = 0.0_f64;
    for exponential in [false, true] {
        for h in [0.1, 0.5] {
            for qmc in [false, true] {
                let p = inputs(exponential);
                let req = request(qmc);
                let preq = pure_request(qmc);
                let b = lsv(&req, model(h, &p, exponential), 1, true);
                let pb = pure(&preq, model(h, &p, exponential), 1);
                let a = b.evaluate_mixed_bergomi_shape_risk().unwrap();
                let pa = pb.evaluate_mixed_bergomi_shape_risk().unwrap();
                assert_eq!(a.price, b.evaluate().unwrap());
                assert_eq!(pa.price, pb.evaluate().unwrap());
                assert_eq!(a.standard_errors.is_some(), qmc);
                assert_eq!(a.parameter_names[0], "weight_transfer[0,2]");
                assert_eq!(a.parameter_names.len(), p.len());
                let old = b
                    .evaluate_mixed_bergomi_parameter_risk_with_hurst()
                    .unwrap();
                assert_ne!(a.risk_fingerprint, old.risk_fingerprint);
                // xi is a deterministic multiplicative scale at each time; the calibrated
                // product ell*V is invariant, EVEN WITH NONZERO component vol-of-vols.
                assert!(a.direct_adjoints[2].abs() > 1.0);
                for j in 2..p.len() {
                    cancellation = cancellation.max(a.parameter_adjoints[j].abs());
                    assert!(a.parameter_adjoints[j].abs() < 1e-8);
                    if let Some(s) = &a.standard_errors {
                        assert!(s[j] < 1e-8);
                    }
                }
                for j in 0..p.len() {
                    close(
                        a.parameter_adjoints[j],
                        a.direct_adjoints[j] + a.calibration_adjoints[j],
                        1e-12,
                    );
                    for e in [1e-6, 5e-7] {
                        let f = |e| {
                            lsv(&req, model(h, &bump(&p, j, e), exponential), 1, false)
                                .evaluate()
                                .unwrap()
                                .value
                        };
                        let pf = |e| {
                            pure(&preq, model(h, &bump(&p, j, e), exponential), 1)
                                .evaluate()
                                .unwrap()
                                .value
                        };
                        for (v, fd) in [
                            (a.parameter_adjoints[j], (f(e) - f(-e)) / (2. * e)),
                            (pa.parameter_adjoints[j], (pf(e) - pf(-e)) / (2. * e)),
                        ] {
                            close(v, fd, 3e-5);
                            gap = gap.max((v - fd).abs());
                            count += 1;
                        }
                    }
                }
                let other = lsv(&req, model(h, &p, exponential), 3, true)
                    .evaluate_mixed_bergomi_shape_risk()
                    .unwrap();
                assert_eq!(a.parameter_adjoints, other.parameter_adjoints);
                assert_eq!(a.standard_errors, other.standard_errors);
                let other = pure(&preq, model(h, &p, exponential), 3)
                    .evaluate_mixed_bergomi_shape_risk()
                    .unwrap();
                assert_eq!(pa.parameter_adjoints, other.parameter_adjoints);
                assert_eq!(pa.standard_errors, other.standard_errors);
                assert_eq!(
                    old,
                    b.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                        .unwrap()
                );
            }
        }
    }
    println!(
        "mixed_shape_price comparisons={count} max_gap={gap:.15e} max_xi_cancellation={cancellation:.15e}"
    );
}

#[test]
fn shape_boundaries_constant_curve_identical_components_and_size_guards() {
    let make = |w, eta, xi, rho| MixedRoughBergomi::new(0.2, rho, w, eta, xi).unwrap().into();
    let xi = ForwardVarianceCurve::constant(0.04).unwrap();
    for rho in [-1., 1.] {
        let p = pure(
            &pure_request(false),
            make(vec![1.], vec![0.7], xi.clone(), rho),
            1,
        );
        let r = p.evaluate_mixed_bergomi_shape_risk().unwrap();
        assert_eq!(r.parameter_names.as_ref(), ["forward_variance[0]"]);
        assert!(p.evaluate_mixed_bergomi_parameter_risk().is_err());
    }
    let p = pure(
        &pure_request(false),
        make(vec![0.2, 0.3, 0.5], vec![0.7; 3], xi.clone(), -0.6),
        1,
    );
    let r = p.evaluate_mixed_bergomi_shape_risk().unwrap();
    for a in &r.parameter_adjoints[..2] {
        assert!(a.abs() < 1e-10);
    }
    for m in [
        make(vec![0., 1.], vec![0.3, 0.8], xi.clone(), -0.6),
        make(
            vec![1.],
            vec![0.3],
            ForwardVarianceCurve::constant(0.).unwrap(),
            -0.6,
        ),
        RoughHeston::new(0.1, 0.04, 0.7, 0.055, 0.15, -0.6)
            .unwrap()
            .into(),
    ] {
        let p = pure(&pure_request(false), m, 1);
        assert!(p.evaluate_mixed_bergomi_shape_risk().is_err());
        assert!(p.evaluate().is_ok());
    }
    let curve = ForwardVarianceCurve::piecewise_linear(
        (0..4097).map(|i| i as f64).collect(),
        vec![0.04; 4097],
    )
    .unwrap();
    let p = pure(
        &pure_request(false),
        make(vec![1.], vec![0.3], curve, -0.6),
        1,
    );
    assert!(p.evaluate_mixed_bergomi_shape_risk().is_err());
    assert!(
        lsv(&request(false), model(0.1, &inputs(false), false), 1, false)
            .evaluate_mixed_bergomi_shape_risk()
            .is_err()
    );
    let b =
        RoughVolatilityPathPlan::compile(model(0.1, &inputs(false), false), vec![0., 1.]).unwrap();
    let z = b.pseudo_shocks(91, 0, RandomDomain::Valuation);
    let rec = b.evolve_mixed_bergomi_shape_path(100., &z).unwrap();
    assert!(rec.reverse(&[0.], &[0.]).is_err());
    assert!(rec.reverse(&[0., f64::NAN], &[0., 0.]).is_err());
}

#[test]
#[ignore = "extended independent constant-variance Black shape reference"]
fn shape_constant_variance_black_reference() {
    let reference = 99.23813686925295_f64;
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
            0.1,
            -0.6,
            vec![0.35, 0.65],
            vec![0., 0.],
            ForwardVarianceCurve::constant(0.04).unwrap(),
        )
        .unwrap();
        let a = pure(&req, m.into(), 1)
            .evaluate_mixed_bergomi_shape_risk()
            .unwrap();
        assert!(a.parameter_adjoints[0].abs() < 1e-10);
        let risk = a.parameter_adjoints[1];
        let se = a.standard_errors[1];
        assert!((risk - reference).abs() <= 5. * se + 0.02);
        assert!(se > 0. && se <= 0.1);
        assert!((a.price.value - 7.965567455405804).abs() <= 5. * a.price.standard_error + 0.002);
        println!(
            "mixed_shape_black seed={seed} xi_risk={risk:.15e} se={se:.15e} ref={reference:.15e}"
        );
    }
}

#[test]
#[ignore = "independent nondegenerate two-step conditional-Black shape law"]
fn shape_independent_two_step_expectations() {
    let data: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/mixed-shape.json"
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
            let r = p.evaluate_mixed_bergomi_shape_risk().unwrap();
            let price_ref = row["price"].as_f64().unwrap();
            assert!((r.price.value - price_ref).abs() <= 5. * r.price.standard_error + 0.003);
            for (j, a) in r.parameter_adjoints.iter().enumerate() {
                let reference = row["parameter_adjoints"][j].as_f64().unwrap();
                let se = r.standard_errors[j];
                assert!(
                    (a - reference).abs() <= 5. * se + 0.003,
                    "H={h} j={j} a={a} ref={reference} se={se}"
                );
                if j == 3 {
                    assert_eq!(*a, 0.);
                    assert_eq!(se, 0.);
                } else {
                    assert!(se > 0. && se <= 0.2);
                }
                println!(
                    "mixed_shape_expectation h={h} seed={seed} j={j} value={a:.15e} ref={reference:.15e} se={se:.15e}"
                );
            }
        }
    }
}
