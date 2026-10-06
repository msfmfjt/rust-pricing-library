//! Finite hybrid-law H derivatives, not continuous-time accuracy certification.
use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};
fn model(h: f64, nu: f64) -> RoughVolatilityModel {
    RoughHeston::new(h, 0.04, 0.7, 0.055, nu, -0.6)
        .unwrap()
        .into()
}
fn payload(qmc: bool, divs: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(100.0);
    if divs {
        v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
            {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}
        ]);
    }
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,
        "scramble_count":4,"master_scramble_seed":819,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":128,"master_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    };
    v
}
fn req(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{a:0.15e} vs {b:0.15e}"
    );
}
fn objective(p: &RoughVolatilityPath, f: &[f64], v: &[f64]) -> f64 {
    p.forwards.iter().zip(f).map(|(a, b)| a * b).sum::<f64>()
        + p.variances.iter().zip(v).map(|(a, b)| a * b).sum::<f64>()
}
#[test]
fn path_hurst_reverse_includes_all_loadings_and_negative_nodes() {
    let ts = vec![0., 0.07, 0.21, 0.63, 1.];
    let f = [0.3, -0.2, 0.15, 0.1, 0.7];
    let v = [0.2, 0.3, -0.7, 0.4, 0.8];
    let mut maxgap: f64 = 0.;
    let mut count = 0;
    let mut negatives = 0;
    for h in [0.03, 0.1, 0.3, 0.5] {
        for nu in [0.15, 0.7] {
            let p = RoughVolatilityPathPlan::compile(model(h, nu), ts.clone()).unwrap();
            let hp = RoughHestonMcHurstPlan::compile(&p).unwrap();
            for i in 0..24 {
                let z = p.pseudo_shocks(91, i, RandomDomain::Valuation);
                let r = hp.evolve_path(100., &z).unwrap();
                assert_eq!(r.path(), &p.evolve_path(100., &z).unwrap());
                negatives += r.path().negative_variance_nodes;
                let g = r.reverse(&f, &v).unwrap();
                let value = objective(r.path(), &f, &v);
                for e in [2e-7, 1e-7] {
                    let eval = |h| {
                        let x = RoughVolatilityPathPlan::compile(model(h, nu), ts.clone())
                            .unwrap()
                            .evolve_path(100., &z)
                            .unwrap();
                        assert_eq!(x.negative_variance_nodes, r.path().negative_variance_nodes);
                        objective(&x, &f, &v)
                    };
                    let fd = if h == 0.5 {
                        (3. * value - 4. * eval(h - e) + eval(h - 2. * e)) / (2. * e)
                    } else {
                        (eval(h + e) - eval(h - e)) / (2. * e)
                    };
                    close(g.hurst, fd, 3e-5);
                    maxgap = maxgap.max((g.hurst - fd).abs());
                    count += 1;
                }
                let base = p
                    .evolve_heston_parameter_path(100., &z)
                    .unwrap()
                    .reverse(&f, &v)
                    .unwrap();
                assert_eq!(g.initial_forward.to_bits(), base.initial_forward.to_bits());
            }
        }
    }
    assert!(negatives > 0);
    println!("HURST_PATH comparisons={count} negative={negatives} max_gap={maxgap:0.15e}");
}
#[test]
fn brownian_boundary_residual_left_derivative_is_not_zero() {
    let p = RoughVolatilityPathPlan::compile(model(0.5, 0.15), vec![0., 1.]).unwrap();
    let hp = RoughHestonMcHurstPlan::compile(&p).unwrap();
    let z = [0., 0., 1.];
    let r = hp.evolve_path(100., &z).unwrap();
    let got = r.reverse(&[0., 0.], &[0., 1.]).unwrap();
    let expected = 0.7 * (0.055 - 0.04) * (0.5772156649015329 - 1.0) - 0.15 * 0.2;
    close(got.hurst, expected, 2e-14);
    // With one asset step the price uses V0 only, so its H derivative is zero.
    assert_eq!(r.reverse(&[0., 1.], &[0., 0.]).unwrap().hurst, 0.);
}
#[test]
fn domain_seed_and_truncation_boundaries_are_explicit() {
    for m in [
        LiftedHeston::new(0.04, 0.7, 0.055, 0.15, -0.6, vec![1.], vec![0.])
            .unwrap()
            .into(),
        Rfsv::new(0.2, 0.7, 0.1, -1.7, None).unwrap().into(),
    ] {
        let p = RoughVolatilityPathPlan::compile(m, vec![0., 1.]).unwrap();
        assert!(RoughHestonMcHurstPlan::compile(&p).is_err());
    }
    for (v, rho) in [(0., -0.6), (0.04, 1.), (0.04, -1.)] {
        let p = RoughVolatilityPathPlan::compile(
            RoughHeston::new(0.2, v, 0.7, 0.055, 0.15, rho)
                .unwrap()
                .into(),
            vec![0., 1.],
        )
        .unwrap();
        assert!(RoughHestonMcHurstPlan::compile(&p).is_err());
    }
    let p = RoughVolatilityPathPlan::compile(model(0.2, 0.15), vec![0., 0.25, 1.]).unwrap();
    let hp = RoughHestonMcHurstPlan::compile(&p).unwrap();
    let z = [0.; 6];
    assert!(hp.evolve_path(100., &[]).is_err());
    assert!(hp.evolve_path(100., &[f64::NAN; 6]).is_err());
    let r = hp.evolve_path(100., &z).unwrap();
    assert!(r.reverse(&[0.], &[0.; 3]).is_err());
    assert!(r.reverse(&[f64::NAN; 3], &[0.; 3]).is_err());
    assert_eq!(r.reverse(&[0.; 3], &[0.; 3]).unwrap().hurst, 0.);
    let zero = RoughVolatilityPathPlan::compile(
        RoughHeston::new(0.5, 0.04, 1., 0., 0., 0.).unwrap().into(),
        vec![0., 1., 2.],
    )
    .unwrap();
    assert!(
        RoughHestonMcHurstPlan::compile(&zero)
            .unwrap()
            .evolve_path(100., &z)
            .is_err()
    );
    let terminal = RoughVolatilityPathPlan::compile(
        RoughHeston::new(0.5, 0.04, 1., 0., 0., 0.).unwrap().into(),
        vec![0., 1.],
    )
    .unwrap();
    let hp = RoughHestonMcHurstPlan::compile(&terminal).unwrap();
    let z = [0.; 3];
    let r = hp.evolve_path(100., &z).unwrap();
    assert!(r.reverse(&[0., 1.], &[0., 0.]).is_ok());
    assert!(r.reverse(&[0., 0.], &[0., 1.]).is_err());
}
#[test]
fn high_level_price_bumps_workers_and_old_parameter_results() {
    let mut maximum: f64 = 0.;
    let mut count = 0;
    for qmc in [false, true] {
        for h in [0.1, 0.3, 0.5] {
            let request = req(payload(qmc, true));
            let build = |h, w| {
                RoughVolatilityPricingPlan::compile(
                    &request,
                    model(h, 0.15),
                    0.25,
                    ExecutionPolicy::new(w, None).unwrap(),
                )
                .unwrap()
            };
            let p = build(h, 1);
            let old = p.evaluate_heston_parameter_risk().unwrap();
            let r = p.evaluate_hurst_risk().unwrap();
            assert_eq!(r.price, p.evaluate().unwrap());
            assert_eq!(p.evaluate_heston_parameter_risk().unwrap(), old);
            let worker = build(h, 3).evaluate_hurst_risk().unwrap();
            assert_eq!(
                worker.hurst_sensitivity.to_bits(),
                r.hurst_sensitivity.to_bits()
            );
            assert_eq!(worker.standard_error.to_bits(), r.standard_error.to_bits());
            assert_ne!(worker.risk_fingerprint, r.risk_fingerprint);
            for e in [1e-6, 5e-7] {
                let eval = |h| build(h, 1).evaluate().unwrap().value;
                let fd = if h == 0.5 {
                    (3. * r.price.value - 4. * eval(h - e) + eval(h - 2. * e)) / (2. * e)
                } else {
                    (eval(h + e) - eval(h - e)) / (2. * e)
                };
                close(r.hurst_sensitivity, fd, 3e-5);
                maximum = maximum.max((r.hurst_sensitivity - fd).abs());
                count += 1;
            }
            assert!(r.standard_error.is_finite() && r.standard_error > 0.);
        }
    }
    println!("HURST_PRICE comparisons={count} max_gap={maximum:0.15e}");
}
#[test]
#[ignore = "release-only independent finite-grid deterministic variance / Black panel"]
fn independently_referenced_deterministic_variance_hurst_risk() {
    let data: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/mc-hurst.json"
    ))
    .unwrap();
    for row in data["black"].as_array().unwrap() {
        for seed in [91, 1973] {
            let h = row["h"].as_f64().unwrap();
            let want = row["hurst"].as_f64().unwrap();
            let price = row["price"].as_f64().unwrap();
            let mut v = payload(true, false);
            for c in ["discount_curve", "dividend_curve"] {
                v["market"][c]["discount_factors"] = json!([1., 1.]);
            }
            v["engine"]["points_per_scramble"] = json!(2048);
            v["engine"]["scramble_count"] = json!(8);
            v["engine"]["master_scramble_seed"] = json!(seed);
            let p = RoughVolatilityPricingPlan::compile(
                &req(v),
                model(h, 0.),
                0.25,
                ExecutionPolicy::new(1, None).unwrap(),
            )
            .unwrap();
            let r = p.evaluate_hurst_risk().unwrap();
            let gap = (r.hurst_sensitivity - want).abs();
            assert!((r.price.value - price).abs() <= 5. * r.price.standard_error + 2e-4);
            assert!(
                gap <= 5. * r.standard_error + 2e-4,
                "{r:?}: reference {want}"
            );
            assert!(r.standard_error > 0. && r.standard_error <= 0.01);
            println!(
                "HURST_BLACK h={h} seed={seed} risk={:0.15e} reference={want:0.15e} se={:0.15e} gap={gap:0.15e}",
                r.hurst_sensitivity, r.standard_error
            );
        }
    }
}

#[test]
fn sampling_error_reconstructs_pairs_and_scramble_means() {
    use pricing::mc::{BrownianBridgePlan, EngineConfig, RqmcPlan, inverse_standard_normal};
    for use_qmc in [false, true] {
        for bridge_on in [false, true] {
            let mut v = payload(use_qmc, false);
            for c in ["discount_curve", "dividend_curve"] {
                v["market"][c]["discount_factors"] = json!([1., 1.]);
            }
            v["engine"]["variance_reduction"]["brownian_bridge"] = json!(bridge_on);
            let request = req(v);
            let p = RoughVolatilityPricingPlan::compile(
                &request,
                model(0.2, 0.15),
                0.25,
                ExecutionPolicy::new(1, None).unwrap(),
            )
            .unwrap();
            let hp = RoughHestonMcHurstPlan::compile(p.path_plan()).unwrap();
            let path = hp.path_plan();
            let n = path.time_nodes().len() - 1;
            let dim = path.random_dimension();
            let bridge = BrownianBridgePlan::compile(path.time_nodes().to_vec(), 1).unwrap();
            let (scrambles, points, qmc) = match request.engine() {
                EngineConfig::RandomizedQuasiMonteCarlo(c) => (
                    c.scramble_count().get(),
                    c.points_per_scramble().get(),
                    Some(RqmcPlan::compile(c, dim).unwrap()),
                ),
                EngineConfig::PseudoMonteCarlo(c) => {
                    (1, c.independent_sampling_units().get(), None)
                }
            };
            let mut units = Vec::new();
            for s in 0..scrambles {
                let mut mean = 0.;
                for i in 0..points {
                    let mut z = if let Some(q) = &qmc {
                        (0..dim)
                            .map(|d| inverse_standard_normal(q.uniform(s, i, d).unwrap()).unwrap())
                            .collect::<Vec<_>>()
                    } else {
                        path.pseudo_shocks(819, i, RandomDomain::Valuation)
                    };
                    if bridge_on {
                        for block in z[..2 * n].chunks_exact_mut(n) {
                            let b = bridge.apply_one_factor(block).unwrap();
                            block.copy_from_slice(&b);
                        }
                    }
                    let mut pair = 0.;
                    for sign in [1., -1.] {
                        let z = z.iter().map(|v| sign * v).collect::<Vec<_>>();
                        let r = hp.evolve_path(100., &z).unwrap();
                        let mut seed = vec![0.; n + 1];
                        seed[n] = f64::from(r.path().forwards[n] > 100.);
                        pair += 0.5 * r.reverse(&seed, &vec![0.; n + 1]).unwrap().hurst;
                    }
                    if use_qmc {
                        mean += pair;
                    } else {
                        units.push(pair);
                    }
                }
                if use_qmc {
                    units.push(mean / points as f64);
                }
            }
            let mean = units.iter().sum::<f64>() / units.len() as f64;
            let se = (units.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (units.len() * (units.len() - 1)) as f64)
                .sqrt();
            let got = p.evaluate_hurst_risk().unwrap();
            close(got.hurst_sensitivity, mean, 2e-12);
            close(got.standard_error, se, 2e-12);
        }
    }
}
