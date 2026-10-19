//! Fixed-kernel MC parameter VJP controls, not continuum-model Greek admission.
use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn model(lift: bool, h: f64, p: [f64; 5]) -> RoughVolatilityModel {
    if lift {
        LiftedHeston::new(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            vec![0.2, 0.8, 1.3],
            vec![0.0, 0.7, 12.0],
        )
        .unwrap()
        .into()
    } else {
        RoughHeston::new(h, p[0], p[1], p[2], p[3], p[4])
            .unwrap()
            .into()
    }
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{a:.15e} vs {b:.15e}"
    );
}
fn objective(p: &RoughVolatilityPath, f: &[f64], v: &[f64]) -> f64 {
    p.forwards.iter().zip(f).map(|(a, b)| a * b).sum::<f64>()
        + p.variances.iter().zip(v).map(|(a, b)| a * b).sum::<f64>()
}
fn payload(qmc: bool, dividends: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(100.0);
    if dividends {
        v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    }
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":4,
        "master_scramble_seed":819,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":128,"master_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    };
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

#[test]
fn all_five_path_adjoints_match_two_bumps_including_truncated_nodes() {
    let times = vec![0.0, 0.07, 0.21, 0.63, 1.0];
    let fs = [0.3, -0.2, 0.15, 0.1, 0.7];
    let vs = [0.2, 0.3, -0.7, 0.4, 0.8];
    let mut negative = 0;
    let mut maximum: f64 = 0.0;
    let mut comparisons = 0;
    for lift in [false, true] {
        for h in [0.1, 0.3, 0.5] {
            for nu in [0.15, 0.7] {
                let p = [0.04, 0.7, 0.055, nu, -0.6];
                let plan =
                    RoughVolatilityPathPlan::compile(model(lift, h, p), times.clone()).unwrap();
                for i in 0..16 {
                    let z = plan.pseudo_shocks(91, i, RandomDomain::Valuation);
                    let record = plan.evolve_heston_parameter_path(100.0, &z).unwrap();
                    assert_eq!(record.path(), &plan.evolve_path(100.0, &z).unwrap());
                    negative += record.path().negative_variance_nodes;
                    let got = record.reverse(&fs, &vs).unwrap();
                    for k in 0..5 {
                        for eps in [2e-7, 1e-7] {
                            let mut up = p;
                            up[k] += eps;
                            let mut dn = p;
                            dn[k] -= eps;
                            let a =
                                RoughVolatilityPathPlan::compile(model(lift, h, up), times.clone())
                                    .unwrap()
                                    .evolve_path(100.0, &z)
                                    .unwrap();
                            let b =
                                RoughVolatilityPathPlan::compile(model(lift, h, dn), times.clone())
                                    .unwrap()
                                    .evolve_path(100.0, &z)
                                    .unwrap();
                            assert_eq!(a.negative_variance_nodes, b.negative_variance_nodes);
                            let want =
                                (objective(&a, &fs, &vs) - objective(&b, &fs, &vs)) / (2.0 * eps);
                            close(got.parameters[k], want, 2e-5);
                            maximum = maximum.max((got.parameters[k] - want).abs());
                            comparisons += 1;
                        }
                    }
                    let eps = 1e-3;
                    let up = plan.evolve_path(100.0 + eps, &z).unwrap();
                    let dn = plan.evolve_path(100.0 - eps, &z).unwrap();
                    close(
                        got.initial_forward,
                        (objective(&up, &fs, &vs) - objective(&dn, &fs, &vs)) / (2.0 * eps),
                        2e-9,
                    );
                }
            }
        }
    }
    assert!(negative > 50, "truncation not materially exercised");
    println!("HESTON_MC_PATH comparisons={comparisons} negative={negative} max_gap={maximum:.14e}");
}

#[test]
fn boundary_and_seed_failures_are_explicit_and_price_still_works() {
    for (v0, rho) in [(0.0, -0.6), (0.04, 1.0), (0.04, -1.0)] {
        let p = RoughVolatilityPathPlan::compile(
            model(false, 0.2, [v0, 0.7, 0.055, 0.15, rho]),
            vec![0.0, 0.25, 1.0],
        )
        .unwrap();
        let z = vec![0.0; p.random_dimension() as usize];
        assert!(p.evolve_path(100.0, &z).is_ok());
        assert!(p.evolve_heston_parameter_path(100.0, &z).is_err());
    }
    let unsupported: RoughVolatilityModel = Rfsv::new(0.2, 0.7, 0.1, -1.7, None).unwrap().into();
    let p = RoughVolatilityPathPlan::compile(unsupported, vec![0.0, 1.0]).unwrap();
    assert!(p.evolve_heston_parameter_path(100.0, &[0.0; 3]).is_err());
    // Exactly zero deterministic Euler node: raw=.04 - .04 = 0.
    let p = RoughVolatilityPathPlan::compile(
        model(false, 0.5, [0.04, 1.0, 0.0, 0.0, 0.0]),
        vec![0.0, 1.0, 2.0],
    )
    .unwrap();
    let z = vec![0.0; p.random_dimension() as usize];
    assert_eq!(p.evolve_path(100.0, &z).unwrap().variances[1], 0.0);
    assert!(p.evolve_heston_parameter_path(100.0, &z).is_err());
    let terminal = RoughVolatilityPathPlan::compile(
        model(false, 0.5, [0.04, 1.0, 0.0, 0.0, 0.0]),
        vec![0.0, 1.0],
    )
    .unwrap();
    let terminal_z = [0.0; 3];
    let terminal_record = terminal
        .evolve_heston_parameter_path(100.0, &terminal_z)
        .unwrap();
    assert_eq!(terminal_record.path().variances[1], 0.0);
    assert!(terminal_record.reverse(&[0.0, 1.0], &[0.0, 0.0]).is_ok());
    assert!(terminal_record.reverse(&[0.0, 0.0], &[0.0, 1.0]).is_err());
    let p = RoughVolatilityPathPlan::compile(
        model(false, 0.2, [0.04, 0.7, 0.055, 0.15, -0.6]),
        vec![0.0, 0.25, 1.0],
    )
    .unwrap();
    assert!(p.evolve_heston_parameter_path(100.0, &[]).is_err());
    assert!(
        p.evolve_heston_parameter_path(100.0, &[f64::NAN; 6])
            .is_err()
    );
    let z = vec![0.0; 6];
    let r = p.evolve_heston_parameter_path(100.0, &z).unwrap();
    assert!(r.reverse(&[0.0], &[0.0; 3]).is_err());
    assert!(r.reverse(&[f64::NAN; 3], &[0.0; 3]).is_err());
    assert_eq!(
        r.reverse(&[0.0; 3], &[0.0; 3]).unwrap().parameters,
        [0.0; 5]
    );
}

#[test]
fn public_mc_rqmc_parameters_recompile_at_fixed_market_and_keep_prices() {
    let p = [0.04, 0.7, 0.055, 0.15, -0.6];
    let mut maximum: f64 = 0.0;
    for lift in [false, true] {
        for qmc in [false, true] {
            let req = request(payload(qmc, true));
            let build = |x, workers| {
                RoughVolatilityPricingPlan::compile(
                    &req,
                    model(lift, 0.2, x),
                    0.2,
                    ExecutionPolicy::new(workers, Some(32)).unwrap(),
                )
                .unwrap()
            };
            let plan = build(p, 1);
            let risk = plan.evaluate_heston_parameter_risk().unwrap();
            assert_eq!(risk.price, plan.evaluate().unwrap());
            let other = build(p, 3).evaluate_heston_parameter_risk().unwrap();
            assert_eq!(risk.parameter_adjoints, other.parameter_adjoints);
            assert_eq!(risk.standard_errors, other.standard_errors);
            assert_ne!(risk.risk_fingerprint, other.risk_fingerprint);
            for k in 0..5 {
                for eps in [1e-6, 5e-7] {
                    let mut up = p;
                    up[k] += eps;
                    let mut dn = p;
                    dn[k] -= eps;
                    let want = (build(up, 1).evaluate().unwrap().value
                        - build(dn, 1).evaluate().unwrap().value)
                        / (2.0 * eps);
                    close(risk.parameter_adjoints[k], want, 3e-5);
                    maximum = maximum.max((risk.parameter_adjoints[k] - want).abs());
                }
            }
            assert!(
                risk.standard_errors
                    .iter()
                    .all(|x| x.is_finite() && *x > 0.0)
            );
            assert_eq!(plan.evaluate_heston_parameter_risk().unwrap(), risk);
        }
    }
    println!("HESTON_MC_PRICE comparisons=40 max_gap={maximum:.14e}");
}

#[test]
fn heston_boundary_lift_and_one_step_derivatives_agree() {
    let h = RoughHeston::new(0.5, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
    let a = RoughVolatilityPathPlan::compile(h.clone().into(), vec![0.0, 0.1, 0.4, 1.0]).unwrap();
    let b = RoughVolatilityPathPlan::compile(
        LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
        vec![0.0, 0.1, 0.4, 1.0],
    )
    .unwrap();
    for i in 0..12 {
        let z = a.pseudo_shocks(41, i, RandomDomain::Valuation);
        let ra = a.evolve_heston_parameter_path(100.0, &z).unwrap();
        let rb = b.evolve_heston_parameter_path(100.0, &z[..6]).unwrap();
        let x = ra.reverse(&[0.0, 0.2, 0.3, 0.4], &[0.0; 4]).unwrap();
        let y = rb.reverse(&[0.0, 0.2, 0.3, 0.4], &[0.0; 4]).unwrap();
        for k in 0..5 {
            close(x.parameters[k], y.parameters[k], 1e-11);
        }
    }
    // Terminal-variance seeding at one Brownian-boundary step has an explicit polynomial derivative.
    let t: f64 = 0.3;
    let rho: f64 = -0.6;
    let v: f64 = 0.04;
    let nu = 0.15;
    let kappa = 0.7;
    let theta = 0.055;
    let p = RoughVolatilityPathPlan::compile(h.into(), vec![0.0, t]).unwrap();
    let z = [0.3, -0.4, 0.2];
    let r = p.evolve_heston_parameter_path(100.0, &z).unwrap();
    let g = r.reverse(&[0.0; 2], &[0.0, 1.0]).unwrap();
    let dw = t.sqrt() * (rho * z[0] + (1.0 - rho * rho).sqrt() * z[1]);
    let expected = [
        1.0 - kappa * t + nu * dw / (2.0 * v.sqrt()),
        (theta - v) * t,
        kappa * t,
        v.sqrt() * dw,
        nu * v.sqrt() * t.sqrt() * (z[0] - rho / (1.0 - rho * rho).sqrt() * z[1]),
    ];
    for (a, b) in g.parameters.iter().zip(expected) {
        close(*a, b, 2e-13);
    }
}

#[test]
#[ignore = "release-only independently referenced Black variance-risk panel"]
fn black_constant_variance_risk_uses_sampling_errors_not_grid_error() {
    // kappa=0 fixes V=v0 at nu=0. dPrice/dv0 has an independent Black reference.
    let reference = 99.23813686925295;
    for lift in [false, true] {
        for seed in [91, 1973] {
            let mut v = payload(true, false);
            v["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
            v["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
            v["engine"]["points_per_scramble"] = json!(4096);
            v["engine"]["scramble_count"] = json!(8);
            v["engine"]["master_scramble_seed"] = json!(seed);
            let req = request(v);
            let p = RoughVolatilityPricingPlan::compile(
                &req,
                model(lift, 0.2, [0.04, 0.0, 0.04, 0.0, -0.6]),
                0.25,
                ExecutionPolicy::new(1, None).unwrap(),
            )
            .unwrap();
            let g = p.evaluate_heston_parameter_risk().unwrap();
            let gap = (g.parameter_adjoints[0] - reference).abs();
            assert!(gap <= 5.0 * g.standard_errors[0] + 0.02, "{g:?}");
            assert!(g.standard_errors[0] > 0.0 && g.standard_errors[0] <= 0.1);
            assert_eq!(g.parameter_adjoints[1], 0.0);
            assert_eq!(g.parameter_adjoints[2], 0.0);
            assert_eq!(g.parameter_adjoints[4], 0.0);
            println!(
                "HESTON_MC_BLACK lift={lift} seed={seed} price={:.14e} variance_risk={:.14e} se={:.14e} gap={gap:.14e}",
                g.price.value, g.parameter_adjoints[0], g.standard_errors[0]
            );
        }
    }
}

#[test]
fn sampling_errors_reconstruct_each_independent_unit_and_scramble() {
    use pricing::mc::{BrownianBridgePlan, EngineConfig, RqmcPlan, inverse_standard_normal};
    for lift in [false, true] {
        for use_qmc in [false, true] {
            let mut v = payload(use_qmc, false);
            for curve in ["discount_curve", "dividend_curve"] {
                v["market"][curve]["discount_factors"] = json!([1.0, 1.0]);
            }
            let req = request(v);
            let p = RoughVolatilityPricingPlan::compile(
                &req,
                model(lift, 0.2, [0.04, 0.7, 0.055, 0.15, -0.6]),
                0.25,
                ExecutionPolicy::new(1, None).unwrap(),
            )
            .unwrap();
            let path = p.path_plan();
            let n = path.time_nodes().len() - 1;
            let dim = path.random_dimension();
            let bridge = BrownianBridgePlan::compile(path.time_nodes().to_vec(), 1).unwrap();
            let (scrambles, points, qmc) = match req.engine() {
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
                let mut sum = [0.0; 5];
                for i in 0..points {
                    let mut z = if let Some(q) = &qmc {
                        (0..dim)
                            .map(|d| inverse_standard_normal(q.uniform(s, i, d).unwrap()).unwrap())
                            .collect::<Vec<_>>()
                    } else {
                        path.pseudo_shocks(819, i, RandomDomain::Valuation)
                    };
                    for block in z[..2 * n].chunks_exact_mut(n) {
                        let x = bridge.apply_one_factor(block).unwrap();
                        block.copy_from_slice(&x);
                    }
                    let mut pair = [0.0; 5];
                    for sign in [1.0, -1.0] {
                        let z = z.iter().map(|v| v * sign).collect::<Vec<_>>();
                        let r = path.evolve_heston_parameter_path(100.0, &z).unwrap();
                        let mut seed = vec![0.0; n + 1];
                        seed[n] = f64::from(r.path().forwards[n] > 100.0);
                        let risk = r.reverse(&seed, &vec![0.0; n + 1]).unwrap();
                        for (total, sensitivity) in pair.iter_mut().zip(risk.parameters) {
                            *total += 0.5 * sensitivity;
                        }
                    }
                    if use_qmc {
                        for k in 0..5 {
                            sum[k] += pair[k];
                        }
                    } else {
                        units.push(pair);
                    }
                }
                if use_qmc {
                    units.push(sum.map(|s| s / points as f64));
                }
            }
            let got = p.evaluate_heston_parameter_risk().unwrap();
            for k in 0..5 {
                let mean = units.iter().map(|u| u[k]).sum::<f64>() / units.len() as f64;
                let se = (units.iter().map(|u| (u[k] - mean).powi(2)).sum::<f64>()
                    / (units.len() * (units.len() - 1)) as f64)
                    .sqrt();
                close(got.parameter_adjoints[k], mean, 3e-13);
                close(got.standard_errors[k], se, 3e-13);
            }
            assert_eq!(got.price.independent_sampling_units, units.len() as u64);
        }
    }
}
