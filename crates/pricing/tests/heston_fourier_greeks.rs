//! Fixed-model Forward Greeks: independent references and finite-grid derivatives.
use pricing::rough_volatility::*;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/fourier-greeks.json"
    ))
    .unwrap()
}
fn rough(h: f64) -> RoughHeston {
    RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65).unwrap()
}
fn lift() -> LiftedHeston {
    LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.2, 0.4, 0.5],
        vec![0.1, 1.0, 8.0],
    )
    .unwrap()
}
fn cfg(n: usize, m: usize, u: f64) -> HestonFourierConfig {
    HestonFourierConfig::new(n, m, u).unwrap()
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
        "actual={a:.16e} expected={b:.16e} gap={:.6e} tolerance={tolerance:.6e}",
        (a - b).abs()
    );
}

#[test]
fn deterministic_kinks_zero_control_and_input_domains() {
    for model in [rough(0.1).into(), lift().into()] {
        let plan = HestonFourierPlan::compile(model, 0.0, cfg(16, 32, 32.0)).unwrap();
        assert!(matches!(
            plan.price_and_greeks(100.0, 100.0, 0.97),
            Err(FourierError::InvalidInput(_))
        ));
        for f in [90.0, 110.0] {
            let g = plan.price_and_greeks(f, 100.0, 0.97).unwrap();
            assert_eq!(g.price, plan.price(f, 100.0, 0.97).unwrap());
            assert_eq!(g.call_forward_delta, if f > 100.0 { 0.97 } else { 0.0 });
            assert_eq!(g.put_forward_delta, if f < 100.0 { -0.97 } else { 0.0 });
            assert_eq!(g.forward_gamma, 0.0);
            assert_eq!(g.delta_tail_indicator, 0.0);
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            for (f, k, d) in [(bad, 100.0, 1.0), (100.0, bad, 1.0), (100.0, 100.0, bad)] {
                assert!(matches!(
                    plan.price_and_greeks(f, k, d),
                    Err(FourierError::InvalidInput(_))
                ));
            }
        }
    }
    let zero = RoughHeston::new(0.1, 0.0, 0.0, 0.05, 0.3, -0.6).unwrap();
    let plan = HestonFourierPlan::compile(zero.into(), 1.0, cfg(32, 64, 32.0)).unwrap();
    assert!(plan.price_and_greeks(100.0, 100.0, 1.0).is_err());
    close(
        plan.price_and_greeks(110.0, 100.0, 1.0)
            .unwrap()
            .forward_gamma,
        0.0,
        0.0,
    );
    // Zero initial variance with positive immigration is NOT the intrinsic law.
    let nonzero = RoughHeston::new(0.5, 0.0, 0.7, 0.055, 0.0, -0.65).unwrap();
    let plan = HestonFourierPlan::compile(nonzero.into(), 1.0, cfg(128, 512, 192.0)).unwrap();
    assert!(plan.price(100.0, 100.0, 0.97).unwrap().call > 0.0);
    assert!(matches!(
        plan.price_and_greeks(100.0, 100.0, 0.97),
        Err(FourierError::InvalidInput(_))
    ));
}

#[test]
fn deterministic_variance_matches_independent_black_greeks() {
    let refs = reference();
    for h in [0.1, 0.3, 0.5] {
        for t in [0.25, 1.0] {
            for kind in ["constant", "mean_reverting"] {
                if kind == "mean_reverting" && h != 0.5 {
                    continue;
                }
                let kappa = if kind == "constant" { 0.0 } else { 0.7 };
                let model = RoughHeston::new(h, 0.04, kappa, 0.055, 0.0, -0.65).unwrap();
                let plan =
                    HestonFourierPlan::compile(model.into(), t, cfg(128, 512, 192.0)).unwrap();
                for row in refs["black_rows"].as_array().unwrap() {
                    if row["kind"] != kind || row["maturity"].as_f64().unwrap() != t {
                        continue;
                    }
                    let g = plan
                        .price_and_greeks(100.0, row["strike"].as_f64().unwrap(), 0.97)
                        .unwrap();
                    let tolerance = if kind == "constant" { 1e-12 } else { 2e-7 };
                    close(
                        g.call_forward_delta,
                        row["call_forward_delta"].as_f64().unwrap(),
                        tolerance,
                    );
                    close(
                        g.put_forward_delta,
                        row["put_forward_delta"].as_f64().unwrap(),
                        tolerance,
                    );
                    close(
                        g.forward_gamma,
                        row["forward_gamma"].as_f64().unwrap(),
                        tolerance,
                    );
                    assert_eq!(
                        g.price,
                        plan.price(100.0, row["strike"].as_f64().unwrap(), 0.97)
                            .unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn analytic_greeks_preserve_price_parity_homogeneity_and_bump_consistency() {
    let refs = reference();
    for model in [rough(0.1).into(), rough(0.3).into(), lift().into()] {
        let plan = HestonFourierPlan::compile(model, 1.0, cfg(256, 512, 192.0)).unwrap();
        for k in [80.0, 100.0, 120.0] {
            let g = plan.price_and_greeks(100.0, k, 0.97).unwrap();
            assert_eq!(g.price, plan.price(100.0, k, 0.97).unwrap());
            close(g.call_forward_delta - g.put_forward_delta, 0.97, 3e-15);
            for scale in [0.01, 0.5, 3.0, 100.0] {
                let s = plan
                    .price_and_greeks(100.0 * scale, k * scale, 0.97)
                    .unwrap();
                close(s.call_forward_delta, g.call_forward_delta, 2e-13);
                close(s.put_forward_delta, g.put_forward_delta, 2e-13);
                close(s.forward_gamma * scale, g.forward_gamma, 2e-13);
                close(s.price.call / scale, g.price.call, 2e-11);
            }
            let d = plan.price_and_greeks(100.0, k, 0.97 / 2.0).unwrap();
            close(2.0 * d.call_forward_delta, g.call_forward_delta, 0.0);
            close(2.0 * d.forward_gamma, g.forward_gamma, 0.0);
            for h in refs["protocol"]["bump_sizes"].as_array().unwrap() {
                let h = h.as_f64().unwrap();
                let up = plan.price(100.0 + h, k, 0.97).unwrap();
                let down = plan.price(100.0 - h, k, 0.97).unwrap();
                close(
                    (up.call - down.call) / (2.0 * h),
                    g.call_forward_delta,
                    refs["protocol"]["bump_delta_tolerance"].as_f64().unwrap(),
                );
                close(
                    (up.put - down.put) / (2.0 * h),
                    g.put_forward_delta,
                    refs["protocol"]["bump_delta_tolerance"].as_f64().unwrap(),
                );
                for (u, b, d) in [
                    (up.call, g.price.call, down.call),
                    (up.put, g.price.put, down.put),
                ] {
                    close(
                        (u - 2.0 * b + d) / (h * h),
                        g.forward_gamma,
                        refs["protocol"]["bump_gamma_tolerance"].as_f64().unwrap(),
                    );
                }
            }
        }
    }
}

#[test]
fn derivative_diagnostics_match_explicit_coarse_grid_and_envelope() {
    let fine = HestonFourierPlan::compile(rough(0.1).into(), 0.25, cfg(128, 128, 32.0)).unwrap();
    let coarse = HestonFourierPlan::compile(rough(0.1).into(), 0.25, cfg(128, 64, 32.0)).unwrap();
    let a = fine.price_and_greeks(100.0, 100.0, 0.97).unwrap();
    let b = coarse.price_and_greeks(100.0, 100.0, 0.97).unwrap();
    close(
        a.delta_quadrature_difference,
        (a.call_forward_delta - b.call_forward_delta).abs(),
        1e-15,
    );
    close(
        a.gamma_quadrature_difference,
        (a.forward_gamma - b.forward_gamma).abs(),
        1e-15,
    );
    // Scalar reconstruction from independent transform requests; no stored
    // difference array, production Simpson or accumulator helpers are accessed.
    let mut delta = 0.0;
    let mut gamma = 0.0;
    for j in 64..=128 {
        let u = j as f64 / 4.0;
        let phi = fine.log_transform(Complex64::new(0.5, u)).unwrap().exp();
        let black = (-0.5 * 0.04 * 0.25 * (u * u + 0.25)).exp();
        let norm = (black - phi.re).hypot(phi.im);
        let weight = if j == 64 || j == 128 {
            1.0
        } else if j % 2 == 0 {
            2.0
        } else {
            4.0
        };
        delta += weight * norm / (u * u + 0.25).sqrt();
        gamma += weight * norm;
    }
    close(
        a.delta_tail_indicator,
        0.97 / std::f64::consts::PI * 0.25 / 3.0 * delta,
        1e-14,
    );
    close(
        a.gamma_tail_indicator,
        0.97 / (100.0 * std::f64::consts::PI) * 0.25 / 3.0 * gamma,
        1e-14,
    );
    assert!(a.delta_tail_indicator > 1e-8 && a.gamma_tail_indicator > 1e-8);
}

#[test]
#[ignore = "independent stochastic Greek acceptance; executed by dedicated release CI"]
fn nondegenerate_greeks_match_independent_share_probability_and_density() {
    let refs = reference();
    let p = &refs["protocol"];
    let config = cfg(
        p["time_steps"].as_u64().unwrap() as usize,
        p["integration_intervals"].as_u64().unwrap() as usize,
        p["cutoff"].as_f64().unwrap(),
    );
    let mut count = 0;
    for t in [0.25, 1.0] {
        for (family, model) in [("heston", rough(0.5).into()), ("lift", lift().into())] {
            let plan = HestonFourierPlan::compile(model, t, config).unwrap();
            for row in refs["rows"].as_array().unwrap() {
                if row["family"] != family || row["maturity"].as_f64().unwrap() != t {
                    continue;
                }
                let k = row["strike"].as_f64().unwrap();
                let got = plan.price_and_greeks(100.0, k, 0.97).unwrap();
                let delta = got.call_forward_delta - row["call_forward_delta"].as_f64().unwrap();
                let gamma = got.forward_gamma - row["forward_gamma"].as_f64().unwrap();
                println!("GREEK_REFERENCE,{family},{t},{k},{delta:.12e},{gamma:.12e}");
                close(delta, 0.0, p["delta_abs_tolerance"].as_f64().unwrap());
                close(
                    got.put_forward_delta,
                    row["put_forward_delta"].as_f64().unwrap(),
                    p["delta_abs_tolerance"].as_f64().unwrap(),
                );
                close(gamma, 0.0, p["gamma_abs_tolerance"].as_f64().unwrap());
                assert_eq!(got.price, plan.price(100.0, k, 0.97).unwrap());
                count += 1;
            }
        }
    }
    assert_eq!(count, 12);
}

#[test]
#[ignore = "time and frequency cutoff checks; executed by dedicated release CI"]
fn rough_and_lifted_greek_time_and_cutoff_refinement() {
    let refs = reference();
    let p = &refs["protocol"];
    let mut count = 0;
    for t in p["refinement_maturities"].as_array().unwrap() {
        let t = t.as_f64().unwrap();
        for (name, model) in [
            ("rough_h01", RoughVolatilityModel::from(rough(0.1))),
            ("rough_h03", rough(0.3).into()),
            ("lift", lift().into()),
        ] {
            let ns = p["refinement_time_steps"].as_array().unwrap();
            let ms = p["refinement_intervals"].as_array().unwrap();
            let us = p["refinement_cutoffs"].as_array().unwrap();
            let a = HestonFourierPlan::compile(
                model.clone(),
                t,
                cfg(
                    ns[0].as_u64().unwrap() as usize,
                    ms[0].as_u64().unwrap() as usize,
                    us[0].as_f64().unwrap(),
                ),
            )
            .unwrap();
            let b = HestonFourierPlan::compile(
                model.clone(),
                t,
                cfg(
                    ns[1].as_u64().unwrap() as usize,
                    ms[0].as_u64().unwrap() as usize,
                    us[0].as_f64().unwrap(),
                ),
            )
            .unwrap();
            let c = HestonFourierPlan::compile(
                model,
                t,
                cfg(
                    ns[1].as_u64().unwrap() as usize,
                    ms[1].as_u64().unwrap() as usize,
                    us[1].as_f64().unwrap(),
                ),
            )
            .unwrap();
            for k in [80.0, 100.0, 120.0] {
                let x = a.price_and_greeks(100.0, k, 0.97).unwrap();
                let y = b.price_and_greeks(100.0, k, 0.97).unwrap();
                let z = c.price_and_greeks(100.0, k, 0.97).unwrap();
                let td = (y.call_forward_delta - x.call_forward_delta).abs();
                let tg = (y.forward_gamma - x.forward_gamma).abs();
                let cd = (z.call_forward_delta - y.call_forward_delta).abs();
                let cg = (z.forward_gamma - y.forward_gamma).abs();
                println!("GREEK_REFINEMENT,{name},{t},{k},{td:.12e},{tg:.12e},{cd:.12e},{cg:.12e}");
                for d in [td, cd] {
                    close(d, 0.0, p["refinement_delta_tolerance"].as_f64().unwrap());
                }
                for g in [tg, cg] {
                    close(g, 0.0, p["refinement_gamma_tolerance"].as_f64().unwrap());
                }
                count += 1;
            }
        }
    }
    assert_eq!(count, 18);
}
