//! Fixed-kernel scalar parameter tangents, NOT Hurst, Spot or calibrated risk.
use pricing::rough_volatility::*;

fn model(kind: usize, p: [f64; 5]) -> RoughVolatilityModel {
    if kind == 3 {
        LiftedHeston::new(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            vec![0.2, 0.4, 0.5],
            vec![0.1, 1.0, 8.0],
        )
        .unwrap()
        .into()
    } else {
        RoughHeston::new([0.1, 0.3, 0.5][kind], p[0], p[1], p[2], p[3], p[4])
            .unwrap()
            .into()
    }
}
const P: [f64; 5] = [0.04, 0.7, 0.055, 0.18, -0.65];
fn plan(
    kind: usize,
    p: [f64; 5],
    t: f64,
    steps: usize,
    intervals: usize,
    cutoff: f64,
) -> HestonFourierPlan {
    HestonFourierPlan::compile(
        model(kind, p),
        t,
        HestonFourierConfig::new(steps, intervals, cutoff).unwrap(),
    )
    .unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= tol,
        "actual={a:.15e} ref={b:.15e} gap={:.4e} tolerance={tol:.4e}",
        (a - b).abs()
    );
}

#[test]
fn scalar_risk_matches_two_bumps_of_call_put_and_log_transform() {
    for kind in 0..4 {
        let base = plan(kind, P, 1.0, 64, 256, 96.0);
        let risk = base.parameter_risk_plan().unwrap();
        for q in 0..5 {
            for h in [1e-5, 5e-6] {
                let mut up = P;
                up[q] += h;
                let mut down = P;
                down[q] -= h;
                let up = plan(kind, up, 1.0, 64, 256, 96.0);
                let down = plan(kind, down, 1.0, 64, 256, 96.0);
                for k in [80.0, 100.0, 120.0] {
                    let g = risk.price(100.0, k, 0.97).unwrap();
                    assert_eq!(g.price, base.price(100.0, k, 0.97).unwrap());
                    let a = up.price(100.0, k, 0.97).unwrap();
                    let b = down.price(100.0, k, 0.97).unwrap();
                    close(
                        g.sensitivities.as_array()[q],
                        (a.call - b.call) / (2.0 * h),
                        5e-6,
                    );
                    close(
                        g.sensitivities.as_array()[q],
                        (a.put - b.put) / (2.0 * h),
                        5e-6,
                    );
                }
                for z in [
                    Complex64::new(0.5, 0.7),
                    Complex64::new(0.0, 2.0),
                    Complex64::new(1.0, 0.7),
                ] {
                    let tangent = risk.log_transform_derivatives(z).unwrap()[q];
                    let bumped =
                        (up.log_transform(z).unwrap() - down.log_transform(z).unwrap()) / (2.0 * h);
                    close(tangent.re, bumped.re, 2e-8);
                    close(tangent.im, bumped.im, 2e-8);
                }
            }
        }
    }
}

#[test]
fn zero_maturity_zero_control_inputs_and_moment_strip() {
    for kind in 0..4 {
        let risk = plan(kind, P, 0.0, 8, 16, 16.0)
            .parameter_risk_plan()
            .unwrap();
        for k in [80.0, 100.0, 120.0] {
            let g = risk.price(100.0, k, 0.97).unwrap();
            assert_eq!(g.sensitivities.as_array(), [0.0; 5]);
            assert_eq!(g.quadrature_differences.as_array(), [0.0; 5]);
            assert_eq!(g.tail_indicators.as_array(), [0.0; 5]);
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            for (f, k, d) in [(bad, 100.0, 1.0), (100.0, bad, 1.0), (100.0, 100.0, bad)] {
                assert!(matches!(
                    risk.price(f, k, d),
                    Err(FourierError::InvalidInput(_))
                ));
            }
        }
        let risk = plan(kind, P, 1.0, 32, 64, 32.0)
            .parameter_risk_plan()
            .unwrap();
        for z in [Complex64::ZERO, Complex64::ONE] {
            assert_eq!(
                risk.log_transform_derivatives(z).unwrap(),
                [Complex64::ZERO; 5]
            );
        }
        let z = Complex64::new(0.5, 2.0);
        let a = risk.log_transform_derivatives(z).unwrap();
        let b = risk
            .log_transform_derivatives(Complex64::new(z.re, -z.im))
            .unwrap();
        for q in 0..5 {
            close(a[q].re, b[q].re, 0.0);
            close(a[q].im, -b[q].im, 0.0);
        }
        for z in [
            Complex64::new(-0.1, 0.0),
            Complex64::new(1.1, 0.0),
            Complex64::new(0.5, f64::NAN),
        ] {
            assert!(matches!(
                risk.log_transform_derivatives(z),
                Err(FourierError::InvalidInput(_))
            ));
        }
        for kappa in [0.0, 0.7] {
            let mut p = P;
            p[0] = 0.0;
            p[1] = kappa;
            let base = plan(kind, p, 1.0, 32, 64, 32.0);
            assert!(matches!(
                base.parameter_risk_plan(),
                Err(FourierError::InvalidInput(_))
            ));
        }
    }
}

#[test]
fn inward_boundary_derivatives_and_exact_affine_exponent_directions() {
    for kind in 0..4 {
        for (q, boundary) in [(1, 0.0), (2, 0.0), (3, 0.0), (4, -1.0), (4, 1.0)] {
            let mut p = P;
            p[q] = boundary;
            let base = plan(kind, p, 0.25, 32, 128, 96.0);
            let risk = base.parameter_risk_plan().unwrap();
            let h = if boundary == 1.0 { -1e-5 } else { 1e-5 };
            let mut one = p;
            one[q] += h;
            let mut two = p;
            two[q] += 2.0 * h;
            let one = plan(kind, one, 0.25, 32, 128, 96.0);
            let two = plan(kind, two, 0.25, 32, 128, 96.0);
            let g = risk.price(100.0, 100.0, 0.97).unwrap();
            let f0 = g.price.call;
            let f1 = one.price(100.0, 100.0, 0.97).unwrap().call;
            let f2 = two.price(100.0, 100.0, 0.97).unwrap().call;
            close(
                g.sensitivities.as_array()[q],
                (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * h),
                5e-6,
            );
            if q == 1 {
                close(g.sensitivities.long_run_variance, 0.0, 0.0);
            }
            if q == 3 {
                close(g.sensitivities.correlation, 0.0, 0.0);
            }
        }
        let base = plan(kind, P, 1.0, 32, 64, 32.0);
        let risk = base.parameter_risk_plan().unwrap();
        let z = Complex64::new(0.5, 3.0);
        for q in [0, 2] {
            let mut p = P;
            p[q] += 0.03;
            let shifted = plan(kind, p, 1.0, 32, 64, 32.0);
            let d = (shifted.log_transform(z).unwrap() - base.log_transform(z).unwrap()) / 0.03;
            let tangent = risk.log_transform_derivatives(z).unwrap()[q];
            close(d.re, tangent.re, 2e-13);
            close(d.im, tangent.im, 2e-13);
        }
    }
}

#[test]
fn scaling_diagnostics_and_fixed_factor_splitting() {
    let base = plan(0, P, 0.25, 64, 128, 32.0);
    let risk = base.parameter_risk_plan().unwrap();
    let r = risk.price(100.0, 100.0, 0.97).unwrap();
    let coarse = plan(0, P, 0.25, 64, 64, 32.0)
        .parameter_risk_plan()
        .unwrap()
        .price(100.0, 100.0, 0.97)
        .unwrap();
    for scale in [0.01, 0.5, 3.0, 100.0] {
        let g = risk.price(100.0 * scale, 100.0 * scale, 0.97).unwrap();
        for q in 0..5 {
            close(
                g.sensitivities.as_array()[q] / scale,
                r.sensitivities.as_array()[q],
                3e-11,
            );
        }
    }
    let half = risk.price(100.0, 100.0, 0.485).unwrap();
    for q in 0..5 {
        close(
            half.sensitivities.as_array()[q] * 2.0,
            r.sensitivities.as_array()[q],
            0.0,
        );
        close(
            r.quadrature_differences.as_array()[q],
            (r.sensitivities.as_array()[q] - coarse.sensitivities.as_array()[q]).abs(),
            3e-14,
        );
        assert!(r.tail_indicators.as_array()[q] > 0.0);
    }
    let a = LiftedHeston::new(P[0], P[1], P[2], P[3], P[4], vec![1.0], vec![0.0]).unwrap();
    let b =
        LiftedHeston::new(P[0], P[1], P[2], P[3], P[4], vec![0.5, 0.5], vec![0.0, 0.0]).unwrap();
    let make = |m| {
        HestonFourierPlan::compile(m, 0.25, HestonFourierConfig::new(64, 128, 64.0).unwrap())
            .unwrap()
            .parameter_risk_plan()
            .unwrap()
    };
    let a = make(a.into()).price(100.0, 100.0, 0.97).unwrap();
    let b = make(b.into()).price(100.0, 100.0, 0.97).unwrap();
    for q in 0..5 {
        close(
            a.sensitivities.as_array()[q],
            b.sensitivities.as_array()[q],
            1e-12,
        );
    }
}

fn references() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/parameter-risk.json"
    ))
    .unwrap()
}

#[test]
#[ignore = "independent continuous-time derivative references; explicit release CI"]
fn independent_continuous_time_price_and_fractional_transform_derivatives() {
    let refs = references();
    let tol = refs["protocol"]["price_abs_tolerance"].as_f64().unwrap();
    let mut worst = 0.0_f64;
    for kind in [2, 3] {
        for t in [0.25, 1.0] {
            let base = plan(kind, P, t, 512, 1024, 256.0);
            let risk = base.parameter_risk_plan().unwrap();
            for row in refs["rows"].as_array().unwrap() {
                if row["family"] != if kind == 2 { "heston" } else { "lift" }
                    || row["maturity"].as_f64().unwrap() != t
                {
                    continue;
                }
                let k = row["strike"].as_f64().unwrap();
                let got = risk.price(100.0, k, 0.97).unwrap();
                for q in 0..5 {
                    let expected = row["derivatives"][q].as_f64().unwrap();
                    let value = got.sensitivities.as_array()[q];
                    worst = worst.max((value - expected).abs());
                    println!(
                        "reference kind={kind} t={t} k={k} parameter={q} value={value:.12e} expected={expected:.12e} gap={:.12e}",
                        value - expected
                    );
                    close(value, expected, tol);
                }
                assert_eq!(got.price, base.price(100.0, k, 0.97).unwrap());
            }
        }
    }
    println!("maximum_independent_price_derivative_gap={worst:.12e}");
    let mut worst = 0.0_f64;
    for kind in [0, 1] {
        let h = [0.1, 0.3][kind];
        for t in [0.25, 1.0] {
            let risk = plan(kind, P, t, 512, 8, 1.0).parameter_risk_plan().unwrap();
            for row in refs["rough_transforms"].as_array().unwrap() {
                if row["hurst"].as_f64().unwrap() != h || row["maturity"].as_f64().unwrap() != t {
                    continue;
                }
                let z = Complex64::new(
                    row["exponent"][0].as_f64().unwrap(),
                    row["exponent"][1].as_f64().unwrap(),
                );
                let d = risk.log_transform_derivatives(z).unwrap();
                for q in 0..5 {
                    let r = Complex64::new(
                        row["derivatives"][q][0].as_f64().unwrap(),
                        row["derivatives"][q][1].as_f64().unwrap(),
                    );
                    let gap = (d[q] - r).abs();
                    worst = worst.max(gap);
                    println!("fractional h={h} t={t} z={z:?} parameter={q} gap={gap:.12e}");
                    close(
                        gap,
                        0.0,
                        refs["protocol"]["transform_abs_tolerance"]
                            .as_f64()
                            .unwrap(),
                    );
                }
            }
        }
    }
    println!("maximum_independent_rough_transform_derivative_gap={worst:.12e}");
}

#[test]
#[ignore = "finite Riccati and cutoff refinement; explicit release CI"]
fn separate_time_and_frequency_cutoff_parameter_risk_refinement() {
    let refs = references();
    let tolerance = refs["protocol"]["refinement_abs_tolerance"]
        .as_f64()
        .unwrap();
    let mut worst_time = 0.0_f64;
    let mut worst_cutoff = 0.0_f64;
    let mut failures = 0;
    for kind in [0, 1, 3] {
        for t in [0.25, 1.0] {
            let a = plan(kind, P, t, 512, 512, 128.0)
                .parameter_risk_plan()
                .unwrap();
            let b = plan(kind, P, t, 1024, 512, 128.0)
                .parameter_risk_plan()
                .unwrap();
            let c = plan(kind, P, t, 1024, 1024, 256.0)
                .parameter_risk_plan()
                .unwrap();
            for k in [80.0, 100.0, 120.0] {
                let a = a.price(100.0, k, 0.97).unwrap();
                let b = b.price(100.0, k, 0.97).unwrap();
                let c = c.price(100.0, k, 0.97).unwrap();
                for q in 0..5 {
                    let dt = (a.sensitivities.as_array()[q] - b.sensitivities.as_array()[q]).abs();
                    let du = (b.sensitivities.as_array()[q] - c.sensitivities.as_array()[q]).abs();
                    worst_time = worst_time.max(dt);
                    worst_cutoff = worst_cutoff.max(du);
                    println!(
                        "refinement kind={kind} t={t} k={k} parameter={q} time_gap={dt:.12e} cutoff_gap={du:.12e}"
                    );
                    if dt > tolerance || du > tolerance {
                        failures += 1;
                    }
                }
            }
        }
    }
    println!(
        "maximum_time_gap={worst_time:.12e} maximum_cutoff_gap={worst_cutoff:.12e} failures={failures}"
    );
    assert_eq!(failures, 0);
}
