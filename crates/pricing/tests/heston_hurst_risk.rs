//! Hurst derivative at fixed scalar parameters, not finite-lift factory risk.
use pricing::rough_volatility::*;
const P: [f64; 5] = [0.04, 0.7, 0.055, 0.18, -0.65];
fn plan(h: f64, p: [f64; 5], t: f64, steps: usize, n: usize, cutoff: f64) -> HestonFourierPlan {
    HestonFourierPlan::compile(
        RoughHeston::new(h, p[0], p[1], p[2], p[3], p[4])
            .unwrap()
            .into(),
        t,
        HestonFourierConfig::new(steps, n, cutoff).unwrap(),
    )
    .unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= tol,
        "actual={a:.15e} reference={b:.15e} gap={:.5e} tol={tol:.5e}",
        (a - b).abs()
    );
}
#[test]
fn two_bumps_of_call_put_and_log_transform_including_hurst_boundary() {
    for h in [0.01, 0.1, 0.3, 0.5] {
        let base = plan(h, P, 1.0, 64, 256, 96.0);
        let risk = base.hurst_risk_plan().unwrap();
        for bump in [1e-4, 5e-5] {
            let (a, b, denom) = if h == 0.5 {
                (
                    plan(h - bump, P, 1.0, 64, 256, 96.0),
                    plan(h - 2.0 * bump, P, 1.0, 64, 256, 96.0),
                    2.0 * bump,
                )
            } else {
                (
                    plan(h + bump, P, 1.0, 64, 256, 96.0),
                    plan(h - bump, P, 1.0, 64, 256, 96.0),
                    2.0 * bump,
                )
            };
            for k in [80.0, 100.0, 120.0] {
                let g = risk.price(100.0, k, 0.97).unwrap();
                let p0 = base.price(100.0, k, 0.97).unwrap();
                assert_eq!(g.price, p0);
                let pa = a.price(100.0, k, 0.97).unwrap();
                let pb = b.price(100.0, k, 0.97).unwrap();
                for (p0, pa, pb) in [(p0.call, pa.call, pb.call), (p0.put, pa.put, pb.put)] {
                    let fd = if h == 0.5 {
                        (3.0 * p0 - 4.0 * pa + pb) / denom
                    } else {
                        (pa - pb) / denom
                    };
                    close(g.hurst_sensitivity, fd, 5e-6);
                }
            }
            for z in [
                Complex64::new(0.5, 0.7),
                Complex64::new(0.0, 2.0),
                Complex64::new(1.0, 0.7),
            ] {
                let p0 = base.log_transform(z).unwrap();
                let pa = a.log_transform(z).unwrap();
                let pb = b.log_transform(z).unwrap();
                let fd = if h == 0.5 {
                    (p0 * 3.0 - pa * 4.0 + pb) / denom
                } else {
                    (pa - pb) / denom
                };
                let actual = risk.log_transform_derivative(z).unwrap();
                close(actual.re, fd.re, 2e-8);
                close(actual.im, fd.im, 2e-8);
            }
        }
    }
}
#[test]
fn domain_moment_strip_zero_time_and_zero_variance_limits() {
    for h in [0.01, 0.1, 0.5] {
        let r = plan(h, P, 0.0, 8, 16, 16.0).hurst_risk_plan().unwrap();
        for k in [80.0, 100.0, 120.0] {
            let g = r.price(100.0, k, 0.97).unwrap();
            assert_eq!(
                (
                    g.hurst_sensitivity,
                    g.quadrature_difference,
                    g.tail_indicator
                ),
                (0.0, 0.0, 0.0)
            );
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            for (f, k, d) in [(bad, 100.0, 1.0), (100.0, bad, 1.0), (100.0, 100.0, bad)] {
                assert!(matches!(
                    r.price(f, k, d),
                    Err(FourierError::InvalidInput(_))
                ));
            }
        }
        for p in [
            [0.04, 0.0, 0.055, 0.0, -0.65],
            [0.04, 0.7, 0.04, 0.0, -0.65],
            [0.0, 0.0, 0.055, 0.18, -0.65],
        ] {
            let r = plan(h, p, 1.0, 64, 128, 64.0).hurst_risk_plan().unwrap();
            close(
                r.price(100.0, 100.0, 0.97).unwrap().hurst_sensitivity,
                0.0,
                2e-13,
            );
        }
        let r = plan(h, P, 1.0, 64, 128, 64.0).hurst_risk_plan().unwrap();
        for z in [Complex64::ZERO, Complex64::ONE] {
            assert_eq!(r.log_transform_derivative(z).unwrap(), Complex64::ZERO);
        }
        let z = Complex64::new(0.5, 0.7);
        let a = r.log_transform_derivative(z).unwrap();
        let b = r
            .log_transform_derivative(Complex64::new(z.re, -z.im))
            .unwrap();
        assert_eq!(a.re, b.re);
        assert_eq!(a.im, -b.im);
        for z in [
            Complex64::new(-0.1, 0.0),
            Complex64::new(1.1, 0.0),
            Complex64::new(0.5, f64::NAN),
            Complex64::new(0.5, 1e7),
        ] {
            assert!(matches!(
                r.log_transform_derivative(z),
                Err(FourierError::InvalidInput(_))
            ));
        }
    }
    // Unlike the v0/Forward Greek methods, Hurst does not differentiate the zero
    // Black control; positive-maturity v0=0 need not be singular for Hurst risk.
    let mut p = P;
    p[0] = 0.0;
    let base = plan(0.1, p, 1.0, 128, 256, 128.0);
    let r = base.hurst_risk_plan().unwrap();
    let up = plan(0.1001, p, 1.0, 128, 256, 128.0);
    let down = plan(0.0999, p, 1.0, 128, 256, 128.0);
    close(
        r.price(100.0, 100.0, 0.97).unwrap().hurst_sensitivity,
        (up.price(100.0, 100.0, 0.97).unwrap().call - down.price(100.0, 100.0, 0.97).unwrap().call)
            / 0.0002,
        5e-6,
    );
    let lift = LiftedHeston::new(P[0], P[1], P[2], P[3], P[4], vec![1.0], vec![0.0]).unwrap();
    let base = HestonFourierPlan::compile(
        lift.into(),
        1.0,
        HestonFourierConfig::new(32, 64, 32.0).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        base.hurst_risk_plan(),
        Err(FourierError::InvalidInput(_))
    ));
}
#[test]
fn homogeneity_coarse_diagnostic_and_nonvacuous_tail_reconstruction() {
    let base = plan(0.1, P, 1.0, 64, 64, 16.0);
    let r = base.hurst_risk_plan().unwrap();
    let g = r.price(100.0, 100.0, 0.97).unwrap();
    for scale in [0.1, 2.0, 10.0] {
        close(
            r.price(100.0 * scale, 100.0 * scale, 0.97)
                .unwrap()
                .hurst_sensitivity,
            scale * g.hurst_sensitivity,
            2e-11,
        );
    }
    close(
        r.price(100.0, 100.0, 0.485).unwrap().hurst_sensitivity,
        0.5 * g.hurst_sensitivity,
        2e-13,
    );
    let coarse = plan(0.1, P, 1.0, 64, 32, 16.0)
        .hurst_risk_plan()
        .unwrap()
        .price(100.0, 100.0, 0.97)
        .unwrap();
    close(
        g.quadrature_difference,
        (g.hurst_sensitivity - coarse.hurst_sensitivity).abs(),
        2e-13,
    );
    let mut envelope = 0.0;
    for j in 32..=64 {
        let u = j as f64 * 0.25;
        let z = Complex64::new(0.5, u);
        let integrand =
            (base.log_transform(z).unwrap().exp() * r.log_transform_derivative(z).unwrap()).abs()
                / (u * u + 0.25);
        let weight = if j == 32 || j == 64 {
            1.0
        } else if j % 2 == 0 {
            2.0
        } else {
            4.0
        };
        envelope += weight * integrand;
    }
    let expected = 0.97 * 100.0 / std::f64::consts::PI * 0.25 / 3.0 * envelope;
    assert!(expected > 1e-4);
    close(g.tail_indicator, expected, 2e-12);
}
#[test]
fn mixed_partial_consistency_with_existing_fixed_kernel_parameter_risk() {
    let h = 0.1;
    let bump = 1e-4;
    let up = plan(h + bump, P, 0.25, 64, 128, 64.0)
        .parameter_risk_plan()
        .unwrap();
    let down = plan(h - bump, P, 0.25, 64, 128, 64.0)
        .parameter_risk_plan()
        .unwrap();
    for q in [1, 3, 4] {
        let mut a = P;
        let mut b = P;
        a[q] += bump;
        b[q] -= bump;
        let a = plan(h, a, 0.25, 64, 128, 64.0).hurst_risk_plan().unwrap();
        let b = plan(h, b, 0.25, 64, 128, 64.0).hurst_risk_plan().unwrap();
        let dh_dp = (a.price(100.0, 100.0, 0.97).unwrap().hurst_sensitivity
            - b.price(100.0, 100.0, 0.97).unwrap().hurst_sensitivity)
            / (2.0 * bump);
        let dp_dh = (up
            .price(100.0, 100.0, 0.97)
            .unwrap()
            .sensitivities
            .as_array()[q]
            - down
                .price(100.0, 100.0, 0.97)
                .unwrap()
                .sensitivities
                .as_array()[q])
            / (2.0 * bump);
        close(dh_dp, dp_dh, 5e-6);
    }
}
fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/hurst-risk.json"
    ))
    .unwrap()
}
#[test]
#[ignore = "independent 60-digit transform and deterministic-price references; run in release CI"]
fn independent_transform_and_deterministic_mean_reversion_references() {
    let f = fixture();
    for h in [0.01, 0.1, 0.3, 0.5] {
        for t in [0.25, 1.0] {
            let r = plan(h, P, t, 1024, 8, 8.0).hurst_risk_plan().unwrap();
            for row in f["transforms"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|row| row["hurst"] == h && row["maturity"] == t)
            {
                let z = Complex64::new(
                    row["exponent"][0].as_f64().unwrap(),
                    row["exponent"][1].as_f64().unwrap(),
                );
                let d = r.log_transform_derivative(z).unwrap();
                let e = Complex64::new(
                    row["derivative"][0].as_f64().unwrap(),
                    row["derivative"][1].as_f64().unwrap(),
                );
                let gap = (d - e).abs();
                println!("HURST_TRANSFORM H={h} T={t} z={z:?} gap={gap:.12e}");
                close(
                    gap,
                    0.0,
                    f["protocol"]["transform_abs_tolerance"].as_f64().unwrap(),
                );
            }
            let mut p = P;
            p[3] = 0.0;
            let r = plan(h, p, t, 1024, 1024, 256.0).hurst_risk_plan().unwrap();
            for row in f["deterministic"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|row| row["hurst"] == h && row["maturity"] == t)
            {
                let k = row["strike"].as_f64().unwrap();
                let g = r.price(100.0, k, 0.97).unwrap();
                let expected = row["derivative"].as_f64().unwrap();
                println!(
                    "HURST_DETERMINISTIC H={h} T={t} K={k} sensitivity={:.12e} gap={:.12e}",
                    g.hurst_sensitivity,
                    (g.hurst_sensitivity - expected).abs()
                );
                close(
                    g.hurst_sensitivity,
                    expected,
                    f["protocol"]["deterministic_price_abs_tolerance"]
                        .as_f64()
                        .unwrap(),
                );
                close(g.price.call, row["call"].as_f64().unwrap(), 1e-4);
            }
        }
    }
}
#[test]
#[ignore = "separate time-grid and frequency-cutoff checks; finite comparisons not error bounds"]
fn nondegenerate_hurst_price_refinement() {
    let f = fixture();
    let tol = f["protocol"]["refinement_abs_tolerance"].as_f64().unwrap();
    for h in [0.1, 0.3] {
        for t in [0.25, 1.0] {
            let a = plan(h, P, t, 512, 512, 128.0).hurst_risk_plan().unwrap();
            let b = plan(h, P, t, 1024, 512, 128.0).hurst_risk_plan().unwrap();
            let c = plan(h, P, t, 1024, 1024, 256.0).hurst_risk_plan().unwrap();
            for k in [80.0, 100.0, 120.0] {
                let a = a.price(100.0, k, 0.97).unwrap();
                let b = b.price(100.0, k, 0.97).unwrap();
                let c = c.price(100.0, k, 0.97).unwrap();
                println!(
                    "HURST_REFINEMENT H={h} T={t} K={k} sensitivity={:.12e} time_gap={:.12e} cutoff_gap={:.12e}",
                    c.hurst_sensitivity,
                    (a.hurst_sensitivity - b.hurst_sensitivity).abs(),
                    (b.hurst_sensitivity - c.hurst_sensitivity).abs()
                );
                close(a.hurst_sensitivity, b.hurst_sensitivity, tol);
                close(b.hurst_sensitivity, c.hurst_sensitivity, tol);
            }
        }
    }
}
