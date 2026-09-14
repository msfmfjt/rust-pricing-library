use pricing::models::{HullWhite1Factor, HybridCorrelation, RoughBergomi};

#[test]
fn rough_covariance_matches_independent_quadrature_and_ho_lee_limit() {
    for h in [0.01, 0.1, 0.3, 0.49, 0.5] {
        for a in [0.0, 1e-12, 0.2, 2.0, 10.0] {
            let model = RoughBergomi::new(h, 1.2, -0.6).unwrap();
            let rates =
                HullWhite1Factor::new(a, vec![0.0, 0.3, 0.9], vec![0.01, 0.023, 0.014]).unwrap();
            let corr = HybridCorrelation::new(-0.6, 0.25, -0.15).unwrap();
            let c = model.hybrid_covariance(&rates, 0.1, 1.4, corr).unwrap();
            let base = rates.transition(0.1, 1.4, 0.0, corr).unwrap();
            for (i, row) in base.covariance.iter().enumerate() {
                assert_eq!(&c[i][..4], row);
            }
            assert!((c[4][4] - 1.3_f64.powf(2.0 * h)).abs() < 1e-14);
            let p = h + 0.5;
            let cov = (2.0 * h).sqrt() * 1.3_f64.powf(p) / p;
            assert!((c[4][0] - corr.equity_vol * cov).abs() < 1e-14);
            assert!((c[4][1] - cov).abs() < 1e-14);
            let mut expected = [0.0; 2];
            for (lo, hi, sigma) in [(0.1, 0.3, 0.01), (0.3, 0.9, 0.023), (0.9, 1.4, 0.014)] {
                let left = (1.4_f64 - hi).powf(p);
                let right = (1.4_f64 - lo).powf(p);
                let n = 24000;
                let dy = (right - left) / n as f64;
                for j in 0..=n {
                    let u = (left + j as f64 * dy).powf(1.0 / p);
                    let weight = dy / 3.0
                        * if j == 0 || j == n {
                            1.0
                        } else if j % 2 == 0 {
                            2.0
                        } else {
                            4.0
                        };
                    let bond = if a < 1e-8 {
                        u - 0.5 * a * u * u
                    } else {
                        -(-a * u).exp_m1() / a
                    };
                    expected[0] +=
                        weight * (-a * u).exp() * sigma * corr.vol_rate * (2.0 * h).sqrt() / p;
                    expected[1] += weight * bond * sigma * corr.vol_rate * (2.0 * h).sqrt() / p;
                }
            }
            for j in 0..2 {
                assert!(
                    (c[4][j + 2] - expected[j]).abs() < 2e-11,
                    "H={h} a={a} j={j}: {} != {}",
                    c[4][j + 2],
                    expected[j]
                );
            }
            if a == 0.0 {
                let mut exact = [0.0; 2];
                for (lo, hi, sigma) in [(0.1, 0.3, 0.01), (0.3, 0.9, 0.023), (0.9, 1.4, 0.014)] {
                    for (j, e) in exact.iter_mut().enumerate() {
                        let q = p + j as f64;
                        *e += sigma
                            * corr.vol_rate
                            * (2.0 * h).sqrt()
                            * ((1.4_f64 - lo).powf(q) - (1.4_f64 - hi).powf(q))
                            / q;
                    }
                }
                assert!((c[4][2] - exact[0]).abs() < 1e-15);
                assert!((c[4][3] - exact[1]).abs() < 1e-15);
            }
        }
    }
}

#[test]
fn rough_parameter_domains_and_kernel_cancellation_are_explicit() {
    for h in [0.0, -0.1, 0.5001, f64::NAN, f64::INFINITY] {
        assert!(RoughBergomi::new(h, 1.0, 0.0).is_err());
    }
    for eta in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(RoughBergomi::new(0.1, eta, 0.0).is_err());
    }
    assert!(RoughBergomi::new(0.1, 1.0, 1.1).is_err());
    let m = RoughBergomi::new(0.1, 0.0, 0.0).unwrap();
    assert!(m.average_kernel(0.0, 0.0).is_err());
    assert!((m.average_kernel(1.0, 1.0 + 1e-12).unwrap() - 0.2_f64.sqrt()).abs() < 1e-12);
    assert_eq!(
        RoughBergomi::new(0.5, 1.0, 0.0)
            .unwrap()
            .average_kernel(0.0, 0.25)
            .unwrap(),
        1.0
    );
    let rates = HullWhite1Factor::new(0.2, vec![0.0], vec![0.01]).unwrap();
    assert!(
        m.hybrid_covariance(
            &rates,
            0.0,
            0.0,
            HybridCorrelation::new(0.0, 0.0, 0.0).unwrap()
        )
        .is_err()
    );
    assert!(
        m.hybrid_covariance(
            &rates,
            0.0,
            1.0,
            HybridCorrelation::new(0.2, 0.0, 0.0).unwrap()
        )
        .is_err()
    );
}
