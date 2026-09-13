use pricing_core::CurveId;
use pricing_market::LogLinearDiscountCurve;
use pricing_models::{HullWhite1Factor, HybridCorrelation};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a:.17e} != {b:.17e}");
}

#[test]
fn joint_covariance_matches_independent_kernel_quadrature() {
    for a in [0.0, 1e-12, 0.2, 2.0] {
        let model =
            HullWhite1Factor::new(a, vec![0.0, 0.3, 0.9], vec![0.01, 0.023, 0.014]).unwrap();
        let corr = HybridCorrelation::new(-0.6, 0.25, -0.15).unwrap();
        let start = 0.1;
        let end = 1.4;
        let k = 0.8;
        let actual = model.transition(start, end, k, corr).unwrap();
        let mut expected = [[0.0; 4]; 4];
        let correlations = [
            [1.0, -0.6, 0.25, 0.25],
            [-0.6, 1.0, -0.15, -0.15],
            [0.25, -0.15, 1.0, 1.0],
            [0.25, -0.15, 1.0, 1.0],
        ];
        for (lo, hi, sigma) in [(0.1, 0.3, 0.01), (0.3, 0.9, 0.023), (0.9, 1.4, 0.014)] {
            let n = 2000;
            let dt = (hi - lo) / n as f64;
            for p in 0..=n {
                let u = lo + p as f64 * dt;
                let v = end - u;
                let integral = if a < 1e-8 {
                    v - a * v * v / 2.0
                } else {
                    (1.0 - (-a * v).exp()) / a
                };
                let f = [
                    1.0,
                    (-k * v).exp(),
                    sigma * (-a * v).exp(),
                    sigma * integral,
                ];
                let weight = dt / 3.0
                    * if p == 0 || p == n {
                        1.0
                    } else if p % 2 == 0 {
                        2.0
                    } else {
                        4.0
                    };
                for i in 0..4 {
                    for j in 0..4 {
                        expected[i][j] += weight * f[i] * f[j] * correlations[i][j];
                    }
                }
            }
        }
        for (i, row) in expected.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                close(actual.covariance[i][j], e, 3e-13);
            }
        }
    }
}

#[test]
fn ho_lee_limit_and_curve_fit_include_piecewise_volatility() {
    let model = HullWhite1Factor::new(0.0, vec![0.0], vec![0.02]).unwrap();
    let t = 3.0;
    close(
        model.integrated_variance(t).unwrap(),
        0.02_f64.powi(2) * t.powi(3) / 3.0,
        1e-16,
    );
    close(
        model.rate_shift(t).unwrap(),
        0.02_f64.powi(2) * t * t / 2.0,
        1e-16,
    );
    let curve =
        LogLinearDiscountCurve::new(CurveId::new(1), vec![0.0, 1.0, 3.0], vec![1.0, 1.01, 0.97])
            .unwrap();
    let model = HullWhite1Factor::new(0.1, vec![0.0, 0.5], vec![0.01, 0.025]).unwrap();
    close(
        model.bond_price(&curve, 0.0, 3.0, 0.0).unwrap(),
        0.97,
        1e-15,
    );
    close(
        model.bond_price(&curve, 0.0, 1.0, 0.0).unwrap(),
        1.01,
        1e-15,
    );
    let call = model.bond_option(&curve, 1.0, 3.0, 0.94, true).unwrap();
    let put = model.bond_option(&curve, 1.0, 3.0, 0.94, false).unwrap();
    close(call - put, 0.97 - 0.94 * 1.01, 1e-14);
    let deterministic = HullWhite1Factor::new(0.1, vec![0.0], vec![0.0]).unwrap();
    close(
        deterministic.bond_price(&curve, 1.0, 3.0, 0.0).unwrap(),
        0.97 / 1.01,
        1e-15,
    );
}

#[test]
fn validation_rejects_inconsistent_grids_and_correlations() {
    assert!(HullWhite1Factor::new(-0.1, vec![0.0], vec![0.01]).is_err());
    assert!(HullWhite1Factor::new(0.1, vec![0.1], vec![0.01]).is_err());
    assert!(HullWhite1Factor::new(0.1, vec![0.0, 0.0], vec![0.01, 0.02]).is_err());
    assert!(HullWhite1Factor::new(0.1, vec![0.0], vec![f64::NAN]).is_err());
    assert!(HybridCorrelation::new(0.9, 0.9, -0.9).is_err());
    assert!(HybridCorrelation::new(1.0, -0.5, -0.5).is_ok());
    let model = HullWhite1Factor::new(0.1, vec![0.0], vec![0.01]).unwrap();
    assert!(model.relative_bond(2.0, 1.0, 0.0).is_err());
}
