use pricing::mc::{LocalVolTimeGrid, hull_white::RoughBergomiDriverPlan};
use pricing::models::{HullWhite1Factor, HybridCorrelation, RoughBergomi};

#[test]
fn rough_driver_covariance_adaptedness_and_singular_correlations() {
    let times = vec![0.0, 0.07, 0.3, 0.55, 1.0];
    let grid = LocalVolTimeGrid::compile(times.clone(), 1.0).unwrap();
    let rates = HullWhite1Factor::new(0.3, vec![0.0, 0.15, 0.62], vec![0.01, 0.02, 0.014]).unwrap();
    for h in [0.01, 0.1, 0.49, 0.5] {
        for (sv, sr, vr) in [
            (-0.6, 0.25, -0.15),
            (1.0, 0.3, 0.3),
            (-1.0, 0.3, -0.3),
            (0.0, 0.0, 1.0),
        ] {
            let model = RoughBergomi::new(h, 1.2, sv).unwrap();
            let corr = HybridCorrelation::new(sv, sr, vr).unwrap();
            let plan = RoughBergomiDriverPlan::new(model, &rates, corr, &grid).unwrap();
            let mut cov = [[0.0; 5]; 5];
            for d in 0..20 {
                let mut z = vec![0.0; 20];
                z[d] = 1.0;
                let x = plan.evolve(&z).unwrap();
                for &v in &x[..=d % 4] {
                    assert_eq!(v, 0.0, "future noise used");
                }
                let opposite = plan
                    .evolve(&z.iter().map(|v| -v).collect::<Vec<_>>())
                    .unwrap();
                for i in 0..5 {
                    assert_eq!(x[i], -opposite[i]);
                    for j in 0..5 {
                        cov[i][j] += x[i] * x[j];
                    }
                }
            }
            for i in 1..5 {
                assert!((cov[i][i] - plan.variances()[i]).abs() < 3e-12);
                for j in i + 1..5 {
                    let mut expected = 0.0;
                    for k in 0..i - 1 {
                        expected += model
                            .average_kernel(times[i] - times[k + 1], times[i] - times[k])
                            .unwrap()
                            * model
                                .average_kernel(times[j] - times[k + 1], times[j] - times[k])
                                .unwrap()
                            * (times[k + 1] - times[k]);
                    }
                    expected += model
                        .average_kernel(times[j] - times[i], times[j] - times[i - 1])
                        .unwrap()
                        * model.average_kernel(0.0, times[i] - times[i - 1]).unwrap()
                        * (times[i] - times[i - 1]);
                    assert!(
                        (cov[i][j] - expected).abs() < 3e-12,
                        "H={h}: {} != {expected}",
                        cov[i][j]
                    );
                }
            }
            assert!(plan.evolve(&[0.0; 16]).is_err());
            assert!(plan.evolve(&[f64::NAN; 20]).is_err());
        }
    }
}

#[test]
fn hybrid_variance_refines_towards_the_volterra_variance() {
    let rates = HullWhite1Factor::new(0.0, vec![0.0], vec![0.0]).unwrap();
    let corr = HybridCorrelation::new(0.0, 0.0, 0.0).unwrap();
    for h in [0.05, 0.1, 0.3] {
        let mut previous = 1.0;
        for n in [8, 32, 128] {
            let grid = LocalVolTimeGrid::compile(vec![1.0], 1.0 / n as f64).unwrap();
            let plan = RoughBergomiDriverPlan::new(
                RoughBergomi::new(h, 1.0, 0.0).unwrap(),
                &rates,
                corr,
                &grid,
            )
            .unwrap();
            let mse = 1.0 - plan.variances().last().unwrap();
            assert!(mse > 0.0 && mse < previous);
            previous = mse;
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // Explicit small covariance matrices in the independent oracle.
fn external_rough_histories_preserve_dated_cross_time_covariance() {
    let times = [0.0, 0.07, 0.3, 0.55, 1.0];
    let correlations = [0.6, -0.2, 0.4, -0.5];
    let grid = LocalVolTimeGrid::compile(times.to_vec(), 1.0).unwrap();
    let rates = HullWhite1Factor::new(0.0, vec![0.0], vec![0.0]).unwrap();
    let corr = HybridCorrelation::new(0.0, 0.0, 0.0).unwrap();
    for hs in [[0.1, 0.35], [0.499, 0.03]] {
        let plans = hs.map(|h| {
            RoughBergomiDriverPlan::new(
                RoughBergomi::new(h, 0.8, 0.0).unwrap(),
                &rates,
                corr,
                &grid,
            )
            .unwrap()
        });
        let mut actual = [[0.0; 5]; 5];
        for step in 0..4 {
            let dt = times[step + 1] - times[step];
            // Independent power-kernel covariance for [dV_A,dV_B,J_A,J_B].
            let powers = [0.0, 0.0, hs[0] - 0.5, hs[1] - 0.5];
            let coeffs = [1.0, 1.0, (2.0 * hs[0]).sqrt(), (2.0 * hs[1]).sqrt()];
            let mut lower = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..=i {
                    let rho = if i % 2 == j % 2 {
                        1.0
                    } else {
                        correlations[step]
                    };
                    let power = 1.0 + powers[i] + powers[j];
                    let cov = rho * coeffs[i] * coeffs[j] * dt.powf(power) / power;
                    let residual = cov - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                    lower[i][j] = if i == j {
                        residual.sqrt()
                    } else {
                        residual / lower[j][j]
                    };
                }
            }
            for basis in 0..4 {
                let paths: [Vec<f64>; 2] = std::array::from_fn(|asset| {
                    let mut dw = [0.0; 4];
                    let mut near = [0.0; 4];
                    dw[step] = lower[asset][basis];
                    near[step] = lower[asset + 2][basis];
                    let x = plans[asset].evolve_with_innovations(&dw, &near).unwrap();
                    assert!(x[..=step].iter().all(|&v| v == 0.0), "future noise used");
                    let opposite = plans[asset]
                        .evolve_with_innovations(&dw.map(|v| -v), &near.map(|v| -v))
                        .unwrap();
                    assert!(x.iter().zip(opposite).all(|(a, b)| *a == -b));
                    x
                });
                for i in 0..5 {
                    for j in 0..5 {
                        actual[i][j] += paths[0][i] * paths[1][j];
                    }
                }
            }
        }
        let average = |h: f64, t: f64, k: usize| {
            let p = h + 0.5;
            (2.0 * h).sqrt() * ((t - times[k]).powf(p) - (t - times[k + 1]).powf(p))
                / (p * (times[k + 1] - times[k]))
        };
        for i in 1..5 {
            for j in 1..5 {
                let expected: f64 = (0..i.min(j))
                    .map(|k| {
                        let dt = times[k + 1] - times[k];
                        let c = if k + 1 == i && k + 1 == j {
                            2.0 * (hs[0] * hs[1]).sqrt() * dt.powf(hs[0] + hs[1]) / (hs[0] + hs[1])
                        } else {
                            average(hs[0], times[i], k) * average(hs[1], times[j], k) * dt
                        };
                        correlations[k] * c
                    })
                    .sum();
                assert!((actual[i][j] - expected).abs() < 3e-12);
            }
        }
        assert!(
            plans[0]
                .evolve_with_innovations(&[0.0; 3], &[0.0; 4])
                .is_err()
        );
        assert!(
            plans[0]
                .evolve_with_innovations(&[0.0; 4], &[f64::NAN; 4])
                .is_err()
        );
    }
}

#[test]
fn rough_joint_kernel_input_validation() {
    let model = RoughBergomi::new(0.1, 0.8, -0.6).unwrap();
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(model.near_near_covariance(model, dt).is_err());
        assert!(model.near_ou_covariance(0.3, dt).is_err());
    }
    for k in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(model.near_ou_covariance(k, 0.2).is_err());
    }
}
