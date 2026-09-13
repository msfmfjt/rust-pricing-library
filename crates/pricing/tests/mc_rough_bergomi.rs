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
