use pricing::mc::{LocalVolTimeGrid, hull_white::*, lsv::LsvLeverageSurface};
use pricing::models::{Bergomi1Factor, Bergomi2Factor, HullWhite1Factor, HybridCorrelation};

fn integrate(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let n = 4096;
    let h = (hi - lo) / n as f64;
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(lo + i as f64 * h);
    }
    sum * h / 3.0
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-11 * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn two_factor_hw_joint_covariance_matches_independent_kernel_quadrature() {
    for (a, k, rs, rv, r12, sv) in [
        (0.13, [3.1, 0.24], 0.25, [-0.1, 0.05], 0.4, [-0.6, -0.2]),
        (0.0, [0.0, 1e-9], 0.0, [0.0, 0.0], 0.5, [-0.4, 0.0]),
        (0.7, [0.2, 2.4], 1.0, [1.0, 1.0], 1.0, [1.0, 1.0]),
    ] {
        let rates = HullWhite1Factor::new(a, vec![0.0, 0.37], vec![0.013, 0.021]).unwrap();
        let factor = Bergomi2Factor::new(k, 0.4, 0.35, sv, r12).unwrap();
        let cov = Bergomi2FactorHullWhiteDriverPlan::covariance(factor, &rates, rs, rv, 0.1, 0.9)
            .unwrap();
        let corr = [
            [1.0, sv[0], rs, rs, sv[1]],
            [sv[0], 1.0, rv[0], rv[0], r12],
            [rs, rv[0], 1.0, 1.0, rv[1]],
            [rs, rv[0], 1.0, 1.0, rv[1]],
            [sv[1], r12, rv[1], rv[1], 1.0],
        ];
        let kernel = |i: usize, u: f64, sigma: f64| {
            let t = 0.9 - u;
            match i {
                0 => 1.0,
                1 => (-k[0] * t).exp(),
                2 => sigma * (-a * t).exp(),
                3 => sigma * if a == 0.0 { t } else { -(-a * t).exp_m1() / a },
                4 => (-k[1] * t).exp(),
                _ => unreachable!(),
            }
        };
        for i in 0..5 {
            for j in 0..5 {
                let expected = corr[i][j]
                    * (integrate(|u| kernel(i, u, 0.013) * kernel(j, u, 0.013), 0.1, 0.37)
                        + integrate(|u| kernel(i, u, 0.021) * kernel(j, u, 0.021), 0.37, 0.9));
                close(cov[i][j], expected);
            }
        }
        let grid = LocalVolTimeGrid::compile(vec![0.0, 0.9], 0.9).unwrap();
        Bergomi2FactorHullWhiteDriverPlan::new(factor, &rates, rs, rv, &grid).unwrap();
    }
    let f = Bergomi2Factor::new([3.0, 0.2], 0.4, 0.3, [-0.5, -0.3], 0.4).unwrap();
    let rates = HullWhite1Factor::new(0.1, vec![0.0], vec![0.0]).unwrap();
    assert!(
        Bergomi2FactorHullWhiteDriverPlan::covariance(f, &rates, 0.9, [0.9, 0.9], 0.0, 1.0)
            .is_err()
    );
    assert!(
        Bergomi2FactorHullWhiteDriverPlan::covariance(f, &rates, 0.1, [0.0, f64::NAN], 0.0, 1.0)
            .is_err()
    );
}

#[test]
fn two_factor_hw_preserves_one_factor_limit_and_reverses_leverage() {
    let times = vec![0.0, 0.2, 0.6, 1.0];
    let grid = LocalVolTimeGrid::compile(times.clone(), 1.0).unwrap();
    let surface = LsvLeverageSurface::new(
        times,
        vec![-0.5, 0.0, 0.5],
        vec![
            0.05, 0.045, 0.04, 0.052, 0.044, 0.039, 0.05, 0.046, 0.041, 0.05, 0.047, 0.043,
        ],
        100.0,
    )
    .unwrap();
    let rates = HullWhite1Factor::new(0.15, vec![0.0, 0.35], vec![0.012, 0.018]).unwrap();
    let corr = HybridCorrelation::new(-0.5, 0.2, -0.1).unwrap();
    let one = HullWhiteEquityPlan::new(
        rates.clone(),
        HybridEquityVolatility::BergomiLsv {
            factor: Bergomi1Factor::new(2.0, 0.35, -0.5).unwrap(),
            leverage: surface.clone(),
        },
        corr,
        &grid,
    )
    .unwrap();
    let build = |theta, leverage| {
        HullWhiteEquityPlan::new(
            rates.clone(),
            HybridEquityVolatility::Bergomi2FactorLsv {
                factor: Bergomi2Factor::new([2.0, 0.12], 0.35, theta, [-0.5, -0.2], 0.4).unwrap(),
                second_vol_rate_correlation: 0.05,
                leverage,
            },
            corr,
            &grid,
        )
        .unwrap()
    };
    let two = build(0.0, surface.clone());
    let shocks = two
        .pseudo_shocks(45, 7, pricing::mc::RandomDomain::Valuation)
        .unwrap();
    let p1 = one.evolve_path(100.0, &shocks[..12]).unwrap();
    let p2 = two.evolve_path(100.0, &shocks).unwrap();
    assert_eq!(p1, p2);
    let two = build(0.4, surface.clone());
    let seeds = [0.1, 0.3, -0.4, 0.9];
    let path = two.record_path(100.0, &shocks).unwrap();
    let adj = path.reverse(&seeds).unwrap();
    let value = |s| {
        build(0.4, s)
            .evolve_path(100.0, &shocks)
            .unwrap()
            .iter()
            .zip(seeds)
            .map(|(s, b)| s.normalized_equity * b)
            .sum::<f64>()
    };
    for j in 0..surface.squared_leverage().len() {
        let shifted = |h| {
            let mut v = surface.squared_leverage().to_vec();
            v[j] += h;
            LsvLeverageSurface::new(
                surface.times().to_vec(),
                surface.log_nodes().to_vec(),
                v,
                100.0,
            )
            .unwrap()
        };
        let fd = (value(shifted(1e-7)) - value(shifted(-1e-7))) / 2e-7;
        assert!(
            (adj.squared_leverage[j] - fd).abs() < 2e-6,
            "node {j}: {} vs {fd}",
            adj.squared_leverage[j]
        );
    }
    assert!(two.record_path(100.0, &shocks[..3]).is_err());
}
