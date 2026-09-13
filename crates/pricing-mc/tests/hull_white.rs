use pricing_mc::hull_white::{HullWhiteEquityPlan, HybridEquityVolatility};
use pricing_mc::{LocalVolTimeGrid, RandomDomain};
use pricing_models::{HullWhite1Factor, HybridCorrelation};

#[test]
fn simulated_innovations_reconstruct_all_joint_covariances() {
    for a in [0.0, 1e-12, 0.3] {
        let rates = HullWhite1Factor::new(a, vec![0.0, 0.4], vec![0.01, 0.024]).unwrap();
        let corr = HybridCorrelation::new(0.0, -0.65, 0.0).unwrap();
        let grid = LocalVolTimeGrid::compile(vec![1.0], 1.0).unwrap();
        let plan = HullWhiteEquityPlan::new(
            rates.clone(),
            HybridEquityVolatility::BlackScholes(0.2),
            corr,
            &grid,
        )
        .unwrap();
        let shift = rates.integrated_shift(0.0, 1.0).unwrap();
        let mut columns = [[0.0; 4]; 4];
        for (j, column) in columns.iter_mut().enumerate() {
            let mut z = [0.0; 4];
            z[j] = 1.0;
            let states = plan.evolve_path(100.0, &z).unwrap();
            let s = states[1];
            *column = [
                ((s.normalized_equity / 100.0).ln() - s.integrated_rate_factor - shift + 0.02)
                    / 0.2,
                s.volatility_factor,
                s.rate_factor,
                s.integrated_rate_factor,
            ];
        }
        let expected = rates.transition(0.0, 1.0, 0.0, corr).unwrap().covariance;
        for (i, row) in expected.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                let actual: f64 = columns.iter().map(|c| c[i] * c[j]).sum();
                assert!((actual - e).abs() < 3e-14, "{i},{j}: {actual} vs {e}");
            }
        }
    }
}

#[test]
fn discount_curve_and_discounted_equity_are_martingales() {
    let rates = HullWhite1Factor::new(0.1, vec![0.0, 0.7], vec![0.012, 0.023]).unwrap();
    let grid = LocalVolTimeGrid::compile(vec![2.0], 0.25).unwrap();
    let plan = HullWhiteEquityPlan::new(
        rates.clone(),
        HybridEquityVolatility::BlackScholes(0.2),
        HybridCorrelation::new(0.0, 0.4, 0.0).unwrap(),
        &grid,
    )
    .unwrap();
    let mut sums = [0.0; 2];
    let mut squares = [0.0; 2];
    let count = 16384;
    for p in 0..count {
        let z = plan
            .pseudo_shocks(981, p, RandomDomain::Diagnostics)
            .unwrap();
        let mut values = [0.0; 2];
        for sign in [-1.0, 1.0] {
            let path = plan
                .evolve_path(100.0, &z.iter().map(|v| v * sign).collect::<Vec<_>>())
                .unwrap();
            let s = path.last().unwrap();
            let d = rates
                .relative_discount(2.0, s.integrated_rate_factor)
                .unwrap();
            values[0] += 0.5 * d;
            values[1] += 0.5 * d * s.normalized_equity;
        }
        for j in 0..2 {
            sums[j] += values[j];
            squares[j] += values[j] * values[j];
        }
    }
    for j in 0..2 {
        let n = count as f64;
        let mean = sums[j] / n;
        let se = ((squares[j] - sums[j] * sums[j] / n) / (n - 1.0) / n).sqrt();
        let target = [1.0, 100.0][j];
        assert!(
            (mean - target).abs() < 5.0 * se,
            "{mean} vs {target}, se={se}"
        );
    }
}

#[test]
fn singular_and_deterministic_limits_remain_valid() {
    let grid = LocalVolTimeGrid::compile(vec![1.0], 1.0).unwrap();
    for sigma in [0.0, 0.02] {
        let rates = HullWhite1Factor::new(0.0, vec![0.0], vec![sigma]).unwrap();
        let plan = HullWhiteEquityPlan::new(
            rates,
            HybridEquityVolatility::BlackScholes(0.0),
            HybridCorrelation::new(0.0, 1.0, 0.0).unwrap(),
            &grid,
        )
        .unwrap();
        assert!(plan.evolve_path(100.0, &[0.3, -0.2, 0.4, 0.6]).is_ok());
        assert!(plan.evolve_path(100.0, &[0.0]).is_err());
        assert!(plan.evolve_path(100.0, &[f64::NAN; 4]).is_err());
    }
}

#[test]
fn incompatible_small_market_variance_is_rejected_without_clipping() {
    use pricing_mc::hull_white::{HullWhiteLsvTarget, calibrate_hull_white_lsv};
    use pricing_mc::lsv::LsvParticleConfig;
    use pricing_models::Bergomi1Factor;
    let target = HullWhiteLsvTarget::flat(
        0.001,
        vec![0.0, 0.5, 1.0],
        vec![-0.002, 0.0, 0.002],
        1e-8,
        4.0,
    )
    .unwrap();
    let err = calibrate_hull_white_lsv(
        &target,
        Bergomi1Factor::new(2.0, 0.0, 0.0).unwrap(),
        &HullWhite1Factor::new(0.1, vec![0.0], vec![0.1]).unwrap(),
        HybridCorrelation::new(0.0, 0.5, 0.0).unwrap(),
        100.0,
        &LsvParticleConfig::new(1024, 321, 0.1, 10.0, false).unwrap(),
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("non_positive_rate_corrected_variance")
    );
}
