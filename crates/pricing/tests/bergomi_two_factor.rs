use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::{
    BergomiLsvPlan, LsvLeverageSurface, LsvParticleConfig, calibrate_bergomi_lsv,
};
use pricing::mc::{LocalVolTimeGrid, RandomDomain};
use pricing::models::{Bergomi1Factor, Bergomi2Factor};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
        "{a} vs {b}"
    );
}
fn target() -> LocalVarianceGrid {
    LocalVarianceGrid::new(
        vec![0.0, 0.1, 0.2, 0.3],
        vec![-0.8, -0.2, 0.12, 0.45, 0.9],
        (0..20)
            .map(|i| 0.04 + 0.001 * (i % 5) as f64 + 0.002 * (i / 5) as f64)
            .collect(),
        1e-8,
        4.0,
    )
    .unwrap()
}
fn config(trace: bool) -> LsvParticleConfig {
    LsvParticleConfig::new(512, 417, 0.3, 8.0, trace).unwrap()
}
fn factor() -> Bergomi2Factor {
    Bergomi2Factor::new([4.0, 0.3], 0.8, 0.4, [-0.65, -0.25], 0.5).unwrap()
}

#[test]
fn joint_transition_matches_kernel_quadrature_including_singular_brownian_drivers() {
    for (ks, r, rho) in [
        ([4.0, 0.3], [-0.65, -0.25], 0.5),
        ([0.0, 1e-12], [0.2, -0.5], -0.3),
        ([3.0, 0.2], [-1.0, -1.0], 1.0),
        ([0.0, 0.0], [-1.0, -1.0], 1.0),
    ] {
        let f = Bergomi2Factor::new(ks, 0.5, 0.4, r, rho).unwrap();
        let t = f.transition(0.5).unwrap();
        let k = [0.0, ks[0], ks[1]];
        let c = [[1.0, r[0], r[1]], [r[0], 1.0, rho], [r[1], rho, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                let n = 2000;
                let h = 0.5 / n as f64;
                let integral = (0..=n)
                    .map(|p| {
                        let w = if p == 0 || p == n {
                            1.0
                        } else if p % 2 == 0 {
                            2.0
                        } else {
                            4.0
                        };
                        w * (-(k[i] + k[j]) * p as f64 * h).exp()
                    })
                    .sum::<f64>()
                    * h
                    / 3.0;
                close(t.covariance[i][j], c[i][j] * integral, 2e-13);
                close(
                    (0..3).map(|a| t.lower[i][a] * t.lower[j][a]).sum(),
                    t.covariance[i][j],
                    2e-14,
                );
            }
        }
        if ks == [3.0, 0.2] {
            // Three different time kernels retain rank three although all the
            // instantaneous Brownian drivers are perfectly correlated.
            assert!(t.lower[1][1] > 0.01 && t.lower[2][2] > 1e-4);
        }
    }
}

#[test]
fn parameters_validate_marginal_psd_normalization_and_degenerate_weights() {
    let f = factor();
    let w = f.normalized_weights();
    close(
        w[0] * w[0] + w[1] * w[1] + 2.0 * f.factor_correlation() * w[0] * w[1],
        1.0,
        2e-15,
    );
    assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, 0.4, [0.9, 0.9], -0.9).is_err());
    assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, 0.5, [0.0, 0.0], -1.0).is_err());
    for bad in [f64::NAN, f64::INFINITY, -0.01] {
        assert!(Bergomi2Factor::new([bad, 0.1], 0.5, 0.4, [0.0, 0.0], 0.5).is_err());
        assert!(Bergomi2Factor::new([1.0, bad], 0.5, 0.4, [0.0, 0.0], 0.5).is_err());
        assert!(Bergomi2Factor::new([1.0, 0.1], bad, 0.4, [0.0, 0.0], 0.5).is_err());
        assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, bad, [0.0, 0.0], 0.5).is_err());
    }
    for bad in [f64::NAN, f64::INFINITY, -1.01, 1.01] {
        assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, 0.4, [bad, 0.0], 0.5).is_err());
        assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, 0.4, [0.0, bad], 0.5).is_err());
        assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, 0.4, [0.0, 0.0], bad).is_err());
    }
    assert!(Bergomi2Factor::new([1.0, 0.1], 0.5, 1.01, [0.0, 0.0], 0.5).is_err());
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(f.transition(dt).is_err());
    }
}

#[test]
fn zero_second_weight_and_zero_nu_recover_the_one_factor_algorithm() {
    let target = target();
    let grid = LocalVolTimeGrid::compile(target.time_nodes().to_vec(), 1.0).unwrap();
    for nu in [0.0, 0.8] {
        let one = calibrate_bergomi_lsv(
            &target,
            Bergomi1Factor::new(4.0, nu, -0.65).unwrap(),
            100.0,
            config(true),
        )
        .unwrap();
        let two = calibrate_bergomi_lsv(
            &target,
            Bergomi2Factor::new([4.0, 0.3], nu, 0.0, [-0.65, -0.25], 0.5).unwrap(),
            100.0,
            config(true),
        )
        .unwrap();
        for (a, b) in one
            .surface()
            .squared_leverage()
            .iter()
            .zip(two.surface().squared_leverage())
        {
            close(*a, *b, 1e-14);
        }
        let p = two.pricing_plan(&grid).unwrap();
        let z = p.pseudo_shocks(432, 7, RandomDomain::Valuation).unwrap();
        assert_eq!(z.len(), 9); // Do not drop the inactive factor's coordinates.
        let a = one
            .pricing_plan(&grid)
            .unwrap()
            .evolve_path(100.0, &z[..6])
            .unwrap();
        let b = p.evolve_path(100.0, &z).unwrap();
        for (a, b) in a.states().iter().zip(b.states()) {
            close(*a, *b, 2e-14);
        }
        let seeds = [0.0, 0.3, 0.3, 0.4];
        let a = one
            .reverse_leverage(&a.reverse(&seeds).unwrap().squared_leverage)
            .unwrap();
        let b = two
            .reverse_leverage(&b.reverse(&seeds).unwrap().squared_leverage)
            .unwrap();
        for (a, b) in a.iter().zip(b) {
            close(*a, b, 2e-12);
        }
    }
}

#[test]
fn path_reverse_covers_every_shock_block_and_nonflat_leverage_node() {
    let target = target();
    let grid = LocalVolTimeGrid::compile(target.time_nodes().to_vec(), 1.0).unwrap();
    let make = |values| {
        BergomiLsvPlan::new(
            factor(),
            LsvLeverageSurface::new(
                target.time_nodes().to_vec(),
                target.log_moneyness_nodes().to_vec(),
                values,
                100.0,
            )
            .unwrap(),
            &grid,
        )
        .unwrap()
    };
    let p = make(target.values().to_vec());
    let z = vec![0.3, -0.6, 0.8, -0.2, 0.7, -0.4, 0.6, -0.8, 0.2];
    let seeds = [0.2, -0.3, 0.5, 1.0];
    let objective = |p: &BergomiLsvPlan<Bergomi2Factor>, f, z: &[f64]| {
        p.evolve_path(f, z)
            .unwrap()
            .states()
            .iter()
            .zip(seeds)
            .map(|(s, w)| s * w)
            .sum::<f64>()
    };
    let adj = p.evolve_path(101.0, &z).unwrap().reverse(&seeds).unwrap();
    close(
        adj.initial_f,
        (objective(&p, 101.0 + 1e-5, &z) - objective(&p, 101.0 - 1e-5, &z)) / 2e-5,
        2e-8,
    );
    for j in 0..z.len() {
        let mut up = z.clone();
        let mut down = z.clone();
        up[j] += 1e-5;
        down[j] -= 1e-5;
        let a = if j < 3 {
            adj.spot_shocks[j]
        } else {
            adj.orthogonal_shocks[j - 3]
        };
        close(
            a,
            (objective(&p, 101.0, &up) - objective(&p, 101.0, &down)) / 2e-5,
            2e-8,
        );
    }
    for j in 0..20 {
        let mut up = target.values().to_vec();
        let mut down = up.clone();
        up[j] += 1e-6;
        down[j] -= 1e-6;
        close(
            adj.squared_leverage[j],
            (objective(&make(up), 101.0, &z) - objective(&make(down), 101.0, &z)) / 2e-6,
            2e-7,
        );
    }
}

#[test]
fn two_factor_particle_vjp_matches_all_recalibrated_buckets_with_tail_feedback() {
    let target = target();
    let c = calibrate_bergomi_lsv(&target, factor(), 100.0, config(true)).unwrap();
    assert!(c.diagnostics().iter().any(|r| r.extrapolated_nodes > 0));
    let seeds = (0..20).map(|i| (0.7 * i as f64).cos()).collect::<Vec<_>>();
    let adj = c.reverse_leverage(&seeds).unwrap();
    let objective = |j, bump| {
        let mut v = target.values().to_vec();
        v[j] += bump;
        let t = LocalVarianceGrid::new(
            target.time_nodes().to_vec(),
            target.log_moneyness_nodes().to_vec(),
            v,
            1e-8,
            4.0,
        )
        .unwrap();
        calibrate_bergomi_lsv(&t, factor(), 100.0, config(false))
            .unwrap()
            .surface()
            .squared_leverage()
            .iter()
            .zip(&seeds)
            .map(|(v, w)| v * w)
            .sum::<f64>()
    };
    let mut feedback: f64 = 0.0;
    for j in 0..20 {
        close(
            adj[j],
            (objective(j, 1e-6) - objective(j, -1e-6)) / 2e-6,
            5e-6,
        );
        feedback = feedback.max((adj[j] - seeds[j] / c.conditional_moments()[j].second).abs());
    }
    assert!(feedback > 1e-3);
}
