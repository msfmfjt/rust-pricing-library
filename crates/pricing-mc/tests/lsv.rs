use pricing_market::LocalVarianceGrid;
use pricing_mc::lsv::{
    BergomiLsvPlan, LsvLeverageSurface, LsvParticleConfig, calibrate_bergomi_lsv,
};
use pricing_mc::{LocalVolLogEulerPlan, LocalVolTimeGrid, RandomDomain};
use pricing_models::Bergomi1Factor;

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
fn factor() -> Bergomi1Factor {
    Bergomi1Factor::new(2.0, 0.8, -0.65).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
        "actual={a:.12e}, expected={b:.12e}"
    );
}

#[test]
fn zero_vol_of_vol_reduces_to_lv_with_identical_spot_shocks() {
    let target = target();
    let model = calibrate_bergomi_lsv(
        &target,
        Bergomi1Factor::new(3.0, 0.0, -1.0).unwrap(),
        100.0,
        config(true),
    )
    .unwrap();
    assert_eq!(model.surface().squared_leverage(), target.values());
    let grid = LocalVolTimeGrid::compile(target.time_nodes().to_vec(), 1.0).unwrap();
    let plan = model.pricing_plan(&grid).unwrap();
    let shocks = plan.pseudo_shocks(432, 7, RandomDomain::Valuation).unwrap();
    let lsv = plan.evolve_path(100.0, &shocks).unwrap();
    let lv = LocalVolLogEulerPlan::with_constant_forward(grid, 100.0)
        .unwrap()
        .evolve_path(&target, 100.0, &shocks[..3])
        .unwrap();
    for (&a, &b) in lsv.states().iter().zip(lv.states()) {
        close(a, b, 1e-15);
    }
    let seed = (0..20).map(|i| (i as f64).sin()).collect::<Vec<_>>();
    assert_eq!(model.reverse_leverage(&seed).unwrap(), seed);
}

#[test]
fn path_reverse_matches_spot_grid_and_both_gaussian_factors() {
    let target = target();
    let surface = LsvLeverageSurface::new(
        target.time_nodes().to_vec(),
        target.log_moneyness_nodes().to_vec(),
        target.values().to_vec(),
        100.0,
    )
    .unwrap();
    let grid = LocalVolTimeGrid::compile(target.time_nodes().to_vec(), 1.0).unwrap();
    let plan = BergomiLsvPlan::new(factor(), surface.clone(), &grid).unwrap();
    let z = vec![0.3, -0.6, 0.8, -0.2, 0.7, -0.4];
    let seeds = [0.2, -0.3, 0.5, 1.0];
    let objective = |p: &BergomiLsvPlan, f, z: &[f64]| {
        p.evolve_path(f, z)
            .unwrap()
            .states()
            .iter()
            .zip(seeds)
            .map(|(s, w)| s * w)
            .sum::<f64>()
    };
    let adj = plan
        .evolve_path(101.0, &z)
        .unwrap()
        .reverse(&seeds)
        .unwrap();
    let h = 1e-5;
    close(
        adj.initial_f,
        (objective(&plan, 101.0 + h, &z) - objective(&plan, 101.0 - h, &z)) / (2.0 * h),
        2e-8,
    );
    for j in 0..z.len() {
        let mut up = z.clone();
        let mut down = z.clone();
        up[j] += h;
        down[j] -= h;
        let expected = (objective(&plan, 101.0, &up) - objective(&plan, 101.0, &down)) / (2.0 * h);
        let actual = if j < 3 {
            adj.spot_shocks[j]
        } else {
            adj.orthogonal_shocks[j - 3]
        };
        close(actual, expected, 2e-8);
    }
    for j in 0..surface.squared_leverage().len() {
        let eval = |bump: f64| {
            let mut v = surface.squared_leverage().to_vec();
            v[j] += bump;
            let s = LsvLeverageSurface::new(
                surface.times().to_vec(),
                surface.log_nodes().to_vec(),
                v,
                100.0,
            )
            .unwrap();
            objective(&BergomiLsvPlan::new(factor(), s, &grid).unwrap(), 101.0, &z)
        };
        close(
            adj.squared_leverage[j],
            (eval(1e-6) - eval(-1e-6)) / 2e-6,
            2e-7,
        );
    }
}

#[test]
fn calibration_vjp_matches_recalibration_and_includes_feedback() {
    let target = target();
    let model = calibrate_bergomi_lsv(&target, factor(), 100.0, config(true)).unwrap();
    assert!(model.diagnostics().iter().any(|r| r.extrapolated_nodes > 0));
    let seed = (0..20).map(|i| (0.7 * i as f64).cos()).collect::<Vec<_>>();
    let adj = model.reverse_leverage(&seed).unwrap();
    let objective = |j: usize, bump: f64| {
        let mut values = target.values().to_vec();
        values[j] += bump;
        let grid = LocalVarianceGrid::new(
            target.time_nodes().to_vec(),
            target.log_moneyness_nodes().to_vec(),
            values,
            1e-8,
            4.0,
        )
        .unwrap();
        let model = calibrate_bergomi_lsv(&grid, factor(), 100.0, config(false)).unwrap();
        model
            .surface()
            .squared_leverage()
            .iter()
            .zip(&seed)
            .map(|(v, w)| v * w)
            .sum::<f64>()
    };
    let mut feedback = 0.0f64;
    for j in 0..20 {
        close(
            adj[j],
            (objective(j, 1e-6) - objective(j, -1e-6)) / 2e-6,
            5e-6,
        );
        feedback = feedback.max((adj[j] - seed[j] / model.conditional_moments()[j].second).abs());
    }
    assert!(
        feedback > 1e-3,
        "test must exercise calibration feedback, not just the explicit variance ratio"
    );
}

#[test]
fn complete_price_path_to_target_vjp_matches_bump_recalibrate_reprice() {
    let target = target();
    let grid = LocalVolTimeGrid::compile(target.time_nodes().to_vec(), 1.0).unwrap();
    let model = calibrate_bergomi_lsv(&target, factor(), 100.0, config(true)).unwrap();
    let plan = model.pricing_plan(&grid).unwrap();
    let z = plan
        .pseudo_shocks(921, 18, RandomDomain::Valuation)
        .unwrap();
    let seeds = [0.0, 0.3, 0.3, 0.4];
    let path_adj = plan
        .evolve_path(100.0, &z)
        .unwrap()
        .reverse(&seeds)
        .unwrap();
    let adj = model.reverse_leverage(&path_adj.squared_leverage).unwrap();
    let direction = (0..20).map(|i| (i as f64 * 0.37).sin()).collect::<Vec<_>>();
    let bump_price = |bump: f64| {
        let values = target
            .values()
            .iter()
            .zip(&direction)
            .map(|(v, d)| v + bump * d)
            .collect();
        let t = LocalVarianceGrid::new(
            target.time_nodes().to_vec(),
            target.log_moneyness_nodes().to_vec(),
            values,
            1e-8,
            4.0,
        )
        .unwrap();
        let m = calibrate_bergomi_lsv(&t, factor(), 100.0, config(false)).unwrap();
        m.pricing_plan(&grid)
            .unwrap()
            .evolve_path(100.0, &z)
            .unwrap()
            .states()
            .iter()
            .zip(seeds)
            .map(|(v, w)| v * w)
            .sum::<f64>()
    };
    close(
        adj.iter().zip(&direction).map(|(a, d)| a * d).sum(),
        (bump_price(1e-6) - bump_price(-1e-6)) / 2e-6,
        2e-6,
    );
}

#[test]
fn independent_vanilla_repricing_and_martingale_check() {
    let times = (0..=32).map(|i| i as f64 / 32.0).collect::<Vec<_>>();
    let nodes = (0..=30)
        .map(|i| -0.75 + i as f64 * 0.05)
        .collect::<Vec<_>>();
    let target =
        LocalVarianceGrid::new(times.clone(), nodes, vec![0.04; 33 * 31], 1e-8, 4.0).unwrap();
    let model = calibrate_bergomi_lsv(
        &target,
        Bergomi1Factor::new(2.0, 0.7, -0.5).unwrap(),
        100.0,
        LsvParticleConfig::new(4096, 42, 0.12, 10.0, false).unwrap(),
    )
    .unwrap();
    let grid = LocalVolTimeGrid::compile(times, 1.0).unwrap();
    let plan = model.pricing_plan(&grid).unwrap();
    let mut mean = 0.0;
    let mut call = 0.0;
    for i in 0..8192 {
        let shocks = plan.pseudo_shocks(99, i, RandomDomain::Valuation).unwrap();
        for sign in [1.0, -1.0] {
            let z = shocks.iter().map(|z| z * sign).collect::<Vec<_>>();
            let path = plan.evolve_path(100.0, &z).unwrap();
            let f = *path.states().last().unwrap();
            mean += f / 16384.0;
            call += (f - 100.0).max(0.0) / 16384.0;
        }
    }
    assert!((mean - 100.0).abs() < 0.5, "martingale mean={mean}");
    assert!(
        (call - 7.965567455405804).abs() < 0.4,
        "independent ATM call={call}"
    );
}

#[test]
fn time_knot_insertion_tail_policies_and_bad_inputs_are_explicit() {
    let target = target();
    let model = calibrate_bergomi_lsv(&target, factor(), 100.0, config(false)).unwrap();
    assert!(model.reverse_leverage(&[0.0; 20]).is_err());
    let missing = LocalVolTimeGrid::compile(vec![0.0, 0.3], 1.0).unwrap();
    assert!(model.pricing_plan(&missing).is_err());
    let inserted = LocalVolTimeGrid::compile(vec![0.0, 0.05, 0.1, 0.2, 0.3], 1.0).unwrap();
    let plan = model.pricing_plan(&inserted).unwrap();
    assert!(plan.evolve_path(100.0, &[0.0; 8]).is_ok());
    assert!(plan.evolve_path(0.0, &[0.0; 8]).is_err());
    assert!(plan.evolve_path(100.0, &[f64::NAN; 8]).is_err());
    assert!(LsvParticleConfig::new(1, 1, 0.1, 1.0, false).is_err());
    assert!(LsvParticleConfig::new(100, 1, 0.0, 1.0, false).is_err());
    assert!(LsvParticleConfig::new(100, 1, 0.1, 101.0, false).is_err());
    assert_eq!(
        plan.surface().squared_leverage_at(0.05, 100.0).unwrap(),
        plan.surface().squared_leverage_at(0.0, 100.0).unwrap()
    );
}
