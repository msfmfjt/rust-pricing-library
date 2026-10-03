//! Fixed-model state/Spot VJPs and full discrete particle target pullbacks.
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::{LsvLeverageSurface, LsvParticleConfig};
use pricing::mc::{DeterministicExecutor, ExecutionPolicy, LocalVolTimeGrid, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn families() -> Vec<RoughVolatilityModel> {
    let h = RoughHeston::new(0.2, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
    let c = ForwardVarianceCurve::constant(0.04).unwrap();
    vec![
        h.clone().into(),
        LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.3, 0.25, 0.05, 0.035)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.2, -0.6, vec![0.35, 0.65], vec![0.25, 0.6], c.clone())
            .unwrap()
            .into(),
        RoughSabr::new(0.2, 0.4, -0.6, 1.0, c).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.12, -1.7, None).unwrap().into(),
    ]
}
fn grid() -> LocalVarianceGrid {
    LocalVarianceGrid::new(
        vec![0.0, 0.15, 0.4, 0.7, 1.0],
        vec![-0.6, -0.2, 0.13, 0.45, 0.8],
        (0..25)
            .map(|i| 0.04 + 0.002 * (i % 5) as f64 + 0.001 * (i / 5) as f64)
            .collect(),
        1e-8,
        4.0,
    )
    .unwrap()
}
fn particles(trace: bool) -> LsvParticleConfig {
    LsvParticleConfig::new(128, 429, 0.5, 3.0, trace).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{a:.14e} vs {b:.14e}; gap {}",
        (a - b).abs()
    );
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn parse(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn payload(lsv: bool, qmc: bool, spot_bump: f64, node_bump: f64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(100.0 + spot_bump);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    if lsv {
        let g = grid();
        let values = g
            .values()
            .iter()
            .enumerate()
            .map(|(i, v)| v + node_bump * (0.7 * i as f64).cos())
            .collect::<Vec<_>>();
        v["model"] = json!({"type":"local_volatility","local_variance_grid":{
            "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),
            "shape":[5,5],"values":values,"floor":1e-8,"cap":4.0}});
    }
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":4,
        "master_scramble_seed":819,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":128,"master_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    };
    v
}

#[test]
fn all_six_initial_state_reverse_matches_two_bumps_and_retains_primal() {
    for model in families() {
        let p = RoughVolatilityPathPlan::compile(model, vec![0.0, 0.08, 0.21, 0.6, 1.0]).unwrap();
        for i in 0..16 {
            let z = p.pseudo_shocks(72, i, RandomDomain::Valuation);
            let r = p.evolve_recorded_path(100.0, &z).unwrap();
            assert_eq!(r.path(), &p.evolve_path(100.0, &z).unwrap());
            let seeds = [0.3, -0.4, 0.1, 0.7, 1.2];
            let bar = r.reverse_initial_forward(&seeds).unwrap();
            for eps in [1e-3, 5e-4] {
                let up = p.evolve_path(100.0 + eps, &z).unwrap();
                let dn = p.evolve_path(100.0 - eps, &z).unwrap();
                close(
                    bar,
                    (dot(&up.forwards, &seeds) - dot(&dn.forwards, &seeds)) / (2.0 * eps),
                    2e-9,
                );
            }
            assert!(r.reverse_initial_forward(&[0.0]).is_err());
        }
    }
}

#[test]
fn fixed_leverage_path_reverse_covers_six_families_and_zero_variance_steps() {
    let g = grid();
    let time = LocalVolTimeGrid::compile(g.time_nodes().to_vec(), 1.0).unwrap();
    let ell = (0..25).map(|i| 0.9 + 0.02 * i as f64).collect::<Vec<_>>();
    let direction = (0..25).map(|i| (i as f64 + 0.1).sin()).collect::<Vec<_>>();
    let seeds = [0.2, -0.4, 0.7, 0.2, 1.0];
    for model in families() {
        let build = |e: f64| {
            RoughFamilyLsvPlan::new(
                model.clone(),
                LsvLeverageSurface::new(
                    g.time_nodes().to_vec(),
                    g.log_moneyness_nodes().to_vec(),
                    ell.iter().zip(&direction).map(|(v, d)| v + e * d).collect(),
                    100.0,
                )
                .unwrap(),
                &time,
            )
            .unwrap()
        };
        let p = build(0.0);
        let z = p.pseudo_shocks(19, 1, RandomDomain::Valuation);
        let r = p.evolve_path(99.0, &z).unwrap();
        let a = r.reverse(&seeds).unwrap();
        let mut states = Vec::new();
        p.evolve_states(99.0, &z, &mut states).unwrap();
        assert_eq!(states, r.states());
        for e in [1e-5, 5e-6] {
            close(
                a.initial_forward,
                (dot(p.evolve_path(99.0 + e, &z).unwrap().states(), &seeds)
                    - dot(p.evolve_path(99.0 - e, &z).unwrap().states(), &seeds))
                    / (2.0 * e),
                2e-8,
            );
            close(
                dot(&a.squared_leverage, &direction),
                (dot(build(e).evolve_path(99.0, &z).unwrap().states(), &seeds)
                    - dot(build(-e).evolve_path(99.0, &z).unwrap().states(), &seeds))
                    / (2.0 * e),
                2e-8,
            );
        }
    }
    let model = RoughHeston::new(0.1, 0.04, 0.7, 0.055, 0.5, -0.6).unwrap();
    let path = RoughVolatilityPathPlan::compile(model.clone().into(), vec![0.0, 0.5, 1.0]).unwrap();
    let normals = [0.0, 0.0, -5.0, 0.0, -5.0, 0.0];
    assert_eq!(path.evolve_path(100.0, &normals).unwrap().variances[1], 0.0);
    let p = RoughFamilyLsvPlan::new(
        model.into(),
        LsvLeverageSurface::new(vec![0.0, 0.5, 1.0], vec![-1.0, 1.0], vec![1.0; 6], 100.0).unwrap(),
        &LocalVolTimeGrid::compile(vec![0.0, 0.5, 1.0], 1.0).unwrap(),
    )
    .unwrap();
    let r = p.evolve_path(100.0, &normals).unwrap();
    assert_eq!(r.states()[1], r.states()[2]);
    let a = r.reverse(&[0.0, 0.0, 1.0]).unwrap();
    assert!(a.squared_leverage.iter().all(|v| v.is_finite()));
    assert_eq!(&a.squared_leverage[2..], &[0.0; 4]);
}

#[test]
fn six_family_particle_pullbacks_include_conditional_estimator_and_initial_moments() {
    let g = grid();
    let direction = (0..25).map(|i| (i as f64 * 0.8).cos()).collect::<Vec<_>>();
    let seeds = (0..25).map(|i| (i as f64 * 0.4).sin()).collect::<Vec<_>>();
    for model in families() {
        let build = |b: f64, trace: bool| {
            let target = LocalVarianceGrid::new(
                g.time_nodes().to_vec(),
                g.log_moneyness_nodes().to_vec(),
                g.values()
                    .iter()
                    .zip(&direction)
                    .map(|(v, d)| v + b * d)
                    .collect(),
                g.floor(),
                g.cap(),
            )
            .unwrap();
            calibrate_rough_family_lsv(&target, model.clone(), 100.0, particles(trace)).unwrap()
        };
        let c = build(0.0, true);
        let a = c.reverse_leverage(&seeds).unwrap();
        for eps in [1e-7, 5e-8] {
            let up = build(eps, false);
            let dn = build(-eps, false);
            assert_eq!(
                c.conditional_moments()
                    .iter()
                    .map(|m| m.source_node)
                    .collect::<Vec<_>>(),
                up.conditional_moments()
                    .iter()
                    .map(|m| m.source_node)
                    .collect::<Vec<_>>()
            );
            close(
                dot(&a, &direction),
                (dot(up.surface().squared_leverage(), &seeds)
                    - dot(dn.surface().squared_leverage(), &seeds))
                    / (2.0 * eps),
                2e-6,
            );
        }
        let initial = c.conditional_moments()[0].second;
        assert!(initial > 0.001 && initial < 0.2); // never assume the base variance is one
        for j in 0..5 {
            close(
                c.surface().squared_leverage()[j] * initial,
                g.values()[j],
                2e-14,
            );
        }
        let parallel = calibrate_rough_family_lsv_parallel(
            &g,
            model.clone(),
            100.0,
            particles(true),
            &DeterministicExecutor::new(ExecutionPolicy::new(3, Some(64)).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(c.surface(), parallel.surface());
        assert_eq!(a, parallel.reverse_leverage(&seeds).unwrap());
        assert!(build(0.0, false).reverse_leverage(&seeds).is_err());
    }
}

#[test]
fn public_spot_delta_includes_cash_escrow_and_replays_across_workers() {
    for model in families() {
        for qmc in [false, true] {
            let build = |b, threads| {
                RoughVolatilityPricingPlan::compile(
                    &parse(payload(false, qmc, b, 0.0)),
                    model.clone(),
                    0.2,
                    ExecutionPolicy::new(threads, Some(64)).unwrap(),
                )
                .unwrap()
            };
            let p = build(0.0, 1);
            let a = p.evaluate_delta().unwrap();
            let parallel = build(0.0, 3).evaluate_delta().unwrap();
            assert_eq!(a.price.value, p.evaluate().unwrap().value);
            assert_eq!(a.price.standard_error, p.evaluate().unwrap().standard_error);
            assert_eq!(a.delta, parallel.delta);
            assert_eq!(a.delta_standard_error, parallel.delta_standard_error);
            for eps in [1e-4, 5e-5] {
                close(
                    a.delta,
                    (build(eps, 1).evaluate().unwrap().value
                        - build(-eps, 1).evaluate().unwrap().value)
                        / (2.0 * eps),
                    2e-8,
                );
            }
            assert!(a.delta_standard_error.is_finite());
        }
    }
}

#[test]
fn public_lsv_recalibrated_node_risk_matches_recalibrated_price_bumps() {
    for model in families() {
        for qmc in [false, true] {
            let build = |b, workers, trace| {
                RoughFamilyLsvPricingPlan::compile(
                    &parse(payload(true, qmc, 0.0, b)),
                    model.clone(),
                    particles(trace),
                    ExecutionPolicy::new(workers, Some(64)).unwrap(),
                )
                .unwrap()
            };
            let p = build(0.0, 1, true);
            let a = p.evaluate_local_variance_risk().unwrap();
            assert_eq!(a.price.value, p.evaluate().unwrap().value);
            let other = build(0.0, 3, true).evaluate_local_variance_risk().unwrap();
            assert_eq!(a.node_adjoints, other.node_adjoints);
            assert_eq!(a.standard_errors, other.standard_errors);
            assert_eq!(a.standard_errors.is_some(), qmc);
            let actual = a
                .node_adjoints
                .iter()
                .enumerate()
                .map(|(i, a)| a * (0.7 * i as f64).cos())
                .sum();
            for eps in [1e-7, 5e-8] {
                close(
                    actual,
                    (build(eps, 1, false).evaluate().unwrap().value
                        - build(-eps, 1, false).evaluate().unwrap().value)
                        / (2.0 * eps),
                    3e-6,
                );
            }
            assert!(build(0.0, 1, false).evaluate_local_variance_risk().is_err());
        }
    }
}

#[test]
fn non_log_sabr_lsv_and_resource_excess_are_rejected_but_pure_delta_is_available() {
    for beta in [0.0, 0.5] {
        let model: RoughVolatilityModel = RoughSabr::new(
            0.2,
            0.2,
            -0.4,
            beta,
            ForwardVarianceCurve::constant(0.04).unwrap(),
        )
        .unwrap()
        .into();
        assert!(
            calibrate_rough_family_lsv(&grid(), model.clone(), 100.0, particles(true)).is_err()
        );
        let p = RoughVolatilityPathPlan::compile(model, vec![0.0, 0.2, 1.0]).unwrap();
        let z = p.pseudo_shocks(1, 0, RandomDomain::Valuation);
        let a = p
            .evolve_recorded_path(100.0, &z)
            .unwrap()
            .reverse_initial_forward(&[0.0, 0.0, 1.0])
            .unwrap();
        let eps = 1e-3;
        close(
            a,
            (p.evolve_path(100.0 + eps, &z).unwrap().forwards[2]
                - p.evolve_path(100.0 - eps, &z).unwrap().forwards[2])
                / (2.0 * eps),
            1e-8,
        );
    }
    let model = families().remove(0);
    assert!(
        calibrate_rough_family_lsv(
            &grid(),
            model,
            100.0,
            LsvParticleConfig::new(usize::MAX, 1, 0.5, 2.0, false).unwrap()
        )
        .is_err()
    );
}

#[test]
fn public_lsv_native_rqmc_layout_matches_independent_payoff_reconstruction() {
    use pricing::mc::{BrownianBridgePlan, EngineConfig, RqmcPlan, inverse_standard_normal};
    let mut v = payload(true, true, 0.0, 0.0);
    v["market"]["discrete_dividends"] = json!([]);
    v["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
    v["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
    let req = parse(v);
    let EngineConfig::RandomizedQuasiMonteCarlo(config) = req.engine() else {
        unreachable!()
    };
    for model in families() {
        let p = RoughFamilyLsvPricingPlan::compile(
            &req,
            model.clone(),
            particles(false),
            ExecutionPolicy::new(1, Some(64)).unwrap(),
        )
        .unwrap();
        let times = p.calibration().surface().times();
        let grid = LocalVolTimeGrid::compile(times.to_vec(), 1.0).unwrap();
        let path = p.calibration().pricing_plan(&grid).unwrap();
        let n = times.len() - 1;
        let blocks = match model {
            RoughVolatilityModel::Rfsv(_) | RoughVolatilityModel::QuadraticRoughHeston(_) => 1,
            _ => 2,
        };
        let bridge = BrownianBridgePlan::compile(times.to_vec(), 1).unwrap();
        let q = RqmcPlan::compile(config, path.random_dimension()).unwrap();
        let mut means = Vec::new();
        for s in 0..config.scramble_count().get() {
            let mut total = 0.0;
            for i in 0..config.points_per_scramble().get() {
                let mut z = (0..path.random_dimension())
                    .map(|d| inverse_standard_normal(q.uniform(s, i, d).unwrap()).unwrap())
                    .collect::<Vec<_>>();
                // Hybrid near-cell residuals and RFSV levels are NOT Brownian blocks.
                for b in z[..blocks * n].chunks_exact_mut(n) {
                    let mapped = bridge.apply_one_factor(b).unwrap();
                    b.copy_from_slice(&mapped);
                }
                for sign in [1.0, -1.0] {
                    let z = z.iter().map(|v| sign * v).collect::<Vec<_>>();
                    let path = path.evolve_path(100.0, &z).unwrap();
                    total += 0.5 * (path.states()[n] - 100.0).max(0.0);
                }
            }
            means.push(total / config.points_per_scramble().get() as f64);
        }
        let mean = means.iter().sum::<f64>() / means.len() as f64;
        let se = (means.iter().map(|m| (m - mean).powi(2)).sum::<f64>()
            / (means.len() * (means.len() - 1)) as f64)
            .sqrt();
        let result = p.evaluate().unwrap();
        close(result.value, mean, 2e-13);
        close(result.standard_error, se, 2e-13);
        assert_eq!(result.independent_sampling_units, means.len() as u64);
    }
}

#[test]
fn lsv_keeps_future_cash_reserve_without_extending_the_target_horizon() {
    let model = families().remove(0);
    let req = parse(payload(true, false, 0.0, 0.0));
    let p = RoughFamilyLsvPricingPlan::compile(
        &req,
        model.clone(),
        particles(true),
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert_eq!(*p.calibration().surface().times().last().unwrap(), 1.0);
    let with_future = p.evaluate_local_variance_risk().unwrap();
    let mut without = payload(true, false, 0.0, 0.0);
    without["market"]["discrete_dividends"]
        .as_array_mut()
        .unwrap()
        .pop();
    let other = RoughFamilyLsvPricingPlan::compile(
        &parse(without),
        model,
        particles(true),
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap()
    .evaluate()
    .unwrap();
    assert!(with_future.price.value.is_finite());
    assert_ne!(with_future.price.value.to_bits(), other.value.to_bits());
}

#[test]
#[ignore = "release numerical Black-limit price/Delta/parallel-variance acceptance"]
fn six_family_black_limit_price_delta_and_lsv_node_risk() {
    let h = RoughHeston::new(0.2, 0.04, 0.7, 0.04, 0.0, -0.6).unwrap();
    let curve = ForwardVarianceCurve::constant(0.04).unwrap();
    let models: Vec<RoughVolatilityModel> = vec![
        h.clone().into(),
        LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.0, 0.0, 0.05, 0.04)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.2, -0.6, vec![0.3, 0.7], vec![0.0, 0.0], curve.clone())
            .unwrap()
            .into(),
        RoughSabr::new(0.2, 0.0, -0.6, 1.0, curve).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.0, 0.2_f64.ln(), None).unwrap().into(),
    ];
    // Analytic Black constants independently computed from the normal CDF/density.
    let (black_price, black_delta, black_variance_risk) =
        (7.965567455405804, 0.539827837277029, 99.23813686925295);
    for (family, model) in models.into_iter().enumerate() {
        for seed in [91_u64, 1973] {
            let mut v = payload(false, true, 0.0, 0.0);
            v["market"]["discrete_dividends"] = json!([]);
            for name in ["discount_curve", "dividend_curve"] {
                v["market"][name]["discount_factors"] = json!([1.0, 1.0]);
            }
            v["engine"]["master_scramble_seed"] = json!(seed);
            v["engine"]["points_per_scramble"] = json!(2048);
            v["engine"]["scramble_count"] = json!(8);
            let policy = ExecutionPolicy::new(2, Some(128)).unwrap();
            let pure = RoughVolatilityPricingPlan::compile(
                &parse(v.clone()),
                model.clone(),
                0.125,
                policy,
            )
            .unwrap()
            .evaluate_delta()
            .unwrap();
            assert!(
                (pure.price.value - black_price).abs() <= 5.0 * pure.price.standard_error + 2e-4
            );
            assert!((pure.delta - black_delta).abs() <= 5.0 * pure.delta_standard_error + 2e-4);
            let times = (0..9).map(|i| i as f64 / 8.0).collect::<Vec<_>>();
            v["model"] = json!({"type":"local_volatility","local_variance_grid":{
                "time_nodes":times, "log_forward_moneyness_nodes":[-1.0,-0.5,0.0,0.5,1.0],
                "shape":[9,5],"values":vec![0.04;45],"floor":1e-8,"cap":4.0}});
            let p = RoughFamilyLsvPricingPlan::compile(
                &parse(v),
                model.clone(),
                particles(true),
                policy,
            )
            .unwrap();
            let lsv = p.evaluate_local_variance_risk().unwrap();
            let sensitivity = lsv.node_adjoints.iter().sum::<f64>();
            let conservative_se = lsv.standard_errors.as_ref().unwrap().iter().sum::<f64>();
            assert!((lsv.price.value - black_price).abs() <= 5.0 * lsv.price.standard_error + 2e-4);
            assert!((sensitivity - black_variance_risk).abs() <= 5.0 * conservative_se + 0.02);
            println!(
                "AAD_LSV_BLACK family={family} seed={seed} price={:.12e} price_se={:.12e} delta={:.12e} delta_se={:.12e} lsv_price={:.12e} lsv_se={:.12e} parallel_variance_risk={sensitivity:.12e} node_se_sum={conservative_se:.12e}",
                pure.price.value,
                pure.price.standard_error,
                pure.delta,
                pure.delta_standard_error,
                lsv.price.value,
                lsv.price.standard_error
            );
        }
    }
}
