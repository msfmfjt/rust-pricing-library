use pricing::lsv::{BergomiLsvPricingPlan, RoughBergomiLsvPricingPlan};
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::{
    LsvLeverageSurface, LsvParticleConfig, RoughBergomiLsvPlan, calibrate_bergomi_lsv,
    calibrate_rough_bergomi_lsv, calibrate_rough_bergomi_lsv_parallel,
};
use pricing::mc::{DeterministicExecutor, ExecutionPolicy, LocalVolTimeGrid, RandomDomain};
use pricing::models::{Bergomi1Factor, RoughBergomi};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

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
fn rough() -> RoughBergomi {
    RoughBergomi::new(0.2, 1.4, -0.7).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
        "actual={a:.12e}, expected={b:.12e}"
    );
}
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

// At H=1/2 the rough driver is the Brownian motion W_v, and eta=2*nu makes
// exp(eta*X - eta^2 V/2) the uncentred Bergomi multiplier exp(2*nu*W) times
// exp(-2*nu^2*t). With zero mean reversion the particles are then the same up
// to roundoff, and the leverage absorbs exactly that deterministic factor.
#[test]
fn half_hurst_rough_lsv_matches_zero_mean_reversion_bergomi() {
    let target = target();
    let (nu, rho) = (0.4, -0.6);
    let bergomi_factor = Bergomi1Factor::new(0.0, nu, rho).unwrap();
    let rough_model = RoughBergomi::new(0.5, 2.0 * nu, rho).unwrap();
    let bergomi = calibrate_bergomi_lsv(&target, bergomi_factor, 100.0, config(true)).unwrap();
    let rough = calibrate_rough_bergomi_lsv(&target, rough_model, 100.0, config(true)).unwrap();
    let times = target.time_nodes();
    let m = target.log_moneyness_nodes().len();
    let centring = |r: usize| (2.0 * nu * nu * times[r]).exp();
    for r in 0..times.len() {
        for j in 0..m {
            let k = r * m + j;
            close(
                rough.surface().squared_leverage()[k],
                bergomi.surface().squared_leverage()[k] * centring(r),
                1e-10,
            );
            let (a, b) = (
                rough.conditional_moments()[k],
                bergomi.conditional_moments()[k],
            );
            close(a.second * centring(r), b.second, 1e-10);
            close(a.effective_samples, b.effective_samples, 1e-10);
            assert_eq!(a.extrapolated, b.extrapolated);
        }
    }
    // <s, L_rough> = <s * c, L_bergomi> for every target, so the VJPs agree.
    let seed = (0..20).map(|i| (0.7 * i as f64).cos()).collect::<Vec<_>>();
    let scaled = seed
        .iter()
        .enumerate()
        .map(|(k, s)| s * centring(k / m))
        .collect::<Vec<_>>();
    for (a, b) in rough
        .reverse_leverage(&seed)
        .unwrap()
        .iter()
        .zip(bergomi.reverse_leverage(&scaled).unwrap())
    {
        close(*a, b, 1e-9);
    }
    // Pricing paths share the spot and orthogonal blocks. The grid is the
    // calibration grid: between leverage knots the rough centring moves with
    // the node time while the absorbed factor is constant per leverage row.
    let grid = LocalVolTimeGrid::compile(times.to_vec(), 1.0).unwrap();
    let bergomi_plan = bergomi.pricing_plan(&grid).unwrap();
    let rough_plan = rough.pricing_plan(&grid).unwrap();
    for path in 0..50 {
        let shocks = bergomi_plan
            .pseudo_shocks(33, path, RandomDomain::Valuation)
            .unwrap();
        let mut rough_shocks = shocks.clone();
        rough_shocks.extend(std::iter::repeat_n(0.37, shocks.len() / 2));
        let a = rough_plan.evolve_path(100.0, &rough_shocks).unwrap();
        let b = bergomi_plan.evolve_path(100.0, &shocks).unwrap();
        for (x, y) in a.states().iter().zip(b.states()) {
            close(*x, *y, 1e-12);
        }
    }
}

// The pathwise reverse covers leverage, initial f and all three shock blocks.
#[test]
fn rough_path_reverse_matches_finite_differences() {
    let target = target();
    let surface = LsvLeverageSurface::new(
        target.time_nodes().to_vec(),
        target.log_moneyness_nodes().to_vec(),
        target.values().iter().map(|v| 25.0 * v).collect(),
        100.0,
    )
    .unwrap();
    // Inserted nodes give a nonuniform Volterra grid with older cells.
    let grid = LocalVolTimeGrid::compile(vec![0.0, 0.05, 0.1, 0.17, 0.2, 0.3], 1.0).unwrap();
    let plan = RoughBergomiLsvPlan::new(rough(), surface.clone(), &grid).unwrap();
    let n = grid.nodes().len() - 1;
    let z = (0..3 * n)
        .map(|i| 0.9 * (1.3 * i as f64 + 0.4).sin())
        .collect::<Vec<_>>();
    let seeds = (0..=n).map(|i| 0.2 + 0.1 * i as f64).collect::<Vec<_>>();
    let objective = |p: &RoughBergomiLsvPlan, f, z: &[f64]| {
        p.evolve_path(f, z)
            .unwrap()
            .states()
            .iter()
            .zip(&seeds)
            .map(|(s, w)| s * w)
            .sum::<f64>()
    };
    let adj = plan
        .evolve_path(101.0, &z)
        .unwrap()
        .reverse(&seeds)
        .unwrap();
    let h = 1e-6;
    close(
        adj.initial_f,
        (objective(&plan, 101.0 + h, &z) - objective(&plan, 101.0 - h, &z)) / (2.0 * h),
        1e-7,
    );
    for j in 0..z.len() {
        let mut up = z.clone();
        let mut down = z.clone();
        up[j] += h;
        down[j] -= h;
        let expected = (objective(&plan, 101.0, &up) - objective(&plan, 101.0, &down)) / (2.0 * h);
        let actual = if j < n {
            adj.spot_shocks[j]
        } else {
            adj.orthogonal_shocks[j - n]
        };
        close(actual, expected, 1e-7);
    }
    for k in 0..surface.squared_leverage().len() {
        let eval = |bump: f64| {
            let mut v = surface.squared_leverage().to_vec();
            v[k] += bump;
            let s = LsvLeverageSurface::new(
                surface.times().to_vec(),
                surface.log_nodes().to_vec(),
                v,
                100.0,
            )
            .unwrap();
            objective(
                &RoughBergomiLsvPlan::new(rough(), s, &grid).unwrap(),
                101.0,
                &z,
            )
        };
        close(
            adj.squared_leverage[k],
            (eval(1e-6) - eval(-1e-6)) / 2e-6,
            1e-6,
        );
    }
}

#[test]
fn rough_calibration_vjp_matches_recalibration() {
    let target = target();
    let model = calibrate_rough_bergomi_lsv(&target, rough(), 100.0, config(true)).unwrap();
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
        calibrate_rough_bergomi_lsv(&grid, rough(), 100.0, config(false))
            .unwrap()
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
        "the test must exercise calibration feedback"
    );
}

#[test]
fn rough_price_only_paths_and_parallel_calibration_are_bit_identical() {
    let target = target();
    let sequential = calibrate_rough_bergomi_lsv(
        &target,
        rough(),
        100.0,
        LsvParticleConfig::new(9_000, 5, 0.3, 8.0, true).unwrap(),
    )
    .unwrap();
    let seed = (0..20).map(|i| (i as f64).sin()).collect::<Vec<_>>();
    for workers in [1, 3, 8] {
        let executor =
            DeterministicExecutor::new(ExecutionPolicy::new(workers, None).unwrap()).unwrap();
        let parallel = calibrate_rough_bergomi_lsv_parallel(
            &target,
            rough(),
            100.0,
            LsvParticleConfig::new(9_000, 5, 0.3, 8.0, true).unwrap(),
            &executor,
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", parallel.surface()),
            format!("{:?}", sequential.surface())
        );
        assert_eq!(
            format!("{:?}", parallel.conditional_moments()),
            format!("{:?}", sequential.conditional_moments())
        );
        assert_eq!(
            format!("{:?}", parallel.reverse_leverage(&seed).unwrap()),
            format!("{:?}", sequential.reverse_leverage(&seed).unwrap())
        );
    }
    let grid = LocalVolTimeGrid::compile(vec![0.0, 0.05, 0.1, 0.17, 0.2, 0.3], 0.04).unwrap();
    let plan = sequential.pricing_plan(&grid).unwrap();
    let mut states = Vec::new();
    for path in 0..200 {
        let shocks = plan
            .pseudo_shocks(91, path, RandomDomain::Valuation)
            .unwrap();
        let recorded = plan.evolve_path(100.0, &shocks).unwrap();
        plan.evolve_states(100.0, &shocks, &mut states).unwrap();
        assert_eq!(bits(&states), bits(recorded.states()), "path {path}");
    }
    assert!(plan.evolve_states(100.0, &[0.0; 4], &mut states).is_err());
}

fn payload(qmc: bool, bump: f64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":[0.0,0.3,0.7,1.0],"log_forward_moneyness_nodes":[-0.8,-0.2,0.12,0.45,0.9],"shape":[4,5],
        "values":(0..20).map(|i|0.04+0.002*(i%5) as f64+bump*(0.7*i as f64).cos()).collect::<Vec<_>>(),"floor":1e-8,"cap":4.0}});
    v["market"]["discrete_dividends"] =
        json!([{"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":1.5}}]);
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,"scramble_count":4,"master_scramble_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    v
}
fn request(qmc: bool, bump: f64) -> PricingRequest {
    parse_request_json(
        &serde_json::to_vec(&payload(qmc, bump)).unwrap(),
        JsonLimits::DEFAULT,
    )
    .unwrap()
}

#[test]
fn rough_facade_replays_across_workers_and_reverses_calibration() {
    for qmc in [false, true] {
        let build = |bump, workers, trace| {
            RoughBergomiLsvPricingPlan::compile(
                &request(qmc, bump),
                rough(),
                LsvParticleConfig::new(256, 429, 0.35, 5.0, trace).unwrap(),
                ExecutionPolicy::new(workers, Some(64)).unwrap(),
            )
            .unwrap()
        };
        let p = build(0.0, 1, true);
        assert!(p.calibration().surface().times().contains(&0.25));
        let r = p.evaluate_local_variance_risk().unwrap();
        let other = build(0.0, 3, true).evaluate_local_variance_risk().unwrap();
        assert_eq!(
            r.price.value.to_bits(),
            p.evaluate().unwrap().value.to_bits()
        );
        assert_eq!(r.price.value.to_bits(), other.price.value.to_bits());
        assert_eq!(r.node_adjoints, other.node_adjoints);
        assert_eq!(r.standard_errors, other.standard_errors);
        assert_eq!(r.standard_errors.is_some(), qmc);
        assert_eq!(r.price.scheme, pricing::mc::lsv::ROUGH_BERGOMI_LSV_SCHEME);
        assert!(build(0.0, 1, false).evaluate_local_variance_risk().is_err());
        let expected = (build(1e-7, 1, false).evaluate().unwrap().value
            - build(-1e-7, 1, false).evaluate().unwrap().value)
            / 2e-7;
        let actual = r
            .node_adjoints
            .iter()
            .enumerate()
            .map(|(i, a)| a * (0.7 * i as f64).cos())
            .sum::<f64>();
        assert!(
            (actual - expected).abs() < 2e-5 * (1.0 + expected.abs()),
            "{actual} vs {expected}"
        );
    }
}

// End to end through the request path: with pseudo-MC the rough plan reads its
// spot and orthogonal blocks from the same coordinates as one-factor Bergomi.
#[test]
fn half_hurst_rough_facade_prices_like_zero_mean_reversion_bergomi() {
    let (nu, rho) = (0.3, -0.5);
    let particles = LsvParticleConfig::new(256, 429, 0.35, 5.0, false).unwrap();
    let policy = ExecutionPolicy::new(2, Some(64)).unwrap();
    let rough = RoughBergomiLsvPricingPlan::compile(
        &request(false, 0.0),
        RoughBergomi::new(0.5, 2.0 * nu, rho).unwrap(),
        particles.clone(),
        policy,
    )
    .unwrap()
    .evaluate()
    .unwrap();
    let bergomi = BergomiLsvPricingPlan::compile(
        &request(false, 0.0),
        Bergomi1Factor::new(0.0, nu, rho).unwrap(),
        particles,
        policy,
    )
    .unwrap()
    .evaluate()
    .unwrap();
    close(rough.value, bergomi.value, 1e-10);
    close(rough.standard_error, bergomi.standard_error, 1e-8);
}
