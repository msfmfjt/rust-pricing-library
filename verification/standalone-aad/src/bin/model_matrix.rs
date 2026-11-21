//! Fixed native path, calibration and end-to-end pricing workloads.
use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::mc::lsv::{
    BergomiLsvPlan, LsvLeverageSurface, LsvParticleConfig, calibrate_bergomi_lsv,
};
use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid, RandomDomain};
use pricing::models::{
    Bergomi1Factor, Bergomi2Factor, BergomiDynamics, HullWhite1Factor, HybridCorrelation,
};
use pricing::{JsonLimits, parse_request_json};
use rust_aad_bench::{FNV_OFFSET, fnv, measure};
use serde_json::{Value, json};
use std::hint::black_box;

fn factor1() -> Bergomi1Factor {
    Bergomi1Factor::new(0.7, 0.3, -0.35).unwrap()
}
fn particles() -> LsvParticleConfig {
    LsvParticleConfig::new(256, 8401, 0.7, 2., true).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= tol * (1. + b.abs()),
        "{a} versus {b}"
    );
}
fn direction(i: usize) -> f64 {
    ((i * 17 % 41) as f64 - 20.) / 32.
}

fn paths<F: BergomiDynamics>(factor: F, state: impl Fn(F::State) -> Vec<f64>) -> Value {
    let times: Vec<_> = (0..=120).map(|i| i as f64 / 120.).collect();
    let grid = LocalVolTimeGrid::compile(times, 1.).unwrap();
    let values = vec![0.03, 0.04, 0.053, 0.038, 0.045, 0.06, 0.04, 0.05, 0.065];
    let compile = |v: Vec<f64>| {
        BergomiLsvPlan::new(
            factor,
            LsvLeverageSurface::new(vec![0., 0.5, 1.], vec![-0.4, 0., 0.4], v, 100.).unwrap(),
            &grid,
        )
        .unwrap()
    };
    let plan = compile(values.clone());
    let shocks: Vec<_> = (0..32)
        .map(|i| plan.pseudo_shocks(971, i, RandomDomain::Valuation).unwrap())
        .collect();
    let hash = shocks
        .iter()
        .flatten()
        .fold(FNV_OFFSET, |h, z| fnv(h, &z.to_le_bytes()));
    let mut seeds = vec![0.; 121];
    seeds[120] = 1.;
    let eps = 1e-5;
    let bumped_plan = |h: f64| {
        compile(
            values
                .iter()
                .enumerate()
                .map(|(i, v)| v + h * direction(i))
                .collect(),
        )
    };
    let up = bumped_plan(eps);
    let down = bumped_plan(-eps);
    let mut mean = Vec::new();
    for z in &shocks {
        let path = plan.evolve_path(101., z).unwrap();
        let adj = path.reverse(&seeds).unwrap();
        let forward = |p: &BergomiLsvPlan<F>, spot: f64, z: &[f64]| {
            p.evolve_path(spot, z).unwrap().states()[120]
        };
        let directional = adj.initial_f
            + adj
                .squared_leverage
                .iter()
                .enumerate()
                .map(|(i, x)| x * direction(i))
                .sum::<f64>()
            + adj
                .spot_shocks
                .iter()
                .chain(&adj.orthogonal_shocks)
                .enumerate()
                .map(|(i, x)| x * direction(i))
                .sum::<f64>();
        let bumped = |h: f64| {
            z.iter()
                .enumerate()
                .map(|(i, v)| v + h * direction(i))
                .collect::<Vec<_>>()
        };
        let fd = (forward(&up, 101. + eps, &bumped(eps))
            - forward(&down, 101. - eps, &bumped(-eps)))
            / (2. * eps);
        close(directional, fd, 2e-6);
        let mut row = vec![path.states()[120], adj.initial_f];
        row.extend(state(adj.initial_factor));
        row.extend(&*adj.squared_leverage);
        row.extend(&*adj.spot_shocks);
        row.extend(&*adj.orthogonal_shocks);
        if mean.is_empty() {
            mean.resize(row.len(), 0.);
        }
        for (a, v) in mean.iter_mut().zip(row) {
            assert!(v.is_finite());
            *a += v / 32.;
        }
    }
    let mut states = Vec::new();
    let forward = measure(32, || {
        for z in &shocks {
            plan.evolve_states(101., black_box(z), &mut states).unwrap();
            black_box(&states);
        }
    });
    let vjp = measure(32, || {
        for z in &shocks {
            let p = plan.evolve_path(101., black_box(z)).unwrap();
            black_box(p.reverse(&seeds).unwrap());
        }
    });
    json!({"steps":120,"paths":32,"seed":971,"shocks_fnv1a64":format!("{hash:016x}"),"output_and_adjoint_means":mean,
        "phases":{"compile":measure(1,||compile(values.clone())),"forward":forward,"forward_vjp":vjp}})
}

fn calibration() -> Value {
    let values: Vec<_> = (0..9).map(|i| 0.04 + i as f64 / 1024.).collect();
    let make =
        |v| LocalVarianceGrid::new(vec![0., 0.5, 1.], vec![-0.4, 0., 0.4], v, 1e-8, 4.).unwrap();
    let target = make(values.clone());
    let build =
        |g: &LocalVarianceGrid| calibrate_bergomi_lsv(g, factor1(), 100., particles()).unwrap();
    let plan = build(&target);
    let seeds: Vec<_> = (0..plan.surface().squared_leverage().len())
        .map(direction)
        .collect();
    let reverse = || plan.reverse_leverage(&seeds).unwrap();
    let actual = reverse();
    for (i, d) in actual.iter().enumerate() {
        let objective = |h: f64| {
            let mut v = values.clone();
            v[i] += h;
            build(&make(v))
                .surface()
                .squared_leverage()
                .iter()
                .zip(&seeds)
                .map(|(x, s)| x * s)
                .sum::<f64>()
        };
        close(*d, (objective(1e-6) - objective(-1e-6)) / 2e-6, 2e-6);
    }
    json!({"particles":256,"seed":8401,"values":plan.surface().squared_leverage(),"seeds":seeds,"target_vjp":actual,
        "phases":{"calibrate":measure(1,||build(&target)),"calibration_vjp":measure(1,reverse)}})
}

fn hybrid() -> Value {
    let target = HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.5, 1.], vec![-0.5, 0., 0.5], vec![0.28; 6]).unwrap(),
        vec![0., 0.25, 0.5, 0.75, 1.],
        vec![-0.35, 0., 0.35],
        1e-8,
        4.,
    )
    .unwrap();
    let grid = target.grid();
    let mut request: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    request["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":grid.time_nodes(),
        "log_forward_moneyness_nodes":grid.log_moneyness_nodes(),"shape":[5,3],"values":grid.values(),"floor":grid.floor(),"cap":grid.cap()}});
    request["market"]["discrete_dividends"] = json!([{"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":2.0,"beta":0.02}}]);
    request["engine"] = json!({"type":"pseudo_monte_carlo","master_seed":971,"independent_sampling_units":64,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let req =
        parse_request_json(&serde_json::to_vec(&request).unwrap(), JsonLimits::DEFAULT).unwrap();
    let build = || {
        HullWhiteEquityPricingPlan::compile_lsv(
            &req,
            &target,
            factor1(),
            HullWhite1Factor::new(0.13, vec![0., 0.37], vec![0.006, 0.008]).unwrap(),
            HybridCorrelation::new(-0.35, 0.1, 0.02).unwrap(),
            particles(),
            ExecutionPolicy::new(1, Some(128)).unwrap(),
        )
        .unwrap()
    };
    let plan = build();
    let price = || plan.evaluate().unwrap();
    let vjp = || plan.evaluate_aad().unwrap();
    let risk = vjp();
    close(price().value, risk.price.value, 2e-13);
    assert_eq!(risk.parameter_labels.len(), risk.derivatives.len());
    assert!(risk.derivatives.iter().all(|x| x.is_finite()));
    json!({"independent_units":64,"paths":risk.price.evaluated_paths,"price":risk.price.value,"standard_error":risk.price.standard_error,
        "labels":risk.parameter_labels,"derivatives":risk.derivatives,"seeds":[971,8401],"particles":256,
        "phases":{"compile_calibrate":measure(1,build),"price":measure(1,price),"price_and_risk":measure(1,vjp)}})
}

fn main() {
    let id = std::env::args().nth(1).expect("model workload ID");
    let row = match id.as_str() {
        "bergomi-1f-path-v1" => paths(factor1(), |s| vec![s]),
        "bergomi-2f-path-v1" => paths(
            Bergomi2Factor::new([0.4, 2.], 0.3, 0.35, [-0.35, -0.2], 0.25).unwrap(),
            |s| s.to_vec(),
        ),
        "lsv-particle-v1" => calibration(),
        "hull-white-lsv-v1" => hybrid(),
        _ => panic!("unknown application benchmark"),
    };
    println!(
        "{}",
        json!({"schema":1,"workload":id,"backend":"pricing-native","result":row})
    );
}
