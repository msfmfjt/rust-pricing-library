//! Independent finite-grid prices and independent sampling-error aggregation.
//! No continuous-time error or convergence-rate claim is made by this panel.

use pricing::mc::{
    BrownianBridgePlan, EngineConfig, ExecutionPolicy, RandomDomain, RqmcPlan,
    inverse_standard_normal,
};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/two-step-prices.json"
    ))
    .unwrap()
}
fn scalar(p: &Value, name: &str) -> f64 {
    p[name].as_f64().unwrap()
}
fn vector(p: &Value, name: &str) -> Vec<f64> {
    p[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn model(case: &Value) -> RoughVolatilityModel {
    let p = &case["parameters"];
    let f = |name| scalar(p, name);
    let forward_variance = || {
        ForwardVarianceCurve::exponential(
            f("initial_forward_variance"),
            f("forward_variance_growth"),
        )
        .unwrap()
    };
    match case["name"].as_str().unwrap() {
        "rough_heston" => RoughHeston::new(
            f("hurst"),
            f("initial_variance"),
            f("mean_reversion"),
            f("long_run_variance"),
            f("vol_of_vol"),
            f("correlation"),
        )
        .unwrap()
        .into(),
        "lifted_heston" => LiftedHeston::new(
            f("initial_variance"),
            f("mean_reversion"),
            f("long_run_variance"),
            f("vol_of_vol"),
            f("correlation"),
            vector(p, "weights"),
            vector(p, "rates"),
        )
        .unwrap()
        .into(),
        "quadratic_rough_heston" => QuadraticRoughHeston::new(
            f("hurst"),
            f("initial_state"),
            f("mean_reversion"),
            f("vol_of_vol"),
            f("quadratic"),
            f("shift"),
            f("variance_floor"),
        )
        .unwrap()
        .into(),
        "mixed_rough_bergomi" => MixedRoughBergomi::new(
            f("hurst"),
            f("correlation"),
            vector(p, "weights"),
            vector(p, "vol_of_vols"),
            forward_variance(),
        )
        .unwrap()
        .into(),
        "rough_sabr" => RoughSabr::new(
            f("hurst"),
            f("vol_of_vol"),
            f("correlation"),
            f("beta"),
            forward_variance(),
        )
        .unwrap()
        .into(),
        "rfsv" => Rfsv::new(
            f("hurst"),
            f("mean_reversion"),
            f("vol_of_log_vol"),
            f("mean_log_vol"),
            None,
        )
        .unwrap()
        .into(),
        other => panic!("unknown reference family {other}"),
    }
}
fn request(
    strike: f64,
    seed: u64,
    qmc: bool,
    antithetic: bool,
    bridge: bool,
    accurate: bool,
) -> PricingRequest {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v2/pricing_request.golden.json"
    ))
    .unwrap();
    value["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
    value["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
    value["product"]["strike"] = json!(strike);
    value["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo", "points_per_scramble":if accurate {4096} else {64},
            "scramble_count":if accurate {16} else {4}, "master_scramble_seed":seed,
            "variance_reduction":{"antithetic":antithetic,"brownian_bridge":bridge}})
    } else {
        json!({"type":"pseudo_monte_carlo", "independent_sampling_units":64, "master_seed":seed,
            "variance_reduction":{"antithetic":antithetic,"brownian_bridge":bridge}})
    };
    parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn compile(request: &PricingRequest, model: RoughVolatilityModel) -> RoughVolatilityPricingPlan {
    let plan = RoughVolatilityPricingPlan::compile(
        request,
        model,
        0.5,
        ExecutionPolicy::new(2, Some(256)).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.time_nodes(), &[0.0, 0.5, 1.0]);
    assert_eq!(plan.risky_spot(), 100.0);
    plan
}

#[test]
#[ignore = "nondegenerate numerical panel; run in focused release CI"]
fn nondegenerate_two_step_prices_match_conditional_gaussian_reference() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let model = model(case);
        for seed in [91, 1973] {
            for (strike, expected) in reference["strikes"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["call_prices"].as_array().unwrap())
            {
                let strike = strike.as_f64().unwrap();
                let expected = expected.as_f64().unwrap();
                let request = request(strike, seed, true, true, true, true);
                let result = compile(&request, model.clone()).evaluate().unwrap();
                let gap = result.value - expected;
                println!(
                    "{} seed={seed} K={strike} price={:.12} reference={expected:.12} gap={gap:.9} se={:.9} bound={:.9}",
                    model.name(),
                    result.value,
                    result.standard_error,
                    gap.abs() + 4.0 * result.standard_error
                );
                assert_eq!(result.independent_sampling_units, 16);
                assert_eq!(result.evaluated_paths, 131_072);
                assert!(result.standard_error > 0.0 && result.standard_error <= 0.005);
                assert!(
                    gap.abs() <= 5.0 * result.standard_error + 2e-7,
                    "{}: biased beyond sampling uncertainty",
                    model.name()
                );
                assert!(
                    gap.abs() + 4.0 * result.standard_error <= 0.03,
                    "{}: failed absolute finite-grid precision budget",
                    model.name()
                );
            }
        }
    }
}

fn two_pass(values: &[f64]) -> (f64, f64) {
    assert!(values.len() >= 2);
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let squared = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    (mean, (squared / (n * (n - 1.0))).sqrt())
}
fn payoff_sample(
    plan: &RoughVolatilityPathPlan,
    mut normals: Vec<f64>,
    antithetic: bool,
    bridge: bool,
) -> f64 {
    let n = plan.time_nodes().len() - 1;
    if bridge {
        let bridge = BrownianBridgePlan::compile(plan.time_nodes().to_vec(), 1).unwrap();
        let blocks = match plan.model() {
            RoughVolatilityModel::Rfsv(_) | RoughVolatilityModel::QuadraticRoughHeston(_) => 1,
            _ => 2,
        };
        // Only genuine Brownian blocks are bridged, never the Gaussian residuals.
        for block in normals[..blocks * n].chunks_exact_mut(n) {
            let transformed = bridge.apply_one_factor(block).unwrap();
            block.copy_from_slice(&transformed);
        }
    }
    let sample = |z: &[f64]| {
        let terminal = *plan.evolve_path(100.0, z).unwrap().forwards.last().unwrap();
        (terminal - 100.0).max(0.0)
    };
    let positive = sample(&normals);
    if antithetic {
        let negative: Vec<_> = normals.iter().map(|x| -x).collect();
        0.5 * (positive + sample(&negative))
    } else {
        positive
    }
}

#[test]
fn independent_primal_payoffs_reconstruct_mc_and_rqmc_sampling_errors() {
    for case in reference()["cases"].as_array().unwrap() {
        for qmc in [false, true] {
            for antithetic in [false, true] {
                for bridge in [false, true] {
                    let request = request(100.0, 91, qmc, antithetic, bridge, false);
                    let plan = compile(&request, model(case));
                    let path = plan.path_plan();
                    let dimension = path.random_dimension();
                    let units: Vec<f64> = match request.engine() {
                        EngineConfig::PseudoMonteCarlo(config) => {
                            (0..config.independent_sampling_units().get())
                                .map(|p| {
                                    payoff_sample(
                                        path,
                                        path.pseudo_shocks(
                                            config.master_seed(),
                                            p,
                                            RandomDomain::Valuation,
                                        ),
                                        antithetic,
                                        bridge,
                                    )
                                })
                                .collect()
                        }
                        EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                            let sampler = RqmcPlan::compile(config, dimension).unwrap();
                            (0..config.scramble_count().get())
                                .map(|scramble| {
                                    let points: Vec<f64> = (0..config.points_per_scramble().get())
                                        .map(|point| {
                                            let z: Vec<f64> = (0..dimension)
                                                .map(|d| {
                                                    inverse_standard_normal(
                                                        sampler
                                                            .uniform(scramble, point, d)
                                                            .unwrap(),
                                                    )
                                                    .unwrap()
                                                })
                                                .collect();
                                            payoff_sample(path, z, antithetic, bridge)
                                        })
                                        .collect();
                                    points.iter().sum::<f64>() / points.len() as f64
                                })
                                .collect()
                        }
                    };
                    let (mean, se) = two_pass(&units);
                    let actual = plan.evaluate().unwrap();
                    assert!(se > 0.0);
                    assert_eq!(actual.independent_sampling_units, units.len() as u64);
                    assert!((actual.value - mean).abs() <= 2e-12);
                    assert!(
                        (actual.standard_error - se).abs() <= 2e-12,
                        "{} qmc={qmc} antithetic={antithetic} bridge={bridge}: actual={} independent={se}",
                        path.model().name(),
                        actual.standard_error
                    );
                }
            }
        }
    }
}
