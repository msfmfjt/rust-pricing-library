use super::*;
use crate::engine::risk::stochastic_dividends::lsv_path_refinement_tests::{
    Contract, request_value,
};
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

fn payload() -> Value {
    let mut v = request_value(Contract::Barrier);
    v["risk"]
        .as_object_mut()
        .unwrap()
        .remove("payoff_smoothing");
    v
}

fn compile(v: &Value, h: f64, correlation: [f64; 3]) -> StochasticDividendPricingPlan {
    let request = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
        &request,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, correlation[0]).unwrap(),
        RoughBergomi::new(h, 0.6, correlation[1]).unwrap(),
        correlation[2],
        LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
        (364.0 / 365.0) / 8.0,
        ExecutionPolicy::new(1, Some(32)).unwrap(),
    )
    .unwrap()
}

fn normals(steps: usize, pattern: usize) -> Vec<f64> {
    (0..4 * steps)
        .map(|k| {
            let (i, factor) = (k / 4 + 1, k % 4);
            ((37 * i * (factor + 3) + 17 * i * i + 13 * pattern) % 101) as f64 / 25.0 - 2.0
        })
        .collect()
}

#[test]
fn conditioned_tangents_match_reanchored_spot_bumps() {
    for h in [0.1, 0.3, 0.5] {
        for terminal_monitor in [false, true] {
            let mut v = payload();
            if !terminal_monitor {
                v["product"]["monitoring_dates"] = json!(["2027-03-05"]);
            }
            let base = compile(&v, h, [-0.25, -0.4, 0.15]);
            let spot = v["market"]["spot"].as_f64().unwrap();
            for bump in [0.001, 0.0005] {
                v["market"]["spot"] = json!(spot - bump);
                let down = compile(&v, h, [-0.25, -0.4, 0.15]);
                v["market"]["spot"] = json!(spot + bump);
                let up = compile(&v, h, [-0.25, -0.4, 0.15]);
                v["market"]["spot"] = json!(spot);
                for pattern in 0..8 {
                    let z = normals(base.path.times().len() - 1, pattern);
                    let eval = |p: &StochasticDividendPricingPlan| {
                        HardBarrierPlan::compile(p).unwrap().sample(p, &z).unwrap()
                    };
                    let actual = eval(&base)[1];
                    let expected = (eval(&up)[0] - eval(&down)[0]) / (2.0 * bump);
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "H={h}, terminal={terminal_monitor}, pattern={pattern}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn pseudo_mc_errors_use_antithetic_units_and_replay_across_workers() {
    for antithetic in [false, true] {
        let mut v = payload();
        v["engine"]["variance_reduction"]["antithetic"] = json!(antithetic);
        let base = compile(&v, 0.1, [-0.25, -0.4, 0.15]);
        let EngineConfig::PseudoMonteCarlo(config) = base.engine else {
            panic!()
        };
        let context = HardBarrierPlan::compile(&base).unwrap();
        let rng = Philox4x32::from_seed(config.master_seed());
        let count = config.independent_sampling_units().get();
        let mut values = [Vec::new(), Vec::new()];
        for p in 0..count {
            let z = (0..base.path.random_dimension())
                .map(|d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation)))
                .collect::<Vec<_>>();
            let mut value = context.sample(&base, &z).unwrap();
            if antithetic {
                let opposite = z.iter().map(|z| -z).collect::<Vec<_>>();
                let other = context.sample(&base, &opposite).unwrap();
                for j in 0..2 {
                    value[j] = (value[j] + other[j]) * 0.5;
                }
            }
            for j in 0..2 {
                values[j].push(value[j]);
            }
        }
        let actual = base.evaluate_lsv_hard_barrier_spot_risk().unwrap();
        for (j, (mean, se)) in [
            (actual.price.value, actual.price.standard_error),
            (actual.delta, actual.delta_standard_error),
        ]
        .into_iter()
        .enumerate()
        {
            let stats = DeterministicStatistics::from_ordered_values_two_pass(&values[j]);
            assert!((mean - stats.sum().total() / count as f64).abs() < 2e-14);
            let variance = values[j].iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (count * (count - 1)) as f64;
            assert!((se - variance.sqrt()).abs() < 2e-14);
        }
        assert_eq!(actual.price.independent_sampling_units, count);
        assert_eq!(
            actual.price.evaluated_paths,
            u128::from(count) * if antithetic { 2 } else { 1 }
        );
        let mut parallel = base.clone();
        parallel.policy = ExecutionPolicy::new(2, Some(32)).unwrap();
        assert_eq!(
            actual,
            parallel.evaluate_lsv_hard_barrier_spot_risk().unwrap()
        );
    }
}

#[test]
fn scope_rejects_unsupported_contracts_and_singular_conditioning() {
    for (field, value) in [
        ("side", json!({"type":"put"})),
        ("direction", json!({"type":"down"})),
        ("rebate", json!(1.0)),
    ] {
        let mut v = payload();
        v["product"][field] = value;
        assert!(
            compile(&v, 0.1, [-0.25, -0.4, 0.15])
                .evaluate_lsv_hard_barrier_spot_risk()
                .is_err()
        );
    }
    let mut v = payload();
    v["product"]["monitoring_dates"] = json!(["2026-09-04", "2027-09-03"]);
    assert!(
        compile(&v, 0.1, [-0.25, -0.4, 0.15])
            .evaluate_lsv_hard_barrier_spot_risk()
            .is_err()
    );
    let asian = request_value(Contract::Asian);
    assert!(
        compile(&asian, 0.1, [-0.25, -0.4, 0.15])
            .evaluate_lsv_hard_barrier_spot_risk()
            .is_err()
    );
    for correlation in [[0.0, 1.0, 0.0], [0.2, 0.2, 1.0]] {
        assert!(
            compile(&payload(), 0.1, correlation)
                .evaluate_lsv_hard_barrier_spot_risk()
                .is_err()
        );
    }
}

#[test]
fn knock_in_out_parity_and_notional_include_delayed_discount() {
    let v = payload();
    let ki = compile(&v, 0.3, [-0.25, -0.4, 0.15]);
    let mut ko_value = v.clone();
    ko_value["product"]["style"] = json!({"type":"knock_out"});
    let ko = compile(&ko_value, 0.3, [-0.25, -0.4, 0.15]);
    let mut scaled_value = ko_value;
    scaled_value["product"]["notional"] = json!(3.0);
    let scaled = compile(&scaled_value, 0.3, [-0.25, -0.4, 0.15]);
    let context = HardBarrierPlan::compile(&ki).unwrap();
    let out_context = HardBarrierPlan::compile(&ko).unwrap();
    let mut vanilla_context = HardBarrierPlan::compile(&ko).unwrap();
    vanilla_context.monitors.fill(false);
    for pattern in 0..8 {
        let z = normals(ki.path.times().len() - 1, pattern);
        let a = context.sample(&ki, &z).unwrap();
        let b = out_context.sample(&ko, &z).unwrap();
        let c = vanilla_context.sample(&ko, &z).unwrap();
        let d = HardBarrierPlan::compile(&scaled)
            .unwrap()
            .sample(&scaled, &z)
            .unwrap();
        for j in 0..2 {
            assert!((a[j] + b[j] - c[j]).abs() < 2e-13);
            assert!((3.0 * b[j] - d[j]).abs() < 2e-13);
        }
    }
}

#[test]
fn tail_transport_preserves_open_quantiles_and_rejects_numerical_failure() {
    let mut v = payload();
    v["product"]["barrier"] = json!(1e8);
    let base = compile(&v, 0.1, [-0.25, -0.4, 0.15]);
    let context = HardBarrierPlan::compile(&base).unwrap();
    let mut z = vec![0.0; base.path.random_dimension() as usize];
    // The monitored free normal has Phi(z)==1 in f64. Complementary tails
    // still yield an interior conditional quantile.
    let monitored = context.monitors.iter().position(|m| *m).unwrap();
    z[4 * (monitored - 1) + 3] = 10.0;
    assert!(context.sample(&base, &z).is_ok());
    z[4 * (monitored - 1) + 3] = -40.0;
    assert!(context.sample(&base, &z).is_err());
    z[0] = f64::NAN;
    assert!(context.sample(&base, &z).is_err());
}
