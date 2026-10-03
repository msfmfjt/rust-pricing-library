use super::tests::{compile, payload};
use super::*;
use crate::risk::SpotBump;
use serde_json::json;

fn rqmc() -> serde_json::Value {
    json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,
        "scramble_count":4,"master_scramble_seed":193,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
}

#[test]
fn continuous_spot_bumps_match_recalibrated_prices_for_all_contract_styles() {
    for direction in ["up", "down"] {
        for side in ["call", "put"] {
            for style in ["knock_in", "knock_out"] {
                let mut v = payload();
                v["engine"] = rqmc();
                v["product"]["direction"] = json!({"type":direction});
                v["product"]["side"] = json!({"type":side});
                v["product"]["style"] = json!({"type":style});
                v["product"]["strike"] = json!(100.0);
                v["product"]["barrier"] = json!(if direction == "up" { 120.0 } else { 80.0 });
                let plan = compile(&v, 0.6).unwrap();
                let r = plan
                    .evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
                    .unwrap();
                assert_eq!(r.price, plan.evaluate().unwrap());
                assert_eq!(r.spot_bumps, [0.5, 1.0, 2.0]);
                assert_eq!(r.payoff_evaluations, 7 * r.price.evaluated_paths);
                for (j, &h) in r.spot_bumps.iter().enumerate() {
                    v["market"]["spot"] = json!(100.0 - h);
                    let down = compile(&v, 0.6).unwrap();
                    v["market"]["spot"] = json!(100.0 + h);
                    let up = compile(&v, 0.6).unwrap();
                    assert!(
                        (r.delta_estimates[j]
                            - (up.evaluate().unwrap().value - down.evaluate().unwrap().value)
                                / (2.0 * h))
                            .abs()
                            < 2e-11,
                        "{direction} {side} {style} h={h}"
                    );
                    for bumped in [&up, &down] {
                        for (a, b) in plan
                            .lsv_squared_leverage()
                            .iter()
                            .zip(bumped.lsv_squared_leverage())
                        {
                            assert!((a - b).abs() < 2e-14);
                        }
                    }
                }
                let relative = plan
                    .evaluate_spot_bump_risk(SpotBump::relative(0.01).unwrap())
                    .unwrap();
                assert_eq!(r.delta_estimates, relative.delta_estimates);
                assert_ne!(r.risk_fingerprint, relative.risk_fingerprint);
                assert_eq!(r.delta(), r.delta_estimates[1]);
                assert_eq!(r.standard_error(), r.delta_standard_errors[1]);
            }
        }
    }
}

#[test]
fn continuous_spot_bump_errors_use_paired_units_and_scramble_means() {
    for qmc in [false, true] {
        for antithetic in [false, true] {
            let mut v = payload();
            if qmc {
                v["engine"] = rqmc();
            }
            v["engine"]["variance_reduction"] =
                json!({"antithetic":antithetic,"brownian_bridge":true});
            let plan = compile(&v, 0.6).unwrap();
            let r = plan
                .evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
                .unwrap();
            let mut parallel = plan.clone();
            parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
            assert_eq!(
                r,
                parallel
                    .evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
                    .unwrap()
            );
            let mut scenarios = Vec::new();
            for h in r.spot_bumps {
                for spot in [100.0 - h, 100.0 + h] {
                    v["market"]["spot"] = json!(spot);
                    scenarios.push(compile(&v, 0.6).unwrap());
                }
            }
            // Independently re-evolve all six bumped/recalibrated plans and
            // aggregate their *paired* differences through the scalar sampler.
            let sample = |z: Vec<f64>, vr: VarianceReduction| {
                let bridge = plan.inner.bridge(vr).unwrap();
                let values = scenarios
                    .iter()
                    .map(|s| {
                        s.inner
                            .sample(z.clone(), bridge.as_ref(), antithetic, &|z| {
                                s.path_payoff(z)
                            })
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let mut row = [0.0; 5];
                for j in 0..3 {
                    row[j] = (values[2 * j + 1] - values[2 * j]) / (2.0 * r.spot_bumps[j]);
                }
                row[3] = row[0] - row[1];
                row[4] = row[1] - row[2];
                row
            };
            let dimension = plan.inner.path.random_dimension();
            let rows = match plan.inner.engine {
                EngineConfig::PseudoMonteCarlo(c) => {
                    let rng = Philox4x32::from_seed(c.master_seed());
                    (0..c.independent_sampling_units().get())
                        .map(|p| {
                            let z = (0..dimension)
                                .map(|d| {
                                    rng.standard_normal(RandomCoordinate::new(
                                        p,
                                        d,
                                        RandomDomain::Valuation,
                                    ))
                                })
                                .collect();
                            sample(z, c.variance_reduction())
                        })
                        .collect::<Vec<_>>()
                }
                EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                    let q = RqmcPlan::compile(c, dimension).unwrap();
                    (0..c.scramble_count().get())
                        .map(|s| {
                            let mut mean = [0.0; 5];
                            for p in 0..c.points_per_scramble().get() {
                                let z = (0..dimension)
                                    .map(|d| {
                                        inverse_standard_normal(q.uniform(s, p, d).unwrap())
                                            .unwrap()
                                    })
                                    .collect();
                                let row = sample(z, c.variance_reduction());
                                for j in 0..5 {
                                    mean[j] += row[j] / c.points_per_scramble().get() as f64;
                                }
                            }
                            mean
                        })
                        .collect::<Vec<_>>()
                }
            };
            assert_eq!(r.price.independent_sampling_units, rows.len() as u64);
            let estimates = r
                .delta_estimates
                .into_iter()
                .chain(r.bump_differences)
                .collect::<Vec<_>>();
            let errors = r
                .delta_standard_errors
                .into_iter()
                .chain(r.bump_difference_standard_errors)
                .collect::<Vec<_>>();
            for j in 0..5 {
                let mean = rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64;
                let se = (rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>()
                    / (rows.len() * (rows.len() - 1)) as f64)
                    .sqrt();
                assert!((estimates[j] - mean).abs() < 2e-11);
                assert!((errors[j] - se).abs() < 2e-11);
            }
        }
    }
}

#[test]
fn continuous_spot_bumps_preserve_history_and_recheck_current_endpoint() {
    let mut v = payload();
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["monitoring_dates"] = json!(["2026-09-03", "2026-09-04"]);
    v["product"]["historical_hit"] = json!(true);
    v["product"]["barrier"] = json!(100.0);
    v["product"]["notional"] = json!(1e18);
    let plan = compile(&v, 0.6).unwrap();
    let r = plan
        .evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
        .unwrap();
    assert_eq!(r.delta_estimates, [0.0; 3]);
    assert_eq!(r.delta_standard_errors, [0.0; 3]);
    assert_eq!(r.bump_difference_standard_errors, [0.0; 2]);
    assert_eq!(r.price.value, 7.0 * plan.inner.base.discount());
    // Current equality is allowed for the explicitly finite-bump estimator.
    // The lower scenario survives and the upper scenario is already hit.
    v["product"]["historical_hit"] = json!(false);
    v["product"]["notional"] = json!(2.0);
    let live = compile(&v, 0.6).unwrap();
    let r = live
        .evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
        .unwrap();
    for (j, h) in r.spot_bumps.into_iter().enumerate() {
        v["market"]["spot"] = json!(100.0 - h);
        let down = compile(&v, 0.6).unwrap().evaluate().unwrap();
        assert!((r.delta_estimates[j] - (r.price.value - down.value) / (2.0 * h)).abs() < 1e-11);
    }
    // Ended-unhit history selects vanilla; current barrier equality has no say.
    v["market"]["spot"] = json!(100.0);
    v["product"]["monitoring_dates"] = json!(["2026-09-03"]);
    let ended = compile(&v, 0.6).unwrap();
    let r = ended
        .evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
        .unwrap();
    assert!(r.delta() > 0.0);
    assert_ne!(
        r.risk_fingerprint,
        live.evaluate_spot_bump_risk(SpotBump::absolute(1.0).unwrap())
            .unwrap()
            .risk_fingerprint
    );
    // Validate the entire ladder before sampling, including resolved contracts.
    for h in [f64::MIN_POSITIVE, f64::MAX, 50.0, 45.0] {
        assert!(
            plan.evaluate_spot_bump_risk(SpotBump::absolute(h).unwrap())
                .is_err()
        );
    }
}
