use super::tests::{compile, gbm_reference_with_volatility, payload};
use super::*;
use serde_json::{Value, json};

fn bumped(v: &Value, shift: f64) -> Value {
    let mut result = v.clone();
    for x in result["model"]["local_variance_grid"]["values"]
        .as_array_mut()
        .unwrap()
    {
        *x = json!((x.as_f64().unwrap().sqrt() + shift).powi(2));
    }
    result
}
fn rqmc() -> Value {
    json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,
        "scramble_count":4,"master_scramble_seed":193,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
}

#[test]
fn parallel_local_volatility_recalibrates_original_nodes_for_all_styles() {
    for direction in ["up", "down"] {
        for side in ["call", "put"] {
            for style in ["knock_out", "knock_in"] {
                let mut v = payload();
                v["engine"] = rqmc();
                v["product"]["direction"] = json!({"type":direction});
                v["product"]["side"] = json!({"type":side});
                v["product"]["style"] = json!({"type":style});
                v["product"]["barrier"] = json!(if direction == "up" { 120.0 } else { 80.0 });
                let plan = compile(&v, 0.6).unwrap();
                let before = plan.evaluate().unwrap();
                let r = plan.evaluate_parallel_local_volatility_risk(0.01).unwrap();
                assert_eq!(r.price, before);
                assert_eq!(r.local_volatility_bumps, [0.005, 0.01, 0.02]);
                assert_eq!(r.scenario_evaluated_paths, 7 * r.price.evaluated_paths);
                assert_eq!(r.payoff_evaluations, r.scenario_evaluated_paths);
                assert_eq!(r.recalibration_count, 6);
                assert_eq!(r.vega_per_vol_point(), 0.01 * r.vega());
                assert_eq!(r.standard_error_per_vol_point(), 0.01 * r.standard_error());
                let scenarios = plan
                    .local_volatility_scenarios(&r.local_volatility_bumps)
                    .unwrap();
                for (j, h) in r.local_volatility_bumps.into_iter().enumerate() {
                    let down = compile(&bumped(&v, -h), 0.6).unwrap();
                    let up = compile(&bumped(&v, h), 0.6).unwrap();
                    assert!(
                        (r.vega_estimates[j]
                            - (up.evaluate().unwrap().value - down.evaluate().unwrap().value)
                                / (2.0 * h))
                            .abs()
                            < 2e-10
                    );
                    for (scenario, external) in
                        [(&scenarios[2 * j], &down), (&scenarios[2 * j + 1], &up)]
                    {
                        assert_eq!(scenario.times(), external.time_nodes());
                        assert_eq!(
                            scenario.lsv_surface().unwrap().squared_leverage(),
                            external.lsv_squared_leverage()
                        );
                    }
                }
                assert_eq!(plan.evaluate().unwrap(), before);
                assert_ne!(
                    r.risk_fingerprint,
                    plan.evaluate_parallel_local_volatility_risk(0.005)
                        .unwrap()
                        .risk_fingerprint
                );
            }
        }
    }
}

#[test]
fn parallel_local_volatility_errors_are_paired_mc_units_and_scrambles() {
    for qmc in [false, true] {
        for antithetic in [false, true] {
            let mut v = payload();
            if qmc {
                v["engine"] = rqmc();
            }
            v["engine"]["variance_reduction"] =
                json!({"antithetic":antithetic,"brownian_bridge":true});
            let plan = compile(&v, 0.6).unwrap();
            let r = plan.evaluate_parallel_local_volatility_risk(0.01).unwrap();
            let mut parallel = plan.clone();
            parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
            assert_eq!(
                r,
                parallel
                    .evaluate_parallel_local_volatility_risk(0.01)
                    .unwrap()
            );
            let mut scenarios = Vec::new();
            for h in r.local_volatility_bumps {
                for shift in [-h, h] {
                    scenarios.push(compile(&bumped(&v, shift), 0.6).unwrap());
                }
            }
            let sample = |z: Vec<f64>, vr: VarianceReduction| {
                let bridge = plan.inner.bridge(vr).unwrap();
                let prices = scenarios
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
                    row[j] =
                        (prices[2 * j + 1] - prices[2 * j]) / (2.0 * r.local_volatility_bumps[j]);
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
            let values = r
                .vega_estimates
                .into_iter()
                .chain(r.bump_differences)
                .collect::<Vec<_>>();
            let errors = r
                .vega_standard_errors
                .into_iter()
                .chain(r.bump_difference_standard_errors)
                .collect::<Vec<_>>();
            assert_eq!(r.price.independent_sampling_units, rows.len() as u64);
            for j in 0..5 {
                let mean = rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64;
                let se = (rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>()
                    / (rows.len() * (rows.len() - 1)) as f64)
                    .sqrt();
                assert!((values[j] - mean).abs() < 2e-10);
                assert!((errors[j] - se).abs() < 2e-10);
            }
        }
    }
}

#[test]
fn parallel_local_volatility_respects_history_and_rejects_invalid_ladders() {
    let mut v = payload();
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["historical_hit"] = json!(true);
    v["product"]["monitoring_dates"] = json!(["2026-09-03", "2027-09-03"]);
    v["product"]["notional"] = json!(1e18);
    let plan = compile(&v, 0.6).unwrap();
    let r = plan.evaluate_parallel_local_volatility_risk(0.01).unwrap();
    assert_eq!(r.vega_estimates, [0.0; 3]);
    assert_eq!(r.vega_standard_errors, [0.0; 3]);
    assert_eq!(r.bump_difference_standard_errors, [0.0; 2]);
    assert_eq!(r.price.value, 7.0 * plan.inner.base.discount());
    for h in [
        0.0,
        -0.01,
        f64::NAN,
        f64::INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
        0.2,
    ] {
        assert!(plan.evaluate_parallel_local_volatility_risk(h).is_err());
    }
    for (field, value) in [("floor", 0.03), ("cap", 0.06)] {
        let mut edge = v.clone();
        edge["model"]["local_variance_grid"][field] = json!(value);
        assert!(
            compile(&edge, 0.6)
                .unwrap()
                .evaluate_parallel_local_volatility_risk(0.01)
                .is_err()
        );
    }
}

#[test]
fn parallel_local_volatility_matches_independent_gbm_quadrature() {
    for direction in ["up", "down"] {
        for side in ["call", "put"] {
            let mut v = payload();
            v["market"]
                .as_object_mut()
                .unwrap()
                .remove("discrete_dividends");
            v["model"]["local_variance_grid"]["values"] = json!(vec![0.04; 9]);
            v["product"]["strike"] = json!(100.0);
            v["product"]["barrier"] = json!(if direction == "up" { 130.0 } else { 70.0 });
            v["product"]["direction"] = json!({"type":direction});
            v["product"]["side"] = json!({"type":side});
            v["product"]["style"] = json!({"type":"knock_out"});
            v["engine"] = rqmc();
            v["engine"]["points_per_scramble"] = json!(8192);
            v["engine"]["scramble_count"] = json!(8);
            let r = compile(&v, 0.0)
                .unwrap()
                .evaluate_parallel_local_volatility_risk(0.01)
                .unwrap();
            for (j, h) in r.local_volatility_bumps.into_iter().enumerate() {
                let up =
                    gbm_reference_with_volatility(100.0, 0.2 + h, direction, side, "knock_out");
                let down =
                    gbm_reference_with_volatility(100.0, 0.2 - h, direction, side, "knock_out");
                let expected = (up - down) / (2.0 * h);
                assert!(
                    (r.vega_estimates[j] - expected).abs() < 5.0 * r.vega_standard_errors[j] + 0.03,
                    "{direction} {side} h={h}: {} +/- {} vs {expected}",
                    r.vega_estimates[j],
                    r.vega_standard_errors[j]
                );
                assert!(r.vega_standard_errors[j] < 1.0);
            }
        }
    }
}
