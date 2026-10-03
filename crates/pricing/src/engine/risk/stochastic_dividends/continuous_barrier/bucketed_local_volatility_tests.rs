use super::tests::{compile, payload};
use super::*;
use serde_json::{Value, json};

fn bumped(v: &Value, node: usize, shift: f64) -> Value {
    let mut result = v.clone();
    let x = &mut result["model"]["local_variance_grid"]["values"][node];
    *x = json!((x.as_f64().unwrap().sqrt() + shift).powi(2));
    result
}
fn rqmc() -> Value {
    json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,
        "scramble_count":4,"master_scramble_seed":193,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
}

#[test]
fn bucketed_local_volatility_recompiles_original_nodes_for_all_styles() {
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
                // Non-sorted selection covers t=0, interior and terminal target rows.
                let r = plan
                    .evaluate_bucketed_local_volatility_risk(0.01, &[4, 0, 8])
                    .unwrap();
                assert_eq!(r.price, before);
                assert_eq!(r.node_indices, [4, 0, 8]);
                assert_eq!(
                    r.time_nodes,
                    plan.original_local_variance_target().unwrap().time_nodes()
                );
                assert_eq!(r.log_moneyness_nodes, [-0.5, 0.0, 0.5]);
                assert!(r.time_nodes.len() < plan.time_nodes().len());
                assert_eq!(r.recalibration_count, 18);
                assert_eq!(r.scenario_evaluated_paths, 19 * r.price.evaluated_paths);
                assert_eq!(r.payoff_evaluations, r.scenario_evaluated_paths);
                let selections = r.node_indices.iter().copied().map(Some).collect::<Vec<_>>();
                let scenarios = plan
                    .local_volatility_scenarios_for_nodes(&r.local_volatility_bumps, &selections)
                    .unwrap();
                for (b, &node) in r.node_indices.iter().enumerate() {
                    for (j, h) in r.local_volatility_bumps.into_iter().enumerate() {
                        let down = compile(&bumped(&v, node, -h), 0.6).unwrap();
                        let up = compile(&bumped(&v, node, h), 0.6).unwrap();
                        let expected = (up.evaluate().unwrap().value
                            - down.evaluate().unwrap().value)
                            / (2.0 * h);
                        assert!((r.vega_estimates[b][j] - expected).abs() < 2e-10);
                        for (scenario, external) in [
                            (&scenarios[6 * b + 2 * j], &down),
                            (&scenarios[6 * b + 2 * j + 1], &up),
                        ] {
                            assert_eq!(scenario.times(), external.time_nodes());
                            assert_eq!(
                                scenario.lsv_surface().unwrap().squared_leverage(),
                                external.lsv_squared_leverage()
                            );
                        }
                    }
                }
                assert_eq!(plan.evaluate().unwrap(), before);
            }
        }
    }
}

#[test]
fn bucketed_local_volatility_errors_include_paired_cross_node_covariance() {
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
                .evaluate_bucketed_local_volatility_risk(0.01, &[4, 0])
                .unwrap();
            let mut parallel = plan.clone();
            parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
            assert_eq!(
                r,
                parallel
                    .evaluate_bucketed_local_volatility_risk(0.01, &[4, 0])
                    .unwrap()
            );
            let mut scenarios = Vec::new();
            for node in [4, 0] {
                for h in r.local_volatility_bumps {
                    for shift in [-h, h] {
                        scenarios.push(compile(&bumped(&v, node, shift), 0.6).unwrap());
                    }
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
                let mut row = [0.0; 15];
                for bucket in 0..2 {
                    for j in 0..3 {
                        row[5 * bucket + j] = (prices[6 * bucket + 2 * j + 1]
                            - prices[6 * bucket + 2 * j])
                            / (2.0 * r.local_volatility_bumps[j]);
                    }
                    row[5 * bucket + 3] = row[5 * bucket] - row[5 * bucket + 1];
                    row[5 * bucket + 4] = row[5 * bucket + 1] - row[5 * bucket + 2];
                }
                for j in 0..5 {
                    row[10 + j] = row[j] + row[5 + j];
                }
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
                            let mut mean = [0.0; 15];
                            for p in 0..c.points_per_scramble().get() {
                                let z = (0..dimension)
                                    .map(|d| {
                                        inverse_standard_normal(q.uniform(s, p, d).unwrap())
                                            .unwrap()
                                    })
                                    .collect();
                                let row = sample(z, c.variance_reduction());
                                for j in 0..15 {
                                    mean[j] += row[j] / c.points_per_scramble().get() as f64;
                                }
                            }
                            mean
                        })
                        .collect::<Vec<_>>()
                }
            };
            let values = (0..2)
                .flat_map(|b| r.vega_estimates[b].into_iter().chain(r.bump_differences[b]))
                .chain(r.sum_vega_estimates)
                .chain(r.sum_bump_differences)
                .collect::<Vec<_>>();
            let errors = (0..2)
                .flat_map(|b| {
                    r.vega_standard_errors[b]
                        .into_iter()
                        .chain(r.bump_difference_standard_errors[b])
                })
                .chain(r.sum_vega_standard_errors)
                .chain(r.sum_bump_difference_standard_errors)
                .collect::<Vec<_>>();
            assert_eq!(r.price.independent_sampling_units, rows.len() as u64);
            for j in 0..15 {
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
fn bucketed_local_volatility_selection_history_and_bounds() {
    let mut v = payload();
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["historical_hit"] = json!(true);
    v["product"]["monitoring_dates"] = json!(["2026-09-03", "2027-09-03"]);
    v["product"]["notional"] = json!(1e18);
    let plan = compile(&v, 0.6).unwrap();
    let r = plan
        .evaluate_bucketed_local_volatility_risk(0.01, &[4, 0])
        .unwrap();
    assert_eq!(r.price.value, 7.0 * plan.inner.base.discount());
    assert_eq!(r.vega_estimates, vec![[0.0; 3]; 2]);
    assert_eq!(r.vega_standard_errors, vec![[0.0; 3]; 2]);
    assert_eq!(r.sum_vega_estimates, [0.0; 3]);
    assert_eq!(r.sum_vega_standard_errors, [0.0; 3]);
    assert_eq!(r.sum_bump_difference_standard_errors, [0.0; 2]);
    for nodes in [&[][..], &[9], &[usize::MAX], &[0, 0]] {
        assert!(
            plan.evaluate_bucketed_local_volatility_risk(0.01, nodes)
                .is_err()
        );
    }
    for h in [
        0.0,
        -0.01,
        f64::NAN,
        f64::INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
        0.2,
    ] {
        assert!(
            plan.evaluate_bucketed_local_volatility_risk(h, &[0])
                .is_err()
        );
    }
    for (h, nodes) in [(0.005, &[4, 0][..]), (0.01, &[0, 4]), (0.01, &[4])] {
        assert_ne!(
            r.risk_fingerprint,
            plan.evaluate_bucketed_local_volatility_risk(h, nodes)
                .unwrap()
                .risk_fingerprint
        );
    }
    for (field, value) in [("floor", 0.04), ("cap", 0.05)] {
        let mut edge = v.clone();
        edge["model"]["local_variance_grid"]["values"] = json!(vec![value; 9]);
        edge["model"]["local_variance_grid"][field] = json!(value);
        assert!(
            compile(&edge, 0.6)
                .unwrap()
                .evaluate_bucketed_local_volatility_risk(0.01, &[4])
                .is_err()
        );
    }
    // Unselected boundary nodes are unchanged and need no positive volatility bump.
    v["model"]["local_variance_grid"]["floor"] = json!(0.035);
    assert!(
        compile(&v, 0.6)
            .unwrap()
            .evaluate_bucketed_local_volatility_risk(0.005, &[4])
            .is_ok()
    );
}
