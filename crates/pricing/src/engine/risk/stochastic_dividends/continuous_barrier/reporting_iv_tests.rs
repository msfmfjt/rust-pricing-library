use super::tests::{compile, payload};
use super::*;
use serde_json::{Value, json};

fn reporting_payload() -> Value {
    let mut v = payload();
    v["model"]["reporting_iv_basis"] = json!({
        "maturity_nodes":[91.0/365.0,364.0/365.0],
        "log_forward_moneyness_nodes":[-0.25,0.25],"shape":[2,2],"implied_volatilities":vec![0.2;4]});
    v
}
fn bumped(v: &Value, node: usize, shift: f64) -> Value {
    let mut result = v.clone();
    let x = &mut result["model"]["local_variance_grid"]["values"][node];
    *x = json!((x.as_f64().unwrap().sqrt() + shift).powi(2));
    result
}
// Independently specified density-hat/bilinear map for this nonmatching basis.
fn weights(node: usize, threshold: f64) -> [f64; 4] {
    let (t, x) = (node / 3, node % 3);
    if t == 0 || (threshold == 0.9 && x != 1) {
        return [0.0; 4];
    }
    let density_scale = if x == 1 { 2.0 } else { 4.0 };
    let spatial = match x {
        0 => [1.0, 0.0],
        1 => [0.5, 0.5],
        _ => [0.0, 1.0],
    };
    let time = if t == 1 {
        [2.0 / 3.0, 1.0 / 3.0]
    } else {
        [0.0, 1.0]
    };
    std::array::from_fn(|b| density_scale * time[b / 2] * spatial[b % 2])
}
fn project(nodes: &[[f64; 3]], threshold: f64) -> Vec<f64> {
    let mut out = vec![0.0; 29]; // four buckets with gaps, then pre/sum/residual triples.
    for b in 0..4 {
        for j in 0..3 {
            out[5 * b + j] = (0..9).map(|i| weights(i, threshold)[b] * nodes[i][j]).sum();
        }
        out[5 * b + 3] = out[5 * b] - out[5 * b + 1];
        out[5 * b + 4] = out[5 * b + 1] - out[5 * b + 2];
    }
    for j in 0..3 {
        out[20 + j] = nodes.iter().map(|r| r[j]).sum();
        out[23 + j] = (0..4).map(|b| out[5 * b + j]).sum();
        out[26 + j] = out[20 + j] - out[23 + j];
    }
    out
}
fn result_columns(r: &StochasticDividendContinuousBarrierReportingIvRisk) -> (Vec<f64>, Vec<f64>) {
    let values = (0..4)
        .flat_map(|b| {
            r.bucket_estimates[b]
                .into_iter()
                .chain(r.bump_differences[b])
        })
        .chain(r.pre_projection_estimates)
        .chain(r.projected_sum_estimates)
        .chain(r.residual_estimates)
        .collect();
    let errors = (0..4)
        .flat_map(|b| {
            r.bucket_standard_errors[b]
                .into_iter()
                .chain(r.bump_difference_standard_errors[b])
        })
        .chain(r.pre_projection_standard_errors)
        .chain(r.projected_sum_standard_errors)
        .chain(r.residual_standard_errors)
        .collect();
    (values, errors)
}
#[test]
fn reporting_iv_map_handles_time_zero_density_cutoff_and_nonmatching_basis() {
    for threshold in [1e-8, 0.9] {
        let plan = compile(&reporting_payload(), 0.6).unwrap();
        let map = plan.reporting_projection(threshold).unwrap();
        for node in 0..9 {
            for b in 0..4 {
                assert!((map.node_weights[node][b] - weights(node, threshold)[b]).abs() < 2e-14);
            }
        }
        assert_eq!(map.density_rows.len(), 2);
        for row in &map.density_rows {
            assert_eq!(
                row.active_domain().start_index(),
                if threshold == 0.9 { 1 } else { 0 }
            );
            assert_eq!(
                row.active_domain().end_index(),
                if threshold == 0.9 { 1 } else { 2 }
            );
        }
        let r = plan
            .evaluate_reporting_iv_projection(0.01, threshold)
            .unwrap();
        let nodes = plan
            .evaluate_bucketed_local_volatility_risk(0.01, &(0..9).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(r.price, nodes.price);
        assert_eq!(r.price, plan.evaluate().unwrap());
        let expected = project(&nodes.vega_estimates, threshold);
        let (actual, _) = result_columns(&r);
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 2e-10);
        }
        assert_eq!(r.recalibration_count, 54);
        assert_eq!(r.scenario_evaluated_paths, 55 * r.price.evaluated_paths);
        assert_eq!(r.payoff_evaluations, r.scenario_evaluated_paths);
        assert_eq!(r.reporting_maturity_nodes, [91.0 / 365.0, 364.0 / 365.0]);
        assert_eq!(r.positive_target_time_nodes, [182.0 / 365.0, 364.0 / 365.0]);
        assert_eq!(r.reporting_implied_volatilities, vec![0.2; 4]);
    }
}
#[test]
fn reporting_iv_errors_project_paired_mc_units_and_rqmc_scrambles() {
    for qmc in [false, true] {
        for antithetic in [false, true] {
            let mut v = reporting_payload();
            if qmc {
                v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":16,
            "scramble_count":4,"master_scramble_seed":193});
            }
            v["engine"]["variance_reduction"] =
                json!({"antithetic":antithetic,"brownian_bridge":true});
            let plan = compile(&v, 0.6).unwrap();
            let r = plan.evaluate_reporting_iv_projection(0.01, 1e-8).unwrap();
            let mut parallel = plan.clone();
            parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
            assert_eq!(
                r,
                parallel
                    .evaluate_reporting_iv_projection(0.01, 1e-8)
                    .unwrap()
            );
            let mut scenarios = Vec::new();
            for node in 0..9 {
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
                let nodes = (0..9)
                    .map(|i| {
                        std::array::from_fn(|j| {
                            (prices[6 * i + 2 * j + 1] - prices[6 * i + 2 * j])
                                / (2.0 * r.local_volatility_bumps[j])
                        })
                    })
                    .collect::<Vec<_>>();
                project(&nodes, 1e-8)
            };
            let dim = plan.inner.path.random_dimension();
            let rows = match plan.inner.engine {
                EngineConfig::PseudoMonteCarlo(c) => {
                    let rng = Philox4x32::from_seed(c.master_seed());
                    (0..c.independent_sampling_units().get())
                        .map(|p| {
                            sample(
                                (0..dim)
                                    .map(|d| {
                                        rng.standard_normal(RandomCoordinate::new(
                                            p,
                                            d,
                                            RandomDomain::Valuation,
                                        ))
                                    })
                                    .collect(),
                                c.variance_reduction(),
                            )
                        })
                        .collect::<Vec<_>>()
                }
                EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                    let q = RqmcPlan::compile(c, dim).unwrap();
                    (0..c.scramble_count().get())
                        .map(|s| {
                            let mut mean = vec![0.0; 29];
                            for p in 0..c.points_per_scramble().get() {
                                let row = sample(
                                    (0..dim)
                                        .map(|d| {
                                            inverse_standard_normal(q.uniform(s, p, d).unwrap())
                                                .unwrap()
                                        })
                                        .collect(),
                                    c.variance_reduction(),
                                );
                                for j in 0..29 {
                                    mean[j] += row[j] / c.points_per_scramble().get() as f64;
                                }
                            }
                            mean
                        })
                        .collect::<Vec<_>>()
                }
            };
            let (values, errors) = result_columns(&r);
            assert_eq!(r.price.independent_sampling_units, rows.len() as u64);
            for j in 0..29 {
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
fn reporting_iv_rejects_invalid_inputs_and_preserves_fixed_history() {
    let missing = compile(&payload(), 0.6).unwrap();
    assert!(matches!(
        missing.evaluate_reporting_iv_projection(0.01, 1e-8),
        Err(MonteCarloError::MissingLocalVolatilityReportingBasis)
    ));
    let mut v = reporting_payload();
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["historical_hit"] = json!(true);
    v["product"]["monitoring_dates"] = json!(["2026-09-03", "2027-09-03"]);
    v["product"]["notional"] = json!(1e18);
    let plan = compile(&v, 0.6).unwrap();
    let r = plan.evaluate_reporting_iv_projection(0.01, 0.9).unwrap();
    assert_eq!(r.price.value, 7.0 * plan.inner.base.discount());
    let (values, errors) = result_columns(&r);
    assert!(values.iter().chain(&errors).all(|v| *v == 0.0));
    for t in [0.0, -0.1, 1.01, f64::NAN, f64::INFINITY] {
        assert!(plan.evaluate_reporting_iv_projection(0.01, t).is_err());
    }
    for h in [0.0, -0.01, 0.2, f64::MIN_POSITIVE, f64::NAN, f64::INFINITY] {
        assert!(plan.evaluate_reporting_iv_projection(h, 0.9).is_err());
    }
    for (h, t) in [(0.005, 0.9), (0.01, 1e-8)] {
        assert_ne!(
            r.risk_fingerprint,
            plan.evaluate_reporting_iv_projection(h, t)
                .unwrap()
                .risk_fingerprint
        );
    }
    v["model"]["reporting_iv_basis"]["maturity_nodes"] = json!([0.75, 1.0]);
    assert!(
        compile(&v, 0.6)
            .unwrap()
            .evaluate_reporting_iv_projection(0.01, 0.9)
            .is_err()
    );
}
