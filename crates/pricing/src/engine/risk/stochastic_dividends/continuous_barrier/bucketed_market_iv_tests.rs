use super::market_iv_tests::{QUOTES, independent_quote_payload, quote_payload, surface};
use super::tests::compile;
use super::*;
use serde_json::{Value, json};

fn bumped(v: &Value, quote: usize, shift: f64) -> Value {
    let mut quotes = QUOTES;
    quotes[quote] += shift;
    independent_quote_payload(v, &quotes)
}
fn rqmc() -> Value {
    json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,
        "scramble_count":4,"master_scramble_seed":193,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
}

#[test]
fn bucketed_market_iv_rebuilds_independent_dupire_for_all_styles() {
    for direction in ["up", "down"] {
        for side in ["call", "put"] {
            for style in ["knock_out", "knock_in"] {
                let mut v = quote_payload();
                v["engine"] = rqmc();
                v["product"]["direction"] = json!({"type":direction});
                v["product"]["side"] = json!({"type":side});
                v["product"]["style"] = json!({"type":style});
                v["product"]["barrier"] = json!(if direction == "up" { 120.0 } else { 80.0 });
                let plan = compile(&v, 0.6)
                    .unwrap()
                    .with_market_iv_surface(surface(0.0))
                    .unwrap();
                let before = plan.evaluate().unwrap();
                // Non-sorted selection covers both quote times, including one beyond expiry.
                let r = plan
                    .evaluate_bucketed_market_iv_risk(0.01, &[4, 0, 5])
                    .unwrap();
                assert_eq!(r.price, before);
                assert_eq!(r.quote_indices, [4, 0, 5]);
                assert_eq!(r.quote_maturity_nodes, [0.25, 1.25]);
                assert_eq!(r.quote_log_moneyness_nodes, [-0.75, 0.0, 0.75]);
                assert_eq!(r.implied_volatilities, QUOTES);
                assert!(r.quote_maturity_nodes[1] > *plan.time_nodes().last().unwrap());
                assert_eq!(r.recalibration_count, 18);
                assert_eq!(r.scenario_evaluated_paths, 19 * r.price.evaluated_paths);
                assert_eq!(r.payoff_evaluations, r.scenario_evaluated_paths);
                let selections = r
                    .quote_indices
                    .iter()
                    .copied()
                    .map(Some)
                    .collect::<Vec<_>>();
                let scenarios = plan
                    .market_iv_scenarios_for_quotes(&r.implied_volatility_bumps, &selections)
                    .unwrap();
                for (b, &node) in r.quote_indices.iter().enumerate() {
                    for (j, h) in r.implied_volatility_bumps.into_iter().enumerate() {
                        let down = compile(&bumped(&v, node, -h), 0.6).unwrap();
                        let up = compile(&bumped(&v, node, h), 0.6).unwrap();
                        let expected = (up.evaluate().unwrap().value
                            - down.evaluate().unwrap().value)
                            / (2.0 * h);
                        assert!((r.vega_estimates[b][j] - expected).abs() < 2e-9);
                        for (scenario, external) in [
                            (&scenarios[6 * b + 2 * j], &down),
                            (&scenarios[6 * b + 2 * j + 1], &up),
                        ] {
                            assert_eq!(scenario.times(), external.time_nodes());
                            for (actual, expected) in scenario
                                .lsv_surface()
                                .unwrap()
                                .squared_leverage()
                                .iter()
                                .zip(external.lsv_squared_leverage())
                            {
                                assert!((actual - expected).abs() < 1e-12);
                            }
                        }
                    }
                }
                assert_eq!(plan.evaluate().unwrap(), before);
            }
        }
    }
}

#[test]
fn bucketed_market_iv_errors_include_paired_cross_node_covariance() {
    for qmc in [false, true] {
        for antithetic in [false, true] {
            let mut v = quote_payload();
            if qmc {
                v["engine"] = rqmc();
            }
            v["engine"]["variance_reduction"] =
                json!({"antithetic":antithetic,"brownian_bridge":true});
            let plan = compile(&v, 0.6)
                .unwrap()
                .with_market_iv_surface(surface(0.0))
                .unwrap();
            let r = plan
                .evaluate_bucketed_market_iv_risk(0.01, &[4, 0])
                .unwrap();
            let mut parallel = plan.clone();
            parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
            assert_eq!(
                r,
                parallel
                    .evaluate_bucketed_market_iv_risk(0.01, &[4, 0])
                    .unwrap()
            );
            let mut scenarios = Vec::new();
            for node in [4, 0] {
                for h in r.implied_volatility_bumps {
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
                            / (2.0 * r.implied_volatility_bumps[j]);
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
                assert!((values[j] - mean).abs() < 2e-9);
                assert!((errors[j] - se).abs() < 2e-9);
            }
        }
    }
}

#[test]
fn bucketed_market_iv_selection_history_and_bounds() {
    let mut v = quote_payload();
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["historical_hit"] = json!(true);
    v["product"]["monitoring_dates"] = json!(["2026-09-03", "2027-09-03"]);
    v["product"]["notional"] = json!(1e18);
    let plan = compile(&v, 0.6)
        .unwrap()
        .with_market_iv_surface(surface(0.0))
        .unwrap();
    let r = plan
        .evaluate_bucketed_market_iv_risk(0.01, &[4, 0])
        .unwrap();
    assert_eq!(r.price.value, 7.0 * plan.inner.base.discount());
    assert_eq!(r.vega_estimates, vec![[0.0; 3]; 2]);
    assert_eq!(r.vega_standard_errors, vec![[0.0; 3]; 2]);
    assert_eq!(r.sum_vega_estimates, [0.0; 3]);
    assert_eq!(r.sum_vega_standard_errors, [0.0; 3]);
    assert_eq!(r.sum_bump_difference_standard_errors, [0.0; 2]);
    for nodes in [&[][..], &[6], &[usize::MAX], &[0, 0]] {
        assert!(plan.evaluate_bucketed_market_iv_risk(0.01, nodes).is_err());
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
        assert!(plan.evaluate_bucketed_market_iv_risk(h, &[0]).is_err());
    }
    for (h, nodes) in [(0.005, &[4, 0][..]), (0.01, &[0, 4]), (0.01, &[4])] {
        assert_ne!(
            r.risk_fingerprint,
            plan.evaluate_bucketed_market_iv_risk(h, nodes)
                .unwrap()
                .risk_fingerprint
        );
    }
    assert!(
        compile(&v, 0.6)
            .unwrap()
            .evaluate_bucketed_market_iv_risk(0.01, &[0])
            .is_err()
    );
    // The half/base steps fit; only the double step crosses the Dupire cap.
    let flat =
        crate::market::MarketIvSurface::new(vec![0.25, 1.25], vec![-0.75, 0.0, 0.75], vec![0.2; 6])
            .unwrap();
    let original = plan.original_local_variance_target().unwrap();
    let target = flat
        .local_variance_grid(
            original.time_nodes().to_vec(),
            original.log_moneyness_nodes().to_vec(),
            1e-8,
            0.05,
        )
        .unwrap();
    v["model"]["local_variance_grid"]["values"] = json!(target.values());
    v["model"]["local_variance_grid"]["cap"] = json!(0.05);
    let capped = compile(&v, 0.6)
        .unwrap()
        .with_market_iv_surface(flat)
        .unwrap();
    assert!(capped.evaluate_bucketed_market_iv_risk(0.005, &[4]).is_ok());
    assert!(capped.evaluate_bucketed_market_iv_risk(0.01, &[4]).is_err());
}

#[test]
fn bucketed_market_iv_order_single_bucket_and_parallel_limit() {
    let mut v = quote_payload();
    v["engine"] = rqmc();
    let plan = compile(&v, 0.6)
        .unwrap()
        .with_market_iv_surface(surface(0.0))
        .unwrap();
    let forward = plan
        .evaluate_bucketed_market_iv_risk(0.001, &[4, 0])
        .unwrap();
    let reverse = plan
        .evaluate_bucketed_market_iv_risk(0.001, &[0, 4])
        .unwrap();
    assert_ne!(forward.risk_fingerprint, reverse.risk_fingerprint);
    for b in 0..2 {
        assert_eq!(forward.vega_estimates[b], reverse.vega_estimates[1 - b]);
        assert_eq!(
            forward.vega_standard_errors[b],
            reverse.vega_standard_errors[1 - b]
        );
    }
    let single = plan.evaluate_bucketed_market_iv_risk(0.001, &[4]).unwrap();
    assert_eq!(single.vega_estimates[0], forward.vega_estimates[0]);
    assert_eq!(
        single.vega_standard_errors[0],
        single.sum_vega_standard_errors
    );
    assert_eq!(
        single.bump_difference_standard_errors[0],
        single.sum_bump_difference_standard_errors
    );
    assert_eq!(single.vega_estimates[0], single.sum_vega_estimates);
    // With smooth, small bumps the sum approaches simultaneous parallel risk;
    // finite single-quote differences need not add up exactly at larger bumps.
    let mut gaps = Vec::new();
    for h in [0.002, 0.0002] {
        let all = plan
            .evaluate_bucketed_market_iv_risk(h, &[0, 1, 2, 3, 4, 5])
            .unwrap();
        let parallel = plan.evaluate_parallel_market_iv_risk(h).unwrap();
        gaps.push((all.sum_vega_estimates[1] - parallel.vega()).abs());
    }
    assert!(gaps[0] > 1e-7);
    assert!(gaps[1] < gaps[0] / 10.0, "{gaps:?}");
}
