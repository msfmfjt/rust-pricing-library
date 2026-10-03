use super::tests::{compile, gbm_reference_with_volatility, payload};
use super::*;
use crate::market::MarketIvSurface;
use serde_json::{Value, json};

pub(super) const QUOTES: [f64; 6] = [0.22, 0.20, 0.21, 0.24, 0.22, 0.23];
pub(super) fn surface(shift: f64) -> MarketIvSurface {
    MarketIvSurface::new(
        vec![0.25, 1.25],
        vec![-0.75, 0.0, 0.75],
        QUOTES.iter().map(|v| v + shift).collect(),
    )
    .unwrap()
}
pub(super) fn quote_payload() -> Value {
    let mut v = payload();
    let plain = compile(&v, 0.6).unwrap();
    let original = plain.original_local_variance_target().unwrap();
    let target = surface(0.0)
        .local_variance_grid(
            original.time_nodes().to_vec(),
            original.log_moneyness_nodes().to_vec(),
            original.floor(),
            original.cap(),
        )
        .unwrap();
    v["model"]["local_variance_grid"]["values"] = json!(target.values());
    v
}
// Independent closed natural-cubic construction for two time rows and three
// equally spaced quote strikes. No production surface/derivative call is used.
fn independent_variance(t: f64, x: f64, quotes: &[f64; 6]) -> f64 {
    let spline = |row: usize| {
        let time = [0.25, 1.25][row];
        let w: [f64; 3] = std::array::from_fn(|j| time * quotes[3 * row + j].powi(2));
        let seconds = [
            0.0,
            1.5 * (w[0] - 2.0 * w[1] + w[2]) / 0.75_f64.powi(2),
            0.0,
        ];
        let cell = usize::from(x >= 0.0);
        let b = (x - [-0.75, 0.0][cell]) / 0.75;
        let a = 1.0 - b;
        [
            a * w[cell]
                + b * w[cell + 1]
                + ((a.powi(3) - a) * seconds[cell] + (b.powi(3) - b) * seconds[cell + 1])
                    * 0.75_f64.powi(2)
                    / 6.0,
            (w[cell + 1] - w[cell]) / 0.75
                + ((1.0 - 3.0 * a * a) * seconds[cell] + (3.0 * b * b - 1.0) * seconds[cell + 1])
                    * 0.75
                    / 6.0,
            a * seconds[cell] + b * seconds[cell + 1],
        ]
    };
    let lo = spline(0);
    let hi = spline(1);
    let b = t - 0.25;
    let w: [f64; 3] = std::array::from_fn(|j| (1.0 - b) * lo[j] + b * hi[j]);
    let density = (1.0 - x * w[1] / (2.0 * w[0])).powi(2)
        - 0.25 * w[1] * w[1] * (1.0 / w[0] + 0.25)
        + 0.5 * w[2];
    (hi[0] - lo[0]) / density
}
fn bumped(v: &Value, shift: f64) -> Value {
    independent_quote_payload(v, &QUOTES.map(|sigma| sigma + shift))
}
pub(super) fn independent_quote_payload(v: &Value, quotes: &[f64; 6]) -> Value {
    let mut v = v.clone();
    let grid = &v["model"]["local_variance_grid"];
    let times = grid["time_nodes"].as_array().unwrap();
    let xs = grid["log_forward_moneyness_nodes"].as_array().unwrap();
    let values = times
        .iter()
        .flat_map(|t| {
            xs.iter().map(|x| {
                let t = t.as_f64().unwrap();
                independent_variance(
                    if t == 0.0 {
                        times[1].as_f64().unwrap()
                    } else {
                        t
                    },
                    x.as_f64().unwrap(),
                    quotes,
                )
            })
        })
        .collect::<Vec<_>>();
    v["model"]["local_variance_grid"]["values"] = json!(values);
    v
}
fn rqmc() -> Value {
    json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,"scramble_count":4,
        "master_scramble_seed":193,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
}
#[test]
fn parallel_market_iv_matches_independent_dupire_rebuild_for_all_barrier_styles() {
    for direction in ["up", "down"] {
        for side in ["call", "put"] {
            for style in ["knock_in", "knock_out"] {
                let mut v = quote_payload();
                v["engine"] = rqmc();
                v["product"]["direction"] = json!({"type":direction});
                v["product"]["side"] = json!({"type":side});
                v["product"]["style"] = json!({"type":style});
                v["product"]["barrier"] = json!(if direction == "up" { 120.0 } else { 80.0 });
                let plain = compile(&v, 0.6).unwrap();
                let plan = plain.clone().with_market_iv_surface(surface(0.0)).unwrap();
                let r = plan.evaluate_parallel_market_iv_risk(0.01).unwrap();
                assert!(plan.supports_market_iv_risk());
                assert!(!plain.supports_market_iv_risk());
                assert_ne!(plan.plan_fingerprint(), plain.plan_fingerprint());
                let before = plain.evaluate().unwrap();
                assert_eq!(r.price, plan.evaluate().unwrap());
                assert_eq!(
                    (r.price.value, r.price.standard_error),
                    (before.value, before.standard_error)
                );
                assert_eq!(r.quote_maturity_nodes, [0.25, 1.25]);
                assert_eq!(r.quote_log_moneyness_nodes, [-0.75, 0.0, 0.75]);
                assert_eq!(r.implied_volatilities, QUOTES);
                assert_eq!(r.recalibration_count, 6);
                assert_eq!(r.payoff_evaluations, 7 * r.price.evaluated_paths);
                assert_eq!(r.scenario_evaluated_paths, r.payoff_evaluations);
                for (j, h) in r.implied_volatility_bumps.iter().enumerate() {
                    let down = compile(&bumped(&v, -h), 0.6).unwrap().evaluate().unwrap();
                    let up = compile(&bumped(&v, *h), 0.6).unwrap().evaluate().unwrap();
                    assert!(
                        (r.vega_estimates[j] - (up.value - down.value) / (2.0 * h)).abs() < 2e-9
                    );
                }
                assert_eq!(r.vega_per_vol_point(), 0.01 * r.vega());
                assert_eq!(r.standard_error_per_vol_point(), 0.01 * r.standard_error());
                assert_ne!(
                    r.risk_fingerprint,
                    plan.evaluate_parallel_market_iv_risk(0.005)
                        .unwrap()
                        .risk_fingerprint
                );
                // The nonflat quote smile makes this different from bumping sqrt(v_local).
                let local = plan.evaluate_parallel_local_volatility_risk(0.01).unwrap();
                assert!((r.vega() - local.vega()).abs() > 1e-5);
            }
        }
    }
}

#[test]
fn parallel_market_iv_errors_are_paired_mc_units_and_scrambles() {
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
            let r = plan.evaluate_parallel_market_iv_risk(0.01).unwrap();
            let mut parallel = plan.clone();
            parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
            assert_eq!(r, parallel.evaluate_parallel_market_iv_risk(0.01).unwrap());
            let mut scenarios = Vec::new();
            for h in r.implied_volatility_bumps {
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
                        (prices[2 * j + 1] - prices[2 * j]) / (2.0 * r.implied_volatility_bumps[j]);
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
                assert!((values[j] - mean).abs() < 2e-9);
                assert!((errors[j] - se).abs() < 2e-9);
            }
        }
    }
}

#[test]
fn market_iv_source_validation_and_fixed_history_are_explicit() {
    let v = quote_payload();
    let plain = compile(&v, 0.6).unwrap();
    assert!(plain.evaluate_parallel_market_iv_risk(0.01).is_err());
    let plan = plain.clone().with_market_iv_surface(surface(0.0)).unwrap();
    assert!(plan.clone().with_market_iv_surface(surface(0.0)).is_err());
    assert!(
        plain
            .clone()
            .with_market_iv_surface(surface(0.001))
            .is_err()
    );
    for h in [0.0, -0.01, 0.2, 1e-300, f64::NAN, f64::INFINITY] {
        assert!(plan.evaluate_parallel_market_iv_risk(h).is_err());
    }
    assert!(
        surface(0.0)
            .local_variance_grid(vec![0.0, 1.0], vec![-0.8, 0.0], 1e-8, 4.0)
            .is_err()
    );
    assert!(
        surface(0.0)
            .local_variance_grid(vec![0.0, 1.0], vec![-0.5, 0.5], 0.2, 4.0)
            .is_err()
    );
    assert!(
        surface(0.0)
            .local_variance_grid(vec![0.0, 1.0], vec![-0.5, 0.5], 1e-8, 0.01)
            .is_err()
    );
    let mut v = v;
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["historical_hit"] = json!(true);
    v["product"]["monitoring_dates"] = json!(["2026-09-03", "2027-09-03"]);
    v["product"]["notional"] = json!(1e18);
    let plan = compile(&v, 0.6)
        .unwrap()
        .with_market_iv_surface(surface(0.0))
        .unwrap();
    let r = plan.evaluate_parallel_market_iv_risk(0.01).unwrap();
    assert_eq!(r.price.value, 7.0 * plan.inner.base.discount());
    assert_eq!(r.vega_estimates, [0.0; 3]);
    assert_eq!(r.vega_standard_errors, [0.0; 3]);
    assert_eq!(r.bump_difference_standard_errors, [0.0; 2]);
}

#[test]
fn parallel_market_iv_matches_independent_gbm_quadrature() {
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
            let source =
                MarketIvSurface::new(vec![0.25, 1.25], vec![-0.75, 0.75], vec![0.2; 4]).unwrap();
            let original = compile(&v, 0.0).unwrap();
            let grid = original.original_local_variance_target().unwrap();
            let target = source
                .local_variance_grid(
                    grid.time_nodes().to_vec(),
                    grid.log_moneyness_nodes().to_vec(),
                    grid.floor(),
                    grid.cap(),
                )
                .unwrap();
            v["model"]["local_variance_grid"]["values"] = json!(target.values());
            let r = compile(&v, 0.0)
                .unwrap()
                .with_market_iv_surface(source)
                .unwrap()
                .evaluate_parallel_market_iv_risk(0.01)
                .unwrap();
            for (j, h) in r.implied_volatility_bumps.into_iter().enumerate() {
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
