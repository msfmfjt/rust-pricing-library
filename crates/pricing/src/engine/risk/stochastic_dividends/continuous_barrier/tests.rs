use super::*;
use crate::engine::risk::stochastic_dividends::lsv_path_refinement_tests::{
    Contract, EXPIRY, PAYMENT, request_value,
};
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

fn payload() -> Value {
    let mut v = request_value(Contract::Barrier);
    v["risk"]
        .as_object_mut()
        .unwrap()
        .remove("payoff_smoothing");
    v["product"]["monitoring"] = json!({"type":"continuous"});
    v["product"]["barrier"] = json!(120.0);
    v["product"]["notional"] = json!(2.0);
    v["product"]["rebate"] = json!(7.0);
    v
}
fn compile(
    v: &Value,
    eta: f64,
) -> Result<StochasticDividendContinuousBarrierPlan, MonteCarloError> {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    StochasticDividendContinuousBarrierPlan::compile_rough_bergomi_lsv(
        &r,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
        RoughBergomi::new(0.1, eta, -0.4).unwrap(),
        0.15,
        LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
        EXPIRY / 8.0,
        ExecutionPolicy::new(1, Some(32)).unwrap(),
    )
}
fn shocks(plan: &StochasticDividendContinuousBarrierPlan, pattern: usize) -> Vec<f64> {
    (0..plan.inner.path.random_dimension() as usize)
        .map(|i| ((13 * i + 7 * pattern) % 31) as f64 / 20.0 - 0.75)
        .collect()
}

#[test]
fn physical_log_variance_includes_reserve_and_signed_correlation() {
    assert!((physical_log_variance(0.1, 0.3, -0.5) - 0.07).abs() < 1e-16);
    assert!((physical_log_variance(0.1, 0.3, 0.5) - 0.13).abs() < 1e-16);
    assert_eq!(physical_log_variance(0.2, 0.2, -1.0), 0.0);
    assert_eq!(physical_log_variance(0.0, 0.0, 1.0), 0.0);
    assert_eq!(physical_log_variance(0.2, 0.0, 0.9), 0.2_f64.powi(2));
}

#[test]
fn causal_volatility_trace_preserves_paths_and_ignores_current_innovation() {
    let plan = compile(&payload(), 0.6).unwrap();
    let mut z = shocks(&plan, 0);
    let (states, volatility) = plan
        .inner
        .path
        .evolve_rough_path_with_volatilities(&z)
        .unwrap();
    assert_eq!(states, plan.inner.path.evolve_path(&z).unwrap());
    assert_eq!(volatility.len(), states.len() - 1);
    let surface = plan.inner.path.lsv_surface().unwrap();
    assert_eq!(
        volatility[0],
        surface
            .squared_leverage_at(0.0, surface.initial_f())
            .unwrap()
            .sqrt()
    );
    z[3] += 2.0;
    let (_, shifted) = plan
        .inner
        .path
        .evolve_rough_path_with_volatilities(&z)
        .unwrap();
    assert_eq!(shifted[0], volatility[0]);
    assert_ne!(shifted[1], volatility[1]);
    assert!(
        plan.inner
            .path
            .evolve_rough_path_with_volatilities(&[])
            .is_err()
    );
    z[0] = f64::NAN;
    assert!(
        plan.inner
            .path
            .evolve_rough_path_with_volatilities(&z)
            .is_err()
    );
}

#[test]
fn continuous_in_out_parity_and_price_only_boundary() {
    for direction in ["up", "down"] {
        for side in ["call", "put"] {
            let mut v = payload();
            v["product"]["direction"] = json!({"type":direction});
            v["product"]["side"] = json!({"type":side});
            v["product"]["barrier"] = json!(if direction == "up" { 120.0 } else { 80.0 });
            v["product"]["style"] = json!({"type":"knock_in"});
            let ki = compile(&v, 0.6).unwrap();
            v["product"]["style"] = json!({"type":"knock_out"});
            let ko = compile(&v, 0.6).unwrap();
            assert_ne!(ki.plan_fingerprint(), ko.plan_fingerprint());
            for pattern in 0..16 {
                let z = shocks(&ki, pattern);
                let states = ki.inner.path.evolve_path(&z).unwrap();
                let terminal = ki
                    .inner
                    .path
                    .nodes()
                    .last()
                    .unwrap()
                    .spots(*states.last().unwrap())
                    .unwrap()
                    .0;
                let signed = if side == "call" {
                    terminal - 80.0
                } else {
                    80.0 - terminal
                };
                let expected = 0.95_f64.powf(PAYMENT) * (2.0 * signed.max(0.0) + 7.0);
                assert!(
                    (ki.path_payoff(&z).unwrap() + ko.path_payoff(&z).unwrap() - expected).abs()
                        < 1e-12
                );
            }
            let req =
                parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
            assert!(
                StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
                    &req,
                    ki.inner.path.model(),
                    RoughBergomi::new(0.1, 0.6, -0.4).unwrap(),
                    0.15,
                    LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
                    EXPIRY / 8.0,
                    ExecutionPolicy::new(1, Some(32)).unwrap()
                )
                .is_err()
            );
        }
    }
    for (field, value) in [
        ("delta", json!(true)),
        ("vega", json!(true)),
        (
            "payoff_smoothing",
            json!({"type":"compact_c2","half_width":2.0}),
        ),
    ] {
        let mut v = payload();
        v["risk"][field] = value;
        assert!(
            compile(&v, 0.6)
                .unwrap_err()
                .to_string()
                .contains("price-only")
        );
    }
    let mut v = payload();
    v["product"]["monitoring"] = json!({"type":"discrete"});
    assert!(compile(&v, 0.6).is_err());
}

#[test]
fn jumps_and_monitoring_end_are_observed_separately() {
    for direction in ["up", "down"] {
        let mut v = payload();
        v["product"]["direction"] = json!({"type":direction});
        v["product"]["barrier"] = json!(100.0);
        v["product"]["monitoring_dates"] = json!(["2027-03-05"]);
        let plan = compile(&v, 0.6).unwrap();
        let end = plan.monitoring_end.unwrap();
        let states = vec![BuehlerDividendState::initial(); plan.time_nodes().len()];
        let volatility = vec![0.2; states.len() - 1];
        let safe = if direction == "up" { 80.0 } else { 120.0 };
        let hit = if direction == "up" { 120.0 } else { 80.0 };
        let mut spots = vec![(safe, None); states.len()];
        // A later crossing is outside the contractual window.
        spots[end + 1] = (hit, Some(hit));
        assert!(
            plan.log_survival(&states, &spots, &volatility)
                .unwrap()
                .is_finite()
        );
        spots[end] = if direction == "up" {
            (safe, Some(hit))
        } else {
            (hit, Some(safe))
        };
        assert_eq!(
            plan.log_survival(&states, &spots, &volatility).unwrap(),
            f64::NEG_INFINITY
        );
        spots[end] = (safe, Some(safe));
        spots[0].0 = 100.0;
        assert_eq!(
            plan.log_survival(&states, &spots, &volatility).unwrap(),
            f64::NEG_INFINITY
        );
    }
}

#[test]
fn resolved_history_and_initial_endpoint_use_exact_branches() {
    for (hit, dates) in [
        (true, json!(["2026-09-03", "2026-09-04", "2027-09-03"])),
        (false, json!(["2026-09-03"])),
    ] {
        for style in ["knock_in", "knock_out"] {
            let mut v = payload();
            v["product"]["historical_hit"] = json!(hit);
            v["product"]["monitoring_dates"] = dates.clone();
            v["product"]["style"] = json!({"type":style});
            v["product"]["barrier"] = json!(100.0);
            let plan = compile(&v, 0.6).unwrap();
            let price = plan.evaluate().unwrap();
            assert_eq!(price.scheme, SCHEME);
            assert_eq!(price.plan_fingerprint, plan.plan_fingerprint());
            // Existing resolved graph is an independent payoff route.
            assert_eq!(price.value, plan.inner.evaluate().unwrap().value);
            if hit != (style == "knock_in") {
                assert_eq!(price.standard_error, 0.0);
                assert!((price.value - 7.0 * 0.95_f64.powf(PAYMENT)).abs() < 2e-14);
            }
        }
    }
    let mut v = payload();
    v["product"]["monitoring_dates"] = json!(["2026-09-04"]);
    v["product"]["style"] = json!({"type":"knock_out"});
    v["product"]["barrier"] = json!(100.0);
    v["product"]["notional"] = json!(1e18);
    let price = compile(&v, 0.6).unwrap().evaluate().unwrap();
    assert!((price.value - 7.0 * 0.95_f64.powf(PAYMENT)).abs() < 2e-14);
    assert_eq!(price.standard_error, 0.0);
}

#[test]
fn mc_antithetic_errors_and_rqmc_scramble_errors_replay() {
    let mut v = payload();
    v["engine"]["variance_reduction"]["brownian_bridge"] = json!(false);
    let plan = compile(&v, 0.6).unwrap();
    let EngineConfig::PseudoMonteCarlo(config) = plan.inner.engine else {
        panic!()
    };
    let rng = Philox4x32::from_seed(config.master_seed());
    let count = config.independent_sampling_units().get();
    let values = (0..count)
        .map(|p| {
            let z = (0..plan.inner.path.random_dimension())
                .map(|d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation)))
                .collect::<Vec<_>>();
            let opposite = z.iter().map(|x| -x).collect::<Vec<_>>();
            (plan.path_payoff(&z).unwrap() + plan.path_payoff(&opposite).unwrap()) * 0.5
        })
        .collect::<Vec<_>>();
    let stats = DeterministicStatistics::from_ordered_values_two_pass(&values);
    let r = plan.evaluate().unwrap();
    assert!((r.value - stats.sum().total() / count as f64).abs() < 2e-14);
    assert!(
        (r.standard_error - (stats.moments().sample_variance().unwrap() / count as f64).sqrt())
            .abs()
            < 2e-14
    );
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,
        "scramble_count":4,"master_scramble_seed":193,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let plan = compile(&v, 0.6).unwrap();
    let r = plan.evaluate().unwrap();
    let mut parallel = plan.clone();
    parallel.inner.policy = ExecutionPolicy::new(3, Some(32)).unwrap();
    assert_eq!(r, parallel.evaluate().unwrap());
    assert_eq!(r.independent_sampling_units, 4);
    assert_eq!(r.evaluated_paths, 256);
    let EngineConfig::RandomizedQuasiMonteCarlo(c) = plan.inner.engine else {
        panic!()
    };
    let q = RqmcPlan::compile(c, plan.inner.path.random_dimension()).unwrap();
    let bridge = plan.inner.bridge(c.variance_reduction()).unwrap();
    let means = (0..4)
        .map(|s| {
            (0..32)
                .map(|p| {
                    let z = (0..plan.inner.path.random_dimension())
                        .map(|d| inverse_standard_normal(q.uniform(s, p, d).unwrap()).unwrap())
                        .collect();
                    plan.inner
                        .sample(z, bridge.as_ref(), true, &|z| plan.path_payoff(z))
                        .unwrap()
                })
                .sum::<f64>()
                / 32.0
        })
        .collect::<Vec<_>>();
    let mean = means.iter().sum::<f64>() / 4.0;
    let se = (means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / 12.0).sqrt();
    assert!((r.value - mean).abs() < 2e-14);
    assert!((r.standard_error - se).abs() < 2e-14);
}

#[test]
fn flat_no_cash_gbm_limit_matches_independent_terminal_quadrature() {
    // Gaussian terminal density times the exact one-interval GBM survival.
    // Simpson integration splits at both the strike and barrier.
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
            v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":4096,
                "scramble_count":8,"master_scramble_seed":193,
                "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
            let plan = compile(&v, 0.0).unwrap();
            let variance = 0.04 * EXPIRY;
            let drift = (0.98_f64 / 0.95).ln() * EXPIRY - 0.5 * variance;
            let level = plan.barrier.barrier().get();
            let z_barrier = ((level / 100.0).ln() - drift) / variance.sqrt();
            let z_strike = -drift / variance.sqrt();
            let density = |z: f64| (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let integrand = |z: f64| {
                let terminal = 100.0 * (drift + variance.sqrt() * z).exp();
                let safe = if direction == "up" {
                    terminal < level
                } else {
                    terminal > level
                };
                let survival = if safe {
                    -(-2.0 * (level / 100.0).ln() * (level / terminal).ln() / variance).exp_m1()
                } else {
                    0.0
                };
                let signed = if side == "call" {
                    terminal - 100.0
                } else {
                    100.0 - terminal
                };
                density(z) * (2.0 * signed.max(0.0) * survival + 7.0 * (1.0 - survival))
            };
            let mut breaks = [-10.0, z_barrier, z_strike, 10.0];
            breaks.sort_by(f64::total_cmp);
            let mut expected = 0.0;
            for w in breaks.windows(2) {
                let n = 4096;
                let dz = (w[1] - w[0]) / n as f64;
                let sum = (0..=n)
                    .map(|i| {
                        let weight = if i == 0 || i == n {
                            1.0
                        } else if i % 2 == 0 {
                            2.0
                        } else {
                            4.0
                        };
                        weight * integrand(w[0] + i as f64 * dz)
                    })
                    .sum::<f64>();
                expected += dz * sum / 3.0;
            }
            expected *= 0.95_f64.powf(PAYMENT);
            let r = plan.evaluate().unwrap();
            assert!(
                (r.value - expected).abs() < 5.0 * r.standard_error + 0.002,
                "{direction} {side}: {} +/- {} vs {expected}",
                r.value,
                r.standard_error
            );
            assert!(r.standard_error < 0.015);
        }
    }
}
