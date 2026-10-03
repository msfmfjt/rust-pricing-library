//! Resolved continuous contracts need no stochastic-dividend bridge law.
use super::lsv_path_refinement_tests::{Contract, EXPIRY, PAYMENT, request_value};
use super::*;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

const KINDS: [&str; 7] = [
    "bs",
    "bergomi",
    "two",
    "rough",
    "lsv",
    "lsv_two",
    "lsv_rough",
];

fn payload(hit: bool, ended: bool) -> Value {
    let mut v = request_value(Contract::Barrier);
    v["risk"]
        .as_object_mut()
        .unwrap()
        .remove("payoff_smoothing");
    v["engine"]["independent_sampling_units"] = json!(16);
    v["product"]["monitoring"] = json!({"type": "continuous"});
    v["product"]["monitoring_dates"] = if ended {
        json!(["2026-09-03"])
    } else {
        // Today's endpoint must disappear when a past hit is absorbing.
        json!(["2026-09-03", "2026-09-04", "2027-03-05", "2027-09-03"])
    };
    v["product"]["historical_hit"] = json!(hit);
    v["product"]["notional"] = json!(2.0);
    v["product"]["rebate"] = json!(7.0);
    // Deliberately at today's barrier. History must not be differentiated.
    v["product"]["barrier"] = json!(100.0);
    v
}

fn compile(v: &Value, kind: &str) -> Result<StochasticDividendPricingPlan, MonteCarloError> {
    let mut v = v.clone();
    if !kind.starts_with("lsv") {
        v["model"] = json!({"type": "black_scholes", "volatility": 0.2});
    }
    let request =
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let model = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
    let one = Bergomi1Factor::new(0.8, 0.3, -0.4).unwrap();
    let two = Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [-0.4, -0.2], 0.3).unwrap();
    let rough = RoughBergomi::new(0.1, 0.6, -0.4).unwrap();
    let particles = LsvParticleConfig::new(64, 42, 0.35, 5.0, true).unwrap();
    let policy = ExecutionPolicy::new(1, Some(16)).unwrap();
    let step = EXPIRY / 8.0;
    match kind {
        "bs" => StochasticDividendPricingPlan::compile_bs(&request, model, step, policy),
        "bergomi" => {
            StochasticDividendPricingPlan::compile_bergomi(&request, model, one, 0.15, step, policy)
        }
        "two" => StochasticDividendPricingPlan::compile_bergomi_two_factor(
            &request,
            model,
            two,
            [0.15, -0.1],
            step,
            policy,
        ),
        "rough" => StochasticDividendPricingPlan::compile_rough_bergomi(
            &request, model, rough, 0.15, step, policy,
        ),
        "lsv" => StochasticDividendPricingPlan::compile_bergomi_lsv(
            &request, model, one, 0.15, particles, step, policy,
        ),
        "lsv_two" => StochasticDividendPricingPlan::compile_bergomi_two_factor_lsv(
            &request,
            model,
            two,
            [0.15, -0.1],
            particles,
            step,
            policy,
        ),
        "lsv_rough" => StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
            &request, model, rough, 0.15, particles, step, policy,
        ),
        _ => panic!("unknown test model"),
    }
}

fn price_delta(plan: &StochasticDividendPricingPlan) -> (f64, f64, f64, f64) {
    if plan.lsv.is_some() {
        let r = plan.evaluate_lsv_spot_risk().unwrap();
        (
            r.price.value,
            r.price.standard_error,
            r.delta,
            r.delta_standard_error,
        )
    } else {
        let r = plan.evaluate_aad().unwrap();
        (
            r.price.value,
            r.price.standard_error,
            r.delta(),
            r.standard_errors[0],
        )
    }
}

#[test]
fn resolved_continuous_prices_and_spot_risk_match_vanilla_or_fixed_cash() {
    let discount = 0.95_f64.powf(PAYMENT);
    let delay_discount = 0.95_f64.powf(PAYMENT - EXPIRY);
    for kind in KINDS {
        for (hit, ended) in [(true, false), (true, true), (false, true)] {
            for direction in ["up", "down"] {
                for side in ["call", "put"] {
                    for style in ["knock_in", "knock_out"] {
                        let mut v = payload(hit, ended);
                        v["product"]["direction"] = json!({"type": direction});
                        v["product"]["side"] = json!({"type": side});
                        v["product"]["style"] = json!({"type": style});
                        let plan = compile(&v, kind).unwrap();
                        let price = plan.evaluate().unwrap();
                        let actual = price_delta(&plan);
                        assert_eq!(price.value, actual.0);
                        assert_eq!(price.standard_error, actual.1);
                        assert_eq!(price.independent_sampling_units, 16);
                        assert_eq!(price.evaluated_paths, 32);
                        if hit == (style == "knock_in") {
                            v["product"] = json!({"type": "european_vanilla",
                                "underlying_id": 1, "currency_id": 2,
                                "expiry": "2027-09-03", "strike": 80.0,
                                "notional": 2.0, "side": {"type": side}});
                            let vanilla = compile(&v, kind).unwrap();
                            assert_eq!(plan.time_nodes(), vanilla.time_nodes());
                            let expected = price_delta(&vanilla);
                            for (a, e) in [actual.0, actual.1, actual.2, actual.3]
                                .into_iter()
                                .zip([expected.0, expected.1, expected.2, expected.3])
                            {
                                assert!(
                                    (a - delay_discount * e).abs() < 5e-13,
                                    "{kind}, {hit}, {ended}, {direction}, {side}, {style}: {a} vs {e}"
                                );
                            }
                        } else {
                            assert!((actual.0 - 7.0 * discount).abs() < 2e-14);
                            assert_eq!((actual.1, actual.2, actual.3), (0.0, 0.0, 0.0));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn resolved_continuous_history_is_frozen_under_spot_and_market_bumps() {
    for hit in [false, true] {
        for style in ["knock_in", "knock_out"] {
            let mut v = payload(hit, !hit);
            v["product"]["style"] = json!({"type": style});
            let plan = compile(&v, "lsv_rough").unwrap();
            let r = plan.evaluate_lsv_market_risk().unwrap();
            for bump in [0.001, 0.0005] {
                v["market"]["spot"] = json!(100.0 + bump);
                let up = compile(&v, "lsv_rough").unwrap().evaluate().unwrap().value;
                v["market"]["spot"] = json!(100.0 - bump);
                let down = compile(&v, "lsv_rough").unwrap().evaluate().unwrap().value;
                v["market"]["spot"] = json!(100.0);
                assert!((r.delta - (up - down) / (2.0 * bump)).abs() < 2e-7);
            }
            let bump = 1e-5_f64;
            for (field, expected) in [
                ("discount_curve", r.discount_log_df_adjoints[1]),
                ("dividend_curve", r.repo_spread_log_df_adjoints[1]),
            ] {
                let original = v["market"][field]["discount_factors"][1].as_f64().unwrap();
                v["market"][field]["discount_factors"][1] = json!(original * bump.exp());
                let up = compile(&v, "lsv_rough").unwrap().evaluate().unwrap().value;
                v["market"][field]["discount_factors"][1] = json!(original * (-bump).exp());
                let down = compile(&v, "lsv_rough").unwrap().evaluate().unwrap().value;
                v["market"][field]["discount_factors"][1] = json!(original);
                assert!((expected - (up - down) / (2.0 * bump)).abs() < 2e-7);
            }
            let variance = plan.evaluate_local_variance_risk().unwrap();
            let original = v["model"]["local_variance_grid"]["values"][4]
                .as_f64()
                .unwrap();
            v["model"]["local_variance_grid"]["values"][4] = json!(original + bump);
            let up = compile(&v, "lsv_rough").unwrap().evaluate().unwrap().value;
            v["model"]["local_variance_grid"]["values"][4] = json!(original - bump);
            let down = compile(&v, "lsv_rough").unwrap().evaluate().unwrap().value;
            assert!((variance.node_adjoints[4] - (up - down) / (2.0 * bump)).abs() < 2e-5);
            if hit != (style == "knock_in") {
                assert!(variance.node_adjoints.iter().all(|v| *v == 0.0));
                assert!(r.cash_mean_adjoints.iter().all(|v| *v == 0.0));
                assert!(r.repo_spread_log_df_adjoints.iter().all(|v| *v == 0.0));
                assert!((r.discount_log_df_adjoints[1] - PAYMENT * r.price.value).abs() < 2e-14);
            }
            // The dedicated survival estimator remains a discrete-contract API.
            assert!(plan.evaluate_lsv_hard_barrier_spot_risk().is_err());
        }
    }
}

#[test]
fn live_continuous_monitoring_and_other_hybrid_adapters_still_reject() {
    for kind in KINDS {
        for dates in [
            json!(["2026-09-03", "2027-09-03"]),
            json!(["2026-09-03", "2026-09-04"]),
        ] {
            let mut v = payload(false, false);
            v["product"]["monitoring_dates"] = dates;
            assert!(
                compile(&v, kind)
                    .unwrap_err()
                    .to_string()
                    .contains("continuous Barrier")
            );
        }
        let mut v = payload(false, false);
        v["product"]["monitoring_dates"] = json!(["2027-09-03"]);
        v["product"]
            .as_object_mut()
            .unwrap()
            .remove("historical_hit");
        assert!(
            compile(&v, kind)
                .unwrap_err()
                .to_string()
                .contains("continuous Barrier")
        );
    }
    let mut v = payload(true, false);
    let request =
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    assert!(
        SimulationPlan::compile_hybrid_base(&request, ExecutionPolicy::new(1, Some(16)).unwrap())
            .is_err()
    );
    v["risk"]["payoff_smoothing"] = json!({"type": "compact_c2", "half_width": 2.0});
    v["schema_version"] = json!(3);
    v["risk"]["payoff_smoothing_width_ladder"] = json!([2.0, 1.0]);
    assert!(
        compile(&v, "lsv_rough")
            .unwrap_err()
            .to_string()
            .contains("smoothing width ladder")
    );
}

#[test]
fn resolved_continuous_rqmc_replay_smoothing_and_rebate_scale() {
    for (hit, ended, style) in [(true, false, "knock_out"), (false, true, "knock_in")] {
        let mut v = payload(hit, ended);
        v["product"]["style"] = json!({"type": style});
        v["product"]["notional"] = json!(1e18);
        v["engine"] = json!({"type": "randomized_quasi_monte_carlo",
            "points_per_scramble": 16, "scramble_count": 4, "master_scramble_seed": 91,
            "variance_reduction": {"antithetic": true, "brownian_bridge": true}});
        let plan = compile(&v, "lsv_rough").unwrap();
        let r = plan.evaluate_lsv_spot_risk().unwrap();
        assert!((r.price.value - 7.0 * 0.95_f64.powf(PAYMENT)).abs() < 2e-14);
        assert_eq!(
            (r.price.standard_error, r.delta, r.delta_standard_error),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(r.price.independent_sampling_units, 4);
        assert_eq!(r.price.evaluated_paths, 128);
        let mut parallel = plan.clone();
        parallel.policy = ExecutionPolicy::new(3, Some(16)).unwrap();
        assert_eq!(r, parallel.evaluate_lsv_spot_risk().unwrap());
        // Even an extremely wide smoother cannot change an absorbing history.
        v["risk"]["payoff_smoothing"] = json!({"type": "compact_c2", "half_width": 200.0});
        let smoothed = compile(&v, "lsv_rough").unwrap();
        assert_eq!(price_delta(&plan), price_delta(&smoothed));
        assert_ne!(plan.plan_fingerprint(), smoothed.plan_fingerprint());
    }
}
