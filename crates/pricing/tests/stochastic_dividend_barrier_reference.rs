//! Independent hard two-date Barrier reference in the exact lognormal limit.
//! The reference Delta includes moving monitoring boundaries. Smoothed public
//! Deltas are compared to it; unsupported hard pathwise risk stays rejected.

use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::RoughBergomi;
use pricing::stochastic_dividends::{BuehlerDividendModel, StochasticDividendPricingPlan as Plan};
use pricing::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/stochastic-dividends/rough-barrier-reference.json"
    ))
    .unwrap()
}

fn plan(h: f64, steps: usize, seed: u64, width: Option<f64>, points: u64, scrambles: u32) -> Plan {
    let r = reference();
    let f = |key: &str| r[key].as_f64().unwrap();
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(f("spot"));
    v["market"]["discount_curve"]["discount_factors"] = json!([1.0, f("annual_discount")]);
    v["market"]["dividend_curve"]["discount_factors"] = json!([1.0, f("annual_carry")]);
    v["market"]["discrete_dividends"] = json!(
        (0..3)
            .map(|i| json!({
                "event_id":i + 1, "ex_time":r["cash_times"][i],
                "quote":{"type":"fixed_cash", "amount":r["cash_means"][i]}
            }))
            .collect::<Vec<_>>()
    );
    v["product"] = json!({"type":"barrier", "underlying_id":1, "currency_id":2,
        "expiry":"2027-09-03", "strike":f("strike"), "barrier":f("barrier"),
        "notional":1.0, "side":{"type":"call"}, "direction":{"type":"up"},
        "style":{"type":"knock_in"}, "monitoring":{"type":"discrete"},
        "monitoring_dates":["2027-03-05", "2027-09-03"], "payment_date":"2027-12-04"});
    v["model"] = json!({"type":"local_volatility", "local_variance_grid":{
        "time_nodes":[0.0, f("fixing_time"), f("expiry_time")],
        "log_forward_moneyness_nodes":[-0.5, 0.0, 0.5], "shape":[3,3],
        "values":vec![f("residual_variance");9], "floor":1e-8, "cap":4.0}});
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
        "points_per_scramble":points, "scramble_count":scrambles, "master_scramble_seed":seed,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    if let Some(width) = width {
        v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2", "half_width":width});
    }
    let request =
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let plan = Plan::compile_rough_bergomi_lsv(
        &request,
        BuehlerDividendModel::new(
            0.0,
            0.6,
            f("dividend_volatility"),
            f("equity_dividend_correlation"),
        )
        .unwrap(),
        RoughBergomi::new(h, 0.0, -0.4).unwrap(),
        0.15,
        LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
        f("expiry_time") / steps as f64,
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.time_nodes().len(), steps + 1);
    assert_eq!(plan.time_nodes()[steps / 2], f("fixing_time"));
    assert_eq!(plan.time_nodes()[steps], f("expiry_time"));
    assert!(
        plan.lsv_squared_leverage()
            .unwrap()
            .iter()
            .all(|x| (*x - f("residual_variance")).abs() < 2e-14)
    );
    plan
}

#[test]
fn hard_barrier_reference_setup_preserves_price_only_contract() {
    for h in [0.01, 0.3, 0.5] {
        for steps in [2, 4] {
            let hard = plan(h, steps, 193, None, 64, 4);
            let before = hard.evaluate().unwrap();
            assert!(hard.evaluate_lsv_spot_risk().is_err());
            assert_eq!(hard.evaluate().unwrap(), before);
            let smooth = plan(h, steps, 193, Some(0.5), 64, 4);
            assert_eq!(hard.lsv_squared_leverage(), smooth.lsv_squared_leverage());
            let risk = smooth.evaluate_lsv_spot_risk().unwrap();
            assert!(risk.price.value > 0.0 && risk.delta > 0.0);
        }
    }
}

#[test]
#[ignore = "release-mode independent hard Barrier price and Delta reference"]
fn rough_lsv_barrier_matches_independent_hard_reference_in_exact_limit() {
    let reference = reference();
    let price = reference["price"].as_f64().unwrap();
    let delta = reference["delta"].as_f64().unwrap();
    let quadrature_tolerance = reference["quadrature_agreement_tolerance"]
        .as_f64()
        .unwrap();
    let mut failures = Vec::new();
    // Both grids simulate the same exact law in this limit; no time bias is
    // absorbed into these price/Delta comparison budgets.
    for (h, steps) in [(0.1, 2), (0.3, 4)] {
        for seed in [193, 877] {
            for width in [None, Some(2.0), Some(1.0), Some(0.5)] {
                let plan = plan(h, steps, seed, width, 65536, 32);
                let (estimate, risk) = if width.is_some() {
                    let risk = plan.evaluate_lsv_spot_risk().unwrap();
                    (
                        risk.price.clone(),
                        Some((risk.delta, risk.delta_standard_error)),
                    )
                } else {
                    (plan.evaluate().unwrap(), None)
                };
                assert_eq!(estimate.independent_sampling_units, 32);
                assert_eq!(estimate.evaluated_paths, 4_194_304);
                for (quantity, value, se, expected) in
                    std::iter::once(("price", estimate.value, estimate.standard_error, price))
                        .chain(risk.map(|(value, se)| ("delta", value, se, delta)))
                {
                    assert!(value.is_finite() && se.is_finite() && se > 0.0);
                    let bound = (value - expected).abs() + 4.0 * se + quadrature_tolerance;
                    println!(
                        "{}",
                        json!({"hurst":h, "steps":steps,"seed":seed,
                        "smoothing_half_width":width, "quantity":quantity, "value":value,
                        "scramble_se":se, "hard_reference":expected, "abs_difference_plus_4se":bound,
                        "quadrature_agreement_tolerance":quadrature_tolerance,
                        "points_per_scramble":65536,"scrambles":32,"antithetic":true,
                        "scope":"eta_zero_kappa_zero_flat_variance_hard_barrier_reference"})
                    );
                    if (width.is_none() || width == Some(0.5)) && !(bound < 0.01 && se < 0.002) {
                        failures.push(format!("H={h}, steps={steps}, seed={seed}, width={width:?}, {quantity}: difference={}, SE={se}, bound={bound}", value - expected));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
