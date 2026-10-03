//! Independent hard Barrier references: exact lognormal limit, two-step
//! conditional quadrature, and multi-step survival-conditioned rough LSV.
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
    configured_plan([h, 0.0, 0.0], steps, seed, width, points, scrambles)
}

fn configured_plan(
    model: [f64; 3],
    steps: usize,
    seed: u64,
    width: Option<f64>,
    points: u64,
    scrambles: u32,
) -> Plan {
    build_plan(model, steps, seed, width, points, scrambles, None, None)
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    model: [f64; 3],
    steps: usize,
    seed: u64,
    width: Option<f64>,
    points: u64,
    scrambles: u32,
    dates: Option<&[&str]>,
    contract: Option<&Value>,
) -> Plan {
    let [h, eta, kappa] = model;
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
    if let Some(dates) = dates {
        v["product"]["monitoring_dates"] = json!(dates);
    }
    if let Some(contract) = contract {
        for key in ["side", "direction", "style"] {
            v["product"][key] = json!({"type":contract[key]});
        }
        for key in ["strike", "barrier", "notional", "rebate"] {
            if let Some(value) = contract.get(key) {
                v["product"][key] = value.clone();
            }
        }
    }
    v["model"] = json!({"type":"local_volatility", "local_variance_grid":{
        "time_nodes":[0.0, f("fixing_time"), f("expiry_time")],
        "log_forward_moneyness_nodes":[-0.5, 0.0, 0.5], "shape":[3,3],
        "values":vec![f("residual_variance");9], "floor":1e-8, "cap":4.0}});
    if eta != 0.0 {
        v["model"]["local_variance_grid"]["values"] =
            conditional_reference()["target_variances"].clone();
    }
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
            kappa,
            0.6,
            f("dividend_volatility"),
            f("equity_dividend_correlation"),
        )
        .unwrap(),
        RoughBergomi::new(h, eta, -0.4).unwrap(),
        0.15,
        LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
        // Contractual quarter dates may differ by an ulp from repeated
        // half-grid arithmetic. Keep ceil(dt / max_step) at the intended count.
        f("expiry_time") / steps as f64
            * if dates.is_some() {
                1.0 + 8.0 * f64::EPSILON
            } else {
                1.0
            },
        ExecutionPolicy::new(1, Some(64)).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.time_nodes().len(), steps + 1);
    assert_eq!(plan.time_nodes()[steps / 2], f("fixing_time"));
    assert_eq!(plan.time_nodes()[steps], f("expiry_time"));
    if eta == 0.0 {
        assert!(
            plan.lsv_squared_leverage()
                .unwrap()
                .iter()
                .all(|x| (*x - f("residual_variance")).abs() < 2e-14)
        );
    }
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

fn conditional_reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/stochastic-dividends/rough-conditional-barrier-reference.json"
    ))
    .unwrap()
}

fn conditional_plan(
    case: &Value,
    seed: u64,
    width: Option<f64>,
    points: u64,
    scrambles: u32,
) -> Plan {
    assert_eq!(case["equity_linkage"], json!(0.6));
    assert_eq!(case["equity_volatility_correlation"], json!(-0.4));
    assert_eq!(case["dividend_volatility_correlation"], json!(0.15));
    let model = ["hurst", "eta", "kappa"].map(|key| case[key].as_f64().unwrap());
    let plan = configured_plan(model, 2, seed, width, points, scrambles);
    let retained = case["squared_leverage"].as_array().unwrap();
    assert_eq!(plan.lsv_squared_leverage().unwrap().len(), retained.len());
    for (&actual, expected) in plan.lsv_squared_leverage().unwrap().iter().zip(retained) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 2e-13);
    }
    assert_eq!(
        json!(plan.lsv_log_moneyness_nodes().unwrap()),
        case["log_nodes"]
    );
    plan
}

#[test]
fn conditional_barrier_calibration_inputs_and_risk_contract() {
    let reference = conditional_reference();
    assert_eq!(
        reference["market_contract_fixture"],
        json!("rough-barrier-reference.json")
    );
    assert_eq!(
        reference["calibration"],
        json!({"particles":64, "seed":42,
        "bandwidth":0.35, "minimum_effective_sample_size":5.0})
    );
    for case in reference["cases"].as_array().unwrap() {
        let hard = conditional_plan(case, 193, None, 64, 4);
        let price = hard.evaluate().unwrap();
        assert!(hard.evaluate_lsv_spot_risk().is_err());
        assert_eq!(hard.evaluate().unwrap(), price);
        let smooth = conditional_plan(case, 193, Some(0.5), 64, 4);
        assert_eq!(smooth.lsv_squared_leverage(), hard.lsv_squared_leverage());
        assert!(smooth.evaluate_lsv_spot_risk().unwrap().delta > 0.0);
    }
}

#[test]
#[ignore = "release-mode nonzero-eta/kappa conditional hard Barrier reference"]
fn rough_lsv_barrier_matches_conditional_hard_reference() {
    let reference = conditional_reference();
    let tolerance = reference["quadrature_agreement_tolerance"]
        .as_f64()
        .unwrap();
    let mut failures = Vec::new();
    for case in reference["cases"].as_array().unwrap() {
        for seed in [193, 877] {
            for width in [None, Some(1.0), Some(0.5)] {
                let plan = conditional_plan(case, seed, width, 65536, 32);
                let (estimate, risk) = if width.is_some() {
                    let risk = plan.evaluate_lsv_spot_risk().unwrap();
                    (risk.price, Some((risk.delta, risk.delta_standard_error)))
                } else {
                    (plan.evaluate().unwrap(), None)
                };
                assert_eq!(estimate.independent_sampling_units, 32);
                assert_eq!(estimate.evaluated_paths, 4_194_304);
                for (quantity, value, se) in
                    std::iter::once(("price", estimate.value, estimate.standard_error))
                        .chain(risk.map(|(value, se)| ("delta", value, se)))
                {
                    let expected = case[quantity].as_f64().unwrap();
                    let bound = (value - expected).abs() + 4.0 * se + tolerance;
                    assert!(value.is_finite() && se.is_finite() && se > 0.0);
                    println!(
                        "{}",
                        json!({"scope":"two_step_frozen_surface_nonzero_eta_kappa_hard_reference",
                        "hurst":case["hurst"], "eta":case["eta"], "kappa":case["kappa"],
                        "steps":2, "seed":seed, "smoothing_half_width":width,
                        "quantity":quantity, "value":value, "scramble_se":se, "hard_reference":expected,
                        "abs_difference_plus_4se":bound, "quadrature_agreement_tolerance":tolerance,
                        "points_per_scramble":65536, "scrambles":32,"antithetic":true})
                    );
                    if (width.is_none() || width == Some(0.5)) && !(bound < 0.01 && se < 0.002) {
                        failures.push(format!("H={}, seed={seed}, width={width:?}, {quantity}: bound={bound}, SE={se}",case["hurst"]));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn survival_reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/stochastic-dividends/rough-survival-barrier-reference.json"
    ))
    .unwrap()
}

fn survival_plan(case: &Value, seed: u64, width: Option<f64>, points: u64, scrambles: u32) -> Plan {
    let times = case["times"].as_array().unwrap();
    let dates = case["monitoring_dates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|date| date.as_str().unwrap())
        .collect::<Vec<_>>();
    let p = build_plan(
        [case["hurst"].as_f64().unwrap(), 0.6, 0.7],
        times.len() - 1,
        seed,
        width,
        points,
        scrambles,
        Some(&dates),
        None,
    );
    assert_eq!(json!(p.time_nodes()), case["times"]);
    assert_eq!(
        json!(p.lsv_log_moneyness_nodes().unwrap()),
        case["log_nodes"]
    );
    let retained = case["squared_leverage"].as_array().unwrap();
    assert_eq!(p.lsv_squared_leverage().unwrap().len(), retained.len());
    for (&actual, expected) in p.lsv_squared_leverage().unwrap().iter().zip(retained) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 2e-13);
    }
    p
}

#[test]
fn survival_reference_grids_and_contracts_are_retained() {
    let reference = survival_reference();
    for case in reference["cases"].as_array().unwrap() {
        for (field, value) in [
            ("eta", 0.6),
            ("kappa", 0.7),
            ("equity_linkage", 0.6),
            ("equity_volatility_correlation", -0.4),
            ("dividend_volatility_correlation", 0.15),
        ] {
            assert_eq!(case[field], json!(value));
        }
        let p = survival_plan(case, 193, None, 64, 4);
        let price = p.evaluate().unwrap();
        assert!(p.evaluate_lsv_spot_risk().is_err());
        assert_eq!(price, p.evaluate().unwrap());
        let smooth = survival_plan(case, 193, Some(0.5), 64, 4);
        assert_eq!(p.lsv_squared_leverage(), smooth.lsv_squared_leverage());
        assert!(smooth.evaluate_lsv_spot_risk().unwrap().delta > 0.0);
    }
}

#[test]
fn dedicated_hard_barrier_spot_method_is_replayable_and_distinct() {
    let hard = plan(0.1, 4, 193, None, 64, 4);
    let before = hard.evaluate().unwrap();
    let risk = hard.evaluate_lsv_hard_barrier_spot_risk().unwrap();
    assert_eq!(risk, hard.evaluate_lsv_hard_barrier_spot_risk().unwrap());
    assert_eq!(before, hard.evaluate().unwrap());
    assert_ne!(risk.price.plan_fingerprint, before.plan_fingerprint);
    assert_eq!(risk.price.independent_sampling_units, 4);
    assert_eq!(risk.price.evaluated_paths, 512);
    assert_eq!(
        risk.price.uncertainty_scope(),
        "pricing_conditional_on_calibration"
    );
    assert!(risk.delta > 0.0 && risk.delta_standard_error > 0.0);
    assert!(risk.method.contains("survival"));
    assert!(hard.evaluate_lsv_spot_risk().is_err());
    assert!(
        plan(0.1, 4, 193, Some(0.5), 64, 4)
            .evaluate_lsv_hard_barrier_spot_risk()
            .is_err()
    );
}

#[test]
#[ignore = "release-mode production hard Barrier survival Spot risk"]
fn production_hard_barrier_spot_matches_independent_references() {
    let exact = reference();
    let conditional = conditional_reference();
    let survival = survival_reference();
    let mut failures = Vec::new();
    for seed in [193, 877] {
        let mut cases = vec![(
            "exact".to_owned(),
            plan(0.1, 4, seed, None, 16384, 16),
            exact["price"].as_f64().unwrap(),
            exact["delta"].as_f64().unwrap(),
            [0.0, 0.0],
        )];
        for case in conditional["cases"].as_array().unwrap() {
            cases.push((
                format!("two_step_h{}", case["hurst"]),
                conditional_plan(case, seed, None, 16384, 16),
                case["price"].as_f64().unwrap(),
                case["delta"].as_f64().unwrap(),
                [0.0, 0.0],
            ));
        }
        for case in survival["cases"].as_array().unwrap() {
            cases.push((
                case["id"].as_str().unwrap().to_owned(),
                survival_plan(case, seed, None, 16384, 16),
                case["price"].as_f64().unwrap(),
                case["delta"].as_f64().unwrap(),
                [
                    case["price_se"].as_f64().unwrap(),
                    case["delta_se"].as_f64().unwrap(),
                ],
            ));
        }
        for (name, plan, price, delta, reference_se) in cases {
            let risk = plan.evaluate_lsv_hard_barrier_spot_risk().unwrap();
            for (j, (quantity, value, se, expected, budget)) in [
                (
                    "price",
                    risk.price.value,
                    risk.price.standard_error,
                    price,
                    0.03,
                ),
                ("delta", risk.delta, risk.delta_standard_error, delta, 0.015),
            ]
            .into_iter()
            .enumerate()
            {
                let combined = se.hypot(reference_se[j]);
                let bound = (value - expected).abs() + 4.0 * combined;
                println!(
                    "{}",
                    json!({"scope":"production_hard_barrier_survival_spot",
                    "case":name,"seed":seed,"quantity":quantity,"value":value,"scramble_se":se,
                    "reference":expected,"reference_se":reference_se[j],"combined_se":combined,
                    "abs_difference_plus_4se":bound,"limit":budget,
                    "points_per_scramble":16384,"scrambles":16,"antithetic":true,"brownian_bridge":true})
                );
                if !(bound < budget && se < 0.003) {
                    failures.push(format!(
                        "{name} seed={seed} {quantity}: bound={bound}, SE={se}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "release-mode production hard Barrier directions and option sides"]
fn production_hard_barrier_directions_and_sides_match_independent_references() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/stochastic-dividends/rough-barrier-styles-reference.json"
    ))
    .unwrap();
    check_contract_references(&fixture, "production_hard_barrier_directions_and_sides");
}

#[test]
#[ignore = "release-mode production hard Barrier fixed cash rebates"]
fn production_hard_barrier_rebates_match_independent_references() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/stochastic-dividends/rough-barrier-rebates-reference.json"
    ))
    .unwrap();
    check_contract_references(&fixture, "production_hard_barrier_fixed_cash_rebates");
}

fn check_contract_references(fixture: &Value, scope: &str) {
    let source = survival_reference();
    let acceptance = &fixture["acceptance"];
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let base = source["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == case["base_case"])
            .unwrap();
        let dates = base["monitoring_dates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_str().unwrap())
            .collect::<Vec<_>>();
        for seed in [193, 877] {
            let plan = build_plan(
                [base["hurst"].as_f64().unwrap(), 0.6, 0.7],
                8,
                seed,
                None,
                16384,
                16,
                Some(&dates),
                Some(&case["contract"]),
            );
            assert_eq!(json!(plan.time_nodes()), base["times"]);
            for (actual, expected) in plan
                .lsv_squared_leverage()
                .unwrap()
                .iter()
                .zip(base["squared_leverage"].as_array().unwrap())
            {
                assert!((actual - expected.as_f64().unwrap()).abs() < 2e-13);
            }
            let risk = plan.evaluate_lsv_hard_barrier_spot_risk().unwrap();
            for (quantity, value, se) in [
                ("price", risk.price.value, risk.price.standard_error),
                ("delta", risk.delta, risk.delta_standard_error),
            ] {
                let expected = case[quantity].as_f64().unwrap();
                let reference_se = case[format!("{quantity}_se")].as_f64().unwrap();
                let combined = se.hypot(reference_se);
                let bound = (value - expected).abs() + 4.0 * combined;
                let limit = acceptance[format!("{quantity}_bound")].as_f64().unwrap();
                let se_limit = acceptance[format!("valuation_{quantity}_se")]
                    .as_f64()
                    .unwrap();
                let reference_limit = acceptance[format!("reference_{quantity}_se")]
                    .as_f64()
                    .unwrap();
                println!(
                    "{}",
                    json!({"scope":scope,
                    "case":case["id"],"seed":seed,"contract":case["contract"],"quantity":quantity,
                    "value":value,"scramble_se":se,"reference":expected,"reference_se":reference_se,
                    "combined_se":combined,"abs_difference_plus_4se":bound,"limit":limit,
                    "points_per_scramble":16384,"scrambles":16,"antithetic":true,"brownian_bridge":true})
                );
                if !(bound < limit && se < se_limit && reference_se < reference_limit) {
                    failures.push(format!(
                        "{} seed={seed} {quantity}: bound={bound}, SE={se}",
                        case["id"]
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "release-mode multi-step independent survival-conditioned Barrier reference"]
fn rough_lsv_barrier_matches_multistep_survival_reference() {
    let reference = survival_reference();
    let acceptance = &reference["acceptance"];
    let mut failures = Vec::new();
    for case in reference["cases"].as_array().unwrap() {
        for seed in [193, 877] {
            for width in [None, Some(0.5)] {
                let plan = survival_plan(case, seed, width, 65536, 32);
                let (price, risk) = if width.is_some() {
                    let r = plan.evaluate_lsv_spot_risk().unwrap();
                    (r.price, Some((r.delta, r.delta_standard_error)))
                } else {
                    (plan.evaluate().unwrap(), None)
                };
                assert_eq!(price.independent_sampling_units, 32);
                assert_eq!(price.evaluated_paths, 4_194_304);
                for (quantity, value, se) in
                    std::iter::once(("price", price.value, price.standard_error))
                        .chain(risk.map(|(v, se)| ("delta", v, se)))
                {
                    let expected = case[quantity].as_f64().unwrap();
                    let reference_se = case[format!("{quantity}_se")].as_f64().unwrap();
                    let combined_se = se.hypot(reference_se);
                    let bound = (value - expected).abs() + 4.0 * combined_se;
                    let limit = acceptance[format!("{quantity}_bound")].as_f64().unwrap();
                    let reference_limit = acceptance[format!("reference_{quantity}_se")]
                        .as_f64()
                        .unwrap();
                    let valuation_limit = acceptance[format!("valuation_{quantity}_se")]
                        .as_f64()
                        .unwrap();
                    assert!(value.is_finite() && se.is_finite() && se > 0.0 && reference_se > 0.0);
                    println!(
                        "{}",
                        json!({"scope":"independent_multistep_hard_barrier_reference",
                        "case":case["id"],"steps":case["times"].as_array().unwrap().len()-1,
                        "monitoring_dates":case["monitoring_dates"],"seed":seed,"smoothing_half_width":width,
                        "quantity":quantity,"value":value,"scramble_se":se,"hard_reference":expected,
                        "reference_batch_se":reference_se,"combined_se":combined_se,
                        "abs_difference_plus_4se":bound,"limit":limit,
                        "points_per_scramble":65536,"scrambles":32,"antithetic":true})
                    );
                    if !(bound < limit && se < valuation_limit && reference_se < reference_limit) {
                        failures.push(format!("{} seed={seed} width={width:?} {quantity}: bound={bound}, SE={se}, reference SE={reference_se}",case["id"]));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
