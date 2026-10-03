//! Public rough residual-LSV correlation risk and independent exact-limit checks.

use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pricing::models::RoughBergomi;
use pricing::stochastic_dividends::{BuehlerDividendModel, StochasticDividendPricingPlan as Plan};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn payload(rqmc: bool, antithetic: bool, bridge: bool) -> Value {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    value["model"] = json!({"type": "local_volatility", "local_variance_grid": {
        "time_nodes": [0.0, 0.5, 1.0], "log_forward_moneyness_nodes": [-0.5, 0.0, 0.5],
        "shape": [3, 3], "values": vec![0.04; 9], "floor": 1e-8, "cap": 4.0
    }});
    // Exercise both a cash event on the grid and the post-expiry reserve.
    value["market"]["discrete_dividends"] = json!([
        {"event_id": 1, "ex_time": 0.5, "quote": {"type": "fixed_cash", "amount": 6.0}},
        {"event_id": 2, "ex_time": 1.4, "quote": {"type": "fixed_cash", "amount": 3.0}}
    ]);
    value["engine"] = if rqmc {
        json!({"type": "randomized_quasi_monte_carlo", "points_per_scramble": 32,
            "scramble_count": 4, "master_scramble_seed": 91})
    } else {
        json!({"type": "pseudo_monte_carlo", "independent_sampling_units": 128,
            "master_seed": 91})
    };
    value["engine"]["variance_reduction"] =
        json!({"antithetic": antithetic, "brownian_bridge": bridge});
    value
}

fn request(rqmc: bool, antithetic: bool, bridge: bool) -> PricingRequest {
    let value = payload(rqmc, antithetic, bridge);
    parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn plan(request: &PricingRequest, eta: f64, rho: [f64; 2], workers: u32) -> Plan {
    Plan::compile_rough_bergomi_lsv(
        request,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
        RoughBergomi::new(0.1, eta, rho[0]).unwrap(),
        rho[1],
        LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
        0.25,
        ExecutionPolicy::new(workers, Some(16)).unwrap(),
    )
    .unwrap()
}

#[test]
fn rough_lsv_correlations_match_recompilation_and_worker_replay() {
    let rho = [-0.4, 0.15];
    let bump = 0.02;
    for rqmc in [false, true] {
        for (antithetic, bridge) in [(false, false), (true, true)] {
            let request = request(rqmc, antithetic, bridge);
            let base = plan(&request, 0.6, rho, 1);
            let risk = base
                .evaluate_lsv_rough_bergomi_correlation_risk(bump, bump)
                .unwrap();
            assert_eq!(risk.price, base.evaluate().unwrap());
            assert_eq!(
                risk.price.independent_sampling_units,
                if rqmc { 4 } else { 128 }
            );
            assert_eq!(
                risk.price.evaluated_paths,
                if antithetic { 256 } else { 128 }
            );
            assert_eq!(risk.correlation_bumps.as_ref(), &[bump, bump]);
            assert!(
                risk.standard_errors
                    .iter()
                    .all(|se| se.is_finite() && *se >= 0.0)
            );

            for coordinate in 0..2 {
                let mut down_rho = rho;
                let mut up_rho = rho;
                down_rho[coordinate] -= bump;
                up_rho[coordinate] += bump;
                let down = plan(&request, 0.6, down_rho, 1);
                let up = plan(&request, 0.6, up_rho, 1);
                for scenario in [&down, &up] {
                    if coordinate == 0 {
                        assert_ne!(scenario.lsv_squared_leverage(), base.lsv_squared_leverage());
                    } else {
                        assert_eq!(scenario.lsv_squared_leverage(), base.lsv_squared_leverage());
                    }
                }
                let fd =
                    (up.evaluate().unwrap().value - down.evaluate().unwrap().value) / (2.0 * bump);
                assert!(
                    (risk.derivatives[coordinate] - fd).abs() <= 2e-10,
                    "rqmc={rqmc}, antithetic={antithetic}, coordinate={coordinate}: {} != {fd}",
                    risk.derivatives[coordinate]
                );
            }
            let replay = plan(&request, 0.6, rho, 3)
                .evaluate_lsv_rough_bergomi_correlation_risk(bump, bump)
                .unwrap();
            assert_eq!(risk.derivatives, replay.derivatives);
            assert_eq!(risk.standard_errors, replay.standard_errors);
            assert_eq!(risk.price.value, replay.price.value);
            assert_eq!(risk.price.standard_error, replay.price.standard_error);
        }
    }
}

#[test]
fn rough_lsv_correlation_risk_vanishes_without_vol_of_vol() {
    // With eta=0, neither volatility-driver correlation enters equity/dividend
    // dynamics or leverage calibration. This is independent of a bump oracle.
    for rqmc in [false, true] {
        let risk = plan(&request(rqmc, true, true), 0.0, [-0.4, 0.15], 1)
            .evaluate_lsv_rough_bergomi_correlation_risk(0.02, 0.02)
            .unwrap();
        for value in risk.derivatives.iter().chain(risk.standard_errors.iter()) {
            assert!(value.abs() <= 1e-11, "zero-eta correlation risk: {value}");
        }
    }
}

#[test]
fn rough_lsv_correlation_bumps_require_scalar_and_joint_domains() {
    let base = plan(&request(false, true, true), 0.6, [-0.4, 0.15], 1);
    for bad in [0.0, -0.02, f64::NAN, f64::INFINITY] {
        assert!(
            base.evaluate_lsv_rough_bergomi_correlation_risk(bad, 0.02)
                .is_err()
        );
        assert!(
            base.evaluate_lsv_rough_bergomi_correlation_risk(0.02, bad)
                .is_err()
        );
    }
    assert!(
        base.evaluate_lsv_rough_bergomi_correlation_risk(0.7, 0.02)
            .is_err()
    );
    // Scalar correlations stay in [-1, 1], but the joint matrix is not PSD.
    assert!(
        base.evaluate_lsv_rough_bergomi_correlation_risk(0.02, 0.84)
            .is_err()
    );
}

#[test]
fn rough_lsv_zero_eta_price_and_delta_match_independent_conditional_black() {
    // eta=0 and a flat target imply constant squared leverage, independent of
    // calibration particles. kappa=0 then makes terminal stock a sum of two
    // correlated lognormals. These references integrate out the dividend
    // normal with conditional Black, not a production path or pricing helper.
    // The Python wheel test recomputes both references at orders 96 and 128.
    const PRICE: f64 = 7.653276188575835;
    const DELTA: f64 = 0.5922534629738484;
    let mut value = payload(true, true, true);
    value["market"]["discrete_dividends"] = json!([
        {"event_id": 1, "ex_time": 1.4, "quote": {"type": "fixed_cash", "amount": 25.0}}
    ]);
    value["engine"]["points_per_scramble"] = json!(2048);
    value["engine"]["scramble_count"] = json!(8);
    value["engine"]["master_scramble_seed"] = json!(612);
    let request =
        parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).unwrap();
    for (hurst, particles, seed) in [(0.01, 32, 42), (0.1, 64, 1973), (0.5, 128, 617)] {
        for step in [0.5, 0.25] {
            let plan = Plan::compile_rough_bergomi_lsv(
                &request,
                BuehlerDividendModel::new(0.0, 0.6, 0.45, -0.35).unwrap(),
                RoughBergomi::new(hurst, 0.0, -0.4).unwrap(),
                0.15,
                LsvParticleConfig::new(particles, seed, 0.35, 5.0, false).unwrap(),
                step,
                ExecutionPolicy::new(1, Some(64)).unwrap(),
            )
            .unwrap();
            assert!(
                plan.lsv_squared_leverage()
                    .unwrap()
                    .iter()
                    .all(|x| (x - 0.04).abs() < 2e-14)
            );
            assert_eq!(*plan.time_nodes().last().unwrap(), 1.0);
            assert_eq!(plan.time_nodes().len(), (1.0 / step) as usize + 1);
            let result = plan.evaluate_lsv_spot_risk().unwrap();
            assert_eq!(result.price, plan.evaluate().unwrap());
            assert_eq!(result.price.independent_sampling_units, 8);
            assert_eq!(result.price.evaluated_paths, 32768);
            assert!(result.price.standard_error < 0.02);
            assert!(result.delta_standard_error < 0.005);
            for (label, actual, expected, se, budget) in [
                (
                    "price",
                    result.price.value,
                    PRICE,
                    result.price.standard_error,
                    0.002,
                ),
                (
                    "delta",
                    result.delta,
                    DELTA,
                    result.delta_standard_error,
                    0.00002,
                ),
            ] {
                println!(
                    "eta=0, H={hurst}, step={step}, {label}: {actual:.12}, reference={expected:.12}, SE={se:.6e}"
                );
                assert!(
                    (actual - expected).abs() <= 6.0 * se + budget,
                    "H={hurst}, step={step}, {label}: {actual} != {expected}, SE={se}"
                );
            }
        }
    }
}
