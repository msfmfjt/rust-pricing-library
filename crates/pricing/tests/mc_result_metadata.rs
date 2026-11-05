//! Sampling counts, uncertainty and replay information at the public result boundary.
use pricing::market::CurveRegion;
use pricing::mc::ExecutionPolicy;
use pricing::{EstimatorKind, JsonLimits, PricingPlan, parse_request_json, result_to_json};
use serde_json::{Value, json};

fn document(qmc: bool, antithetic: bool, risk: bool, volatility: f64) -> Value {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    let reduction = json!({"antithetic": antithetic, "brownian_bridge": true});
    value["engine"] = if qmc {
        json!({"type": "randomized_quasi_monte_carlo", "points_per_scramble": 16,
            "scramble_count": 4, "master_scramble_seed": 7, "variance_reduction": reduction})
    } else {
        json!({"type": "pseudo_monte_carlo", "independent_sampling_units": 64,
            "master_seed": 7, "variance_reduction": reduction})
    };
    value["model"]["volatility"] = json!(volatility);
    value["risk"]["delta"] = json!(risk);
    value["risk"]["vega"] = json!(risk);
    value["risk"]["aad_tile_capacity"] = json!(13);
    value["risk"]["checkpoint_interval"] = json!(3);
    value
}

fn plan(value: &Value, workers: u32) -> PricingPlan {
    let request =
        parse_request_json(&serde_json::to_vec(value).unwrap(), JsonLimits::DEFAULT).unwrap();
    PricingPlan::compile(&request, ExecutionPolicy::new(workers, Some(17)).unwrap()).unwrap()
}

#[test]
fn sampled_result_keeps_engine_units_variances_and_optional_risks() {
    for qmc in [false, true] {
        for antithetic in [false, true] {
            for risk in [false, true] {
                for volatility in [0.0, 0.2] {
                    let result = plan(&document(qmc, antithetic, risk, volatility), 2)
                        .evaluate()
                        .unwrap();
                    let units = if qmc { 4 } else { 64 };
                    let estimator = if qmc {
                        EstimatorKind::RandomizedQuasiMonteCarlo
                    } else {
                        EstimatorKind::PseudoMonteCarlo
                    };
                    let estimate = result.pricing_result.value;
                    assert_eq!(result.independent_sampling_units, units);
                    assert_eq!(estimate.effective_sampling_units().get(), units);
                    assert_eq!(estimate.estimator(), estimator);
                    assert_eq!(result.evaluated_paths, if antithetic { 128 } else { 64 });
                    assert_eq!(result.diagnostics.master_seed, 7);
                    assert_eq!(result.diagnostics.estimator, estimator);
                    assert_eq!(result.diagnostics.antithetic, antithetic);
                    assert_eq!(result.diagnostics.scramble_count, qmc.then_some(4));
                    assert_eq!(result.diagnostics.direction_checksum.is_some(), qmc);
                    assert_eq!(result.diagnostics.scramble_checksum.is_some(), qmc);
                    assert_eq!(result.diagnostics.worker_threads, 2);
                    assert_eq!(result.diagnostics.reduction_block_size, 17);
                    assert_eq!(result.diagnostics.aad_tile_capacity, 13);
                    assert_eq!(result.diagnostics.checkpoint_interval, 3);
                    assert_eq!(result.early_exercise_diagnostics, None);
                    assert_eq!(
                        result.estimator_variance,
                        result.sampling_variance / units as f64
                    );
                    if volatility == 0.0 {
                        assert_eq!(result.sampling_variance, 0.0);
                        assert_eq!(estimate.standard_error().get(), 0.0);
                    } else {
                        assert!(result.sampling_variance > 0.0);
                        let reported_variance = estimate.standard_error().get().powi(2);
                        assert!(
                            (reported_variance - result.estimator_variance).abs()
                                <= 1e-12 * result.estimator_variance
                        );
                    }
                    assert_eq!(result.pricing_result.risks.delta.is_some(), risk);
                    assert_eq!(result.pricing_result.risks.vega.is_some(), risk);
                    assert_eq!(result.risk_diagnostics.delta_validation.is_some(), risk);
                    assert_eq!(result.risk_diagnostics.vega_validation.is_some(), risk);
                    for risk in [
                        result.pricing_result.risks.delta,
                        result.pricing_result.risks.vega,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        assert_eq!(risk.raw().estimator(), estimator);
                        assert_eq!(risk.raw().effective_sampling_units().get(), units);
                    }
                }
            }
        }
    }
}

#[test]
fn extrapolation_warnings_and_replay_survive_worker_changes() {
    for qmc in [false, true] {
        let mut value = document(qmc, true, true, 0.2);
        value["market"]["discount_curve"]["times"] = json!([0.0, 0.5]);
        value["market"]["dividend_curve"]["times"] = json!([0.0, 0.5]);
        let serial_plan = plan(&value, 1);
        let mut serial = serial_plan.evaluate().unwrap();
        let parallel = plan(&value, 3).evaluate().unwrap();
        let warnings = serial.pricing_result.diagnostics.warnings();
        assert_eq!(
            warnings.iter().map(|w| w.code()).collect::<Vec<_>>(),
            [
                "discount_curve_extrapolation",
                "dividend_curve_extrapolation"
            ]
        );
        assert_eq!(
            serial.diagnostics.discount_region,
            CurveRegion::RightExtrapolated
        );
        assert_eq!(
            serial.diagnostics.dividend_region,
            CurveRegion::RightExtrapolated
        );
        let serialized: Value =
            serde_json::from_str(&result_to_json(&serial.pricing_result).unwrap()).unwrap();
        assert_eq!(
            serialized["replay"]["request_fingerprint"],
            serial_plan.request_fingerprint().to_string()
        );
        serial.diagnostics.worker_threads = 3;
        assert_eq!(serial, parallel);
    }
}
