//! Task errors must identify the actual nested input, including asset indices.
use pricing::{JsonLimits, WireError, parse_pricing_task_json};
use serde_json::{Value, json};

fn check(fixture: &str, input: &str, bad: Value, expected: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/task-v1")
        .join(format!("{fixture}.golden.json"));
    let mut document: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    *document.pointer_mut(input).expect("existing fixture input") = bad;
    let error =
        parse_pricing_task_json(&serde_json::to_vec(&document).unwrap(), JsonLimits::DEFAULT)
            .unwrap_err();
    let WireError::DomainAt { pointer, .. } = error else {
        panic!("{input}: expected a domain error, received {error}");
    };
    assert_eq!(pointer, expected, "{fixture}: {input}");
    assert!(
        document.pointer(&pointer).is_some(),
        "{pointer} must resolve"
    );
    assert!(
        input == pointer || input.starts_with(&format!("{pointer}/")),
        "{pointer} must locate the rejected value or its containing object"
    );
}

#[test]
fn embedded_specification_errors_keep_the_task_source_prefix() {
    for (fixture, prefix) in [
        ("standard_risk", "/source/specification"),
        ("stochastic_dividend_bs", "/source/equity"),
    ] {
        for (input, bad, expected) in [
            ("/numerics/worker_threads", json!(0), "/numerics"),
            ("/market/spot", json!(-1.0), "/market/spot"),
            ("/valuation_date", json!("invalid"), "/valuation_date"),
        ] {
            check(
                fixture,
                &format!("{prefix}{input}"),
                bad,
                &format!("{prefix}{expected}"),
            );
        }
    }
}

#[test]
fn basket_errors_keep_asset_and_calibration_coordinates() {
    for (input, bad, expected) in [
        (
            "/source/models/1/volatility",
            json!(-0.2),
            "/source/models/1/volatility",
        ),
        (
            "/source/markets/1/spot",
            json!(-1.0),
            "/source/markets/1/spot",
        ),
        (
            "/source/markets/1/discount_curve/discount_factors/1",
            json!(-1.0),
            "/source/markets/1/discount_curve",
        ),
        (
            "/source/correlation/entries/0/date",
            json!("invalid"),
            "/source/correlation/entries/0/date",
        ),
        (
            "/source/engine/independent_sampling_units",
            json!(0),
            "/source/engine",
        ),
    ] {
        check("multi_asset_bs", input, bad, expected);
    }
    for (input, bad, expected) in [
        (
            "/source/lsv/0/value/particles/particle_count",
            json!(0),
            "/source/lsv/0/value/particles",
        ),
        (
            "/source/lsv/0/value/factor/vol_of_vol",
            json!(-0.2),
            "/source/lsv/0/value/factor",
        ),
        (
            "/source/hull_white/mean_reversion",
            json!(-0.1),
            "/source/hull_white",
        ),
        (
            "/source/hull_white/lsv_targets/0/value/log_densities",
            json!([]),
            "/source/hull_white/lsv_targets/0/value",
        ),
        (
            "/source/local_correlation/particles/particle_count",
            json!(0),
            "/source/local_correlation/particles",
        ),
        (
            "/source/local_correlation/second_correlation/entries/0/date",
            json!("invalid"),
            "/source/local_correlation/second_correlation/entries/0/date",
        ),
        (
            "/source/local_correlation/hull_white_target/log_densities",
            json!([]),
            "/source/local_correlation/hull_white_target",
        ),
    ] {
        check("multi_asset_joint", input, bad, expected);
    }
}

#[test]
fn composition_and_operation_errors_point_to_their_own_scope() {
    for (fixture, input, bad, expected) in [
        (
            "multi_asset_bs",
            "/source/numerics/maximum_step",
            json!(0.0),
            "/source/numerics/maximum_step",
        ),
        (
            "multi_asset_bs",
            "/source/markets/1/underlying_id",
            json!(77),
            "/source/markets",
        ),
        ("multi_asset_bs", "/source/lsv", json!([]), "/source"),
        (
            "stochastic_dividend_bs",
            "/source/maximum_step",
            json!(0.0),
            "/source/maximum_step",
        ),
        (
            "stochastic_dividend_bs",
            "/source/volatility_correlations",
            json!([0.0]),
            "/source/volatility_correlations",
        ),
        (
            "multi_asset_bs",
            "/operation/gamma_relative_bump",
            json!(0.0),
            "/operation",
        ),
    ] {
        check(fixture, input, bad, expected);
    }
}
