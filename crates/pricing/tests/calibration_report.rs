//! Report migration, numerical compatibility and malformed diagnostic rejection.
use pricing::*;
use serde_json::{Value, json};

fn task(name: &str) -> PricingTask {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/task-v1")
        .join(format!("{name}.golden.json"));
    parse_pricing_task_json(&std::fs::read(path).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn report(name: &str) -> ValuationReport {
    task(name).compile().unwrap().evaluate().unwrap().report
}
fn wire(r: &ValuationReport) -> Value {
    serde_json::from_str(&valuation_report_to_json(r).unwrap()).unwrap()
}
fn parse(v: &Value) -> Result<ValuationReport, WireError> {
    parse_valuation_report_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT)
}

#[test]
fn v1_migration_preserves_unknown_diagnostics_and_frozen_numerical_fields() {
    let v1: Value = serde_json::from_str(include_str!(
        "../../../fixtures/report-v1/multi_asset_joint.golden.json"
    ))
    .unwrap();
    let old = parse(&v1).unwrap();
    assert_eq!(old.calibration_diagnostics(), None);
    let mut migrated = v1.clone();
    migrated["schema_version"] = json!(2);
    assert_eq!(wire(&old), migrated);
    assert_eq!(parse(&migrated).unwrap(), old);
    let current = report("multi_asset_joint");
    let mut new = wire(&current);
    assert!(
        !new.as_object_mut()
            .unwrap()
            .remove("calibration_diagnostics")
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(current.task_fingerprint(), old.task_fingerprint());
    // Compiled plan identity also hashes calibrated floating-point values.
    // Exact numerical replay follows the existing platform-specific contract;
    // source/task identity above remains independent of the replay platform.
    if current.platform() == old.platform() {
        assert_eq!(current.plan_fingerprint(), old.plan_fingerprint());
        assert_eq!(
            new, migrated,
            "diagnostics must not alter existing numerical fields"
        );
    }
}

#[test]
fn native_v1_migration_preserves_all_fields_and_plan_identity_on_every_platform() {
    for name in [
        "standard_risk",
        "multi_asset_bs",
        "multi_asset_joint",
        "stochastic_dividend_bs",
        "stochastic_dividend_hw",
        "stochastic_dividend_lsv",
    ] {
        let current = report(name);
        let mut expected = wire(&current);
        expected
            .as_object_mut()
            .unwrap()
            .remove("calibration_diagnostics")
            .unwrap();
        let mut legacy = expected.clone();
        legacy["schema_version"] = json!(1);
        let migrated = parse(&legacy).unwrap();
        assert_eq!(migrated.calibration_diagnostics(), None, "{name}");
        assert_eq!(
            migrated.task_fingerprint(),
            current.task_fingerprint(),
            "{name}"
        );
        assert_eq!(
            migrated.plan_fingerprint(),
            current.plan_fingerprint(),
            "{name}"
        );
        // No numerical tolerance: retain the entire native report, including
        // every price/risk/covariance value, seed, convention and compiled hash.
        assert_eq!(
            serde_json::to_vec(&wire(&migrated)).unwrap(),
            serde_json::to_vec(&expected).unwrap(),
            "{name}"
        );
        assert_eq!(parse(&expected).unwrap(), migrated, "{name}");
    }
}

#[test]
fn fresh_reports_distinguish_empty_from_unavailable_and_survive_reselection() {
    for name in [
        "standard_risk",
        "multi_asset_bs",
        "stochastic_dividend_bs",
        "stochastic_dividend_hw",
    ] {
        let r = report(name);
        assert_eq!(r.calibration_diagnostics(), Some([].as_slice()));
        assert_eq!(parse(&wire(&r)).unwrap(), r);
    }
    for name in ["multi_asset_joint", "stochastic_dividend_lsv"] {
        let compiled = task(name).compile().unwrap();
        let original = compiled.evaluate().unwrap();
        let price = compiled.with_operation(PricingOperation::Evaluate).unwrap();
        assert_eq!(
            original.report.calibration_diagnostics(),
            price.evaluate().unwrap().report.calibration_diagnostics()
        );
        assert_eq!(parse(&wire(&original.report)).unwrap(), original.report);
    }
}

#[test]
fn diagnostic_reader_rejects_invalid_shapes_identities_units_and_versions() {
    let original = wire(&report("multi_asset_joint"));
    let block = &original["calibration_diagnostics"][0];
    let metric = &block["metrics"][0];
    for (pointer, value) in [
        ("/schema_version", json!(3)),
        ("/calibration_diagnostics", Value::Null),
        ("/calibration_diagnostics", json!([block, block])),
        ("/calibration_diagnostics/0/kind", json!("unknown")),
        ("/calibration_diagnostics/0/underlying_ids", json!([1, 1])),
        ("/calibration_diagnostics/0/seed", json!(-1)),
        (
            "/calibration_diagnostics/0/axes/0/values",
            json!([0.0, 0.0]),
        ),
        ("/calibration_diagnostics/0/axes/0/name", json!("")),
        ("/calibration_diagnostics/0/metrics", json!([])),
        (
            "/calibration_diagnostics/0/metrics",
            json!([metric, metric]),
        ),
        ("/calibration_diagnostics/0/metrics/0/values", json!([1.0])),
        (
            "/calibration_diagnostics/0/metrics/0/unit",
            json!("unscaled"),
        ),
        ("/calibration_diagnostics/0/metrics/0/convention", json!("")),
    ] {
        let mut v = original.clone();
        *v.pointer_mut(pointer).unwrap() = value;
        assert!(parse(&v).is_err(), "accepted {pointer}");
    }
    for (unit, value) in [
        ("count", -1.0),
        ("count", 0.5),
        ("count", 1e20),
        ("indicator", 2.0),
    ] {
        let mut v = original.clone();
        v["calibration_diagnostics"][0]["metrics"][0]["unit"] = json!(unit);
        v["calibration_diagnostics"][0]["metrics"][0]["values"][0] = json!(value);
        assert!(
            matches!(parse(&v), Err(WireError::DomainAt { pointer, .. }) if pointer == "/calibration_diagnostics/0")
        );
    }
    for location in [
        "/calibration_diagnostics/0",
        "/calibration_diagnostics/0/metrics/0",
    ] {
        let mut v = original.clone();
        v.pointer_mut(location).unwrap()["unknown"] = json!(1);
        assert!(parse(&v).is_err());
    }
    let mut v = original;
    v["schema_version"] = json!(1);
    assert!(parse(&v).is_err(), "v1 must continue to reject new fields");
    v["calibration_diagnostics"] = json!([]);
    assert!(
        parse(&v).is_err(),
        "even an empty v2 field is invalid in v1"
    );
}

#[test]
fn metric_names_extend_with_explicit_units_and_conventions() {
    let mut v = wire(&report("multi_asset_joint"));
    let mut extra = v["calibration_diagnostics"][0]["metrics"][0].clone();
    extra["name"] = json!("future_native_metric");
    extra["convention"] = json!("Explicit future diagnostic convention.");
    v["calibration_diagnostics"][0]["metrics"]
        .as_array_mut()
        .unwrap()
        .push(extra);
    assert_eq!(wire(&parse(&v).unwrap()), v);
}
