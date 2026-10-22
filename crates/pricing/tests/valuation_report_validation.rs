//! Domain-error pointers resolve in both supported report document versions.
use pricing::{JsonLimits, WireError, parse_valuation_report_json};
use serde_json::{Value, json};

fn documents() -> [Value; 2] {
    [
        include_str!("../../../fixtures/report-v1/multi_asset_joint.golden.json"),
        include_str!("../../../fixtures/report-v2/multi_asset_joint.golden.json"),
    ]
    .map(|text| serde_json::from_str(text).unwrap())
}

fn assert_pointer(value: &Value, expected: &str) {
    assert!(
        value.pointer(expected).is_some(),
        "expected pointer must resolve in the supplied report"
    );
    let error =
        parse_valuation_report_json(&serde_json::to_vec(value).unwrap(), JsonLimits::DEFAULT)
            .expect_err("invalid report");
    match error {
        WireError::DomainAt { pointer, .. } => assert_eq!(pointer, expected),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn report_scalar_and_replay_failures_identify_their_actual_fields() {
    for original in documents() {
        for (pointer, value) in [
            ("/standard_error", json!(-1.0)),
            ("/independent_sampling_units", json!(0)),
            ("/evaluated_paths", json!("0")),
            ("/library_version", json!("")),
            ("/platform", json!("")),
            ("/plan_fingerprint", json!("blake3-256:invalid")),
        ] {
            let mut report = original.clone();
            *report.pointer_mut(pointer).unwrap() = value;
            assert_pointer(&report, pointer);
        }
    }
}

#[test]
fn report_risk_failures_identify_the_affected_block() {
    for original in documents() {
        for (changed, value, expected) in [
            ("/risks/0/values", json!([]), "/risks/0"),
            ("/risks/1/standard_errors", json!([-1.0]), "/risks/1"),
            ("/risks/1/convention", json!(""), "/risks/1"),
        ] {
            let mut report = original.clone();
            *report.pointer_mut(changed).unwrap() = value;
            assert_pointer(&report, expected);
        }
        let mut report = original;
        let risks = report["risks"].as_array_mut().unwrap();
        let duplicate_index = risks.len();
        risks.push(risks[0].clone());
        assert_pointer(&report, &format!("/risks/{duplicate_index}"));
    }
}
