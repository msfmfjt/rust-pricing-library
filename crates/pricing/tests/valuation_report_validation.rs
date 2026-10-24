//! Domain-error pointers resolve in both supported report document versions.
use pricing::{
    JsonLimits, WireError, parse_valuation_report_json, parse_valuation_report_json_with_context,
    valuation_report_to_json,
};
use serde_json::{Value, json};
use std::error::Error;

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
    let error = context_error(
        &serde_json::to_vec(value).unwrap(),
        JsonLimits::DEFAULT,
        Some(value["schema_version"].as_u64().unwrap() as u32),
    );
    match error {
        WireError::DomainAt { pointer, .. } => assert_eq!(pointer, expected),
        other => panic!("unexpected error: {other}"),
    }
}

fn context_error(input: &[u8], limits: JsonLimits, version: Option<u32>) -> WireError {
    let detailed = parse_valuation_report_json_with_context(input, limits).unwrap_err();
    assert_eq!(detailed.schema_version(), version);
    let legacy = parse_valuation_report_json(input, limits).unwrap_err();
    assert_eq!(detailed.wire_error(), &legacy);
    assert_eq!(detailed.to_string(), legacy.to_string());
    assert_eq!(
        detailed.source().unwrap().downcast_ref::<WireError>(),
        Some(&legacy)
    );
    assert_eq!(detailed.into_wire_error(), legacy);
    legacy
}

#[test]
fn report_wire_failures_keep_the_validated_envelope_version() {
    for original in documents() {
        let version = original["schema_version"].as_u64().unwrap() as u32;
        let mut report = original.clone();
        report["standard_error"] = json!("not a number");
        assert!(matches!(
            context_error(
                &serde_json::to_vec(&report).unwrap(),
                JsonLimits::DEFAULT,
                Some(version)
            ),
            WireError::Json(_)
        ));
        report = original.clone();
        report["document_kind"] = json!("pricing_task");
        assert_eq!(
            context_error(
                &serde_json::to_vec(&report).unwrap(),
                JsonLimits::DEFAULT,
                Some(version)
            ),
            WireError::WrongDocumentKind {
                expected: "valuation_report",
                actual: "pricing_task".into()
            }
        );
        report = original;
        report["schema_version"] = json!(99);
        assert_eq!(
            context_error(
                &serde_json::to_vec(&report).unwrap(),
                JsonLimits::DEFAULT,
                Some(99)
            ),
            WireError::UnsupportedSchemaVersion(99)
        );
    }
    let mut v1 = documents()[0].clone();
    v1["calibration_diagnostics"] = json!([]);
    assert_pointer(&v1, "/calibration_diagnostics");
}

#[test]
fn report_context_never_trusts_an_envelope_rejected_by_syntax_or_limits() {
    for input in [
        br#"{"document_kind":"valuation_report","schema_version":1"#.as_slice(),
        br#"{"document_kind":"valuation_report","schema_version":1,"schema_version":2}"#,
        br#"{"document_kind":"valuation_report","schema_version":1,"x":{"a":0,"a":1}}"#,
        br#"{"schema_version":1}"#,
        br#"{"document_kind":"valuation_report","schema_version":-1}"#,
        b"\xff",
    ] {
        assert!(matches!(
            context_error(input, JsonLimits::DEFAULT, None),
            WireError::Json(_)
        ));
    }
    assert_eq!(
        context_error(b"\xef\xbb\xbf{}", JsonLimits::DEFAULT, None),
        WireError::Utf8Bom
    );
    let input = serde_json::to_vec(&documents()[0]).unwrap();
    assert_eq!(
        context_error(
            &input,
            JsonLimits {
                max_input_bytes: input.len() - 1,
                ..JsonLimits::DEFAULT
            },
            None
        ),
        WireError::ResourceLimit {
            name: "input_bytes",
            observed: input.len(),
            limit: input.len() - 1
        }
    );
    let deep = format!(
        r#"{{"document_kind":"valuation_report","schema_version":1,"x":{}0{}}}"#,
        "[".repeat(65),
        "]".repeat(65)
    );
    assert!(matches!(
        context_error(deep.as_bytes(), JsonLimits::DEFAULT, None),
        WireError::ResourceLimit {
            name: "nesting_depth",
            ..
        }
    ));
    assert_eq!(
        context_error(
            &input,
            JsonLimits {
                max_input_bytes: usize::MAX,
                ..JsonLimits::DEFAULT
            },
            None
        ),
        WireError::LimitOverrideExceedsHardCap
    );
}

#[test]
fn report_context_reader_preserves_valid_values_and_v1_migration() {
    for mut document in documents() {
        let input = serde_json::to_vec(&document).unwrap();
        let detailed =
            parse_valuation_report_json_with_context(&input, JsonLimits::DEFAULT).unwrap();
        assert_eq!(
            detailed,
            parse_valuation_report_json(&input, JsonLimits::DEFAULT).unwrap()
        );
        document["schema_version"] = json!(2);
        let saved: Value =
            serde_json::from_str(&valuation_report_to_json(&detailed).unwrap()).unwrap();
        assert_eq!(saved, document);
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
