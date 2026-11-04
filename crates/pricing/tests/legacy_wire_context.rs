//! Legacy wire readers retain input versions without changing their errors or migrations.
use pricing::*;
use serde_json::{Value, json};
use std::error::Error;

type Reader = fn(&[u8], JsonLimits) -> Result<(), WireError>;
type ContextReader = fn(&[u8], JsonLimits) -> Result<(), WireParseError>;

fn readers() -> [(Reader, ContextReader); 4] {
    [
        (
            |s, l| parse_request_json(s, l).map(drop),
            |s, l| parse_request_json_with_context(s, l).map(drop),
        ),
        (
            |s, l| parse_result_json(s, l).map(drop),
            |s, l| parse_result_json_with_context(s, l).map(drop),
        ),
        (
            |s, l| parse_monte_carlo_result_json(s, l).map(drop),
            |s, l| parse_monte_carlo_result_json_with_context(s, l).map(drop),
        ),
        (
            |s, l| parse_specification_json(s, l).map(drop),
            |s, l| parse_specification_json_with_context(s, l).map(drop),
        ),
    ]
}

fn same_error(
    readers: (Reader, ContextReader),
    input: &[u8],
    limits: JsonLimits,
    version: Option<u32>,
) -> WireError {
    let original = readers.0(input, limits).unwrap_err();
    let context = readers.1(input, limits).unwrap_err();
    assert_eq!(context.schema_version(), version);
    assert_eq!(context.wire_error(), &original);
    assert_eq!(context.to_string(), original.to_string());
    assert_eq!(
        context.source().unwrap().downcast_ref::<WireError>(),
        Some(&original)
    );
    assert_eq!(context.into_wire_error(), original);
    original
}

#[test]
fn all_legacy_readers_retain_validated_envelope_versions() {
    for (reader, kind, maximum) in [
        (readers()[0], "pricing_request", 3),
        (readers()[1], "pricing_result", 3),
        (readers()[2], "pricing_result", 3),
        (readers()[3], "pricing_specification", 2),
    ] {
        for version in 1..=maximum {
            let incomplete = json!({"document_kind": kind, "schema_version": version});
            assert!(matches!(
                same_error(
                    reader,
                    &serde_json::to_vec(&incomplete).unwrap(),
                    JsonLimits::DEFAULT,
                    Some(version)
                ),
                WireError::Json(_)
            ));
            let wrong_kind = json!({"document_kind":"wrong", "schema_version":version});
            assert!(matches!(
                same_error(
                    reader,
                    &serde_json::to_vec(&wrong_kind).unwrap(),
                    JsonLimits::DEFAULT,
                    Some(version)
                ),
                WireError::WrongDocumentKind { .. }
            ));
        }
        let unsupported = json!({"document_kind":kind,"schema_version":99});
        assert_eq!(
            same_error(
                reader,
                &serde_json::to_vec(&unsupported).unwrap(),
                JsonLimits::DEFAULT,
                Some(99)
            ),
            WireError::UnsupportedSchemaVersion(99)
        );
    }
}

#[test]
fn no_legacy_reader_trusts_a_rejected_envelope() {
    for reader in readers() {
        for input in [
            br#"{"document_kind":"pricing_request","schema_version":1"#.as_slice(),
            br#"{"document_kind":"pricing_request","schema_version":1,"schema_version":2}"#,
            br#"{"document_kind":"pricing_request","schema_version":1,"x":{"a":0,"a":1}}"#,
            br#"{"document_kind":7,"schema_version":1}"#,
            br#"{"schema_version":1}"#,
            br#"{"document_kind":"pricing_request","schema_version":-1}"#,
            b"\xff",
        ] {
            assert!(matches!(
                same_error(reader, input, JsonLimits::DEFAULT, None),
                WireError::Json(_)
            ));
        }
        assert_eq!(
            same_error(reader, b"\xef\xbb\xbf{}", JsonLimits::DEFAULT, None),
            WireError::Utf8Bom
        );
        let input = br#"{"document_kind":"pricing_request","schema_version":1}"#;
        assert!(matches!(
            same_error(
                reader,
                input,
                JsonLimits {
                    max_input_bytes: input.len() - 1,
                    ..JsonLimits::DEFAULT
                },
                None
            ),
            WireError::ResourceLimit {
                name: "input_bytes",
                ..
            }
        ));
        assert_eq!(
            same_error(
                reader,
                input,
                JsonLimits {
                    max_input_bytes: usize::MAX,
                    ..JsonLimits::DEFAULT
                },
                None
            ),
            WireError::LimitOverrideExceedsHardCap
        );
        let deep = format!(
            r#"{{"document_kind":"pricing_request","schema_version":1,"x":{}0{}}}"#,
            "[".repeat(65),
            "]".repeat(65)
        );
        assert!(matches!(
            same_error(reader, deep.as_bytes(), JsonLimits::DEFAULT, None),
            WireError::ResourceLimit { .. }
        ));
    }
}

#[test]
fn domain_and_feature_failures_keep_context_and_original_pointers() {
    for version in 1..=3 {
        let request = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            format!("../../fixtures/v{version}/pricing_request.golden.json"),
        ))
        .unwrap();
        let mut value: Value = serde_json::from_slice(&request).unwrap();
        value["market"]["spot"] = json!(-1.0);
        assert!(matches!(
            same_error(readers()[0],&serde_json::to_vec(&value).unwrap(),JsonLimits::DEFAULT,Some(version)),
            WireError::DomainAt { pointer, .. } if pointer == "/market/spot"
        ));
        value["product"] = json!({"type":"graph"});
        assert!(matches!(
            same_error(readers()[0],&serde_json::to_vec(&value).unwrap(),JsonLimits::DEFAULT,Some(version)),
            WireError::UnsupportedSchemaFeature { schema_version, .. } if schema_version == version
        ));
        let result = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../fixtures/v{version}/pricing_result.golden.json"
            )),
        )
        .unwrap();
        let mut value: Value = serde_json::from_slice(&result).unwrap();
        value["value"]["standard_error"] = json!(-1.0);
        assert!(matches!(
            same_error(readers()[1],&serde_json::to_vec(&value).unwrap(),JsonLimits::DEFAULT,Some(version)),
            WireError::DomainAt { pointer, .. } if pointer == "/value"
        ));
        assert!(matches!(
            same_error(readers()[2],&result,JsonLimits::DEFAULT,Some(version)),
            WireError::DomainAt { pointer, .. } if pointer == "/monte_carlo"
        ));
    }
    for version in 1..=2 {
        let mut value: Value = serde_json::from_str(include_str!(
            "../../../fixtures/specification-v1/black_scholes.golden.json"
        ))
        .unwrap();
        value["schema_version"] = json!(version);
        value["market"]["spot"] = json!(-1.0);
        assert!(matches!(
            same_error(readers()[3],&serde_json::to_vec(&value).unwrap(),JsonLimits::DEFAULT,Some(version)),
            WireError::DomainAt { pointer, .. } if pointer == "/market/spot"
        ));
    }
}

#[test]
fn successful_legacy_migrations_and_fingerprints_are_unchanged() {
    for version in 1..=3 {
        let request = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            format!("../../fixtures/v{version}/pricing_request.golden.json"),
        ))
        .unwrap();
        let old = parse_request_json(&request, JsonLimits::DEFAULT).unwrap();
        let detailed = parse_request_json_with_context(&request, JsonLimits::DEFAULT).unwrap();
        assert_eq!(
            request_to_json(&old).unwrap(),
            request_to_json(&detailed).unwrap()
        );
        assert_eq!(old.wire_migration(), detailed.wire_migration());
        assert_eq!(
            fingerprint_request(&old).unwrap(),
            fingerprint_request(&detailed).unwrap()
        );
        let result = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../fixtures/v{version}/pricing_result.golden.json"
            )),
        )
        .unwrap();
        let old = parse_result_json(&result, JsonLimits::DEFAULT).unwrap();
        let detailed = parse_result_json_with_context(&result, JsonLimits::DEFAULT).unwrap();
        assert_eq!(
            result_to_json(&old).unwrap(),
            result_to_json(&detailed).unwrap()
        );
    }
    for version in 1..=2 {
        let mut value: Value = serde_json::from_str(include_str!(
            "../../../fixtures/specification-v1/black_scholes.golden.json"
        ))
        .unwrap();
        value["schema_version"] = json!(version);
        let input = serde_json::to_vec(&value).unwrap();
        let old = parse_specification_json(&input, JsonLimits::DEFAULT).unwrap();
        let detailed = parse_specification_json_with_context(&input, JsonLimits::DEFAULT).unwrap();
        assert_eq!(
            specification_to_json(&old).unwrap(),
            specification_to_json(&detailed).unwrap()
        );
        assert_eq!(
            fingerprint_specification(&old).unwrap(),
            fingerprint_specification(&detailed).unwrap()
        );
    }
}
