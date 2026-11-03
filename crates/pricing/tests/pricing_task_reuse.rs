//! Reusing compilation must preserve complete native results and replay identity.
use pricing::*;
use serde_json::{Value, json};

const AAD: PricingOperation = PricingOperation::Aad {
    gamma_relative_bump: None,
};

fn fixture(name: &str) -> PricingTask {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/task-v1")
        .join(format!("{name}.golden.json"));
    parse_pricing_task_json(&std::fs::read(path).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn from_value(value: &Value) -> PricingTask {
    parse_pricing_task_json(&serde_json::to_vec(value).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn assert_reuse(task: PricingTask) {
    let operation = task.operation();
    let price_task = task.with_operation(PricingOperation::Evaluate).unwrap();
    let price = price_task.compile().unwrap();
    let price_before = price.evaluate().unwrap();
    assert_eq!(price.capability().operation, PricingOperation::Evaluate);
    assert_eq!(
        price.capability_for(operation).status,
        CapabilityStatus::Supported
    );

    let reused = price.with_operation(operation).unwrap();
    assert_eq!(reused.operation(), operation);
    assert_eq!(reused.task_fingerprint(), task.fingerprint().unwrap());
    assert_eq!(reused.plan_fingerprint(), price.plan_fingerprint());
    assert_eq!(
        pricing_task_to_json(reused.task()).unwrap(),
        pricing_task_to_json(&task).unwrap()
    );
    let actual = reused.evaluate().unwrap();
    let independent = task.compile().unwrap();
    assert_eq!(actual, independent.evaluate().unwrap());
    assert_eq!(actual.report.task_fingerprint(), reused.task_fingerprint());
    assert_eq!(actual.report.plan_fingerprint(), reused.plan_fingerprint());
    assert_eq!(price.evaluate().unwrap(), price_before);
    let restored = reused.with_operation(PricingOperation::Evaluate).unwrap();
    assert_eq!(restored.task_fingerprint(), price.task_fingerprint());
    assert_eq!(restored.evaluate().unwrap(), price_before);
    // Dropping both the source handle and the original compiled owner is safe.
    drop(task);
    drop(price);
    assert_eq!(reused.evaluate().unwrap(), actual);
}

#[test]
fn all_source_families_preserve_complete_native_results() {
    for name in [
        "standard_risk",
        "multi_asset_bs",
        "multi_asset_joint",
        "stochastic_dividend_bs",
        "stochastic_dividend_hw",
        "stochastic_dividend_lsv",
    ] {
        assert_reuse(fixture(name));
    }
    for (text, operation) in [
        (
            include_str!("../../../fixtures/specification-v1/bergomi_hull_white.golden.json"),
            AAD,
        ),
        (
            include_str!("../../../fixtures/specification-v1/bergomi_lsv.golden.json"),
            PricingOperation::LocalVarianceRisk,
        ),
        (
            include_str!("../../../fixtures/specification-v1/rough_heston.golden.json"),
            PricingOperation::Evaluate,
        ),
    ] {
        let source = parse_specification_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_reuse(PricingTask::new(source, operation).unwrap());
    }
}

#[test]
fn changing_gamma_changes_the_task_but_not_the_compiled_plan() {
    let base = fixture("multi_asset_bs").compile().unwrap();
    for bump in [None, Some(0.005), Some(0.02)] {
        let operation = PricingOperation::Aad {
            gamma_relative_bump: bump,
        };
        let reused = base.with_operation(operation).unwrap();
        assert_ne!(reused.task_fingerprint(), base.task_fingerprint());
        assert_eq!(reused.plan_fingerprint(), base.plan_fingerprint());
        let fresh = fixture("multi_asset_bs")
            .with_operation(operation)
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(reused.evaluate().unwrap(), fresh.evaluate().unwrap());
    }
}

#[test]
fn failed_selection_preserves_native_preflight_and_the_original_plan() {
    let mut value: Value =
        serde_json::from_str(&pricing_task_to_json(&fixture("multi_asset_bs")).unwrap()).unwrap();
    value["operation"] = json!({"type":"evaluate"});
    value["source"]["markets"][0]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":90.0}}
    ]);
    let task = from_value(&value);
    let operation = PricingOperation::Aad {
        gamma_relative_bump: Some(0.2),
    };
    assert_eq!(
        task.source().capability(operation).status,
        CapabilityStatus::Conditional
    );
    let compiled = task.compile().unwrap();
    let before = compiled.evaluate().unwrap();
    let capability = compiled.capability_for(operation);
    assert_eq!(capability.status, CapabilityStatus::Unsupported);
    assert_eq!(
        capability.reason,
        Some(CapabilityReason::NativePreflightRequired)
    );
    let failure = compiled.with_operation(operation).unwrap_err();
    let fresh_failure = task
        .with_operation(operation)
        .unwrap()
        .compile()
        .unwrap_err();
    assert_eq!(failure.to_string(), fresh_failure.to_string());
    assert_eq!(compiled.evaluate().unwrap(), before);
    assert_eq!(
        compiled.with_operation(AAD).unwrap().capability().status,
        CapabilityStatus::Supported
    );
}

#[test]
fn unrepresentable_gamma_bumps_fail_native_preflight_before_pricing() {
    for name in ["multi_asset_bs", "multi_asset_joint"] {
        let task = fixture(name)
            .with_operation(PricingOperation::Evaluate)
            .unwrap();
        let compiled = task.compile().unwrap();
        let before = compiled.evaluate().unwrap();
        for bump in [f64::from_bits(1), f64::MIN_POSITIVE, 1e-18] {
            let operation = PricingOperation::Aad {
                gamma_relative_bump: Some(bump),
            };
            assert_eq!(
                task.source().capability(operation).status,
                CapabilityStatus::Conditional
            );
            let capability = compiled.capability_for(operation);
            assert_eq!(
                capability.status,
                CapabilityStatus::Unsupported,
                "{name}: bump={bump}"
            );
            assert_eq!(
                capability.reason,
                Some(CapabilityReason::NativePreflightRequired)
            );
            assert!(
                capability
                    .detail
                    .contains("unrepresentable Gamma Spot bump")
            );
            let reused = compiled.with_operation(operation).unwrap_err();
            let fresh = task
                .with_operation(operation)
                .unwrap()
                .compile()
                .unwrap_err();
            assert_eq!(reused.to_string(), fresh.to_string());
        }
        assert_eq!(compiled.evaluate().unwrap(), before);
        assert_eq!(
            compiled.capability_for(AAD).status,
            CapabilityStatus::Supported
        );
    }
}

#[test]
fn source_prerequisites_cannot_be_upgraded_by_reusing_a_plan() {
    let mut value: Value =
        serde_json::from_str(&pricing_task_to_json(&fixture("stochastic_dividend_lsv")).unwrap())
            .unwrap();
    value["operation"] = json!({"type":"evaluate"});
    value["source"]["equity"]["model"]["equity"]["particles"]["retain_reverse_trace"] =
        json!(false);
    let task = from_value(&value);
    let compiled = task.compile().unwrap();
    let operation = PricingOperation::LocalVarianceRisk;
    assert_eq!(
        compiled.capability_for(operation).reason,
        Some(CapabilityReason::ReverseTraceRequired)
    );
    assert!(compiled.with_operation(operation).is_err());
    assert!(task.with_operation(operation).is_err());
    assert!(compiled.evaluate().is_ok());
    let basket = fixture("multi_asset_bs").compile().unwrap();
    for bump in [0.0, 1.0, -0.01, f64::NAN, f64::INFINITY] {
        let operation = PricingOperation::Aad {
            gamma_relative_bump: Some(bump),
        };
        assert_eq!(
            basket.capability_for(operation).reason,
            Some(CapabilityReason::InvalidGammaBump)
        );
        assert!(basket.with_operation(operation).is_err());
    }
}

#[test]
fn cloned_plans_have_independent_evaluation_workspaces() {
    let risk = fixture("multi_asset_joint").compile().unwrap();
    let price = risk.with_operation(PricingOperation::Evaluate).unwrap();
    let expected_risk = risk.evaluate().unwrap();
    let expected_price = price.evaluate().unwrap();
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let plan = if i % 2 == 0 {
                risk.clone()
            } else {
                price.clone()
            };
            std::thread::spawn(move || plan.evaluate().unwrap())
        })
        .collect();
    drop(risk);
    drop(price);
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(
            handle.join().unwrap(),
            if i % 2 == 0 {
                expected_risk.clone()
            } else {
                expected_price.clone()
            }
        );
    }
}
