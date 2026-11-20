//! Acceptance for the `Measures` operation on standard single-asset sources
//! (ADR 0043, phase 1): a measure list is the same calculation as a request
//! that had set the corresponding Greeks itself, bit for bit.
use pricing::mc::ExecutionPolicy;
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};
use pricing::*;
use serde_json::{Value, json};

fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(1, Some(128)).unwrap()
}
fn black_scholes() -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"]["independent_sampling_units"] = json!(256);
    v["risk"] =
        json!({"delta":false,"vega":false,"smile_dynamics":{"type":"sticky_log_moneyness"}});
    v
}
fn time(date: &str) -> f64 {
    pricing::core::DayCountConvention::Act365F.year_fraction(
        "2026-09-04".parse::<pricing::core::Date>().unwrap(),
        date.parse().unwrap(),
    )
}
fn bass() -> Value {
    let mut v = black_scholes();
    v["model"] = json!({"type":"bass_local_volatility","parameters":{
        "maturity_nodes":[time("2027-03-04"),1.0],"log_forward_moneyness_nodes":[-2.0,0.0,2.0],"implied_volatilities":vec![0.2;6],
        "projection_nodes":(0..=800).map(|i|-2.0+4.0*i as f64/800.0).collect::<Vec<_>>(),
        "grid_points":401,"grid_width":8.0,"max_iterations":2000,"cdf_tolerance":1e-7,"tail_tolerance":1e-7,
        "projection_tail_tolerance":1e-7,"projection_mean_tolerance":1e-4,"iv_bump":1e-4}});
    v
}
fn rqmc(mut v: Value) -> Value {
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":8,"master_scramble_seed":41,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    v
}
fn request(v: &Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn source(v: &Value) -> PricingSpecification {
    PricingSpecification::from_request(request(v), policy())
}
fn bump() -> SpotBump {
    SpotBump::relative(0.002).unwrap()
}
fn grid() -> VegaKtConfig {
    VegaKtConfig::new(
        vec!["2027-03-04".parse().unwrap(), "2027-09-04".parse().unwrap()],
        vec![-2.0, 0.0, 2.0],
        1e-5,
        true,
    )
    .unwrap()
}
fn greeks() -> Vec<MeasureRequest> {
    vec![
        MeasureRequest::native(Measure::SpotDelta),
        MeasureRequest::spot_gamma(bump()),
        MeasureRequest::native(Measure::ParallelMarketIvVega),
    ]
}
fn measures(requests: Vec<MeasureRequest>) -> PricingOperation {
    PricingOperation::Measures(MeasureSet::new(requests).unwrap())
}
/// The same Greeks selected the old way, on a request built in memory.
fn flagged(v: &Value, vega_kt: Option<VegaKtConfig>) -> PricingRequest {
    request(v)
        .with_risk(
            RiskRequest::new(
                true,
                Some(GammaConfig::new(bump())),
                true,
                vega_kt,
                SmileDynamics::StickyLogMoneyness,
                None,
                None,
            )
            .unwrap(),
        )
        .unwrap()
}
fn standard(result: &PricingTaskResult) -> &MonteCarloPrice {
    let PricingTaskOutput::SingleAsset(r) = &result.native_result else {
        panic!("single-asset result")
    };
    let SpecificationOutput::Standard(price) = &r.output else {
        panic!("standard result")
    };
    price
}

#[test]
fn measures_equal_request_flags_bitwise_under_mc_and_rqmc() {
    for (label, v, vega_kt) in [
        ("bs mc", black_scholes(), None),
        ("bs rqmc", rqmc(black_scholes()), None),
        ("bass mc", bass(), Some(grid())),
        ("bass rqmc", rqmc(bass()), Some(grid())),
    ] {
        let mut requests = greeks();
        if let Some(grid) = &vega_kt {
            requests.push(MeasureRequest::native(Measure::VegaKt(grid.clone())));
        }
        let task = PricingTask::new(source(&v), measures(requests)).unwrap();
        let compiled = task.compile().unwrap();
        assert_eq!(compiled.capability().status, CapabilityStatus::Supported);
        let result = compiled.evaluate().unwrap();

        let reference = flagged(&v, vega_kt.clone());
        let native = PricingPlan::compile(&reference, policy()).unwrap();
        let expected = native.evaluate().unwrap();
        // Every field: price, Greeks, VegaKT, diagnostics and replay metadata.
        assert_eq!(standard(&result), &expected, "{label}");
        assert_eq!(
            compiled.plan_fingerprint(),
            native.plan_fingerprint().to_string(),
            "{label}"
        );
        // And equal to the frozen nested-request path through a task.
        let legacy = PricingTask::new(
            PricingSpecification::from_request(reference, policy()),
            PricingOperation::Evaluate,
        )
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap();
        assert_eq!(standard(&legacy), &expected, "{label}");

        let report = &result.report;
        let risks = &expected.pricing_result.risks;
        for (kind, estimate) in [
            (MeasureKind::SpotDelta, risks.delta.unwrap()),
            (MeasureKind::SpotGamma, risks.gamma.unwrap()),
            (MeasureKind::ParallelMarketIvVega, risks.vega.unwrap()),
        ] {
            let row = report.risk(kind).unwrap();
            assert_eq!(row.values, [estimate.raw().value().get()], "{label}");
            assert_eq!(
                row.standard_errors.as_deref(),
                Some(&[estimate.raw().standard_error().get()][..]),
                "{label}"
            );
            assert_eq!(
                report
                    .risks()
                    .iter()
                    .filter(|r| r.measure == Some(kind))
                    .count(),
                2
            );
        }
        assert_eq!(
            report.risk(MeasureKind::SpotGamma).unwrap().method_detail,
            Some(MeasureMethod::NativeSpotBump { bump: bump() })
        );
        assert_eq!(report.vega_kt(), risks.vega_kt.as_ref(), "{label}");
        assert_eq!(report.vega_kt().is_some(), vega_kt.is_some());
        // The typed rows carry the same numbers as the legacy named rows.
        for (typed, named) in report.risks().iter().zip(legacy.report.risks()) {
            assert_eq!(named.measure, None);
            assert_eq!(
                (
                    &typed.name,
                    &typed.values,
                    &typed.standard_errors,
                    typed.unit
                ),
                (
                    &named.name,
                    &named.values,
                    &named.standard_errors,
                    named.unit
                )
            );
        }
    }
}

#[test]
fn measures_task_and_report_round_trip_as_version_three() {
    let v = black_scholes();
    let task = PricingTask::new(source(&v), measures(greeks())).unwrap();
    let json = pricing_task_to_json(&task).unwrap();
    let document: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(document["schema_version"], 3);
    assert_eq!(document["operation"]["type"], "measures");
    let restored = parse_pricing_task_json(json.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(restored.operation(), task.operation());
    assert_eq!(json, pricing_task_to_json(&restored).unwrap());
    assert_eq!(task.fingerprint().unwrap(), restored.fingerprint().unwrap());

    // Order is part of the identity; the numbers do not depend on it.
    let mut reversed = greeks();
    reversed.reverse();
    let other = PricingTask::new(source(&v), measures(reversed)).unwrap();
    assert_ne!(task.fingerprint().unwrap(), other.fingerprint().unwrap());
    let result = task.compile().unwrap().evaluate().unwrap();
    let swapped = other.compile().unwrap().evaluate().unwrap();
    assert_eq!(standard(&result), standard(&swapped));

    let saved = valuation_report_to_json(&result.report).unwrap();
    let report: Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(report["schema_version"], 3);
    assert_eq!(report["risks"][0]["measure"], "spot_delta");
    assert_eq!(
        report["risks"][2]["method_detail"]["type"],
        "native_spot_bump"
    );
    assert_eq!(
        parse_valuation_report_json(saved.as_bytes(), JsonLimits::DEFAULT).unwrap(),
        result.report
    );

    // The measures operation and typed rows are refused under older versions.
    let mut old = document.clone();
    old["schema_version"] = json!(2);
    assert!(matches!(
        parse_pricing_task_json(&serde_json::to_vec(&old).unwrap(), JsonLimits::DEFAULT),
        Err(WireError::UnsupportedSchemaFeature {
            schema_version: 2,
            ..
        })
    ));
    let mut old = report.clone();
    old["schema_version"] = json!(2);
    assert!(
        parse_valuation_report_json(&serde_json::to_vec(&old).unwrap(), JsonLimits::DEFAULT)
            .is_err()
    );
    let mut duplicate = document;
    let first = duplicate["operation"]["measures"][0].clone();
    duplicate["operation"]["measures"]
        .as_array_mut()
        .unwrap()
        .push(first);
    assert!(
        parse_pricing_task_json(
            &serde_json::to_vec(&duplicate).unwrap(),
            JsonLimits::DEFAULT
        )
        .is_err()
    );
}

#[test]
fn legacy_operations_keep_their_document_versions() {
    let v = black_scholes();
    let task = PricingTask::new(source(&v), PricingOperation::Evaluate).unwrap();
    let json: Value = serde_json::from_str(&pricing_task_to_json(&task).unwrap()).unwrap();
    assert_eq!(json["schema_version"], 1);
    let flagged = PricingTask::new(
        PricingSpecification::from_request(flagged(&v, None), policy()),
        PricingOperation::Evaluate,
    )
    .unwrap();
    let report = flagged.compile().unwrap().evaluate().unwrap().report;
    assert!(!report.risks().is_empty());
    let saved: Value = serde_json::from_str(&valuation_report_to_json(&report).unwrap()).unwrap();
    assert_eq!(saved["schema_version"], 2);
    assert!(saved["risks"][0].get("measure").is_none());
}

#[test]
fn risk_cannot_be_selected_in_two_places() {
    let v = black_scholes();
    let nested = PricingSpecification::from_request(flagged(&v, None), policy());
    let error = PricingTask::new(
        nested.clone(),
        measures(vec![MeasureRequest::native(Measure::SpotDelta)]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SpecificationValidationError::UnsupportedOperation {
            reason: Some(CapabilityReason::AmbiguousRiskSelection),
            ..
        }
    ));
    // An empty list is still a measures operation, so it is refused as well.
    assert!(PricingTask::new(nested, measures(vec![])).is_err());
}

#[test]
fn unsupported_requests_are_refused_with_a_reason() {
    let v = black_scholes();
    let s: PricingSource = source(&v).into();
    let reason = |request: MeasureRequest| {
        let capability = s.measure_capabilities(&[request]).remove(0);
        assert_eq!(capability.status, CapabilityStatus::Unsupported);
        capability.reason.unwrap()
    };
    assert_eq!(
        reason(MeasureRequest::native(Measure::SpotGamma)),
        CapabilityReason::NoNativeEstimator
    );
    assert_eq!(
        reason(MeasureRequest::new(
            Measure::SpotDelta,
            MeasureMethod::NativeSpotBump { bump: bump() }
        )),
        CapabilityReason::NoNativeEstimator
    );
    // Bucketed Vega needs a volatility surface.
    assert_eq!(
        reason(MeasureRequest::native(Measure::VegaKt(grid()))),
        CapabilityReason::ModelCombination
    );
    assert!(
        MeasureSet::new(vec![
            MeasureRequest::native(Measure::SpotDelta),
            MeasureRequest::native(Measure::SpotDelta)
        ])
        .is_err()
    );
    assert!(
        PricingTask::new(
            source(&v),
            measures(vec![MeasureRequest::native(Measure::SpotGamma)])
        )
        .is_err()
    );
}

#[test]
fn a_standard_plan_serves_only_the_selection_it_was_compiled_with() {
    let v = black_scholes();
    let operation = measures(greeks());
    let compiled = PricingTask::new(source(&v), operation.clone())
        .unwrap()
        .compile()
        .unwrap();
    assert!(compiled.with_operation(operation).is_ok());
    for other in [
        PricingOperation::Evaluate,
        measures(vec![MeasureRequest::native(Measure::SpotDelta)]),
        measures(vec![]),
    ] {
        assert_eq!(
            compiled.capability_for(other.clone()).status,
            CapabilityStatus::Unsupported
        );
        assert!(compiled.with_operation(other).is_err());
    }
    let capabilities = compiled.measure_capabilities(&[
        MeasureRequest::native(Measure::Price),
        MeasureRequest::native(Measure::SpotDelta),
        MeasureRequest::spot_gamma(SpotBump::relative(0.5).unwrap()),
    ]);
    assert_eq!(
        capabilities.iter().map(|c| c.status).collect::<Vec<_>>(),
        [
            CapabilityStatus::Supported,
            CapabilityStatus::Supported,
            CapabilityStatus::Unsupported
        ]
    );

    // Price-only measures and evaluate are the same plan and the same number.
    let price = PricingTask::new(source(&v), PricingOperation::Evaluate)
        .unwrap()
        .compile()
        .unwrap();
    let listed = price
        .with_operation(measures(vec![MeasureRequest::native(Measure::Price)]))
        .unwrap();
    assert_eq!(price.plan_fingerprint(), listed.plan_fingerprint());
    assert_eq!(
        standard(&price.evaluate().unwrap()),
        standard(&listed.evaluate().unwrap())
    );
    assert!(listed.evaluate().unwrap().report.risks().is_empty());
}

#[test]
fn every_source_family_prices_through_measures() {
    for name in [
        "multi_asset_bs.golden.json",
        "stochastic_dividend_bs.golden.json",
        "stochastic_dividend_hw.golden.json",
    ] {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/task-v1")
                .join(name),
        )
        .unwrap();
        let task = parse_pricing_task_json(&bytes, JsonLimits::DEFAULT).unwrap();
        let price = task
            .with_operation(PricingOperation::Evaluate)
            .unwrap()
            .compile()
            .unwrap();
        let listed = price
            .with_operation(measures(vec![MeasureRequest::native(Measure::Price)]))
            .unwrap();
        let (a, b) = (price.evaluate().unwrap(), listed.evaluate().unwrap());
        assert_eq!(a.native_result, b.native_result, "{name}");
        assert_eq!(a.report.value().to_bits(), b.report.value().to_bits());
        // Stochastic dividends (ADR 0043 step D1) and baskets (step E1)
        // have a native Delta.
        assert!(
            task.with_operation(measures(vec![MeasureRequest::native(Measure::SpotDelta)]))
                .is_ok(),
            "{name}"
        );
    }
}

#[test]
fn records_flatten_the_price_and_every_risk_coordinate() {
    let v = black_scholes();
    let report = PricingTask::new(source(&v), measures(greeks()))
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap()
        .report;
    let records = report.to_records();
    assert_eq!(records.len(), 1 + report.risks().len());
    assert_eq!(records[0].measure, "price");
    assert_eq!(records[0].value, report.value());
    assert_eq!(records[0].standard_error, Some(report.standard_error()));
    assert_eq!(records[1].measure, "spot_delta");
    assert_eq!(records[1].name, "delta.raw");
    assert_eq!(records[1].unit, "currency_per_spot");
    assert_eq!(records[1].value, report.risks()[0].values[0]);

    // A gridded row from a legacy operation flattens in row-major order.
    let bytes = include_bytes!("../../../fixtures/task-v1/stochastic_dividend_lsv.golden.json");
    let task = parse_pricing_task_json(bytes, JsonLimits::DEFAULT).unwrap();
    let report = task.compile().unwrap().evaluate().unwrap().report;
    let row = &report.risks()[0];
    assert_eq!(row.axes.len(), 2);
    let flat: Vec<_> = report
        .to_records()
        .into_iter()
        .filter(|r| r.name == row.name)
        .collect();
    assert_eq!(flat.len(), row.values.len());
    let width = row.axes[1].values.len();
    for (i, record) in flat.iter().enumerate() {
        assert_eq!(record.measure, row.name);
        assert_eq!(record.value, row.values[i]);
        assert_eq!(record.coordinate[0].1, row.axes[0].values[i / width]);
        assert_eq!(record.coordinate[1].1, row.axes[1].values[i % width]);
    }
}

#[test]
fn version_three_fixtures_are_stable() {
    let bytes = include_bytes!("../../../fixtures/task-v3/standard_measures.golden.json");
    let task = parse_pricing_task_json(bytes, JsonLimits::DEFAULT).unwrap();
    assert_eq!(
        format!("{}\n", pricing_task_to_pretty_json(&task).unwrap()).as_bytes(),
        bytes
    );
    assert_eq!(
        task.fingerprint().unwrap().to_string(),
        "blake3-256:6d50c4aba82d09b721a73d54a77793d405d873f2b714e8aa22a4de03b738432a"
    );
    assert_eq!(task.measures().unwrap().requests().len(), 4);

    // The saved report is a document check only: numbers are not replayed
    // across platforms here.
    let saved = include_bytes!("../../../fixtures/report-v3/standard_measures.golden.json");
    let report = parse_valuation_report_json(saved, JsonLimits::DEFAULT).unwrap();
    assert_eq!(report.task_fingerprint(), task.fingerprint().unwrap());
    assert_eq!(
        format!("{}\n", valuation_report_to_pretty_json(&report).unwrap()).as_bytes(),
        saved
    );
    assert!(report.risk(MeasureKind::SpotGamma).is_some());
}
