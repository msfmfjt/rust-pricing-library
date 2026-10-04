use crate::mc::ExecutionPolicy;

use super::*;

fn request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::EuropeanVanilla(
        EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            "2027-09-04".parse().expect("date"),
            100.0,
            1.0,
            OptionSide::Call,
        )
        .expect("product"),
    );
    let forward = EquityForward::new(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.98),
    );
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn american_request() -> PricingRequest {
    let base = request();
    let product = ProductSpec::AmericanVanilla(
        AmericanVanillaSpec::new(
            base.product().underlying(),
            base.product().currency(),
            base.product().expiry(),
            100.0,
            1.0,
            OptionSide::Put,
            vec![
                "2027-03-04".parse().expect("exercise date"),
                base.product().expiry(),
            ],
        )
        .expect("American product"),
    );
    let training_engine = EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(19, 512, VarianceReduction::new(true, false)).expect("training engine"),
    );
    let lsm = LsmConfig::new(
        training_engine,
        vec![LsmStateVariable::Spot],
        PolynomialBasisSpec::new(1, 2, 3, 3).expect("basis"),
        1.0e-12,
        CpqrConfig::new(1.0e-12, 1.0e-10).expect("cpqr"),
        4096,
    )
    .expect("lsm");
    PricingRequest::new_with_lsm(
        base.valuation_date(),
        product,
        base.market().clone(),
        base.model().clone(),
        base.engine(),
        base.risk().clone(),
        Some(lsm),
    )
    .expect("American request")
}

fn black_76_request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::EuropeanVanilla(
        EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            "2027-09-04".parse().expect("date"),
            100.0,
            1.0,
            OptionSide::Call,
        )
        .expect("product"),
    );
    let forward = EquityForward::new(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.95),
    );
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::Black76(Black76Spec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn digital_request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::Digital(
        DigitalSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            "2027-09-04".parse().expect("date"),
            100.0,
            10.0,
            OptionSide::Call,
            DigitalPayout::Cash,
        )
        .expect("product"),
    );
    let forward = EquityForward::new(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.98),
    );
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn barrier_request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::Barrier(
        BarrierSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            "2027-09-04".parse().expect("expiry"),
            100.0,
            120.0,
            1.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            BarrierMonitoring::Continuous,
            vec![
                "2027-03-04".parse().expect("monitoring"),
                "2027-09-04".parse().expect("expiry"),
            ],
            Some(3.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("product"),
    );
    let forward = EquityForward::new(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.98),
    );
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn asian_request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::ArithmeticAsian(
        ArithmeticAsianSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            100.0,
            1.0,
            OptionSide::Call,
            vec![
                AsianObservation::unknown("2026-09-04".parse().expect("date"), 0.25)
                    .expect("first"),
                AsianObservation::unknown("2027-09-04".parse().expect("date"), 0.75)
                    .expect("second"),
            ],
            "2027-09-04".parse().expect("payment"),
        )
        .expect("product"),
    );
    let forward = EquityForward::new(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.98),
    );
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn lookback_request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::FixedLookback(
        FixedLookbackSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            100.0,
            1.0,
            OptionSide::Put,
            vec![
                "2026-03-04".parse().expect("date"),
                "2027-09-04".parse().expect("date"),
            ],
            Some(92.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("product"),
    );
    let forward = EquityForward::new(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.98),
    );
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn dividend_request() -> PricingRequest {
    let curve = |id, discount| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("curve"),
        )
    };
    let product = ProductSpec::EuropeanVanilla(
        EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            "2027-09-04".parse().expect("date"),
            95.0,
            1.0,
            OptionSide::Call,
        )
        .expect("product"),
    );
    let event = EventId::new(77);
    let forward = EquityForward::with_discrete_dividends(
        UnderlyingId::new(1),
        PositiveF64::new(100.0, "spot").expect("spot"),
        curve(10, 0.95),
        curve(11, 0.98),
        vec![
            DividendEvent::new(
                event,
                0.25,
                DividendQuote::fixed_cash_and_proportional(1.5, 0.02, event).expect("quote"),
            )
            .expect("event"),
        ],
    )
    .expect("forward");
    PricingRequest::new(
        "2026-09-04".parse().expect("date"),
        product,
        MarketContext::Equity(EquityMarket::new(CurrencyId::new(2), forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
        ),
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn local_vol_request() -> PricingRequest {
    local_vol_request_with_model(
        LocalVolatilitySpec::from_explicit_grid(
            vec![0.25, 1.0],
            vec![-0.1, 0.0, 0.2],
            vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
            1.0e-8,
            4.0,
        )
        .expect("local vol"),
    )
}

fn local_vol_request_with_model(model: LocalVolatilitySpec) -> PricingRequest {
    let mut request = request();
    request = PricingRequest::new(
        request.valuation_date(),
        request.product().clone(),
        request.market().clone(),
        ModelSpec::LocalVolatility(model),
        request.engine(),
        request.risk().clone(),
    )
    .expect("request");
    request
}

fn local_vol_vega_kt_request() -> PricingRequest {
    let request = local_vol_request();
    PricingRequest::new(
        request.valuation_date(),
        request.product().clone(),
        request.market().clone(),
        request.model().clone(),
        request.engine(),
        RiskRequest::new(
            true,
            None,
            true,
            Some(
                VegaKtConfig::new(
                    vec![
                        "2027-03-04".parse().expect("date"),
                        "2027-09-04".parse().expect("date"),
                    ],
                    vec![-0.2, 0.0, 0.2],
                    1.0e-8,
                    true,
                )
                .expect("vega kt"),
            ),
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(256),
        )
        .expect("risk"),
    )
    .expect("request")
}

#[test]
fn request_round_trip_and_noncanonical_input_have_same_fingerprint() {
    let request = request();
    let compact = request_to_json(&request).expect("json");
    assert_eq!(
        compact,
        include_str!("../../../../fixtures/v3/pricing_request.golden.json")
    );
    assert_json_text_contract(&compact);
    let parsed = parse_request_json(compact.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(request, parsed);
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    let pretty = request_to_pretty_json(&parsed).expect("pretty");
    assert_json_text_contract(&pretty);
    let reparsed =
        parse_request_json(pretty.as_bytes(), JsonLimits::DEFAULT).expect("parse pretty");
    assert_eq!(
        fingerprint_request(&parsed).expect("fingerprint"),
        fingerprint_request(&reparsed).expect("fingerprint")
    );
}

#[test]
fn schema_v1_documents_migrate_forward_and_remain_strict() {
    assert_eq!(MigrationRegistry.accepted_source_versions(), &[1, 2, 3]);
    let request_v1 = include_str!("../../../../fixtures/v1/pricing_request.golden.json");
    let request =
        parse_request_json(request_v1.as_bytes(), JsonLimits::DEFAULT).expect("migrate request");
    let request_migration = request.wire_migration().expect("request migration");
    assert_eq!(request_migration.original_schema_version().get(), 1);
    assert_eq!(request_migration.current_schema_version().get(), 3);
    assert_eq!(
        request_migration.migration_ids(),
        &[
            MIGRATION_REQUEST_V1_TO_V2.to_owned(),
            MIGRATION_REQUEST_V2_TO_V3.to_owned(),
        ]
    );
    assert_ne!(
        request_migration.pre_migration_fingerprint(),
        request_migration.post_migration_fingerprint()
    );
    let migrated_request_result = crate::SimulationPlan::compile(
        &request,
        ExecutionPolicy::new(1, Some(256)).expect("policy"),
    )
    .expect("plan")
    .execute()
    .expect("result");
    assert_eq!(
        migrated_request_result.pricing_result.replay.migration(),
        request_migration
    );
    assert_eq!(
        request_to_json(&request).expect("current request"),
        include_str!("../../../../fixtures/v3/pricing_request.golden.json")
    );
    let invalid_v1 = request_v1.replacen(
        "\"smile_dynamics\"",
        "\"payoff_smoothing_width_ladder\":[2.0,1.0],\"smile_dynamics\"",
        1,
    );
    assert!(parse_request_json(invalid_v1.as_bytes(), JsonLimits::DEFAULT).is_err());

    let result_v1 = include_str!("../../../../fixtures/v1/pricing_result.golden.json");
    let result =
        parse_result_json(result_v1.as_bytes(), JsonLimits::DEFAULT).expect("migrate result");
    assert_eq!(
        result_to_json(&result).expect("current result"),
        include_str!("../../../../fixtures/v3/pricing_result_v1_migrated.golden.json")
    );
    assert_eq!(result.replay.schema_version(), SchemaVersion::CURRENT);
    assert_eq!(
        result.replay.migration().migration_ids(),
        &[
            MIGRATION_RESULT_V1_TO_V2.to_owned(),
            MIGRATION_RESULT_V2_TO_V3.to_owned(),
        ]
    );
}

#[test]
fn schema_v2_documents_migrate_to_v3_with_adjacent_provenance() {
    let request_v2 = include_str!("../../../../fixtures/v2/pricing_request.golden.json");
    let request = parse_request_json(request_v2.as_bytes(), JsonLimits::DEFAULT).expect("request");
    let migration = request.wire_migration().expect("migration");
    assert_eq!(migration.original_schema_version().get(), 2);
    assert_eq!(migration.current_schema_version().get(), 3);
    assert_eq!(
        migration.migration_ids(),
        &[MIGRATION_REQUEST_V2_TO_V3.to_owned()]
    );
    assert_ne!(
        migration.pre_migration_fingerprint(),
        migration.post_migration_fingerprint()
    );
    assert_eq!(
        request_to_json(&request).expect("current request"),
        include_str!("../../../../fixtures/v3/pricing_request.golden.json")
    );

    let result_v2 = include_str!("../../../../fixtures/v2/pricing_result.golden.json");
    let result = parse_result_json(result_v2.as_bytes(), JsonLimits::DEFAULT).expect("result");
    assert_eq!(result.replay.migration().original_schema_version().get(), 2);
    assert_eq!(result.replay.migration().current_schema_version().get(), 3);
    assert_eq!(
        result.replay.migration().migration_ids(),
        &[MIGRATION_RESULT_V2_TO_V3.to_owned()]
    );
    assert_eq!(
        result_to_json(&result).expect("current result"),
        include_str!("../../../../fixtures/v3/pricing_result_v2_migrated.golden.json")
    );
}

#[test]
fn request_json_round_trips_american_product_and_lsm_configuration() {
    let request = american_request();
    let expected_lsm_fingerprint = request.lsm().expect("lsm").fingerprint();

    let json = request_to_json(&request).expect("json");
    assert_eq!(
        json,
        include_str!("../../../../fixtures/v3/pricing_request_american.golden.json")
    );
    assert!(json.contains("\"type\":\"american_vanilla\""));
    assert!(json.contains("\"exercise_dates\":[\"2027-03-04\",\"2027-09-04\"]"));
    assert!(json.contains("\"lsm\":{"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");

    assert_eq!(request, parsed);
    assert_eq!(
        parsed.lsm().expect("parsed lsm").fingerprint(),
        expected_lsm_fingerprint
    );
    assert_eq!(request_to_json(&parsed).expect("json"), json);
    assert_eq!(
        fingerprint_request(&parsed).expect("fingerprint"),
        fingerprint_request(&request).expect("fingerprint")
    );

    let mut legacy_value: Value = serde_json::from_str(&json).expect("request value");
    legacy_value["schema_version"] = 2.into();
    legacy_value
        .as_object_mut()
        .expect("request object")
        .remove("lsm");
    assert!(
        parse_request_json(
            serde_json::to_string(&legacy_value)
                .expect("legacy-shaped JSON")
                .as_bytes(),
            JsonLimits::DEFAULT,
        )
        .is_err()
    );

    let oversized_basis = json.replacen("\"max_degree\":2", "\"max_degree\":4294967295", 1);
    assert!(matches!(
        parse_request_json(oversized_basis.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/lsm/basis/max_degree" && message.contains("resource limit")
    ));
}

#[test]
fn request_json_round_trips_black_76_model() {
    let request = black_76_request();

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"type\":\"black_76\""));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert!(matches!(parsed.model(), ModelSpec::Black76(_)));
    assert_eq!(request_to_json(&parsed).expect("json"), json);
}

#[test]
fn request_json_round_trips_digital_product() {
    let request = digital_request();

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"type\":\"digital\""));
    assert!(json.contains("\"payout_kind\":{\"type\":\"cash\"}"));
    assert!(json.contains("\"payment_date\":\"2027-09-04\""));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert!(matches!(parsed.product(), ProductSpec::Digital(_)));
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    assert_eq!(request_to_json(&parsed).expect("json"), json);
}

#[test]
fn request_json_round_trips_digital_payoff_smoothing() {
    let base = digital_request();
    let risk = RiskRequest::new(
        true,
        None,
        true,
        None,
        SmileDynamics::StickyLogMoneyness,
        None,
        None,
    )
    .expect("risk")
    .with_payoff_smoothing(PayoffSmoothing::compact_c2(2.0).expect("smoothing"));
    let request = PricingRequest::new(
        base.valuation_date(),
        base.product().clone(),
        base.market().clone(),
        base.model().clone(),
        base.engine(),
        risk,
    )
    .expect("request");

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"payoff_smoothing\":{\"type\":\"compact_c2\",\"half_width\":2.0}"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(
        parsed.risk().payoff_smoothing(),
        request.risk().payoff_smoothing()
    );
    assert_eq!(fingerprint_request(&parsed), fingerprint_request(&request));
}

#[test]
fn request_json_round_trips_payoff_smoothing_width_ladder() {
    let base = digital_request();
    let risk = RiskRequest::new(
        true,
        None,
        true,
        None,
        SmileDynamics::StickyLogMoneyness,
        None,
        None,
    )
    .expect("risk")
    .with_payoff_smoothing(PayoffSmoothing::compact_c2(3.0).expect("primary"))
    .with_payoff_smoothing_width_ladder(
        PayoffSmoothingWidthLadder::new(vec![4.0, 2.0, 1.0]).expect("ladder"),
    );
    let request = PricingRequest::new(
        base.valuation_date(),
        base.product().clone(),
        base.market().clone(),
        base.model().clone(),
        base.engine(),
        risk,
    )
    .expect("request");

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"payoff_smoothing_width_ladder\":[4.0,2.0,1.0]"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(
        parsed
            .risk()
            .payoff_smoothing_width_ladder()
            .expect("ladder")
            .half_widths(),
        request
            .risk()
            .payoff_smoothing_width_ladder()
            .expect("ladder")
            .half_widths()
    );
    assert_eq!(request_to_json(&parsed).expect("json"), json);
    assert_eq!(fingerprint_request(&parsed), fingerprint_request(&request));
}

#[test]
fn digital_request_json_defaults_missing_payment_date_to_expiry() {
    let json = request_to_json(&digital_request()).expect("json");
    let legacy_json = json.replace(",\"payment_date\":\"2027-09-04\"", "");

    let parsed = parse_request_json(legacy_json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    let ProductSpec::Digital(product) = parsed.product() else {
        panic!("digital product");
    };
    assert_eq!(product.payment_date(), product.expiry());
    assert_eq!(request_to_json(&parsed).expect("json"), json);
}

#[test]
fn request_json_round_trips_barrier_product() {
    let request = barrier_request();

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"type\":\"barrier\""));
    assert!(json.contains("\"direction\":{\"type\":\"up\"}"));
    assert!(json.contains("\"style\":{\"type\":\"knock_out\"}"));
    assert!(json.contains("\"monitoring\":{\"type\":\"continuous\"}"));
    assert!(json.contains("\"rebate\":3.0"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert!(matches!(
        parsed.product(),
        ProductSpec::Barrier(product)
            if product.monitoring() == BarrierMonitoring::Continuous
    ));
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    assert_eq!(request_to_json(&parsed).expect("json"), json);
}

#[test]
fn barrier_history_round_trips_and_validates_schedule_and_monitoring() {
    let mut value: serde_json::Value =
        serde_json::from_str(&request_to_json(&barrier_request()).unwrap()).unwrap();
    value["product"]["monitoring"]["type"] = "discrete".into();
    let parse = |v: &serde_json::Value| {
        parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT)
    };
    let original = request_to_json(&parse(&value).unwrap()).unwrap();
    assert!(!original.contains("historical_hit"));
    let mut fingerprints = Vec::new();
    for hit in [false, true] {
        value["product"]["historical_hit"] = hit.into();
        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("requires monitoring dates before")
        );
        value["product"]["monitoring_dates"][0] = "2026-09-03".into();
        let request = parse(&value).unwrap();
        let json = request_to_json(&request).unwrap();
        assert_eq!(
            request,
            parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).unwrap()
        );
        let ProductSpec::Barrier(spec) = request.product() else {
            panic!()
        };
        assert_eq!(spec.historical_hit(), Some(hit));
        fingerprints.push(fingerprint_request(&request).unwrap());
        for version in [1, 2, 3] {
            value["schema_version"] = version.into();
            let migrated = parse(&value).unwrap();
            assert_eq!(migrated.product(), request.product());
            assert!(
                request_to_json(&migrated)
                    .unwrap()
                    .contains(&format!("\"historical_hit\":{hit}"))
            );
        }
        value["product"]["monitoring"]["type"] = "continuous".into();
        let continuous = parse(&value).unwrap();
        let ProductSpec::Barrier(spec) = continuous.product() else {
            panic!()
        };
        assert_eq!(spec.historical_hit(), Some(hit));
        assert_eq!(spec.monitoring(), BarrierMonitoring::Continuous);
        let round_trip = parse_request_json(
            request_to_json(&continuous).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(round_trip, continuous);
        value["product"]["monitoring"]["type"] = "discrete".into();
        value["product"]["monitoring_dates"][0] = "2027-03-04".into();
    }
    assert_ne!(fingerprints[0], fingerprints[1]);
    value["product"]["monitoring_dates"][0] = "2026-09-03".into();
    value["product"]
        .as_object_mut()
        .unwrap()
        .remove("historical_hit");
    assert!(
        parse(&value)
            .unwrap_err()
            .to_string()
            .contains("requires historical barrier state")
    );
    for invalid in [serde_json::json!(1), serde_json::json!("false")] {
        value["product"]["historical_hit"] = invalid;
        assert!(parse(&value).is_err());
    }
}

#[test]
fn request_json_round_trips_arithmetic_asian_product() {
    let request = asian_request();

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"type\":\"arithmetic_asian\""));
    assert!(json.contains("\"observations\""));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert!(matches!(parsed.product(), ProductSpec::ArithmeticAsian(_)));
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    assert_eq!(request_to_json(&parsed).expect("json"), json);
}

#[test]
fn request_json_round_trips_fixed_lookback_product() {
    let request = lookback_request();

    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"type\":\"fixed_lookback\""));
    assert!(json.contains("\"historical_extremum\":92.0"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert!(matches!(parsed.product(), ProductSpec::FixedLookback(_)));
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    assert_eq!(request_to_json(&parsed).expect("json"), json);
}

#[test]
fn absent_optional_product_values_are_omitted_instead_of_serialized_as_null() {
    let barrier = ProductV1::Barrier {
        underlying_id: 1,
        currency_id: 2,
        expiry: "2027-09-04".to_owned(),
        strike: 100.0,
        barrier: 120.0,
        notional: 1.0,
        side: SideV1::Call,
        direction: BarrierDirectionV1::Up,
        style: BarrierStyleV1::KnockOut,
        monitoring: BarrierMonitoringV1::Discrete,
        monitoring_dates: vec!["2027-09-04".to_owned()],
        rebate: None,
        historical_hit: None,
        payment_date: "2027-09-04".to_owned(),
    };
    let lookback = ProductV1::FixedLookback {
        underlying_id: 1,
        currency_id: 2,
        strike: 100.0,
        notional: 1.0,
        side: SideV1::Put,
        monitoring_dates: vec!["2027-09-04".to_owned()],
        historical_extremum: None,
        payment_date: "2027-09-04".to_owned(),
    };
    let barrier_json = serde_json::to_value(barrier).expect("Barrier JSON");
    let lookback_json = serde_json::to_value(lookback).expect("Lookback JSON");
    assert!(barrier_json.get("rebate").is_none());
    assert!(barrier_json.get("historical_hit").is_none());
    assert!(lookback_json.get("historical_extremum").is_none());
}

#[test]
fn request_json_round_trips_discrete_dividends() {
    let request = dividend_request();
    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"discrete_dividends\""));
    assert!(json.contains("\"fixed_cash_and_proportional\""));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    let dividends = parsed
        .market()
        .equity()
        .forward()
        .discrete_dividends()
        .expect("dividends");
    assert_eq!(dividends.events()[0].event(), EventId::new(77));
    assert_eq!(dividends.events()[0].fixed_cash(), 1.5);
    assert_eq!(dividends.events()[0].beta(), 0.02);
}

#[test]
fn request_json_rejects_coincident_discrete_dividend_events() {
    let json = request_to_json(&dividend_request()).expect("json");
    let mut value: Value = serde_json::from_str(&json).expect("json value");
    let dividends = value
        .pointer_mut("/market/discrete_dividends")
        .and_then(Value::as_array_mut)
        .expect("dividends");
    let mut duplicate_time_event = dividends[0].clone();
    duplicate_time_event["event_id"] = Value::from(78);
    dividends.push(duplicate_time_event);
    let invalid = serde_json::to_string(&value).expect("invalid json");

    assert!(matches!(
        parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/market/discrete_dividends"
                && message.contains("not strictly increasing")
    ));
}

#[test]
fn request_json_round_trips_local_volatility_grid_shape() {
    let request = local_vol_request();
    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"local_volatility\""));
    assert!(json.contains("\"shape\":[2,3]"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    let ModelSpec::LocalVolatility(model) = parsed.model() else {
        panic!("local vol model");
    };
    assert_eq!(model.local_variance_grid().values()[4], 0.045);
    assert!(model.reporting_iv_basis().is_none());
}

#[test]
fn request_json_round_trips_local_volatility_reporting_iv_basis() {
    let basis = LocalVolatilityReportingBasis::new(
        vec![0.25, 1.0],
        vec![-0.2, 0.0, 0.2],
        vec![0.22, 0.20, 0.21, 0.24, 0.22, 0.23],
    )
    .expect("basis");
    let model = LocalVolatilitySpec::from_explicit_grid(
        vec![0.25, 1.0],
        vec![-0.1, 0.0, 0.2],
        vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
        1.0e-8,
        4.0,
    )
    .expect("local vol")
    .with_reporting_iv_basis(basis);
    let request = local_vol_request_with_model(model);
    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"reporting_iv_basis\""));
    assert!(json.contains("\"shape\":[2,3]"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    let ModelSpec::LocalVolatility(model) = parsed.model() else {
        panic!("local vol model");
    };
    let basis = model.reporting_iv_basis().expect("basis");
    assert_eq!(basis.maturity_nodes(), [0.25, 1.0]);
    assert_eq!(basis.log_forward_moneyness_nodes(), [-0.2, 0.0, 0.2]);
    assert_eq!(basis.implied_volatilities()[4], 0.22);
}

#[test]
fn request_json_rejects_mismatched_reporting_iv_basis_shape() {
    let basis = LocalVolatilityReportingBasis::new(
        vec![0.25, 1.0],
        vec![-0.2, 0.0, 0.2],
        vec![0.22, 0.20, 0.21, 0.24, 0.22, 0.23],
    )
    .expect("basis");
    let model = LocalVolatilitySpec::from_explicit_grid(
        vec![0.25, 1.0],
        vec![-0.1, 0.0, 0.2],
        vec![0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
        1.0e-8,
        4.0,
    )
    .expect("local vol")
    .with_reporting_iv_basis(basis);
    let json = request_to_json(&local_vol_request_with_model(model)).expect("json");
    let invalid = json.replacen(
            "\"reporting_iv_basis\":{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[2,3]",
            "\"reporting_iv_basis\":{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[3,2]",
            1,
        );
    assert!(matches!(
        parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/model/reporting_iv_basis" && message.contains("reporting_iv_basis shape")
    ));

    for value in ["1.0", "1e0", "-0", "\"1\""] {
        let invalid = json.replacen(
                "\"reporting_iv_basis\":{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[2,3]",
                &format!("\"reporting_iv_basis\":{{\"maturity_nodes\":[0.25,1.0],\"log_forward_moneyness_nodes\":[-0.2,0.0,0.2],\"shape\":[{value},3]"),
                1,
            );
        assert!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "request JSON accepted reporting_iv_basis shape value {value}"
        );
    }
}

#[test]
fn request_json_rejects_mismatched_local_volatility_grid_shape() {
    let json = request_to_json(&local_vol_request()).expect("json");
    let invalid = json.replacen("\"shape\":[2,3]", "\"shape\":[3,2]", 1);
    assert!(matches!(
        parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/model/local_variance_grid" && message.contains("shape")
    ));

    for value in ["1.0", "1e0", "-0", "\"1\""] {
        let invalid = json.replacen("\"shape\":[2,3]", &format!("\"shape\":[{value},3]"), 1);
        assert!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "request JSON accepted local_variance_grid shape value {value}"
        );
    }
}

#[test]
fn request_json_domain_date_errors_include_instance_path() {
    let json = request_to_json(&request()).expect("json");
    let invalid = json.replacen(
        "\"valuation_date\":\"2026-09-04\"",
        "\"valuation_date\":\"2026-02-31\"",
        1,
    );
    assert!(matches!(
        parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/valuation_date" && message.contains("invalid date")
    ));
}

#[test]
fn request_json_round_trips_local_volatility_vega_kt_request() {
    let request = local_vol_vega_kt_request();
    let json = request_to_json(&request).expect("json");
    assert!(json.contains("\"vega_kt\""));
    assert!(json.contains("\"full_bucket_covariance\":true"));
    let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    assert_eq!(
        fingerprint_request(&request).expect("fingerprint"),
        fingerprint_request(&parsed).expect("fingerprint")
    );
    let vega_kt = parsed.risk().vega_kt().expect("vega kt");
    assert_eq!(vega_kt.maturity_nodes()[0].to_string(), "2027-03-04");
    assert!(vega_kt.full_bucket_covariance());
}

#[test]
fn strict_reader_rejects_unknown_null_future_and_limits() {
    let json = request_to_json(&request()).expect("json");
    let unknown = json.replacen("\"valuation_date\"", "\"unknown\":1,\"valuation_date\"", 1);
    assert!(parse_request_json(unknown.as_bytes(), JsonLimits::DEFAULT).is_err());
    let null = json.replacen("\"risk\":{", "\"risk\":{\"aad_tile_capacity\":null,", 1);
    assert!(parse_request_json(null.as_bytes(), JsonLimits::DEFAULT).is_err());
    let future = json.replacen("\"schema_version\":3", "\"schema_version\":4", 1);
    assert!(matches!(
        parse_request_json(future.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::UnsupportedSchemaVersion(4))
    ));
    let limits = JsonLimits {
        max_input_bytes: 8,
        ..JsonLimits::DEFAULT
    };
    assert!(matches!(
        parse_request_json(json.as_bytes(), limits),
        Err(WireError::ResourceLimit {
            name: "input_bytes",
            ..
        })
    ));
    for constant in ["NaN", "Infinity", "-Infinity"] {
        let invalid = json.replacen("100.0", constant, 1);
        assert!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "request JSON accepted {constant}"
        );
    }
    for schema_version in ["1.0", "1e0", "-0", "\"1\""] {
        let invalid = json.replacen(
            "\"schema_version\":3",
            &format!("\"schema_version\":{schema_version}"),
            1,
        );
        assert!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "request JSON accepted schema_version {schema_version}"
        );
    }
    for (field, original) in [
        ("underlying_id", "\"underlying_id\":1"),
        ("curve_id", "\"curve_id\":10"),
        ("master_seed", "\"master_seed\":7"),
        (
            "independent_sampling_units",
            "\"independent_sampling_units\":1024",
        ),
    ] {
        for value in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = json.replacen(original, &format!("\"{field}\":{value}"), 1);
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted {field} {value}"
            );
        }
    }

    let rqmc_request = PricingRequest::new(
        request().valuation_date(),
        request().product().clone(),
        request().market().clone(),
        request().model().clone(),
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(256, 4, 11, VarianceReduction::new(true, false)).expect("rqmc engine"),
        ),
        request().risk().clone(),
    )
    .expect("rqmc request");
    let rqmc_json = request_to_json(&rqmc_request).expect("rqmc json");
    for (field, original) in [
        ("points_per_scramble", "\"points_per_scramble\":256"),
        ("scramble_count", "\"scramble_count\":4"),
        ("master_scramble_seed", "\"master_scramble_seed\":11"),
    ] {
        for value in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = rqmc_json.replacen(original, &format!("\"{field}\":{value}"), 1);
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted {field} {value}"
            );
        }
    }

    let risk_json = request_to_json(&local_vol_vega_kt_request()).expect("risk json");
    for (field, original) in [
        ("checkpoint_interval", "\"checkpoint_interval\":16"),
        ("aad_tile_capacity", "\"aad_tile_capacity\":256"),
    ] {
        for value in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = risk_json.replacen(original, &format!("\"{field}\":{value}"), 1);
            assert!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "request JSON accepted {field} {value}"
            );
        }
    }
}

#[test]
fn strict_reader_rejects_malformed_tagged_union_types() {
    let request_json = request_to_json(&request()).expect("request json");
    for (name, invalid) in [
        (
            "numeric document kind",
            request_json.replacen(
                "\"document_kind\":\"pricing_request\"",
                "\"document_kind\":1",
                1,
            ),
        ),
        (
            "unknown document kind",
            request_json.replacen(
                "\"document_kind\":\"pricing_request\"",
                "\"document_kind\":\"PricingRequest\"",
                1,
            ),
        ),
        (
            "missing product type",
            request_json.replacen("\"type\":\"european_vanilla\",", "", 1),
        ),
        (
            "numeric product type",
            request_json.replacen("\"type\":\"european_vanilla\"", "\"type\":1", 1),
        ),
        (
            "unknown product type",
            request_json.replacen(
                "\"type\":\"european_vanilla\"",
                "\"type\":\"EuropeanVanilla\"",
                1,
            ),
        ),
        (
            "bare side variant",
            request_json.replacen("\"side\":{\"type\":\"call\"}", "\"side\":\"call\"", 1),
        ),
    ] {
        assert!(
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "request JSON accepted {name}"
        );
    }

    let result_json = include_str!("../../../../fixtures/v1/pricing_result.golden.json");
    for (name, invalid) in [
        (
            "numeric document kind",
            result_json.replacen(
                "\"document_kind\":\"pricing_result\"",
                "\"document_kind\":1",
                1,
            ),
        ),
        (
            "unknown document kind",
            result_json.replacen(
                "\"document_kind\":\"pricing_result\"",
                "\"document_kind\":\"PricingResult\"",
                1,
            ),
        ),
        (
            "missing estimator type",
            result_json.replacen(
                "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                "\"estimator\":{}",
                1,
            ),
        ),
        (
            "numeric estimator type",
            result_json.replacen(
                "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                "\"estimator\":{\"type\":1}",
                1,
            ),
        ),
        (
            "unknown estimator type",
            result_json.replacen(
                "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                "\"estimator\":{\"type\":\"PseudoMonteCarlo\"}",
                1,
            ),
        ),
        (
            "bare estimator variant",
            result_json.replacen(
                "\"estimator\":{\"type\":\"pseudo_monte_carlo\"}",
                "\"estimator\":\"pseudo_monte_carlo\"",
                1,
            ),
        ),
    ] {
        assert!(
            parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "result JSON accepted {name}"
        );
    }
}

#[test]
fn strict_reader_rejects_duplicate_object_members_recursively() {
    let request_json = request_to_json(&request()).expect("request json");
    let duplicate_request_root = request_json.replacen(
        "\"valuation_date\"",
        "\"schema_version\":3,\"valuation_date\"",
        1,
    );
    let duplicate_request_nested =
        request_json.replacen("\"spot\":100.0", "\"spot\":100.0,\"spot\":101.0", 1);
    for invalid in [duplicate_request_root, duplicate_request_nested] {
        assert!(
            matches!(
                parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT),
                Err(WireError::Json(message)) if message.contains("duplicate object member")
            ),
            "request JSON accepted a duplicate object member"
        );
    }

    let result_json = include_str!("../../../../fixtures/v1/pricing_result.golden.json");
    let duplicate_result_root =
        result_json.replacen("\"value\"", "\"schema_version\":1,\"value\"", 1);
    let duplicate_result_nested = result_json.replacen(
        "\"standard_error\":0.5",
        "\"standard_error\":0.5,\"standard_error\":0.6",
        1,
    );
    for invalid in [duplicate_result_root, duplicate_result_nested] {
        assert!(
            matches!(
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT),
                Err(WireError::Json(message)) if message.contains("duplicate object member")
            ),
            "result JSON accepted a duplicate object member"
        );
    }
}

#[test]
fn strict_reader_rejects_bom_comments_and_trailing_tokens() {
    let request_json = request_to_json(&request()).expect("request json");
    let result_json = include_str!("../../../../fixtures/v1/pricing_result.golden.json");
    let cases = [
        ("\u{feff}".to_owned() + &request_json, "request BOM"),
        (
            request_json.replacen("{", "{// comment\n", 1),
            "request comment",
        ),
        (request_json.clone() + "{}", "request trailing token"),
        ("\u{feff}".to_owned() + result_json, "result BOM"),
        (
            result_json.replacen("{", "{// comment\n", 1),
            "result comment",
        ),
        (result_json.to_owned() + "{}", "result trailing token"),
    ];

    assert!(parse_request_json(&[0xff], JsonLimits::DEFAULT).is_err());
    assert!(parse_result_json(&[0xff], JsonLimits::DEFAULT).is_err());

    for (invalid, name) in cases {
        let rejected = if name.starts_with("request") {
            parse_request_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err()
        } else {
            parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err()
        };
        assert!(rejected, "{name} was accepted");
    }
}

#[test]
fn json_limits_report_stable_resource_names() {
    let cases = [
        (
            br#"{"document_kind":"pricing_request"}"#.as_slice(),
            JsonLimits {
                max_string_bytes: 8,
                ..JsonLimits::DEFAULT
            },
            "string_bytes",
            13,
            8,
        ),
        (
            b"1234".as_slice(),
            JsonLimits {
                max_number_token_bytes: 3,
                ..JsonLimits::DEFAULT
            },
            "number_token_bytes",
            4,
            3,
        ),
        (
            b"[[[]]]".as_slice(),
            JsonLimits {
                max_nesting_depth: 2,
                ..JsonLimits::DEFAULT
            },
            "nesting_depth",
            3,
            2,
        ),
        (
            b"[1,2]".as_slice(),
            JsonLimits {
                max_array_elements: 1,
                ..JsonLimits::DEFAULT
            },
            "array_elements",
            2,
            1,
        ),
        (
            br#"{"a":1,"b":2}"#.as_slice(),
            JsonLimits {
                max_object_members: 1,
                ..JsonLimits::DEFAULT
            },
            "object_members",
            2,
            1,
        ),
        (
            b"[1]".as_slice(),
            JsonLimits {
                max_total_values: 1,
                ..JsonLimits::DEFAULT
            },
            "total_values",
            2,
            1,
        ),
    ];

    for (input, limits, expected_name, expected_observed, expected_limit) in cases {
        assert!(
            matches!(
                parse_request_json(input, limits),
                Err(WireError::ResourceLimit { name, observed, limit })
                    if name == expected_name
                        && observed == expected_observed
                        && limit == expected_limit
            ),
            "expected request resource limit {expected_name}"
        );
        assert!(
            matches!(
                parse_result_json(input, limits),
                Err(WireError::ResourceLimit { name, observed, limit })
                    if name == expected_name
                        && observed == expected_observed
                        && limit == expected_limit
            ),
            "expected result resource limit {expected_name}"
        );
    }
}

#[test]
fn json_limits_have_stable_default_and_hard_cap_values() {
    assert_eq!(
        JsonLimits::DEFAULT,
        JsonLimits {
            max_input_bytes: 4 * 1024 * 1024,
            max_nesting_depth: 64,
            max_string_bytes: 1024 * 1024,
            max_number_token_bytes: 128,
            max_array_elements: 100_000,
            max_object_members: 10_000,
            max_total_values: 250_000,
        }
    );
    assert_eq!(
        JsonLimits::HARD_CAP,
        JsonLimits {
            max_input_bytes: 64 * 1024 * 1024,
            max_nesting_depth: 256,
            max_string_bytes: 16 * 1024 * 1024,
            max_number_token_bytes: 1024,
            max_array_elements: 2_000_000,
            max_object_members: 250_000,
            max_total_values: 4_000_000,
        }
    );
    assert_eq!(JsonLimits::default(), JsonLimits::DEFAULT);
}

#[test]
fn json_limit_overrides_reject_each_hard_cap_excess() {
    let hard = JsonLimits::HARD_CAP;
    assert_eq!(hard.checked(), Ok(hard));

    let cases = [
        JsonLimits {
            max_input_bytes: hard.max_input_bytes + 1,
            ..hard
        },
        JsonLimits {
            max_nesting_depth: hard.max_nesting_depth + 1,
            ..hard
        },
        JsonLimits {
            max_string_bytes: hard.max_string_bytes + 1,
            ..hard
        },
        JsonLimits {
            max_number_token_bytes: hard.max_number_token_bytes + 1,
            ..hard
        },
        JsonLimits {
            max_array_elements: hard.max_array_elements + 1,
            ..hard
        },
        JsonLimits {
            max_object_members: hard.max_object_members + 1,
            ..hard
        },
        JsonLimits {
            max_total_values: hard.max_total_values + 1,
            ..hard
        },
    ];

    for limits in cases {
        assert_eq!(
            limits.checked(),
            Err(WireError::LimitOverrideExceedsHardCap)
        );
    }
}

#[test]
fn bundled_schemas_are_draft_2020_12_json() {
    for schema in [current_request_schema(), current_result_schema()] {
        let value: Value = serde_json::from_str(schema).expect("schema JSON");
        assert_eq!(
            value["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }
}

#[test]
fn result_compact_json_matches_golden_and_round_trips() {
    let value = Estimate::new(10.0, 0.5, 9.0, 11.0, EstimatorKind::PseudoMonteCarlo, 1024)
        .expect("estimate");
    let result = PricingResult {
        value,
        risks: RiskReport::default(),
        diagnostics: Diagnostics::new(vec![PricingWarning::new(
            "curve_extrapolation",
            "discount curve extrapolated",
        )]),
        replay: ReplayMetadata::new(SchemaVersion::CURRENT, [0; 32], "0.1.0", "acceptance-test"),
    };
    let json = result_to_json(&result).expect("json");
    assert_eq!(
        json,
        include_str!("../../../../fixtures/v3/pricing_result.golden.json")
    );
    assert_json_text_contract(&json);
    assert_json_text_contract(&result_to_pretty_json(&result).expect("pretty json"));
    assert_eq!(
        parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("round trip"),
        result
    );
}

#[test]
fn schema_v3_rejects_experimental_calibration_domain_on_write() {
    let mut result = parse_monte_carlo_result_json(
        include_bytes!("../../../../fixtures/v3/pricing_result_american.golden.json"),
        JsonLimits::DEFAULT,
    )
    .expect("accepted American replay fixture");
    result
        .early_exercise_diagnostics
        .as_mut()
        .expect("LSM diagnostics")
        .training_random_domain = RandomDomain::LsvCalibration;
    assert!(monte_carlo_result_to_json(&result).is_err());
    assert!(monte_carlo_result_to_pretty_json(&result).is_err());
}

#[test]
fn monte_carlo_result_round_trips_complete_american_lsm_replay_state() {
    let result = crate::price_monte_carlo(
        &american_request(),
        ExecutionPolicy::new(2, Some(256)).expect("execution policy"),
    )
    .expect("American result");
    let json = monte_carlo_result_to_json(&result).expect("JSON");
    if std::env::var_os("UPDATE_AMERICAN_RESULT_GOLDEN").is_some() {
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/v3/pricing_result_american.golden.json"
            ),
            &json,
        )
        .expect("write American result golden");
    }
    let golden = include_str!("../../../../fixtures/v3/pricing_result_american.golden.json");
    let frozen_platform = "\"platform\":\"macos-aarch64\"";
    assert_eq!(golden.matches(frozen_platform).count(), 1);
    let expected = golden.replacen(
        frozen_platform,
        &format!(
            "\"platform\":{}",
            serde_json::to_string(result.pricing_result.replay.platform())
                .expect("platform JSON string")
        ),
        1,
    );
    assert_eq!(json, expected);
    assert_json_text_contract(&json);
    assert!(json.contains("\"early_exercise\""));
    assert!(json.contains("\"decision_models\""));
    assert_eq!(
        parse_monte_carlo_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("round trip"),
        result
    );
    assert_eq!(
        parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("core result"),
        result.pricing_result
    );

    let mut invalid: Value = serde_json::from_str(&json).expect("JSON value");
    invalid["monte_carlo"]["early_exercise"]["stopping_indices"][0] = serde_json::json!(999);
    assert!(matches!(
        parse_monte_carlo_result_json(
            serde_json::to_string(&invalid).expect("JSON").as_bytes(),
            JsonLimits::DEFAULT,
        ),
        Err(WireError::DomainAt { pointer, .. })
            if pointer == "/monte_carlo/early_exercise/stopping_indices"
    ));
    assert!(
        parse_result_json(
            serde_json::to_string(&invalid).expect("JSON").as_bytes(),
            JsonLimits::DEFAULT,
        )
        .is_err()
    );

    let mut unknown: Value = serde_json::from_str(&json).expect("JSON value");
    unknown["monte_carlo"]["early_exercise"]["unknown"] = serde_json::json!(true);
    assert!(
        parse_monte_carlo_result_json(
            serde_json::to_string(&unknown).expect("JSON").as_bytes(),
            JsonLimits::DEFAULT,
        )
        .is_err()
    );
}

fn assert_json_text_contract(json: &str) {
    let without_final_lf = json.strip_suffix('\n').expect("final newline");
    assert!(!without_final_lf.ends_with('\n'));
    assert!(!json.as_bytes().starts_with(&[0xef, 0xbb, 0xbf]));
    assert!(!json.contains("\r\n"));
}

#[test]
fn result_json_domain_errors_include_instance_paths() {
    let json = include_str!("../../../../fixtures/v3/pricing_result.golden.json");
    let invalid_estimate = json.replacen("\"standard_error\":0.5", "\"standard_error\":-0.5", 1);
    assert!(matches!(
        parse_result_json(invalid_estimate.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/value" && message.contains("standard_error")
    ));

    let invalid_fingerprint = json.replacen(
            "\"request_fingerprint\":\"blake3-256:0000000000000000000000000000000000000000000000000000000000000000\"",
            "\"request_fingerprint\":\"not-a-fingerprint\"",
            1,
        );
    assert!(matches!(
        parse_result_json(invalid_fingerprint.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/replay/request_fingerprint" && message.contains("fingerprint")
    ));
    let uppercase_fingerprint = json.replacen(
            "\"request_fingerprint\":\"blake3-256:0000000000000000000000000000000000000000000000000000000000000000\"",
            "\"request_fingerprint\":\"blake3-256:ABCDEF0000000000000000000000000000000000000000000000000000000000\"",
            1,
        );
    assert!(matches!(
        parse_result_json(uppercase_fingerprint.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/replay/request_fingerprint" && message.contains("fingerprint")
    ));

    let future_replay_schema = json.replace(
        "\"replay\":{\"schema_version\":3",
        "\"replay\":{\"schema_version\":4",
    );
    assert!(matches!(
        parse_result_json(future_replay_schema.as_bytes(), JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/replay/schema_version"
                && message.contains("unsupported replay schema_version 4")
    ));

    for schema_version in ["1.0", "1e0", "-0", "\"1\""] {
        let invalid_top_level = json.replacen(
            "\"schema_version\":3",
            &format!("\"schema_version\":{schema_version}"),
            1,
        );
        assert!(
            parse_result_json(invalid_top_level.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "result JSON accepted schema_version {schema_version}"
        );
        let invalid_replay = json.replacen(
            "\"replay\":{\"schema_version\":3",
            &format!("\"replay\":{{\"schema_version\":{schema_version}"),
            1,
        );
        assert!(
            parse_result_json(invalid_replay.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "result JSON accepted replay schema_version {schema_version}"
        );
    }

    for effective_sampling_units in ["1.0", "1e0", "-0", "\"1\""] {
        let invalid = json.replacen(
            "\"effective_sampling_units\":1024",
            &format!("\"effective_sampling_units\":{effective_sampling_units}"),
            1,
        );
        assert!(
            parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "result JSON accepted effective_sampling_units {effective_sampling_units}"
        );
    }

    for (field, original, pointer) in [
        ("library_version", "0.1.0", "/replay/library_version"),
        ("platform", "acceptance-test", "/replay/platform"),
    ] {
        let invalid = json.replacen(
            &format!("\"{field}\":\"{original}\""),
            &format!("\"{field}\":\"\""),
            1,
        );
        assert!(matches!(
            parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT),
            Err(WireError::DomainAt { pointer: actual, message })
                if actual == pointer && message.contains("empty")
        ));
    }

    for constant in ["NaN", "Infinity", "-Infinity"] {
        let invalid = json.replacen("10.0", constant, 1);
        assert!(
            parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "result JSON accepted {constant}"
        );
    }
}

#[test]
fn result_json_rejects_inconsistent_migration_provenance() {
    let json = include_str!("../../../../fixtures/v3/pricing_result.golden.json");

    let mut wrong_ids: serde_json::Value = serde_json::from_str(json).expect("fixture");
    wrong_ids["replay"]["migration"]["original_schema_version"] = 1.into();
    assert!(matches!(
        parse_result_json(
            serde_json::to_string(&wrong_ids).expect("json").as_bytes(),
            JsonLimits::DEFAULT
        ),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/replay/migration/migration_ids"
                && message.contains("schema path")
    ));

    let mut wrong_post: serde_json::Value = serde_json::from_str(json).expect("fixture");
    wrong_post["replay"]["migration"]["post_migration_fingerprint"] =
        "blake3-256:1111111111111111111111111111111111111111111111111111111111111111".into();
    assert!(matches!(
        parse_result_json(
            serde_json::to_string(&wrong_post).expect("json").as_bytes(),
            JsonLimits::DEFAULT
        ),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/replay/migration/post_migration_fingerprint"
                && message.contains("request_fingerprint")
    ));

    let mut wrong_pre: serde_json::Value = serde_json::from_str(json).expect("fixture");
    wrong_pre["replay"]["migration"]["pre_migration_fingerprint"] =
        "blake3-256:1111111111111111111111111111111111111111111111111111111111111111".into();
    assert!(matches!(
        parse_result_json(
            serde_json::to_string(&wrong_pre).expect("json").as_bytes(),
            JsonLimits::DEFAULT
        ),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/replay/migration/pre_migration_fingerprint"
                && message.contains("identical")
    ));
}

#[test]
fn result_json_round_trips_vega_kt_report_without_null_covariance_entries() {
    let estimate =
        Estimate::new(10.0, 0.5, 9.0, 11.0, EstimatorKind::PseudoMonteCarlo, 64).expect("estimate");
    let reporting_stats = VegaKtResultReportingStats::new(1, 2, -0.5, 0.25).expect("stats");
    let vega_kt = VegaKtResult::new(
        vec![
            VegaKtResultCoordinate::new(0.5, -0.1, 0.2).expect("coordinate"),
            VegaKtResultCoordinate::new(0.5, 0.1, 0.21).expect("coordinate"),
        ],
        vec![
            VegaKtResultBucketEstimate::new(1.0, 0.01, Some(0.5), Some(-0.1)).expect("bucket"),
            VegaKtResultBucketEstimate::new(2.0, 0.02, None, Some(0.2)).expect("bucket"),
        ],
        vec![1.0, 2.0],
        Some(vec![Some(0.5), None, None, Some(0.75)]),
        VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor,
        VegaKtResultProjection::new(3.0, -0.25, 2.75, reporting_stats).expect("projection"),
        VegaKtResultResidualDiagnostics::new(1, 2, 1, 0.001, -0.25, 2.75, reporting_stats)
            .expect("residual"),
        VegaKtResultUnit::CurrencyPerUnitAbsoluteVolatility,
        VegaKtResultUnit::CurrencyPerVolatilityPoint,
        "equation_11_first_order_v1",
        "O(delta_t_k)",
    )
    .expect("vega kt");
    let result = PricingResult {
        value: estimate,
        risks: RiskReport {
            vega_kt: Some(vega_kt),
            ..RiskReport::default()
        },
        diagnostics: Diagnostics::default(),
        replay: ReplayMetadata::new(SchemaVersion::CURRENT, [7; 32], "0.1.0", "test-platform"),
    };
    let json = result_to_json(&result).expect("json");
    assert!(json.contains("\"vega_kt\""));
    assert!(json.contains("\"unavailable\""));
    assert!(!json.contains("null"));

    for (field, original) in [
        ("active_domain_start_index", "1"),
        ("active_domain_end_index", "2"),
        ("active_domain_forward_index", "1"),
        ("left_edge_count", "1"),
        ("right_edge_count", "2"),
    ] {
        for value in ["1.0", "1e0", "-0", "\"1\""] {
            let invalid = json.replacen(
                &format!("\"{field}\":{original}"),
                &format!("\"{field}\":{value}"),
                1,
            );
            assert!(
                parse_result_json(invalid.as_bytes(), JsonLimits::DEFAULT).is_err(),
                "result JSON accepted {field} {value}"
            );
        }
    }

    let parsed = parse_result_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
    let parsed_vega_kt = parsed.risks.vega_kt.expect("vega kt");
    assert_eq!(parsed_vega_kt.coordinates().len(), 2);
    assert_eq!(parsed_vega_kt.estimates()[1].sample_variance(), None);
    assert_eq!(
        parsed_vega_kt.full_bucket_covariance().expect("covariance")[1],
        None
    );
    assert_eq!(parsed_vega_kt.projection().scalar_vega().get(), 3.0);

    let mut missing_full_covariance: Value = serde_json::from_str(&json).expect("result JSON");
    missing_full_covariance["risks"]["vega_kt"]
        .as_object_mut()
        .expect("vega kt")
        .remove("full_bucket_covariance");
    let missing_full_covariance = serde_json::to_vec(&missing_full_covariance).expect("JSON");
    assert!(matches!(
        parse_result_json(&missing_full_covariance, JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/risks/vega_kt"
                && message.contains("requires full_bucket_covariance")
    ));

    let mut unexpected_full_covariance: Value = serde_json::from_str(&json).expect("result JSON");
    unexpected_full_covariance["risks"]["vega_kt"]["covariance_layout"] =
        serde_json::json!({"type": "price_and_bucket_variance_only"});
    let unexpected_full_covariance = serde_json::to_vec(&unexpected_full_covariance).expect("JSON");
    assert!(matches!(
        parse_result_json(&unexpected_full_covariance, JsonLimits::DEFAULT),
        Err(WireError::DomainAt { pointer, message })
            if pointer == "/risks/vega_kt"
                && message.contains("must not include full_bucket_covariance")
    ));
}

#[test]
fn result_writer_preserves_negative_zero() {
    let estimate =
        Estimate::new(-0.0, 0.0, -0.0, 0.0, EstimatorKind::Analytical, 1).expect("estimate");
    let result = PricingResult {
        value: estimate,
        risks: RiskReport::default(),
        diagnostics: Diagnostics::default(),
        replay: ReplayMetadata::new(SchemaVersion::CURRENT, [1; 32], "0.1.0", "test"),
    };
    assert!(
        result_to_json(&result)
            .expect("json")
            .contains("\"value\":-0.0")
    );
}
