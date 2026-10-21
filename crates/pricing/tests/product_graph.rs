use pricing::core::{CurrencyId, Date, NodeId, UnderlyingId};
use pricing::mc::{EngineConfig, ExecutionPolicy, RqmcConfig, VarianceReduction};
use pricing::models::{Bergomi1Factor, HullWhite1Factor};
use pricing::product::{
    GraphProduct, ProductSpec, SourceGraph, SourceGraphBuilder, SourceNode, SourceOpcode as Op,
};
use pricing::{
    JsonLimits, PricingPlan, PricingRequest, PricingSpecification, fingerprint_product_graph,
    parse_product_graph_json, parse_request_json, parse_specification_json, product_graph_to_json,
    request_to_json, specification_to_json,
};
use serde_json::{Value, json};

fn date(s: &str) -> Date {
    s.parse().unwrap()
}
fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(2, Some(32)).unwrap()
}
fn base(risk: bool) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"]["independent_sampling_units"] = json!(128);
    v["risk"]["delta"] = json!(risk);
    v["risk"]["vega"] = json!(risk);
    parse_request_json(
        serde_json::to_string(&v).unwrap().as_bytes(),
        JsonLimits::DEFAULT,
    )
    .unwrap()
}
fn replace(r: &PricingRequest, product: ProductSpec, engine: EngineConfig) -> PricingRequest {
    PricingRequest::new(
        r.valuation_date(),
        product,
        r.market().clone(),
        r.model().clone(),
        engine,
        r.risk().clone(),
    )
    .unwrap()
}
fn vanilla(r: &PricingRequest) -> GraphProduct {
    GraphProduct::new(
        r.product().currency(),
        vec![r.product().underlying()],
        r.product().source_graph(r.valuation_date()).unwrap(),
        vec![r.product().payment_date()],
    )
    .unwrap()
}
fn parse(v: &Value) -> Result<GraphProduct, pricing::WireError> {
    parse_product_graph_json(
        serde_json::to_string(v).unwrap().as_bytes(),
        JsonLimits::DEFAULT,
    )
}
fn multiple(r: &PricingRequest) -> GraphProduct {
    let mut b = SourceGraphBuilder::new();
    let first = b.literal(7.0).unwrap();
    let last = b
        .push(Op::TerminalSpot {
            underlying: r.product().underlying(),
            observation_date: r.product().expiry(),
        })
        .unwrap();
    GraphProduct::new(
        r.product().currency(),
        vec![r.product().underlying()],
        b.finish(vec![first, last]),
        vec![date("2027-03-04"), date("2028-03-04")],
    )
    .unwrap()
}

#[test]
fn graph_vanilla_matches_native_price_delta_vega_and_qmc() {
    let base = base(true);
    for engine in [
        base.engine(),
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(64, 4, 91, VarianceReduction::new(true, true)).unwrap(),
        ),
    ] {
        let native = replace(&base, base.product().clone(), engine);
        let graph = replace(&base, ProductSpec::Graph(vanilla(&base)), engine);
        let expected = PricingPlan::compile(&native, policy())
            .unwrap()
            .evaluate()
            .unwrap();
        let actual = PricingPlan::compile(&graph, policy())
            .unwrap()
            .evaluate()
            .unwrap();
        assert_eq!(actual.pricing_result.value, expected.pricing_result.value);
        assert_eq!(actual.pricing_result.risks, expected.pricing_result.risks);
        assert!(request_to_json(&graph).is_err());
        let source = PricingSpecification::from_request(graph, policy());
        let encoded = specification_to_json(&source).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&encoded).unwrap()["schema_version"],
            2
        );
        let restored = parse_specification_json(encoded.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(
            source.fingerprint().unwrap(),
            restored.fingerprint().unwrap()
        );
        assert_eq!(
            source.compile().unwrap().evaluate().unwrap(),
            restored.compile().unwrap().evaluate().unwrap()
        );
        let mut obsolete: Value = serde_json::from_str(&encoded).unwrap();
        obsolete["schema_version"] = json!(1);
        assert!(
            parse_specification_json(
                serde_json::to_string(&obsolete).unwrap().as_bytes(),
                JsonLimits::DEFAULT
            )
            .is_err()
        );
    }
}

#[test]
fn cashflow_discounting_and_aggregate_sampling_statistics_are_correct() {
    use pricing::core::DayCountConvention;
    use pricing::market::DiscountCurve;
    let r = base(true);
    let graph = multiple(&r);
    let source = replace(&r, ProductSpec::Graph(graph.clone()), r.engine());
    let actual = PricingPlan::compile(&source, policy())
        .unwrap()
        .evaluate()
        .unwrap();
    let mut b = SourceGraphBuilder::new();
    let spot = b
        .push(Op::TerminalSpot {
            underlying: r.product().underlying(),
            observation_date: r.product().expiry(),
        })
        .unwrap();
    let last = GraphProduct::new(
        r.product().currency(),
        vec![r.product().underlying()],
        b.finish(vec![spot]),
        vec![graph.last_payment_date()],
    )
    .unwrap();
    let expected =
        PricingPlan::compile(&replace(&r, ProductSpec::Graph(last), r.engine()), policy())
            .unwrap()
            .evaluate()
            .unwrap();
    let time =
        DayCountConvention::Act365F.year_fraction(r.valuation_date(), graph.payment_dates()[0]);
    let cash = 7.0
        * r.market()
            .equity()
            .forward()
            .discount_curve()
            .evaluate(time)
            .unwrap()
            .discount;
    assert!(
        (actual.pricing_result.value.value().get()
            - expected.pricing_result.value.value().get()
            - cash)
            .abs()
            < 1e-12
    );
    assert!(
        (actual.pricing_result.value.standard_error().get()
            - expected.pricing_result.value.standard_error().get())
        .abs()
            < 1e-12
    );
    assert!(
        (actual
            .pricing_result
            .risks
            .delta
            .unwrap()
            .raw()
            .value()
            .get()
            - expected
                .pricing_result
                .risks
                .delta
                .unwrap()
                .raw()
                .value()
                .get())
        .abs()
            < 1e-12
    );
}

#[test]
fn deterministic_sv_accepts_cashflows_and_stochastic_rates_reject_unsupported_discounting() {
    let r = base(false);
    let request = replace(&r, ProductSpec::Graph(multiple(&r)), r.engine());
    let sv = pricing::stochastic_volatility::StochasticVolatilityPricingPlan::compile_bergomi(
        &request,
        Bergomi1Factor::new(1.0, 0.3, -0.4).unwrap(),
        0.25,
        policy(),
    )
    .unwrap();
    assert!(sv.evaluate().unwrap().value.is_finite());
    let rates = HullWhite1Factor::new(0.1, vec![0.0], vec![0.01]).unwrap();
    assert!(
        pricing::hull_white::HullWhiteEquityPricingPlan::compile_bs(
            &request,
            rates,
            0.0,
            0.25,
            policy()
        )
        .is_err()
    );
}

#[test]
fn source_graph_round_trip_is_order_independent_and_cashflows_change_identity() {
    let r = base(false);
    let graph = vanilla(&r);
    let mut v: Value = serde_json::from_str(&product_graph_to_json(&graph).unwrap()).unwrap();
    v["product"]["nodes"].as_array_mut().unwrap().reverse();
    assert_eq!(
        fingerprint_product_graph(&graph).unwrap(),
        fingerprint_product_graph(&parse(&v).unwrap()).unwrap()
    );
    assert_eq!(
        product_graph_to_json(&graph).unwrap(),
        product_graph_to_json(&parse(&v).unwrap()).unwrap()
    );
    v["product"]["cashflows"][0]["payment_date"] = json!("2028-01-01");
    assert_ne!(
        fingerprint_product_graph(&graph).unwrap(),
        fingerprint_product_graph(&parse(&v).unwrap()).unwrap()
    );
    let native =
        pricing::product::multi_asset::MultiAssetProduct::from_product_graph(graph.clone());
    assert_eq!(native.to_product_graph().unwrap(), graph);
}

#[test]
fn malformed_and_anticipative_graphs_are_rejected() {
    let base: Value =
        serde_json::from_str(&product_graph_to_json(&vanilla(&base(false))).unwrap()).unwrap();
    for case in 0..9 {
        let mut v = base.clone();
        match case {
            0 => v["product"]["nodes"][0]["ignored"] = json!(1),
            1 => v["product"]["nodes"][0]["operation"]["type"] = json!("unknown"),
            2 => v["product"]["cashflows"][0]["output_node"] = json!(99999),
            3 => v["product"]["nodes"][0] = json!({"id":0,"operation":{"type":"negate","input":0}}),
            4 => v["product"]["underlying_ids"] = json!([99]),
            5 => v["product"]["cashflows"][0]["payment_date"] = json!("2027-01-01"),
            6 => {
                let node = v["product"]["nodes"][0].clone();
                v["product"]["nodes"].as_array_mut().unwrap().push(node);
            }
            7 => v["product"]["underlying_ids"] = json!([1, 1]),
            _ => v["product"]["cashflows"] = json!([]),
        }
        assert!(parse(&v).is_err(), "case {case}");
    }
    let encoded = serde_json::to_string(&base).unwrap();
    assert!(
        parse_product_graph_json(
            encoded.as_bytes(),
            JsonLimits {
                max_input_bytes: 10,
                ..JsonLimits::DEFAULT
            }
        )
        .is_err()
    );
    let duplicate = encoded.replacen(
        "\"schema_version\":1",
        "\"schema_version\":1,\"schema_version\":1",
        1,
    );
    assert!(parse_product_graph_json(duplicate.as_bytes(), JsonLimits::DEFAULT).is_err());
}

#[test]
fn past_observations_payments_and_hard_indicator_risk_are_rejected() {
    let r = base(true);
    for (observation, payment) in [("2026-01-01", "2027-09-04"), ("2026-01-01", "2026-02-01")] {
        let node = SourceNode::new(
            NodeId::new(0),
            Op::TerminalSpot {
                underlying: UnderlyingId::new(1),
                observation_date: date(observation),
            },
        );
        let graph = GraphProduct::new(
            CurrencyId::new(2),
            vec![UnderlyingId::new(1)],
            SourceGraph::new(vec![node], vec![NodeId::new(0)]),
            vec![date(payment)],
        )
        .unwrap();
        assert!(
            PricingRequest::new(
                r.valuation_date(),
                ProductSpec::Graph(graph),
                r.market().clone(),
                r.model().clone(),
                r.engine(),
                r.risk().clone()
            )
            .is_err()
        );
    }
    let mut b = SourceGraphBuilder::new();
    let spot = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: r.product().expiry(),
        })
        .unwrap();
    let hit = b.push(Op::Indicator { input: spot }).unwrap();
    let graph = GraphProduct::new(
        CurrencyId::new(2),
        vec![UnderlyingId::new(1)],
        b.finish(vec![hit]),
        vec![r.product().expiry()],
    )
    .unwrap();
    assert!(!graph.supports_pathwise_risk());
    assert!(
        PricingRequest::new(
            r.valuation_date(),
            ProductSpec::Graph(graph),
            r.market().clone(),
            r.model().clone(),
            r.engine(),
            r.risk().clone()
        )
        .is_err()
    );
}

#[test]
fn graph_sources_replay_across_existing_model_adapters_and_legacy_result_wire() {
    let graph = vanilla(&base(false));
    let product: Value = serde_json::from_str(&product_graph_to_json(&graph).unwrap()).unwrap();
    for fixture in [
        include_str!("../../../fixtures/specification-v1/black_scholes.golden.json"),
        include_str!("../../../fixtures/specification-v1/bergomi_hull_white.golden.json"),
        include_str!("../../../fixtures/specification-v1/bergomi_lsv.golden.json"),
        include_str!("../../../fixtures/specification-v1/rough_heston.golden.json"),
    ] {
        let original = parse_specification_json(fixture.as_bytes(), JsonLimits::DEFAULT).unwrap();
        let mut value: Value = serde_json::from_str(fixture).unwrap();
        value["schema_version"] = json!(2);
        value["product"] = json!({"type": "graph", "graph": product["product"]});
        let source =
            parse_specification_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT)
                .unwrap();
        let expected = original.compile().unwrap().evaluate().unwrap();
        let actual = source.compile().unwrap().evaluate().unwrap();
        assert_eq!(actual.value(), expected.value());
        assert_eq!(actual.standard_error(), expected.standard_error());
    }
    let request = replace(&base(true), ProductSpec::Graph(graph), base(true).engine());
    let result = PricingPlan::compile(&request, policy())
        .unwrap()
        .evaluate()
        .unwrap();
    let saved = pricing::monte_carlo_result_to_json(&result).unwrap();
    assert_eq!(
        result,
        pricing::parse_monte_carlo_result_json(saved.as_bytes(), JsonLimits::DEFAULT).unwrap()
    );
    for version in 1..=3 {
        let mut value: Value = serde_json::from_str(include_str!(
            "../../../fixtures/v3/pricing_request.golden.json"
        ))
        .unwrap();
        value["schema_version"] = json!(version);
        value["product"] = json!({"type": "graph", "graph": product["product"]});
        assert!(
            parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).is_err()
        );
    }
}

#[test]
fn constant_cashflows_and_shared_output_covariance_use_each_payment_discount() {
    use pricing::core::DayCountConvention;
    use pricing::market::DiscountCurve;
    let r = base(true);
    let dates = vec![date("2027-09-04"), date("2028-03-04")];
    let discount_sum: f64 = dates
        .iter()
        .map(|date| {
            r.market()
                .equity()
                .forward()
                .discount_curve()
                .evaluate(DayCountConvention::Act365F.year_fraction(r.valuation_date(), *date))
                .unwrap()
                .discount
        })
        .sum();
    let mut b = SourceGraphBuilder::new();
    let cash = b.literal(7.0).unwrap();
    // Dead observations/indicators do not turn fixed cash into a path-dependent risk.
    let dead = b
        .push(Op::TerminalSpot {
            underlying: r.product().underlying(),
            observation_date: date("2020-01-01"),
        })
        .unwrap();
    b.push(Op::Indicator { input: dead }).unwrap();
    let graph = GraphProduct::new(
        r.product().currency(),
        vec![r.product().underlying()],
        b.finish(vec![cash, cash]),
        dates.clone(),
    )
    .unwrap();
    let actual = PricingPlan::compile(
        &replace(&r, ProductSpec::Graph(graph), r.engine()),
        policy(),
    )
    .unwrap()
    .evaluate()
    .unwrap();
    assert!((actual.pricing_result.value.value().get() - 7.0 * discount_sum).abs() < 1e-12);
    assert_eq!(actual.pricing_result.value.standard_error().get(), 0.0);
    assert_eq!(
        actual
            .pricing_result
            .risks
            .delta
            .unwrap()
            .raw()
            .value()
            .get(),
        0.0
    );
    let mut b = SourceGraphBuilder::new();
    let spot = b
        .push(Op::TerminalSpot {
            underlying: r.product().underlying(),
            observation_date: r.product().expiry(),
        })
        .unwrap();
    let make = |dates: Vec<Date>| {
        GraphProduct::new(
            r.product().currency(),
            vec![r.product().underlying()],
            b.clone().finish(vec![spot; dates.len()]),
            dates,
        )
        .unwrap()
    };
    let single = PricingPlan::compile(
        &replace(&r, ProductSpec::Graph(make(vec![dates[0]])), r.engine()),
        policy(),
    )
    .unwrap()
    .evaluate()
    .unwrap();
    let both = PricingPlan::compile(
        &replace(&r, ProductSpec::Graph(make(dates.clone())), r.engine()),
        policy(),
    )
    .unwrap()
    .evaluate()
    .unwrap();
    let first_discount = r
        .market()
        .equity()
        .forward()
        .discount_curve()
        .evaluate(1.0)
        .unwrap()
        .discount;
    let ratio = discount_sum / first_discount;
    assert!(
        (both.pricing_result.value.value().get()
            - ratio * single.pricing_result.value.value().get())
        .abs()
            < 1e-12
    );
    assert!(
        (both.pricing_result.value.standard_error().get()
            - ratio * single.pricing_result.value.standard_error().get())
        .abs()
            < 1e-12
    );
    assert!(
        (both.pricing_result.risks.delta.unwrap().raw().value().get()
            - ratio
                * single
                    .pricing_result
                    .risks
                    .delta
                    .unwrap()
                    .raw()
                    .value()
                    .get())
        .abs()
            < 1e-12
    );
}
