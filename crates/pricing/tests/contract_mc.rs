use pricing::contract::*;
use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::market::DiscountCurve;
use pricing::mc::ExecutionPolicy;
use pricing::product::{
    GraphComparison, GraphLimitPolicy, ProductSpec, SourceGraphBuilder, SourceOpcode as Op,
};
use pricing::{
    JsonLimits, PricingOperation, PricingPlan, PricingRequest, PricingSpecification,
    parse_pricing_task_json, parse_product_graph_json, parse_request_json, pricing_task_to_json,
    product_graph_to_json,
};
use serde_json::{Value, json};
fn d(s: &str) -> Date {
    s.parse().unwrap()
}
fn base(risk: bool) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"]["independent_sampling_units"] = json!(256);
    v["risk"]["delta"] = json!(risk);
    v["risk"]["vega"] = json!(risk);
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(2, Some(32)).unwrap()
}
fn source(e: NumericExpr) -> Contract {
    Contract::new(
        "payoff",
        vec![Observation::spot(
            "spot",
            UnderlyingId::new(1),
            d("2027-09-04"),
            "close",
        )],
        Cashflow::new(
            "pay",
            CurrencyId::new(2),
            e,
            d("2027-09-04"),
            d("2027-09-06"),
        )
        .into(),
    )
    .unwrap()
}
#[test]
fn native_lazy_select_does_not_observe_or_evaluate_inactive_branch() {
    let mut b = SourceGraphBuilder::new();
    let s = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: d("2027-01-01"),
        })
        .unwrap();
    let zero = b.literal(0.0).unwrap();
    let one = b.literal(1.0).unwrap();
    let positive = b
        .push(Op::Compare {
            left: s,
            right: zero,
            operation: GraphComparison::Greater,
        })
        .unwrap();
    let invalid = b
        .push(Op::Divide {
            numerator: one,
            denominator: zero,
        })
        .unwrap();
    let select = b
        .push(Op::Select {
            condition: positive,
            then_value: s,
            else_value: invalid,
        })
        .unwrap();
    let p = b
        .finish(vec![select])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
    assert_eq!(p.evaluate(|_, _| Some(3.0)).unwrap(), vec![3.0]);
    assert!(p.evaluate(|_, _| Some(-1.0)).is_err());
    let reverse = p
        .evaluate_single_with_terminal_adjoint(|_, _| Some(3.0))
        .unwrap();
    assert_eq!(reverse.value, 3.0);
    assert_eq!(reverse.terminal_adjoints[0].value, 1.0);
    let mut workspace = p.workspace();
    for value in [2.0, 5.0, 1.0] {
        assert_eq!(
            p.evaluate_reusing(&mut workspace, |_, _| Some(value), |_, _| None)
                .unwrap(),
            &[value]
        );
    }
}
#[test]
fn contract_condition_comparisons_aggregates_and_cse_match_semantics() {
    let s = NumericExpr::observation("spot");
    for compare in [
        Comparison::Less,
        Comparison::LessEqual,
        Comparison::Equal,
        Comparison::NotEqual,
        Comparison::GreaterEqual,
        Comparison::Greater,
    ] {
        let e = NumericExpr::select(
            s.clone().compare(compare, 100.0.into()),
            7.0.into(),
            3.0.into(),
        );
        let c = source(e.clone());
        let p = c
            .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
            .unwrap();
        for x in [99.0, 100.0, 101.0] {
            let expected = e
                .evaluate(&std::collections::BTreeMap::from([("spot".into(), x)]))
                .unwrap();
            assert_eq!(
                p.groups()[0].payoff().evaluate(|_, _| Some(x)).unwrap(),
                vec![expected]
            );
        }
    }
    let e = NumericExpr::Aggregate {
        operation: Aggregate::GeometricAverage,
        values: vec![s.clone(), 4.0.into()],
    };
    let c = source(e);
    let p = c
        .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
        .unwrap();
    assert!((p.groups()[0].payoff().evaluate(|_, _| Some(16.0)).unwrap()[0] - 8.0).abs() < 1e-13);
    let adj = p.groups()[0]
        .payoff()
        .evaluate_single_with_terminal_adjoint(|_, _| Some(16.0))
        .unwrap();
    assert!((adj.terminal_adjoints[0].value - 0.25).abs() < 1e-14);
    let e = s.clone().plus(s);
    let p = source(e)
        .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
        .unwrap();
    assert_eq!(p.groups()[0].payoff().terminal_observations().len(), 1);
    assert_eq!(
        p.groups()[0]
            .payoff()
            .opcodes()
            .iter()
            .filter(|op| matches!(op, pricing::product::CompiledOpcode::TerminalSpot { .. }))
            .count(),
        1
    );
}
#[test]
fn vanilla_reuses_native_mc_prices_greeks_and_aad() {
    let r = base(true);
    let p = r.product();
    let native = match p {
        ProductSpec::EuropeanVanilla(v) => v,
        _ => panic!(),
    };
    let o = Observation::spot("s", p.underlying(), p.expiry(), "close");
    let amount = NumericExpr::observation("s")
        .minus(native.strike().get().into())
        .maximum(0.0.into())
        .times(native.notional().get().into());
    let c = Contract::new(
        "vanilla",
        vec![o],
        Cashflow::new("pay", p.currency(), amount, p.expiry(), p.payment_date()).into(),
    )
    .unwrap();
    let lowered = c
        .compile_mc(r.valuation_date(), vec![p.underlying()])
        .unwrap();
    let template = PricingSpecification::from_request(r.clone(), policy());
    let result = lowered
        .pricing_task(&template, PricingOperation::Evaluate)
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap();
    let expected = PricingPlan::compile(&r, policy())
        .unwrap()
        .evaluate()
        .unwrap();
    assert_eq!(
        result.report.value(),
        expected.pricing_result.value.value().get()
    );
    assert_eq!(
        result.report.standard_error(),
        expected.pricing_result.value.standard_error().get()
    );
    assert!(!result.report.risks().is_empty());
    let graph = lowered.groups()[0].product().clone();
    let request = PricingRequest::new(
        r.valuation_date(),
        ProductSpec::Graph(graph),
        r.market().clone(),
        r.model().clone(),
        r.engine(),
        r.risk().clone(),
    )
    .unwrap();
    let actual = PricingPlan::compile(&request, policy())
        .unwrap()
        .evaluate()
        .unwrap();
    assert_eq!(actual.pricing_result.risks, expected.pricing_result.risks);
}
#[test]
fn graph_and_task_extensions_use_new_versions_without_reinterpreting_v1() {
    let c = source(NumericExpr::select(
        NumericExpr::observation("spot").compare(Comparison::GreaterEqual, 100.0.into()),
        1.0.into(),
        0.0.into(),
    ));
    let compiled = c
        .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
        .unwrap();
    let graph = compiled.groups()[0].product();
    let json = product_graph_to_json(graph).unwrap();
    let mut v: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["schema_version"], 2);
    assert_eq!(
        &parse_product_graph_json(json.as_bytes(), JsonLimits::DEFAULT).unwrap(),
        graph
    );
    v["schema_version"] = json!(1);
    assert!(
        parse_product_graph_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).is_err()
    );
    let r = base(false);
    let template = PricingSpecification::from_request(r, policy());
    let task = compiled
        .pricing_task(&template, PricingOperation::Evaluate)
        .unwrap();
    let text = pricing_task_to_json(&task).unwrap();
    let mut v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["schema_version"], 2);
    assert_eq!(v["source"]["specification"]["schema_version"], 3);
    let restored = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(task.fingerprint().unwrap(), restored.fingerprint().unwrap());
    assert_eq!(
        task.compile().unwrap().evaluate().unwrap(),
        restored.compile().unwrap().evaluate().unwrap()
    );
    v["schema_version"] = json!(1);
    assert!(
        parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).is_err()
    );
}
#[test]
fn missing_history_unsupported_model_observation_and_currency_mixing_fail_explicitly() {
    let c = source(NumericExpr::observation("spot"));
    assert!(matches!(
        c.compile_mc(d("2028-01-01"), vec![UnderlyingId::new(1)]),
        Err(ContractError::Unsupported(_)) | Err(ContractError::MissingFixing(_))
    ));
    assert!(matches!(
        c.compile_mc(d("2026-09-04"), vec![UnderlyingId::new(2)]),
        Err(ContractError::Unsupported(_))
    ));
    let mut o = c.observations()[0].clone();
    o.kind = ObservationKind::Rate;
    let c = Contract::new("rates", vec![o], c.root().clone()).unwrap();
    assert!(matches!(
        c.compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)]),
        Err(ContractError::Unsupported(_))
    ));
    let c = Contract::new(
        "mixed",
        vec![],
        Term::bundle(vec![
            Cashflow::new(
                "usd",
                CurrencyId::new(2),
                10.0.into(),
                d("2027-09-04"),
                d("2027-09-06"),
            )
            .into(),
            Cashflow::new(
                "eur",
                CurrencyId::new(3),
                20.0.into(),
                d("2027-09-04"),
                d("2027-09-06"),
            )
            .into(),
        ]),
    )
    .unwrap();
    let p = c
        .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
        .unwrap();
    assert_eq!(p.groups().len(), 2);
    assert_eq!(
        p.groups()[0].payoff().evaluate(|_, _| None).unwrap(),
        vec![10.0]
    );
    assert_eq!(
        p.groups()[1].payoff().evaluate(|_, _| None).unwrap(),
        vec![20.0]
    );
}

#[test]
fn reduction_preserves_a_future_lazy_branch_and_prunes_it_after_fixing() {
    let o = Observation::spot("spot", UnderlyingId::new(1), d("2027-09-04"), "close");
    let e = NumericExpr::select(
        NumericExpr::observation("spot").compare(Comparison::Greater, 0.0.into()),
        7.0.into(),
        NumericExpr::from(1.0).divided_by(0.0.into()),
    );
    let c = source(e);
    let r = c.advance(&[], d("2026-09-04")).unwrap();
    let compiled = r
        .residual_contract
        .compile_mc(r.as_of(), vec![UnderlyingId::new(1)])
        .unwrap();
    assert_eq!(
        compiled.groups()[0]
            .payoff()
            .evaluate(|_, _| Some(1.0))
            .unwrap(),
        vec![7.0]
    );
    assert!(
        compiled.groups()[0]
            .payoff()
            .evaluate(|_, _| Some(-1.0))
            .is_err()
    );
    let event = LifecycleEvent {
        id: "fix".into(),
        effective_date: o.date,
        event: Event::Fixing {
            observation: o,
            value: 1.0,
        },
    };
    assert_eq!(
        r.advance(&[event], d("2027-09-04"))
            .unwrap()
            .outstanding_payments[0]
            .amount,
        7.0
    );
}

#[test]
fn early_redemption_does_not_read_later_nested_decisions() {
    let first = d("2027-03-01");
    let last = d("2027-09-04");
    let currency = CurrencyId::new(2);
    let early = Cashflow::new("early", currency, 103.0.into(), first, first).into();
    let later = Cashflow::new("later", currency, 106.0.into(), last, last).into();
    let loss = Cashflow::new(
        "loss",
        currency,
        NumericExpr::observation("last"),
        last,
        last,
    )
    .into();
    let root = Term::conditional(
        NumericExpr::observation("first").compare(Comparison::GreaterEqual, 100.0.into()),
        early,
        Term::conditional(
            NumericExpr::observation("last").compare(Comparison::GreaterEqual, 100.0.into()),
            later,
            loss,
        ),
    );
    let c = Contract::new(
        "autocall",
        vec![
            Observation::spot("first", UnderlyingId::new(1), first, "close"),
            Observation::spot("last", UnderlyingId::new(1), last, "close"),
        ],
        root,
    )
    .unwrap();
    let p = c
        .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
        .unwrap();
    assert_eq!(
        p.groups()[0]
            .payoff()
            .evaluate(|_, date| if date == first { Some(100.0) } else { None })
            .unwrap(),
        vec![103.0, 0.0, 0.0]
    );
}

#[test]
fn separate_payment_discounts_and_stochastic_rate_capability_are_native() {
    let r = base(false);
    let currency = r.product().currency();
    let c = Contract::new(
        "fixed",
        vec![],
        Term::bundle(vec![
            Cashflow::new("a", currency, 10.0.into(), d("2027-03-04"), d("2027-03-04")).into(),
            Cashflow::new("b", currency, 20.0.into(), d("2028-03-04"), d("2028-03-04")).into(),
        ]),
    )
    .unwrap();
    let compiled = c
        .compile_mc(r.valuation_date(), vec![r.product().underlying()])
        .unwrap();
    let template = PricingSpecification::from_request(r.clone(), policy());
    let result = compiled
        .pricing_task(&template, PricingOperation::Evaluate)
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap();
    let expected = [(10.0, d("2027-03-04")), (20.0, d("2028-03-04"))]
        .into_iter()
        .map(|(amount, date)| {
            amount
                * r.market()
                    .equity()
                    .forward()
                    .discount_curve()
                    .evaluate(
                        pricing::core::DayCountConvention::Act365F
                            .year_fraction(r.valuation_date(), date),
                    )
                    .unwrap()
                    .discount
        })
        .sum::<f64>();
    assert!((result.report.value() - expected).abs() < 1e-12);
    assert_eq!(result.report.standard_error(), 0.0);
    let hw = pricing::parse_specification_json(
        include_bytes!("../../../fixtures/specification-v1/bergomi_hull_white.golden.json"),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    let task = compiled
        .pricing_task(&hw, PricingOperation::Evaluate)
        .unwrap();
    assert!(task.compile().is_err());
}
#[test]
fn correlated_multi_asset_contract_uses_existing_model_and_aad() {
    let template = pricing::parse_pricing_task_json(
        include_bytes!("../../../fixtures/task-v1/multi_asset_bs.golden.json"),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    let pricing::PricingSource::MultiAsset(s) = template.source() else {
        panic!()
    };
    let ids = s.product().underlyings();
    let expiry = s.product().expiry();
    let observations = ids
        .iter()
        .enumerate()
        .map(|(i, &u)| Observation::spot(format!("s{i}"), u, expiry, "close"))
        .collect();
    let amount = NumericExpr::observation("s0").plus(NumericExpr::observation("s1"));
    let c = Contract::new(
        "basket",
        observations,
        Cashflow::new(
            "sum",
            s.product().currency(),
            amount,
            expiry,
            s.product().last_payment_date(),
        )
        .into(),
    )
    .unwrap();
    let compiled = c.compile_mc(s.valuation_date(), ids.to_vec()).unwrap();
    assert_eq!(
        compiled.groups()[0]
            .payoff()
            .evaluate(|u, _| Some(if u == ids[0] { 90.0 } else { 110.0 }))
            .unwrap(),
        vec![200.0]
    );
    let task = compiled
        .multi_asset_task(
            s,
            PricingOperation::Aad {
                gamma_relative_bump: None,
            },
        )
        .unwrap();
    let result = task.compile().unwrap().evaluate().unwrap();
    assert!(result.report.value() > 0.0);
    assert!(!result.report.risks().is_empty());
}
#[test]
fn hard_contract_predicates_reject_pathwise_greeks() {
    let c = source(NumericExpr::select(
        NumericExpr::observation("spot").compare(Comparison::GreaterEqual, 100.0.into()),
        1.0.into(),
        0.0.into(),
    ));
    let compiled = c
        .compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
        .unwrap();
    let template = PricingSpecification::from_request(base(true), policy());
    match compiled.pricing_task(&template, PricingOperation::Evaluate) {
        Err(_) => {}
        Ok(task) => assert!(task.compile().is_err()),
    }
}

#[test]
fn folded_select_keeps_invalid_inactive_nodes_unexecuted() {
    let mut b = SourceGraphBuilder::new();
    let one = b.literal(1.0).unwrap();
    let zero = b.literal(0.0).unwrap();
    let bad = b
        .push(Op::Divide {
            numerator: one,
            denominator: zero,
        })
        .unwrap();
    let selected = b
        .push(Op::Select {
            condition: one,
            then_value: one,
            else_value: bad,
        })
        .unwrap();
    let p = b
        .finish(vec![selected])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
    assert_eq!(p.evaluate(|_, _| None).unwrap(), [1.0]);
    assert_eq!(
        p.evaluate_single_with_terminal_adjoint(|_, _| None)
            .unwrap()
            .value,
        1.0
    );
}

#[test]
fn reused_workspace_reprepares_constants_when_switching_between_tapes() {
    fn payoff(payout: f64) -> pricing::product::CompiledPayoff {
        let mut b = SourceGraphBuilder::new();
        let observed = b
            .push(Op::TerminalSpot {
                underlying: UnderlyingId::new(1),
                observation_date: d("2027-01-01"),
            })
            .unwrap();
        let zero = b.literal(0.0).unwrap();
        let cash = b.literal(payout).unwrap();
        let output = b
            .push(Op::Select {
                condition: observed,
                then_value: cash,
                else_value: zero,
            })
            .unwrap();
        b.finish(vec![output])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap()
    }
    let a = payoff(3.0);
    let b = payoff(-7.0);
    let mut graph = SourceGraphBuilder::new();
    let x = graph.literal(99.0).unwrap();
    let plain = graph
        .finish(vec![x])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
    let mut workspace = a.workspace();
    for p in [&a, &b, &plain, &b, &a] {
        assert_eq!(
            p.evaluate_reusing(&mut workspace, |_, _| Some(1.0), |_, _| None)
                .unwrap(),
            p.evaluate(|_, _| Some(1.0)).unwrap()
        );
    }
    assert!(
        a.evaluate_reusing(&mut workspace, |_, _| None, |_, _| None)
            .is_err()
    );
    assert_eq!(
        a.evaluate_reusing(&mut workspace, |_, _| Some(0.0), |_, _| None)
            .unwrap(),
        [0.0]
    );
}
