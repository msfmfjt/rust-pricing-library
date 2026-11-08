use pricing::contract::*;
use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::mc::ExecutionPolicy;
use pricing::product::{
    BarrierDirection as NativeDirection, BarrierMonitoring, BarrierSpec, BarrierStyle,
    GraphLimitPolicy, OptionSide, ProductSpec, SourceGraphBuilder, SourceOpcode,
};
use pricing::{
    JsonLimits, PricingOperation, PricingSpecification, PricingTask, parse_request_json,
};

fn d(s: &str) -> Date {
    s.parse().unwrap()
}
fn start() -> Date {
    d("2026-09-04")
}
fn middle() -> Date {
    d("2027-03-04")
}
fn expiry() -> Date {
    d("2027-09-04")
}
fn contract(feature: OptionFeature) -> Contract {
    InstrumentBuilder::new(
        "put",
        CurrencyId::new(2),
        UnderlyingId::new(1),
        "official.close",
    )
    .unwrap()
    .vanilla_option(OptionSide::Put, 100., 1., expiry(), feature)
    .unwrap()
}
fn bermudan() -> Contract {
    contract(OptionFeature::Bermudan {
        dates: vec![middle(), expiry()],
    })
}
fn american() -> Contract {
    contract(OptionFeature::American { start: start() })
}
fn source(lsm: bool) -> PricingSpecification {
    let bytes = if lsm {
        include_bytes!("../../../fixtures/v3/pricing_request_american.golden.json").as_slice()
    } else {
        include_bytes!("../../../fixtures/v3/pricing_request.golden.json").as_slice()
    };
    PricingSpecification::from_request(
        parse_request_json(bytes, JsonLimits::DEFAULT).unwrap(),
        ExecutionPolicy::new(2, Some(32)).unwrap(),
    )
}
fn event(id: &str, date: Date, event: Event) -> LifecycleEvent {
    LifecycleEvent {
        id: id.into(),
        effective_date: date,
        event,
    }
}

#[test]
fn intermediate_settlement_does_not_require_unrelated_monitoring_data_early() {
    let option = contract(OptionFeature::ContinuousBarrier(terms(
        BarrierDirection::Up,
        BarrierStyle::KnockOut,
    )));
    let fixed = Contract::new(
        "fixed",
        vec![],
        Cashflow::new("fee", CurrencyId::new(2), 7.0.into(), middle(), middle()).into(),
    )
    .unwrap();
    let bundle = Contract::bundle("portfolio", vec![option, fixed]).unwrap();
    let paid = event(
        "paid",
        middle(),
        Event::Settlement {
            cashflow_id: "fee".into(),
        },
    );
    let later = middle().next_day().unwrap();
    let status = event(
        "monitoring",
        later,
        Event::BarrierStatus {
            barrier_id: "put/option".into(),
            hit: false,
        },
    );
    let result = bundle.advance(&[paid, status], later).unwrap();
    assert!(result.outstanding_payments.is_empty());
    assert_eq!(result.determined_cashflows[0].id, "fee");
    assert!(result.determined_cashflows[0].settled);
}

#[test]
fn analysis_exposes_legal_opportunities_and_mutually_exclusive_cashflows() {
    let analysis = bermudan().analyze().unwrap();
    assert_eq!(analysis.option_features[0].expiry, expiry());
    assert_eq!(
        analysis
            .potential_cashflows
            .iter()
            .map(|c| c.payment_date)
            .collect::<Vec<_>>(),
        [middle(), expiry()]
    );
    assert!(analysis.determined_cashflows.is_empty());
    assert_eq!(
        analysis.potential_cashflows[0].id,
        analysis.potential_cashflows[1].id
    );
    let rolled = american().advance(&[], middle()).unwrap().residual_contract;
    assert_eq!(
        rolled.analyze().unwrap().option_features[0].feature,
        OptionFeature::American { start: middle() }
    );
}

#[test]
fn exercise_rejects_inactive_rights_and_mismatched_off_grid_sources() {
    let right = american();
    let day = middle();
    let expected = right.exercise_observation("put/option", day).unwrap();
    let mut foreign = expected.clone();
    foreign.fixing_source = "different.close".into();
    let mut registry = right.observations().to_vec();
    registry.push(foreign.clone());
    assert!(Contract::new("bad", registry, right.root().clone()).is_err());
    let notice = event(
        "notice",
        day,
        Event::Exercise {
            exercise_id: "put/option".into(),
        },
    );
    let fix = |o| {
        event(
            "fix",
            day,
            Event::Fixing {
                observation: o,
                value: 90.,
            },
        )
    };
    assert!(right.advance(&[notice.clone(), fix(foreign)], day).is_err());
    let gated = Contract::new(
        "gated",
        right.observations().to_vec(),
        Term::conditional(false.into(), right.root().clone(), Term::Empty),
    )
    .unwrap();
    assert!(gated.advance(&[notice, fix(expected)], day).is_err());
}

#[test]
fn confirmed_rebate_remains_payable_after_subsequent_termination() {
    let option = contract(OptionFeature::ContinuousBarrier(terms(
        BarrierDirection::Up,
        BarrierStyle::KnockOut,
    )));
    let hit = event(
        "hit",
        middle(),
        Event::BarrierStatus {
            barrier_id: "put/option".into(),
            hit: true,
        },
    );
    let later = middle().next_day().unwrap();
    let another = event(
        "also-hit",
        later,
        Event::BarrierStatus {
            barrier_id: "put/option".into(),
            hit: true,
        },
    );
    let terminated = event(
        "terminate",
        later,
        Event::Termination {
            contract_id: "put".into(),
        },
    );
    let result = option.advance(&[another, terminated, hit], later).unwrap();
    assert_eq!(result.outstanding_payments.len(), 1);
    assert_eq!(result.outstanding_payments[0].determination_date, middle());
    assert_eq!(result.outstanding_payments[0].amount, 2.);
    assert_eq!(result.outstanding_payments[0].payment_date, expiry());
}
fn terms(direction: BarrierDirection, style: BarrierStyle) -> ContinuousBarrierTerms {
    ContinuousBarrierTerms {
        start: start(),
        end: expiry(),
        payment_date: expiry(),
        level: if direction == BarrierDirection::Up {
            115.
        } else {
            85.
        },
        direction,
        style,
        rebate: 2.,
    }
}
#[test]
fn lsm_uses_native_out_of_sample_policy_and_explicit_american_grid() {
    let template = source(true);
    let native = PricingTask::new(template.clone(), PricingOperation::Evaluate).unwrap();
    let reference = native.compile().unwrap().evaluate().unwrap();
    let dates = vec![middle(), expiry()];
    let settings = AdvancedMcSettings {
        american_grid: Some(dates.clone()),
        ..Default::default()
    };
    for (option, config) in [
        (bermudan(), AdvancedMcSettings::default()),
        (american(), settings.clone()),
    ] {
        let task = option
            .pricing_task(&template, PricingOperation::Evaluate, &config)
            .unwrap();
        assert_eq!(task.fingerprint().unwrap(), native.fingerprint().unwrap());
        assert_eq!(task.compile().unwrap().evaluate().unwrap(), reference);
        let wire = option.to_json().unwrap();
        assert_eq!(
            Contract::from_json(wire.as_bytes(), JsonLimits::DEFAULT).unwrap(),
            option
        );
        assert!(
            Contract::from_json(
                wire.replace("\"schema_version\":2", "\"schema_version\":1")
                    .as_bytes(),
                JsonLimits::DEFAULT
            )
            .is_err()
        );
    }
    assert!(
        american()
            .pricing_task(&template, PricingOperation::Evaluate, &Default::default())
            .is_err()
    );
    assert!(
        bermudan()
            .pricing_task(&template, PricingOperation::Evaluate, &settings)
            .is_err()
    );
    for grid in [
        vec![expiry(), middle()],
        vec![middle()],
        vec![middle(), middle(), expiry()],
        vec![d("2026-01-01"), expiry()],
    ] {
        assert!(
            american()
                .pricing_task(
                    &template,
                    PricingOperation::Evaluate,
                    &AdvancedMcSettings {
                        american_grid: Some(grid),
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }
}
#[test]
fn exercise_replay_uses_the_actual_notice_and_fixing_on_or_off_the_pricing_grid() {
    let off_grid = d("2027-01-13");
    for (option, date) in [(bermudan(), middle()), (american(), off_grid)] {
        let observation = option.exercise_observation("put/option", date).unwrap();
        let fix = event(
            "fix",
            date,
            Event::Fixing {
                observation,
                value: 85.,
            },
        );
        let notice = event(
            "notice",
            date,
            Event::Exercise {
                exercise_id: "put/option".into(),
            },
        );
        let events = vec![notice.clone(), fix.clone(), notice.clone()];
        let prior = option.advance(&events, start()).unwrap();
        assert!(prior.applied_event_ids.is_empty());
        assert!(prior.determined_cashflows.is_empty());
        let result = prior.advance(&[], date).unwrap();
        let cash = &result.outstanding_payments[0];
        assert_eq!(
            (cash.amount, cash.determination_date, cash.payment_date),
            (15., date, date)
        );
        assert_eq!(result, option.advance(&events, date).unwrap());
        assert_eq!(
            LifecycleResult::from_json(result.to_json().unwrap().as_bytes(), JsonLimits::DEFAULT)
                .unwrap(),
            result
        );
        let settled = event(
            "paid",
            date,
            Event::Settlement {
                cashflow_id: "put/option".into(),
            },
        );
        assert!(
            result
                .advance(&[settled], expiry())
                .unwrap()
                .outstanding_payments
                .is_empty()
        );
        assert!(option.advance(&[notice], date).is_err());
        assert!(
            result
                .advance(
                    &[event(
                        "second",
                        expiry(),
                        Event::Exercise {
                            exercise_id: "put/option".into()
                        }
                    )],
                    expiry()
                )
                .is_err()
        );
    }
    assert!(
        bermudan()
            .exercise_observation("put/option", off_grid)
            .is_err()
    );
    let remaining = bermudan()
        .advance(&[], middle().next_day().unwrap())
        .unwrap();
    assert_eq!(remaining.remaining_observations.len(), 1);
    assert_eq!(remaining.remaining_observations[0].date, expiry());
}
#[test]
fn bridge_styles_and_directions_route_through_native_model_correction() {
    let mut raw: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    raw["risk"]["delta"] = false.into();
    raw["risk"]["vega"] = false.into();
    let template = PricingSpecification::from_request(
        parse_request_json(&serde_json::to_vec(&raw).unwrap(), JsonLimits::DEFAULT).unwrap(),
        ExecutionPolicy::new(2, Some(32)).unwrap(),
    );
    let config = AdvancedMcSettings {
        brownian_bridge: true,
        ..Default::default()
    };
    for (direction, native_direction) in [
        (BarrierDirection::Up, NativeDirection::Up),
        (BarrierDirection::Down, NativeDirection::Down),
    ] {
        for style in [BarrierStyle::KnockIn, BarrierStyle::KnockOut] {
            let b = terms(direction, style);
            let option = contract(OptionFeature::ContinuousBarrier(b.clone()));
            assert!(
                option
                    .pricing_task(&template, PricingOperation::Evaluate, &Default::default())
                    .is_err()
            );
            let task = option
                .pricing_task(&template, PricingOperation::Evaluate, &config)
                .unwrap();
            let product = BarrierSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                expiry(),
                100.,
                b.level,
                1.,
                OptionSide::Put,
                native_direction,
                style,
                BarrierMonitoring::Continuous,
                vec![expiry()],
                Some(2.),
                expiry(),
            )
            .unwrap();
            let native = PricingSpecification::new(
                start(),
                ProductSpec::Barrier(product),
                template.market().clone(),
                template.model().clone(),
                template.engine(),
                template.risk().clone(),
                template.numerics().clone(),
            )
            .unwrap();
            let expected = PricingTask::new(native, PricingOperation::Evaluate)
                .unwrap()
                .compile()
                .unwrap()
                .evaluate()
                .unwrap();
            assert_eq!(task.compile().unwrap().evaluate().unwrap(), expected);
        }
    }
}
#[test]
fn continuous_monitoring_requires_published_history_and_touch_is_absorbing() {
    let option = contract(OptionFeature::ContinuousBarrier(terms(
        BarrierDirection::Down,
        BarrierStyle::KnockOut,
    )));
    let status = |id: &str, date: Date, hit| {
        event(
            id,
            date,
            Event::BarrierStatus {
                barrier_id: "put/option".into(),
                hit,
            },
        )
    };
    assert!(matches!(
        option.advance(&[], middle()),
        Err(ContractError::MissingFixing(_))
    ));
    let clear = option
        .advance(&[status("clear", middle(), false)], middle())
        .unwrap();
    assert!(!clear.barrier_states["put/option"]);
    assert!(clear.advance(&[], middle().next_day().unwrap()).is_err());
    let hit_date = middle().next_day().unwrap();
    let hit = clear
        .advance(&[status("hit", hit_date, true)], hit_date)
        .unwrap();
    assert_eq!(hit.outstanding_payments[0].amount, 2.);
    assert_eq!(hit.outstanding_payments[0].determination_date, hit_date);
    assert!(hit.remaining_observations.is_empty());
    assert!(
        hit.advance(&[status("contradiction", expiry(), false)], expiry())
            .is_err()
    );
    let touch = event(
        "terminal",
        expiry(),
        Event::Fixing {
            observation: option.observations()[0].clone(),
            value: 85.,
        },
    );
    assert_eq!(
        option
            .advance(std::slice::from_ref(&touch), expiry())
            .unwrap()
            .outstanding_payments[0]
            .amount,
        2.
    );
    assert!(
        option
            .advance(&[touch, status("false", expiry(), false)], expiry())
            .is_err()
    );
}

#[test]
fn forward_liveness_preserves_shared_outputs_adjoints_and_reused_workspaces() {
    let mut builder = SourceGraphBuilder::new();
    let spot = builder
        .push(SourceOpcode::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: expiry(),
        })
        .unwrap();
    let one = builder.literal(1.).unwrap();
    let mut sum = spot;
    for _ in 0..128 {
        sum = builder
            .push(SourceOpcode::Add {
                left: sum,
                right: one,
            })
            .unwrap();
    }
    let graph = builder.finish(vec![sum, spot, sum]);
    let standard = graph.compile(GraphLimitPolicy::DEFAULT).unwrap();
    let program = graph.compile_compact(GraphLimitPolicy::DEFAULT).unwrap();
    assert_eq!(standard.tape_fingerprint(), program.tape_fingerprint());
    assert!(standard.workspace().allocated_bytes() > program.workspace().allocated_bytes());
    assert_eq!(standard.evaluation_program_bytes(), 0);
    assert!(program.forward_value_slots() <= 3);
    assert!(program.opcodes().len() > 128);
    let mut workspace = program.workspace();
    let bytes = workspace.allocated_bytes();
    for s in [0.25, 80., 103.] {
        assert_eq!(
            program
                .evaluate_reusing(&mut workspace, |_, _| Some(s), |_, _| None)
                .unwrap(),
            [s + 128., s, s + 128.]
        );
        assert_eq!(
            program.evaluate(|_, _| Some(s)).unwrap(),
            [s + 128., s, s + 128.]
        );
        let reverse = program
            .evaluate_output_with_observation_adjoints(0, |_, _| Some(s), |_, _| None)
            .unwrap();
        assert_eq!(reverse.terminal_adjoints[0].value, 1.);
        assert_eq!(workspace.allocated_bytes(), bytes);
    }
}
