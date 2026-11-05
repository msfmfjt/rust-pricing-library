use pricing::JsonLimits;
use pricing::contract::*;
use pricing::core::{CurrencyId, Date, UnderlyingId};
use std::collections::BTreeMap;
fn d(s: &str) -> Date {
    s.parse().unwrap()
}
fn obs(id: &str, date: &str) -> Observation {
    Observation::spot(id, UnderlyingId::new(1), d(date), "exchange.close")
}
fn cf(id: &str, amount: NumericExpr, date: &str) -> Term {
    Cashflow::new(id, CurrencyId::new(2), amount, d(date), d("2027-06-10")).into()
}
fn event(id: &str, date: &str, event: Event) -> LifecycleEvent {
    LifecycleEvent {
        id: id.into(),
        effective_date: d(date),
        event,
    }
}
fn fixing(id: &str, o: &Observation, value: f64) -> LifecycleEvent {
    LifecycleEvent {
        id: id.into(),
        effective_date: o.date,
        event: Event::Fixing {
            observation: o.clone(),
            value,
        },
    }
}
fn two() -> Contract {
    Contract::new(
        "two",
        vec![obs("first", "2027-03-01"), obs("last", "2027-06-01")],
        Term::bundle(vec![
            cf("coupon", 3.0.into(), "2027-03-01"),
            cf(
                "payoff",
                NumericExpr::observation("first").plus(NumericExpr::observation("last")),
                "2027-06-01",
            ),
        ]),
    )
    .unwrap()
}
#[test]
fn replay_is_idempotent_order_independent_and_checkpoint_recomputes() {
    let c = two();
    let first = fixing("fix-first", &c.observations()[0], 80.0);
    let r = c
        .advance(&[first.clone(), first.clone()], d("2027-03-02"))
        .unwrap();
    assert_eq!(r.applied_event_ids, vec!["fix-first"]);
    assert_eq!(r.outstanding_payments.len(), 1);
    assert_eq!(r.remaining_observations.len(), 1);
    let last = fixing("fix-last", &c.observations()[1], 110.0);
    let complete = r.advance(&[last.clone()], d("2027-06-01")).unwrap();
    let replay = c.advance(&[last, first], d("2027-06-01")).unwrap();
    assert_eq!(complete, replay);
    assert_eq!(complete.outstanding_payments[1].amount, 190.0);
    assert_eq!(complete.advance(&[], d("2027-06-01")).unwrap(), complete);
    assert_eq!(
        LifecycleResult::from_json(complete.to_json().unwrap().as_bytes(), JsonLimits::DEFAULT)
            .unwrap(),
        complete
    );
    assert!(r.advance(&[], d("2027-03-01")).is_err());
}
#[test]
fn missing_history_and_wrong_source_never_use_spot() {
    let c = two();
    assert!(matches!(
        c.advance(&[], d("2027-03-02")),
        Err(ContractError::MissingFixing(_))
    ));
    let mut wrong = c.observations()[0].clone();
    wrong.fixing_source = "other.close".into();
    assert!(matches!(
        c.advance(&[fixing("bad", &wrong, 100.0)], d("2027-03-02")),
        Err(ContractError::EventConflict(_))
    ));
    let a = fixing("same-id", &c.observations()[0], 100.0);
    let b = fixing("same-id", &c.observations()[0], 101.0);
    assert!(c.advance(&[a, b], d("2027-03-02")).is_err());
    assert!(
        c.advance(
            &[
                fixing("a", &c.observations()[0], 100.0),
                fixing("b", &c.observations()[0], 101.0)
            ],
            d("2027-03-02")
        )
        .is_err()
    );
}
#[test]
fn future_fixings_are_not_applied_early_and_cash_is_not_settled_by_time() {
    let c = two();
    let events = [
        fixing("first", &c.observations()[0], 90.0),
        fixing("last", &c.observations()[1], 100.0),
    ];
    let r = c.advance(&events, d("2027-02-01")).unwrap();
    assert!(r.applied_event_ids.is_empty());
    assert_eq!(r.remaining_observations.len(), 2);
    let r = r.advance(&[], d("2027-06-11")).unwrap();
    assert_eq!(r.outstanding_payments.len(), 2);
    let settled = r
        .advance(
            &[event(
                "settle",
                "2027-06-10",
                Event::Settlement {
                    cashflow_id: "payoff".into(),
                },
            )],
            d("2027-06-11"),
        )
        .unwrap();
    assert_eq!(settled.outstanding_payments.len(), 1);
    assert!(settled.determined_cashflows[1].settled);
    assert_eq!(
        settled
            .residual_contract
            .analyze()
            .unwrap()
            .potential_cashflows
            .len(),
        1
    );
}
#[test]
fn same_day_order_is_fixing_then_confirmation_then_settlement() {
    let o = obs("close", "2027-06-10");
    let c = Contract::new(
        "spot",
        vec![o.clone()],
        cf("pay", NumericExpr::observation("close"), "2027-06-10"),
    )
    .unwrap();
    let settle = event(
        "a-settle",
        "2027-06-10",
        Event::Settlement {
            cashflow_id: "pay".into(),
        },
    );
    let confirm = event(
        "b-confirm",
        "2027-06-10",
        Event::Determination {
            cashflow_id: "pay".into(),
            currency: CurrencyId::new(2),
            amount: 101.0,
        },
    );
    let f = fixing("z-fix", &o, 101.0);
    let r = c
        .advance(&[settle.clone(), confirm, f.clone()], d("2027-06-10"))
        .unwrap();
    assert!(r.outstanding_payments.is_empty());
    assert!(matches!(r.residual_contract.root(), Term::Empty));
    let mut late = f;
    late.effective_date = d("2027-06-11");
    assert!(c.advance(&[settle, late], d("2027-06-12")).is_err());
}
#[test]
fn barrier_hit_is_inclusive_and_removes_future_monitoring() {
    for (direction, values) in [
        (BarrierDirection::Up, [99.0, 100.0, 101.0]),
        (BarrierDirection::Down, [101.0, 100.0, 99.0]),
    ] {
        for (i, value) in values.into_iter().enumerate() {
            let a = obs("a", "2027-03-01");
            let b = obs("b", "2027-06-01");
            let c = Contract::new(
                "barrier",
                vec![a.clone(), b],
                Term::Barrier {
                    id: "barrier-state".into(),
                    observations: vec!["a".into(), "b".into()],
                    direction,
                    level: 100.0,
                    hit: false,
                    on_hit: Box::new(cf("rebate", 5.0.into(), "2027-06-01")),
                    on_no_hit: Box::new(cf("principal", 100.0.into(), "2027-06-01")),
                },
            )
            .unwrap();
            let r = c
                .advance(&[fixing("fix", &a, value)], d("2027-03-02"))
                .unwrap();
            assert_eq!(r.barrier_states["barrier-state"], i > 0);
            assert_eq!(r.remaining_observations.len(), usize::from(i == 0));
            if i > 0 {
                assert_eq!(
                    r.residual_contract.analyze().unwrap().potential_cashflows[0].id,
                    "rebate"
                );
            }
        }
    }
}
#[test]
fn autocall_style_condition_cancels_all_later_cashflows() {
    let a = obs("first", "2027-03-01");
    let b = obs("last", "2027-06-01");
    let c = Contract::new(
        "autocall",
        vec![a.clone(), b],
        Term::conditional(
            NumericExpr::observation("first").compare(Comparison::GreaterEqual, 100.0.into()),
            cf("early", 103.0.into(), "2027-03-01"),
            cf("later", NumericExpr::observation("last"), "2027-06-01"),
        ),
    )
    .unwrap();
    let r = c
        .advance(&[fixing("fix", &a, 100.0)], d("2027-03-02"))
        .unwrap();
    assert!(r.remaining_observations.is_empty());
    assert_eq!(
        r.outstanding_payments
            .iter()
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>(),
        vec!["early"]
    );
}
#[test]
fn termination_preserves_prior_obligations_and_cancels_future_dependence() {
    let c = two();
    let r = c
        .advance(
            &[event(
                "terminate",
                "2027-03-01",
                Event::Termination {
                    contract_id: "two".into(),
                },
            )],
            d("2027-06-20"),
        )
        .unwrap();
    assert_eq!(r.outstanding_payments.len(), 1);
    assert_eq!(r.outstanding_payments[0].id, "coupon");
    assert!(r.remaining_observations.is_empty());
}
fn path(t: &Term, f: &BTreeMap<String, f64>) -> Vec<(String, f64)> {
    match t {
        Term::Empty => vec![],
        Term::Cashflow { cashflow: c } => vec![(
            c.id.clone(),
            c.amount.evaluate(f).unwrap() * c.direction.sign(),
        )],
        Term::Bundle { contracts } => contracts.iter().flat_map(|t| path(t, f)).collect(),
        Term::Conditional {
            condition,
            then_contract,
            else_contract,
        } => path(
            if condition.evaluate(f).unwrap() {
                then_contract
            } else {
                else_contract
            },
            f,
        ),
        Term::Reverse { contract } => path(contract, f)
            .into_iter()
            .map(|(id, a)| (id, -a))
            .collect(),
        Term::Scale { factor, contract } => path(contract, f)
            .into_iter()
            .map(|(id, a)| (id, a * factor.evaluate(f).unwrap()))
            .collect(),
        _ => panic!("test helper only covers the generated source family"),
    }
}
#[test]
fn residual_future_cashflows_match_original_across_paths_and_signs() {
    for historical in [70.0, 100.0, 130.0] {
        let c = two();
        let r = c
            .advance(
                &[fixing("first", &c.observations()[0], historical)],
                d("2027-03-02"),
            )
            .unwrap();
        for terminal in [0.0, 70.0, 100.0, 140.0, 1000.0] {
            let f = BTreeMap::from([("first".into(), historical), ("last".into(), terminal)]);
            assert_eq!(path(c.root(), &f), path(r.residual_contract.root(), &f));
        }
    }
    let c = Contract::new(
        "signs",
        vec![],
        cf("fixed", 3.0.into(), "2027-03-01")
            .reverse()
            .scale(2.0.into()),
    )
    .unwrap();
    assert_eq!(
        c.advance(&[], d("2027-03-01"))
            .unwrap()
            .outstanding_payments[0]
            .amount,
        -6.0
    );
}
