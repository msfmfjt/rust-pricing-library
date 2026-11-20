use pricing::contract::*;
use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::product::{
    self, BarrierMonitoring, BarrierStyle, DigitalPayout, GraphLimitPolicy, OptionSide,
};
fn d(s: &str) -> Date {
    s.parse().unwrap()
}
fn builder(id: &str) -> InstrumentBuilder {
    InstrumentBuilder::new(
        id,
        CurrencyId::new(2),
        UnderlyingId::new(1),
        "exchange.close",
    )
    .unwrap()
}
fn payoff(c: &Contract, path: &[(Date, f64)]) -> f64 {
    let compiled = c
        .compile_mc(d("2026-01-01"), vec![UnderlyingId::new(1)])
        .unwrap();
    compiled.groups()[0]
        .payoff()
        .evaluate(|_, date| path.iter().find(|(t, _)| *t == date).map(|(_, v)| *v))
        .unwrap()
        .iter()
        .sum()
}
fn fixing(c: &Contract, index: usize, value: f64) -> LifecycleEvent {
    let o = c.analyze().unwrap().required_observations[index].clone();
    LifecycleEvent {
        id: format!("fix/{index}"),
        effective_date: o.date,
        event: Event::Fixing {
            observation: o,
            value,
        },
    }
}
#[test]
fn forward_european_digital_and_composition_match_native_paths() {
    let expiry = d("2027-01-01");
    let b = builder("instrument");
    let f = b.forward(100.0, 2.0, expiry, expiry).unwrap();
    for side in [OptionSide::Call, OptionSide::Put] {
        let vanilla = b.european(side, 100.0, 2.0, expiry, expiry).unwrap();
        let native = product::EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            2.0,
            side,
        )
        .unwrap()
        .source_graph()
        .unwrap()
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
        let digital = b.digital(side, 100.0, 7.0, expiry, expiry).unwrap();
        let old_digital = product::DigitalSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            7.0,
            side,
            DigitalPayout::Cash,
        )
        .unwrap()
        .source_graph()
        .unwrap()
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
        for spot in [80.0, 100.0, 120.0] {
            assert_eq!(
                payoff(&vanilla, &[(expiry, spot)]),
                native.evaluate(|_, _| Some(spot)).unwrap()[0]
            );
            assert_eq!(
                payoff(&digital, &[(expiry, spot)]),
                old_digital.evaluate(|_, _| Some(spot)).unwrap()[0]
            );
            assert_eq!(payoff(&f, &[(expiry, spot)]), 2.0 * (spot - 100.0));
        }
    }
    let opposite = builder("opposite")
        .forward(100.0, 2.0, expiry, expiry)
        .unwrap()
        .reversed("reverse")
        .unwrap();
    let bundled =
        Contract::bundle("hedged", vec![f.scaled("two", 2.0).unwrap(), opposite]).unwrap();
    assert_eq!(payoff(&bundled, &[(expiry, 120.0)]), 40.0);
    assert_eq!(
        Contract::from_json(
            bundled.to_json().unwrap().as_bytes(),
            pricing::JsonLimits::DEFAULT
        )
        .unwrap(),
        bundled
    );
}
#[test]
fn all_path_aggregations_preserve_history_and_native_asian_values() {
    let dates = vec![d("2026-07-01"), d("2027-01-01")];
    let path = [(dates[0], 80.0), (dates[1], 125.0)];
    for (operation, expected) in [
        (Aggregate::ArithmeticAverage, 22.5),
        (Aggregate::GeometricAverage, 20.0),
        (Aggregate::RunningMinimum, 0.0),
        (Aggregate::RunningMaximum, 45.0),
    ] {
        let c = builder("asian")
            .asian(
                operation,
                OptionSide::Call,
                80.0,
                1.0,
                dates.clone(),
                dates[1],
            )
            .unwrap();
        assert!((payoff(&c, &path) - expected).abs() < 1e-12);
        let r = c.advance(&[fixing(&c, 0, 80.0)], dates[0]).unwrap();
        assert_eq!(r.remaining_observations.len(), 1);
        assert!((payoff(&r.residual_contract, &path[1..]) - expected).abs() < 1e-12);
        if operation == Aggregate::ArithmeticAverage {
            let old = product::ArithmeticAsianSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                80.0,
                1.0,
                OptionSide::Call,
                dates
                    .iter()
                    .map(|t| product::AsianObservation::unknown(*t, 0.5).unwrap())
                    .collect(),
                dates[1],
            )
            .unwrap()
            .source_graph()
            .unwrap()
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
            assert_eq!(
                payoff(&c, &path),
                old.evaluate(|_, t| path.iter().find(|(d, _)| *d == t).map(|(_, v)| *v))
                    .unwrap()[0]
            );
        }
    }
}
#[test]
fn barriers_match_old_native_payoff_and_lifecycle_at_inclusive_boundary() {
    let dates = vec![d("2026-07-01"), d("2027-01-01")];
    for direction in [BarrierDirection::Up, BarrierDirection::Down] {
        for style in [BarrierStyle::KnockIn, BarrierStyle::KnockOut] {
            let level = if direction == BarrierDirection::Up {
                120.0
            } else {
                80.0
            };
            let c = builder("barrier")
                .discrete_barrier(DiscreteBarrierTerms {
                    side: OptionSide::Call,
                    strike: 90.0,
                    notional: 1.0,
                    expiry: dates[1],
                    payment_date: dates[1],
                    monitoring_dates: dates.clone(),
                    direction,
                    style,
                    level,
                    rebate: 3.0,
                })
                .unwrap();
            let old = product::BarrierSpec::new(
                UnderlyingId::new(1),
                CurrencyId::new(2),
                dates[1],
                90.0,
                level,
                1.0,
                OptionSide::Call,
                if direction == BarrierDirection::Up {
                    product::BarrierDirection::Up
                } else {
                    product::BarrierDirection::Down
                },
                style,
                BarrierMonitoring::Discrete,
                dates.clone(),
                Some(3.0),
                dates[1],
            )
            .unwrap()
            .source_graph()
            .unwrap()
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
            for first in [level - 0.01, level, level + 0.01] {
                let path = [(dates[0], first), (dates[1], 100.0)];
                let expected = old
                    .evaluate(|_, t| path.iter().find(|(d, _)| *d == t).map(|(_, v)| *v))
                    .unwrap()[0];
                assert_eq!(payoff(&c, &path), expected);
                let residual = c.advance(&[fixing(&c, 0, first)], dates[0]).unwrap();
                assert_eq!(payoff(&residual.residual_contract, &path[1..]), expected);
                assert_eq!(
                    residual.barrier_states["barrier/barrier"],
                    if direction == BarrierDirection::Up {
                        first >= level
                    } else {
                        first <= level
                    }
                );
            }
        }
    }
}
#[test]
fn swap_fixings_are_explicit_and_legs_remain_identifiable() {
    let date = d("2027-01-01");
    let c = builder("swap")
        .swap(
            1000.0,
            0.03,
            vec![SwapPeriod {
                observation_date: date,
                payment_date: d("2027-07-01"),
                year_fraction: 0.5,
            }],
            true,
        )
        .unwrap();
    assert!(matches!(
        c.compile_mc(d("2026-01-01"), vec![UnderlyingId::new(1)]),
        Err(ContractError::Unsupported(_))
    ));
    let r = c.advance(&[fixing(&c, 0, 0.05)], date).unwrap();
    assert_eq!(r.determined_cashflows.len(), 2);
    assert_eq!(
        r.determined_cashflows.iter().map(|c| c.amount).sum::<f64>(),
        10.0
    );
    assert_eq!(payoff(&r.residual_contract, &[]), 10.0);
    assert!(c.advance(&[], d("2027-01-02")).is_err());
}
#[test]
fn autocall_cancels_future_observations_and_flows_with_replay_and_settlement() {
    let dates = [d("2026-07-01"), d("2027-01-01")];
    let c = builder("autocall")
        .autocall(
            100.0,
            100.0,
            70.0,
            dates
                .iter()
                .map(|t| AutocallPeriod {
                    observation_date: *t,
                    payment_date: *t,
                    call_level: 100.0,
                    coupon_level: 80.0,
                    coupon_rate: 0.05,
                })
                .collect(),
        )
        .unwrap();
    // No terminal fixing is available after first-date redemption.
    assert_eq!(payoff(&c, &[(dates[0], 100.0)]), 105.0);
    assert_eq!(payoff(&c, &[(dates[0], 90.0), (dates[1], 90.0)]), 110.0);
    assert_eq!(payoff(&c, &[(dates[0], 60.0), (dates[1], 60.0)]), 60.0);
    let event = fixing(&c, 0, 100.0);
    let r = c
        .advance(&[event.clone(), event.clone()], dates[0])
        .unwrap();
    assert!(r.remaining_observations.is_empty());
    assert_eq!(r.outstanding_payments.len(), 2);
    assert_eq!(
        r.outstanding_payments.iter().map(|c| c.amount).sum::<f64>(),
        105.0
    );
    let settlements: Vec<_> = r
        .outstanding_payments
        .iter()
        .enumerate()
        .map(|(i, c)| LifecycleEvent {
            id: format!("settle/{i}"),
            effective_date: dates[0],
            event: Event::Settlement {
                cashflow_id: c.id.clone(),
            },
        })
        .collect();
    let done = r.advance(&settlements, dates[1]).unwrap();
    assert!(done.outstanding_payments.is_empty());
    assert!(matches!(done.residual_contract.root(), Term::Empty));
    let mut replay = settlements;
    replay.push(event);
    assert_eq!(done, c.advance(&replay, dates[1]).unwrap());
}
#[test]
fn invalid_schedules_duplicate_barriers_and_overflows_are_rejected() {
    let date = d("2027-01-01");
    assert!(
        builder("x")
            .asian(
                Aggregate::ArithmeticAverage,
                OptionSide::Call,
                1.0,
                1.0,
                vec![date, date],
                date
            )
            .is_err()
    );
    assert!(builder("x").forward(1.0, f64::NAN, date, date).is_err());
    assert!(builder("x").autocall(100.0, 0.0, 70.0, vec![]).is_err());
    let obs = Observation::spot("s", UnderlyingId::new(1), date, "close");
    let barrier = Term::Barrier {
        id: "same".into(),
        observations: vec!["s".into()],
        direction: BarrierDirection::Up,
        level: 100.0,
        hit: false,
        on_hit: Box::new(Term::Empty),
        on_no_hit: Box::new(Term::Empty),
    };
    assert!(
        Contract::new(
            "duplicate",
            vec![obs],
            Term::bundle(vec![barrier.clone(), barrier])
        )
        .is_err()
    );
}
#[test]
fn logical_short_circuit_and_partial_evaluation_preserve_domain_errors() {
    let bad = NumericExpr::from(1.0)
        .divided_by(0.0.into())
        .compare(Comparison::Greater, 0.0.into());
    let f = std::collections::BTreeMap::new();
    assert!(bad.clone().and(false.into()).evaluate(&f).is_err());
    assert!(bad.clone().or(true.into()).evaluate(&f).is_err());
    assert!(
        !BooleanExpr::from(false)
            .and(bad.clone())
            .evaluate(&f)
            .unwrap()
    );
    assert!(BooleanExpr::from(true).or(bad).evaluate(&f).unwrap());
}

#[test]
fn cached_branch_workspaces_isolate_payoffs_errors_threads_and_reentrant_callbacks() {
    let expiry = d("2027-01-01");
    let c = builder("cache")
        .digital(OptionSide::Call, 100.0, 7.0, expiry, expiry)
        .unwrap();
    let compiled = c
        .compile_mc(d("2026-01-01"), vec![UnderlyingId::new(1)])
        .unwrap();
    let p = compiled.groups()[0].payoff();
    assert!(p.evaluate(|_, _| None).is_err());
    assert_eq!(p.evaluate(|_, _| Some(100.0)).unwrap(), [7.0]);
    // Evaluating another lazy tape while an observation callback is active
    // cannot borrow the cached workspace twice or corrupt either result.
    assert_eq!(
        p.evaluate(|_, _| {
            assert_eq!(p.evaluate(|_, _| Some(99.0)).unwrap(), [0.0]);
            Some(101.0)
        })
        .unwrap(),
        [7.0]
    );
    let mut w = p.workspace();
    for i in 0..20 {
        let spot = if i % 2 == 0 { 90.0 } else { 110.0 };
        assert_eq!(
            p.evaluate(|_, _| Some(spot)).unwrap(),
            p.evaluate_reusing(&mut w, |_, _| Some(spot), |_, _| None)
                .unwrap()
        );
    }
    std::thread::scope(|scope| {
        for spot in [99.0, 100.0] {
            scope.spawn(move || {
                for _ in 0..20 {
                    assert_eq!(
                        p.evaluate(|_, _| Some(spot)).unwrap(),
                        [if spot >= 100.0 { 7.0 } else { 0.0 }]
                    );
                }
            });
        }
    });
}
