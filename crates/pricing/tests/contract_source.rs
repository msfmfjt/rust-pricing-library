use pricing::JsonLimits;
use pricing::contract::*;
use pricing::core::{CurrencyId, Date, UnderlyingId};
use std::collections::BTreeMap;
fn date(s: &str) -> Date {
    s.parse().unwrap()
}
fn observation() -> Observation {
    Observation::spot(
        "close",
        UnderlyingId::new(1),
        date("2027-01-01"),
        "exchange.close",
    )
}
fn cashflow(amount: NumericExpr) -> Cashflow {
    Cashflow::new(
        "payment",
        CurrencyId::new(2),
        amount,
        date("2027-01-01"),
        date("2027-01-03"),
    )
}
fn contract() -> Contract {
    Contract::new(
        "call",
        vec![observation()],
        cashflow(
            NumericExpr::observation("close")
                .minus(100.0.into())
                .maximum(0.0.into()),
        )
        .into(),
    )
    .unwrap()
}
#[test]
fn typed_expressions_and_strict_json_preserve_identity() {
    let c = contract();
    let json = c.to_json().unwrap();
    let d = Contract::from_json(json.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(c, d);
    assert_eq!(c.fingerprint().unwrap(), d.fingerprint().unwrap());
    for bad in [
        json.replace("\"schema_version\":1", "\"schema_version\":2"),
        json.replace(
            "\"document_kind\":\"contract\"",
            "\"document_kind\":\"other\"",
        ),
        json.replacen("{", "{\"unknown\":0,", 1),
        json.replacen("{", "{\"id\":\"duplicate\",", 1),
        json.replace("\"value\":100.0", "\"value\":true"),
        json.replace("\"2027-01-01\"", "\"2027-02-30\""),
    ] {
        assert!(
            Contract::from_json(bad.as_bytes(), JsonLimits::DEFAULT).is_err(),
            "{bad}"
        );
    }
}
#[test]
fn dependency_dates_currency_direction_and_reference_validation() {
    let a = contract().analyze().unwrap();
    assert_eq!(a.required_observations, vec![observation()]);
    assert_eq!(a.horizon, Some(date("2027-01-03")));
    assert_eq!(a.underlying_dependencies, vec![UnderlyingId::new(1)]);
    assert!(a.determined_cashflows.is_empty());
    assert!(
        Contract::new(
            "bad",
            vec![],
            cashflow(NumericExpr::observation("close")).into()
        )
        .is_err()
    );
    let mut c = cashflow(NumericExpr::observation("close"));
    c.determination_date = date("2026-12-31");
    assert!(Contract::new("bad", vec![observation()], c.into()).is_err());
    assert!(Contract::new("bad", vec![observation(), observation()], Term::Empty).is_err());
    let mut other = cashflow(2.0.into());
    other.id = "eur".into();
    other.currency = CurrencyId::new(3);
    let c = Contract::new(
        "mixed",
        vec![],
        Term::bundle(vec![cashflow(1.0.into()).into(), other.into()]),
    )
    .unwrap();
    assert_eq!(
        c.analyze().unwrap().currencies,
        vec![CurrencyId::new(2), CurrencyId::new(3)]
    );
    assert!(Contract::new("bad", vec![], cashflow(f64::NAN.into()).into()).is_err());
}
#[test]
fn numeric_boolean_date_semantics_and_conditional_laziness() {
    let f = BTreeMap::from([("close".into(), 120.0)]);
    let x = NumericExpr::observation("close");
    let condition = x.clone().compare(Comparison::GreaterEqual, 120.0.into());
    let invalid = NumericExpr::from(1.0).divided_by(0.0.into());
    assert_eq!(
        NumericExpr::select(condition.clone(), 7.0.into(), invalid.clone())
            .evaluate(&f)
            .unwrap(),
        7.0
    );
    assert!(invalid.evaluate(&f).is_err());
    assert!(x.evaluate(&BTreeMap::new()).is_err());
    let dates = BooleanExpr::DateCompare {
        operation: Comparison::Less,
        left: Box::new(date("2026-01-01").into()),
        right: Box::new(date("2027-01-01").into()),
    };
    assert!(dates.evaluate(&f).unwrap());
    assert!(condition.and(dates).evaluate(&f).unwrap());
    assert_eq!(NumericExpr::from(-3.0).abs().evaluate(&f).unwrap(), 3.0);
    for (operation, expected) in [
        (Aggregate::ArithmeticAverage, 5.0),
        (Aggregate::GeometricAverage, 4.0),
        (Aggregate::RunningMinimum, 2.0),
        (Aggregate::RunningMaximum, 8.0),
    ] {
        let e = NumericExpr::Aggregate {
            operation,
            values: vec![2.0.into(), 8.0.into()],
        };
        assert!((e.evaluate(&f).unwrap() - expected).abs() < 1e-14);
    }
}
#[test]
fn algebra_properties_over_deterministic_sample_population() {
    let mut state = 1234567_u64;
    for _ in 0..1000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 32) as i32 as f64) / 65536.0;
        let f = BTreeMap::from([("x".into(), x)]);
        let e = NumericExpr::observation("x");
        assert_eq!(e.clone().minus(e.clone()).evaluate(&f).unwrap(), 0.0);
        assert_eq!(e.clone().maximum(e.clone()).evaluate(&f).unwrap(), x);
        assert_eq!(
            e.clone()
                .compare(Comparison::Less, 0.0.into())
                .negate()
                .evaluate(&f)
                .unwrap(),
            x >= 0.0
        );
        assert_eq!(e.clone().abs().evaluate(&f).unwrap(), x.abs());
    }
}
#[test]
fn unsupported_exercise_is_not_reinterpreted_as_a_maximum() {
    let e = Term::Exercise {
        id: "exercise".into(),
        date: date("2027-01-01"),
        contract: Box::new(Term::Empty),
    };
    assert!(matches!(
        Contract::new("x", vec![], e),
        Err(ContractError::Unsupported(_))
    ));
}
