use pricing::analytical::{BlackForwardOracleInputs, evaluate_black_forward};
use pricing::product::OptionSide;
use pricing_numerics::{CenteredMoment, Complex64};

fn main() {
    check_number_domains();
    check_identifiers();
    check_gregorian_calendar();
    let mut violations = 0;
    let strict = std::env::args().nth(1).as_deref() == Some("--strict");
    for volatility in [0.2, 2.0_f64.powi(-60)] {
        for side in [OptionSide::Call, OptionSide::Put] {
            let inputs = BlackForwardOracleInputs {
                side,
                forward: 1.0,
                strike: 1.0,
                notional: 1.0,
                discount: 1.0,
                volatility,
                time: 1.0,
            };
            let result = evaluate_black_forward(inputs).expect("finite inputs and finite Greeks");
            let positive = result.price > 0.0;
            violations += usize::from(!positive);
            println!(
                "Black-76 {side:?} sigma={volatility:e} price={:.17e} bits={:016x} positive={positive}",
                result.price,
                result.price.to_bits()
            );
        }
    }
    for value in [1.0, f64::MAX] {
        let quotient = Complex64::new(value, value) / Complex64::new(1.0, 1.0);
        let correct = quotient.re == value && quotient.im == 0.0;
        violations += usize::from(!correct);
        println!(
            "Complex division x={value:e} re={:e} im={:e} reconstructed={correct}",
            quotient.re, quotient.im
        );
    }
    let mut left = CenteredMoment::new();
    let mut right = CenteredMoment::new();
    for x in [1.0, 3.0] {
        left.add(x);
    }
    for x in [5.0, 7.0] {
        right.add(x);
    }
    let stats = CenteredMoment::merged(left, right);
    assert_eq!(
        (stats.count(), stats.mean(), stats.second_moment()),
        (4, 4.0, 20.0)
    );
    println!("Disagreements with exact contracts: {violations}");
    if strict && violations != 0 {
        std::process::exit(1);
    }
}

fn check_gregorian_calendar() {
    use Weekday::{Friday, Monday, Saturday, Sunday, Thursday, Tuesday, Wednesday};
    use pricing::core::{Date, DayCountConvention, Weekday};
    let weekdays = [
        Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,
    ];
    let start = Date::MIN;
    let mut ordinal = 0_i32;
    let mut previous: Option<Date> = None;
    for year in 1..=9999_u16 {
        let leap =
            year.is_multiple_of(400) || (year.is_multiple_of(4) && !year.is_multiple_of(100));
        let lengths = [
            31,
            28 + u8::from(leap),
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        for (i, days) in lengths.into_iter().enumerate() {
            let month = u8::try_from(i + 1).unwrap();
            assert!(Date::new(year, month, 0).is_err());
            assert!(Date::new(year, month, days + 1).is_err());
            for day in 1..=days {
                let date = Date::new(year, month, day).unwrap();
                assert_eq!(start.days_until(date), ordinal);
                assert_eq!(date.weekday(), weekdays[(ordinal % 7) as usize]);
                assert_eq!(
                    DayCountConvention::Act365F.year_fraction(start, date),
                    f64::from(ordinal) / 365.0
                );
                assert_eq!(
                    DayCountConvention::Act360.year_fraction(date, start),
                    -f64::from(ordinal) / 360.0
                );
                if let Some(prior) = previous {
                    assert_eq!(prior.next_day().unwrap(), date);
                    assert_eq!(date.previous_day().unwrap(), prior);
                }
                previous = Some(date);
                ordinal += 1;
            }
        }
    }
    assert_eq!(previous, Some(Date::MAX));
    assert!(Date::MIN.previous_day().is_err() && Date::MAX.next_day().is_err());
    assert_eq!(ordinal, 3_652_059);
    println!("Gregorian public API: {ordinal} valid dates checked");
}

fn check_number_domains() {
    use pricing::core::{FiniteF64, NonNegativeF64, PositiveF64};
    let fractions = [0, 1, (1_u64 << 51) - 1, 1 << 51, (1 << 52) - 1];
    let mut checked = 0;
    for sign in [0_u64, 1] {
        for exponent in 0_u64..2048 {
            for fraction in fractions {
                let bits = (sign << 63) | (exponent << 52) | fraction;
                let value = f64::from_bits(bits);
                let finite = exponent < 2047;
                let zero = exponent == 0 && fraction == 0;
                assert_eq!(
                    FiniteF64::new(value, "x").ok().map(|x| x.to_bits()),
                    finite.then_some(bits)
                );
                assert_eq!(
                    PositiveF64::new(value, "x").ok().map(|x| x.get().to_bits()),
                    (finite && sign == 0 && !zero).then_some(bits)
                );
                assert_eq!(
                    NonNegativeF64::new(value, "x")
                        .ok()
                        .map(|x| x.get().to_bits()),
                    (finite && (sign == 0 || zero)).then_some(bits)
                );
                checked += 1;
            }
        }
    }
    println!("Numeric wrappers: {checked} bit patterns, three public constructors each");
}

fn check_identifiers() {
    use pricing::core::{CurrencyId, CurveId, EventId, NodeId, PathIndex, TimeIndex, UnderlyingId};
    for n in u16::MIN..=u16::MAX {
        let id = CurrencyId::new(n);
        assert_eq!(id.get(), n);
        assert_eq!(id.checked_next().ok().map(|x| x.get()), n.checked_add(1));
    }
    macro_rules! check_boundaries {
        ($id:ident, $raw:ty) => {
            for n in [0, 1, <$raw>::MAX - 1, <$raw>::MAX] {
                assert_eq!(
                    $id::new(n).checked_next().ok().map(|x| x.get()),
                    n.checked_add(1)
                );
            }
        };
    }
    check_boundaries!(CurveId, u32);
    check_boundaries!(EventId, u32);
    check_boundaries!(NodeId, u32);
    check_boundaries!(TimeIndex, u32);
    check_boundaries!(UnderlyingId, u32);
    check_boundaries!(PathIndex, u64);
    println!("IDs: all 65536 CurrencyId values and boundaries of the other six ID types");
}
