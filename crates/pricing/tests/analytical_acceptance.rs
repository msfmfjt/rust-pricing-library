use pricing::analytical::{BlackScholesOracleInputs, evaluate};
use pricing::product::OptionSide;

const GRID: &str = include_str!("../../../fixtures/acceptance/european_bs_analytical.csv");
const SPOT: f64 = 100.0;

#[derive(Debug)]
struct Case<'a> {
    name: &'a str,
    side: OptionSide,
    moneyness: f64,
    time: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    expected_price: f64,
    expected_delta: f64,
    expected_gamma: f64,
    expected_vega: f64,
}

#[test]
fn analytical_oracle_matches_frozen_acceptance_grid() {
    let cases = GRID.lines().skip(1).map(parse_case).collect::<Vec<_>>();
    assert_eq!(
        cases.len(),
        18,
        "the reviewed grid size is part of the fixture"
    );

    for case in cases {
        let discount = (-case.rate * case.time).exp();
        let dividend_discount = (-case.dividend_yield * case.time).exp();
        let forward = SPOT * dividend_discount / discount;
        let result = evaluate(BlackScholesOracleInputs {
            side: case.side,
            spot: SPOT,
            strike: case.moneyness * forward,
            notional: 1.0,
            discount,
            dividend_discount,
            volatility: case.volatility,
            time: case.time,
        })
        .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));

        assert_close(
            case.name,
            "price",
            result.price,
            case.expected_price,
            5.0e-13,
        );
        assert_close(
            case.name,
            "delta",
            result.delta,
            case.expected_delta,
            5.0e-14,
        );
        assert_close(
            case.name,
            "gamma",
            result.gamma,
            case.expected_gamma,
            5.0e-15,
        );
        assert_close(
            case.name,
            "vega",
            result.vega,
            case.expected_vega,
            5.0e-13,
        );
    }
}

fn parse_case(line: &str) -> Case<'_> {
    let fields = line.split(',').collect::<Vec<_>>();
    assert_eq!(fields.len(), 11, "invalid fixture row: {line}");
    let number = |index: usize| {
        fields[index]
            .parse::<f64>()
            .unwrap_or_else(|error| panic!("invalid number in {}: {error}", fields[0]))
    };
    Case {
        name: fields[0],
        side: match fields[1] {
            "call" => OptionSide::Call,
            "put" => OptionSide::Put,
            side => panic!("invalid side in {}: {side}", fields[0]),
        },
        moneyness: number(2),
        time: number(3),
        rate: number(4),
        dividend_yield: number(5),
        volatility: number(6),
        expected_price: number(7),
        expected_delta: number(8),
        expected_gamma: number(9),
        expected_vega: number(10),
    }
}

fn assert_close(case: &str, field: &str, actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    let bound = tolerance * expected.abs().max(1.0);
    assert!(
        error <= bound,
        "{case} {field}: actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}, bound={bound:.3e}"
    );
}
