use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::product::{
    ArithmeticAsianSpec, AsianObservation, CompiledPayoff, EuropeanVanillaSpec, GraphComparison,
    GraphLimitPolicy, OptionSide, SourceGraphBuilder, SourceOpcode as Op,
};
use pricing_standalone_aad_check::{AdaptedPayoff, CompileError, Observation};
use std::collections::BTreeMap;

fn check_paths(native: &CompiledPayoff, rows: &[Vec<f64>], seeds: &[f64]) {
    let translated = AdaptedPayoff::compile(native).unwrap();
    let mut ws = translated.graph.workspace();
    for row in rows {
        assert_eq!(row.len(), translated.observations.len());
        let values: BTreeMap<_, _> = translated
            .observations
            .iter()
            .copied()
            .zip(row.iter().copied())
            .collect();
        let standalone = translated.graph.evaluate(row, &mut ws).unwrap().to_vec();
        let mut wanted = BTreeMap::<Observation, f64>::new();
        for (i, &seed) in seeds.iter().enumerate() {
            let native = native
                .evaluate_output_with_observation_adjoints(
                    i,
                    |u, d| values.get(&Observation::Terminal(u, d)).copied(),
                    |u, d| values.get(&Observation::PreDividend(u, d)).copied(),
                )
                .unwrap();
            assert_eq!(standalone[i].to_bits(), native.value.to_bits());
            for a in native.terminal_adjoints {
                *wanted
                    .entry(Observation::Terminal(a.underlying, a.observation_date))
                    .or_default() += seed * a.value;
            }
            for a in native.pre_dividend_adjoints {
                *wanted
                    .entry(Observation::PreDividend(a.underlying, a.observation_date))
                    .or_default() += seed * a.value;
            }
        }
        let actual = translated.graph.vjp(row, seeds, &mut ws).unwrap();
        for (key, &v) in translated.observations.iter().zip(actual) {
            assert_eq!(v, wanted.remove(key).unwrap_or(0.0), "{key:?}");
        }
        assert!(wanted.is_empty());
    }
}

#[test]
fn vanilla_call_and_put_cover_moneyness_and_equality() {
    let expiry = "2027-01-04".parse().unwrap();
    for side in [OptionSide::Call, OptionSide::Put] {
        let vanilla = EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            expiry,
            100.0,
            4.0,
            side,
        )
        .unwrap();
        let native = vanilla
            .source_graph()
            .unwrap()
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        check_paths(&native, &[vec![101.0], vec![100.0], vec![99.0]], &[1.0]);
    }
}

#[test]
fn asian_paths_map_input_adjoints_to_dates() {
    let d1 = "2027-01-04".parse().unwrap();
    let d2 = "2027-02-04".parse().unwrap();
    let asian = ArithmeticAsianSpec::new(
        UnderlyingId::new(1),
        CurrencyId::new(1),
        100.0,
        2.0,
        OptionSide::Call,
        vec![
            AsianObservation::unknown(d1, 0.25).unwrap(),
            AsianObservation::unknown(d2, 0.75).unwrap(),
        ],
        d2,
    )
    .unwrap();
    let native = asian
        .source_graph()
        .unwrap()
        .compile_compact(GraphLimitPolicy::DEFAULT)
        .unwrap();
    check_paths(
        &native,
        &[vec![80.0, 104.0], vec![112.0, 100.0], vec![100.0, 100.0]],
        &[0.5],
    );
}

#[test]
fn shared_pre_and_post_dividend_inputs_lazy_domains_and_output_seeds() {
    let date: Date = "2027-01-04".parse().unwrap();
    let u = UnderlyingId::new(1);
    let mut b = SourceGraphBuilder::new();
    let spot = Op::TerminalSpot {
        underlying: u,
        observation_date: date,
    };
    let x = b.push(spot).unwrap();
    let repeated = b.push(spot).unwrap();
    let pre = b
        .push(Op::PreDividendSpot {
            underlying: u,
            observation_date: date,
        })
        .unwrap();
    let diff = b
        .push(Op::Subtract {
            left: pre,
            right: x,
        })
        .unwrap();
    let zero = b.literal(0.0).unwrap();
    let positive = b
        .push(Op::Compare {
            left: diff,
            right: zero,
            operation: GraphComparison::Greater,
        })
        .unwrap();
    let log = b.push(Op::Log { input: diff }).unwrap();
    let guarded = b
        .push(Op::Select {
            condition: positive,
            then_value: log,
            else_value: repeated,
        })
        .unwrap();
    let tape = b
        .finish(vec![guarded, x])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
    let translated = AdaptedPayoff::compile(&tape).unwrap();
    assert_eq!(
        translated.observations,
        [
            Observation::Terminal(u, date),
            Observation::PreDividend(u, date)
        ]
    );
    check_paths(
        &tape,
        &[vec![100.0, 102.0], vec![100.0, 99.0], vec![100.0, 100.0]],
        &[2.0, -0.5],
    );
}

#[test]
fn unsupported_operations_are_reported_before_replay() {
    let mut b = SourceGraphBuilder::new();
    let spot = b
        .push(Op::TerminalSpot {
            underlying: UnderlyingId::new(1),
            observation_date: "2027-01-04".parse().unwrap(),
        })
        .unwrap();
    let out = b.push(Op::Indicator { input: spot }).unwrap();
    let tape = b
        .finish(vec![out])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
    assert!(matches!(
        AdaptedPayoff::compile(&tape),
        Err(CompileError::Unsupported(_))
    ));
}
